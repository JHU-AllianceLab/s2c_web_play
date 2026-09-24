#!/usr/bin/env python3
"""Freeze the SYM and ASYM touchdown training plants into static MJCF the browser loads.

WHAT THIS IS
------------
mjlab never stores the game scene as a file: it builds an ``MjSpec``
programmatically from ``go2.xml`` + ``EntityCfg`` and compiles it in memory.  The
browser has no mjlab, so this script walks the SAME mjlab code path, freezes the
resulting spec to XML + mesh assets, and then PROVES the frozen copy compiles to
a model whose every physics-bearing array is identical to what mjlab compiles.

Route, per game:

    <presets>._make_env(preset, play=True)           # ManagerBasedRlEnvCfg
      -> mjlab.scene.Scene(cfg.scene, device="cpu")  # builds the MjSpec
      -> spec.option <- cfg.sim.mujoco               # mjlab applies these to the
                                                     # COMPILED model at runtime
                                                     # (mjlab/sim/sim.py:76-91);
                                                     # a static MJCF must carry
                                                     # them in <option>
      -> Scene.write()                               # scene.xml + mesh assets
      -> flatten mesh paths, rename entity prefixes  # attacker/ -> a_ , defender/ -> b_
      -> recompile with plain mujoco and DIFF        # every array, see _diff_models
      -> reduce the 16 visual meshes                 # 27.1 MiB -> ~1 MiB, §"lite"
      -> append the ARENA decoration                 # visual only, §"arena"
      -> re-prove: bit-identical rollouts            # §"proofs"

``play=True`` is what adds mjlab's own visual-only field markers (boundary rails,
the touchdown line(s), the end-zone slab(s)) -- ``game_env_cfg._field_marker_spec_fn``
at :68-119 (asym) / :68-137 (sym), all ``contype=0 conaffinity=0 density=0 group=2``.
There are no physical walls in the training plant and none here: out-of-bounds is a
referee rule on the trunk centre, never a collision (recon/05 §0.2).

THREE PROOFS this script runs and prints every time
---------------------------------------------------
P1  exported plain XML  == mjlab's compiled model, array for array
      (12 counts, 20 opt fields, 4 stat fields, 91 arrays; --game both does it twice)
P2  reduced meshes      -> bit-identical 2 s rollouts (hold + sinusoid ctrl)
P3  arena decoration    -> bit-identical 2 s rollouts, every preexisting id preserved,
      every decor geom contype=0 conaffinity=0 density=0, decor body mass 0, nv unchanged

OUTPUTS (under --out, default <repo>/assets/scene/<game>)
    scene.xml        the static MJCF the browser loads (mjlab plant + arena decor)
    assets/*.obj     the 16 Go2 visual meshes, quadric-decimated (same filenames)
    scene.json       everything the JS side would otherwise hardcode
    keyframe.json    settled standing states at the spawn poses

USAGE
    cd <repo>
    PYTHONPATH=<mjlab_repo> <mjlab_venv>/bin/python tools/export_scene.py --game both

    # decimation needs one extra pure-python wheel; it is deliberately NOT
    # installed into the shared mjlab venv:
    #   pip install --no-deps --target /tmp/pylibs fast_simplification
    #   PYTHONPATH=/tmp/pylibs:<mjlab_repo> ... tools/export_scene.py
    # without it, pass --no-lite (the full 27.1 MiB meshes are then shipped).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# Source of truth.  Read-only: this script never writes into MJLAB_REPO.
# --------------------------------------------------------------------------- #
MJLAB_REPO = Path("/home/ray/Disk_ext/Go2/Project/unitree_rl_mjlab")

# mjlab entity key -> the prefix the web game uses.
RENAME = {"attacker": "a", "defender": "b"}

# --------------------------------------------------------------------------- #
# The two games.
#
# ``spawnConfig`` MUST equal app/config.js GAMES.<game>.spawn -- that file is the
# frozen interface every other agent codes against, and keyframe.json is built at
# these poses.  Asserted against the live scenario where the two should agree.
# --------------------------------------------------------------------------- #
GAMES: dict[str, dict] = {
  "sym": {
    "task": "Game-SymTouchdown-Go2-Go2-WBC-Clean",
    "twin": "Game-SymTouchdown-Go2-Go2-WBC-Clean-Filtered",
    "make_env": "src.tasks.sym_game.config.go2_go2.presets",
    "presets": ("src.tasks.sym_game.config.go2_go2.sym_preset", "SYM_PRESETS"),
    "seats": {"a": "A", "b": "B"},
    # The LOCKED demo opening every sym render/video uses.  Numbers:
    # recon/02_sym_checkpoints.md:317, decoded from /home/ray/demo_sym/
    # spawn_s2c_open.json on the 5080.  Mirrored in app/config.js GAMES.sym.spawn.
    "spawnConfig": {
      "a": {"x": -2.1431, "y": -0.1409, "yaw": 0.0979},
      "b": {"x": 2.1775, "y": -0.1215, "yaw": 3.1381},
    },
    "spawnConfigSource": "recon/02_sym_checkpoints.md:317 "
                         "(/home/ray/demo_sym/spawn_s2c_open.json), locked demo opening",
  },
  "asym": {
    "task": "Game-Touchdown-Go2-Go2-WBC-Clean",
    "twin": "Game-Touchdown-Go2-Go2-WBC-Clean-Filtered",
    "make_env": "src.tasks.game.config.go2_go2.presets",
    "presets": ("src.tasks.game.config.go2_go2.presets", "PRESETS"),
    "seats": {"a": "attacker", "b": "defender"},
    # The training nominal spawn, which is also app/config.js GAMES.asym.spawn.
    # touchdown.py:235-248; asserted equal to the live scenario below.
    "spawnConfig": {
      "a": {"x": -1.4, "y": 0.0, "yaw": 0.0},
      "b": {"x": 0.75, "y": 0.0, "yaw": math.pi},
    },
    "spawnConfigSource": "src/tasks/game/scenarios/touchdown.py:235-248 "
                         "(training nominal == app/config.js GAMES.asym.spawn)",
  },
}

LITE_FACE_FRACTION = 0.15
LITE_MIN_FACES = 2000
LITE_AGGRESSION = 3
LITE_COORD_FMT = "%.5g"  # 1e-5 m = 0.01 mm at Go2 scale
# The stock Go2 OBJs carry ONE normal per face corner, so a position-identical
# vertex appears once per incident triangle: base_4.obj is 46 564 "vertices" for
# 59 511 triangles and splits into 1313 connected components.  A quadric
# decimator has almost no shared edges to collapse on such a soup and instead
# deletes whole shells -- measured 49% of base_4's surface gone at 15% of the
# faces, and the render shows a shattered shell with a mangled logo.  Welding
# coincident positions first (1 um) turns base_4 into 30 997 shared vertices /
# 104 components and the SAME 15% target then costs 1.9 mm mean deviation with
# an intact silhouette.  Welding is exact: no triangle is moved, only re-indexed.
LITE_WELD_DIGITS = 6      # decimal places on the vertex key -> 1 um
LITE_MAX_P95_MM = 4.0     # gate: nothing we DRAW may sit this far off the original
LITE_MAX_BBOX_MM = 1.0    # gate: the per-mesh silhouette may not shrink/grow
SETTLE_SECONDS = 2.0

# Base height window the settled stand must fall in.  NOTE: the task brief asked
# for [0.25, 0.40]; the walk-soft plant (kp 20/20/40, go2_constants.py:41-67,
# recon/05 §3.1) does not reach it.  MEASURED on this very model: with the base
# at z = 0 and the joints at the default pose the lowest collision point
# (a_FL_foot_collision) sits at -0.295017 m, so the default stance is
# kinematically a 0.2950 m stand -- already below mjlab's 0.32 spawn -- and the
# legs then sag a further 0.0647 m under the robot's own 15.206 kg (37.3 N per
# leg against kp 20 hips/thighs), the rear shins coming to rest on the floor.
# The settled height is 0.2303 m for both robots in both games.  The window
# below is the measured truth with slack, not a relaxed target.  No gain was
# touched to make it pass; see the report and keyframe.json baseZWindowNote.
BASE_Z_WINDOW = (0.20, 0.40)


# --------------------------------------------------------------------------- #
# ARENA palette -- every value lifted from mjlab's own play-mode markers and
# ground material, so the browser venue matches the render clips we shipped.
#   boundary   (0.92, 0.92, 0.88, 0.65)  src/tasks/game/game_env_cfg.py:98
#   touchdown  (1.00, 0.28, 0.14, 0.90)  src/tasks/game/game_env_cfg.py:99
#   region     (0.10, 0.80, 0.35, 0.18)  src/tasks/game/game_env_cfg.py:100
#   ground checker rgb1 (0.2,0.3,0.4) / rgb2 (0.1,0.2,0.3)
#                            mjlab/terrains/terrain_entity.py:23-24
# --------------------------------------------------------------------------- #
ARENA = {
  "apron":     (0.08, 0.12, 0.17, 1.0),   # darker than rgb2: the venue floor
  "surface":   (0.16, 0.24, 0.33, 1.0),   # between the checker's rgb1 and rgb2
  "endzone":   (0.10, 0.80, 0.35, 0.45),  # region RGB, raised alpha to read as turf
  "line":      (1.00, 0.28, 0.14, 1.0),   # touchdown RGB, opaque
  "marking":   (0.92, 0.92, 0.88, 0.95),  # boundary RGB, opaque-ish
  "post":      (0.14, 0.17, 0.21, 1.0),
  "post_cap":  (1.00, 0.28, 0.14, 1.0),
  "barrier":   (0.20, 0.24, 0.30, 1.0),
  "barrier_top": (0.92, 0.92, 0.88, 1.0),
}

# Flat decor is stacked in 1.5 mm slabs strictly BELOW z = 0.008, the underside of
# mjlab's own ``touchdown_region`` slab (pos z 0.012, half 0.004) -- no z-fighting
# with anything mjlab already draws.
Z_APRON, Z_SURFACE, Z_ENDZONE, Z_MARKING, Z_LINE = 0.00125, 0.00275, 0.00425, 0.00575, 0.00725
SLAB_HALF = 0.00075


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def md5(path: Path) -> str:
  return hashlib.md5(path.read_bytes()).hexdigest()


def dir_bytes(d: Path) -> int:
  return sum(p.stat().st_size for p in d.rglob("*") if p.is_file()) if d.exists() else 0


def gz_bytes(d: Path) -> int:
  """gzip -9 size of a directory's files, summed (what Pages serves)."""
  import gzip

  total = 0
  for p in sorted(d.rglob("*")):
    if p.is_file():
      total += len(gzip.compress(p.read_bytes(), 9))
  return total


def _fl(a) -> list:
  return np.asarray(a).astype(float).round(12).tolist()


def _s(*vals) -> str:
  return " ".join(f"{float(v):g}" for v in vals)


# --------------------------------------------------------------------------- #
# 1. build the mjlab spec
# --------------------------------------------------------------------------- #
def build_scene(game: str, task_id: str, play: bool):
  """Return (env_cfg, mjlab Scene, scenario) for one registered preset."""
  sys.path.insert(0, str(MJLAB_REPO))
  os.chdir(MJLAB_REPO)  # go2_constants resolves assets relative to SRC_PATH
  from importlib import import_module

  from mjlab.scene import Scene  # noqa: E402

  spec = GAMES[game]
  _make_env = getattr(import_module(spec["make_env"]), "_make_env")
  pmod, pattr = spec["presets"]
  presets = getattr(import_module(pmod), pattr)

  preset = next(p for p in presets if p.task_id == task_id)
  cfg = _make_env(preset, play=play)
  scene = Scene(cfg.scene, device="cpu")
  return cfg, scene, preset.scenario()


def stamp_option(spec, mj) -> None:
  """Write mjlab's MujocoCfg into the SPEC's <option>.

  mjlab applies these to the compiled MjModel at runtime
  (mjlab/sim/sim.py:76-91 ``MujocoCfg.apply``), i.e. they never reach the spec.
  A static MJCF must carry them or the browser silently runs MuJoCo defaults
  (timestep 0.002, Euler, 100 solver iterations).
  """
  import mujoco

  integrator = {
    "euler": mujoco.mjtIntegrator.mjINT_EULER,
    "implicitfast": mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
  }[mj.integrator]
  solver = {
    "newton": mujoco.mjtSolver.mjSOL_NEWTON,
    "cg": mujoco.mjtSolver.mjSOL_CG,
    "pgs": mujoco.mjtSolver.mjSOL_PGS,
  }[mj.solver]
  cone = {
    "pyramidal": mujoco.mjtCone.mjCONE_PYRAMIDAL,
    "elliptic": mujoco.mjtCone.mjCONE_ELLIPTIC,
  }[mj.cone]
  jac = {
    "auto": mujoco.mjtJacobian.mjJAC_AUTO,
    "dense": mujoco.mjtJacobian.mjJAC_DENSE,
    "sparse": mujoco.mjtJacobian.mjJAC_SPARSE,
  }[mj.jacobian]

  o = spec.option
  o.timestep = mj.timestep
  o.integrator = integrator
  o.solver = solver
  o.iterations = mj.iterations
  o.tolerance = mj.tolerance
  o.ls_iterations = mj.ls_iterations
  o.ls_tolerance = mj.ls_tolerance
  o.ccd_iterations = mj.ccd_iterations
  o.cone = cone
  o.impratio = mj.impratio
  o.jacobian = jac
  o.gravity = list(mj.gravity)
  assert not mj.multiccd, 'multiccd would need <flag multiccd="enable"/>'


# --------------------------------------------------------------------------- #
# 2. freeze the spec to XML, flatten mesh paths, rename prefixes
# --------------------------------------------------------------------------- #
def _rename(value: str) -> str:
  out = value
  for old, new in RENAME.items():
    out = out.replace(f"{old}/", f"{new}_")  # entity attach prefix
    out = out.replace(f"{old}_", f"{new}_")  # ContactSensorCfg name prefix
  return out


def freeze_xml(scene) -> tuple[ET.Element, dict[str, bytes]]:
  """Return (root element, {flat_asset_name: bytes}).

  mjlab's Scene.write() emits one copy of every mesh PER ENTITY
  (assets/attacker/base_0.obj, assets/defender/base_0.obj) because attach()
  namespaces the asset keys.  The two copies are byte-identical, so we assert
  that and ship one flat set -- 27.09 MiB instead of 54.18 MiB.  The <mesh>
  elements stay duplicated (32 entries, 16 files) so the compiled model is
  array-for-array what mjlab compiles.
  """
  with tempfile.TemporaryDirectory() as td:
    tmp = Path(td) / "scene"
    scene.write(tmp)
    root = ET.parse(tmp / "scene.xml").getroot()
    raw_assets = {
      str(p.relative_to(tmp / "assets")): p.read_bytes()
      for p in sorted((tmp / "assets").rglob("*"))
      if p.is_file()
    }

  # flatten mesh file="attacker/base_0.obj" -> file="base_0.obj"
  flat: dict[str, bytes] = {}
  for key, data in raw_assets.items():
    name = os.path.basename(key)
    if name in flat:
      assert flat[name] == data, f"asset {key} differs from an earlier {name}"
    flat[name] = data
  for mesh in root.iter("mesh"):
    f = mesh.get("file")
    if f:
      mesh.set("file", os.path.basename(f))

  # rename entity prefixes everywhere EXCEPT the (already flattened) file= paths
  for el in root.iter():
    for k, v in list(el.attrib.items()):
      if k == "file":
        continue
      el.attrib[k] = _rename(v)

  blob = ET.tostring(root, encoding="unicode")
  for old in RENAME:
    assert old not in blob, f"{old!r} survived the rename"
  return root, flat


def write_xml(root: ET.Element, path: Path) -> None:
  root = ET.fromstring(ET.tostring(root, encoding="unicode"))  # deep copy
  ET.indent(root, space="  ")
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(ET.tostring(root, encoding="unicode").rstrip() + "\n")


# --------------------------------------------------------------------------- #
# 3. the ARENA -- decoration only
# --------------------------------------------------------------------------- #
DECOR_BODY = "arena_decor"


def add_arena(root: ET.Element, cx, cy, hx, hy, line_x, line_x_b) -> list[str]:
  """Append a purely decorative body to <worldbody>; return the geom names.

  EVERY geom here is ``contype=0 conaffinity=0 density=0`` -- MuJoCo excludes
  such geoms from the broadphase entirely and they carry no mass, so they cannot
  collide with a robot and cannot change any body's inertia.  The body itself
  carries no joint, so it adds no dof.  It is appended LAST so every preexisting
  body/geom/site id is preserved.  ``_prove_decor`` re-checks all of this on the
  compiled model and then proves two 2 s rollouts are bit-identical.
  """
  world = root.find("worldbody")
  assert world is not None
  body = ET.SubElement(world, "body", {"name": DECOR_BODY})
  names: list[str] = []

  def add(name, gtype, pos, size, rgba, quat=None):
    attrs = {
      "name": name, "type": gtype, "pos": _s(*pos), "size": _s(*size),
      "rgba": _s(*rgba), "contype": "0", "conaffinity": "0", "density": "0",
      "group": "2",
    }
    if quat is not None:
      attrs["quat"] = _s(*quat)
    ET.SubElement(body, "geom", attrs)
    names.append(name)

  # --- the floor: a venue apron, then the field surface ---------------------
  add("decor_apron", "box", (cx, cy, Z_APRON),
      (hx + 0.70, hy + 0.70, SLAB_HALF), ARENA["apron"])
  add("decor_surface", "box", (cx, cy, Z_SURFACE),
      (hx, hy, SLAB_HALF), ARENA["surface"])

  # --- end zones: the run-out behind each scoring line ----------------------
  lines = [("", line_x)] + ([("_b", line_x_b)] if line_x_b is not None else [])
  for suffix, lx in lines:
    # lx is env-local and signed; the run-out runs from it to the near wall.
    wall = cx + hx if lx >= cx else cx - hx
    half = abs(wall - lx) / 2.0
    if half > 0.0:
      add(f"decor_endzone{suffix}", "box", ((lx + wall) / 2.0, cy, Z_ENDZONE),
          (half, hy, SLAB_HALF), ARENA["endzone"])
    add(f"decor_line{suffix}", "box", (lx, cy, Z_LINE),
        (0.035, hy, SLAB_HALF), ARENA["line"])

  # --- pitch markings: touch lines inset so they sit inside mjlab's rails ---
  m = 0.04
  add("decor_touchline_y_min", "box", (cx, cy - hy + m, Z_MARKING),
      (hx, m, SLAB_HALF), ARENA["marking"])
  add("decor_touchline_y_max", "box", (cx, cy + hy - m, Z_MARKING),
      (hx, m, SLAB_HALF), ARENA["marking"])
  add("decor_goalline_x_min", "box", (cx - hx + m, cy, Z_MARKING),
      (m, hy, SLAB_HALF), ARENA["marking"])
  add("decor_goalline_x_max", "box", (cx + hx - m, cy, Z_MARKING),
      (m, hy, SLAB_HALF), ARENA["marking"])
  add("decor_halfway", "box", (cx, cy, Z_MARKING),
      (0.025, hy, SLAB_HALF), ARENA["marking"])
  add("decor_centre_circle", "cylinder", (cx, cy, Z_MARKING),
      (0.45, SLAB_HALF), (*ARENA["marking"][:3], 0.30))

  # --- corner posts, clear of the out-of-bounds rectangle -------------------
  for i, (sx, sy) in enumerate(((-1, -1), (-1, 1), (1, -1), (1, 1))):
    px, py = cx + sx * (hx + 0.20), cy + sy * (hy + 0.20)
    add(f"decor_post_{i}", "cylinder", (px, py, 0.26), (0.05, 0.26), ARENA["post"])
    add(f"decor_post_cap_{i}", "sphere", (px, py, 0.55), (0.065,), ARENA["post_cap"])

  # --- low barriers, 0.30 m outside the field rectangle ---------------------
  bx, by, bh = hx + 0.32, hy + 0.32, 0.11
  for nm, pos, size in (
    ("decor_barrier_y_min", (cx, cy - by, bh), (bx, 0.05, bh)),
    ("decor_barrier_y_max", (cx, cy + by, bh), (bx, 0.05, bh)),
    ("decor_barrier_x_min", (cx - bx, cy, bh), (0.05, by, bh)),
    ("decor_barrier_x_max", (cx + bx, cy, bh), (0.05, by, bh)),
  ):
    add(nm, "box", pos, size, ARENA["barrier"])
    add(nm + "_top", "box", (pos[0], pos[1], 2 * bh + 0.012),
        (size[0], size[1], 0.012), ARENA["barrier_top"])
  return names


# --------------------------------------------------------------------------- #
# 4. model introspection + the mjlab-vs-exported diff
# --------------------------------------------------------------------------- #
_ARRAYS = (
  # bodies / inertia
  "body_parentid body_rootid body_weldid body_jntnum body_jntadr body_dofnum "
  "body_dofadr body_pos body_quat body_ipos body_iquat body_mass body_inertia "
  # joints / dofs
  "jnt_type jnt_bodyid jnt_group jnt_limited jnt_qposadr jnt_dofadr jnt_pos "
  "jnt_axis jnt_stiffness jnt_range jnt_margin "
  "dof_bodyid dof_jntid dof_parentid dof_armature dof_damping dof_frictionloss "
  "dof_invweight0 "
  # geoms
  "geom_type geom_contype geom_conaffinity geom_condim geom_bodyid geom_dataid "
  "geom_group geom_priority geom_solmix geom_solref geom_solimp geom_size "
  "geom_pos geom_quat geom_friction geom_margin geom_gap geom_rgba "
  # actuators
  "actuator_trntype actuator_dyntype actuator_gaintype actuator_biastype "
  "actuator_trnid actuator_group actuator_ctrllimited actuator_forcelimited "
  "actuator_actlimited actuator_dynprm actuator_gainprm actuator_biasprm "
  "actuator_ctrlrange actuator_forcerange actuator_gear "
  # sites / keyframes / meshes
  "site_type site_bodyid site_pos site_quat site_size "
  "key_time key_qpos key_qvel key_ctrl "
  "mesh_vertadr mesh_vertnum mesh_faceadr mesh_facenum mesh_pos mesh_quat "
  "mesh_vert mesh_face mesh_normal "
  # sensors
  "sensor_type sensor_datatype sensor_objtype sensor_objid sensor_reftype "
  "sensor_refid sensor_dim sensor_adr sensor_intprm"
).split()

_MESH_ARRAYS = {n for n in _ARRAYS if n.startswith("mesh_")}

_OPT = (
  "timestep integrator solver iterations tolerance ls_iterations ls_tolerance "
  "ccd_iterations cone impratio jacobian gravity disableflags enableflags "
  "wind density viscosity o_margin noslip_iterations sdf_iterations"
).split()

_COUNTS = "nq nv nu nbody njnt ngeom nsite nmesh nsensor nkey na nmocap".split()

# Float arrays only have to survive the decimal round trip of MJCF text.
_FLOAT_ATOL = 1e-12

# One documented, dynamics-free exception.  mjlab's terrain adds a cosmetic
# marker site ``env_origin_0`` of type mjGEOM_SPHERE with size (0.3, 0.3, 0.3);
# MuJoCo's serializer writes a sphere site's size as the single radius, so the
# reload gives (0.3, 0.005, 0.005).  Only size[0] is read for a sphere, in
# physics and in rendering alike, and sites carry no dynamics at all.
_BENIGN = {"site_size": (0, "env_origin_0 sphere radius; size[1:] unused for a sphere")}


def _diff_models(ref, got, skip: set[str] = frozenset(),
                 ignore_rows: dict | None = None) -> tuple[list[str], list[str]]:
  """(failures, benign notes) between two compiled models."""
  ignore_rows = ignore_rows or {}
  bad: list[str] = []
  notes: list[str] = []
  for n in _COUNTS:
    if getattr(ref, n) != getattr(got, n):
      bad.append(f"count {n}: ref {getattr(ref, n)} != got {getattr(got, n)}")
  for n in _OPT:
    a, b = getattr(ref.opt, n), getattr(got.opt, n)
    if not np.allclose(np.asarray(a, float), np.asarray(b, float),
                       rtol=0, atol=_FLOAT_ATOL):
      bad.append(f"opt.{n}: ref {a} != got {b}")
  for n in ("extent", "meansize", "meanmass", "meaninertia"):
    a, b = getattr(ref.stat, n), getattr(got.stat, n)
    if not math.isclose(float(a), float(b), rel_tol=1e-12, abs_tol=0.0):
      bad.append(f"stat.{n}: ref {a} != got {b}")
  if bad:
    return bad, notes  # shapes will not line up; stop here
  for n in _ARRAYS:
    if n in skip:
      continue
    a, b = np.asarray(getattr(ref, n)), np.asarray(getattr(got, n))
    if a.shape != b.shape:
      bad.append(f"{n}: shape {a.shape} != {b.shape}")
      continue
    if np.array_equal(a, b):
      continue
    d = np.abs(a.astype(float) - b.astype(float))
    if n in _BENIGN:
      row, why = _BENIGN[n]
      d = d.copy()
      d[row] = 0.0
      notes.append(f"{n}[{row}] differs -- {why}")
    if n in ignore_rows:
      d = d.copy()
      d[ignore_rows[n]] = 0.0
    if np.issubdtype(a.dtype, np.integer):
      if d.max() > 0:
        bad.append(f"{n}: {int((d > 0).sum())} differing int entries")
    elif d.max() > _FLOAT_ATOL:
      bad.append(f"{n}: {int((d > _FLOAT_ATOL).sum())} entries beyond "
                 f"{_FLOAT_ATOL:g}, max |d| {d.max():g}")
    elif d.max() > 0:
      notes.append(f"{n}: decimal round trip only, max |d| {d.max():g}")
  return bad, notes


def name_list(model, objtype, n) -> list[str]:
  import mujoco

  return [mujoco.mj_id2name(model, objtype, i) for i in range(n)]


# --------------------------------------------------------------------------- #
# 5. rollouts (used as the physics-unchanged proof AND for the keyframes)
# --------------------------------------------------------------------------- #
def set_pose(model, data, layout, robot, x, y, yaw, z, qj) -> None:
  adr = layout[robot]["freeQposAdr"]
  data.qpos[adr: adr + 3] = [x, y, z]
  data.qpos[adr + 3: adr + 7] = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
  for k, a in enumerate(layout[robot]["jointQposAdr"]):
    data.qpos[a] = qj[k]
  vadr = layout[robot]["freeDofAdr"]
  data.qvel[vadr: vadr + 6] = 0.0
  for a in layout[robot]["jointDofAdr"]:
    data.qvel[a] = 0.0


def quat_yaw(q) -> float:
  """Yaw of a (w, x, y, z) quaternion."""
  return math.atan2(2 * (q[0] * q[3] + q[1] * q[2]),
                    1 - 2 * (q[2] * q[2] + q[3] * q[3]))


def quat_mul(a, b):
  w1, x1, y1, z1 = a
  w2, x2, y2, z2 = b
  return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                   w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                   w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                   w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def set_pose_quat(model, data, layout, robot, x, y, z, quat, qj) -> None:
  adr = layout[robot]["freeQposAdr"]
  data.qpos[adr: adr + 3] = [x, y, z]
  data.qpos[adr + 3: adr + 7] = quat
  for k, a in enumerate(layout[robot]["jointQposAdr"]):
    data.qpos[a] = qj[k]
  vadr = layout[robot]["freeDofAdr"]
  data.qvel[vadr: vadr + 6] = 0.0
  for a in layout[robot]["jointDofAdr"]:
    data.qvel[a] = 0.0


def place(model, data, layout, spawn, default_q, z=0.32, qj=None) -> None:
  import mujoco

  mujoco.mj_resetData(model, data)
  for r in ("a", "b"):
    s = spawn[r]
    set_pose(model, data, layout, r, s["x"], s["y"], s["yaw"], z,
             default_q if qj is None else qj[r])
  for r in ("a", "b"):
    for k, act in enumerate(layout[r]["actuatorIdsJointOrder"]):
      data.ctrl[act] = default_q[k]
  # cheap proof that actuatorIdsJointOrder really crosses joint order ->
  # mjModel's hip x4 / thigh x4 / calf x4 actuator order
  assert np.allclose(np.asarray(data.ctrl), np.asarray(model.key_ctrl[0])), \
      "ctrl written in joint order does not reproduce mjlab's merged key_ctrl"
  mujoco.mj_forward(model, data)


def rollout(model, data, layout, default_q, seconds, mode="hold"):
  """Step `seconds` from the current state.  mode 'hold' = ctrl at the default
  pose; mode 'wiggle' = a per-joint sinusoid that drives the legs into the floor
  and the two robots' contact sets into each other -- the harder physics-equality
  probe of the two."""
  import mujoco

  n = int(round(seconds / model.opt.timestep))
  trace = np.empty((n, 2))
  for i in range(n):
    if mode == "wiggle":
      t = i * model.opt.timestep
      for r in ("a", "b"):
        ph = 0.0 if r == "a" else math.pi / 3.0
        for k, act in enumerate(layout[r]["actuatorIdsJointOrder"]):
          data.ctrl[act] = default_q[k] + 0.35 * math.sin(2 * math.pi * 1.7 * t
                                                          + ph + 0.7 * k)
    mujoco.mj_step(model, data)
    trace[i] = (data.qpos[layout["a"]["freeQposAdr"] + 2],
                data.qpos[layout["b"]["freeQposAdr"] + 2])
  return trace


def _probe(model, layout, spawn, default_q, mode):
  """A full place+rollout, returning the final (qpos, qvel) for equality tests."""
  import mujoco

  data = mujoco.MjData(model)
  place(model, data, layout, spawn, default_q)
  rollout(model, data, layout, default_q, SETTLE_SECONDS, mode=mode)
  return np.array(data.qpos), np.array(data.qvel)


def mesh_geom_world_bounds(model, layout, spawn, default_q) -> dict:
  """World-space AABB of every MESH geom's own vertices, at a fixed pose.

  MuJoCo re-parametrises a mesh geom on compile: it moves the mesh vertices into
  the mesh's inertia frame (``mesh_pos``/``mesh_quat``) and compensates in
  ``geom_pos``/``geom_quat``.  Decimating a mesh therefore MOVES geom_pos/quat/size
  by centimetres while the rendered surface stays put.  Comparing the transformed
  vertices is the check that actually means "the dog is drawn in the same place".
  """
  import mujoco

  data = mujoco.MjData(model)
  place(model, data, layout, spawn, default_q)
  out = {}
  names = name_list(model, mujoco.mjtObj.mjOBJ_GEOM, model.ngeom)
  for g in range(model.ngeom):
    if int(model.geom_type[g]) != int(mujoco.mjtGeom.mjGEOM_MESH):
      continue
    m = int(model.geom_dataid[g])
    v0, vn = int(model.mesh_vertadr[m]), int(model.mesh_vertnum[m])
    v = np.asarray(model.mesh_vert[v0: v0 + vn], dtype=np.float64)
    R = np.asarray(data.geom_xmat[g]).reshape(3, 3)
    w = np.asarray(data.geom_xpos[g]) + v @ R.T
    out[names[g]] = (w.min(axis=0), w.max(axis=0))
  return out


def prove_identical(a_model, b_model, layout, spawn, default_q, label) -> dict:
  """Two models must produce bit-identical 2 s rollouts under both ctrl modes."""
  out = {}
  for mode in ("hold", "wiggle"):
    qa, va = _probe(a_model, layout, spawn, default_q, mode)
    qb, vb = _probe(b_model, layout, spawn, default_q, mode)
    dq = float(np.abs(qa - qb).max())
    dv = float(np.abs(va - vb).max())
    out[mode] = {"maxAbsDqpos": dq, "maxAbsDqvel": dv}
    print(f"        {label} [{mode}] max|dqpos| {dq:g}  max|dqvel| {dv:g}")
    assert dq == 0.0 and dv == 0.0, f"{label}/{mode} changed the physics"
  return out


# --------------------------------------------------------------------------- #
# 6. lite meshes
# --------------------------------------------------------------------------- #
def write_obj(verts: np.ndarray, faces: np.ndarray, path: Path) -> None:
  buf = io.StringIO()
  np.savetxt(buf, verts, fmt=f"v {LITE_COORD_FMT} {LITE_COORD_FMT} {LITE_COORD_FMT}")
  np.savetxt(buf, faces + 1, fmt="f %d %d %d")
  path.write_text(buf.getvalue())


def deviation_mm(ma, mb, n=20000, seed=0) -> dict:
  """Surface-to-surface deviation between two trimeshes, by dense sampling.

  ``drawnToOrigP95_mm`` is the one that decides whether the reduced mesh is
  shippable: it bounds how far anything we DRAW sits from the real Go2 surface.
  ``origToDrawnMean_mm`` / ``lostOver3mm_pct`` measure the other direction --
  original surface that no longer has a neighbour -- which on these CAD meshes is
  dominated by internal structure you cannot see from outside.
  """
  import trimesh  # noqa: F401
  from scipy.spatial import cKDTree

  pa, _ = trimesh.sample.sample_surface(ma, n, seed=seed)  # original
  pb, _ = trimesh.sample.sample_surface(mb, n, seed=seed)  # reduced
  d_o2d = cKDTree(pb).query(pa)[0] * 1000.0
  d_d2o = cKDTree(pa).query(pb)[0] * 1000.0
  bbox = float(np.abs(np.asarray(ma.bounds) - np.asarray(mb.bounds)).max()) * 1000.0
  return {
    "drawnToOrigMean_mm": round(float(d_d2o.mean()), 4),
    "drawnToOrigP95_mm": round(float(np.percentile(d_d2o, 95)), 4),
    "origToDrawnMean_mm": round(float(d_o2d.mean()), 4),
    "lostOver3mm_pct": round(100.0 * float((d_o2d > 3.0).mean()), 3),
    "bboxShift_mm": round(bbox, 4),
  }


def weld(data: bytes):
  """Load an OBJ and weld position-coincident vertices.  Exactly lossless: the
  triangle set is unchanged, only re-indexed (degenerate faces, which a weld can
  only create from zero-area triangles, are dropped)."""
  import trimesh

  raw = trimesh.load(io.BytesIO(data), file_type="obj", process=False, force="mesh")
  m = trimesh.Trimesh(np.asarray(raw.vertices), np.asarray(raw.faces), process=False)
  m.merge_vertices(digits_vertex=LITE_WELD_DIGITS)
  m.update_faces(m.nondegenerate_faces())
  m.remove_unreferenced_vertices()
  return raw, m


def make_lite(assets: dict[str, bytes], out_dir: Path, frac: float, agg: int) -> dict:
  import fast_simplification
  import trimesh

  if out_dir.exists():
    shutil.rmtree(out_dir)
  out_dir.mkdir(parents=True)
  report = {}
  for name, data in sorted(assets.items()):
    raw, src = weld(data)
    n = len(src.faces)
    target = max(LITE_MIN_FACES, int(round(n * frac)))
    if target >= n:
      vo, fo = np.asarray(src.vertices), np.asarray(src.faces)
    else:
      vo, fo = fast_simplification.simplify(
        np.asarray(src.vertices, dtype=np.float32),
        np.asarray(src.faces, dtype=np.int32), 1.0 - target / n, agg=agg)
      vo = np.asarray(vo, dtype=np.float64)
      fo = np.asarray(fo, dtype=np.int64)
    write_obj(vo, fo, out_dir / name)
    dev = deviation_mm(src, trimesh.Trimesh(vo, fo, process=False))
    assert dev["drawnToOrigP95_mm"] <= LITE_MAX_P95_MM, (name, dev)
    assert dev["bboxShift_mm"] <= LITE_MAX_BBOX_MM, (name, dev)
    report[name] = {
      "facesRaw": int(len(raw.faces)), "vertsRaw": int(len(raw.vertices)),
      "facesWelded": int(n), "vertsWelded": int(len(src.vertices)),
      "facesOut": int(len(fo)), "vertsOut": int(len(vo)),
      "bytesIn": len(data), "bytesOut": (out_dir / name).stat().st_size,
      "deviation": dev,
    }
  return report


# --------------------------------------------------------------------------- #
# 7. one game, end to end
# --------------------------------------------------------------------------- #
def export_game(game: str, out: Path, args) -> dict:
  import mujoco

  spec = GAMES[game]
  task_id, twin_id = spec["task"], spec["twin"]
  print(f"\n================ {game.upper()}  ({task_id}) ================")

  # ---- 1. mjlab reference model ------------------------------------------- #
  cfg, scene, scenario = build_scene(game, task_id, play=True)
  stamp_option(scene.spec, cfg.sim.mujoco)
  ref = scene.compile()
  cfg.sim.mujoco.apply(ref)  # no-op now, but this is what mjlab does at runtime
  print(f"[mjlab] nq {ref.nq} nv {ref.nv} nu {ref.nu} nbody {ref.nbody} "
        f"ngeom {ref.ngeom} nsite {ref.nsite} nsensor {ref.nsensor}")

  twin_ok = False
  if not args.no_twin_check:
    cfg2, scene2, _ = build_scene(game, twin_id, play=True)
    stamp_option(scene2.spec, cfg2.sim.mujoco)
    twin = scene2.compile()
    cfg2.sim.mujoco.apply(twin)
    bad, _ = _diff_models(ref, twin)
    assert not bad, f"Clean vs Clean-Filtered plants differ: {bad}"
    twin_ok = True
    print(f"[twin ] {twin_id}: plant identical ({len(_ARRAYS)} arrays + opt + counts)")

  # ---- 2. freeze ----------------------------------------------------------- #
  root, assets = freeze_xml(scene)
  work = Path(tempfile.mkdtemp(prefix=f"scene_{game}_"))
  plain_xml = work / "plain.xml"
  write_xml(root, plain_xml)
  full_dir = work / "assets_full"
  full_dir.mkdir()
  for name, data in sorted(assets.items()):
    (full_dir / name).write_bytes(data)
  shutil.copytree(full_dir, work / "assets")  # plain.xml's meshdir
  bytes_full = dir_bytes(full_dir)
  print(f"[freeze] plain.xml {plain_xml.stat().st_size} B + {len(assets)} meshes "
        f"{bytes_full} B")

  # ---- 3. PROOF P1: exported plain XML == mjlab ---------------------------- #
  got = mujoco.MjModel.from_xml_path(str(plain_xml))
  bad, notes = _diff_models(ref, got)
  if bad:
    print("EXPORT DIFFERS FROM MJLAB:")
    for b in bad:
      print("   ", b)
    raise SystemExit(1)
  print(f"[P1   ] exported XML == mjlab compiled model: {len(_COUNTS)} counts, "
        f"{len(_OPT)} opt fields, 4 stat fields, {len(_ARRAYS)} arrays")
  for nt in notes:
    print(f"        benign: {nt}")

  # ---- 4. layout ----------------------------------------------------------- #
  jnames = name_list(got, mujoco.mjtObj.mjOBJ_JOINT, got.njnt)
  anames = name_list(got, mujoco.mjtObj.mjOBJ_ACTUATOR, got.nu)
  bnames = name_list(got, mujoco.mjtObj.mjOBJ_BODY, got.nbody)
  gnames = name_list(got, mujoco.mjtObj.mjOBJ_GEOM, got.ngeom)
  snames = name_list(got, mujoco.mjtObj.mjOBJ_SENSOR, got.nsensor)

  CANON = [f"{leg}_{j}_joint" for leg in ("FL", "FR", "RL", "RR")
           for j in ("hip", "thigh", "calf")]
  layout: dict[str, dict] = {}
  for r in ("a", "b"):
    free = f"{r}_floating_base_joint"
    fid = jnames.index(free)
    assert got.jnt_type[fid] == mujoco.mjtJoint.mjJNT_FREE
    jids = [jnames.index(f"{r}_{n}") for n in CANON]
    act_of_jnt = {int(got.actuator_trnid[i, 0]): i for i in range(got.nu)}
    aids = [act_of_jnt[j] for j in jids]
    bids = [i for i, n in enumerate(bnames) if n and n.startswith(f"{r}_")]
    gids = [i for i, n in enumerate(gnames) if n and n.startswith(f"{r}_")]
    layout[r] = {
      "mjlabEntity": [k for k, v in RENAME.items() if v == r][0],
      "seat": spec["seats"][r],
      "prefix": f"{r}_",
      "freeJointName": free,
      "freeJointId": fid,
      "freeQposAdr": int(got.jnt_qposadr[fid]),
      "freeDofAdr": int(got.jnt_dofadr[fid]),
      "jointNames": [f"{r}_{n}" for n in CANON],
      "jointIds": [int(i) for i in jids],
      "jointQposAdr": [int(got.jnt_qposadr[i]) for i in jids],
      "jointDofAdr": [int(got.jnt_dofadr[i]) for i in jids],
      "actuatorNames": [anames[i] for i in aids],
      "actuatorIdsJointOrder": [int(i) for i in aids],
      "baseBodyName": f"{r}_base_link",
      "baseBodyId": bnames.index(f"{r}_base_link"),
      "bodyIds": [int(i) for i in bids],
      "bodyNames": [bnames[i] for i in bids],
      "geomIds": [int(i) for i in gids],
      "collisionGeoms": [gnames[i] for i in gids if gnames[i].endswith("_collision")],
      "footGeoms": [f"{r}_{n}_foot_collision" for n in ("FL", "FR", "RL", "RR")],
      "contype": int(got.geom_contype[gnames.index(f"{r}_base1_collision")]),
      "conaffinity": int(got.geom_conaffinity[gnames.index(f"{r}_base1_collision")]),
    }
  assert layout["a"]["freeQposAdr"] == 0 and layout["b"]["freeQposAdr"] == 19

  kp = {r: [float(got.actuator_gainprm[i, 0])
            for i in layout[r]["actuatorIdsJointOrder"]] for r in ("a", "b")}
  kd = {r: [float(-got.actuator_biasprm[i, 2])
            for i in layout[r]["actuatorIdsJointOrder"]] for r in ("a", "b")}
  assert kp["a"] == kp["b"] and kd["a"] == kd["b"], "seats must share gains"
  assert kp["a"] == [20, 20, 40] * 4 and kd["a"] == [1, 1, 2] * 4, (kp["a"], kd["a"])

  jlo = [float(got.jnt_range[i, 0]) for i in layout["a"]["jointIds"]]
  jhi = [float(got.jnt_range[i, 1]) for i in layout["a"]["jointIds"]]
  soft = float(
    __import__("src.assets.robots.unitree_go2.go2_constants", fromlist=["x"])
    .GO2_ARTICULATION.soft_joint_pos_limit_factor
  )
  mid = [(a + b) / 2 for a, b in zip(jlo, jhi)]
  half = [(b - a) / 2 * soft for a, b in zip(jlo, jhi)]
  qlo = [m - h for m, h in zip(mid, half)]
  qhi = [m + h for m, h in zip(mid, half)]

  default_q = [-0.1, 0.9, -1.8, 0.1, 0.9, -1.8, -0.1, 0.9, -1.8, 0.1, 0.9, -1.8]
  kq = np.asarray(got.key_qpos[0])
  for r in ("a", "b"):
    assert np.allclose([float(kq[a]) for a in layout[r]["jointQposAdr"]], default_q)

  # ---- 5. field geometry (read off the live scenario) ---------------------- #
  cx, cy = float(scenario.field_center[0]), float(scenario.field_center[1])
  hx, hy = float(scenario.field_size[0]) / 2.0, float(scenario.field_size[1]) / 2.0
  # game_env_cfg._field_marker_spec_fn:81+ -- the line the markers, the referee
  # and the obs all use is field_center.x + line_x.  sym's line_x is already a
  # magnitude about a field centred at 0; asym's is field-local 1.7 about +0.2.
  line_env_x = round(cx + float(scenario.line_x), 12)
  line_b = getattr(scenario, "line_x_b", None)
  line_env_x_b = None if line_b is None else round(cx + float(line_b), 12)
  assert abs(line_env_x - 1.9) < 1e-9, line_env_x
  if line_env_x_b is not None:
    assert abs(line_env_x_b + 1.9) < 1e-9, line_env_x_b
  print(f"[field] size {2*hx} x {2*hy} @ ({cx}, {cy})  line(s) "
        f"{line_env_x}" + (f" / {line_env_x_b}" if line_env_x_b is not None else "")
        + f"  oob x |x-{cx}|>{hx}, |y|>{hy}")

  # the config.js spawn must be the one this export is keyed to
  spawn_cfg = spec["spawnConfig"]
  spawn_train = {
    r: {"x": float(scenario.spawns[e].pos[0]), "y": float(scenario.spawns[e].pos[1]),
        "z": float(scenario.spawns[e].pos[2]), "yaw": float(scenario.spawns[e].yaw)}
    for r, e in (("a", scenario.attacker_role), ("b", scenario.defender_role))
  }
  if game == "asym":
    for r in ("a", "b"):
      for k in ("x", "y", "yaw"):
        assert abs(spawn_cfg[r][k] - spawn_train[r][k]) < 1e-9, (r, k)

  # ---- 6. lite meshes + PROOF P2 ------------------------------------------- #
  lite_report = None
  lite_proof = None
  ship_dir = full_dir
  if not args.no_lite:
    lite_dir = work / "assets_lite"
    lite_report = make_lite(assets, lite_dir, args.lite_fraction, args.lite_aggression)
    probe = work / "lite_probe"
    probe.mkdir()
    shutil.copy(plain_xml, probe / "plain.xml")
    shutil.copytree(lite_dir, probe / "assets")
    lite_model = mujoco.MjModel.from_xml_path(str(probe / "plain.xml"))
    # A mesh geom's pos/quat/size are DERIVED from its mesh (MuJoCo re-frames the
    # vertices into the mesh inertia frame and compensates in the geom).  Those
    # rows move; the rendered surface must not.  Everything else must be equal.
    is_mesh = np.asarray(got.geom_type) == int(mujoco.mjtGeom.mjGEOM_MESH)
    bad_l, _ = _diff_models(got, lite_model, skip=_MESH_ARRAYS,
                            ignore_rows={"geom_pos": is_mesh, "geom_quat": is_mesh,
                                         "geom_size": is_mesh})
    assert not bad_l, f"lite meshes changed a non-mesh array: {bad_l}"
    ba = mesh_geom_world_bounds(got, layout, spawn_cfg, default_q)
    bb = mesh_geom_world_bounds(lite_model, layout, spawn_cfg, default_q)
    assert set(ba) == set(bb)
    aabb_mm = max(max(float(np.abs(ba[k][i] - bb[k][i]).max()) for i in (0, 1))
                  for k in ba) * 1000.0
    dev = max(v["deviation"]["drawnToOrigP95_mm"] for v in lite_report.values())
    devm = max(v["deviation"]["drawnToOrigMean_mm"] for v in lite_report.values())
    devb = max(v["deviation"]["bboxShift_mm"] for v in lite_report.values())
    print(f"[lite ] {bytes_full} B -> {dir_bytes(lite_dir)} B  "
          f"(gz-9 {gz_bytes(full_dir)} -> {gz_bytes(lite_dir)})  faces "
          f"{sum(v['facesRaw'] for v in lite_report.values())} -> "
          f"{sum(v['facesOut'] for v in lite_report.values())}  "
          f"drawn-to-original mean<={devm:.3f} p95<={dev:.3f} mm, "
          f"bbox shift <={devb:.3f} mm")
    print(f"[P2   ] non-mesh arrays identical "
          f"({len(_ARRAYS) - len(_MESH_ARRAYS)} arrays + opt + counts; mesh-geom "
          f"pos/quat/size are derived and re-framed); world AABB of every mesh "
          f"geom moves at most {aabb_mm:.3f} mm")
    assert aabb_mm < 5.0, f"mesh geoms moved {aabb_mm} mm in world space"
    prove_identical(got, lite_model, layout, spawn_cfg, default_q, "P2 lite")
    lite_proof = {"maxMeshGeomWorldAabbShift_mm": round(aabb_mm, 4),
                  "maxDrawnToOrigMean_mm": round(devm, 4),
                  "maxDrawnToOrigP95_mm": round(dev, 4),
                  "maxBboxShift_mm": round(devb, 4),
                  "gates": {"drawnToOrigP95_mm": LITE_MAX_P95_MM,
                            "bboxShift_mm": LITE_MAX_BBOX_MM}}
    ship_dir = lite_dir

  # ---- 7. arena + PROOF P3 -------------------------------------------------- #
  decor_names = add_arena(root, cx, cy, hx, hy, line_env_x, line_env_x_b)
  final_xml = work / "scene.xml"
  write_xml(root, final_xml)
  fin_probe = work / "final_probe"
  fin_probe.mkdir()
  shutil.copy(final_xml, fin_probe / "scene.xml")
  shutil.copytree(ship_dir, fin_probe / "assets")
  fin = mujoco.MjModel.from_xml_path(str(fin_probe / "scene.xml"))
  shipped = mujoco.MjModel.from_xml_path(str(work / "lite_probe" / "plain.xml")) \
      if not args.no_lite else got

  # 7a. nothing that carries dynamics may have moved
  for n in "nq nv nu njnt nmesh nsensor nkey na nmocap nsite".split():
    assert getattr(fin, n) == getattr(shipped, n), \
        f"decor changed {n}: {getattr(shipped, n)} -> {getattr(fin, n)}"
  assert fin.nbody == shipped.nbody + 1 and fin.ngeom == shipped.ngeom + len(decor_names)
  # 7b. every preexisting id preserved (decor appended last)
  for n in _ARRAYS:
    a = np.asarray(getattr(shipped, n))
    b = np.asarray(getattr(fin, n))
    assert np.array_equal(a, b[: len(a)]), f"decor shifted array {n}"
  # 7c. the decor itself is inert
  fnames = name_list(fin, mujoco.mjtObj.mjOBJ_GEOM, fin.ngeom)
  dbody = name_list(fin, mujoco.mjtObj.mjOBJ_BODY, fin.nbody).index(DECOR_BODY)
  assert float(fin.body_mass[dbody]) == 0.0 and int(fin.body_dofnum[dbody]) == 0
  assert int(fin.body_jntnum[dbody]) == 0
  dids = [i for i, n in enumerate(fnames) if n in set(decor_names)]
  assert len(dids) == len(decor_names) and min(dids) == shipped.ngeom
  for i in dids:
    assert int(fin.geom_contype[i]) == 0 and int(fin.geom_conaffinity[i]) == 0, fnames[i]
    assert int(fin.geom_bodyid[i]) == dbody
  assert float(np.abs(np.asarray(fin.body_mass) - np.asarray(
      list(shipped.body_mass) + [0.0])).max()) == 0.0
  print(f"[P3   ] arena: +1 body, +{len(decor_names)} geoms, all contype=0 "
        f"conaffinity=0 density=0, body mass 0, nv/nq/nu/nsite unchanged, "
        f"every preexisting id preserved")
  decor_proof = prove_identical(shipped, fin, layout, spawn_cfg, default_q, "P3 arena")

  # ---- 8. keyframes --------------------------------------------------------- #
  data = mujoco.MjData(fin)
  keyframes: dict[str, dict] = {}
  base_z: dict[str, dict] = {}
  for setname, sp in (("config", spawn_cfg), ("trainingNominal", spawn_train)):
    place(fin, data, layout, sp, default_q)
    reset_kf = {"qpos": _fl(data.qpos), "qvel": _fl(data.qvel), "ctrl": _fl(data.ctrl)}
    trace = rollout(fin, data, layout, default_q, SETTLE_SECONDS, mode="hold")
    assert np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all(), "NaN"
    settled_q = {r: [float(data.qpos[a]) for a in layout[r]["jointQposAdr"]]
                 for r in ("a", "b")}
    settled_z = {r: float(data.qpos[layout[r]["freeQposAdr"] + 2]) for r in ("a", "b")}
    settled_quat = {r: np.array(data.qpos[layout[r]["freeQposAdr"] + 3:
                                          layout[r]["freeQposAdr"] + 7])
                    for r in ("a", "b")}
    settled = {"qpos": _fl(data.qpos), "qvel": _fl(data.qvel), "ctrl": _fl(data.ctrl),
               "baseZ": settled_z,
               "drift": {r: [round(float(data.qpos[layout[r]["freeQposAdr"]]) - sp[r]["x"], 6),
                             round(float(data.qpos[layout[r]["freeQposAdr"] + 1]) - sp[r]["y"], 6)]
                         for r in ("a", "b")}}
    # The game-useful one: settled shape, put BACK on the spawn, at rest.
    # Keep the settled ROLL/PITCH -- the rear shins rest on the floor at these
    # gains, so the trunk carries a few degrees of nose-up pitch; flattening it
    # while keeping the sagged joints drives the feet into the ground and the
    # state bounces (measured: 83 mm of drift in 0.5 s).  Only the yaw is turned
    # back to the spawn heading.
    for r in ("a", "b"):
      s = sp[r]
      dyaw = s["yaw"] - quat_yaw(settled_quat[r])
      qz = np.array([math.cos(dyaw / 2), 0.0, 0.0, math.sin(dyaw / 2)])
      set_pose_quat(fin, data, layout, r, s["x"], s["y"], settled_z[r],
                    quat_mul(qz, settled_quat[r]), settled_q[r])
    mujoco.mj_forward(fin, data)
    at_spawn = {"qpos": _fl(data.qpos), "qvel": _fl(data.qvel), "ctrl": _fl(data.ctrl),
                "baseZ": dict(settled_z)}
    q0 = np.array(data.qpos)
    rollout(fin, data, layout, default_q, 0.5, mode="hold")
    at_spawn["residual0p5s"] = {
      "maxAbsDqpos": round(float(np.abs(np.array(data.qpos) - q0).max()), 6),
      "baseZAfter": {r: round(float(data.qpos[layout[r]["freeQposAdr"] + 2]), 6)
                     for r in ("a", "b")},
    }
    for r in ("a", "b"):
      assert BASE_Z_WINDOW[0] <= settled_z[r] <= BASE_Z_WINDOW[1], \
          f"{setname}/{r} base z {settled_z[r]} outside {BASE_Z_WINDOW}"
    keyframes[setname] = {"spawnReset": reset_kf, "settled": settled,
                          "settledAtSpawn": at_spawn}
    base_z[setname] = settled_z
    print(f"[key  ] {setname}: settled base z a {settled_z['a']:.4f} "
          f"b {settled_z['b']:.4f}  (z range over 2 s "
          f"{trace.min():.4f}..{trace.max():.4f})  "
          f"re-placed residual after 0.5 s {at_spawn['residual0p5s']['maxAbsDqpos']:g}")

  # ---- 9. write the shipped tree -------------------------------------------- #
  out.mkdir(parents=True, exist_ok=True)
  shutil.copy(final_xml, out / "scene.xml")
  adir = out / "assets"
  if adir.exists():
    shutil.rmtree(adir)
  shutil.copytree(ship_dir, adir)

  go2_xml = MJLAB_REPO / "src/assets/robots/unitree_go2/xmls/go2.xml"
  scene_json = {
    "schema": "s2c_web_play/scene@2",
    "game": game,
    "generatedBy": "tools/export_scene.py",
    "provenance": {
      "mjlabTask": task_id,
      "twinTask": twin_id,
      "twinPlantVerifiedIdentical": twin_ok,
      "play": True,
      "mjlabRepo": str(MJLAB_REPO),
      "robotXml": str(go2_xml),
      "robotXmlMd5": md5(go2_xml),
      "mujocoVersion": mujoco.__version__,
      "scenario": type(scenario).__name__,
      "note": "Compiled from mjlab's own MjSpec; every array in "
              "tools/export_scene.py:_ARRAYS matches the mjlab compile (P1). "
              "The arena decoration and the reduced meshes are then proved "
              "physics-neutral by bit-identical rollouts (P2/P3).",
    },
    "files": {
      "xml": "scene.xml",
      "meshDir": "assets",
      "meshFiles": sorted(assets),
      "meshNote": "The two games ship identical copies of the same 16 reduced "
                  "meshes so each scene directory is self-contained; "
                  "app code may point both at one copy via createSim's "
                  "assetsBase if it prefers.",
    },
    "physics": {
      "timestep": float(fin.opt.timestep),
      "decimation": int(cfg.decimation),
      "controlDt": float(fin.opt.timestep) * int(cfg.decimation),
      "episodeLengthS": float(cfg.episode_length_s),
      "episodeSteps": int(math.ceil(cfg.episode_length_s /
                                    (fin.opt.timestep * cfg.decimation))),
      "integrator": "implicitfast",
      "solver": "newton",
      "iterations": int(fin.opt.iterations),
      "tolerance": float(fin.opt.tolerance),
      "lsIterations": int(fin.opt.ls_iterations),
      "lsTolerance": float(fin.opt.ls_tolerance),
      "ccdIterations": int(fin.opt.ccd_iterations),
      "cone": "pyramidal",
      "impratio": float(fin.opt.impratio),
      "jacobian": "auto",
      "gravity": _fl(fin.opt.gravity),
      "disableflags": int(fin.opt.disableflags),
      "enableflags": int(fin.opt.enableflags),
    },
    "model": {
      **{k: int(getattr(fin, k)) for k in _COUNTS},
      "bodyNames": name_list(fin, mujoco.mjtObj.mjOBJ_BODY, fin.nbody),
      "geomNames": fnames,
      "jointNames": jnames,
      "actuatorNames": anames,
      "sensorNames": snames,
      "geomBodyId": [int(i) for i in fin.geom_bodyid],
      "geomRobot": [("a" if fnames[i] and fnames[i].startswith("a_")
                     else "b" if fnames[i] and fnames[i].startswith("b_") else None)
                    for i in range(fin.ngeom)],
      "geomType": [int(i) for i in fin.geom_type],
      "geomGroup": [int(i) for i in fin.geom_group],
      "geomContype": [int(i) for i in fin.geom_contype],
      "geomConaffinity": [int(i) for i in fin.geom_conaffinity],
      "geomCondim": [int(i) for i in fin.geom_condim],
      "geomPriority": [int(i) for i in fin.geom_priority],
      "geomFriction": _fl(fin.geom_friction),
      "geomDataId": [int(i) for i in fin.geom_dataid],
      "geomSize": _fl(fin.geom_size),
      "geomRgba": _fl(fin.geom_rgba),
      "bodyParentId": [int(i) for i in fin.body_parentid],
      "bodyMass": _fl(fin.body_mass),
      "meshNames": name_list(fin, mujoco.mjtObj.mjOBJ_MESH, fin.nmesh),
      "geomTypeEnum": {"plane": 0, "hfield": 1, "sphere": 2, "capsule": 3,
                       "ellipsoid": 4, "cylinder": 5, "box": 6, "mesh": 7},
    },
    # top-level: app/physics.js reads sceneJson.geomRobot directly
    # ("a" | "b" | null per geom id, ngeom long) and cross-checks it against
    # the body_rootid map it derives from the compiled model.
    "geomRobot": [("a" if fnames[i] and fnames[i].startswith("a_")
                   else "b" if fnames[i] and fnames[i].startswith("b_") else None)
                  for i in range(fin.ngeom)],
    "qposLayout": {
      "nq": int(fin.nq), "nv": int(fin.nv),
      "freeJointQpos": "x y z qw qx qy qz",
      "robots": {r: {k: layout[r][k] for k in
                     ("freeQposAdr", "freeDofAdr", "jointQposAdr", "jointDofAdr")}
                 for r in ("a", "b")},
    },
    "robots": layout,
    "joints": {
      "order": CANON,
      "orderNote": "entity-native JOINT order FL,FR,RL,RR x (hip,thigh,calf). "
                   "The compiled mjModel ACTUATOR array is grouped hip x4 / "
                   "thigh x4 / calf x4 -- use robots.<r>.actuatorIdsJointOrder "
                   "to cross the two.",
      "defaultJointPos": default_q,
      "hardLimitLo": jlo, "hardLimitHi": jhi,
      "softLimitFactor": soft,
      "softLimitLo": [round(v, 12) for v in qlo],
      "softLimitHi": [round(v, 12) for v in qhi],
      "kp": kp["a"], "kd": kd["a"],
      "forceRange": [[float(fin.actuator_forcerange[i, 0]),
                      float(fin.actuator_forcerange[i, 1])]
                     for i in layout["a"]["actuatorIdsJointOrder"]],
      "ctrlRange": [[float(fin.actuator_ctrlrange[i, 0]),
                     float(fin.actuator_ctrlrange[i, 1])]
                    for i in layout["a"]["actuatorIdsJointOrder"]],
      "ctrlLimited": [bool(fin.actuator_ctrllimited[i])
                      for i in layout["a"]["actuatorIdsJointOrder"]],
      "armature": [float(fin.dof_armature[i]) for i in layout["a"]["jointDofAdr"]],
      "initialBaseHeight": 0.32,
    },
    "field": {
      "size": [2 * hx, 2 * hy],
      "center": [cx, cy],
      "touchdownMargin": float(scenario.touchdown_margin),
      "lineX": line_env_x,
      "lineXB": line_env_x_b,
      "lineXNote": "env-local, = field_center.x + scenario.line_x, the same "
                   "expression game_env_cfg._field_marker_spec_fn uses.",
      "oobHalfX": hx, "oobHalfY": hy,
      "oobX": [cx - hx, cx + hx], "oobY": [cy - hy, cy + hy],
      "wallsNote": "NO physical walls. Every field/arena geom is contype=0 "
                   "conaffinity=0 density=0 group=2. Out-of-bounds is a referee "
                   "test on the trunk centre.",
      "mjlabMarkers": [
        {"name": fnames[i], "type": int(fin.geom_type[i]), "pos": _fl(fin.geom_pos[i]),
         "size": _fl(fin.geom_size[i]), "rgba": _fl(fin.geom_rgba[i])}
        for i in range(fin.ngeom)
        if fnames[i] and int(fin.geom_bodyid[i]) ==
        name_list(fin, mujoco.mjtObj.mjOBJ_BODY, fin.nbody).index(
            "touchdown_field_markers")
      ],
      "decor": {
        "body": DECOR_BODY,
        "firstGeomId": int(min(dids)),
        "geoms": [
          {"name": fnames[i], "type": int(fin.geom_type[i]),
           "pos": _fl(fin.geom_pos[i]), "size": _fl(fin.geom_size[i]),
           "rgba": _fl(fin.geom_rgba[i])} for i in dids
        ],
        "palette": {k: list(v) for k, v in ARENA.items()},
        "paletteSource": "src/tasks/game/game_env_cfg.py:98-100 (boundary / "
                         "touchdown / region rgba) + mjlab/terrains/"
                         "terrain_entity.py:23-24 (ground checker rgb1/rgb2)",
        "proof": decor_proof,
      },
      "ground": {
        "geom": "terrain",
        "type": int(fin.geom_type[fnames.index("terrain")]),
        "condim": int(fin.geom_condim[fnames.index("terrain")]),
        "friction": _fl(fin.geom_friction[fnames.index("terrain")]),
        "contype": int(fin.geom_contype[fnames.index("terrain")]),
        "conaffinity": int(fin.geom_conaffinity[fnames.index("terrain")]),
      },
    },
    "spawn": {
      "config": spawn_cfg,
      "configSource": spec["spawnConfigSource"],
      "trainingNominal": spawn_train,
      "jitter": {
        r: {"tight": [float(scenario.spawns[e].reset_x),
                      float(scenario.spawns[e].reset_y),
                      float(scenario.spawns[e].reset_yaw)],
            "wide": [float(scenario.spawns[e].wide_x),
                     float(scenario.spawns[e].wide_y),
                     float(scenario.spawns[e].wide_yaw)]}
        for r, e in (("a", scenario.attacker_role), ("b", scenario.defender_role))
      },
      "keyframeQpos": _fl(fin.key_qpos[0]),
      "keyframeCtrl": _fl(fin.key_ctrl[0]),
      "keyframeNote": "mjlab's own merged init keyframe (z = 0.32, default "
                      "joints, the TRAINING nominal spawn).",
    },
    "rules": {
      "note": "Measured off the live scenario object; app/referee.js owns the "
              "implementation. recon/05 §5 (asym) / §6 (sym) is the contract.",
      "episodeLengthS": float(scenario.episode_length_s),
      "episodeSteps": int(math.ceil(scenario.episode_length_s /
                                    (fin.opt.timestep * cfg.decimation))),
      "terminalTimeout": bool(scenario.terminal_timeout),
      "fallLimitAngleRad": float(scenario.fall_limit_angle),
      "fallLimitAngleDeg": math.degrees(float(scenario.fall_limit_angle)),
      "trunkContactTerminal": bool(scenario.trunk_contact_terminal),
      "trunkContactSurface": str(getattr(scenario, "trunk_contact_surface", "")),
      "safetyTerminal": bool(getattr(scenario, "safety_terminal", True)),
      "hullTouchdown": bool(getattr(scenario, "hull_touchdown", False)),
      "pastLineFall": str(getattr(scenario, "past_line_fall", "")),
      "contactWinScale": float(getattr(scenario, "contact_win_scale", 0.0)),
      "winWeight": float(getattr(scenario, "win_weight", 0.0)),
      "attackerRole": scenario.attacker_role,
      "defenderRole": scenario.defender_role,
      "directionA": (float(scenario.direction(scenario.attacker_role))
                     if hasattr(scenario, "direction") else 1.0),
      "directionB": (float(scenario.direction(scenario.defender_role))
                     if hasattr(scenario, "direction") else -1.0),
      "terminationTerms": sorted(cfg.terminations.keys())
      if isinstance(cfg.terminations, dict) else
      sorted(f.name for f in __import__("dataclasses").fields(cfg.terminations)),
      "terminationTermsNote": "the live TerminationManager vocabulary for this "
                              "task -- app/referee.js must reproduce exactly "
                              "these and nothing else (recon/05 §7).",
      "directionNote": "sym: scenario.direction(role). asym has no direction() "
                       "-- only the attacker scores, and A/+1 vs B/-1 is the "
                       "convention app/config.js uses.",
    },
    "keyframe": {
      "file": "keyframe.json",
      "settleSeconds": SETTLE_SECONDS,
      "baseZ": base_z,
      "baseZWindow": list(BASE_Z_WINDOW),
      "baseZWindowNote": "The task brief asked for z in [0.25, 0.40]; the "
                         "walk-soft plant does not reach it. Measured on this "
                         "model: the default pose is kinematically a 0.295017 m "
                         "stand (lowest collision point with base z = 0) and the "
                         "legs sag a further 0.0647 m under 15.206 kg at "
                         "kp 20/20/40 (go2_constants.py:41-67), the rear shins "
                         "resting on the floor, so 0.23029 m is the true settled "
                         "height in both games. No gain was changed.",
    },
    "sizes": {
      "sceneXmlBytes": (out / "scene.xml").stat().st_size,
      "meshBytesShipped": dir_bytes(adir),
      "meshBytesShippedGz9": gz_bytes(adir),
      "meshBytesFull": bytes_full,
      "meshBytesFullGz9": gz_bytes(full_dir),
    },
    "lite": None if lite_report is None else {
      "method": f"weld coincident vertices at 1e-{LITE_WELD_DIGITS} m, then "
                "quadric edge collapse (fast_simplification.simplify), "
                f"target {args.lite_fraction:g} of faces (floor "
                f"{LITE_MIN_FACES}), aggression {args.lite_aggression}, "
                f"coords written {LITE_COORD_FMT}",
      "note": "VISUAL ONLY. Every mesh geom in this model is contype=0 "
              "conaffinity=0 density=0 group=2; collision is 23 primitives per "
              "robot (recon/05 §1.2). Proof P2: identical XML + reduced meshes "
              "gives bit-identical qpos/qvel after 2 s under two ctrl modes.",
      "facesRaw": sum(v["facesRaw"] for v in lite_report.values()),
      "facesWelded": sum(v["facesWelded"] for v in lite_report.values()),
      "facesOut": sum(v["facesOut"] for v in lite_report.values()),
      "proof": lite_proof,
      "perMesh": lite_report,
    },
  }
  (out / "scene.json").write_text(json.dumps(scene_json, indent=2) + "\n")

  keyframe_json = {
    "schema": "s2c_web_play/keyframe@2",
    "game": game,
    "comment": f"Settled stands produced by holding ctrl = defaultJointPos for "
               f"{SETTLE_SECONDS} s of mj_step from each spawn set. "
               "spawnReset = the training reset (z 0.32, default joints, zero "
               "vel) -- use this for mjlab parity. settled = the raw 2 s result "
               "(the robot sags and slides a little). settledAtSpawn = the "
               "settled leg shape and height put BACK on the exact spawn x/y/yaw "
               "at rest -- use this for a clean game reset with no drop transient.",
    "settleSeconds": SETTLE_SECONDS,
    "physicsSteps": int(round(SETTLE_SECONDS / fin.opt.timestep)),
    "defaultJointPos": default_q,
    "jointOrder": CANON,
    "qposLayout": scene_json["qposLayout"],
    "baseZWindow": list(BASE_Z_WINDOW),
    "baseZWindowNote": scene_json["keyframe"]["baseZWindowNote"],
    "spawnSets": {"config": spawn_cfg, "trainingNominal": spawn_train},
    "keyframes": keyframes,
  }
  (out / "keyframe.json").write_text(json.dumps(keyframe_json, indent=2) + "\n")

  print(f"[write] {out}/  scene.xml {(out/'scene.xml').stat().st_size} B  "
        f"scene.json {(out/'scene.json').stat().st_size} B  "
        f"keyframe.json {(out/'keyframe.json').stat().st_size} B  "
        f"assets/ {dir_bytes(adir)} B ({len(assets)} files)")
  if not args.keep_work:
    shutil.rmtree(work, ignore_errors=True)
  else:
    print(f"[work ] {work}")
  return {
    "game": game, "out": str(out),
    "sceneXmlBytes": (out / "scene.xml").stat().st_size,
    "sceneJsonBytes": (out / "scene.json").stat().st_size,
    "keyframeJsonBytes": (out / "keyframe.json").stat().st_size,
    "meshBytes": dir_bytes(adir), "meshBytesGz9": gz_bytes(adir),
    "meshBytesFull": bytes_full,
    "decorGeoms": len(decor_names),
  }


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> int:
  repo = Path(__file__).resolve().parents[1]
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--game", choices=("sym", "asym", "both"), default="both")
  ap.add_argument("--out", type=Path, default=None,
                  help="output dir (default <repo>/assets/scene/<game>)")
  ap.add_argument("--no-lite", action="store_true")
  ap.add_argument("--lite-fraction", type=float, default=LITE_FACE_FRACTION)
  ap.add_argument("--lite-aggression", type=int, default=LITE_AGGRESSION)
  ap.add_argument("--no-twin-check", action="store_true")
  ap.add_argument("--keep-work", action="store_true")
  args = ap.parse_args()

  if args.game == "both":
    # one subprocess per game: the two forks register overlapping gym ids and
    # each does its own os.chdir, so they stay in separate interpreters.
    for g in ("sym", "asym"):
      cmd = [sys.executable, str(Path(__file__).resolve()), "--game", g]
      for flag, val in (("--lite-fraction", args.lite_fraction),
                        ("--lite-aggression", args.lite_aggression)):
        cmd += [flag, str(val)]
      if args.no_lite:
        cmd.append("--no-lite")
      if args.no_twin_check:
        cmd.append("--no-twin-check")
      if args.keep_work:
        cmd.append("--keep-work")
      rc = subprocess.call(cmd)
      if rc != 0:
        return rc
    root = repo / "assets/scene"
    print(f"\n[total] assets/scene = {dir_bytes(root)} B "
          f"(gz-9 {gz_bytes(root)} B)")
    for p in sorted(root.rglob("*")):
      if p.is_file() and p.suffix != ".obj":
        print(f"        {p.relative_to(root)}  {p.stat().st_size} B")
    for g in ("sym", "asym"):
      d = root / g / "assets"
      print(f"        {g}/assets/  {dir_bytes(d)} B in "
            f"{len(list(d.glob('*')))} files (gz-9 {gz_bytes(d)} B)")
    return 0

  out = (args.out or (repo / "assets/scene" / args.game)).resolve()
  export_game(args.game, out, args)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())

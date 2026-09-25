#!/usr/bin/env python3
"""Export the S2C safety certificate (QCBF) to the game's dependency-free (.bin + .json) format.

Run with the mjlab venv (torch + numpy + mjlab on the path):

    /home/ray/Disk_ext/Go2/envs/mjlab_venv/bin/python tools/export_filter.py

WHAT THIS EXPORTS
-----------------
The official certificate is `agent_15000` of the v5prox collision line, already
frozen into a deploy bundle:

    deploy/robots/go2_go2/config/policy/safety_game/collision_v5prox_15k_62d_game/
      params/deploy.yaml
      exported/filter_nets.pt      md5 96fcae4443cc1e51f421a608ea1780b5
      exported/{ctrl,dstb,critic}.onnx

`recon/03_safety_filter.md:22-47` names it, `src/tasks/game/config/go2_go2/presets.py:60-62`
(`_V5PROX_15K_GAME_FILTER`) is the preset that loads it, and
`deploy/robots/go2_go2/config/config.yaml:50` is the C++ deploy agreeing.

Three nets, all `sin`-activated MLPs (`src/tasks/safety/rl/networks.py:42-46`,
`meta['activation'] == 'sin'`) — which app/policy.js already supports, so the
browser needs no new runtime:

    filter_ctrl   pi_shield   62 -> 512 -> 512 -> 512 -> 12   head tanh
    filter_dstb   adversary   74 -> 256 -> 256 -> 256 ->  6   head tanh  (74 = 62 obs + 12 ctrl)
    filter_q1     twin head 1 80 -> 256 -> 256 -> 256 ->  1   head linear (80 = 62 + 12 + 6)
    filter_q2     twin head 2 80 -> 256 -> 256 -> 256 ->  1   head linear

`critic.onnx` folds `max(q1, q2)` into the graph; the .bin format is one linear
chain per file, so the twin heads ship as two files and app/filter.js takes the
`max` (`pessimistic = "max"`, `filtered_action.py:334`). Identical arithmetic,
identical MAC count.

NO NORMALIZER. `QcbfNets.__init__` (`filtered_action.py:296-322`) loads only
ctrl/dstb/critic; the source checkpoint has no normalizer key. The raw 62-D
physical observation goes straight into layer 0. (The GAME POLICY that produces
the proposal is a different network with its own normalizer — that one is
tools/export_policy.py's job.)

`action_scale` / `action_bias` on both TanhGaussianActors are asserted to be
identity (all 1.0 / all 0.0), so `forward = tanh(mean_net(x))` exactly
(`networks.py:138-144`), which is what `head_activation: "tanh"` means in the json.
`log_std_net` is never evaluated at deploy and is NOT exported.

WHAT ELSE IT WRITES
-------------------
    assets/policies/filter_{ctrl,dstb,q1,q2}.{bin,json}
    assets/policies/manifest.json      <- a `filter` key (the `policies` array is NOT touched)
    tests/parity.json                  <- 8 fixed inputs per net + the torch float32 outputs
    tests/filter_fixture.json          <- 62-D observation fixtures and whole-filter traces,
                                          produced by calling the REAL training code

The fixtures in tests/filter_fixture.json are not a transcription: this script
duck-types an env and calls

    GameQcbfShieldAction._assemble_obs(fake_self)   # src/tasks/game/mdp/game_qcbf_action.py:295-331
    make_inmemory_shield(...) -> BatchedQcbfFilter  # src/tasks/safety_field/rl/exit_driver.py:647
      .task_increment / .step / .apply_increment    # src/tasks/game/mdp/filtered_action.py:438-627

so every number the JS is held to came out of the training implementation itself,
including `opponent_kinematics` (`safety_collision/mdp/observations.py:54-124`)
and `rect_rect_distance` (`safety_collision/mdp/collide.py:211-244`).

THE LAW THIS BUNDLE RUNS UNDER (asymmetric arm)
-----------------------------------------------
`GameQcbfShieldAction` builds the shield with `sup_enabled=False,
guard_enabled=False` (literals, game_qcbf_action.py:181-183) and
`kin_enabled = bool(filter.tilt_guard is not None)` = False for this bundle,
so `value_guard = -1e9` and the RUN/GUARD/HANDBACK machine never leaves RUN
(`exit_driver.py:700-703`; recon/03:216-228). There is no state machine to port.
`intervention` is absent from this bundle's yaml -> "line_search"; the
projected-gradient copy is the SYMMETRIC arm's (`..._62d_game_pg`,
`sym_preset.py:114-131`), and this build only ships the asymmetric game.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parent.parent
POLICY_DIR = REPO / "assets" / "policies"
TESTS_DIR = REPO / "tests"

MJLAB = Path("/home/ray/Disk_ext/Go2/Project/unitree_rl_mjlab")
BUNDLE = (
    MJLAB
    / "deploy/robots/go2_go2/config/policy/safety_game/collision_v5prox_15k_62d_game"
)

# --- constants taken from the training code, each with its source line -------

# exit_driver.py:710-714 (make_inmemory_shield hard-codes these; the yaml copies
# under `filter:` agree but are NOT what the python path reads -- recon/03:185).
CBF_MAX_ITERS = 20
CBF_TOL = 0.01
PESSIMISTIC = "max"

# exit_driver.py:700-703 -- guard_enabled=False parks the value channel here,
# which is what makes `danger` identically False (recon/03:216-228).
VALUE_GUARD = -1.0e9
VALUE_RELEASE = -1.0e9 + 1e-6

# filtered_action.py:496 / safety_filter.h:700 -- dead in this lane (never
# evaluated because `enter` is identically False), recorded for completeness.
RESEED_TRACK_ERR = 0.15

# safety_collision/mdp/collide.py:56-57, :78, :83
COLLIDE_HALF = (0.29, 0.193)
COLLIDE_FWD_OFFSET = 0.05
D_VIS_COL = 1.5
D_MAX_COL = 3.0
NO_OPPONENT_TAIL = (D_MAX_COL, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)

# go2_constants.py:72-81, JOINT order FL,FR,RL,RR x (hip,thigh,calf)
DEFAULT_JOINT_POS = [-0.1, 0.9, -1.8, 0.1, 0.9, -1.8, -0.1, 0.9, -1.8, 0.1, 0.9, -1.8]

# soft_joint_pos_limits = 0.9 x the compiled jnt_range about its midpoint
# (go2_constants.py:125); recon/03:196-199, and identical to the numbers already
# in app/action.js / assets/scene/asym/scene.json joints.softLimitLo/Hi.
SOFT_LO = [
    -0.94248, -1.317725, -2.628453,
    -0.94248, -1.317725, -2.628453,
    -0.94248, -0.270525, -2.628453,
    -0.94248, -0.270525, -2.628453,
]
SOFT_HI = [
    0.94248, 3.237625, -0.932007,
    0.94248, 3.237625, -0.932007,
    0.94248, 4.284825, -0.932007,
    0.94248, 4.284825, -0.932007,
]

# touchdown_driver.py:42-44 / deploy.yaml walk_* + safety_*; the rise step is
# 1/8 in the ENV lane (GameQcbfShieldActionCfg.gain_rise_steps default 8,
# game_qcbf_action.py:382) and 1/15 in the C++ lane (deploy.yaml gain_rise_steps).
GAINS = {
    "walk": {"kp": [20, 20, 40] * 4, "kd": [1, 1, 2] * 4},
    "safety": {"kp": [100, 100, 200] * 4, "kd": [1, 1, 2] * 4},
    "rise_steps_env": 8,
    "rise_steps_cpp": 15,
    "contact_gated": True,
}

# game_qcbf_action.py:295-331 -- the 62-D layout, in order.
OBS_LAYOUT = [
    {"term": "root_link_lin_vel_b", "at": 0, "dim": 3},
    {"term": "roll", "at": 3, "dim": 1},
    {"term": "pitch", "at": 4, "dim": 1},
    {"term": "root_link_ang_vel_b", "at": 5, "dim": 3},
    {"term": "joint_pos", "at": 8, "dim": 12},
    {"term": "joint_vel", "at": 20, "dim": 12},
    {"term": "foot_contact", "at": 32, "dim": 4},
    {"term": "prev_ctrl", "at": 36, "dim": 12},
    {"term": "wall_margins", "at": 48, "dim": 4},
    {"term": "heading", "at": 52, "dim": 2},
    {"term": "opponent_tail", "at": 54, "dim": 8},
]

NETS = {
    "filter_ctrl": {
        "role": "pi_shield",
        "sd_key": "ctrl",
        "prefix": "mean_net",
        "head_activation": "tanh",
        "inputs": ["obs62"],
        "out": "u_safe (certified increment, [-1,1]^12)",
    },
    "filter_dstb": {
        "role": "adversary",
        "sd_key": "dstb",
        "prefix": "mean_net",
        "head_activation": "tanh",
        "inputs": ["obs62", "ctrl12"],
        "out": "d (worst-case disturbance, [-1,1]^6)",
    },
    "filter_q1": {
        "role": "twin_q_head_1",
        "sd_key": "critic",
        "prefix": "q1",
        "head_activation": None,
        "inputs": ["obs62", "ctrl12", "dstb6"],
        "out": "Q1(obs, u, d)",
    },
    "filter_q2": {
        "role": "twin_q_head_2",
        "sd_key": "critic",
        "prefix": "q2",
        "head_activation": None,
        "inputs": ["obs62", "ctrl12", "dstb6"],
        "out": "Q2(obs, u, d)",
    },
}


# =============================================================================
# weights -> .bin + .json
# =============================================================================


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def layer_shapes(sd: dict, prefix: str) -> list[tuple[int, int]]:
    """[(in, out), ...] in forward order for the Linear stack `<prefix>.{0,2,4,6}`."""
    idxs = sorted(
        int(k.split(".")[1])
        for k in sd
        if k.startswith(f"{prefix}.") and k.endswith(".weight")
    )
    out = []
    for i in idxs:
        w = sd[f"{prefix}.{i}.weight"]
        out.append((int(w.shape[1]), int(w.shape[0])))
    for a, b in zip(out, out[1:]):
        assert a[1] == b[0], f"{prefix}: layer widths do not chain: {out}"
    return out


def write_net(name: str, sd: dict, spec: dict, bundle_md5: str, meta: dict) -> dict:
    prefix = spec["prefix"]
    shapes = layer_shapes(sd, prefix)
    idxs = sorted(
        int(k.split(".")[1])
        for k in sd
        if k.startswith(f"{prefix}.") and k.endswith(".weight")
    )
    blob = bytearray()
    for i in idxs:
        w = sd[f"{prefix}.{i}.weight"].to(torch.float32).contiguous().numpy()
        b = sd[f"{prefix}.{i}.bias"].to(torch.float32).contiguous().numpy()
        blob += w.astype("<f4").tobytes(order="C")
        blob += b.astype("<f4").tobytes(order="C")
    (POLICY_DIR / f"{name}.bin").write_bytes(bytes(blob))

    js = {
        "name": name,
        "role": spec["role"],
        "kind": "safety_certificate",
        "game": "filter",
        "seat": None,
        "method": "s2c",
        "display": f"S2C certificate — {spec['role']}",
        "source_checkpoint": meta["source_checkpoint"],
        "source_bundle": str(BUNDLE.relative_to(MJLAB)),
        "filter_nets_md5": bundle_md5,
        "obs_dim": shapes[0][0],
        "act_dim": shapes[-1][1],
        "input_concat": spec["inputs"],
        "output": spec["out"],
        "cert_obs_dim": int(meta["obs_dim"]),
        "ctrl_dim": int(meta["ctrl_dim"]),
        "dstb_dim": int(meta["dstb_dim"]),
        "layers": [{"in": a, "out": b} for a, b in shapes],
        "activation": meta["activation"],       # "sin" -- networks.py:42-46
        "activation_alpha": 1.0,                # unused by sin; app/policy.js wants the field
        "head_activation": spec["head_activation"],
        # QcbfNets loads no normalizer (filtered_action.py:296-322) and the
        # source checkpoint has none: the RAW physical obs enters layer 0.
        "norm": None,
        "rotation_baked": False,
        "action_path": None,
        "bin": f"{name}.bin",
        "bin_bytes": len(blob),
        "bin_layout": (
            "little-endian float32; for each Linear in order: weight (row-major, "
            "out x in) then bias (out)"
        ),
        "forward": (
            "x = concat(" + ", ".join(spec["inputs"]) + "); "
            "h = sin(L0 x); h = sin(L1 h); h = sin(L2 h); y = L3 h"
            + ("; y = tanh(y)" if spec["head_activation"] == "tanh" else "")
        ),
        "notes": (
            "TanhGaussianActor.forward is tanh(mean_net(x)) * action_scale + "
            "action_bias with action_scale == 1 and action_bias == 0 (asserted at "
            "export), so the affine is a no-op (networks.py:138-144). log_std_net "
            "is never evaluated at deploy and is not exported."
            if spec["head_activation"] == "tanh"
            else "TwinnedQNetwork head is a bare Linear; app/filter.js takes "
            "max(Q1, Q2) because pessimistic == 'max' (filtered_action.py:334)."
        ),
    }
    (POLICY_DIR / f"{name}.json").write_text(json.dumps(js, indent=2) + "\n", "utf-8")
    return js


def numpy_load(name: str):
    """Re-read <name>.bin + <name>.json with nothing but numpy — the independent
    check that the shipped bytes are the network, and the reference app/policy.js
    must reproduce."""
    meta = json.loads((POLICY_DIR / f"{name}.json").read_text())
    raw = (POLICY_DIR / meta["bin"]).read_bytes()
    want = sum(l["in"] * l["out"] + l["out"] for l in meta["layers"]) * 4
    assert len(raw) == want, f"{name}: bin is {len(raw)} B, layers need {want} B"
    buf = np.frombuffer(raw, dtype="<f4")
    layers, off = [], 0
    for l in meta["layers"]:
        n_in, n_out = l["in"], l["out"]
        w = buf[off : off + n_out * n_in].reshape(n_out, n_in)
        off += n_out * n_in
        b = buf[off : off + n_out]
        off += n_out
        layers.append((w, b))
    assert off == buf.size, f"{name}: {buf.size - off} trailing float32 in the bin"
    head = meta["head_activation"]

    def forward(x: np.ndarray, dtype=np.float32) -> np.ndarray:
        x = np.asarray(x, dtype=dtype)
        for k, (w, b) in enumerate(layers):
            x = (x @ w.astype(dtype).T + b.astype(dtype)).astype(dtype)
            if k < len(layers) - 1:
                x = np.sin(x).astype(dtype)
            elif head == "tanh":
                x = np.tanh(x).astype(dtype)
        return x

    return meta, forward


# =============================================================================
# the real training code, driven on duck-typed state
# =============================================================================


def _import_mjlab():
    sys.path.insert(0, str(MJLAB))
    from types import SimpleNamespace  # noqa: F401  (used by callers)

    from src.tasks.game.mdp.filtered_action import QcbfNets  # noqa: E402
    from src.tasks.game.mdp.game_qcbf_action import GameQcbfShieldAction  # noqa: E402
    from src.tasks.safety_field.rl.exit_driver import make_inmemory_shield  # noqa: E402

    return GameQcbfShieldAction, QcbfNets, make_inmemory_shield


def quat_from_rpy(roll: float, pitch: float, yaw: float) -> list[float]:
    """w-first quaternion for the XYZ-extrinsic euler angles
    `euler_xyz_from_quat` inverts (mjlab/utils/lab_api/math.py:460-472)."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return [
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ]


def make_states(seed: int = 0xC0FFEE) -> list[dict]:
    """Physically plausible (ego, opponent) states, hand-picked plus random.

    The three hand-picked ones are recon/03 section 2f's numerical anchors, so the
    fixture re-measures the published V values as a side effect.
    """
    rng = np.random.default_rng(seed)
    states: list[dict] = []

    def st(label, **kw):
        s = dict(
            label=label,
            ego_pos=[0.0, 0.0, 0.32],
            ego_rpy=[0.0, 0.0, 0.0],
            ego_lin_vel_w=[0.0, 0.0, 0.0],
            ego_ang_vel_b=[0.0, 0.0, 0.0],
            joint_pos=list(DEFAULT_JOINT_POS),
            joint_vel=[0.0] * 12,
            feet=[1.0, 1.0, 1.0, 1.0],
            prev_ctrl=[0.0] * 12,
            opp_pos=[20.0, 0.0, 0.32],
            opp_rpy=[0.0, 0.0, math.pi],
            opp_lin_vel_w=[0.0, 0.0, 0.0],
        )
        s.update(kw)
        s["label"] = label
        return s

    # recon/03:378 -- default stance at the field centre, no opponent in range.
    states.append(st("stance_no_opponent"))
    # recon/03:379 -- 0.1 m from the filter's +x wall (the bundle's hx = 2.4).
    states.append(st("near_wall_x2.3", ego_pos=[2.3, 0.0, 0.32]))
    # recon/03:380 -- head-on opponent at d = 0.6 m, closing at 1.5 m/s.
    # d is the rect-rect gap, so the centres sit 0.6 + 2*0.29 apart along x and
    # the +0.05 hull offsets face each other: ego hull at +0.34, opp hull at -0.34.
    states.append(
        st(
            "headon_d0.6_closing1.5",
            ego_lin_vel_w=[0.75, 0.0, 0.0],
            opp_pos=[0.6 + 2 * (COLLIDE_HALF[0] + COLLIDE_FWD_OFFSET), 0.0, 0.32],
            opp_lin_vel_w=[-0.75, 0.0, 0.0],
        )
    )
    # A few more shapes the JS has to get right: yawed hulls, a corner-vs-edge
    # closest pair, penetration (d < 0), the split-horizon boundary, a tilted
    # body (roll/pitch non-zero), feet in flight, a live prev_ctrl.
    states.append(
        st(
            "yawed_pair_corner",
            ego_pos=[-0.4, 0.2, 0.31],
            ego_rpy=[0.05, -0.03, 0.7],
            ego_lin_vel_w=[0.8, -0.3, 0.02],
            ego_ang_vel_b=[0.1, -0.2, 0.4],
            opp_pos=[0.35, -0.25, 0.33],
            opp_rpy=[0.0, 0.0, -2.1],
            opp_lin_vel_w=[-0.6, 0.4, 0.0],
            feet=[1.0, 0.0, 0.0, 1.0],
            prev_ctrl=[0.3, -0.2, 0.5, -0.1, 0.05, -0.6, 0.2, 0.2, -0.3, 0.0, 0.1, -0.4],
        )
    )
    states.append(
        st(
            "penetrating",
            ego_pos=[0.0, 0.0, 0.30],
            ego_rpy=[0.0, 0.0, 0.1],
            ego_lin_vel_w=[1.2, 0.0, 0.0],
            opp_pos=[0.45, 0.05, 0.30],
            opp_rpy=[0.0, 0.0, math.pi - 0.2],
            opp_lin_vel_w=[-0.9, 0.0, 0.0],
            feet=[0.0, 0.0, 0.0, 0.0],
        )
    )
    states.append(
        st(
            "split_horizon_just_far",  # raw d slightly ABOVE d_vis -> tail blanked
            opp_pos=[1.55 + 2 * (COLLIDE_HALF[0] + COLLIDE_FWD_OFFSET), 0.3, 0.32],
            opp_lin_vel_w=[-1.0, 0.2, 0.0],
            ego_lin_vel_w=[1.0, 0.0, 0.0],
        )
    )
    states.append(
        st(
            "split_horizon_just_near",  # raw d slightly BELOW d_vis -> tail live
            opp_pos=[1.45 + 2 * (COLLIDE_HALF[0] + COLLIDE_FWD_OFFSET), 0.3, 0.32],
            opp_lin_vel_w=[-1.0, 0.2, 0.0],
            ego_lin_vel_w=[1.0, 0.0, 0.0],
        )
    )
    states.append(
        st(
            "outside_the_filter_box",  # margins go NEGATIVE past the bundle rect
            ego_pos=[2.6, 1.6, 0.32],
            ego_rpy=[0.0, 0.0, -0.8],
            ego_lin_vel_w=[1.4, 0.6, 0.0],
        )
    )
    states.append(
        st(
            "tilted_falling",
            ego_rpy=[0.55, -0.35, 1.2],
            ego_pos=[-1.2, -0.7, 0.26],
            ego_lin_vel_w=[0.4, -1.1, -0.6],
            ego_ang_vel_b=[1.4, -0.8, 0.3],
            joint_vel=[3.0, -2.0, 4.0, -1.0, 0.5, -3.0, 2.0, 2.0, -4.0, 0.0, 1.0, -2.0],
            feet=[0.0, 1.0, 0.0, 0.0],
        )
    )
    # ... and random draws around stance so the fixture is not only hand-picked.
    for i in range(6):
        states.append(
            st(
                f"random_{i}",
                ego_pos=[
                    float(rng.uniform(-2.2, 2.6)),
                    float(rng.uniform(-1.3, 1.3)),
                    float(rng.uniform(0.24, 0.36)),
                ],
                ego_rpy=[
                    float(rng.uniform(-0.4, 0.4)),
                    float(rng.uniform(-0.4, 0.4)),
                    float(rng.uniform(-math.pi, math.pi)),
                ],
                ego_lin_vel_w=[float(rng.uniform(-1.5, 2.5)), float(rng.uniform(-1, 1)), float(rng.uniform(-0.4, 0.4))],
                ego_ang_vel_b=[float(rng.uniform(-2, 2)) for _ in range(3)],
                joint_pos=[
                    float(np.clip(DEFAULT_JOINT_POS[k] + rng.normal(0, 0.25), SOFT_LO[k], SOFT_HI[k]))
                    for k in range(12)
                ],
                joint_vel=[float(rng.normal(0, 3.0)) for _ in range(12)],
                feet=[float(rng.integers(0, 2)) for _ in range(4)],
                prev_ctrl=[float(rng.uniform(-1, 1)) for _ in range(12)],
                opp_pos=[
                    float(rng.uniform(-2.2, 2.6)),
                    float(rng.uniform(-1.3, 1.3)),
                    0.32,
                ],
                opp_rpy=[0.0, 0.0, float(rng.uniform(-math.pi, math.pi))],
                opp_lin_vel_w=[float(rng.uniform(-2, 2)), float(rng.uniform(-1, 1)), 0.0],
            )
        )
    return states


def build_fake(state: dict, shield, field_half, field_center, d_vis):
    """A duck-typed `GameQcbfShieldAction` self + env carrying exactly `state`."""
    from types import SimpleNamespace

    def t(v):
        return torch.tensor([v], dtype=torch.float32)

    def ent(pos, rpy, jp, jv, lvw, avb):
        quat = torch.tensor([quat_from_rpy(*rpy)], dtype=torch.float32)
        # root_link_lin_vel_b = R(q)^T v_w. quat_apply_inverse, math.py:653-671.
        w, x, y, z = (float(c) for c in quat[0])
        vx, vy, vz = lvw
        tx = 2 * (y * vz - z * vy)
        ty = 2 * (z * vx - x * vz)
        tz = 2 * (x * vy - y * vx)
        lvb = [
            vx - w * tx + (y * tz - z * ty),
            vy - w * ty + (z * tx - x * tz),
            vz - w * tz + (x * ty - y * tx),
        ]
        return SimpleNamespace(
            data=SimpleNamespace(
                root_link_pos_w=t(pos),
                root_link_quat_w=quat,
                root_link_lin_vel_w=t(lvw),
                root_link_lin_vel_b=t(lvb),
                root_link_ang_vel_b=t(avb),
                joint_pos=t(jp),
                joint_vel=t(jv),
            )
        )

    ego = ent(
        state["ego_pos"], state["ego_rpy"], state["joint_pos"], state["joint_vel"],
        state["ego_lin_vel_w"], state["ego_ang_vel_b"],
    )
    opp = ent(
        state["opp_pos"], state["opp_rpy"], DEFAULT_JOINT_POS, [0.0] * 12,
        state["opp_lin_vel_w"], [0.0, 0.0, 0.0],
    )
    sensor = SimpleNamespace(
        data=SimpleNamespace(found=torch.tensor([[[f] for f in state["feet"]]], dtype=torch.float32))
    )

    class Scene(dict):
        env_origins = torch.zeros(1, 3)

    scene = Scene(ego=ego, opp=opp, ego_feet=sensor)
    env = SimpleNamespace(scene=scene, device=torch.device("cpu"))
    return SimpleNamespace(
        _entity=ego,
        _env=env,
        _shield=shield,
        _has_field=True,
        _field_half=torch.tensor(field_half, dtype=torch.float32),
        _field_center=torch.tensor(field_center, dtype=torch.float32),
        _d_vis=d_vis,
        _opp_name="opp",
        cfg=SimpleNamespace(entity_name="ego", foot_contact_sensor="ego_feet"),
    )


# =============================================================================
# main
# =============================================================================


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bundle", type=Path, default=BUNDLE)
    # 5e-6, not export_policy.py's 1e-6: these nets are 512-wide and 4 layers
    # deep, so torch's own blocked float32 GEMM differs from a float64 replay by
    # more than the 60-D game actors do. The residual is torch's rounding, not a
    # weight mismatch -- the .bin bytes are copied verbatim from the state dict.
    ap.add_argument("--tol", type=float, default=5e-6,
                    help="float64 numpy-replay vs torch gate")
    ap.add_argument("--js-tol", type=float, default=1e-5,
                    help="the budget the JS runtime is held to (written into parity.json)")
    args = ap.parse_args()

    bundle = args.bundle
    nets_file = bundle / "exported" / "filter_nets.pt"
    yaml_file = bundle / "params" / "deploy.yaml"
    for p in (nets_file, yaml_file):
        if not p.exists():
            print(f"missing {p}", file=sys.stderr)
            return 2

    GameQcbfShieldAction, QcbfNets, make_inmemory_shield = _import_mjlab()

    y = yaml.safe_load(yaml_file.read_text())
    blob = torch.load(nets_file, map_location="cpu", weights_only=False)
    meta = blob["meta"]
    bundle_md5 = md5_of(nets_file)
    print(f"bundle   {bundle}")
    print(f"nets     {nets_file.name}  {nets_file.stat().st_size:,} B  md5 {bundle_md5}")
    print(f"meta     {meta}")

    assert int(meta["obs_dim"]) == 62 and int(meta["ctrl_dim"]) == 12
    assert int(meta["dstb_dim"]) == 6 and meta["activation"] == "sin"
    assert meta["pessimistic"] == PESSIMISTIC == str(y["filter"]["pessimistic"])
    for k in ("ctrl", "dstb"):
        sc = blob[k]["action_scale"].numpy()
        bi = blob[k]["action_bias"].numpy()
        assert np.allclose(sc, 1.0) and np.allclose(bi, 0.0), (
            f"{k}: action_scale/bias are not identity ({sc}, {bi}); the exported "
            "head_activation=tanh would then be wrong")
    print("assert   ctrl/dstb action_scale == 1, action_bias == 0 (tanh head is exact)")

    # yaml constants the port is allowed to read (recon/03:169-183).
    kappa = float(y["filter"]["kappa"])
    inc_scale = float(y["increment"]["scale"])
    inc_smoothing = float(y["increment"]["smoothing"])
    d_vis = float(y["field"]["d_vis"])
    field_len, field_wid = float(y["field"]["length"]), float(y["field"]["width"])
    assert (kappa, inc_scale, inc_smoothing, d_vis) == (0.9, 0.5, 0.3, 1.5)
    assert float(y["filter"]["value_eps"]) == -1.0e9
    assert int(y["filter"]["cbf_max_iters"]) == CBF_MAX_ITERS
    assert float(y["filter"]["cbf_tol"]) == CBF_TOL
    assert "intervention" not in y["filter"], (
        "this bundle names an intervention; the asym arm must stay on the line search")
    assert "tilt_guard" not in y["filter"], "tilt_guard present -> kin channel would arm"
    assert tuple(y["collision"]["half_extents"]) == COLLIDE_HALF
    assert float(y["collision"]["fwd_offset"]) == COLLIDE_FWD_OFFSET
    assert tuple(y["collision"]["no_opponent_tail"]) == NO_OPPONENT_TAIL
    print("assert   deploy.yaml constants match the module constants used here")

    POLICY_DIR.mkdir(parents=True, exist_ok=True)

    # ---- 1. the four .bin/.json ---------------------------------------------
    print("\n--- 1. weights -------------------------------------------------------")
    written = {}
    for name, spec in NETS.items():
        js = write_net(name, blob[spec["sd_key"]], spec, bundle_md5, meta)
        written[name] = js
        print(f"  {name:<12} {js['obs_dim']:>3} -> {js['act_dim']:<3} "
              f"{'->'.join(str(l['out']) for l in js['layers'])}"
              f"   head {js['head_activation'] or 'linear'}"
              f"   {js['bin_bytes']:,} B")

    # ---- 2. torch reference + numpy replay -----------------------------------
    print("\n--- 2. parity (numpy replay of the SHIPPED bin vs torch) -------------")
    nets = QcbfNets(nets_file, meta["pessimistic"], "cpu")

    shield = make_inmemory_shield(
        nets.ctrl, nets.dstb, nets.critic,
        __import__("types").SimpleNamespace(
            cfg=__import__("types").SimpleNamespace(
                increment_scale=inc_scale, action_smoothing=inc_smoothing),
            _q_lo=torch.tensor([SOFT_LO], dtype=torch.float32),
            _q_hi=torch.tensor([SOFT_HI], dtype=torch.float32),
            action_dim=12,
        ),
        obs_dim=62, num_envs=1, device="cpu", kappa=kappa,
        kin_enabled=False, sup_enabled=False, guard_enabled=False,
    )
    p = shield.p
    assert p.cbf_max_iters == CBF_MAX_ITERS and p.cbf_tol == CBF_TOL
    assert p.value_guard == VALUE_GUARD and p.intervention == "line_search"
    assert not p.kin_enabled and not p.sup_enabled and not p.full_takeover
    assert not p.fallback_line_search

    # Build the 62-D observations by running the REAL _assemble_obs.
    states = make_states()
    field_half = (field_len / 2.0, field_wid / 2.0)
    obs_rows = []
    for s in states:
        shield.reset_(slice(None), torch.tensor([s["joint_pos"]], dtype=torch.float32))
        shield.prev_ctrl = torch.tensor([s["prev_ctrl"]], dtype=torch.float32)
        fake = build_fake(s, shield, field_half, (0.0, 0.0), d_vis)
        obs_rows.append(GameQcbfShieldAction._assemble_obs(fake)[0].clone())
    OBS = torch.stack(obs_rows)  # (N, 62)

    rng = np.random.default_rng(0x5AFE)
    # 8 rows per net, drawn from the operating distribution: the first 4 use the
    # certificate's own (u, d), the last 4 push u to the corners of the box the
    # line search actually explores.
    pick = torch.tensor(rng.choice(len(states), size=8, replace=False).copy())
    obs8 = OBS[pick]
    u_self = nets.ctrl(obs8)
    u_rand = torch.tensor(rng.uniform(-1, 1, (8, 12)), dtype=torch.float32)
    u8 = torch.cat([u_self[:4], u_rand[4:]], dim=0)
    d8 = nets.dstb(obs8, cond=u8)

    inputs = {
        "filter_ctrl": obs8,
        "filter_dstb": torch.cat([obs8, u8], dim=-1),
        "filter_q1": torch.cat([obs8, u8, d8], dim=-1),
        "filter_q2": torch.cat([obs8, u8, d8], dim=-1),
    }
    torch_out = {
        "filter_ctrl": nets.ctrl(obs8),
        "filter_dstb": nets.dstb(obs8, cond=u8),
        "filter_q1": nets.critic.q1(inputs["filter_q1"]),
        "filter_q2": nets.critic.q2(inputs["filter_q2"]),
    }

    parity_rows = {}
    worst64 = worst32 = 0.0
    for name in NETS:
        _, fwd = numpy_load(name)
        x = inputs[name].numpy().astype(np.float32)
        ref = torch_out[name].detach().numpy().astype(np.float32).reshape(8, -1)
        e64 = float(np.abs(fwd(x, np.float64) - ref.astype(np.float64)).max())
        e32 = float(np.abs(fwd(x, np.float32) - ref).max())
        worst64, worst32 = max(worst64, e64), max(worst32, e32)
        assert e64 < args.tol, f"{name}: numpy(bin) vs torch = {e64:.3e} > {args.tol:.0e}"
        print(f"  {name:<12} max|numpy(bin) - torch| = {e64:.3e} (f64, gate "
              f"{args.tol:.0e})  |  {e32:.3e} (f32, JS budget {args.js_tol:.0e})")
        parity_rows[name] = {
            "seed": None,
            "obs_dim": int(written[name]["obs_dim"]),
            "act_dim": int(written[name]["act_dim"]),
            "rotation_baked": False,
            "obs": [[float(v) for v in row] for row in x],
            "act": [[float(v) for v in row] for row in ref],
        }

    # ---- 3. whole-filter traces ---------------------------------------------
    print("\n--- 3. whole-filter reference (the REAL BatchedQcbfFilter) -----------")
    cases = []
    for i, s in enumerate(states):
        obs = OBS[i : i + 1]
        shield.reset_(slice(None), torch.tensor([s["joint_pos"]], dtype=torch.float32))
        shield.prev_ctrl = torch.tensor([s["prev_ctrl"]], dtype=torch.float32)
        u_safe = nets.ctrl(obs)
        V = nets.robust_q(obs, u_safe)
        # Two proposals: "hold" (q_des = the measured pose -> u_task ~ 0) and
        # "ram" (drive every joint hard, which is what a player charging does).
        for tag, q_des in (
            ("hold", torch.tensor([s["joint_pos"]], dtype=torch.float32)),
            ("ram", torch.tensor([[jp + 1.0 for jp in s["joint_pos"]]], dtype=torch.float32)),
        ):
            shield.reset_(slice(None), torch.tensor([s["joint_pos"]], dtype=torch.float32))
            shield.prev_ctrl = torch.tensor([s["prev_ctrl"]], dtype=torch.float32)
            u_task = shield.task_increment(q_des)
            q_task = nets.robust_q(obs, u_task)
            u_sel = shield.step(obs, u_task, q_des)
            target = shield.apply_increment(u_sel)
            cases.append({
                "state": s["label"],
                "proposal": tag,
                "obs": [float(v) for v in obs[0]],
                "q_des": [float(v) for v in q_des[0]],
                "target_in": [float(v) for v in s["joint_pos"]],
                "prev_ctrl_in": [float(v) for v in s["prev_ctrl"]],
                "u_task": [float(v) for v in u_task[0]],
                "u_safe": [float(v) for v in u_safe[0]],
                "V": float(V),
                "thr": float(kappa * V),
                "q_task": float(q_task),
                "u_sel": [float(v) for v in u_sel[0]],
                "alpha": float(shield.alpha[0]),
                "decision": int(shield.decision[0]),
                "target_out": [float(v) for v in target[0]],
            })
            print(f"  {s['label']:<26} {tag:<5} V {float(V):+.6f}  thr {kappa*float(V):+.6f}"
                  f"  q_task {float(q_task):+.6f}  -> decision {int(shield.decision[0])}"
                  f"  alpha {float(shield.alpha[0]):.4f}")

    fixture = {
        "_readme": (
            "Reference values for app/filter.js, produced by CALLING the training "
            "code (not transcribing it): GameQcbfShieldAction._assemble_obs for the "
            "62-D observation (which pulls in opponent_kinematics and "
            "rect_rect_distance), and BatchedQcbfFilter.task_increment / .step / "
            ".apply_increment for the decision. Regenerate with "
            "tools/export_filter.py."
        ),
        "source_bundle": str(bundle.relative_to(MJLAB)),
        "filter_nets_md5": bundle_md5,
        "tolerance": args.js_tol,
        "obs_layout": OBS_LAYOUT,
        "field": {"length": field_len, "width": field_wid,
                  "center": [0.0, 0.0], "d_vis": d_vis,
                  "note": "the ASYMMETRIC arm leaves cfg.field_size None, so the "
                          "certificate sees the BUNDLE's 4.8 x 3.0 at env origin "
                          "(0,0) while the scenario field is 5.2 x 3.0 at (0.2, 0) "
                          "-- recon/03:534-541. Reproduce, do not fix."},
        "params": {
            "kappa": kappa, "cbf_max_iters": CBF_MAX_ITERS, "cbf_tol": CBF_TOL,
            "pessimistic": PESSIMISTIC, "inc_scale": inc_scale,
            "inc_smoothing": inc_smoothing, "value_guard": VALUE_GUARD,
            "value_release": VALUE_RELEASE, "intervention": "line_search",
            "q_lo": SOFT_LO, "q_hi": SOFT_HI,
        },
        "decisions": {"0": "task_pass", "1": "fallback", "2": "line_search",
                      "3": "guard", "4": "handback", "5": "projected_gradient"},
        "states": states,
        "obs": [[float(v) for v in row] for row in OBS],
        "cases": cases,
    }
    (TESTS_DIR / "filter_fixture.json").write_text(
        json.dumps(fixture, indent=1) + "\n", encoding="utf-8")
    n_ls = sum(1 for c in cases if c["decision"] == 2)
    n_fb = sum(1 for c in cases if c["decision"] == 1)
    n_tp = sum(1 for c in cases if c["decision"] == 0)
    print(f"\n  {len(cases)} cases: {n_tp} task_pass, {n_ls} line_search, {n_fb} fallback")

    # ---- 4. parity.json (append, never rewrite the policy rows) --------------
    parity_path = TESTS_DIR / "parity.json"
    parity = json.loads(parity_path.read_text())
    before = len(parity["policies"])
    parity["policies"].update(parity_rows)
    parity["_filter_readme"] = (
        "filter_ctrl / filter_dstb / filter_q1 / filter_q2 are the S2C certificate "
        "(collision_v5prox_15k_62d_game). They carry NO normalizer, the activation "
        "is sin, and the two actor heads are tanh; `obs` for filter_dstb is "
        "concat(obs62, u12) and for the twin heads concat(obs62, u12, d6). Written "
        "by tools/export_filter.py."
    )
    parity_path.write_text(json.dumps(parity, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote tests/parity.json ({before} -> {len(parity['policies'])} entries)")

    # ---- 5. manifest ---------------------------------------------------------
    man_path = POLICY_DIR / "manifest.json"
    man = json.loads(man_path.read_text())
    man["filter"] = {
        "available": True,
        "exporter": "tools/export_filter.py",
        "display": "S2C safety certificate (Q-CBF)",
        "source_checkpoint": meta["source_checkpoint"],
        "source_bundle": str(bundle.relative_to(MJLAB)),
        "filter_nets_md5": bundle_md5,
        "authority": "recon/03_safety_filter.md; src/tasks/game/mdp/game_qcbf_action.py, "
                     "src/tasks/game/mdp/filtered_action.py",
        "nets": {
            "ctrl": {"json": "filter_ctrl.json", "bin": "filter_ctrl.bin",
                     "in": 62, "out": 12, "head": "tanh"},
            "dstb": {"json": "filter_dstb.json", "bin": "filter_dstb.bin",
                     "in": 74, "out": 6, "head": "tanh"},
            "q1": {"json": "filter_q1.json", "bin": "filter_q1.bin",
                   "in": 80, "out": 1, "head": None},
            "q2": {"json": "filter_q2.json", "bin": "filter_q2.bin",
                   "in": 80, "out": 1, "head": None},
        },
        "bin_bytes": sum(written[n]["bin_bytes"] for n in NETS),
        "activation": "sin",
        "normalizer": None,
        "params": {
            "kappa": kappa,                       # deploy.yaml filter.kappa
            "cbf_max_iters": CBF_MAX_ITERS,       # exit_driver.py:712 (hardcoded)
            "cbf_tol": CBF_TOL,                   # exit_driver.py:713 (hardcoded)
            "pessimistic": PESSIMISTIC,           # exit_driver.py:711
            "value_eps": float(y["filter"]["value_eps"]),   # C++ lane only
            "value_guard": VALUE_GUARD,           # exit_driver.py:700-703
            "value_release": VALUE_RELEASE,
            "intervention": "line_search",        # absent from the yaml -> default
            "full_takeover": False,
            "fallback_line_search": False,        # make_inmemory_shield default
            "kin_enabled": False,                 # no tilt_guard key in the yaml
            "sup_enabled": False,                 # game_qcbf_action.py:182 literal
            "guard_enabled": False,               # game_qcbf_action.py:183 literal
            "reseed_track_err": RESEED_TRACK_ERR,
        },
        "increment": {
            "scale": inc_scale,                   # deploy.yaml increment.scale
            "smoothing": inc_smoothing,           # deploy.yaml increment.smoothing
            "inverse_denom": inc_scale * inc_smoothing,
            "q_lo": SOFT_LO,
            "q_hi": SOFT_HI,
            "q_limits_source": "entity soft_joint_pos_limits (game_qcbf_action.py:147), "
                               "numerically equal to deploy.yaml increment.joint_pos_*",
        },
        "obs": {
            "dim": 62,
            "layout": OBS_LAYOUT,
            "slices": {
                "proprio": [0, 48], "wall_margins": [48, 52],
                "heading": [52, 54], "opponent_tail": [54, 62],
            },
            "joint_order": "FL, FR, RL, RR x (hip, thigh, calf) -- sim order; "
                           "joint_ids_map / foot_ids_map in the yaml are the "
                           "hardware permutation and are NOT used here",
            "joint_pos_is_absolute": True,
            "prev_ctrl_is_filter_output": True,
        },
        "field": {
            "length": field_len, "width": field_wid, "center": [0.0, 0.0],
            "d_vis": d_vis,
            "note": "the asym preset leaves field_size None, so the certificate "
                    "reads the BUNDLE rectangle (4.8 x 3.0 at the env origin), not "
                    "the scenario's 5.2 x 3.0 at (0.2, 0) -- recon/03:534-541",
        },
        "collision": {
            "half_extents": list(COLLIDE_HALF), "fwd_offset": COLLIDE_FWD_OFFSET,
            "d_vis": D_VIS_COL, "d_max": D_MAX_COL,
            "no_opponent_tail": list(NO_OPPONENT_TAIL),
        },
        "gains": GAINS,
        "gain_blend_default": False,
        "gain_blend_note": "game_qcbf_action.py:238 disables the blend at "
                           "num_envs == 1, which is the browser's case and the "
                           "case every single-env demo render ran on; app/filter.js "
                           "keeps it off by default and exposes it as a flag",
        "cost_per_filtered_robot_per_step": {
            "mlp_forwards_floor": 7, "mlp_forwards_typical": "13-19",
            "mlp_forwards_ceiling": 67,
            "mac_floor": 1472512, "mac_per_line_search_iter": 455168,
            "note": "1 ctrl + 2 robust_q at the floor; +1 robust_q per line-search "
                    "iteration (recon/03:452-482)",
        },
    }
    man_path.write_text(json.dumps(man, indent=2) + "\n", encoding="utf-8")
    print(f"wrote assets/policies/manifest.json (added the `filter` key; "
          f"`policies` still has {len(man['policies'])} rows)")

    total = sum(written[n]["bin_bytes"] for n in NETS)
    print(f"\nTOTAL filter payload: {total:,} B of .bin")
    print(f"WORST parity: {worst64:.3e} float64 (gate {args.tol:.0e})  |  "
          f"{worst32:.3e} float32 (JS budget {args.js_tol:.0e})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

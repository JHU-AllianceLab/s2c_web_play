#!/usr/bin/env python3
"""Export every actor the S2C web game offers to a dependency-free (.bin + .json) pair.

Run with the mjlab venv (torch + numpy):

    /home/ray/Disk_ext/Go2/envs/mjlab_venv/bin/python tools/export_policy.py

What it writes (all paths relative to the repo root):

    assets/policies/<name>.bin       little-endian float32, per Linear layer:
                                     weight (row-major, out x in) then bias (out)
    assets/policies/<name>.json      shapes, activation, normalizer, provenance
    assets/policies/manifest.json    the machine-readable asset list, grouped by
                                     game and seat, with menu display names
    tests/parity.json                8 fixed obs per policy + the exact float32
                                     outputs of the ORIGINAL torch actor

THE ROSTER
----------
DESIGN.md section 3 is the only authority; recon/01 (asym) and recon/02 (sym)
carry the md5s and the seat/half evidence. One checkpoint per (method, seat):

  player walk   fastwalk_v3 model_9000                                47 -> 12
  sym   (AI always sits at seat B; the human takes seat A)            60 -> 12
        S2C  sym_v5_s17/game_3200     defender seat = B half  native
        ET   sym_n_v2_s23/game_1700   attacker seat = A half  ROTATED
        Nom  sym_p_v3_s29/game_3300   attacker seat = A half  ROTATED
        CPO  sym_c_v3_s59/game_2700   defender seat = B half  native
        Lag  sym_l_v8_s103/game_1000  defender seat = B half  native
  asym  (the human picks a role, so the AI needs BOTH seats)          60 -> 12
        S2C  v133fdrw/game_6000 (atk)      v135fdrw/game_5000 (def)
        ET   v90_g3600 (atk)               v90_g2000 (def)
        Nom  wbc_p25_s23/game_1700 (atk)   game_1400 (def)
        CPO  wbc_c6_s71/game_1500 (atk)    game_4400 (def)
        Lag  wbc_l4_d4_s71/game_2700 (atk) game_1800 (def)

THE NETWORK, verbatim from the training stack
---------------------------------------------
Every actor in this project is an ``rsl_rl`` ``MLPModel``:

    y = mlp( (x - obs_normalizer._mean) / (obs_normalizer._std + 1e-2) )

* ``EmpiricalNormalization.forward`` is ``(x - _mean) / (_std + eps)``,
  ``eps = 1e-2``
  (rsl_rl/modules/normalization.py:18 declares the default, :46-48 is the forward).
  ``_std`` is the buffer the module reads -- do not recompute ``sqrt(_var)``.
* ``mlp`` is ``rsl_rl.modules.mlp.MLP``: Linear, ELU, Linear, ELU, Linear, ELU,
  Linear -- ``last_activation=None`` means the head is bare
  (rsl_rl/modules/mlp.py:52-63). State-dict keys are ``mlp.{0,2,4,6}.{weight,bias}``;
  1/3/5 are the parameterless ELUs. ``resolve_nn_activation("elu")`` is
  ``torch.nn.ELU()`` (rsl_rl/utils/utils.py:48), so alpha = 1.0.
* Hidden dims (512, 256, 128), activation elu, obs_normalization true:
  src/tasks/game/rl/cfg.py:22-32 and src/tasks/sym_game/rl/cfg.py (same
  ``_default_actor``); every exported run bundle manifest agrees.
* Deterministic inference is the raw MLP output: ``GaussianDistribution``'s
  ``deterministic_output`` returns its argument unchanged
  (rsl_rl/modules/distribution.py:182-184). ``distribution.std_param`` is
  training-time exploration only and is NOT exported. No tanh, no clip:
  ``clip_actions`` is None for these tasks, so the 12-vector is unbounded.

NORMALIZER PLACEMENT -- the choice this exporter makes
------------------------------------------------------
The normalizer is kept OUT of the weights and shipped in the json as
``norm = {mean, std, eps}``.  The JS runtime must do

    z[i] = (obs[i] - norm.mean[i]) / (norm.std[i] + norm.eps)

before the first Linear.  It is NOT folded into layer 0.  Reason: three critic
dims have ``_std == 0`` in this family and the ``+ eps`` is what keeps that
finite; folding would hide the eps in a fused bias and make the seat rotation
(which negates ``mean[42:46]`` only) impossible to express.

THE pi-ROTATION (sym seats) -- BAKED IN AT EXPORT TIME
------------------------------------------------------
Seat A attacks +x, seat B attacks -x (sym_touchdown.py:104-110, ``direction``).
Of the 60 actor dims exactly one term is in the absolute frame: ``arena_pose =
(x, y, cos yaw, sin yaw)`` at ``[42:46)`` (sym_game/mdp/observations.py:24-34,
concat order src/tasks/sym_game/game_env_cfg.py:332-402).  Rotating the world
by pi about (0,0) maps (x,y) -> (-x,-y) and yaw -> yaw+pi, so ALL FOUR
components negate; every other term is body-frame or a clock, and
``{seat}_line = line_x - direction*x`` (sym_game/mdp/sym.py:145) is invariant
because seating a half at the other seat flips both ``x`` and ``direction``.

``pi_rotate_actor()`` bakes it the same way ``scripts/splice_sym_v5_g3500B.py``
and ``/home/ray/demo_sym/make_sym_matchup.py`` do: negate ``_mean[42:46]`` and
``mlp.0.weight[:, 42:46]``; ``_var``/``_std``/every bias/every deeper layer
untouched.  ``--check-rotation`` (on by default for rotated exports) runs the
full-net equivalence assert ``rot(x) == orig(Rx)`` and the not-a-no-op assert.

The exported ``<name>.json`` records ``rotation_baked``, and the browser must
NOT rotate again.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
POLICY_DIR = REPO / "assets" / "policies"
TESTS_DIR = REPO / "tests"
CACHE = REPO / ".cache"

MJLAB = Path("/home/ray/Disk_ext/Go2/Project/unitree_rl_mjlab")
BACKUP5090 = Path("/home/ray/Disk_ext/Go2/backup_gpu5090/rescue_2026-08-23")
SPLICES = Path("/home/ray/Disk_ext/Go2/Meeting/9th_meeting/cells_w40/matchups")

# --- constants taken from the training code, with their source lines ---------

# rsl_rl/modules/normalization.py:18 (default), :46-48 (the forward)
NORM_EPS = 1e-2

# The absolute-frame slice of the 60-D game actor observation.
# 3 base_ang_vel + 3 projected_gravity + 12 joint_pos + 12 joint_vel
# + 12 actions = 42, then arena_pose is 4 wide.
# src/tasks/sym_game/game_env_cfg.py:332-402 (concat order),
# src/tasks/sym_game/mdp/observations.py:24-34 (root_pose_2d -> 4).
# Same layout on the asym side: src/tasks/game/game_env_cfg.py:313-386.
ARENA_POSE_LO, ARENA_POSE_HI = 42, 46

# FL(hip,thigh,calf) FR RL RR -- src/assets/robots/unitree_go2/go2_constants.py:72-81
# (INIT_STATE.joint_pos: ".*thigh_joint" 0.9, ".*calf_joint" -1.8,
#  ".*R_hip_joint" +0.1, ".*L_hip_joint" -0.1)
DEFAULT_JOINT_POS = [
    -0.1, 0.9, -1.8,   # FL
    +0.1, 0.9, -1.8,   # FR
    -0.1, 0.9, -1.8,   # RL
    +0.1, 0.9, -1.8,   # RR
]

JOINT_ORDER = ["FL_hip", "FL_thigh", "FL_calf",
               "FR_hip", "FR_thigh", "FR_calf",
               "RL_hip", "RL_thigh", "RL_calf",
               "RR_hip", "RR_thigh", "RR_calf"]

# src/tasks/velocity/velocity_env_cfg.py:156-157 (walk: scale 0.25, use_default_offset)
# src/tasks/game/robots.py:166 and src/tasks/sym_game/robots.py:166 (game: 0.25)
ACTION_SCALE = 0.25

# Menu labels -- DESIGN.md section 1 / the task brief.
DISPLAY = {
    "s2c": "S2C",
    "et": "ET",
    "nom": "Nominal",
    "cpo": "CPO",
    "lag": "Lagrangian",
}

# The v25 increment integrator every game actor's output goes through
# (src/tasks/safety/mdp/ctrl_action.py:179-191). Reproduced in the json so the
# JS action path cannot drift from the training code.
INCREMENT = {
    "scale": 0.5,           # increment_scale
    "smoothing": 0.3,       # action_smoothing (EMA on the persistent target)
    "inverse_denom": 0.15,  # 0.5 * 0.3 -- the certified box, s*alpha
    "source": "src/tasks/safety/mdp/ctrl_action.py:179-191",
    "reset": "target is re-seeded from the MEASURED joint_pos on every reset "
             "(ctrl_action.py:107-122)",
}


# --- the asset list: DESIGN.md section 3 is the only authority ---------------
#
# seat            : the seat in the BROWSER GAME this file is loaded for
# ckpt_seat       : the key inside the .pt the actor is lifted from
# native_half     : sym only -- the half the weights were trained on
# upstream        : the canonical (5080) path, i.e. what the eval spec names
# upstream_md5    : md5 of THAT file, from recon/01 / recon/02
# source          : the bytes this script actually reads

POLICIES: list[dict] = [
    dict(
        name="player_walk_fastwalk_v3_9000",
        corroborate=None,
        expect_bytes=4741877,
        role="player_walk",
        game="walk",
        seat=None,
        method=None,
        display="Player walk",
        # DESIGN.md 3.1 / recon/04.
        source=MJLAB / "logs/rsl_rl/go2_velocity_fast"
                       "/2026-08-09_17-32-07_fastwalk_v3/model_9000.pt",
        upstream="/home/ray/Go2/Project/unitree_rl_mjlab/logs/rsl_rl/"
                 "go2_velocity_fast/2026-08-09_17-32-07_fastwalk_v3/model_9000.pt",
        upstream_md5="120c17c78d1dd10b732840b5189a42d6",
        provenance="local original; md5 matches the 5080 copy (DESIGN.md 3.1)",
        ckpt_seat=None,            # flat checkpoint: ckpt["actor_state_dict"]
        obs_dim=47,
        act_dim=12,
        native_half=None,
        action_path="walk_affine",
        notes=(
            "Unitree-Go2-Flat-Fast walker driven by the human's WASD+QE command. "
            "47-D obs = base_ang_vel(3) | projected_gravity(3) | command(3) | "
            "phase(2, sin/cos of a 0.6 s clock, zeroed when |cmd| < 0.1) | "
            "joint_pos-default(12) | joint_vel(12) | last_raw_action(12); every "
            "obs scale is 1.0, no clip, no noise at play time. "
            "Action path: q_des = default_joint_pos + 0.25*a straight to the "
            "position actuators (NO integrator). Store the RAW action for the "
            "next step's last_action block. Command box "
            "vx [-1.5, 3.0], vy [-1.0, 1.0], wz [-2.0, 2.0]. 50 Hz."
        ),
    ),

    # ---- SYMMETRIC: the AI always sits at seat B ----------------------------
    dict(
        name="sym_s2c_B",
        corroborate=None,
        expect_bytes=9554571,
        role="ai_opponent",
        game="sym",
        seat="B",
        method="s2c",
        display=DISPLAY["s2c"],
        # Pool member F_v8g1800A. S2C / Shield. Chosen for the GAME, in tests/head_on.mjs: as hard to beat as the pool leader (you win 25% at 1.4 m/s, 50% at 1.6) and by far the steadiest -- 0 falls and a 28 deg worst tilt over eight head-on approaches, against 70 deg for F_v7g2000A. Pool: 71.3% win, 5.15% fall.
        source=CACHE / "F_v8g1800A.pt",
        upstream="/home/ray/Go2/Project/unitree_rl_mjlab/logs/rsl_rl/game_sym_touchdown_go2_go2_wbc/2026-08-22_01-34-41_sym_v8_s59/game_1800.pt",
        upstream_md5="67dd9bdda02871d14717e6d2d252579f",
        provenance="pulled from the 5080 into .cache/; md5 verified byte-identical",
        ckpt_seat="attacker",
        obs_dim=60,
        act_dim=12,
        native_half="A",
        action_path="increment_integrator",
        notes="F_v8g1800A, the A half; seated at B, pi-rotation baked in.",
    ),
    dict(
        name="sym_et_B",
        corroborate=None,
        expect_bytes=9555531,
        role="ai_opponent",
        game="sym",
        seat="B",
        method="et",
        display=DISPLAY["et"],
        # Pool member ET_v3s31g1900A. ET / early termination. The only one of the five ET members that stays on its feet when a person walks into it: 13-25% falls against 75% for ET_v2g1700A, ET_g3700A and ET_v5s71g2100A. It is also the hardest baseline to beat, which is what an early-termination arm looks like when it does not topple.
        source=CACHE / "ET_v3s31g1900A.pt",
        upstream="/home/ray/Go2/Project/unitree_rl_mjlab/logs/rsl_rl/game_sym_touchdown_go2_go2_wbc/2026-08-27_02-33-54_sym_n_v3_s31/game_1900.pt",
        upstream_md5="260971c5c80e44845ee5559710befef8",
        provenance="pulled from the 5080 into .cache/; md5 verified byte-identical",
        ckpt_seat="attacker",
        obs_dim=60,
        act_dim=12,
        native_half="A",
        action_path="increment_integrator",
        notes="ET_v3s31g1900A, the A half; seated at B, pi-rotation baked in.",
    ),
    dict(
        name="sym_nom_B",
        corroborate=None,
        expect_bytes=9553547,
        role="ai_opponent",
        game="sym",
        seat="B",
        method="nom",
        display=DISPLAY["nom"],
        # Pool member P_g1600A. Nominal / safety penalty. The beatable one that keeps its feet: at 1.6 m/s you win 5 of 8, every win a clean touchdown, 0 falls, 39 deg worst tilt. The stronger P members win by falling over (P_v2s59g3700A topples in 8 of 8).
        source=CACHE / "P_g1600.pt",
        upstream="/home/ray/Go2/Project/unitree_rl_mjlab/archive/sym_anchors/P_g1600.pt",
        upstream_md5="425bb3cf7d73e14b28a42010befb09a1",
        provenance="pulled from the 5080 into .cache/; md5 verified byte-identical",
        ckpt_seat="attacker",
        obs_dim=60,
        act_dim=12,
        native_half="A",
        action_path="increment_integrator",
        notes="P_g1600A, the A half; seated at B, pi-rotation baked in.",
    ),
    dict(
        name="sym_cpo_B",
        corroborate=None,
        expect_bytes=11186763,
        role="ai_opponent",
        game="sym",
        seat="B",
        method="cpo",
        display=DISPLAY["cpo"],
        # Pool member C_v2s41g3700B. CPO / constrained policy optimisation. Beatable and steady: 75% of the head-on lines go to the player at 1.6 m/s, five of them on a touchdown, with 0-13% falls against 13% and a 74 deg tilt for C_v3s59g2700B.
        source=CACHE / "C_v2s41g3700B.pt",
        upstream="/home/ray/Go2/Project/unitree_rl_mjlab/archive/sym_anchors/C_v2s41g3700.pt",
        upstream_md5="4163bee804721494da21a37dd2bbc868",
        provenance="pulled from the 5080 into .cache/; md5 verified byte-identical",
        ckpt_seat="defender",
        obs_dim=60,
        act_dim=12,
        native_half="B",
        action_path="increment_integrator",
        notes="C_v2s41g3700B, the B half; seated at B, its native half, no rotation.",
    ),
    dict(
        name="sym_lag_B",
        corroborate=None,
        expect_bytes=14351627,
        role="ai_opponent",
        game="sym",
        seat="B",
        method="lag",
        display=DISPLAY["lag"],
        # Pool member L_v3s41g2800A. Lagrangian. Beatable at 63% of the head-on lines at 1.6 m/s on clean touchdowns, 0-13% falls. L_v8s103g1000B was steadier still but the player only took 38% off it.
        source=CACHE / "L_v3s41g2800A.pt",
        upstream="/home/ray/Go2/Project/unitree_rl_mjlab/archive/sym_anchors/L_v3s41g2800.pt",
        upstream_md5="da8b5e8e3acb6e1209dda2b1c63cd697",
        provenance="pulled from the 5080 into .cache/; md5 verified byte-identical",
        ckpt_seat="attacker",
        obs_dim=60,
        act_dim=12,
        native_half="A",
        action_path="increment_integrator",
        notes="L_v3s41g2800A, the A half; seated at B, pi-rotation baked in.",
    ),

    # ---- ASYMMETRIC: the human picks a role, so both seats ship -------------
    dict(
        name="asym_s2c_attacker",
        corroborate=None,
        expect_bytes=9560075,
        role="ai_opponent",
        game="asym",
        seat="attacker",
        method="s2c",
        display=DISPLAY["s2c"],
        # DESIGN.md 3.3 / recon/01 section 1 (OURS_drw). 5080-only original.
        source=CACHE / "asym_v133fdrw_game_6000.pt",
        upstream="/home/ray/Go2/Project/unitree_rl_mjlab/logs/rsl_rl/"
                 "game_touchdown_go2_go2_wbc/2026-08-17_22-23-27_v133fdrw/"
                 "game_6000.pt",
        upstream_md5="d8f3672494f744bf316a9d2e0f2a2c30",
        provenance="pulled from the 5080 into .cache/; md5 verified byte-identical",
        ckpt_seat="attacker",
        obs_dim=60,
        act_dim=12,
        native_half=None,
        action_path="increment_integrator",
        notes=(
            "S2C / Shield attacker, OURS_drw, v133fdrw game_6000 "
            "(seats['attacker']), seed 71, wide-DR Shield run. This is the "
            "attacker of the ICRA demo clips. The QCBF certificate overrode this "
            "policy on 21-52% of control steps in the scored cells "
            "(recon/01 section 5) -- shipping the bare MLP is a different, less "
            "safe policy, and the game must say so wherever the filter is off."
        ),
    ),
    dict(
        name="asym_s2c_defender",
        corroborate=None,
        expect_bytes=9558027,
        role="ai_opponent",
        game="asym",
        seat="defender",
        method="s2c",
        display=DISPLAY["s2c"],
        # DESIGN.md 3.3 / recon/01 section 1 (OURS_drw). 5080-only original.
        source=CACHE / "asym_v135fdrw_game_5000.pt",
        upstream="/home/ray/Go2/Project/unitree_rl_mjlab/logs/rsl_rl/"
                 "game_touchdown_go2_go2_wbc/2026-08-18_11-55-46_v135fdrw/"
                 "game_5000.pt",
        upstream_md5="fec34c66bd90487351d05c32fe1284bc",
        provenance="pulled from the 5080 into .cache/; md5 verified byte-identical",
        ckpt_seat="defender",
        obs_dim=60,
        act_dim=12,
        native_half=None,
        action_path="increment_integrator",
        notes=(
            "S2C / Shield defender, OURS_drw, v135fdrw game_5000 "
            "(seats['defender']), seed 103, wide-DR Shield run. Same filter "
            "caveat as the attacker seat. NOTE the run's own exported bundle "
            "names game_6000, not this pick -- lift the actor out of the "
            "game_*.pt seat, never out of bundles/ (recon/01 section 6)."
        ),
    ),
    dict(
        name="asym_et_attacker",
        corroborate=None,
        expect_bytes=9553291,
        role="ai_opponent",
        game="asym",
        seat="attacker",
        method="et",
        display=DISPLAY["et"],
        # DESIGN.md 3.3 / recon/01 (TERM). Local backup, byte-identical.
        source=BACKUP5090 / "a3run/evalck_v90sweep/v90_g3600.pt",
        upstream="/home/ray/Go2/a3run/evalck_v90sweep/v90_g3600.pt",
        upstream_md5="6a78cbbe6038d5d395941553ef26ddf2",
        provenance="local backup copy; md5 matches the 5080 original (recon/01 "
                   "section 1)",
        ckpt_seat="attacker",
        obs_dim=60,
        act_dim=12,
        native_half=None,
        action_path="increment_integrator",
        notes=(
            "ET / TERM attacker, v90_g3600 (seats['attacker']). Older "
            "checkpoint format: 15 top-level keys, no 'ref_arm'. Trained on the "
            "4.8 m field but scored and rendered at 5.2 x 3.0 like everything "
            "else (recon/01 section 0)."
        ),
    ),
    dict(
        name="asym_et_defender",
        corroborate=None,
        expect_bytes=9550155,
        role="ai_opponent",
        game="asym",
        seat="defender",
        method="et",
        display=DISPLAY["et"],
        source=BACKUP5090 / "a3run/evalck_v90sweep/v90_g2000.pt",
        upstream="/home/ray/Go2/a3run/evalck_v90sweep/v90_g2000.pt",
        upstream_md5="e52999a6309a076521a77e6fea8b64e2",
        provenance="local backup copy; md5 matches the 5080 original (recon/01 "
                   "section 1)",
        ckpt_seat="defender",
        obs_dim=60,
        act_dim=12,
        native_half=None,
        action_path="increment_integrator",
        notes="ET / TERM defender, v90_g2000 (seats['defender']). Same run family "
              "as the attacker seat, a different iteration.",
    ),
    dict(
        name="asym_nom_attacker",
        corroborate=SPLICES / "m1__PEN_s23__vs__CPO_s71.pt",
        expect_bytes=9547341,
        role="ai_opponent",
        game="asym",
        seat="attacker",
        method="nom",
        display=DISPLAY["nom"],
        # DESIGN.md 3.3 / recon/01 (PEN_s23). The upstream file is 5080-only; the
        # local spliced cell carries the identical seat state and records its
        # source in infos.eval_matchup (harness.py:419-470 _load_seat deep-copies
        # ckpt["seats"]["attacker"] verbatim and only resets "iter").
        source=SPLICES / "m0__PEN_s23__vs__PEN_s23.pt",
        upstream="/home/ray/Go2/Project/unitree_rl_mjlab/logs/rsl_rl/"
                 "game_touchdown_go2_go2_wbc_penalty/"
                 "2026-08-16_15-04-49_wbc_p25_s23/game_1700.pt",
        upstream_md5="cc9d1b9e692b61d83b256e6195f9deba",
        provenance="spliced eval cell; infos.eval_matchup.attacker_checkpoint "
                   "names the upstream file, whose md5 is recorded in recon/01 "
                   "but NOT recomputed here (the original is 5080-only)",
        ckpt_seat="attacker",
        obs_dim=60,
        act_dim=12,
        native_half=None,
        action_path="increment_integrator",
        notes="Nom / Penalty attacker, PEN_s23 = wbc_p25_s23 game_1700 "
              "(seats['attacker']). Never filtered.",
    ),
    dict(
        name="asym_nom_defender",
        corroborate=SPLICES / "m1__CPO_s71__vs__PEN_s23.pt",
        expect_bytes=9547341,
        role="ai_opponent",
        game="asym",
        seat="defender",
        method="nom",
        display=DISPLAY["nom"],
        source=SPLICES / "m0__PEN_s23__vs__PEN_s23.pt",
        upstream="/home/ray/Go2/Project/unitree_rl_mjlab/logs/rsl_rl/"
                 "game_touchdown_go2_go2_wbc_penalty/"
                 "2026-08-16_15-04-49_wbc_p25_s23/game_1400.pt",
        upstream_md5="0e6d9a6771a2aaf158bb319569c76271",
        provenance="spliced eval cell; infos.eval_matchup.defender_checkpoint "
                   "names the upstream file, whose md5 is recorded in recon/01 "
                   "but NOT recomputed here (the original is 5080-only)",
        ckpt_seat="defender",
        obs_dim=60,
        act_dim=12,
        native_half=None,
        action_path="increment_integrator",
        notes="Nom / Penalty defender, PEN_s23 = wbc_p25_s23 game_1400 "
              "(seats['defender']). Never filtered.",
    ),
    dict(
        name="asym_cpo_attacker",
        corroborate=SPLICES / "m1__CPO_s71__vs__PEN_s23.pt",
        expect_bytes=11170907,
        role="ai_opponent",
        game="asym",
        seat="attacker",
        method="cpo",
        display=DISPLAY["cpo"],
        source=SPLICES / "m0__CPO_s71__vs__CPO_s71.pt",
        upstream="/home/ray/Go2/Project/unitree_rl_mjlab/logs/rsl_rl/"
                 "game_touchdown_go2_go2_wbc_cpo/"
                 "2026-08-16_06-30-45_wbc_c6_s71/game_1500.pt",
        upstream_md5="1dd6483dde5bdaebbda6c8bc07b1de46",
        provenance="spliced eval cell; infos.eval_matchup.attacker_checkpoint "
                   "names the upstream file, whose md5 is recorded in recon/01 "
                   "but NOT recomputed here (the original is 5080-only)",
        ckpt_seat="attacker",
        obs_dim=60,
        act_dim=12,
        native_half=None,
        action_path="increment_integrator",
        notes="CPO attacker, CPO_s71 = wbc_c6_s71 game_1500 (seats['attacker']).",
    ),
    dict(
        name="asym_cpo_defender",
        corroborate=SPLICES / "m1__PEN_s23__vs__CPO_s71.pt",
        expect_bytes=11170907,
        role="ai_opponent",
        game="asym",
        seat="defender",
        method="cpo",
        display=DISPLAY["cpo"],
        source=SPLICES / "m0__CPO_s71__vs__CPO_s71.pt",
        upstream="/home/ray/Go2/Project/unitree_rl_mjlab/logs/rsl_rl/"
                 "game_touchdown_go2_go2_wbc_cpo/"
                 "2026-08-16_06-30-45_wbc_c6_s71/game_4400.pt",
        upstream_md5="da9ba043c5f4139c16cb56b280d842cd",
        provenance="spliced eval cell; infos.eval_matchup.defender_checkpoint "
                   "names the upstream file, whose md5 is recorded in recon/01 "
                   "but NOT recomputed here (the original is 5080-only)",
        ckpt_seat="defender",
        obs_dim=60,
        act_dim=12,
        native_half=None,
        action_path="increment_integrator",
        notes="CPO defender, CPO_s71 = wbc_c6_s71 game_4400 (seats['defender']).",
    ),
    dict(
        name="asym_lag_attacker",
        corroborate=SPLICES / "m1__LAG_s71__vs__CPO_s71.pt",
        expect_bytes=14340831,
        role="ai_opponent",
        game="asym",
        seat="attacker",
        method="lag",
        display=DISPLAY["lag"],
        source=SPLICES / "m0__LAG_s71__vs__LAG_s71.pt",
        upstream="/home/ray/Go2/Project/unitree_rl_mjlab/logs/rsl_rl/"
                 "game_touchdown_go2_go2_wbc_lag/"
                 "2026-08-15_18-50-22_wbc_l4_d4_s71/game_2700.pt",
        upstream_md5="1dbf684ebb16e691e10a8489095edf2f",
        provenance="spliced eval cell; infos.eval_matchup.attacker_checkpoint "
                   "names the upstream file, whose md5 is recorded in recon/01 "
                   "but NOT recomputed here (the original is 5080-only)",
        ckpt_seat="attacker",
        obs_dim=60,
        act_dim=12,
        native_half=None,
        action_path="increment_integrator",
        notes="Lagrangian attacker, LAG_s71 = wbc_l4_d4_s71 game_2700 "
              "(seats['attacker']).",
    ),
    dict(
        name="asym_lag_defender",
        corroborate=SPLICES / "m1__CPO_s71__vs__LAG_s71.pt",
        expect_bytes=14340831,
        role="ai_opponent",
        game="asym",
        seat="defender",
        method="lag",
        display=DISPLAY["lag"],
        source=SPLICES / "m0__LAG_s71__vs__LAG_s71.pt",
        upstream="/home/ray/Go2/Project/unitree_rl_mjlab/logs/rsl_rl/"
                 "game_touchdown_go2_go2_wbc_lag/"
                 "2026-08-15_18-50-22_wbc_l4_d4_s71/game_1800.pt",
        upstream_md5="ea9765653f19243a3fa37c852030e014",
        provenance="spliced eval cell; infos.eval_matchup.defender_checkpoint "
                   "names the upstream file, whose md5 is recorded in recon/01 "
                   "but NOT recomputed here (the original is 5080-only)",
        ckpt_seat="defender",
        obs_dim=60,
        act_dim=12,
        native_half=None,
        action_path="increment_integrator",
        notes="Lagrangian defender, LAG_s71 = wbc_l4_d4_s71 game_1800 "
              "(seats['defender']).",
    ),
]


# --- checkpoint loading ------------------------------------------------------

def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


_CKPT_CACHE: dict[Path, dict] = {}


def _load_ckpt(path: Path) -> dict:
    """torch.load once per file -- the splices serve two policies each."""
    key = path.resolve()
    if key not in _CKPT_CACHE:
        _CKPT_CACHE[key] = torch.load(path, map_location="cpu", weights_only=False)
    return _CKPT_CACHE[key]


def load_actor(path: Path, seat: str | None) -> "OrderedDict[str, torch.Tensor]":
    """Return the actor state dict of a walk (flat) or game (per-seat) checkpoint.

    Walk checkpoints: ckpt["actor_state_dict"].
    Game checkpoints (game_format 1): ckpt["seats"][seat]["actor_state_dict"].
    """
    ckpt = _load_ckpt(path)
    if seat is None:
        sd = ckpt["actor_state_dict"]
    else:
        assert ckpt.get("game_format") == 1, f"{path}: not a game checkpoint"
        seats = ckpt["seats"]
        if seat not in seats:
            raise KeyError(f"{path}: seats are {sorted(seats)}, asked for {seat!r}")
        sd = seats[seat]["actor_state_dict"]
    return OrderedDict((k, v.detach().clone()) for k, v in sd.items())


def splice_infos(path: Path) -> dict | None:
    """infos.eval_matchup of a spliced cell, if this file is one."""
    ckpt = _load_ckpt(path)
    return (ckpt.get("infos") or {}).get("eval_matchup")


def layer_shapes(sd) -> list[tuple[int, int]]:
    """[(in, out), ...] for mlp.0, mlp.2, mlp.4, mlp.6 in order."""
    idxs = sorted(
        int(k.split(".")[1]) for k in sd if k.startswith("mlp.") and k.endswith(".weight")
    )
    out = []
    for i in idxs:
        w = sd[f"mlp.{i}.weight"]
        b = sd[f"mlp.{i}.bias"]
        assert b.shape == (w.shape[0],), f"mlp.{i}: bias {tuple(b.shape)} vs weight {tuple(w.shape)}"
        out.append((int(w.shape[1]), int(w.shape[0])))
    for (i_in, i_out), (j_in, _) in zip(out, out[1:]):
        assert i_out == j_in, f"layer chain broken: {i_out} -> {j_in}"
    return out


# --- the pi-rotation ---------------------------------------------------------

def pi_rotate_actor(sd, obs_dim: int, lo: int = ARENA_POSE_LO, hi: int = ARENA_POSE_HI):
    """A copy of one actor's state dict that reads the pi-rotated world.

    Rotation by pi about (0,0) maps (x, y) -> (-x, -y) and yaw -> yaw + pi, so all
    four ``arena_pose`` components negate and every other term (body-frame IMU /
    joints / actions / rel_*, and the own-goal-relative ``<seat>_line``) is
    invariant. The bake is exact because ``_std`` is even under the sign flip:

        y = mlp((x - mean) / (std + eps)),  R = diag(+-1), R^2 = I
        (Rx - mean)/std = R * ((x - R*mean)/std)
        with mean' = R*mean and W0' = W0*R:
            W0' ((x - mean')/std) = W0 ((Rx - mean)/std)

    so the residual R is absorbed by layer 0's input columns. Deeper layers and
    every bias are untouched. Matches scripts/splice_sym_v5_g3500B.py (LO,HI=42,46)
    and /home/ray/demo_sym/make_sym_matchup.py element for element.
    """
    out = OrderedDict((k, v.clone() if torch.is_tensor(v) else v) for k, v in sd.items())
    w = out["mlp.0.weight"]
    assert w.shape[1] == obs_dim, f"first layer takes {w.shape[1]}, expected {obs_dim}"
    m = out["obs_normalizer._mean"]
    assert m.shape[-1] == obs_dim, f"normalizer is {m.shape[-1]}-D, expected {obs_dim}"
    w[:, lo:hi] = -w[:, lo:hi]
    m[..., lo:hi] = -m[..., lo:hi]
    return out


def check_rotation_equivalence(sd, rot, obs_dim: int, n: int = 512, seed: int = 0):
    """Prove rot(x) == orig(Rx) on the FULL net, and that it is not a no-op.

    This is the assert from scripts/splice_sym_v5_g3500B.py, widened to the whole
    4-layer ELU net instead of just layer 0.
    """
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, obs_dim, generator=g) * 3.0
    rx = x.clone()
    rx[:, ARENA_POSE_LO:ARENA_POSE_HI] = -rx[:, ARENA_POSE_LO:ARENA_POSE_HI]
    net_rot, net_orig = TorchActor(rot).double(), TorchActor(sd).double()
    with torch.no_grad():
        xd, rxd = x.double(), rx.double()
        err = (net_rot(xd) - net_orig(rxd)).abs().max().item()
        diff = (net_rot(xd) - net_orig(xd)).abs().max().item()
    # The identity is exact in IEEE arithmetic: negating W0's column and mean's
    # entry negates both factors of the same product, so layer 0 sums the same
    # terms in the same order. Anything above float64 noise means the sign-flip
    # identity broke, not that rounding crept in.
    assert err < 1e-12, f"rotation bake-in is not exact: max err {err}"
    assert diff > 1e-6, "rotated copy equals the original -- arena_pose slice wrong?"
    return err, diff


def check_obs_layout(path: Path, seat: str, name: str) -> list[int]:
    """Pin arena_pose at [42:46) from the checkpoint itself, not from a document.

    recon/02 section 1.2: the CRITIC normalizer's ``_var`` has exactly three zero
    entries, at 55/56/57 -- the ``cmd`` term, which is identically zero in this
    game (observations.py:275-286, zero_twist). A 66-D critic whose only
    zero-variance dims are 55/56/57 pins index 54 as ``{seat}_line`` and therefore
    ``arena_pose`` at [42:46), which is the slice the pi-rotation negates. If a
    checkpoint ever failed this, the rotation would be silently wrong.

    The ACTOR normalizer does NOT have zeros there -- its running stats were
    inherited through a warm start from bootstraps that ran with a live pilot cmd
    -- so this is a critic-only probe, and the actor's stored mean/std must be
    shipped verbatim.
    """
    ck = _load_ckpt(path)
    var = ck["seats"][seat]["critic_state_dict"]["obs_normalizer._var"].reshape(-1)
    zeros = (var == 0).nonzero().reshape(-1).tolist()
    assert int(var.numel()) == 66, f"{name}: critic normalizer is {var.numel()}-D, expected 66"
    assert zeros == [55, 56, 57], (
        f"{name}: critic zero-variance dims are {zeros}, expected [55, 56, 57] -- "
        f"the 60-D obs layout is not what the rotation slice assumes"
    )
    return zeros


def _actor_fingerprint(sd) -> str:
    """sha256 over the actor's tensors, key-sorted -- a bit-level seat identity."""
    h = hashlib.sha256()
    for k in sorted(sd):
        h.update(k.encode())
        h.update(np.ascontiguousarray(sd[k].numpy()).tobytes())
    return h.hexdigest()


def check_splice_corroboration(sd, other: Path, seat: str, upstream: str, name: str) -> dict:
    """A second, independently written splice must carry the same seat, bit for bit.

    The picked baseline seats live only on the 5080, so their md5 cannot be
    recomputed here; provenance rests on infos.eval_matchup. This raises that
    from one record to two: m0__X__vs__X.pt and m1__X__vs__Y.pt were produced by
    separate ``build_matchup_checkpoint`` calls (harness.py:432) from the same
    upstream .pt, so their X seats must be identical AND must name the same
    source. A mismatch would mean one of them was built from something else.
    """
    ck = _load_ckpt(other)
    sd2 = ck["seats"][seat]["actor_state_dict"]
    src2 = ck["infos"]["eval_matchup"][f"{seat}_checkpoint"]
    fa, fb = _actor_fingerprint(sd), _actor_fingerprint(sd2)
    assert src2 == upstream, (
        f"{name}: {other.name} records {src2}, the asset list says {upstream}"
    )
    assert fa == fb, (
        f"{name}: the {seat} seat differs between {other.name} and the export source"
    )
    return {"file": other.name, "seat_sha256": fa, "records": src2}


def check_native_half(sd, native_half: str, name: str) -> float:
    """arena_pose's cos-yaw running mean must point the way the half attacks.

    recon/02 section 1.6: an A half faces +x (mean[44] > 0), a B half faces -x
    (mean[44] < 0). This is an independent, numeric test of the seat claim in the
    asset list -- it does not trust the checkpoint's key name.
    """
    cos_yaw = float(sd["obs_normalizer._mean"].reshape(-1)[ARENA_POSE_LO + 2])
    want_positive = native_half == "A"
    assert (cos_yaw > 0) == want_positive, (
        f"{name}: claimed half {native_half} but arena_pose mean cos(yaw) = "
        f"{cos_yaw:+.5f}"
    )
    return cos_yaw


# --- the reference torch actor ----------------------------------------------

class TorchActor(torch.nn.Module):
    """The ORIGINAL actor, rebuilt from a checkpoint state dict.

    Uses rsl_rl's own EmpiricalNormalization so the eps handling is the library's,
    and plain Linear/ELU for the mlp (which is exactly what rsl_rl.modules.mlp.MLP
    builds: Linear, ELU, Linear, ELU, Linear, ELU, Linear).
    """

    def __init__(self, sd):
        super().__init__()
        from rsl_rl.modules.normalization import EmpiricalNormalization

        obs_dim = int(sd["obs_normalizer._mean"].shape[-1])
        self.obs_normalizer = EmpiricalNormalization(obs_dim, eps=NORM_EPS)
        self.obs_normalizer._mean.copy_(sd["obs_normalizer._mean"])
        self.obs_normalizer._var.copy_(sd["obs_normalizer._var"])
        self.obs_normalizer._std.copy_(sd["obs_normalizer._std"])

        shapes = layer_shapes(sd)
        mods, idxs = [], sorted(
            int(k.split(".")[1]) for k in sd if k.startswith("mlp.") and k.endswith(".weight")
        )
        for n, (i, (i_in, i_out)) in enumerate(zip(idxs, shapes)):
            lin = torch.nn.Linear(i_in, i_out)
            lin.weight.data.copy_(sd[f"mlp.{i}.weight"])
            lin.bias.data.copy_(sd[f"mlp.{i}.bias"])
            mods.append(lin)
            if n < len(shapes) - 1:
                mods.append(torch.nn.ELU())      # alpha = 1.0, torch default
        self.mlp = torch.nn.Sequential(*mods)
        self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.obs_normalizer(x))


# --- the exported format -----------------------------------------------------

def export_policy(spec: dict, sd, rotation_baked: bool, extra: dict,
                  local_md5: str) -> dict:
    """Write <name>.bin + <name>.json. Returns the json dict."""
    shapes = layer_shapes(sd)
    obs_dim, act_dim = shapes[0][0], shapes[-1][1]
    assert obs_dim == spec["obs_dim"], f"{spec['name']}: obs {obs_dim} != {spec['obs_dim']}"
    assert act_dim == spec["act_dim"], f"{spec['name']}: act {act_dim} != {spec['act_dim']}"

    idxs = sorted(
        int(k.split(".")[1]) for k in sd if k.startswith("mlp.") and k.endswith(".weight")
    )
    blob = bytearray()
    for i in idxs:
        w = sd[f"mlp.{i}.weight"].to(torch.float32).contiguous().numpy()  # (out, in) row-major
        b = sd[f"mlp.{i}.bias"].to(torch.float32).contiguous().numpy()
        blob += w.astype("<f4").tobytes(order="C")
        blob += b.astype("<f4").tobytes(order="C")

    (POLICY_DIR / f"{spec['name']}.bin").write_bytes(bytes(blob))

    mean = sd["obs_normalizer._mean"].reshape(-1).to(torch.float32).numpy()
    std = sd["obs_normalizer._std"].reshape(-1).to(torch.float32).numpy()

    src = Path(spec["source"])
    meta = {
        "name": spec["name"],
        "role": spec["role"],
        "game": spec["game"],
        "seat": spec["seat"],
        "method": spec["method"],
        "display": spec["display"],
        "source_checkpoint": spec["upstream"],
        "md5": local_md5,
        "obs_dim": obs_dim,
        "act_dim": act_dim,
        "layers": [{"in": a, "out": b} for a, b in shapes],
        "activation": "elu",                 # between layers; the head is linear
        "activation_alpha": 1.0,
        "head_activation": None,
        "norm": {
            "mean": [float(v) for v in mean],
            "std": [float(v) for v in std],
            "eps": NORM_EPS,
        },
        "rotation_baked": rotation_baked,
        "action_path": spec["action_path"],
        # --- everything below is context, not part of the frozen field list ---
        "local_copy": str(src),
        "local_md5": local_md5,
        "local_bytes": src.stat().st_size,
        "upstream_checkpoint_md5": spec.get("upstream_md5"),
        "upstream_md5_verified": bool(extra.get("upstream_md5_verified")),
        "provenance": spec["provenance"],
        "seat_in_checkpoint": spec["ckpt_seat"],
        "native_half": spec["native_half"],
        "norm_placement": (
            "separate -- the runtime computes (obs - norm.mean) / (norm.std + "
            "norm.eps) before layer 0; it is NOT folded into the weights"
        ),
        "action_scale": ACTION_SCALE,
        "default_joint_pos": DEFAULT_JOINT_POS,
        "joint_order": JOINT_ORDER,
        "control_hz": 50.0,
        "bin": f"{spec['name']}.bin",
        "bin_bytes": len(blob),
        "bin_layout": (
            "little-endian float32; for each Linear in order: weight (row-major, "
            "out x in) then bias (out)"
        ),
        "forward": (
            "z = (obs - norm.mean) / (norm.std + norm.eps); "
            "h = ELU(L0 z); h = ELU(L1 h); h = ELU(L2 h); action = L3 h"
        ),
        "notes": spec["notes"],
    }
    if spec["action_path"] == "increment_integrator":
        meta["increment"] = dict(INCREMENT)
    if rotation_baked:
        meta["rotation"] = {
            "slice": [ARENA_POSE_LO, ARENA_POSE_HI],
            "term": "arena_pose (x, y, cos yaw, sin yaw), env-origin frame",
            "applied_to": ["obs_normalizer._mean[42:46]", "mlp.0.weight[:, 42:46]"],
            "from_half": spec["native_half"],
            "to_seat": spec["seat"],
            "equivalence_max_abs_err": extra.get("rot_err"),
            "not_a_noop_max_abs_diff": extra.get("rot_diff"),
            "warning": "the browser must NOT rotate the observation again",
        }
    if extra.get("splice"):
        meta["splice_infos"] = extra["splice"]
    if extra.get("corroboration"):
        meta["splice_corroboration"] = extra["corroboration"]
    if extra.get("arena_cos_yaw") is not None:
        meta["native_half_probe"] = {
            "arena_pose_mean_cos_yaw": extra["arena_cos_yaw"],
            "measured_on": "the AS-LOADED weights, BEFORE any rotation bake",
            "rule": "an A half faces +x (cos yaw > 0), a B half faces -x "
                    "(cos yaw < 0); recon/02 section 1.6",
        }
    if extra.get("critic_zero_var_dims") is not None:
        meta["obs_layout_probe"] = {
            "critic_zero_variance_dims": extra["critic_zero_var_dims"],
            "meaning": "the cmd term is identically zero in this game, so these "
                       "three dims pin index 54 = {seat}_line and arena_pose at "
                       "[42:46)",
        }

    (POLICY_DIR / f"{spec['name']}.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    return meta


# --- the numpy re-reader (the independent check of our own format) -----------

def numpy_load(name: str):
    """Re-read <name>.bin + <name>.json with nothing but numpy.

    Returns (meta, layers, norm, forward). ``forward(obs, dtype)`` is the exact
    computation the JS runtime has to perform; ``dtype=np.float32`` is what the
    browser will actually do, ``np.float64`` removes float32 summation noise so
    the export itself can be checked against torch at 1e-6.
    """
    meta = json.loads((POLICY_DIR / f"{name}.json").read_text())
    raw = (POLICY_DIR / meta["bin"]).read_bytes()
    want = sum(a * b + b for a, b in ((l["in"], l["out"]) for l in meta["layers"])) * 4
    assert len(raw) == want, f"{name}: bin is {len(raw)} B, layers need {want} B"
    buf = np.frombuffer(raw, dtype="<f4")
    layers, off = [], 0
    for l in meta["layers"]:
        n_in, n_out = l["in"], l["out"]
        w = buf[off:off + n_out * n_in].reshape(n_out, n_in); off += n_out * n_in
        b = buf[off:off + n_out]; off += n_out
        layers.append((w, b))
    assert off == buf.size, f"{name}: {buf.size - off} trailing float32 in the bin"
    norm = dict(
        mean=np.asarray(meta["norm"]["mean"], dtype=np.float32),
        std=np.asarray(meta["norm"]["std"], dtype=np.float32),
        eps=float(meta["norm"]["eps"]),   # exact 0.01, cast at the working precision
    )
    alpha = float(meta["activation_alpha"])

    def forward(obs: np.ndarray, dtype=np.float32) -> np.ndarray:
        x = ((np.asarray(obs, dtype=dtype) - norm["mean"].astype(dtype))
             / (norm["std"].astype(dtype) + dtype(norm["eps"]))).astype(dtype)
        for k, (w, b) in enumerate(layers):
            x = (x @ w.astype(dtype).T + b.astype(dtype)).astype(dtype)
            if k < len(layers) - 1:
                # ELU(x) = x if x > 0 else alpha*(exp(x) - 1); expm1 is evaluated
                # only on the non-positive branch so large positives cannot overflow.
                x = np.where(x > 0, x, dtype(alpha) * np.expm1(np.minimum(x, 0))).astype(dtype)
        return x

    return meta, layers, norm, forward


# --- the parity fixture ------------------------------------------------------

def obs_seed(name: str) -> int:
    """A seed tied to the policy NAME, so reordering the roster cannot move it."""
    return int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)


def make_obs(sd, obs_dim: int, seed: int) -> np.ndarray:
    """8 fixed observations drawn from the policy's OWN running statistics.

    obs = mean + std*z, so every dim lands where the policy saw data in training
    and the normalized input is O(1) in both signs (both ELU branches). Rows 0-3
    use z ~ N(0, 1), rows 4-7 use z ~ N(0, 3) to push the activations wider --
    measured float32 error against torch stays at 2.7e-6 even on the wide rows,
    inside the 1e-5 the JS runtime is held to.
    """
    rng = np.random.default_rng(seed)
    mean = sd["obs_normalizer._mean"].reshape(-1).to(torch.float32).numpy()
    std = sd["obs_normalizer._std"].reshape(-1).to(torch.float32).numpy()
    z = np.concatenate([rng.standard_normal((4, obs_dim)),
                        rng.standard_normal((4, obs_dim)) * 3.0]).astype(np.float32)
    return (mean + std * z).astype(np.float32)


def verify_export(name: str, sd, obs: np.ndarray, act_torch32: np.ndarray, tol: float):
    """Re-read our own .bin/.json and prove it IS the torch actor.

    Three checks, strongest first:

    1. BIT IDENTITY -- every float32 in the .bin and in json["norm"] equals the
       corresponding (exported) tensor exactly (np.array_equal). Nothing was lost,
       transposed or reordered.
    2. FLOAT64 FORWARD -- numpy replays the 8 observations in float64 against a
       float64 copy of the torch actor: max abs error must be < tol (1e-6). This
       is the gated check; it isolates layout/order errors from arithmetic noise.
    3. FLOAT32 FORWARD -- the same replay in float32 against the float32 torch
       actor. Reported, not gated: torch's blocked GEMM and numpy's BLAS sum in
       different orders, so a few ULPs of the O(1) outputs is the floor, not a
       defect. This is the number the JS runtime (tolerance 1e-5) will see.
    """
    meta, layers, norm, fwd = numpy_load(name)

    idxs = sorted(int(k.split(".")[1]) for k in sd
                  if k.startswith("mlp.") and k.endswith(".weight"))
    for (w, b), i in zip(layers, idxs):
        ref_w = sd[f"mlp.{i}.weight"].to(torch.float32).numpy()
        ref_b = sd[f"mlp.{i}.bias"].to(torch.float32).numpy()
        assert np.array_equal(w, ref_w), f"{name}: mlp.{i}.weight differs from the checkpoint"
        assert np.array_equal(b, ref_b), f"{name}: mlp.{i}.bias differs from the checkpoint"
    assert np.array_equal(norm["mean"],
                          sd["obs_normalizer._mean"].reshape(-1).to(torch.float32).numpy()), \
        f"{name}: norm.mean differs from the checkpoint"
    assert np.array_equal(norm["std"],
                          sd["obs_normalizer._std"].reshape(-1).to(torch.float32).numpy()), \
        f"{name}: norm.std differs from the checkpoint"

    actor64 = TorchActor(sd).double()
    act64 = actor64(torch.from_numpy(obs).double()).numpy()
    err64 = float(np.abs(fwd(obs, np.float64) - act64).max())
    err32 = float(np.abs(fwd(obs, np.float32) - act_torch32).max())
    assert err64 < tol, f"{name}: bin/json does not reproduce torch ({err64:.3e} >= {tol:.0e})"
    return err64, err32


# --- verify-only: the shipped files, replayed without any checkpoint ---------

def verify_only(tol_js: float = 1e-5) -> int:
    """Replay tests/parity.json against the shipped .bin/.json. No torch, no .pt.

    This is the same arithmetic the browser has to do, so a pass here means the
    artifacts and the fixture agree and the JS runtime has a reachable target.
    It needs nothing but assets/policies/ and tests/parity.json, which is why it
    can run on a machine that has none of the checkpoints.
    """
    manifest = json.loads((POLICY_DIR / "manifest.json").read_text())
    fixture = json.loads((TESTS_DIR / "parity.json").read_text())
    cases = fixture["policies"]
    rows = manifest["policies"]

    names_m = [r["name"] for r in rows]
    missing_fix = [n for n in names_m if n not in cases]
    extra_fix = [n for n in cases if n not in names_m]
    if missing_fix or extra_fix:
        print(f"FIXTURE MISMATCH: no parity rows for {missing_fix}; "
              f"parity rows with no policy {extra_fix}", file=sys.stderr)
        return 2

    worst32 = worst64 = 0.0
    for r in rows:
        name = r["name"]
        meta, _, _, fwd = numpy_load(name)
        for field in ("obs_dim", "act_dim", "rotation_baked", "bin_bytes"):
            assert meta[field] == r[field], f"{name}: manifest/json disagree on {field}"
        assert (POLICY_DIR / meta["bin"]).stat().st_size == meta["bin_bytes"], \
            f"{name}: .bin size does not match the json"
        c = cases[name]
        assert c["obs_dim"] == meta["obs_dim"] and c["act_dim"] == meta["act_dim"], \
            f"{name}: parity fixture dims disagree with the json"
        assert c["rotation_baked"] == meta["rotation_baked"], \
            f"{name}: parity fixture rotation flag disagrees with the json"
        obs = np.asarray(c["obs"], dtype=np.float32)
        ref = np.asarray(c["act"], dtype=np.float32)
        assert obs.shape == (8, meta["obs_dim"]) and ref.shape == (8, meta["act_dim"]), \
            f"{name}: parity fixture should be 8 x {meta['obs_dim']} -> 8 x {meta['act_dim']}"
        e32 = float(np.abs(fwd(obs, np.float32) - ref).max())
        e64 = float(np.abs(fwd(obs, np.float64) - ref).max())
        worst32, worst64 = max(worst32, e32), max(worst64, e64)
        flag = "OK " if e32 < tol_js else "FAIL"
        print(f"  {flag} {name:32s} float32 {e32:.3e}   float64 {e64:.3e}")
        if e32 >= tol_js:
            return 1
    print(f"\n{len(rows)} policies replayed from the shipped .bin/.json alone.")
    print(f"WORST |numpy(bin) - parity.act| = {worst32:.3e} float32, {worst64:.3e} "
          f"float64  (budget {tol_js:.0e})")
    if not manifest.get("complete", False):
        print("NOTE: manifest.complete is false -- this is a PARTIAL export.")
        return 1
    return 0


# --- main --------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tol", type=float, default=1e-6,
                    help="max |numpy(bin) - torch| allowed (default 1e-6)")
    ap.add_argument("--only", action="append", default=None, metavar="NAME",
                    help="export only these policies (repeatable). Rewrites their "
                         ".bin/.json; leaves manifest.json and tests/parity.json "
                         "alone, since those describe the whole roster.")
    ap.add_argument("--skip-missing", action="store_true",
                    help="export what is on disk and list the rest as missing "
                         "instead of failing (the manifest is then PARTIAL)")
    ap.add_argument("--verify-only", action="store_true",
                    help="export nothing: replay tests/parity.json against the "
                         "shipped .bin/.json in numpy. Needs no checkpoint.")
    args = ap.parse_args()

    if args.verify_only:
        return verify_only()

    POLICY_DIR.mkdir(parents=True, exist_ok=True)
    TESTS_DIR.mkdir(parents=True, exist_ok=True)

    policies = POLICIES if not args.only else [p for p in POLICIES if p["name"] in args.only]
    if args.only:
        unknown = set(args.only) - {p["name"] for p in POLICIES}
        if unknown:
            print(f"unknown policy name(s): {sorted(unknown)}", file=sys.stderr)
            return 2
        print("NOTE: --only set; the per-policy .bin/.json are rewritten but "
              "manifest.json and tests/parity.json are LEFT ALONE (they describe "
              "the whole roster). Re-run without --only to refresh them.",
              file=sys.stderr)

    # A source is usable only when it is there AND the whole file is there: a
    # .cache pull that is still streaming is a truncated zip, and torch.load
    # fails on it with a miniz "central directory" error several minutes in.
    def _why_unusable(p: dict) -> str | None:
        src = Path(p["source"])
        if not src.exists():
            return "not on disk"
        have, want = src.stat().st_size, p["expect_bytes"]
        if have != want:
            return (f"{have:,} of {want:,} B"
                    + (" -- still transferring?" if have < want else " -- OVERSIZE"))
        return None

    unusable = {p["name"]: _why_unusable(p) for p in policies}
    missing = [p for p in policies if unusable[p["name"]]]
    if missing:
        for p in missing:
            print(f"MISSING {p['name']}: {p['source']}  [{unusable[p['name']]}]",
                  file=sys.stderr)
            print(f"        upstream (5080): {p['upstream']}", file=sys.stderr)
        if not args.skip_missing:
            return 2
        print(f"WARNING: --skip-missing; {len(missing)} policies are NOT exported "
              f"and manifest.json is PARTIAL.", file=sys.stderr)
        policies = [p for p in policies if p not in missing]

    rows, parity, worst, worst32 = [], {}, 0.0, 0.0
    torch.set_grad_enabled(False)

    for spec in policies:
        name = spec["name"]
        src = Path(spec["source"])
        assert src.stat().st_size == spec["expect_bytes"], (
            f"{name}: {src} is {src.stat().st_size:,} B, the asset list says "
            f"{spec['expect_bytes']:,} B"
        )
        sd_native = load_actor(src, spec["ckpt_seat"])
        local_md5 = md5_of(src)
        extra: dict = {}

        print(f"\n== {name}   [{spec['game']}/{spec['seat'] or '-'}]  {spec['display']}")
        print(f"   upstream {spec['upstream']}")
        print(f"   read     {src}")
        print(f"   md5      {local_md5}"
              + ("  == upstream" if local_md5 == spec.get("upstream_md5") else ""))
        extra["upstream_md5_verified"] = local_md5 == spec.get("upstream_md5")

        info = splice_infos(src) if spec["ckpt_seat"] else None
        if info:
            want = info[f"{spec['ckpt_seat']}_checkpoint"]
            assert want == spec["upstream"], (
                f"{name}: the splice names {want}, the asset list says {spec['upstream']}"
            )
            print(f"   splice   infos.eval_matchup.{spec['ckpt_seat']}_checkpoint "
                  f"MATCHES the asset list")
            extra["splice"] = info

        if spec["corroborate"] is not None:
            corr = check_splice_corroboration(sd_native, Path(spec["corroborate"]),
                                              spec["ckpt_seat"], spec["upstream"], name)
            extra["corroboration"] = corr
            print(f"   2nd copy {corr['file']} carries a BIT-IDENTICAL {spec['ckpt_seat']} "
                  f"seat (sha256 {corr['seat_sha256'][:16]}) and records the same source")

        shapes = layer_shapes(sd_native)
        print(f"   layers   {' -> '.join(str(a) for a, _ in shapes)} -> {shapes[-1][1]}  (elu)")

        # arena_pose at [42:46) proved from this checkpoint's own critic
        if spec["obs_dim"] == 60:
            zeros = check_obs_layout(src, spec["ckpt_seat"], name)
            extra["critic_zero_var_dims"] = zeros
            print(f"   layout   critic zero-variance dims {zeros} (the cmd term) "
                  f"=> arena_pose at [42:46)")

        # the seat claim, tested numerically instead of trusted
        if spec["native_half"]:
            cos_yaw = check_native_half(sd_native, spec["native_half"], name)
            extra["arena_cos_yaw"] = cos_yaw
            print(f"   half     native {spec['native_half']}, arena_pose mean "
                  f"cos(yaw) = {cos_yaw:+.5f}  (A faces +x, B faces -x)")

        # bake the pi-rotation when the half and the seat disagree
        rotate = bool(spec["native_half"]) and spec["native_half"] != spec["seat"]
        sd = sd_native
        if rotate:
            sd = pi_rotate_actor(sd_native, spec["obs_dim"])
            err, diff = check_rotation_equivalence(sd_native, sd, spec["obs_dim"])
            extra["rot_err"], extra["rot_diff"] = err, diff
            m0 = sd_native["obs_normalizer._mean"].reshape(-1)[ARENA_POSE_LO:ARENA_POSE_HI]
            m1 = sd["obs_normalizer._mean"].reshape(-1)[ARENA_POSE_LO:ARENA_POSE_HI]
            print(f"   pi-rot   BAKED {spec['native_half']} half -> seat {spec['seat']}")
            print(f"            mean[42:46] {[round(float(v), 6) for v in m0]}"
                  f" -> {[round(float(v), 6) for v in m1]}")
            shown = "0.0 (EXACT)" if err == 0.0 else f"{err:.3e}"
            print(f"            full-net max|rot(x) - orig(Rx)| = {shown}"
                  f"   (unrotated differs by {diff:.6f})")

        meta = export_policy(spec, sd, rotation_baked=rotate, extra=extra,
                             local_md5=local_md5)
        print(f"   wrote    {meta['bin']}  {meta['bin_bytes']:,} B  +  {name}.json")

        # the 8 fixed observations, and the EXPORTED actor's own torch answer
        seed = obs_seed(name)
        obs = make_obs(sd, spec["obs_dim"], seed=seed)
        act = TorchActor(sd)(torch.from_numpy(obs)).numpy().astype(np.float32)

        err64, err32 = verify_export(name, sd, obs, act, args.tol)
        worst, worst32 = max(worst, err64), max(worst32, err32)

        if rotate:
            # The strongest form of the rotation proof: the SHIPPED .bin, read
            # back with nothing but numpy, evaluated on a plain seat-B obs, must
            # equal the ORIGINAL unrotated torch actor evaluated on R*obs.
            _, _, _, fwd = numpy_load(name)
            robs = obs.copy()
            robs[:, ARENA_POSE_LO:ARENA_POSE_HI] = -robs[:, ARENA_POSE_LO:ARENA_POSE_HI]
            ref = TorchActor(sd_native).double()(torch.from_numpy(robs).double()).numpy()
            bake_err = float(np.abs(fwd(obs, np.float64) - ref).max())
            assert bake_err < args.tol, (
                f"{name}: the shipped bin does not equal the original actor on "
                f"R*obs ({bake_err:.3e})"
            )
            print(f"   pi-proof max|numpy(bin)(obs) - torch(original)(R*obs)| = "
                  f"{bake_err:.3e}  (gated < {args.tol:.0e})")
            extra["rot_bin_err"] = bake_err
            meta["rotation"]["shipped_bin_vs_original_on_Rx"] = bake_err
            (POLICY_DIR / f"{name}.json").write_text(
                json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        print("   verify   bin/json weights + normalizer are BIT-IDENTICAL to the "
              "exported tensors")
        print(f"   parity   max|numpy(bin/json) - torch| = {err64:.3e} (float64, gated "
              f"< {args.tol:.0e})  |  {err32:.3e} (float32, informational)")

        parity[name] = {
            "seed": seed,
            "obs_dim": spec["obs_dim"],
            "act_dim": spec["act_dim"],
            "rotation_baked": rotate,
            "obs": [[float(v) for v in row] for row in obs],
            "act": [[float(v) for v in row] for row in act],
        }
        rows.append({
            "name": name,
            "display": spec["display"],
            "method": spec["method"],
            "role": spec["role"],
            "game": spec["game"],
            "seat": spec["seat"],
            "bin": meta["bin"],
            "json": f"{name}.json",
            "obs_dim": meta["obs_dim"],
            "act_dim": meta["act_dim"],
            "source_checkpoint": meta["source_checkpoint"],
            "md5": meta["md5"],
            "upstream_checkpoint_md5": meta["upstream_checkpoint_md5"],
            "upstream_md5_verified": meta["upstream_md5_verified"],
            "provenance": meta["provenance"],
            "native_half": meta["native_half"],
            "rotation_baked": meta["rotation_baked"],
            "action_path": meta["action_path"],
            "bin_bytes": meta["bin_bytes"],
        })

    by_name = {r["name"]: r for r in rows}
    games: dict = {}
    for r in rows:
        if r["game"] == "walk":
            continue
        games.setdefault(r["game"], {"seats": {}})
        games[r["game"]]["seats"].setdefault(r["seat"], []).append(r["name"])

    manifest = {
        "schema": 2,
        "exporter": "tools/export_policy.py",
        "authority": "DESIGN.md section 3 (asset list); recon/01 asym, recon/02 sym",
        "complete": not (args.only or args.skip_missing),
        "norm_placement": "json (norm.mean / norm.std / norm.eps), not folded",
        "activation": "elu (alpha 1.0) between layers; the head is linear",
        "bin_layout": "little-endian float32; per Linear: weight (row-major, out x in) then bias",
        "norm_eps": NORM_EPS,
        "action_scale": ACTION_SCALE,
        "default_joint_pos": DEFAULT_JOINT_POS,
        "joint_order": JOINT_ORDER,
        "control_hz": 50.0,
        "increment_integrator": dict(INCREMENT),
        "display_names": DISPLAY,
        "_howto": "games.<game>.seats.<seat> lists policy NAMES; look each one up "
                  "in the flat `policies` array (or load assets/policies/<name>.json). "
                  "Never hard-code a checkpoint path -- every one is in here.",
        "seating": {
            "sym": "the human takes seat A (scores at x = +1.9); the AI takes "
                   "seat B (scores at x = -1.9). Every sym file below is the "
                   "seat-B build; A-half members carry the pi-rotation baked in.",
            "asym": "the human picks attacker or defender; the AI takes the other "
                    "seat. No rotation on this side -- the roles are fixed.",
        },
        "player_walk": by_name.get("player_walk_fastwalk_v3_9000"),
        "games": games,
        "policies": rows,
    }
    if missing and args.skip_missing:
        manifest["missing"] = [
            {"name": p["name"], "game": p["game"], "seat": p["seat"],
             "method": p["method"], "display": p["display"],
             "source_checkpoint": p["upstream"],
             "expected_local_copy": str(p["source"]),
             "expect_bytes": p["expect_bytes"],
             "reason": unusable[p["name"]]}
            for p in missing
        ]
    if args.only:
        print("\n--only: manifest.json and tests/parity.json NOT rewritten.")
        print(f"exported {len(rows)} policy file(s): "
              + ", ".join(r["name"] for r in rows))
        return 0

    (POLICY_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                              encoding="utf-8")

    fixture = {
        "_readme": (
            "Fixed observations and the exact float32 outputs of the torch actor "
            "AS EXPORTED (rsl_rl MLPModel: (x - mean)/(std + 1e-2), then "
            "Linear/ELU x3, bare Linear head, deterministic mean; for a policy "
            "with rotation_baked = true these are the ROTATED weights, i.e. what "
            "<name>.bin holds). The JS runtime must reproduce act to < 1e-5. obs "
            "were drawn as mean + std*z from each policy's own normalizer stats "
            "with numpy default_rng(seed), seed = int(sha256(name)[:8], 16); rows "
            "0-3 use z ~ N(0,1), rows 4-7 z ~ N(0,3)."
        ),
        "tolerance": 1e-5,
        "policies": parity,
    }
    (TESTS_DIR / "parity.json").write_text(json.dumps(fixture, indent=2) + "\n",
                                           encoding="utf-8")

    bin_total = sum(r["bin_bytes"] for r in rows)
    json_total = sum((POLICY_DIR / r["json"]).stat().st_size for r in rows)
    man_bytes = (POLICY_DIR / "manifest.json").stat().st_size
    print(f"\nwrote assets/policies/manifest.json ({len(rows)} policies, {man_bytes:,} B)")
    print(f"wrote tests/parity.json ({(TESTS_DIR / 'parity.json').stat().st_size:,} B)")
    print(f"WORST parity error over all policies: {worst:.3e} float64 (tol {args.tol:.0e})"
          f"  |  {worst32:.3e} float32 vs torch (JS budget 1e-5)")
    print(f"TOTAL exported: {bin_total:,} B of .bin + {json_total:,} B of .json "
          f"+ {man_bytes:,} B manifest = {bin_total + json_total + man_bytes:,} B")
    if missing and args.skip_missing:
        print(f"STILL MISSING ({len(missing)}): "
              + ", ".join(p["name"] for p in missing))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

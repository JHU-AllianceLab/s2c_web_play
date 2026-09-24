#!/usr/bin/env python3
"""Record GROUND TRUTH for app/obs.js and app/action.js out of the live mjlab envs.

The point of this file is that `tests/node_policy_parity.mjs` must not be able
to pass by agreeing with my own reading of the code. It replays a fixed action
sequence through the three REAL training environments

    Game-Touchdown-Go2-Go2-WBC-Clean        (asym, 60-D actor x 2 seats)
    Game-SymTouchdown-Go2-Go2-WBC-Clean     (sym,  60-D actor x 2 seats)
    Unitree-Go2-Flat-Fast                   (walk, 47-D actor)

and writes, per control step:

  * the INPUTS an observation builder is allowed to see -- exactly the entity
    quantities `app/physics.js` exposes: root_link_pos_w, root_link_quat_w
    (w-first), root_link_lin_vel_w, root_link_ang_vel_b, projected_gravity_b,
    joint_pos, joint_vel -- plus episode_length_buf, the env origin and, for the
    walk env, the twist command the observation actually saw;
  * the OUTPUT the ObservationManager produced (the 60-D / 47-D actor vector);
  * for the game envs, each seat's raw action and the increment integrator's
    resulting persistent target (= the ctrl written to the position actuators),
    together with the soft joint limits it clamps against.

node then rebuilds the observation with `app/obs.js` from the inputs alone and
diffs it against the recorded output, and replays the action sequence through
`app/action.js` and diffs the target. Nothing in the JS is consulted here and
nothing in here is consulted by the JS.

Run (about 40 s, CPU only, nothing is launched on any GPU box):

    cd /home/ray/Disk_ext/Go2/Meeting/web_game
    PYTHONPATH=/home/ray/Disk_ext/Go2/Project/unitree_rl_mjlab \
      /home/ray/Disk_ext/Go2/envs/mjlab_venv/bin/python tools/export_obs_fixture.py

Writes tests/obs_fixture.json (local ground truth; gitignored is fine, but it is
small enough to ship and the test skips gracefully when it is absent).
"""

from __future__ import annotations

import json
import math
import os
import sys
from importlib import import_module
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MJLAB_REPO = Path("/home/ray/Disk_ext/Go2/Project/unitree_rl_mjlab")

N_STEPS = 32
SEED = 20260924

# The walk command schedule: a few live commands, then a deliberate sub-0.1
# command so the observation's STAND GATE (phase -> (0,0),
# src/tasks/velocity/mdp/observations.py:74-75) is exercised by the real env
# rather than only by a unit test.
WALK_CMDS = [
    (1.4, 0.0, 0.0),
    (2.6, 0.3, -0.8),
    (-1.2, -0.7, 1.5),
    (0.02, 0.01, 0.0),   # |cmd| = 0.0224 < 0.1  -> phase must be (0, 0)
    (0.0, 0.0, 0.0),     # dead stop           -> phase must be (0, 0)
    (3.0, -1.0, 2.0),    # the corners of the trained box
]


def _np(t):
    return t.detach().cpu().numpy()


def _row(t, i=0):
    return [float(v) for v in _np(t)[i]]


def entity_state(entity):
    """Exactly the quantities app/physics.js `getBase`/`getJointPos`/... expose."""
    d = entity.data
    return {
        "pos": _row(d.root_link_pos_w),
        "quat": _row(d.root_link_quat_w),       # w-FIRST
        "linVelW": _row(d.root_link_lin_vel_w),
        "angVelB": _row(d.root_link_ang_vel_b),
        "projGrav": _row(d.projected_gravity_b),
        "jointPos": _row(d.joint_pos),
        "jointVel": _row(d.joint_vel),
    }


# --------------------------------------------------------------------------- #
# the two game envs
# --------------------------------------------------------------------------- #

GAME_SPECS = {
    "asym": {
        "task": "Game-Touchdown-Go2-Go2-WBC-Clean",
        "presets": ("src.tasks.game.config.go2_go2.presets", "PRESETS"),
        "make_env": "src.tasks.game.config.go2_go2.presets",
        "seats": {"a": "attacker", "b": "defender"},
    },
    "sym": {
        "task": "Game-SymTouchdown-Go2-Go2-WBC-Clean",
        "presets": ("src.tasks.sym_game.config.go2_go2.sym_preset", "SYM_PRESETS"),
        "make_env": "src.tasks.sym_game.config.go2_go2.presets",
        "seats": {"a": "attacker", "b": "defender"},
    },
}


def record_game(game: str) -> dict:
    import torch
    from mjlab.envs import ManagerBasedRlEnv

    spec = GAME_SPECS[game]
    pmod, pattr = spec["presets"]
    presets = getattr(import_module(pmod), pattr)
    make_env = getattr(import_module(spec["make_env"]), "_make_env")
    preset = next(p for p in presets if p.task_id == spec["task"])
    cfg = make_env(preset, play=True)
    cfg.scene.num_envs = 1
    scenario = preset.scenario()

    env = ManagerBasedRlEnv(cfg, device="cpu")
    torch.manual_seed(SEED)
    obs, _ = env.reset(seed=SEED)

    seats = spec["seats"]                       # robot id -> mjlab entity name
    ents = {r: env.scene[n] for r, n in seats.items()}
    terms = {r: env.action_manager.get_term(f"{n}_ctrl") for r, n in seats.items()}

    # --- the constants the JS side must get from scene.json / config.js -------
    soft = _np(ents["a"].data.soft_joint_pos_limits)[0]
    meta = {
        "game": game,
        "task": spec["task"],
        "lineX": float(getattr(scenario, "line_x")),
        "fieldCenter": [float(v) for v in scenario.field_center],
        "fieldSize": [float(v) for v in scenario.field_size],
        "envOrigin": _row(env.scene.env_origins),
        "episodeSteps": int(env.max_episode_length),
        "defaultJointPos": _row(ents["a"].data.default_joint_pos),
        "jointNames": list(ents["a"].joint_names),
        "softLimitLo": [float(v) for v in soft[:, 0]],
        "softLimitHi": [float(v) for v in soft[:, 1]],
        "increment": {
            "scale": float(terms["a"].cfg.scale),
            "incrementScale": float(terms["a"].cfg.increment_scale),
            "smoothing": float(terms["a"].cfg.action_smoothing),
            "delayMaxSteps": int(getattr(terms["a"].cfg, "action_delay_max_steps", 0)),
            "termClass": type(terms["a"]).__name__,
        },
        # The ONE signed thing that differs between the games. The sym scenario
        # owns `direction(role)` (sym_touchdown.py:104-110); the asym scenario
        # has no such method and both seats read the single +1.9 line, i.e. +1
        # (touchdown.py:286-301).
        "direction": {
            r: (
                float(scenario.direction(
                    scenario.attacker_role if seats[r] == "attacker"
                    else scenario.defender_role))
                if hasattr(scenario, "direction") else 1.0
            )
            for r in seats
        },
        "obsTermOrder": {
            g: list(env.observation_manager._group_obs_term_names[g])
            for g in env.observation_manager.active_terms
            if g.endswith("_actor")
        },
        "obsTermDims": {
            g: [int(d[0]) for d in env.observation_manager._group_obs_term_dim[g]]
            for g in env.observation_manager.active_terms
            if g.endswith("_actor")
        },
    }

    # The integrator seed rule: target == the MEASURED post-reset joint_pos.
    seed_check = {
        r: {
            "target": [float(v) for v in _np(terms[r]._target)[0]],
            "measuredJointPos": _row(ents[r].data.joint_pos),
        }
        for r in seats
    }

    rng = torch.Generator().manual_seed(SEED)
    steps = []

    def snapshot(action_by_seat):
        return {
            "episodeLengthBuf": int(env.episode_length_buf[0]),
            "robots": {r: entity_state(ents[r]) for r in seats},
            "actor": {
                r: [float(v) for v in _np(obs[f"{seats[r]}_actor"])[0]] for r in seats
            },
            "rawAction": {
                r: [float(v) for v in _np(terms[r].raw_action)[0]] for r in seats
            },
            "target": {
                r: [float(v) for v in _np(terms[r]._target)[0]] for r in seats
            },
            "actionIn": action_by_seat,
        }

    steps.append(snapshot({r: [0.0] * 12 for r in seats}))

    for _ in range(N_STEPS):
        # Deterministic, vigorous, and well outside the tiny-increment regime so
        # the integrator's saturation branch and its joint-limit clamp both fire.
        act = (torch.rand((1, 24), generator=rng) * 4.0 - 2.0)
        per_seat = {"a": [float(v) for v in act[0, :12]], "b": [float(v) for v in act[0, 12:]]}
        obs, _, term, trunc, _ = env.step(act)
        steps.append(snapshot(per_seat))
        if bool(term[0]) or bool(trunc[0]):
            # A reset re-seeds the integrator; the fixture stays a single
            # uninterrupted episode so the JS replay has no hidden state change.
            break

    env.close()
    return {"meta": meta, "seedCheck": seed_check, "steps": steps}


# --------------------------------------------------------------------------- #
# the walk env
# --------------------------------------------------------------------------- #


def record_walk() -> dict:
    import torch
    from mjlab.envs import ManagerBasedRlEnv

    mod = import_module("src.tasks.velocity.config.go2.env_cfgs")
    # play=False, because play=True silently halves the yaw command box
    # (env_cfgs.py:379-382, recon/04:273) and we want the box the walker was
    # actually trained on. The ONE play-mode override that matters here is
    # turned on by hand: the actor group's observation noise
    # (`cfg.observations["actor"].enable_corruption = False`, env_cfgs.py:124).
    # Without it the recorded 47-D vector carries U(+/-1.5) rad/s of joint-vel
    # noise and is not a function of the recorded state at all.
    cfg = mod.unitree_go2_flat_fast_env_cfg(play=False)
    cfg.scene.num_envs = 1
    cfg.observations["actor"].enable_corruption = False
    twist = cfg.commands["twist"]
    # Drive the command by hand, the way teleop_fastwalk_record.py:156-157 does
    # (otherwise the heading controller overwrites wz and Q/E "does nothing").
    # heading_command stays True (the cfg sets ranges.heading and the term
    # refuses the pair otherwise); rel_heading_envs = 0 is what actually
    # disables the overwrite, and it is exactly what teleop_fastwalk_record.py
    # :156-157 does.
    twist.rel_heading_envs = 0.0
    twist.rel_standing_envs = 0.0
    twist.resampling_time_range = (1.0e6, 1.0e6)

    env = ManagerBasedRlEnv(cfg, device="cpu")
    torch.manual_seed(SEED)
    obs, _ = env.reset(seed=SEED)

    ent = env.scene["robot"]
    term = env.action_manager.get_term("joint_pos")
    cmd_term = env.command_manager.get_term("twist")

    meta = {
        "task": "Unitree-Go2-Flat-Fast",
        "defaultJointPos": _row(ent.data.default_joint_pos),
        "jointNames": list(ent.joint_names),
        "actionScale": float(term.cfg.scale),
        "useDefaultOffset": bool(term.cfg.use_default_offset),
        "termClass": type(term).__name__,
        "obsTermOrder": list(env.observation_manager._group_obs_term_names["actor"]),
        "obsTermDims": [
            int(d[0]) for d in env.observation_manager._group_obs_term_dim["actor"]
        ],
        "enableCorruption": bool(cfg.observations["actor"].enable_corruption),
        "commandRanges": {
            "lin_vel_x": list(twist.ranges.lin_vel_x),
            "lin_vel_y": list(twist.ranges.lin_vel_y),
            "ang_vel_z": list(twist.ranges.ang_vel_z),
        },
    }

    rng = torch.Generator().manual_seed(SEED + 1)
    steps = []

    def snapshot(action_in):
        return {
            "episodeLengthBuf": int(env.episode_length_buf[0]),
            "robot": entity_state(ent),
            # The command the OBSERVATION saw, read back after the step.
            "cmd": _row(env.command_manager.get_command("twist")),
            "actor": [float(v) for v in _np(obs["actor"])[0]],
            "rawAction": [float(v) for v in _np(term.raw_action)[0]],
            "processed": [float(v) for v in _np(term._processed_actions)[0]],
            "actionIn": action_in,
        }

    steps.append(snapshot([0.0] * 12))

    for k in range(N_STEPS):
        vx, vy, wz = WALK_CMDS[k % len(WALK_CMDS)]
        cmd_term.vel_command_b[:, 0] = vx
        cmd_term.vel_command_b[:, 1] = vy
        cmd_term.vel_command_b[:, 2] = wz
        act = (torch.rand((1, 12), generator=rng) * 2.0 - 1.0)
        obs, _, t, tr, _ = env.step(act)
        steps.append(snapshot([float(v) for v in act[0]]))
        if bool(t[0]) or bool(tr[0]):
            break

    env.close()
    return {"meta": meta, "steps": steps}


# --------------------------------------------------------------------------- #


def main() -> int:
    sys.path.insert(0, str(MJLAB_REPO))
    os.chdir(MJLAB_REPO)  # go2_constants resolves the meshes relative to SRC_PATH

    out = {
        "_readme": (
            "Ground truth for app/obs.js and app/action.js, recorded from the LIVE "
            "mjlab training envs by tools/export_obs_fixture.py. `robots.<r>` holds "
            "exactly the entity quantities app/physics.js exposes; `actor` is what "
            "the env's ObservationManager produced from them; `target` is the v25 "
            "increment integrator's persistent target after applying `actionIn`. "
            "Step 0 is the post-reset frame (episodeLengthBuf 0, raw action zeros)."
        ),
        "generatedBy": "tools/export_obs_fixture.py",
        "nSteps": N_STEPS,
        "seed": SEED,
        "games": {},
    }
    for game in ("asym", "sym"):
        print(f"[fixture] {game} ...", flush=True)
        out["games"][game] = record_game(game)
    print("[fixture] walk ...", flush=True)
    out["walk"] = record_walk()

    dest = REPO / "tests" / "obs_fixture.json"
    dest.write_text(json.dumps(out))
    n = sum(len(out["games"][g]["steps"]) for g in out["games"]) + len(out["walk"]["steps"])
    print(f"[fixture] wrote {dest} ({dest.stat().st_size} B, {n} frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

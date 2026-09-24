// app/obs.js — the two observation vectors, assembled exactly as training does.
//
// =============================================================================
// THE 60-D GAME OBSERVATION  (identical layout in BOTH games)
// =============================================================================
//
// Built by the loop at src/tasks/game/game_env_cfg.py:313-387 (the sym fork's
// file is byte-identical here apart from a +18 line offset,
// src/tasks/sym_game/game_env_cfg.py:331-405), with the last three terms coming
// from the scenario/recipe. Measured live on a `ManagerBasedRlEnv`
// (recon/05:187-210): group dims {actor 60, critic 66} in exactly this order.
//
// `enable_corruption=False`, `history_length=1`, and EVERY term has
// `scale=None` and `clip=None` — no scaling, no clipping, no noise, no history.
//
//  #  slice  term                 dim  definition                                     source
//  1   0:3   base_ang_vel          3   root_link_ang_vel_b                            mjlab/envs/mdp/observations.py:31-35
//  2   3:6   projected_gravity     3   quat_apply_inverse(q, (0,0,-1)), UNIT vector   mjlab/envs/mdp/observations.py:38-43
//  3   6:18  joint_pos            12   q - q_default, JOINT order                     mjlab/envs/mdp/observations.py:51-61
//  4  18:30  joint_vel            12   qdot - 0  (default_joint_vel is zero)          mjlab/envs/mdp/observations.py:64-72
//  5  30:42  actions              12   this seat's RAW last action (pre-affine)       mjlab/envs/mdp/observations.py:80-83
//  6  42:46  arena_pose            4   (x, y, cos yaw, sin yaw), ENV-ORIGIN frame     src/tasks/game/mdp/observations.py:24-34
//  7  46:49  rel_pos_<opp>         3   quat_apply_inverse(q_me, p_opp - p_me)         src/tasks/game/mdp/observations.py:50-57
//  8  49:51  rel_yaw_<opp>         2   (cos, sin) of (yaw_opp - yaw_me)               src/tasks/game/mdp/observations.py:60-69
//  9  51:54  rel_vel_<opp>         3   quat_apply_inverse(q_me, v_opp_WORLD)          src/tasks/game/mdp/observations.py:72-84
// 10  54:55  <seat>_line           1   lineX - dir * x        (see below)             src/tasks/game/mdp/observations.py:37-47
//                                                             src/tasks/sym_game/mdp/sym.py:131-145
// 11  55:58  cmd                   3   CONSTANT ZEROS (mdp.zero_twist)                src/tasks/game/mdp/observations.py:275-286
//                                                             recipes/outcome_pbrs.py:590-599 (pilot_enabled False)
// 12  58:60  phase                 2   free-running trot clock, period 0.6 s          src/tasks/game/mdp/observations.py:289-310
//                                                             recipes/outcome_pbrs.py:596-599 / :641-644 (command_name=None)
//
// THE `line` TERM, and why one formula serves both games:
//   asym  `mdp.line_rel_x` returns `line_x - (x_env - field_center.x)` with
//         line_x = 1.7 (FIELD-local) and field_center = (0.2, 0), i.e. 1.9 - x.
//         BOTH seats get it with their OWN asset — the defender also reads the
//         distance to the single line at +1.9 (touchdown.py:286-301, "Both
//         roles see the signed distance to the line").  => dir = +1 for both.
//   sym   `sym.line_rel_x_signed` returns `line_x - direction * x_env` with
//         line_x = 1.9 and field_center = (0,0) (sym_touchdown.py:142-158).
//         direction: attacker/seat A = +1, defender/seat B = -1
//         (sym_touchdown.py:104-110).
//   Both collapse to `lineX - dir * x_env` with lineX = 1.9. That is the frozen
//   `gameObs(..., dir, lineX, ...)` signature, and `seatDirection()` below is
//   the single place the dir is decided.
//
// arena_pose is the ONE absolutely-framed term: it is why an A-half policy
// seated at B needs the pi-rotation. That rotation is BAKED INTO THE WEIGHTS at
// export time (see tools/export_policy.py and every <name>.json `rotation`
// block). This file therefore always emits a plain, un-rotated, seat-correct
// arena_pose. Never rotate here.
//
// =============================================================================
// THE 47-D WALK OBSERVATION  (the human's dog, Unitree-Go2-Flat-Fast)
// =============================================================================
//
// src/tasks/velocity/velocity_env_cfg.py:58-91 in dict order, with
// `height_scan` deleted for the flat task (src/tasks/velocity/config/go2/env_cfgs.py:161-162).
// The Flat-Fast fork keeps the layout untouched on purpose ("KEPT: the 47-D
// observation layout is untouched, INCLUDING the phase term on its stock fixed
// 0.6 s clock", env_cfgs.py:216-221).
//
//  dims  slice   term                source
//   3    0:3     base_ang_vel        builtin gyro at site `imu`, velocity_env_cfg.py:59-63.
//                                    The site sits on base_link with NO quat attribute
//                                    (go2.xml:52), so the gyro reads the body frame and
//                                    equals root_link_ang_vel_b (recon/04:157, :302).
//   3    3:6     projected_gravity   velocity_env_cfg.py:64-67
//   3    6:9     command             [vx, vy, wz] body frame — the WASD/QE input
//   2    9:11    phase               (sin, cos), period 0.6 s, ZEROED while standing
//                                    src/tasks/velocity/mdp/observations.py:49-76
//  12   11:23    joint_pos - default velocity_env_cfg.py:75-78
//  12   23:35    joint_vel           velocity_env_cfg.py:79-82
//  12   35:47    last raw action     velocity_env_cfg.py:83
//
// ⚠ TWO DIFFERENCES between the two phase clocks, both load-bearing:
//   * the WALK clock is GATED: `phase = (0,0)` while `|[vx,vy,wz]| < 0.1`
//     (observations.py:74-75 — the norm is over ALL THREE command dims).
//   * the GAME clock is FREE-RUNNING: `command_name=None`, so no gate
//     (observations.py:303-310; recipes/outcome_pbrs.py:596-599).
//
// OBSERVATION SCALES: every term is scale 1.0 in both tasks. There is no 0.25
// ang-vel scale and no 0.05 joint-vel scale — this is not an IsaacGym-lineage
// walker. Verified against the run's own params/env.yaml (recon/04:173) and by
// reading velocity_env_cfg.py:58-91 / game_env_cfg.py:313-387, where no term
// carries `scale=`. Raw SI units are correct.
//
// Training noise (`enable_corruption`) is OFF at play time for the walker
// (env_cfgs.py:124) and was never on for the game actor groups
// (game_env_cfg.py:377-387). Add nothing.

/** Control period: decimation 4 x timestep 0.005 (game_env_cfg.py:50-51). */
export const CONTROL_DT = 0.02;

/** Trot clock period, both tasks. game/recipes/outcome_pbrs.py:36 (_GAIT_PERIOD)
 *  and velocity_env_cfg.py:72-74. */
export const GAIT_PERIOD = 0.6;

/** The walk clock's stand gate: |command| below this zeros the phase.
 *  src/tasks/velocity/mdp/observations.py:74 */
export const WALK_STAND_EPS = 0.1;

export const WALK_OBS_DIM = 47;
export const GAME_OBS_DIM = 60;

/** Nominal stance, JOINT order FL,FR,RL,RR x (hip,thigh,calf).
 *  go2_constants.py:72-81 (INIT_STATE.joint_pos). Used only when neither the
 *  caller nor the sim supplies one. */
export const DEFAULT_JOINT_POS = Object.freeze([
  -0.1, 0.9, -1.8, 0.1, 0.9, -1.8, -0.1, 0.9, -1.8, 0.1, 0.9, -1.8,
]);

/** Byte-level map of the 47-D walk vector, for tests and the HUD. */
export const WALK_OBS_LAYOUT = Object.freeze([
  Object.freeze({ term: 'base_ang_vel', at: 0, dim: 3 }),
  Object.freeze({ term: 'projected_gravity', at: 3, dim: 3 }),
  Object.freeze({ term: 'command', at: 6, dim: 3 }),
  Object.freeze({ term: 'phase', at: 9, dim: 2 }),
  Object.freeze({ term: 'joint_pos', at: 11, dim: 12 }),
  Object.freeze({ term: 'joint_vel', at: 23, dim: 12 }),
  Object.freeze({ term: 'actions', at: 35, dim: 12 }),
]);

/** Byte-level map of the 60-D game vector, for tests and the HUD. */
export const GAME_OBS_LAYOUT = Object.freeze([
  Object.freeze({ term: 'base_ang_vel', at: 0, dim: 3 }),
  Object.freeze({ term: 'projected_gravity', at: 3, dim: 3 }),
  Object.freeze({ term: 'joint_pos', at: 6, dim: 12 }),
  Object.freeze({ term: 'joint_vel', at: 18, dim: 12 }),
  Object.freeze({ term: 'actions', at: 30, dim: 12 }),
  Object.freeze({ term: 'arena_pose', at: 42, dim: 4 }),
  Object.freeze({ term: 'rel_pos_opp', at: 46, dim: 3 }),
  Object.freeze({ term: 'rel_yaw_opp', at: 49, dim: 2 }),
  Object.freeze({ term: 'rel_vel_opp', at: 51, dim: 3 }),
  Object.freeze({ term: 'line', at: 54, dim: 1 }),
  Object.freeze({ term: 'cmd', at: 55, dim: 3 }),
  Object.freeze({ term: 'phase', at: 58, dim: 2 }),
]);

/** The pi-rotation slice, for assertions elsewhere. recon/02 section 1.5. */
export const ARENA_POSE_SLICE = Object.freeze([42, 46]);

// ---------------------------------------------------------------------------
// frame math — the same formulas mjlab uses, nothing invented
// ---------------------------------------------------------------------------

/**
 * quat_apply_inverse(q, v): rotate a world vector into the body frame.
 * mjlab/utils/lab_api/math.py:653-671 —
 *   t = 2 * (xyz x v);  out = v - q_w * t + xyz x t
 * @param {ArrayLike<number>} q  quaternion, W-FIRST (w, x, y, z)
 * @param {ArrayLike<number>} v  vector (x, y, z)
 * @param {Float64Array|number[]} [out]
 */
export function quatApplyInverse(q, v, out) {
  const w = q[0], x = q[1], y = q[2], z = q[3];
  const vx = v[0], vy = v[1], vz = v[2];
  // t = 2 * cross(xyz, v)
  const tx = 2 * (y * vz - z * vy);
  const ty = 2 * (z * vx - x * vz);
  const tz = 2 * (x * vy - y * vx);
  const o = out ?? new Float64Array(3);
  o[0] = vx - w * tx + (y * tz - z * ty);
  o[1] = vy - w * ty + (z * tx - x * tz);
  o[2] = vz - w * tz + (x * ty - y * tx);
  return o;
}

/**
 * The yaw of `euler_xyz_from_quat` (mjlab/utils/lab_api/math.py:460-463):
 *   yaw = atan2(2(w z + x y), 1 - 2(y^2 + z^2))
 * Algebraically identical to mjlab's `heading_w` (atan2 of the body +x axis in
 * world), which is what app/physics.js `yaw()` returns — the parity test asserts
 * the two agree.
 * @param {ArrayLike<number>} q  quaternion, W-FIRST
 */
export function yawFromQuat(q) {
  const w = q[0], x = q[1], y = q[2], z = q[3];
  return Math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z));
}

/**
 * The trot clock, `(sin, cos)` of `2*pi * ((n * dt) mod period) / period`.
 * src/tasks/game/mdp/observations.py:302-306 and
 * src/tasks/velocity/mdp/observations.py:68-73.
 *
 * `n` is `env.episode_length_buf`: the number of control steps COMPLETED since
 * the last reset. It is incremented before the terminations and the observation
 * (mjlab/envs/manager_based_rl_env.py:395-418), and zeroed by the reset, so the
 * first observation of an episode sees n = 0 and phase = (0, 1).
 *
 * FLOAT32: training evaluates this whole chain in float32 (a long tensor times a
 * python float promotes to the default dtype), so we emulate it with
 * Math.fround. It only matters at the wrap: in float64 `30*0.02 % 0.6` is
 * 1.1e-16 and sin is ~0, in float32 the product lands just BELOW 0.6 and sin is
 * -6e-7. Both are noise, but matching costs nothing and removes a question.
 *
 * @param {number} n        episode_length_buf
 * @param {boolean} standing  zero the clock (walk task only)
 * @param {number} period
 * @param {number} dt
 * @returns {[number, number]} [sin, cos]
 */
export function gaitPhase(n, standing = false, period = GAIT_PERIOD, dt = CONTROL_DT) {
  if (standing) return [0, 0];
  const f = Math.fround;
  const t = f(n * f(dt));
  const p = f(f(t % f(period)) / f(period));
  const ang = f(f(p * f(Math.PI)) * 2);
  return [f(Math.sin(ang)), f(Math.cos(ang))];
}

/**
 * The `direction` a seat attacks in FOR THE OBSERVATION's `line` term. Use this
 * function; do not take the number from anywhere else.
 *
 * ⚠ It is NOT the same as `scene.json.rules.directionA/directionB`, which is the
 * REFEREE's convention (which line a seat scores at) and reads -1 for the asym
 * defender. In the observation the asym defender reads its own distance to the
 * SAME +1.9 line the attacker is chasing, i.e. +1 — verified against the live
 * `Game-Touchdown-Go2-Go2-WBC-Clean` env in tests/node_policy_parity.mjs
 * section 5. Feeding -1 there silently gives the defender a mirrored world.
 *
 *   sym : attacker/A = +1, defender/B = -1   (sym_touchdown.py:104-110)
 *   asym: +1 for BOTH seats — there is one line, at env-local x = +1.9, and
 *         both roles observe their own distance to it (touchdown.py:286-301).
 * @param {'sym'|'asym'} game
 * @param {'A'|'B'|'attacker'|'defender'} seat
 */
export function seatDirection(game, seat) {
  if (game === 'sym') {
    if (seat === 'A' || seat === 'attacker') return 1;
    if (seat === 'B' || seat === 'defender') return -1;
    throw new Error(`obs.js: unknown sym seat "${seat}"`);
  }
  if (game === 'asym') {
    if (seat === 'attacker' || seat === 'defender' || seat === 'A' || seat === 'B') return 1;
    throw new Error(`obs.js: unknown asym seat "${seat}"`);
  }
  throw new Error(`obs.js: unknown game "${game}"`);
}

// ---------------------------------------------------------------------------
// internals
// ---------------------------------------------------------------------------

const ZERO_ORIGIN = Object.freeze([0, 0, 0]);

// Scratch, so a 50 Hz loop allocates nothing. Synchronous, single-threaded,
// and never held across a call.
const _delta = new Float64Array(3);
const _rot = new Float64Array(3);

function defaults(sim, opts) {
  return opts.defaultJointPos ?? sim.defaultJointPos ?? DEFAULT_JOINT_POS;
}

function checkLen(v, n, what) {
  if (!v || v.length !== n) {
    throw new Error(`obs.js: ${what} must have ${n} entries, got ${v ? v.length : v}`);
  }
}

// ---------------------------------------------------------------------------
// the two builders
// ---------------------------------------------------------------------------

/**
 * The 47-D observation for the human's walk policy.
 *
 * @param {object} sim            app/physics.js Sim
 * @param {'a'|'b'} robot
 * @param {ArrayLike<number>} cmd3  [vx, vy, wz], body frame, already clamped to
 *                                  the trained box vx[-1.5,3.0] vy[-1,1] wz[-2,2]
 *                                  (CMD_BOX in app/config.js; recon/04:271)
 * @param {number} stepCount        episode_length_buf (0 on the first obs)
 * @param {ArrayLike<number>} lastAction12  the walk policy's own previous RAW
 *                                  output (zeros right after a reset)
 * @param {object} [opts]
 * @param {ArrayLike<number>} [opts.defaultJointPos]
 * @param {Float32Array} [opts.out]  reuse a destination buffer
 * @returns {Float32Array} length 47
 */
export function walkObs(sim, robot, cmd3, stepCount, lastAction12, opts = {}) {
  checkLen(cmd3, 3, 'cmd3');
  checkLen(lastAction12, 12, 'lastAction12');
  const out = opts.out ?? new Float32Array(WALK_OBS_DIM);
  if (out.length !== WALK_OBS_DIM) throw new Error('obs.js: out must be 47 long');
  const dflt = defaults(sim, opts);

  const base = sim.getBase(robot);
  const g = sim.projectedGravity(robot);
  const q = sim.getJointPos(robot);
  const qd = sim.getJointVel(robot);

  // 0:3 base_ang_vel — the gyro at `imu`, which is root_link_ang_vel_b here.
  out[0] = base.angVelB[0];
  out[1] = base.angVelB[1];
  out[2] = base.angVelB[2];
  // 3:6 projected_gravity (unit)
  out[3] = g[0];
  out[4] = g[1];
  out[5] = g[2];
  // 6:9 command
  out[6] = cmd3[0];
  out[7] = cmd3[1];
  out[8] = cmd3[2];
  // 9:11 phase, gated on the command norm over ALL THREE dims
  // torch.linalg.norm, i.e. a plain root-sum-of-squares (observations.py:74).
  const cn = cmd3[0] * cmd3[0] + cmd3[1] * cmd3[1] + cmd3[2] * cmd3[2];
  const standing = Math.sqrt(cn) < WALK_STAND_EPS;
  const ph = gaitPhase(stepCount, standing);
  out[9] = ph[0];
  out[10] = ph[1];
  // 11:23 joint_pos - default | 23:35 joint_vel | 35:47 last raw action
  for (let i = 0; i < 12; i++) {
    out[11 + i] = q[i] - dflt[i];
    out[23 + i] = qd[i];
    out[35 + i] = lastAction12[i];
  }
  return out;
}

/**
 * The 60-D observation for a game policy. Same layout for sym and asym; only
 * `dir` and the resulting `line` differ.
 *
 * @param {object} sim            app/physics.js Sim
 * @param {'a'|'b'} me            the seat this observation is FOR
 * @param {'a'|'b'} opp           the other robot
 * @param {1|-1} dir              this seat's attack direction (seatDirection())
 * @param {number} lineX          1.9 in both games (GAMES.*.lineX)
 * @param {number} stepCount      episode_length_buf (0 on the first obs)
 * @param {ArrayLike<number>} lastAction12  THIS seat's previous RAW policy
 *                                output (zeros right after a reset)
 * @param {object} [opts]
 * @param {ArrayLike<number>} [opts.defaultJointPos]
 * @param {ArrayLike<number>} [opts.envOrigin]  default (0,0,0): the exported
 *                                scene is a single env at the world origin, so
 *                                world coordinates ARE env-local ones.
 * @param {Float32Array} [opts.out]
 * @returns {Float32Array} length 60
 */
export function gameObs(sim, me, opp, dir, lineX, stepCount, lastAction12, opts = {}) {
  checkLen(lastAction12, 12, 'lastAction12');
  if (dir !== 1 && dir !== -1) throw new Error(`obs.js: dir must be +1 or -1, got ${dir}`);
  const out = opts.out ?? new Float32Array(GAME_OBS_DIM);
  if (out.length !== GAME_OBS_DIM) throw new Error('obs.js: out must be 60 long');
  const dflt = defaults(sim, opts);
  const org = opts.envOrigin ?? ZERO_ORIGIN;

  const A = sim.getBase(me);
  const B = sim.getBase(opp);
  const g = sim.projectedGravity(me);
  const q = sim.getJointPos(me);
  const qd = sim.getJointVel(me);

  // 0:3 base_ang_vel | 3:6 projected_gravity
  out[0] = A.angVelB[0];
  out[1] = A.angVelB[1];
  out[2] = A.angVelB[2];
  out[3] = g[0];
  out[4] = g[1];
  out[5] = g[2];
  // 6:18 joint_pos - default | 18:30 joint_vel | 30:42 last raw action
  for (let i = 0; i < 12; i++) {
    out[6 + i] = q[i] - dflt[i];
    out[18 + i] = qd[i];
    out[30 + i] = lastAction12[i];
  }

  // 42:46 arena_pose — (x, y, cos yaw, sin yaw) in the ENV-ORIGIN frame.
  // NOTE: the field centre is NOT subtracted here (root_pose_2d only removes
  // env_origins), so in the asym game x is the raw env x, not x - 0.2.
  const ex = A.pos[0] - org[0];
  const ey = A.pos[1] - org[1];
  const yawMe = yawFromQuat(A.quat);
  out[42] = ex;
  out[43] = ey;
  out[44] = Math.cos(yawMe);
  out[45] = Math.sin(yawMe);

  // 46:49 rel_pos — the opponent's root position in MY body frame (3-D, z kept)
  _delta[0] = B.pos[0] - A.pos[0];
  _delta[1] = B.pos[1] - A.pos[1];
  _delta[2] = B.pos[2] - A.pos[2];
  const d = quatApplyInverse(A.quat, _delta, _rot);
  out[46] = d[0];
  out[47] = d[1];
  out[48] = d[2];

  // 49:51 rel_yaw — (cos, sin) of (yaw_opp - yaw_me)
  const dy = yawFromQuat(B.quat) - yawMe;
  out[49] = Math.cos(dy);
  out[50] = Math.sin(dy);

  // 51:54 rel_vel — the opponent's WORLD linear velocity in MY body frame.
  // (Not a relative velocity: rel_lin_vel_b rotates v_opp alone,
  //  src/tasks/game/mdp/observations.py:72-84.)
  const v = quatApplyInverse(A.quat, B.linVelW, _rot);
  out[51] = v[0];
  out[52] = v[1];
  out[53] = v[2];

  // 54 line — lineX - dir * x_env. See the header for why one formula serves
  // both games.
  out[54] = lineX - dir * ex;

  // 55:58 cmd — constant zeros in every shipped arm (mdp.zero_twist).
  out[55] = 0;
  out[56] = 0;
  out[57] = 0;

  // 58:60 phase — FREE-RUNNING here (command_name=None), no stand gate.
  const ph = gaitPhase(stepCount, false);
  out[58] = ph[0];
  out[59] = ph[1];

  return out;
}

export default {
  walkObs,
  gameObs,
  gaitPhase,
  quatApplyInverse,
  yawFromQuat,
  seatDirection,
  WALK_OBS_DIM,
  GAME_OBS_DIM,
  WALK_OBS_LAYOUT,
  GAME_OBS_LAYOUT,
  DEFAULT_JOINT_POS,
  CONTROL_DT,
  GAIT_PERIOD,
  WALK_STAND_EPS,
  ARENA_POSE_SLICE,
};

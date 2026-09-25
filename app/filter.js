/**
 * app/filter.js — the S2C safety certificate (Q-CBF), in the browser.
 *
 * This is the paper's whole claim, running: a frozen certificate sits between
 * whatever the policy (or the human) asks for and the plant, and edits the
 * request only as much as the learned value function says it must.
 *
 * =============================================================================
 * WHAT IT IS
 * =============================================================================
 *
 * Three frozen networks from `agent_15000` of the v5prox collision line
 * (`recon/03_safety_filter.md`), shipped as .bin/.json by tools/export_filter.py
 * and run through the SAME app/policy.js MLP the game policies use — the
 * activation is `sin` (`src/tasks/safety/rl/networks.py:42-46`), which that
 * runtime already supports, and there is NO observation normalizer
 * (`filtered_action.py:296-322`; the raw 62-D physical vector enters layer 0):
 *
 *     pi_shield(obs62)          -> u_safe in [-1,1]^12      filter_ctrl
 *     dstb(obs62, u)            -> d      in [-1,1]^6       filter_dstb
 *     Qhat(obs62, u)            = max( Q1(obs,u,d), Q2(obs,u,d) )    filter_q1/q2
 *                                 ("pessimistic = max", filtered_action.py:334)
 *
 * Per robot, per 50 Hz control step (`game_qcbf_action.py:274-293` and
 * `filtered_action.py:438-627`, quoted line by line at each site below):
 *
 *     q_des  = 0.25 * a + q_default            the SAME affine an unfiltered seat uses
 *     u_task = clamp((q_des - target)/0.15, -1, 1)          integrator inverse
 *     u_safe = pi_shield(obs);  V = Qhat(obs, u_safe);  thr = kappa * V
 *     q_task = Qhat(obs, u_task)
 *       q_task >= thr  ->  u_sel = u_task, alpha 0         TASK_PASS
 *       V      <  thr  ->  u_sel = u_safe, alpha 1         FALLBACK
 *       else           ->  u_sel = secant line search      LINE_SEARCH
 *     target = applyIncrement(u_sel)           the SAME v25 integrator
 *     prev_ctrl = u_sel                        -> obs[36:48] on the NEXT step
 *
 * =============================================================================
 * WHAT IS DELIBERATELY ABSENT
 * =============================================================================
 *
 * There is no RUN/GUARD/HANDBACK state machine. `GameQcbfShieldAction` builds
 * the shield with `sup_enabled=False, guard_enabled=False` (literals,
 * game_qcbf_action.py:181-183) and `kin_enabled = (filter.tilt_guard is not
 * None)` = False for this bundle, so `make_inmemory_shield` parks
 * `value_guard = -1e9` (exit_driver.py:700-703) and the `danger` test at
 * filtered_action.py:495 is identically False. recon/03:216-228 spells this out:
 * the mode machine never leaves RUN, the integrator reseed never fires, and
 * `guard_min_dwell / handback_ramp / handback_gate_*` are dead code. Porting
 * them would be porting dead branches.
 *
 * `fallback_line_search` is False (make_inmemory_shield default, never
 * overridden by the action term), so the out-of-set rows really do snap to
 * `u_safe` at alpha 1 — NOT to the `_best_effort` argmax a neighbouring lane
 * uses (recon/03:260-265).
 *
 * The solver is the SECANT LINE SEARCH. `intervention` is absent from this
 * bundle's deploy.yaml, so it defaults to "line_search"
 * (game_qcbf_action.py:118-119). The projected-gradient variant belongs to the
 * SYMMETRIC arm, which passes `intervention="projected_gradient"` explicitly
 * (`sym_preset.py:114-131`, and the `..._62d_game_pg` bundle copy); it needs
 * autograd through Qhat and this build only ships the asymmetric game.
 *
 * =============================================================================
 * THE 62-D OBSERVATION
 * =============================================================================
 *
 * `game_qcbf_action._assemble_obs` (lines 295-331) is the code of record:
 *
 *   idx    dim  content                                            source
 *   0:3    3    root_link_lin_vel_b     base linear velocity, BODY frame
 *   3      1    roll                    euler_xyz_from_quat, re-wrapped atan2(sin,cos)
 *   4      1    pitch                   same
 *   5:8    3    root_link_ang_vel_b     body-frame angular velocity
 *   8:20   12   joint_pos               ABSOLUTE radians (not q - q_default!)
 *   20:32  12   joint_vel
 *   32:36  4    foot contact flags      ContactSensor.data.found > 0, FL FR RL RR
 *   36:48  12   prev_ctrl               the FILTER's own last output u_sel
 *   48:52  4    wall margins            [hx-x, hx+x, hy-y, hy+y], clamped ABOVE at 1.5
 *   52:54  2    heading                 (cos yaw, sin yaw), L2-normalized
 *   54:62  8    opponent tail           opponent_kinematics, ego yaw frame
 *
 * Four traps, all of which silently give a plausible-looking wrong answer:
 *   1. obs[8:20] is ABSOLUTE joint position. The 60-D game obs uses
 *      `q - q_default`; this one does not (recon/03:546-548).
 *   2. obs[36:48] is the FILTER's output, not the policy's action
 *      (recon/03:543-545).
 *   3. The wall rectangle is the BUNDLE's 4.8 x 3.0 at the env origin, NOT the
 *      scenario's 5.2 x 3.0 at (0.2, 0). The asym preset leaves `field_size`
 *      None, so the certificate reads a first margin of `2.4 - x` and only goes
 *      negative in the last 0.4 m of the 0.9 m run-out. That is what it was
 *      trained on; reproduce it, do not fix it (recon/03:534-541).
 *   4. `d` is clamped only from ABOVE (3.0) and stays negative while
 *      penetrating; the margins are clamped only from above (1.5) and go
 *      negative outside the box (recon/03:552-554).
 *
 * The opponent tail (`safety_collision/mdp/observations.py:54-124`) is a
 * rect-rect distance between two COLLIDE_HALF (0.29, 0.193) hulls whose centres
 * sit 0.05 m ahead of base_link along body x (`collide.py:_planar_pose`), plus a
 * SPLIT HORIZON: past a raw `d >= 1.5` the seven direction/motion channels blank
 * to `[d, 0,0,0,0,0, 1, 0]` while `d` itself stays live out to 3.0.
 *
 * =============================================================================
 * GAIN BLEND — off by default, and that is the faithful choice
 * =============================================================================
 *
 * The filter can stiffen its seat from walk-soft 20/20/40 to 100/100/200 while
 * it intervenes (`touchdown_driver.py:42-44`, `gain_blend.py:63-64`). The
 * training env disables that at `num_envs == 1` (`game_qcbf_action.py:238`:
 * `self._gain_blend = cfg.gain_blend and self.num_envs > 1 and self._has_field`),
 * which is exactly the browser's case and the case every single-environment demo
 * render ran on. So `gainBlend` defaults to FALSE here and the plant stays at
 * 20/20/40. It is implemented and switchable (`{gainBlend: true}`) because the
 * C++ hardware lane does blend; turning it on changes the plant, so it is an
 * explicit opt-in, never a default.
 *
 * Citation root: /home/ray/Disk_ext/Go2/Project/unitree_rl_mjlab/
 */

import { loadPolicy, loadManifest } from './policy.js';
import { IncrementIntegrator, DEFAULT_JOINT_POS } from './action.js';

/** The certificate's observation width. deploy.yaml `obs_dim`. */
export const FILTER_OBS_DIM = 62;

/** Byte-level map of the 62-D vector, for tests and diagnostics.
 *  game_qcbf_action.py:295-331 */
export const FILTER_OBS_LAYOUT = Object.freeze([
  Object.freeze({ term: 'root_link_lin_vel_b', at: 0, dim: 3 }),
  Object.freeze({ term: 'roll', at: 3, dim: 1 }),
  Object.freeze({ term: 'pitch', at: 4, dim: 1 }),
  Object.freeze({ term: 'root_link_ang_vel_b', at: 5, dim: 3 }),
  Object.freeze({ term: 'joint_pos', at: 8, dim: 12 }),
  Object.freeze({ term: 'joint_vel', at: 20, dim: 12 }),
  Object.freeze({ term: 'foot_contact', at: 32, dim: 4 }),
  Object.freeze({ term: 'prev_ctrl', at: 36, dim: 12 }),
  Object.freeze({ term: 'wall_margins', at: 48, dim: 4 }),
  Object.freeze({ term: 'heading', at: 52, dim: 2 }),
  Object.freeze({ term: 'opponent_tail', at: 54, dim: 8 }),
]);

/**
 * The collision-hull constants. `src/tasks/safety_collision/mdp/collide.py`
 * lines 56, 57, 78, 83; duplicated (and asserted equal) in the bundle's
 * deploy.yaml `collision:` block.
 */
export const COLLIDE = Object.freeze({
  halfX: 0.29,          // collide.py:56  COLLIDE_HALF[0]
  halfY: 0.193,         // collide.py:56  COLLIDE_HALF[1]
  fwdOffset: 0.05,      // collide.py:57  COLLIDE_FWD_OFFSET
  dVis: 1.5,            // collide.py:78  D_VIS_COL  (direction horizon)
  dMax: 3.0,            // collide.py:83  D_MAX_COL  (distance clamp)
});

/**
 * Decision codes, mirroring filtered_action.py:65-70 so a JS trace and a python
 * log say the same thing. 3/4 (GUARD/HANDBACK) and 5 (projected gradient) can
 * never occur in this lane; they exist so the numbers line up.
 */
export const DECISION = Object.freeze({
  TASK_PASS: 0, FALLBACK: 1, LINE_SEARCH: 2, GUARD: 3, HANDBACK: 4, QP: 5,
});
export const DECISION_NAME = Object.freeze([
  'task_pass', 'fallback', 'line_search', 'guard', 'handback', 'projected_gradient',
]);

/**
 * PD gains, JOINT order. touchdown_driver.py:42-44 (`_WALK_KP`, `_STIFF_KP`,
 * `_KD`), identical to the bundle's walk_/safety_stiffness + damping.
 */
export const GAIN_TABLE = Object.freeze({
  walkKp: Object.freeze([20, 20, 40, 20, 20, 40, 20, 20, 40, 20, 20, 40]),
  safetyKp: Object.freeze([100, 100, 200, 100, 100, 200, 100, 100, 200, 100, 100, 200]),
  kd: Object.freeze([1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2]),
  /** 1/8 per control step in the ENV lane (GameQcbfShieldActionCfg.gain_rise_steps
   *  default 8, game_qcbf_action.py:382). The C++ lane reads 15 from the yaml. */
  risePerStep: 1 / 8,
  /** Hold the rise while all four feet are airborne (derive_gain_alpha). */
  contactGated: true,
});

const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);

// ===========================================================================
// 1. geometry — a line-for-line port of safety_collision/mdp/collide.py
// ===========================================================================

// Scratch. One filter step is synchronous and single-threaded, and nothing
// below is held across a call, so a 50 Hz loop allocates nothing here.
const _pa = new Float64Array(10);   // 5 points on A  (4 corners + centre), xy
const _pb = new Float64Array(10);
const _cpOnB = new Float64Array(10);
const _cpOnA = new Float64Array(10);
const _sdfA = new Float64Array(5);
const _sdfB = new Float64Array(5);

/** The 4 rectangle corners then the centre, written as 5 xy pairs into `out`.
 *  collide.py:164-178 (`_corners`), plus the centre row `torch.cat` at :218. */
function hullPoints(cx, cy, yaw, hx, hy, out) {
  const c = Math.cos(yaw), s = Math.sin(yaw);
  // signs, in collide.py's order: (+,+) (+,-) (-,+) (-,-)
  const sx = [1, 1, -1, -1], sy = [1, -1, 1, -1];
  for (let k = 0; k < 4; k++) {
    const lx = sx[k] * hx, ly = sy[k] * hy;
    out[2 * k] = cx + (lx * c - ly * s);
    out[2 * k + 1] = cy + (lx * s + ly * c);
  }
  out[8] = cx;
  out[9] = cy;
  return out;
}

/**
 * Signed distance of 5 world points to an oriented box, and the closest point
 * on the box surface. collide.py:181-206 (`_point_box`):
 *   q = |p_local| - half;  sdf = ||max(q,0)|| + min(max(q), 0)
 * Negative inside. The closest point clamps p_local to +/-half and maps back.
 */
function pointBox(pts, cx, cy, yaw, hx, hy, sdfOut, cpOut) {
  const c = Math.cos(yaw), s = Math.sin(yaw);
  for (let k = 0; k < 5; k++) {
    const rx = pts[2 * k] - cx, ry = pts[2 * k + 1] - cy;
    const lx = rx * c + ry * s;         // world -> body: R^T
    const ly = -rx * s + ry * c;
    const qx = Math.abs(lx) - hx, qy = Math.abs(ly) - hy;
    const ox = qx > 0 ? qx : 0, oy = qy > 0 ? qy : 0;
    const qmax = qx > qy ? qx : qy;
    sdfOut[k] = Math.hypot(ox, oy) + (qmax < 0 ? qmax : 0);
    const kx = clamp(lx, -hx, hx), ky = clamp(ly, -hy, hy);
    cpOut[2 * k] = cx + (kx * c - ky * s);   // body -> world
    cpOut[2 * k + 1] = cy + (kx * s + ky * c);
  }
}

/**
 * Rect-rect closest distance and the A->B unit normal. Pure geometry, exactly
 * `collide.rect_rect_distance` (collide.py:211-244): the min over 8 corner-vs-box
 * SDFs PLUS 2 centre-vs-box SDFs (the deep-overlap sign fix — all 8 corner SDFs
 * can read positive while the boxes interpenetrate), with the winning
 * candidate's closest-point direction as the normal and the centre line as the
 * degenerate fallback.
 *
 * `torch.min` returns the FIRST minimal index, so the scan below uses a strict
 * `<` and visits the A-points before the B-points, in collide.py's `torch.cat`
 * order.
 *
 * @returns {{d:number, nx:number, ny:number}}
 */
export function rectRectDistance(ax, ay, aYaw, bx, by, bYaw,
                                 hx = COLLIDE.halfX, hy = COLLIDE.halfY) {
  hullPoints(ax, ay, aYaw, hx, hy, _pa);
  hullPoints(bx, by, bYaw, hx, hy, _pb);
  pointBox(_pa, bx, by, bYaw, hx, hy, _sdfA, _cpOnB);   // A points vs box B
  pointBox(_pb, ax, ay, aYaw, hx, hy, _sdfB, _cpOnA);   // B points vs box A

  let d = Infinity, nx = 0, ny = 0;
  for (let k = 0; k < 5; k++) {
    if (_sdfA[k] < d) {                  // dir A->B: p -> cp_on_b
      d = _sdfA[k];
      nx = _cpOnB[2 * k] - _pa[2 * k];
      ny = _cpOnB[2 * k + 1] - _pa[2 * k + 1];
    }
  }
  for (let k = 0; k < 5; k++) {
    if (_sdfB[k] < d) {                  // dir A->B: cp_on_a -> p
      d = _sdfB[k];
      nx = _pb[2 * k] - _cpOnA[2 * k];
      ny = _pb[2 * k + 1] - _cpOnA[2 * k + 1];
    }
  }
  const nn = Math.hypot(nx, ny);
  if (nn > 1e-6) {
    const inv = 1 / Math.max(nn, 1e-6);
    return { d, nx: nx * inv, ny: ny * inv };
  }
  const lx = bx - ax, ly = by - ay;
  const ln = Math.max(Math.hypot(lx, ly), 1e-6);
  return { d, nx: lx / ln, ny: ly / ln };
}

/**
 * The hull centre of one robot: env-local xy shifted `COLLIDE_FWD_OFFSET` ahead
 * along body x. collide.py:145-160 (`_planar_pose`) — done there so `g_collide`
 * and the observation can never drift apart.
 */
export function hullCentre(x, y, yaw, out) {
  const o = out ?? new Float64Array(2);
  o[0] = x + COLLIDE.fwdOffset * Math.cos(yaw);
  o[1] = y + COLLIDE.fwdOffset * Math.sin(yaw);
  return o;
}

/**
 * The 8-D opponent tail, in the EGO yaw frame.
 * `safety_collision/mdp/observations.py:54-124` (`opponent_kinematics`).
 *
 * @param {number[]} ego  [x, y, yaw, vx, vy]  env-local pose + WORLD planar velocity
 * @param {number[]} opp  same, for the other robot
 * @param {Float32Array|Float64Array} [out]  length 8 (or a 62-D buffer + offset)
 * @param {number} [at] offset into `out`
 */
export function opponentTail(ego, opp, out, at = 0) {
  const o = out ?? new Float64Array(8);
  const ca = hullCentre(ego[0], ego[1], ego[2], _hullA);
  const cb = hullCentre(opp[0], opp[1], opp[2], _hullB);
  const { d, nx, ny } = rectRectDistance(ca[0], ca[1], ego[2], cb[0], cb[1], opp[2]);

  const vrx = opp[3] - ego[3];
  const vry = opp[4] - ego[4];
  const dClamped = d < COLLIDE.dMax ? d : COLLIDE.dMax;   // clamp(max=d_max)

  if (d < COLLIDE.dVis) {
    // Split horizon, near side: everything live. The world->ego rotation is
    // R(-yaw_ego), the same convention _point_box uses (observations.py:76-81).
    const c = Math.cos(ego[2]), s = Math.sin(ego[2]);
    o[at] = dClamped;
    o[at + 1] = nx * c + ny * s;
    o[at + 2] = -nx * s + ny * c;
    o[at + 3] = nx * vrx + ny * vry;      // d_dot = n_world . v_rel (frame-invariant)
    o[at + 4] = vrx * c + vry * s;
    o[at + 5] = -vrx * s + vry * c;
    const dpsi = opp[2] - ego[2];
    o[at + 6] = Math.cos(dpsi);
    o[at + 7] = Math.sin(dpsi);
  } else {
    // Far side: the DISTANCE stays live, the 7 direction/motion channels blank
    // (observations.py:104-124 — gated on the RAW d, pre-clamp).
    o[at] = dClamped;
    o[at + 1] = 0; o[at + 2] = 0; o[at + 3] = 0;
    o[at + 4] = 0; o[at + 5] = 0;
    o[at + 6] = 1; o[at + 7] = 0;
  }
  return o;
}

const _hullA = new Float64Array(2);
const _hullB = new Float64Array(2);

// ===========================================================================
// 2. the 62-D observation
// ===========================================================================

/**
 * roll and pitch exactly as `_assemble_obs` computes them:
 * `euler_xyz_from_quat` (mjlab/utils/lab_api/math.py:459-472, XYZ extrinsic)
 * followed by `atan2(sin, cos)` re-wrapping (game_qcbf_action.py:300-302).
 * @param {ArrayLike<number>} q  quaternion, W-FIRST
 * @returns {[number, number]} [roll, pitch]
 */
export function rollPitchFromQuat(q) {
  const w = q[0], x = q[1], y = q[2], z = q[3];
  let roll = Math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y));
  const sp = 2 * (w * y - z * x);
  let pitch = Math.abs(sp) >= 1 ? Math.sign(sp) * (Math.PI / 2) : Math.asin(sp);
  roll = Math.atan2(Math.sin(roll), Math.cos(roll));
  pitch = Math.atan2(Math.sin(pitch), Math.cos(pitch));
  return [roll, pitch];
}

/**
 * Build the certificate's 62-D observation for one robot.
 *
 * Deliberately a SEPARATE function from app/obs.js `gameObs`: the two vectors
 * share nothing but the joint order (60-D uses `q - q_default` and the absolute
 * arena pose; this one uses absolute joints, wall margins, and the filter's own
 * previous output). app/obs.js is untouched.
 *
 * @param {object} sim                app/physics.js Sim
 * @param {'a'|'b'} robot             the seat this observation is FOR
 * @param {'a'|'b'} opp               the other robot
 * @param {ArrayLike<number>} prevCtrl12  the filter's own last u_sel
 * @param {ArrayLike<number>} feet4   foot-ground contact flags, FL FR RL RR
 * @param {object} field              {halfX, halfY, cx, cy, dVis}
 * @param {Float32Array} [out]        length 62
 * @returns {Float32Array}
 */
export function filterObs(sim, robot, opp, prevCtrl12, feet4, field, out) {
  const o = out ?? new Float32Array(FILTER_OBS_DIM);
  if (o.length !== FILTER_OBS_DIM) throw new Error('filter.js: out must be 62 long');
  if (!prevCtrl12 || prevCtrl12.length !== 12) {
    throw new Error('filter.js: prevCtrl must have 12 entries');
  }
  if (!feet4 || feet4.length !== 4) throw new Error('filter.js: feet must have 4 entries');

  const A = sim.getBase(robot);
  const B = sim.getBase(opp);
  const q = sim.getJointPos(robot);
  const qd = sim.getJointVel(robot);

  // 0:3 root_link_lin_vel_b — the WORLD velocity rotated into the body frame.
  const w = A.quat[0], x = A.quat[1], y = A.quat[2], z = A.quat[3];
  {
    const vx = A.linVelW[0], vy = A.linVelW[1], vz = A.linVelW[2];
    const tx = 2 * (y * vz - z * vy);
    const ty = 2 * (z * vx - x * vz);
    const tz = 2 * (x * vy - y * vx);
    o[0] = vx - w * tx + (y * tz - z * ty);
    o[1] = vy - w * ty + (z * tx - x * tz);
    o[2] = vz - w * tz + (x * ty - y * tx);
  }
  // 3, 4 roll / pitch
  const rp = rollPitchFromQuat(A.quat);
  o[3] = rp[0];
  o[4] = rp[1];
  // 5:8 root_link_ang_vel_b
  o[5] = A.angVelB[0];
  o[6] = A.angVelB[1];
  o[7] = A.angVelB[2];
  // 8:20 joint_pos, ABSOLUTE | 20:32 joint_vel | 36:48 prev_ctrl
  for (let i = 0; i < 12; i++) {
    o[8 + i] = q[i];
    o[20 + i] = qd[i];
    o[36 + i] = prevCtrl12[i];
  }
  // 32:36 foot contact
  for (let i = 0; i < 4; i++) o[32 + i] = feet4[i] ? 1 : 0;

  // 48:52 wall margins [hx-x, hx+x, hy-y, hy+y], clamped ABOVE at d_vis only.
  const px = A.pos[0] - field.cx;
  const py = A.pos[1] - field.cy;
  const dv = field.dVis;
  o[48] = Math.min(field.halfX - px, dv);
  o[49] = Math.min(field.halfX + px, dv);
  o[50] = Math.min(field.halfY - py, dv);
  o[51] = Math.min(field.halfY + py, dv);

  // 52:54 heading (cos yaw, sin yaw), L2-normalized from the quaternion the same
  // way _assemble_obs does (game_qcbf_action.py:324-329).
  const siny = 2 * (w * z + x * y);
  const cosy = 1 - 2 * (y * y + z * z);
  const nrm = Math.max(Math.sqrt(siny * siny + cosy * cosy), 1e-8);
  o[52] = cosy / nrm;
  o[53] = siny / nrm;

  // 54:62 the opponent tail, in the ego yaw frame.
  const yawA = Math.atan2(siny, cosy);
  const yawB = Math.atan2(
    2 * (B.quat[0] * B.quat[3] + B.quat[1] * B.quat[2]),
    1 - 2 * (B.quat[2] * B.quat[2] + B.quat[3] * B.quat[3]),
  );
  _egoPose[0] = A.pos[0]; _egoPose[1] = A.pos[1]; _egoPose[2] = yawA;
  _egoPose[3] = A.linVelW[0]; _egoPose[4] = A.linVelW[1];
  _oppPose[0] = B.pos[0]; _oppPose[1] = B.pos[1]; _oppPose[2] = yawB;
  _oppPose[3] = B.linVelW[0]; _oppPose[4] = B.linVelW[1];
  opponentTail(_egoPose, _oppPose, o, 54);

  return o;
}

const _egoPose = new Float64Array(5);
const _oppPose = new Float64Array(5);

// ===========================================================================
// 3. foot-ground contact
// ===========================================================================

/**
 * The `found` channel of the per-player foot contact sensor
 * (`game_env_cfg.py:157-172`: primary = the 4 foot geoms, secondary = the
 * `terrain` body, `fields=("found", "force")`). `found > 0` means "a contact
 * between this foot and the ground exists on this frame" — which in MuJoCo is
 * exactly a `data.contact` entry pairing the foot geom with a terrain geom.
 *
 * One scan serves both seats and is cached for the frame, because reading
 * `data.contact` COPIES the whole contact vector onto the wasm heap (see
 * app/physics.js `anyContactBetweenRobots`); doing it twice per control step
 * would double that cost for nothing.
 */
export function createFootContacts(sim) {
  const geoms = sim.describeGeoms();
  const slot = new Int32Array(sim.ngeom).fill(-1);
  const isTerrain = new Uint8Array(sim.ngeom);
  const byName = new Map(geoms.map((g) => [g.name, g.geomId]));

  for (const g of geoms) {
    // Anything that belongs to neither robot and can collide is ground. In the
    // shipped scene that is the single `terrain` plane; the decoration is
    // contype/conaffinity 0 and never appears in data.contact at all.
    if (g.robot !== 'a' && g.robot !== 'b' && (g.contype | g.conaffinity) !== 0) {
      isTerrain[g.geomId] = 1;
    }
  }
  if (!isTerrain.some((v) => v)) {
    throw new Error('filter.js: the scene has no ground geom for the foot-contact channel');
  }

  const order = ['FL', 'FR', 'RL', 'RR'];   // recon/03:330-340 (model order)
  const seats = { a: 0, b: 4 };
  for (const [seat, base] of Object.entries(seats)) {
    const cfg = (sim.sceneJson && sim.sceneJson.robots && sim.sceneJson.robots[seat]) || {};
    const names = cfg.footGeoms;
    if (!Array.isArray(names) || names.length !== 4) {
      throw new Error(`filter.js: scene.json robots.${seat}.footGeoms must list 4 geoms`);
    }
    names.forEach((nm, i) => {
      if (!nm.includes(`${order[i]}_foot`)) {
        throw new Error(
          `filter.js: foot geom ${i} of seat ${seat} is "${nm}", expected ${order[i]} — ` +
          'the certificate reads the flags in model order FL, FR, RL, RR (recon/03:330-340)');
      }
      const gid = byName.get(nm);
      if (gid === undefined) throw new Error(`filter.js: no geom named "${nm}"`);
      slot[gid] = base + i;
    });
  }

  const flags = { a: new Float64Array(4), b: new Float64Array(4) };
  let stamp = NaN;
  let dirty = true;

  function scan() {
    flags.a.fill(0);
    flags.b.fill(0);
    const data = sim.data;
    const n = data.ncon;
    if (n === 0) return;
    const vec = data.contact;          // heap COPY — must be .delete()d
    try {
      for (let i = 0; i < n; i++) {
        const c = vec.get(i);
        if (c === undefined) continue;
        if (c.exclude === 0) {
          const g1 = c.geom1, g2 = c.geom2;
          let s = -1;
          if (slot[g1] >= 0 && isTerrain[g2]) s = slot[g1];
          else if (slot[g2] >= 0 && isTerrain[g1]) s = slot[g2];
          if (s >= 0) {
            if (s < 4) flags.a[s] = 1;
            else flags.b[s - 4] = 1;
          }
        }
        c.delete();
      }
    } finally {
      vec.delete();
    }
  }

  return {
    /** Foot flags for one seat on the CURRENT frame. */
    read(robot) {
      const t = sim.data.time;
      if (dirty || t !== stamp) {
        scan();
        stamp = t;
        dirty = false;
      }
      return flags[robot];
    },
    /** Force a rescan (the clock restarts at 0 on `sim.resetAll`). */
    invalidate() { dirty = true; },
  };
}

/** One scanner per sim, shared by both seats. */
const _footCache = new WeakMap();
export function footContactsFor(sim) {
  let fc = _footCache.get(sim);
  if (!fc) {
    fc = createFootContacts(sim);
    _footCache.set(sim, fc);
  }
  return fc;
}

// ===========================================================================
// 4. the certificate
// ===========================================================================

/**
 * Wrap the four loaded nets into the two quantities the filter law needs.
 * `pessimistic = "max"` (filtered_action.py:334), so `Qhat = max(Q1, Q2)`.
 */
export function createCertificate(nets) {
  const { ctrl, dstb, q1, q2 } = nets;
  if (ctrl.obsDim !== 62 || ctrl.actDim !== 12) throw new Error('filter.js: ctrl must be 62->12');
  if (dstb.obsDim !== 74 || dstb.actDim !== 6) throw new Error('filter.js: dstb must be 74->6');
  if (q1.obsDim !== 80 || q1.actDim !== 1) throw new Error('filter.js: q1 must be 80->1');
  if (q2.obsDim !== 80 || q2.actDim !== 1) throw new Error('filter.js: q2 must be 80->1');

  const xDstb = new Float32Array(74);
  const xQ = new Float32Array(80);
  const dOut = new Float32Array(6);
  const q1Out = new Float32Array(1);
  const q2Out = new Float32Array(1);
  const uSafe = new Float32Array(12);
  let evals = 0;

  return {
    /** u_safe = pi_shield(obs). REUSED buffer. */
    fallback(obs62) {
      return ctrl.forward(obs62, uSafe);
    },
    /** Qhat(obs, u) = max over the twin heads of Q(obs, u, dstb(obs, u)). */
    robustQ(obs62, u12) {
      xDstb.set(obs62, 0);
      for (let i = 0; i < 12; i++) xDstb[62 + i] = u12[i];
      dstb.forward(xDstb, dOut);
      xQ.set(obs62, 0);
      for (let i = 0; i < 12; i++) xQ[62 + i] = u12[i];
      xQ.set(dOut, 74);
      const a = q1.forward(xQ, q1Out)[0];
      const b = q2.forward(xQ, q2Out)[0];
      evals += 1;
      return a > b ? a : b;
    },
    /** How many robust_q evaluations since the last `resetCost()`. */
    cost() { return evals; },
    resetCost() { evals = 0; },
  };
}

/**
 * The filter law for ONE robot: the decision cascade plus the secant line
 * search. Pure numbers in, pure numbers out — no sim, no env.
 *
 * @param {object} cert   from `createCertificate`
 * @param {object} params {kappa, cbfMaxIters, cbfTol}
 */
export function createShield(cert, params) {
  const kappa = params.kappa;
  const maxIters = params.cbfMaxIters;
  const tol = params.cbfTol;

  const uUn = new Float64Array(12);
  const uSb = new Float64Array(12);
  const uNew = new Float64Array(12);
  const uRes = new Float64Array(12);
  const uSafeCopy = new Float64Array(12);
  const out = new Float64Array(12);

  /** alpha = the projection of u_filt onto the [u_task, u_safe] segment.
   *  filtered_action.py:832-840 (`_compute_alpha`). */
  function computeAlpha(uTask, uSafe, uFilt) {
    let nsq = 0, dot = 0;
    for (let i = 0; i < 12; i++) {
      const diff = uSafe[i] - uTask[i];
      nsq += diff * diff;
      dot += (uFilt[i] - uTask[i]) * diff;
    }
    if (nsq < 1e-12) return 0;
    return clamp(dot / Math.max(nsq, 1e-12), 0, 1);
  }

  /**
   * One decision.
   * @param {Float32Array} obs62
   * @param {ArrayLike<number>} uTask  the integrator-inverse increment, in [-1,1]^12
   * @returns {{u:Float64Array, alpha:number, decision:number, V:number,
   *            qTask:number, thr:number, iters:number, qEvals:number}}
   */
  function step(obs62, uTask) {
    cert.resetCost();
    const uSafe = cert.fallback(obs62);        // REUSED buffer, copy before reuse
    uSafeCopy.set(uSafe);
    const V = cert.robustQ(obs62, uSafeCopy);
    const thr = kappa * V;
    const qTask = cert.robustQ(obs62, uTask);

    // filtered_action.py:565-568 — the guard/handback machine is dead in this
    // lane (see the header), so the cascade is these three branches only.
    if (qTask >= thr) {
      for (let i = 0; i < 12; i++) out[i] = uTask[i];
      return { u: out, alpha: 0, decision: DECISION.TASK_PASS, V, qTask, thr,
        iters: 0, qEvals: cert.cost() };
    }
    if (V < thr) {
      // OUT OF SET (V < 0, since kappa < 1). fallback_line_search is False, so
      // this really is u_safe at alpha 1 (recon/03:260-265).
      out.set(uSafeCopy);
      return { u: out, alpha: 1, decision: DECISION.FALLBACK, V, qTask, thr,
        iters: 0, qEvals: cert.cost() };
    }

    // Secant on [u_task (infeasible), u_safe (feasible)].
    // filtered_action.py:629-675; the C++ twin is safety_filter.h:713-746.
    uUn.set(uTask); let qUn = qTask;
    uSb.set(uSafeCopy); let qSb = V;
    uRes.set(uSafeCopy);
    let done = false, iters = 0;
    for (let n = 0; n < maxIters; n++) {
      const den = qUn - qSb;
      if (Math.abs(den) < 1e-12) break;        // degenerate -> keep the safe bracket
      const t = (qUn - thr) / den;
      for (let i = 0; i < 12; i++) uNew[i] = uUn[i] + t * (uSb[i] - uUn[i]);
      const qNew = cert.robustQ(obs62, uNew);
      iters += 1;
      if (Math.abs(qNew - thr) <= tol) { uRes.set(uNew); done = true; break; }
      if (qNew < thr) { uUn.set(uNew); qUn = qNew; }
      else { uSb.set(uNew); qSb = qNew; }
    }
    if (!done) uRes.set(uSb);                  // exhausted / degenerate
    out.set(uRes);
    return {
      u: out,
      alpha: computeAlpha(uTask, uSafeCopy, uRes),
      decision: DECISION.LINE_SEARCH,
      V, qTask, thr, iters, qEvals: cert.cost(),
    };
  }

  return { step, computeAlpha, params: { kappa, cbfMaxIters: maxIters, cbfTol: tol } };
}

// ===========================================================================
// 5. loading
// ===========================================================================

/**
 * Load the certificate out of `assets/policies/manifest.json`. No filename and
 * no constant is spelled by the caller — DESIGN.md section 3.5.
 *
 * @param {object} [opts]
 * @param {string} [opts.manifestUrl]
 * @param {object} [opts.manifest]  pre-parsed manifest (skips the fetch)
 * @returns {Promise<{nets, params, field, collide, gains, entry}>}
 */
export async function loadFilter(opts = {}) {
  const manifestUrl = opts.manifestUrl || 'assets/policies/manifest.json';
  // Through app/policy.js `loadManifest`, so the node harnesses and the browser
  // read it the same way (it handles both fetch and node:fs).
  const manifest = opts.manifest ?? (await loadManifest(manifestUrl)).manifest;
  const entry = manifest.filter;
  if (!entry || entry.available !== true) {
    throw new Error('filter.js: the manifest carries no `filter` block — run tools/export_filter.py');
  }
  const dir = manifestUrl.slice(0, manifestUrl.lastIndexOf('/') + 1);
  const [ctrl, dstb, q1, q2] = await Promise.all(
    ['ctrl', 'dstb', 'q1', 'q2'].map((k) => loadPolicy(dir + entry.nets[k].json)),
  );

  // The manifest is the source of truth; these asserts only catch an exporter
  // and a runtime that have drifted apart.
  const p = entry.params;
  if (p.pessimistic !== 'max') throw new Error(`filter.js: pessimistic "${p.pessimistic}" is not max`);
  if (p.intervention !== 'line_search') {
    throw new Error(`filter.js: this build implements the line search, the bundle asks for "${p.intervention}"`);
  }
  if (p.full_takeover || p.fallback_line_search || p.kin_enabled || p.sup_enabled || p.guard_enabled) {
    throw new Error('filter.js: the bundle arms a latch this build does not implement');
  }
  const c = entry.collision;
  if (c.half_extents[0] !== COLLIDE.halfX || c.half_extents[1] !== COLLIDE.halfY
      || c.fwd_offset !== COLLIDE.fwdOffset || c.d_vis !== COLLIDE.dVis
      || c.d_max !== COLLIDE.dMax) {
    throw new Error('filter.js: the manifest collision block disagrees with COLLIDE');
  }

  return {
    entry,
    nets: { ctrl, dstb, q1, q2 },
    params: {
      kappa: p.kappa,
      cbfMaxIters: p.cbf_max_iters,
      cbfTol: p.cbf_tol,
      incScale: entry.increment.scale,
      incSmoothing: entry.increment.smoothing,
      qLo: entry.increment.q_lo,
      qHi: entry.increment.q_hi,
    },
    field: {
      halfX: entry.field.length / 2,
      halfY: entry.field.width / 2,
      cx: entry.field.center[0],
      cy: entry.field.center[1],
      dVis: entry.field.d_vis,
    },
    collide: entry.collision,
    gains: entry.gains,
  };
}

// ===========================================================================
// 6. the action path app/match.js plugs in
// ===========================================================================

/**
 * A shielded action path: `{kind, reset(measured12), step(a12, ctx), info()}`,
 * the shape app/match.js `actionPaths` expects.
 *
 * This is `GameQcbfShieldAction` (game_qcbf_action.py:274-293) with the same
 * pieces in the same order, and it serves BOTH seats — the human's walker and
 * the AI's game policy propose through the identical affine
 * (`scale=0.25, use_default_offset=True`), so one implementation covers both
 * (DESIGN.md section 11 rule 4).
 *
 * @param {object} o
 * @param {object} o.filter         from `loadFilter()`
 * @param {object} o.sim            app/physics.js Sim
 * @param {'a'|'b'} o.robot         the seat this path drives
 * @param {'a'|'b'} o.opponentRobot the other seat
 * @param {boolean} [o.gainBlend]   default FALSE — see the header
 * @param {object}  [o.footContacts] override the shared scanner (tests)
 */
export function makeShieldPath({
  filter, sim, robot, opponentRobot, gainBlend = false, footContacts = null,
}) {
  if (!filter || !filter.nets) throw new Error('makeShieldPath: pass the loadFilter() result');
  if (robot !== 'a' && robot !== 'b') throw new Error(`makeShieldPath: bad robot "${robot}"`);
  if (opponentRobot !== 'a' && opponentRobot !== 'b') {
    throw new Error(`makeShieldPath: bad opponentRobot "${opponentRobot}"`);
  }

  const cert = createCertificate(filter.nets);
  const shield = createShield(cert, filter.params);
  const feet = footContacts || footContactsFor(sim);
  // ONE integrator, shared by the proposal inverse and the v25 apply — the same
  // object the unfiltered seats use, so there is no second copy of that maths.
  const integ = new IncrementIntegrator(
    DEFAULT_JOINT_POS, filter.params.qLo, filter.params.qHi,
    { scale: 0.25, incrementScale: filter.params.incScale,
      smoothing: filter.params.incSmoothing },
  );

  const prevCtrl = new Float64Array(12);
  const obs = new Float32Array(FILTER_OBS_DIM);
  const uTask = new Float64Array(12);
  const kp = new Float64Array(12);
  const kd = Float64Array.from(GAIN_TABLE.kd);

  let gainAlpha = 0;
  let last = null;
  let stepMs = 0;

  function writeGains(a) {
    for (let i = 0; i < 12; i++) {
      kp[i] = (1 - a) * GAIN_TABLE.walkKp[i] + a * GAIN_TABLE.safetyKp[i];
    }
    sim.setGains(robot, kp, kd);
  }

  return {
    kind: 'qcbf_shield',
    filter: true,

    reset(measured12) {
      integ.reset(measured12);
      prevCtrl.fill(0);           // BatchedQcbfFilter.reset_ zeroes prev_ctrl
      gainAlpha = 0;
      last = null;
      feet.invalidate?.();
      if (gainBlend) writeGains(0);
    },

    step(a12) {
      const t0 = now();
      const f4 = feet.read(robot);
      filterObs(sim, robot, opponentRobot, prevCtrl, f4, filter.field, obs);
      integ.taskIncrement(a12, uTask);
      const r = shield.step(obs, uTask);
      const ctrl = integ.applyIncrement(r.u);
      prevCtrl.set(r.u);

      if (gainBlend) {
        // derive_gain_alpha (touchdown_driver.py:129-143): rate-limited,
        // contact-gated rise; the descent tracks alpha exactly.
        const feetCount = f4[0] + f4[1] + f4[2] + f4[3];
        const airborne = GAIN_TABLE.contactGated && feetCount < 0.5;
        gainAlpha = r.alpha >= gainAlpha
          ? (airborne ? gainAlpha : Math.min(r.alpha, gainAlpha + GAIN_TABLE.risePerStep))
          : r.alpha;
        writeGains(gainAlpha);
      }

      stepMs = now() - t0;
      last = {
        alpha: r.alpha,
        active: r.decision !== DECISION.TASK_PASS,
        intervening: r.decision !== DECISION.TASK_PASS,
        decision: r.decision,
        decisionName: DECISION_NAME[r.decision],
        value: r.V,
        qTask: r.qTask,
        thr: r.thr,
        iters: r.iters,
        qEvals: r.qEvals,
        gainAlpha,
        stepMs,
      };
      return ctrl;
    },

    info() { return last; },

    /** Diagnostics for the tests: the live 62-D vector and the integrator state. */
    debug() {
      return { obs, prevCtrl, target: integ.target, gainAlpha, gainBlend };
    },
  };
}

const now = (typeof performance !== 'undefined' && performance.now)
  ? () => performance.now()
  : () => Date.now();

export default {
  FILTER_OBS_DIM, FILTER_OBS_LAYOUT, COLLIDE, DECISION, DECISION_NAME, GAIN_TABLE,
  rectRectDistance, hullCentre, opponentTail, rollPitchFromQuat, filterObs,
  createFootContacts, footContactsFor, createCertificate, createShield,
  loadFilter, makeShieldPath,
};

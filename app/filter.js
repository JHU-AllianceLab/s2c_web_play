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
 *       else           ->  u_sel = projected gradient      QP  (see below)
 *     target = applyIncrement(u_sel)           the SAME v25 integrator
 *     prev_ctrl = u_sel                        -> obs[36:48] on the NEXT step
 *
 * =============================================================================
 * THE SOLVER IS THE PROJECTED GRADIENT, NOT THE SECANT
 * =============================================================================
 *
 * `intervention` is absent from this bundle's params/deploy.yaml, so the BUNDLE
 * default is "line_search" — but the bundle does not get the last word. The
 * action term takes the solver from its CFG when the cfg sets it and only falls
 * back to the yaml otherwise (game_qcbf_action.py:117-118), and both shipped
 * S2C seats were trained with that cfg set:
 *
 *   asym_s2c_attacker  logs/.../2026-08-17_22-23-27_v133fdrw/game_6000.pt
 *   asym_s2c_defender  logs/.../2026-08-18_11-55-46_v135fdrw/game_5000.pt
 *
 * whose run_config.yaml both carry, for BOTH seats,
 *
 *   filter: {attacker_ctrl, defender_ctrl}:
 *     ckpt_dir: .../safety_game/collision_v5prox_15k_62d_game
 *     intervention: projected_gradient
 *     pg_step: 0.2   pg_max_iters: 25   pg_backtrack_iters: 12
 *     travel_budget: 5.0
 *
 * (`travel_budget` is a DERIVED display field — runcfg.py:338 writes
 * `pg_step * pg_max_iters` into the view — and gates nothing.)
 *
 * The two solvers resolve the same rows and differ only in WHERE they land:
 * the secant is confined to the 1-D segment towards `u_safe`, the projected
 * gradient solves the actual problem, `argmin ||u - u_task||^2 s.t. Qhat >=
 * kappa V`, in the full 12-D box (filtered_action.py:169-212). Measured on the
 * six intervened fixture cases, the secant moves the action 1.22x to 3.03x
 * further than the gradient solve does for the same safety. That extra travel
 * is what a policy trained under the gradient solve does not expect, and it is
 * what toppled the AI in tests/fall_study.mjs.
 *
 * `_projected_gradient` (filtered_action.py:677-785, itself ported from the
 * vendored reference at safety_GP/filters/intervention.py:281-352) is four
 * steps, all reproduced verbatim below:
 *
 *   1. normalized ascent   u <- clamp(u + eta * g/||g||, -1, 1),  g = grad_u Qhat
 *      tested BEFORE each step, so the first test reads u_task (infeasible by
 *      construction on every row that gets here) and the loop exits the moment
 *      Qhat >= thr;
 *   2. one more test after the loop, because the loop steps after testing;
 *   3. `pg_backtrack_iters` BISECTIONS back along [u_task, u_feas] — bisection,
 *      not a secant, because Qhat is a sin-MLP and is not monotone along that
 *      segment, and because bisection always returns the FEASIBLE side;
 *   4. NON-CONVERGENCE: a row the ascent never made feasible returns `u_safe`
 *      at alpha 1 (exactly what the secant does when it exhausts its budget)
 *      and is counted in `pgInfeasible`, so a mistuned eta/N stays visible.
 *
 * THE GRADIENT. Python gets it from autograd through
 * `robust_q(obs, u) = critic(obs, u, dstb(obs, u))`, so what it returns is the
 * TOTAL derivative dQ/du + (dQ/dd)(d pi_d/du) — the adversary is conditioned on
 * u and is deliberately NOT detached (filtered_action.py:696-698). Here that is
 * a hand-written reverse pass through the two small MLPs (`makeTapedNet`):
 * backprop 1 through the twin head that WON the pessimistic max, take its
 * [62:74] slice for dQ/du and its [74:80] slice as the seed for backprop 2
 * through the adversary, take THAT [62:74] slice, add. One reverse pass costs
 * about one forward pass; finite differences would cost 12 forwards per
 * iteration and would not fit the 20 ms control step.
 *
 * The secant is kept (`intervention: 'line_search'`) because tests/filter_fixture.json
 * is a python trace of it and it is the only reference for the shared cascade.
 *
 * =============================================================================
 * WHAT IS DELIBERATELY ABSENT
 * =============================================================================
 *
 * There is no RUN/GUARD/HANDBACK state machine. `GameQcbfShieldAction` builds
 * the shield with `sup_enabled=False, guard_enabled=False` (literals,
 * game_qcbf_action.py:181-183) and `kin_enabled = (filter.tilt_guard is not
 * None)` = False for this bundle — its deploy.yaml `filter:` block has no
 * `tilt_guard` key, only the comment saying why (bundle params/deploy.yaml:166,
 * "Tilt/kin channel OFF"). So `make_inmemory_shield` parks
 * `value_guard = -1e9` (exit_driver.py:700-703) and the `danger` test at
 * filtered_action.py:494 is identically False. The yaml's own `value_eps:
 * -1.0e9` says the same thing from the deploy side. recon/03:216-228 spells it
 * out: the mode machine never leaves RUN, the integrator reseed never fires,
 * and `guard_min_dwell / handback_ramp / handback_gate_*` are dead code.
 * Porting them would be porting dead branches.
 *
 * `fallback_line_search` is False (make_inmemory_shield default, never
 * overridden by the action term), so the out-of-set rows really do snap to
 * `u_safe` at alpha 1 — NOT to the `_best_effort` argmax a neighbouring lane
 * uses (recon/03:260-265). `full_takeover` is False the same way, so the
 * `need_ls` rows really do reach a solver.
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
 * GAIN BLEND — ON by default, because that is the plant the policy learned on
 * =============================================================================
 *
 * The filter stiffens its seat from walk-soft 20/20/40 to 100/100/200 while it
 * intervenes (`touchdown_driver.py:42-44` `_WALK_KP`/`_STIFF_KP`/`_KD`,
 * `gain_blend.py:63-64` for the blend formula), through a rate-limited,
 * contact-gated `gain_alpha` (`derive_gain_alpha`, touchdown_driver.py:129-143)
 * that rises 1/8 per control step (`GameQcbfShieldActionCfg.gain_rise_steps = 8`,
 * game_qcbf_action.py:382 — the bundle yaml's `gain_rise_steps: 15` is the C++
 * lane, not this one) and holds while all four feet are airborne
 * (`gain_contact_gated = True`, game_qcbf_action.py:385).
 *
 * `game_qcbf_action.py:238` disables the blend when `num_envs == 1`:
 * `self._gain_blend = cfg.gain_blend and self.num_envs > 1 and self._has_field`.
 * The browser IS a single environment — but the browser is also the thing that
 * has to LOOK right, and the 8192-env fleet that produced these weights trained
 * with the blend ON for both seats (`cfg.gain_blend` defaults True,
 * game_qcbf_action.py:378; the DR-Wide arm leaves it alone, game_dr/__init__.py:392-400
 * — "本来filter都是那个kpkd，一直都是这个逻辑"). The `num_envs == 1` guard is a
 * MuJoCo tiling limitation, not a filter law: mjlab only tiles
 * `actuator_gainprm` per env when nworld > 1 (gain_blend.py:12-15), and this
 * runtime has no such limit. So we follow the MULTI-ENV behaviour and blend;
 * a soft-legged brace is not what the policy was trained to brace with.
 * `{gainBlend: false}` pins walk-soft for an A/B.
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
 * log say the same thing. 3/4 (GUARD/HANDBACK) can never occur in this lane;
 * they exist so the numbers line up. 5 (QP) is what an intervened step emits
 * under the projected gradient, 2 under the secant — filtered_action.py:598-603
 * picks between them by `intervention`, exactly so telemetry can tell which
 * mechanism produced the intervention.
 */
export const DECISION = Object.freeze({
  TASK_PASS: 0, FALLBACK: 1, LINE_SEARCH: 2, GUARD: 3, HANDBACK: 4, QP: 5,
});
export const DECISION_NAME = Object.freeze([
  'task_pass', 'fallback', 'line_search', 'guard', 'handback', 'projected_gradient',
]);

/** The two solvers `intervention` selects. filtered_action.py:72-74. */
export const INTERVENTION = Object.freeze({
  LINE_SEARCH: 'line_search',
  PROJECTED_GRADIENT: 'projected_gradient',
});

/**
 * Projected-gradient constants. NOT the QcbfParams dataclass defaults (which
 * happen to be the same three numbers, filtered_action.py:210-212) — these are
 * read off the run_config.yaml of the two checkpoints this build ships, where
 * the action cfg sets them per seat:
 *
 *   logs/rsl_rl/game_touchdown_go2_go2_wbc/2026-08-17_22-23-27_v133fdrw   (attacker)
 *   logs/rsl_rl/game_touchdown_go2_go2_wbc/2026-08-18_11-55-46_v135fdrw   (defender)
 *     filter.{attacker,defender}_ctrl:
 *       intervention: projected_gradient
 *       pg_step: 0.2   pg_max_iters: 25   pg_backtrack_iters: 12
 *
 * The one number that matters is the TRAVEL BUDGET `step * maxIters` = 5.0: the
 * ascent takes normalized steps, so it can move at most that far from u_task,
 * and a budget below the displacement the constraint demands degrades every row
 * to u_safe (filtered_action.py:192-209 has the 256-case measurement).
 */
export const PG = Object.freeze({
  step: 0.2,            // eta, the normalized ascent step
  maxIters: 25,         // N ascent steps before giving up on feasibility
  backtrackIters: 12,   // M bisections back towards u_task once feasible
  /** A gradient this small means ascent cannot help; freeze the row rather
   *  than divide by ~0. filtered_action.py:750-754. */
  gradEps: 1e-8,
});

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
 * A loaded app/policy.js MLP, re-run so the reverse pass can see the tape.
 *
 * The forward is the SAME arithmetic as `MlpPolicy.forward` — same four-way
 * float64 accumulation, same float32 store per layer, so `robustQ` below is
 * numerically what it has always been — with one addition: the PRE-activation
 * of every layer is kept, because `sin` is not invertible from its output and
 * the backward pass needs `cos(z)`.
 *
 * The backward is the ordinary reverse-mode chain, in float64 (a gradient is a
 * direction; nothing downstream rounds it to float32):
 *
 *     head:   dL/dz_last = dL/dy * (1 - y^2)   for the tanh heads (ctrl, dstb)
 *                        = dL/dy               for the bare linear twin heads
 *     layer:  dL/dx      = W^T dL/dz
 *     sin:    dL/dz_prev = dL/dx * cos(z_prev)
 *
 * It costs the same multiply-accumulates as the forward, which is the whole
 * reason `_projected_gradient` is affordable at 50 Hz: 12-dimensional finite
 * differences would be 12 forwards per ascent step, 25 ascent steps deep.
 *
 * Only the shipped architecture is accepted (sin between layers; tanh or linear
 * head) — an unknown activation must fail loudly, not silently differentiate
 * the wrong function.
 */
function makeTapedNet(net, label) {
  if (net.norm) {
    throw new Error(`filter.js: ${label} carries an observation normalizer; the ` +
      'certificate nets have none (filtered_action.py:296-322)');
  }
  const act = String(net.meta.activation ?? '').toLowerCase();
  if (act !== 'sin') {
    throw new Error(`filter.js: ${label} activation is "${act}", the reverse pass ` +
      'implements sin only (networks.py:42-46)');
  }
  const head = net.meta.head_activation == null
    ? null : String(net.meta.head_activation).toLowerCase();
  if (head !== null && head !== 'tanh') {
    throw new Error(`filter.js: ${label} head activation is "${head}", the reverse ` +
      'pass implements tanh and linear only');
  }

  const L = net.layers.map((l) => ({
    nIn: l.in, nOut: l.out, w: l.w, b: l.b,
    z: new Float32Array(l.out),   // pre-activation (the tape)
    a: new Float32Array(l.out),   // post-activation
  }));
  const last = L.length - 1;
  let wide = 0;
  for (const l of L) wide = Math.max(wide, l.nIn, l.nOut);
  // Two float64 scratch rows, ping-ponged down the stack.
  let gA = new Float64Array(wide);
  let gB = new Float64Array(wide);

  return {
    nIn: L[0].nIn,
    nOut: L[last].nOut,
    /** Forward, taping every pre-activation. Returns the REUSED output row. */
    forward(x) {
      let inp = x;
      for (let li = 0; li <= last; li++) {
        const { nIn, nOut, w, b, z, a } = L[li];
        const n4 = nIn - (nIn & 3);
        // TWO output rows at a time, so each `inp[c]` is loaded once for both.
        // Each row still accumulates in the SAME four-way order as
        // app/policy.js:310-318, so this is bit-identical to that kernel (the
        // fixture's 1e-5 parity budget is measured against it) and ~1.2x faster
        // — which matters because the ascent can run 25 of these per step.
        let r = 0, base = 0;
        for (; r + 1 < nOut; r += 2, base += 2 * nIn) {
          const base1 = base + nIn;
          let p0 = 0, p1 = 0, p2 = 0, p3 = 0;
          let q0 = 0, q1 = 0, q2 = 0, q3 = 0, c = 0;
          for (; c < n4; c += 4) {
            const x0 = inp[c], x1 = inp[c + 1], x2 = inp[c + 2], x3 = inp[c + 3];
            p0 += w[base + c] * x0;
            p1 += w[base + c + 1] * x1;
            p2 += w[base + c + 2] * x2;
            p3 += w[base + c + 3] * x3;
            q0 += w[base1 + c] * x0;
            q1 += w[base1 + c + 1] * x1;
            q2 += w[base1 + c + 2] * x2;
            q3 += w[base1 + c + 3] * x3;
          }
          let sp = p0 + p1 + p2 + p3;
          let sq = q0 + q1 + q2 + q3;
          for (; c < nIn; c++) { sp += w[base + c] * inp[c]; sq += w[base1 + c] * inp[c]; }
          z[r] = sp + b[r];         // float32 store, like policy.js
          z[r + 1] = sq + b[r + 1];
        }
        for (; r < nOut; r++, base += nIn) {
          let s0 = 0, s1 = 0, s2 = 0, s3 = 0, c = 0;
          for (; c < n4; c += 4) {
            s0 += w[base + c] * inp[c];
            s1 += w[base + c + 1] * inp[c + 1];
            s2 += w[base + c + 2] * inp[c + 2];
            s3 += w[base + c + 3] * inp[c + 3];
          }
          let s = s0 + s1 + s2 + s3;
          for (; c < nIn; c++) s += w[base + c] * inp[c];
          z[r] = s + b[r];
        }
        if (li < last) {
          for (let r = 0; r < nOut; r++) a[r] = Math.sin(z[r]);
        } else if (head === 'tanh') {
          for (let r = 0; r < nOut; r++) a[r] = Math.tanh(z[r]);
        } else {
          a.set(z);
        }
        inp = a;
      }
      return L[last].a;
    },
    /**
     * Reverse pass over the LAST forward. `gOut` seeds dL/dy on the output;
     * `gIn` receives dL/dx on the input row.
     *
     * `lo`/`hi` restrict which INPUT components are produced (layer 0 only).
     * Both callers want a slice and nothing else — dQ/du and dQ/dd out of the
     * twin head, dQ/du out of the adversary — and layer 0 is the widest input
     * in the stack, so skipping the observation columns is ~10% of the pass for
     * exactly the same numbers.
     */
    backward(gOut, gIn, lo = 0, hi = L[0].nIn) {
      const tail = L[last];
      if (head === 'tanh') {
        const y = tail.a;
        for (let r = 0; r < tail.nOut; r++) gA[r] = gOut[r] * (1 - y[r] * y[r]);
      } else {
        for (let r = 0; r < tail.nOut; r++) gA[r] = gOut[r];
      }
      for (let li = last; li >= 0; li--) {
        const { nIn, nOut, w } = L[li];
        const c0 = li === 0 ? lo : 0;
        const c1 = li === 0 ? hi : nIn;
        gB.fill(0, c0, c1);
        // dL/dx = W^T dL/dz, two rows of W at a time so the read-modify-write
        // on gB is paid once per pair.
        let r = 0, base = 0;
        for (; r + 1 < nOut; r += 2, base += 2 * nIn) {
          const g0 = gA[r], g1 = gA[r + 1];
          if (g0 === 0 && g1 === 0) continue;
          const base1 = base + nIn;
          for (let c = c0; c < c1; c++) gB[c] += w[base + c] * g0 + w[base1 + c] * g1;
        }
        for (; r < nOut; r++, base += nIn) {
          const gr = gA[r];
          if (gr === 0) continue;
          for (let c = c0; c < c1; c++) gB[c] += w[base + c] * gr;
        }
        if (li > 0) {
          const zPrev = L[li - 1].z;    // dL/dz = dL/da * cos(z)
          for (let c = 0; c < nIn; c++) gB[c] *= Math.cos(zPrev[c]);
        }
        const t = gA; gA = gB; gB = t;  // gA now holds this layer's input grad
      }
      for (let c = lo; c < hi; c++) gIn[c] = gA[c];
      return gIn;
    },
  };
}

/**
 * Wrap the four loaded nets into the three quantities the filter law needs:
 * `u_safe`, `Qhat`, and `grad_u Qhat`. `pessimistic = "max"`
 * (filtered_action.py:322/333 via exit_driver.py:637), so `Qhat = max(Q1, Q2)`.
 */
export function createCertificate(nets) {
  const { ctrl, dstb, q1, q2 } = nets;
  if (ctrl.obsDim !== 62 || ctrl.actDim !== 12) throw new Error('filter.js: ctrl must be 62->12');
  if (dstb.obsDim !== 74 || dstb.actDim !== 6) throw new Error('filter.js: dstb must be 74->6');
  if (q1.obsDim !== 80 || q1.actDim !== 1) throw new Error('filter.js: q1 must be 80->1');
  if (q2.obsDim !== 80 || q2.actDim !== 1) throw new Error('filter.js: q2 must be 80->1');

  const ctrlT = makeTapedNet(ctrl, 'ctrl');
  const dstbT = makeTapedNet(dstb, 'dstb');
  const q1T = makeTapedNet(q1, 'q1');
  const q2T = makeTapedNet(q2, 'q2');

  const xDstb = new Float32Array(74);
  const xQ = new Float32Array(80);
  const uSafe = new Float32Array(12);
  const gQ = new Float64Array(1);
  const gXQ = new Float64Array(80);
  const gD = new Float64Array(6);
  const gXD = new Float64Array(74);
  const tapeU = new Float64Array(12);
  let winner = null;     // which twin head took the pessimistic max, last call
  let evals = 0;

  return {
    /** u_safe = pi_shield(obs). REUSED buffer. */
    fallback(obs62) {
      uSafe.set(ctrlT.forward(obs62));
      return uSafe;
    },
    /**
     * Qhat(obs, u) = max over the twin heads of Q(obs, u, dstb(obs, u)).
     * Leaves the tape in place so `gradU` can differentiate THIS call.
     */
    robustQ(obs62, u12) {
      xDstb.set(obs62, 0);
      for (let i = 0; i < 12; i++) { xDstb[62 + i] = u12[i]; tapeU[i] = u12[i]; }
      const d = dstbT.forward(xDstb);
      xQ.set(obs62, 0);
      for (let i = 0; i < 12; i++) xQ[62 + i] = u12[i];
      xQ.set(d, 74);
      const a = q1T.forward(xQ)[0];
      const b = q2T.forward(xQ)[0];
      winner = a > b ? q1T : q2T;
      evals += 1;
      return a > b ? a : b;
    },
    /**
     * grad_u Qhat at the (obs, u) of the LAST `robustQ` — the TOTAL derivative
     * dQ/du + (dQ/dd)(d pi_d/du), because the adversary is conditioned on u and
     * is deliberately not detached (filtered_action.py:696-698).
     *
     * `torch.maximum` routes the gradient to the larger element, so only the
     * winning twin head is differentiated.
     *
     * @param {Float64Array} out12  receives dQ/du
     * @param {ArrayLike<number>} u12  the u the caller believes is on the tape
     */
    gradU(out12, u12) {
      if (winner === null) throw new Error('filter.js: gradU before any robustQ');
      for (let i = 0; i < 12; i++) {
        if (tapeU[i] !== u12[i]) {
          throw new Error('filter.js: gradU asked for a u the tape does not hold — ' +
            'robustQ(obs, u) must be the immediately preceding call');
        }
      }
      gQ[0] = 1;
      winner.backward(gQ, gXQ, 62, 80);  // dQ/d[u, d]; dQ/dobs is not wanted
      for (let i = 0; i < 6; i++) gD[i] = gXQ[74 + i];
      dstbT.backward(gD, gXD, 62, 74);   // (dQ/dd)(d pi_d/du)
      for (let i = 0; i < 12; i++) out12[i] = gXQ[62 + i] + gXD[62 + i];
      return out12;
    },
    /** How many robust_q evaluations since the last `resetCost()`. */
    cost() { return evals; },
    resetCost() { evals = 0; },
  };
}

/**
 * The filter law for ONE robot: the decision cascade plus whichever solver
 * `params.intervention` names. Pure numbers in, pure numbers out — no sim, no
 * env.
 *
 * @param {object} cert   from `createCertificate`
 * @param {object} params {kappa, cbfMaxIters, cbfTol, intervention,
 *                         pgStep, pgMaxIters, pgBacktrackIters}
 */
export function createShield(cert, params) {
  const kappa = params.kappa;
  const maxIters = params.cbfMaxIters;
  const tol = params.cbfTol;
  const intervention = params.intervention ?? INTERVENTION.PROJECTED_GRADIENT;
  if (intervention !== INTERVENTION.LINE_SEARCH
      && intervention !== INTERVENTION.PROJECTED_GRADIENT) {
    throw new Error(`filter.js: unknown intervention "${intervention}" ` +
      `(valid: ${Object.values(INTERVENTION).join(', ')})`);
  }
  const pgStep = params.pgStep ?? PG.step;
  const pgMaxIters = params.pgMaxIters ?? PG.maxIters;
  const pgBacktrackIters = params.pgBacktrackIters ?? PG.backtrackIters;

  const uUn = new Float64Array(12);
  const uSb = new Float64Array(12);
  const uNew = new Float64Array(12);
  const uRes = new Float64Array(12);
  const uSafeCopy = new Float64Array(12);
  const out = new Float64Array(12);
  // Projected-gradient scratch.
  const uAsc = new Float64Array(12);
  const uFeas = new Float64Array(12);
  const uLo = new Float64Array(12);
  const uHi = new Float64Array(12);
  const uMid = new Float64Array(12);
  const grad = new Float64Array(12);
  let pgSolved = 0, pgInfeasible = 0;

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
   * The SECANT on [u_task (infeasible), u_safe (feasible)], into `uRes`.
   * filtered_action.py:629-675; the C++ twin is safety_filter.h:713-746.
   * @returns {number} how many robust_q evaluations the search spent
   */
  function lineSearch(obs62, uTask, qTask, V, thr) {
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
    return iters;
  }

  /**
   * The PROJECTED GRADIENT, into `uRes`: the KKT solve of
   * `argmin ||u - u_task||^2 s.t. Qhat(x, u) >= kappa V(x)`.
   * filtered_action.py:677-785, one env wide.
   *
   * `qTask` is passed in because the caller has just evaluated
   * `robustQ(obs, u_task)` and the ascent's first feasibility test reads
   * exactly that point — python re-evaluates it ("costs nothing extra", :737-739)
   * and gets the same number; here the tape from that call is still warm, so
   * the first iteration only pays for the reverse pass.
   *
   * Differences from a literal transcription, both exact for a single env:
   *   * python freezes a vanishing-gradient row and lets the remaining
   *     iterations re-test the SAME u (which cannot become feasible, it was
   *     just tested); one env can leave the loop instead, and must then skip
   *     the post-loop re-test of that same u.
   *   * python's batched `newly` bookkeeping collapses to one boolean.
   *
   * @returns {number} the number of ascent steps actually taken
   */
  function projectedGradient(obs62, uTask, thr, qTask) {
    uAsc.set(uTask);
    let found = false, stalled = false, ascent = 0;
    let q = qTask;                       // the tape holds robustQ(obs, u_task)
    for (let n = 0; n < pgMaxIters; n++) {
      if (n > 0) q = cert.robustQ(obs62, uAsc);
      if (q >= thr) { uFeas.set(uAsc); found = true; break; }
      cert.gradU(grad, uAsc);
      let gsq = 0;
      for (let i = 0; i < 12; i++) gsq += grad[i] * grad[i];
      const gnorm = Math.sqrt(gsq);
      if (gnorm < PG.gradEps) { stalled = true; break; }
      const s = pgStep / Math.max(gnorm, PG.gradEps);
      for (let i = 0; i < 12; i++) uAsc[i] = clamp(uAsc[i] + s * grad[i], -1, 1);
      ascent += 1;
    }
    // "The loop steps after testing, so the last iterate is still untested."
    // A stalled row's last iterate WAS tested (it is what produced the zero
    // gradient), so only a budget-exhausted row needs this.
    if (!found && !stalled && cert.robustQ(obs62, uAsc) >= thr) {
      uFeas.set(uAsc);
      found = true;
    }
    if (found) {
      // Bisection towards u_task — deliberately NOT the secant: Qhat is a
      // sin-MLP and is not monotone along this segment, and bisection always
      // returns the FEASIBLE side. filtered_action.py:765-778.
      uLo.set(uTask); uHi.set(uFeas);
      for (let m = 0; m < pgBacktrackIters; m++) {
        for (let i = 0; i < 12; i++) uMid[i] = 0.5 * (uLo[i] + uHi[i]);
        if (cert.robustQ(obs62, uMid) >= thr) uHi.set(uMid);
        else uLo.set(uMid);
      }
      uRes.set(uHi);
      pgSolved += 1;
    } else {
      // NON-CONVERGENCE: u_safe at alpha 1, the same thing the secant does when
      // it exhausts its budget — and counted, so a mistuned eta/N is visible
      // instead of quietly degrading. filtered_action.py:708-714.
      uRes.set(uSafeCopy);
      pgInfeasible += 1;
    }
    return ascent;
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

    // filtered_action.py:556-560 — the guard/handback machine is dead in this
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

    // The ONE swapped step (filtered_action.py:567-574): both solvers resolve
    // the same rows and return the same (u, alpha) pair, so nothing above or
    // below this changes.
    const pg = intervention === INTERVENTION.PROJECTED_GRADIENT;
    const iters = pg
      ? projectedGradient(obs62, uTask, thr, qTask)
      : lineSearch(obs62, uTask, qTask, V, thr);
    out.set(uRes);
    return {
      u: out,
      alpha: computeAlpha(uTask, uSafeCopy, uRes),
      decision: pg ? DECISION.QP : DECISION.LINE_SEARCH,
      V, qTask, thr, iters, qEvals: cert.cost(),
    };
  }

  return {
    step,
    computeAlpha,
    /** Rows the ascent could not make feasible within pg_max_iters (and so fell
     *  back to u_safe), vs rows it solved. filtered_action.py:370-374. */
    stats: () => ({ pgSolved, pgInfeasible }),
    resetStats() { pgSolved = 0; pgInfeasible = 0; },
    params: {
      kappa, cbfMaxIters: maxIters, cbfTol: tol, intervention,
      pgStep, pgMaxIters, pgBacktrackIters,
    },
  };
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
/**
 * The rectangle the certificate measures its wall margins against.
 *
 * The two games do NOT share it, and this is the one filter knob they differ
 * on (recon/03:109-113). The asymmetric preset leaves `field_size` unset, so
 * the certificate reads the BUNDLE's 4.8 x 3.0 at the origin even though the
 * pitch is 5.2 x 3.0 at (0.2, 0) — trained that way, reproduced, not fixed.
 * `sym_preset.py:114-131` passes `_SYM_FIELD = (5.6, 3.0)` and
 * `_SYM_CENTER = (0, 0)` explicitly, so on the symmetric pitch the certificate
 * sees the real walls. Feeding it the bundle rectangle there would put the
 * boundary 0.4 m inside each end zone and have it fight a dog that is safely
 * on the pitch.
 */
function fieldFor(entry, game) {
  const base = {
    halfX: entry.field.length / 2,
    halfY: entry.field.width / 2,
    cx: entry.field.center[0],
    cy: entry.field.center[1],
    dVis: entry.field.d_vis,
  };
  if (game !== 'sym') return base;
  return { ...base, halfX: 2.8, halfY: 1.5, cx: 0, cy: 0 };
}

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
  // The two games share the value function and the adversary and differ ONLY in
  // the fallback controller: the symmetric members were played behind the
  // stage-1 ctrl bundle (recon/02:342-348, proved tensor by tensor in
  // tools/export_filter_s1ctrl.py). Running them behind the asymmetric ctrl
  // proposes a `u_safe` they were never certified against, and the dog topples.
  const ctrlKey = opts.game === 'sym' && entry.nets.ctrl_s1 ? 'ctrl_s1' : 'ctrl';
  const [ctrl, dstb, q1, q2] = await Promise.all(
    [ctrlKey, 'dstb', 'q1', 'q2'].map((k) => loadPolicy(dir + entry.nets[k].json)),
  );

  // The manifest is the source of truth for everything the BUNDLE decides;
  // these asserts only catch an exporter and a runtime that have drifted apart.
  const p = entry.params;
  if (p.pessimistic !== 'max') throw new Error(`filter.js: pessimistic "${p.pessimistic}" is not max`);
  if (p.full_takeover || p.fallback_line_search || p.kin_enabled || p.sup_enabled || p.guard_enabled) {
    throw new Error('filter.js: the bundle arms a latch this build does not implement');
  }
  // The value channel really is parked: the yaml's own value_eps and the
  // guard/release pair make `danger` identically false, which is why no
  // GUARD/HANDBACK machine is ported (exit_driver.py:700-703).
  if (!(p.value_guard < -1e8) || !(p.value_eps < -1e8)) {
    throw new Error(`filter.js: the value channel is armed (value_guard ${p.value_guard}, ` +
      `value_eps ${p.value_eps}); this build ports the pure Q-CBF lane only`);
  }
  // THE SOLVER IS NOT THE BUNDLE'S TO CHOOSE. `p.intervention` is whatever
  // params/deploy.yaml says ("line_search" for this bundle, which never named
  // one); the action term lets the run cfg override it, and both shipped S2C
  // checkpoints were trained with that override set to "projected_gradient"
  // (game_qcbf_action.py:117-118, and the run_config.yaml quoted in the
  // header). Ship the solver the WEIGHTS were trained under, not the one the
  // bundle's yaml defaults to; `opts.intervention` forces the other for an A/B.
  const bundleIntervention = p.intervention ?? INTERVENTION.LINE_SEARCH;
  const intervention = opts.intervention ?? INTERVENTION.PROJECTED_GRADIENT;
  if (bundleIntervention !== INTERVENTION.LINE_SEARCH
      && bundleIntervention !== INTERVENTION.PROJECTED_GRADIENT) {
    throw new Error(`filter.js: the bundle names an unknown intervention "${bundleIntervention}"`);
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
      intervention,
      bundleIntervention,
      // The bundle carries no pg_* keys (it never named a solver), so these are
      // the run-config values; a bundle that DOES carry them wins, the same
      // precedence game_qcbf_action.py:119-131 gives the yaml over the defaults.
      pgStep: p.pg_step ?? PG.step,
      pgMaxIters: p.pg_max_iters ?? PG.maxIters,
      pgBacktrackIters: p.pg_backtrack_iters ?? PG.backtrackIters,
      incScale: entry.increment.scale,
      incSmoothing: entry.increment.smoothing,
      qLo: entry.increment.q_lo,
      qHi: entry.increment.q_hi,
    },
    field: fieldFor(entry, opts.game),
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
 * @param {boolean} [o.gainBlend]   default TRUE — see the header
 * @param {object}  [o.footContacts] override the shared scanner (tests)
 */
export function makeShieldPath({
  filter, sim, robot, opponentRobot, gainBlend = true, footContacts = null,
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
      const st = shield.stats();
      last = {
        alpha: r.alpha,
        active: r.decision !== DECISION.TASK_PASS,
        intervening: r.decision !== DECISION.TASK_PASS,
        decision: r.decision,
        decisionName: DECISION_NAME[r.decision],
        value: r.V,
        qTask: r.qTask,
        thr: r.thr,
        iters: r.iters,          // ascent steps (QP) / secant iterations (LS)
        qEvals: r.qEvals,
        pgSolved: st.pgSolved,
        pgInfeasible: st.pgInfeasible,
        gainAlpha,
        stepMs,
      };
      return ctrl;
    },

    info() { return last; },

    /** Which solver this seat runs, and its constants. */
    solver() { return shield.params; },

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
  INTERVENTION, PG,
  rectRectDistance, hullCentre, opponentTail, rollPitchFromQuat, filterObs,
  createFootContacts, footContactsFor, createCertificate, createShield,
  loadFilter, makeShieldPath,
};

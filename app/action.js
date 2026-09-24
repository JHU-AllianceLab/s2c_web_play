// app/action.js — the two action paths. They are NOT the same, and getting the
// second one wrong is the single most expensive mistake in this port.
//
// =============================================================================
// PATH 1 — the human's walk policy: WalkAffine
// =============================================================================
//
//     q_des = default_joint_pos + 0.25 * a        -> straight to the PD target
//
// `JointPositionActionCfg(scale=0.25, use_default_offset=True)`
// (src/tasks/velocity/velocity_env_cfg.py:152-159), processed by
// `BaseAction.process_actions` = `raw * scale + offset`
// (mjlab/envs/mdp/actions/actions.py:145-148) with
// `offset = default_joint_pos` (actions.py:203-204). No clipping anywhere
// (`clip: null` in the run's agent.yaml / deploy.yaml, recon/04:206).
//
// =============================================================================
// PATH 2 — every game policy (N / P / C / L, and the S2C shield's own internal
//          proposal): IncrementIntegrator
// =============================================================================
//
// `WalkAffineIncrementJointPositionAction`
// (src/tasks/safety/mdp/ctrl_action.py:153-193), configured by
// `NoSafetyDeployWalkIncrement.action_term` (src/tasks/game/players.py:145-152)
// as `scale=0.25, use_default_offset=True, increment_scale=0.5,
// action_smoothing=0.3, action_delay_max_steps=0`:
//
//     q_des   = 0.25 * a + q_default                     # the walker affine
//     u       = clamp((q_des - target) / (s*alpha), -1, 1)   # s*alpha = 0.15
//     inc     = clamp(u * s, -s, +s)                     # s = 0.5
//     clamped = clamp(target + inc, q_lo, q_hi)          # SOFT joint limits
//     target += alpha * (clamped - target)               # alpha = 0.3
//     ctrl    = target                                   # no encoder bias off-DR
//
// `target` is PERSISTENT STATE, re-seeded at every reset from the MEASURED
// post-reset joint positions (ctrl_action.py:107-123) — never from the default
// pose. `q_lo`/`q_hi` are the SOFT limits, `mid +/- 0.9*halfrange`
// (soft_joint_pos_limit_factor 0.9, go2_constants.py:125; formula at
// mjlab/entity/entity.py:610-623).
//
// WHAT THE MATH ACTUALLY DOES (worth knowing, and unit-tested):
//   * away from the joint limits and while |q_des - target| <= 0.15, the
//     alpha in the EMA exactly cancels the (s*alpha) in the integrator inverse,
//     so `target_new == q_des` EXACTLY, in one step;
//   * beyond that, u saturates and the target moves exactly
//     alpha*s = 0.15 rad per control step (7.5 rad/s).
//   So the term is a 0.15 rad/step SLEW-RATE LIMITER on the walker's q_des,
//   plus a limit clamp applied to the FULL (pre-EMA) increment.
//
// ⚠ "A zero action HOLDS the last target" (DESIGN.md section 5, recon/05:178)
//   is true of the PARENT term `IncrementJointPositionAction`
//   (ctrl_action.py:42-127, where `inc = clamp(a*s, +/-s)` so a = 0 means
//   inc = 0). It is NOT true of the `WalkAffine` subclass the shipped game arms
//   actually use: there a = 0 means `q_des = q_default`, so the target slews to
//   the DEFAULT POSE at 0.15 rad/step. Both are implemented here — `affine:
//   true` (the default) is the shipped one. The part of that warning that DOES
//   apply to both, and that matters: the target must be seeded from the
//   measured joint positions, never zeroed, or the dog marches in place.
//
// Neither path clips the action (`clip = None` on both terms, recon/05:166) and
// at play time the action is the deterministic MLP mean.

/** Walker affine scale. src/tasks/game/robots.py:166 (`action_scale = 0.25`),
 *  src/tasks/sym_game/robots.py:166, velocity_env_cfg.py:154. */
export const ACTION_SCALE = 0.25;

/** v25 per-joint increment range. ctrl_action.py:135 (`increment_scale = 0.5`). */
export const INCREMENT_SCALE = 0.5;

/** v25 EMA factor on the target. ctrl_action.py:138 (`action_smoothing = 0.3`). */
export const ACTION_SMOOTHING = 0.3;

/** The integrator-inverse denominator, s*alpha. deploy_filter.py:468-472. */
export const INVERSE_DENOM = INCREMENT_SCALE * ACTION_SMOOTHING; // 0.15

/** Nominal stance, JOINT order FL,FR,RL,RR x (hip,thigh,calf).
 *  go2_constants.py:72-81. */
export const DEFAULT_JOINT_POS = Object.freeze([
  -0.1, 0.9, -1.8, 0.1, 0.9, -1.8, -0.1, 0.9, -1.8, 0.1, 0.9, -1.8,
]);

/**
 * SOFT joint limits, `mid +/- 0.9*halfrange`, JOINT order. Measured on the live
 * entity (recon/05:156-161) and reproduced in
 * assets/scene/<game>/scene.json `joints.softLimitLo/Hi`, which is where
 * app/match.js should read them from. These constants exist only as a fallback
 * and as the value the tests check the scene against.
 */
export const SOFT_JOINT_POS_LIMIT_LO = Object.freeze([
  -0.94248, -1.317725, -2.628453,
  -0.94248, -1.317725, -2.628453,
  -0.94248, -0.270525, -2.628453,
  -0.94248, -0.270525, -2.628453,
]);
export const SOFT_JOINT_POS_LIMIT_HI = Object.freeze([
  0.94248, 3.237625, -0.932007,
  0.94248, 3.237625, -0.932007,
  0.94248, 4.284825, -0.932007,
  0.94248, 4.284825, -0.932007,
]);

function f64(src, n, what) {
  if (!src || src.length !== n) {
    throw new Error(`action.js: ${what} must have ${n} entries, got ${src ? src.length : src}`);
  }
  return Float64Array.from(src);
}

const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);

/**
 * Path 1. `q_des = default + scale * a`, stateless.
 *
 * Returned buffer is REUSED between calls — copy it if you need to keep it.
 */
export class WalkAffine {
  /**
   * @param {ArrayLike<number>} [defaultJointPos]  12, JOINT order
   * @param {object} [opts]
   * @param {number} [opts.scale] default 0.25
   */
  constructor(defaultJointPos = DEFAULT_JOINT_POS, opts = {}) {
    this.def = f64(defaultJointPos, 12, 'defaultJointPos');
    this.scale = opts.scale ?? ACTION_SCALE;
    this._out = new Float64Array(12);
  }

  /** No state to clear; present so both classes share one lifecycle. */
  reset() {
    return this;
  }

  /**
   * @param {ArrayLike<number>} a12  raw policy output
   * @returns {Float64Array} q_des, JOINT order — feed straight to sim.setCtrl()
   */
  step(a12) {
    if (a12.length !== 12) throw new Error('action.js: step expects 12 actions');
    const out = this._out, d = this.def, s = this.scale;
    for (let i = 0; i < 12; i++) out[i] = s * a12[i] + d[i];
    return out;
  }
}

/**
 * Path 2. The v25 walker-affine increment integrator with a persistent target.
 *
 * Returned buffer is REUSED between calls — copy it if you need to keep it.
 */
export class IncrementIntegrator {
  /**
   * @param {ArrayLike<number>} [defaultJointPos] 12, JOINT order
   * @param {ArrayLike<number>} [lo]  soft joint lower limits (scene.json
   *                                  joints.softLimitLo); null -> the constant
   * @param {ArrayLike<number>} [hi]  soft joint upper limits
   * @param {object} [opts]
   * @param {number}  [opts.scale]          walker affine scale, default 0.25
   * @param {number}  [opts.incrementScale] s, default 0.5
   * @param {number}  [opts.smoothing]      alpha, default 0.3
   * @param {boolean} [opts.affine]         default true = the shipped
   *        `WalkAffineIncrementJointPositionAction`. false reproduces the
   *        PARENT `IncrementJointPositionAction` (`inc = clamp(a*s, +/-s)`),
   *        used by no shipped game arm.
   * @param {ArrayLike<number>} [opts.seed] seed the target immediately
   */
  constructor(defaultJointPos = DEFAULT_JOINT_POS, lo = null, hi = null, opts = {}) {
    this.def = f64(defaultJointPos, 12, 'defaultJointPos');
    this.lo = f64(lo ?? SOFT_JOINT_POS_LIMIT_LO, 12, 'lo');
    this.hi = f64(hi ?? SOFT_JOINT_POS_LIMIT_HI, 12, 'hi');
    for (let i = 0; i < 12; i++) {
      if (!(this.lo[i] < this.hi[i])) {
        throw new Error(`action.js: joint ${i} has lo ${this.lo[i]} >= hi ${this.hi[i]}`);
      }
    }
    this.scale = opts.scale ?? ACTION_SCALE;
    this.s = opts.incrementScale ?? INCREMENT_SCALE;
    this.alpha = opts.smoothing ?? ACTION_SMOOTHING;
    this.affine = opts.affine !== false;

    this.target = new Float64Array(12);
    this._seeded = false;
    this._out = new Float64Array(12);
    if (opts.seed) this.reset(opts.seed);
  }

  /**
   * Seed the persistent target from the MEASURED post-reset joint positions —
   * ctrl_action.py:107-123. The env writes qpos before `action_manager.reset`,
   * so the value used is the pose the robot is actually in.
   *
   * There is no default: a target of zeros (or of the default pose when the
   * robot was reset somewhere else) is exactly the bug that makes a game policy
   * march in place, so this must be called, with `sim.getJointPos(robot)`.
   *
   * @param {ArrayLike<number>} measured12
   */
  reset(measured12) {
    if (!measured12 || measured12.length !== 12) {
      throw new Error(
        'action.js: IncrementIntegrator.reset needs the 12 MEASURED joint positions ' +
          '(sim.getJointPos(robot)) — ctrl_action.py:107-123',
      );
    }
    for (let i = 0; i < 12; i++) this.target[i] = measured12[i];
    this._seeded = true;
    return this;
  }

  /**
   * One control step.
   * @param {ArrayLike<number>} a12  raw policy output
   * @returns {Float64Array} the new target = ctrl, JOINT order
   */
  step(a12) {
    if (!this._seeded) {
      throw new Error(
        'action.js: IncrementIntegrator.step() before reset(measured joint_pos). ' +
          'The v25 target is persistent state and must be seeded from the post-reset ' +
          'pose (ctrl_action.py:107-123).',
      );
    }
    if (a12.length !== 12) throw new Error('action.js: step expects 12 actions');

    const t = this.target, d = this.def, lo = this.lo, hi = this.hi;
    const s = this.s, alpha = this.alpha, scale = this.scale;
    const denom = s * alpha;

    for (let i = 0; i < 12; i++) {
      let inc;
      if (this.affine) {
        const qDes = scale * a12[i] + d[i];
        const u = clamp((qDes - t[i]) / denom, -1, 1);
        inc = clamp(u * s, -s, s);
      } else {
        inc = clamp(a12[i] * s, -s, s);
      }
      const clamped = clamp(t[i] + inc, lo[i], hi[i]);
      t[i] += alpha * (clamped - t[i]);
    }
    this._out.set(t);
    return this._out;
  }

  // ------------------------------------------------------------------------
  // The two halves, exposed separately so the QCBF shield does not need a
  // second copy of this arithmetic (DESIGN.md section 11 rule 4: "the player
  // and the AI go through ONE filter implementation, not two").
  // `GameQcbfShieldAction.process_actions` (game_qcbf_action.py:274-293) is
  // exactly:
  //     u_task = taskIncrement(a)          // deploy_filter.py:468-472
  //     u_sel  = shield.step(obs62, u_task, qDes(a))
  //     ctrl   = applyIncrement(u_sel)     // deploy_filter.py:779-786
  // and `step(a)` above is `applyIncrement(taskIncrement(a))`.
  // ------------------------------------------------------------------------

  /**
   * The integrator INVERSE: the certified [-1,1] increment that moves the
   * persistent target toward `q_des = 0.25*a + default`.
   * deploy_filter.py:468-472 / ctrl_action.py:186-188.
   */
  taskIncrement(a12, out) {
    if (!this._seeded) throw new Error('action.js: taskIncrement before reset(measured joint_pos)');
    const o = out ?? new Float64Array(12);
    const denom = this.s * this.alpha;
    for (let i = 0; i < 12; i++) {
      o[i] = clamp((this.scale * a12[i] + this.def[i] - this.target[i]) / denom, -1, 1);
    }
    return o;
  }

  /**
   * The v25 integrate step for an already-chosen increment `u` in [-1,1]:
   * clip, clamp to the soft limits, EMA. deploy_filter.py:779-786.
   * @returns {Float64Array} the new target = ctrl
   */
  applyIncrement(u12) {
    if (!this._seeded) throw new Error('action.js: applyIncrement before reset(measured joint_pos)');
    const t = this.target, lo = this.lo, hi = this.hi, s = this.s, alpha = this.alpha;
    for (let i = 0; i < 12; i++) {
      const inc = clamp(u12[i] * s, -s, s);
      const clamped = clamp(t[i] + inc, lo[i], hi[i]);
      t[i] += alpha * (clamped - t[i]);
    }
    this._out.set(t);
    return this._out;
  }

  /** The proposal the integrator is chasing, `0.25*a + default` (diagnostics). */
  qDes(a12, out) {
    const o = out ?? new Float64Array(12);
    for (let i = 0; i < 12; i++) o[i] = this.scale * a12[i] + this.def[i];
    return o;
  }

  /** True once reset() has seeded the target. */
  get seeded() {
    return this._seeded;
  }
}

export default {
  WalkAffine,
  IncrementIntegrator,
  ACTION_SCALE,
  INCREMENT_SCALE,
  ACTION_SMOOTHING,
  INVERSE_DENOM,
  DEFAULT_JOINT_POS,
  SOFT_JOINT_POS_LIMIT_LO,
  SOFT_JOINT_POS_LIMIT_HI,
};

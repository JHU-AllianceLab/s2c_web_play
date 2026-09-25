/**
 * app/config.js — the single source of truth for every behaviour constant.
 *
 * RULE: no magic numbers anywhere else in this codebase. If a number decides
 * how the game behaves, it lives here with a `file:line` citation into the
 * training repo, and every other module imports it.
 *
 * Citation root (all `src/...`, `deploy/...` paths below are relative to it):
 *   /home/ray/Disk_ext/Go2/Project/unitree_rl_mjlab/
 * Verified deep reports (local only, gitignored):
 *   Meeting/web_game/recon/05_env_physics_contract.md   — the physics/rules bible
 *   Meeting/web_game/recon/03_safety_filter.md          — the QCBF stack
 *   Meeting/web_game/recon/04_player_walk_policy.md     — the WASD walker
 *
 * Two numbers-that-are-not-here, on purpose:
 *   * Everything about the compiled plant (solver, gains, friction, geom masks)
 *     is baked into `assets/scene/<game>/scene.xml` by `tools/export_scene.py`
 *     and re-published as data in `assets/scene/<game>/scene.json`. The mirrors
 *     in `PHYS` / `JOINTS` below exist so JS code never has to guess; they are
 *     cross-checked against the shipped scene.json by `tests/node_referee.mjs`.
 *   * Every checkpoint path lives in `assets/policies/manifest.json`. Hard-coding
 *     one here would be a DESIGN.md §3.5 violation.
 */

/** Deep-freeze so a downstream module cannot quietly retune the game. */
function deepFreeze(o) {
  if (o && (typeof o === 'object') && !Object.isFrozen(o)) {
    Object.freeze(o);
    for (const k of Object.getOwnPropertyNames(o)) deepFreeze(o[k]);
  }
  return o;
}

// ---------------------------------------------------------------------------
// 1. PHYSICS CLOCK
// ---------------------------------------------------------------------------

/**
 * The control/physics clock. Mirrors the compiled `<option>` block; the browser
 * gets these from the XML, not from here — this is for JS-side arithmetic
 * (obs phase clock, episode length, HUD timers).
 *
 *   dt         0.005  = SIM_TIMESTEP        game_env_cfg.py:50
 *   decimation 4      = CONTROL_DECIMATION  game_env_cfg.py:51
 *   controlDt  0.02   = dt * decimation     (env.step_dt, 50 Hz)
 */
export const PHYS = deepFreeze({
  dt: 0.005,            // src/tasks/game/game_env_cfg.py:50
  decimation: 4,        // src/tasks/game/game_env_cfg.py:51
  controlDt: 0.02,      // dt * decimation; mjlab env.step_dt
  controlHz: 50,        // 1 / controlDt
  /** Solver block, for cross-checking scene.xml. src/tasks/game/game_env_cfg.py:556-566 */
  solver: {
    integrator: 'implicitfast',  // MujocoCfg default, measured on the built env (recon/05:96)
    solver: 'newton',            // measured (recon/05:97)
    iterations: 10,              // game_env_cfg.py:562
    lsIterations: 20,            // game_env_cfg.py:563
    ccdIterations: 100,          // game_env_cfg.py:564
    tolerance: 1e-8,             // measured (recon/05:98)
    lsTolerance: 0.01,           // measured (recon/05:99)
    cone: 'pyramidal',           // measured (recon/05:102)
    impratio: 1.0,               // measured (recon/05:103)
    gravity: [0, 0, -9.81],      // measured (recon/05:105)
  },
});

// ---------------------------------------------------------------------------
// 2. THE ROBOT
// ---------------------------------------------------------------------------

/**
 * Entity-native JOINT order. The action vector, `joint_pos`/`joint_vel` obs and
 * every 12-vector in this file are in THIS order.
 * ⚠ The compiled mjModel ACTUATOR array is grouped hip x4 / thigh x4 / calf x4.
 * `app/physics.js` crosses the two via scene.json `actuatorIdsJointOrder`;
 * nothing else may index actuators directly.
 * mjlab/entity/entity.py:459-481 (find_joints_by_actuator_names); recon/05:146-150
 */
export const JOINT_ORDER = deepFreeze([
  'FL_hip', 'FL_thigh', 'FL_calf',
  'FR_hip', 'FR_thigh', 'FR_calf',
  'RL_hip', 'RL_thigh', 'RL_calf',
  'RR_hip', 'RR_thigh', 'RR_calf',
]);

/** src/assets/robots/unitree_go2/go2_constants.py:76-81 (INIT_STATE.joint_pos) */
export const DEFAULT_JOINT_POS = deepFreeze([
  -0.1, 0.9, -1.8,   // FL
  0.1, 0.9, -1.8,    // FR
  -0.1, 0.9, -1.8,   // RL
  0.1, 0.9, -1.8,    // RR
]);

/**
 * Joint tables. `softLo/softHi` are the ONLY limits the increment integrator
 * clamps against (`ctrl_action.py:100`); `ctrlRange` is deliberately NOT
 * enforced (`ctrllimited = false`, mjlab/utils/spec.py:171-178).
 *
 * soft = mid +/- 0.9 * halfrange, soft_joint_pos_limit_factor = 0.9
 *   go2_constants.py:126; formula mjlab/entity/entity.py:610-623
 * Values measured on the live entity (recon/05:157-160) and re-measured into
 * assets/scene/<game>/scene.json `joints` by tools/export_scene.py.
 */
export const JOINTS = deepFreeze({
  order: JOINT_ORDER,
  defaultPos: DEFAULT_JOINT_POS,
  softLimitFactor: 0.9,   // go2_constants.py:126
  softLo: [
    -0.94248, -1.317725, -2.628453,
    -0.94248, -1.317725, -2.628453,
    -0.94248, -0.270525, -2.628453,
    -0.94248, -0.270525, -2.628453,
  ],
  softHi: [
    0.94248, 3.237625, -0.932007,
    0.94248, 3.237625, -0.932007,
    0.94248, 4.284825, -0.932007,
    0.94248, 4.284825, -0.932007,
  ],
  /** go2.xml joint ranges; recon/05:50 */
  hardLo: [
    -1.0472, -1.5708, -2.7227,
    -1.0472, -1.5708, -2.7227,
    -1.0472, -0.5236, -2.7227,
    -1.0472, -0.5236, -2.7227,
  ],
  hardHi: [
    1.0472, 3.4907, -0.83776,
    1.0472, 3.4907, -0.83776,
    1.0472, 4.5379, -0.83776,
    1.0472, 4.5379, -0.83776,
  ],
  /** Spawn base height. go2_constants.py:75 (INIT_STATE.pos) */
  spawnZ: 0.32,
});

/**
 * Position-actuator gains, JOINT order. The compiled Clean env measures
 * walk-soft 20/20/40 on all 24 actuators (recon/05:133-138) because every
 * shipped safety condition overrides `entity_cfg()` to `_walk_entity`
 * (src/tasks/game/players.py:68-74, robots.py:205).
 *
 * `safety` is the shield's blend ENDPOINT (touchdown_driver.py:42-44), reached
 * only while the QCBF intervenes — and disabled entirely at num_envs == 1
 * (game_qcbf_action.py:238), which is exactly our case. The browser therefore
 * runs pure walk-soft unless the filter agent deliberately turns the blend on.
 */
export const GAINS = deepFreeze({
  walk: {
    kp: [20, 20, 40, 20, 20, 40, 20, 20, 40, 20, 20, 40],  // go2_constants.py:41-67
    kd: [1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2],
  },
  safety: {
    kp: [100, 100, 200, 100, 100, 200, 100, 100, 200, 100, 100, 200], // touchdown_driver.py:42-44
    kd: [1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2],
  },
  /** 1/8 per control step in the ENV path (game_qcbf_action.py:239 + :382). */
  blendRisePerStep: 1 / 8,
  /** src/tasks/game/mdp/game_qcbf_action.py:238 — blend OFF at num_envs == 1. */
  blendEnabledSingleEnv: false,
});

// ---------------------------------------------------------------------------
// 3. ACTION PATHS — two of them, and they are NOT the same
// ---------------------------------------------------------------------------

/**
 * The human's walker (velocity task) and the AI's game policy use different
 * action paths. Getting this wrong is the single most common way to produce a
 * robot that "marches in place" and looks plausible.
 *
 * WALK (`WalkAffine`): q_des = default + 0.25 * a, written straight to ctrl.
 *   scale 0.25 / offset = default_joint_pos:
 *   src/tasks/velocity/velocity_env_cfg.py:152-159; recon/04:206
 *
 * GAME (`IncrementIntegrator`): affine + the v25 increment integrator, with
 * `target` as PERSISTENT state. A zero action HOLDS the last target; it does
 * NOT return to the default pose.
 *   src/tasks/safety/mdp/ctrl_action.py:179-191
 *     q_des   = 0.25 * a + q_default
 *     u       = clamp((q_des - target) / (scale * smoothing), -1, +1)   // /0.15
 *     inc     = clamp(u * scale, -scale, +scale)                        // +/-0.5
 *     clamped = clamp(target + inc, softLo, softHi)
 *     target += smoothing * (clamped - target)                          // alpha 0.3
 *     ctrl    = target
 *   `target` is re-seeded from the MEASURED joint_pos at every reset
 *   (ctrl_action.py:107-122).
 */
export const ACTION = deepFreeze({
  /** Both paths share the affine. src/tasks/game/robots.py:166; sym_game/robots.py:166 */
  scale: 0.25,
  /** No clipping anywhere; eval takes the deterministic MLP mean. recon/05:166 */
  clip: null,
  walkAffine: {
    scale: 0.25,        // recon/04:206 (velocity_env_cfg.py:152-159)
    useDefaultOffset: true,
  },
  increment: {
    scale: 0.5,           // ctrl_action.py:135 (increment_scale)
    smoothing: 0.3,       // ctrl_action.py:138 (action_smoothing)
    inverseDenom: 0.15,   // = scale * smoothing, ctrl_action.py:186
    delayMaxSteps: 0,     // ctrl_action.py:141; 0 in every shipped arm
    /** encoder_bias = 0 unless DR. mjlab actions.py:206-209; recon/05:176 */
    encoderBias: 0,
  },
});

/** Which action path each side runs. Matches players.py's action-term choice. */
export const POLICY_ROLES = deepFreeze({
  /** The human. 47-D walker -> WalkAffine. */
  player: { obsDim: 47, actionPath: 'walk_affine' },
  /**
   * The AI. 60-D game policy -> IncrementIntegrator.
   * NoSafetyDeployWalkIncrement (src/tasks/game/players.py:128-153) is the term
   * every N/P/C/L arm gets; the S2C weights we ship are the same 60-D MLP with
   * the QCBF certificate REMOVED (see SHIELD below).
   */
  ai: { obsDim: 60, actionPath: 'increment_integrator' },
});

// ---------------------------------------------------------------------------
// 4. OBSERVATIONS
// ---------------------------------------------------------------------------

/**
 * Layout contracts. `app/obs.js` owns the implementation; these are the numbers
 * it must not re-derive.
 *
 * `stepCount` in `walkObs(...)` / `gameObs(...)` is mjlab's `episode_length_buf`
 * = the number of control steps COMPLETED, so it is 0 on the very first tick of
 * an episode and 499 on the last. mjlab increments it BEFORE the obs are built
 * (manager_based_rl_env.py:395-418), and `app/match.js` reproduces that by
 * building the obs for step k with `stepCount = k - 1`.
 */
export const OBS = deepFreeze({
  walkDim: 47,   // recon/04:153-163
  gameDim: 60,   // recon/05:189-204 (measured on a live ManagerBasedRlEnv)
  actDim: 12,
  /** Free-running trot clock, BOTH obs. mdp/observations.py:289-310 */
  phasePeriod: 0.6,
  /**
   * The walker's phase is GATED to (0,0) when |cmd| < 0.1 over all three dims
   * (mdp/observations.py:303-308 with command_name set; recon/04:167-169).
   * The GAME policies pass command_name=None, so their clock ALWAYS runs
   * (observations.py:295-299) — do not gate it.
   */
  walkStandGateNorm: 0.1,
  gamePhaseGated: false,
  /** `cmd` dims 55:58 of the 60-D actor are constant zeros. mdp.zero_twist, observations.py:275-286 */
  gameCmdZeros: [0, 0, 0],
  /**
   * No scale, no clip, no noise, no history on either actor group.
   * game_env_cfg.py:376-387 (enable_corruption=False, history_length=1)
   */
  corruption: false,
  historyLength: 1,
  /** rsl_rl EmpiricalNormalization: (x - mean) / (std + eps). normalization.py:18,46-48 */
  normEps: 0.01,
  /**
   * 60-D actor slices, recon/05:189-204. Published so obs.js, the renderer's
   * debug overlay and the shield agent all index the same vector.
   */
  gameSlices: {
    baseAngVel: [0, 3],
    projectedGravity: [3, 6],
    jointPos: [6, 18],       // q - q_default
    jointVel: [18, 30],
    lastAction: [30, 42],    // this seat's RAW last action, pre-affine
    arenaPose: [42, 46],     // (x, y, cos yaw, sin yaw), ENV-ORIGIN frame
    relPosOpp: [46, 49],
    relYawOpp: [49, 51],
    relVelOpp: [51, 54],
    line: [54, 55],
    cmd: [55, 58],
    phase: [58, 60],
  },
  /** 47-D walker slices, recon/04:153-163. */
  walkSlices: {
    baseAngVel: [0, 3],
    projectedGravity: [3, 6],
    command: [6, 9],         // (vx, vy, wz), body frame — the WASD/QE input
    phase: [9, 11],
    jointPos: [11, 23],
    jointVel: [23, 35],
    lastAction: [35, 47],
  },
});

// ---------------------------------------------------------------------------
// 5. THE PLAYER'S COMMAND BOX
// ---------------------------------------------------------------------------

/**
 * The trained command box of `fastwalk_v3/model_9000`. Clamp to THIS, not to
 * the stale deploy-v0 caps (proximity_driver.py:136-144 still carries those).
 * recon/04:20, :271; src/tasks/velocity/config/go2/env_cfgs.py:177
 */
export const CMD_BOX = deepFreeze({
  vx: [-1.5, 3.0],
  vy: [-1.0, 1.0],
  wz: [-2.0, 2.0],
});

// ---------------------------------------------------------------------------
// 6. SEATS
// ---------------------------------------------------------------------------

/**
 * The MJCF prefixes are `a_` / `b_`, so `app/physics.js` calls the robots 'a'
 * and 'b'. The seat -> robot map is fixed by the export:
 *   sym : a = seat A (scores +1.9), b = seat B (scores -1.9)
 *   asym: a = attacker,             b = defender
 * Asserted against assets/scene/<game>/scene.json `robots.<r>.seat` by
 * tests/node_referee.mjs.
 */
export const SEAT_ROBOT = deepFreeze({
  sym: { A: 'a', B: 'b' },
  asym: { attacker: 'a', defender: 'b' },
});

/**
 * ⚠ TWO DIFFERENT DIRECTIONS, and they disagree for the asym defender.
 *
 * goalDir — the direction this seat SCORES in. Used by the referee.
 *   sym : scenario.direction(role) = +1 / -1   (sym_touchdown.py:104-110)
 *   asym: only the attacker has a line at all  (touchdown.py:378-403)
 *
 * obsDir — the sign in the `{seat}_line` OBSERVATION, dim 54.
 *   sym : sym.line_rel_x_signed = line_x - direction * x   (sym.py:131-145)
 *   asym: mdp.line_rel_x        = line_x - x, with NO direction, for BOTH
 *         seats (touchdown.py:286-301, observations.py:37-47). The asym
 *         defender therefore reads +1, not -1: at its (0.75, 0) spawn the obs
 *         is 1.9 - 0.75 = 1.15, which is exactly what recon/05:236 measures.
 *
 * In both games `obsLine = LINE_X - obsDir * xEnv` with LINE_X = 1.9, because
 * asym's field-local line_x 1.7 plus field_center.x 0.2 IS 1.9 (touchdown.py:52
 * `_LINE_ENV_X`, asserted at :259).
 */
export const SEAT_GOAL_DIR = deepFreeze({
  sym: { A: 1, B: -1 },
  asym: { attacker: 1, defender: 0 },   // 0 = this seat has no scoring line
});
export const SEAT_OBS_DIR = deepFreeze({
  sym: { A: 1, B: -1 },
  asym: { attacker: 1, defender: 1 },   // <-- not a typo; see above
});

// ---------------------------------------------------------------------------
// 7. THE TWO GAMES
// ---------------------------------------------------------------------------

/**
 * Field geometry, spawns and the per-game rule table.
 *
 * `field` / `center` give the OOB rectangle directly: `mdp.base_oob` fires on
 * |x - cx| > field[0]/2 or |y - cy| > field[1]/2, on the trunk CENTRE
 * (terminations.py:224-236). There are NO physical walls — every field and
 * arena geom is contype=0 conaffinity=0 (game_env_cfg.py:68-119; verified on
 * the shipped XML by tools/export_scene.py proof P3).
 */
export const GAMES = deepFreeze({
  sym: {
    // --- geometry -----------------------------------------------------------
    field: [5.6, 3.0],            // sym_touchdown.py:60
    center: [0, 0],               // sym_touchdown.py:61  (revalidate() pins it at 0)
    lineX: 1.9,                   // = field[0]/2 - touchdown_margin, sym_touchdown.py:90-96
    touchdownMargin: 0.9,         // sym_touchdown.py:62
    seats: ['A', 'B'],            // scenario roles attacker/defender, renamed for the UI
    episodeSteps: 500,            // 10.0 s / 0.02 s; presets.py:295 + PHYS.controlDt

    // --- the opening --------------------------------------------------------
    /**
     * NOT the training nominal. This is the LOCKED demo opening
     * `/home/ray/demo_sym/spawn_s2c_open.json`, decoded in recon/02:317 — the
     * same opening every sym render on the website plays from, so a web match
     * is comparable to the clips. Training nominal was A(-1.4,0,0) / B(+1.4,0,pi)
     * (sym_touchdown.py:76-83) and is kept below as `spawnTraining`.
     * Run-up: A 4.043 m, B 4.078 m (both inside |x| <= 2.8).
     */
    spawn: {
      A: { x: -2.1431, y: -0.1409, yaw: 0.0979 },
      B: { x: 2.1775, y: -0.1215, yaw: 3.1381 },
    },
    spawnTraining: {
      A: { x: -1.4, y: 0, yaw: 0 },              // sym_touchdown.py:76
      B: { x: 1.4, y: 0, yaw: Math.PI },         // sym_touchdown.py:81
    },
    /** Match / robustness jitter envelopes. sym_touchdown.py:77-78, :82-83 */
    jitter: { tight: { x: 0.15, y: 0.15, yaw: 0.10 }, wide: { x: 1.0, y: 1.2, yaw: Math.PI } },

    /**
     * The human's speed limit IN THIS GAME ONLY (app/input.js setLimits).
     *
     * Both dogs run for a line here, so the match is a race and whoever is
     * faster wins it without playing. The walker the human drives is not the
     * locomotion the game policies have: measured on this pitch, a held W runs
     * 1.78 m/s and Shift 2.39 m/s, against S2C 1.12, Nominal 1.03, Lagrangian
     * 1.52, CPO 1.75 and ET 1.91. So a straight line plus one sidestep beat
     * three of the five opponents with no skill involved.
     *
     * Narrowed to vx 2.2 with a 0.65 cruise fraction: W gives 1.43 m/s and
     * Shift 2.2 m/s, which sits between the slow opponents and the fast ones —
     * you out-run Nominal and S2C, ET and CPO out-run you, and none of it is
     * decided before the first stride. Still strictly inside CMD_BOX, so the
     * walk policy is never asked for a command it was not trained on.
     *
     * The asymmetric game keeps the full box: there the roles are not
     * symmetric, the attacker has a clock, and speed is the attacker's job.
     */
    playerCmd: { vx: [-1.1, 2.2], vy: [-0.8, 0.8], wz: [-2.0, 2.0], cruiseFrac: 0.65 },

    // --- rules --------------------------------------------------------------
    /** The live TerminationManager vocabulary. scene.json rules.terminationTerms */
    terminationTerms: [
      'touchdown', 'touchdown_def', 'time_out', 'trunk_contact',
      'attacker_fell', 'attacker_oob', 'defender_fell', 'defender_oob',
    ],
    rules: {
      /** Both seats score; seat B's term is literally named `touchdown_def`. sym_touchdown.py:216-226 */
      bothSeatsScore: true,
      /** time_out -> 0 = DRAW. sym.py:334-340 ("time_out never enters"); verdict.py:94-143 */
      timeoutWinner: 'DRAW',
      /** Same-step double cross -> DRAW. sym.py:337 (`where(td_a|td_b, 0, ...)` before the exclusives) */
      doubleCross: 'DRAW',
      /** Same-step double bust -> DRAW ("the parent's ordering scored this +1"). sym.py:335-336 */
      doubleBust: 'DRAW',
      /** A crossing CLEARS a same-step bust: a dive over the line that tips on landing scores. sym.py:337 */
      touchdownClearsBust: true,
    },
    /** Seat -> the sim's own termination-term names, for the ledger/HUD. */
    seatTerms: {
      A: { touchdown: 'touchdown', fell: 'attacker_fell', oob: 'attacker_oob' },
      B: { touchdown: 'touchdown_def', fell: 'defender_fell', oob: 'defender_oob' },
    },
    /** Scenario role behind each UI seat (checkpoints/bundles keep these names). sym_touchdown.py:284 */
    seatRole: { A: 'attacker', B: 'defender' },
    /** Fixed by the shipped roster: every sym AI file is a seat-B build. manifest.json `seating.sym` */
    playerSeats: ['A'],
    aiSeat: 'B',
  },

  asym: {
    // --- geometry -----------------------------------------------------------
    field: [5.2, 3.0],            // touchdown.py:77
    center: [0.2, 0],             // touchdown.py:78
    lineX: 1.9,                   // _LINE_ENV_X, touchdown.py:52; asserted at :259
    touchdownMargin: 0.9,         // touchdown.py:81
    seats: ['attacker', 'defender'],
    episodeSteps: 500,            // presets.py:295 (episode_length_s = 10.0) / PHYS.controlDt

    // --- the opening --------------------------------------------------------
    /** touchdown.py:238 / :243 — the training nominal, unchanged. */
    spawn: {
      attacker: { x: -1.4, y: 0, yaw: 0 },
      defender: { x: 0.75, y: 0, yaw: Math.PI },
    },
    spawnTraining: {
      attacker: { x: -1.4, y: 0, yaw: 0 },
      defender: { x: 0.75, y: 0, yaw: Math.PI },
    },
    /** touchdown.py:239-241, :244-246 (clean preset widens to 1.0/1.2/pi). recon/05:237-238 */
    jitter: { tight: { x: 0.15, y: 0.15, yaw: 0.10 }, wide: { x: 1.0, y: 1.2, yaw: Math.PI } },

    // --- rules --------------------------------------------------------------
    terminationTerms: [
      'touchdown', 'time_out', 'trunk_contact',
      'attacker_fell', 'attacker_oob', 'defender_fell', 'defender_oob',
    ],
    rules: {
      /** Only the attacker has a line. touchdown.py:386-403 (one `touchdown` term, asset = attacker) */
      bothSeatsScore: false,
      /** `lose = ... | time_out` -> the DEFENDER holds. rewards.py:227; terminal_timeout touchdown.py:63 */
      timeoutWinner: 'B',
      doubleCross: null,          // impossible: one line
      /**
       * Same-step double bust -> the ATTACKER wins. `score = where(lose, -1, 0)`
       * then `score = where(win, +1, score)`, so win takes precedence on a tie.
       * rewards.py:226-228 (and its docstring, :140-141). This is the one place
       * the two games genuinely disagree — sym calls it a draw.
       */
      doubleBust: 'A',
      /** `win = touchdown | ...` beats `lose`, so a crossing clears a same-step bust. rewards.py:226-228 */
      touchdownClearsBust: true,
    },
    seatTerms: {
      attacker: { touchdown: 'touchdown', fell: 'attacker_fell', oob: 'attacker_oob' },
      defender: { touchdown: null, fell: 'defender_fell', oob: 'defender_oob' },
    },
    seatRole: { attacker: 'attacker', defender: 'defender' },
    /** The human picks; the AI takes the other seat. manifest.json `seating.asym` */
    playerSeats: ['attacker', 'defender'],
    aiSeat: null,
  },
});

// ---------------------------------------------------------------------------
// 8. THE REFEREE
// ---------------------------------------------------------------------------

/**
 * The canonical body rect, shared by the collision certificate, the hull
 * touchdown rule and the hull OOB rule so they can never disagree about where
 * the body is. The rect centre sits COLLIDE_FWD_OFFSET ahead of base_link along
 * body +x (the Go2 reaches +0.34 m at the head, -0.24 m at the rear).
 * src/tasks/safety_collision/mdp/collide.py:56-57, _planar_pose :137-157,
 * _corners :164-178
 */
export const HULL = deepFreeze({
  half: [0.29, 0.193],   // collide.py:56  COLLIDE_HALF
  fwdOffset: 0.05,       // collide.py:57  COLLIDE_FWD_OFFSET
});

/**
 * THE RULE SET THIS GAME PLAYS. One set, written down, no mixing.
 *
 * ── touchdown reference point: BASE-LINK CENTRE (`hullTouchdown: false`) ──
 *
 * Three candidate rules exist upstream and they are not interchangeable:
 *   a) `mdp.crossed_line`      — base-link centre. terminations.py:37-61
 *   b) `mdp.crossed_line_hull` — any corner of the body rect. terminations.py:63-105
 *   c) the eval harness's "past-line referee" = (b) PLUS voiding a topple whose
 *      hull is past the line, applied to SHIELDED SEATS ONLY.
 *      harness.py:253 (hull_touchdown_filtered), :264 (void_fall_past_line)
 *
 * We play (a), on both seats, in both games. Reasons, in order:
 *
 *   1. It is what the shipped weights were TRAINED under. Every arm on the
 *      roster has `hull_touchdown = False` and `past_line_fall = "strict"`
 *      (touchdown.py:135, :140; measured into scene.json rules). The AI's own
 *      notion of "I scored" is the centre rule; scoring it any other way makes
 *      the opponent play a game it never learned.
 *   2. Rule (c) is a per-seat LENIENCY for a policy the certificate is braking.
 *      Our S2C weights ship UNSHIELDED (the 62-D QCBF stack is the filter
 *      agent's separate lane), so the leniency has nothing to compensate for —
 *      and handing it to one seat in a two-player game is simply a handicap.
 *   3. It is what the videos on the website show. The 11th-meeting renders run
 *      `--no-past-line-referee`, i.e. the base-centre rule
 *      (Meeting/Video/material_sim/demo_sim_source/tools/render_demo.sh;
 *      recon/05:373). A web match and a clip are then the same game.
 *   4. Legibility: one point per dog, one line, and the HUD's "distance to
 *      line" is literally the number the referee thresholds. The hull rule
 *      needs a rotating rectangle drawn on the pitch before a player can see
 *      why a nose-over did or did not count.
 *
 * The hull path IS implemented in `app/referee.js` and tested, so flipping
 * `hullTouchdown` is one boolean and cannot drift from the centre rule.
 * `pastLineFall` accepts only 'strict'; `app/referee.js` throws on anything
 * else rather than silently half-applying rule (c).
 */
export const REFEREE = deepFreeze({
  /** acos(-projected_gravity_b[2]) > this. touchdown.py:123 = math.radians(70) */
  fallLimitAngle: 1.2217304763960306,
  fallLimitAngleDeg: 70,
  /**
   * There is NO base-height fall test in training. `root_height_below_minimum`
   * exists (mjlab/envs/mdp/terminations.py:35-42) but no game scenario installs
   * it. Do not invent one.
   */
  fallBaseZMin: -Infinity,

  /** See the essay above. */
  hullTouchdown: false,     // touchdown.py:135 / sym_touchdown.py (inherited)
  pastLineFall: 'strict',   // touchdown.py:140; the ONLY value app/referee.js accepts

  /** presets.py:297 — any robot-on-robot contact ends the episode. */
  trunkContactTerminal: true,
  /**
   * ⚠ `trunk_contact_surface = "trunk"` is OVERRIDDEN. It is set at
   * presets.py:366 in the horizon ancestor and measured "trunk" on the built
   * clean scenario (scene.json rules.trunkContactSurface) — but with
   * `illegal_contact_geoms` = all 23 collision geoms (robots.py:194), the
   * branch at game_env_cfg.py:245-254 makes the live rule the v81 STRICT one:
   * ANY geom of one robot touching ANY part of the other. That is exactly what
   * `sim.anyContactBetweenRobots()` reports.
   */
  contactSurface: 'any_geom_vs_whole_robot',

  /**
   * touchdown.py:107 (flag) / :419 and sym_touchdown.py:242 (where the terms
   * are built) — the *_fell / *_oob terminals exist in every shipped arm.
   */
  safetyTerminal: true,

  /** touchdown.py:63 — clock-out is a REAL terminal, never a truncation. */
  terminalTimeout: true,

  // --- collision fault attribution ----------------------------------------
  /**
   * The initiator rule. rewards.py:149-150, :174-196; sym.py:299-318.
   *
   *   n_hat = unit(pos_B - pos_A) in the plane           (A -> B)
   *   vA    = vel_A . n_hat            vB = vel_B . (-n_hat)
   *   emaX  = beta * emaX + (1 - beta) * vX,  beta = 0.5 ** (controlDt / 0.12)
   *   verdict uses the PRIOR steps' EMA, folded in AFTER the verdict, so a
   *   collision can never contaminate its own attribution.
   *   emaA >= emaB  =>  A initiated  =>  A LOSES.  (ties go against A)
   *
   * Velocities are `root_link_lin_vel_w` (world frame) of base_link, planar.
   * EMAs reset to 0 at every episode reset (rewards.py:251-255).
   */
  initiator: {
    emaHalfLifeS: 0.12,                              // rewards.py:149
    beta: Math.pow(0.5, 0.02 / 0.12),                // rewards.py:150 -> 0.8908987181403393
    /** `ema_va >= ema_vd` -> -scale, i.e. A at fault. rewards.py:174-176 */
    tieFavours: 'B',
  },
  /**
   * contact_win_scale = 0.0 in every shipped arm (presets.py:730): the zero-sum
   * MIRROR scores contact 0 and the runner charges the initiator out of band.
   * The EPISODE ATTRIBUTION used by eval, the renders and this game credits the
   * VICTIM with the win — harness.py:1062-1066 and verdict.py:131-139 both do
   * `winner = (contact_dir < 0)`. That is the ledger this game reports.
   */
  contactWinScale: 0.0,
  contactAttribution: 'victim_wins',

  /**
   * Terminal-name precedence when several terms fire on the same control step.
   * Transcribed from the SCORE precedence, so the label can never disagree with
   * the winner:
   *   touchdown > trunk_contact > fell/oob > time_out
   * - touchdown over contact: `contact & ~touchdown` (rewards.py:243, sym.py:345)
   * - contact over fell/oob/timeout: the contact branch runs LAST and overwrites
   *   (rewards.py:239-244; harness.py:1060-1067)
   * - fell/oob over time_out: in sym a bust at step 500 scores +/-1 while the
   *   clock alone scores 0, so the label must follow the bust (sym.py:335-336).
   */
  terminalPrecedence: ['touchdown', 'touchdown_def', 'trunk_contact', 'fell', 'oob', 'time_out'],
});

/**
 * The six terminal names `app/referee.js` returns. Same vocabulary as the sim's
 * TerminationManager, collapsed per the frozen interface: `attacker_fell` and
 * `defender_fell` both surface as `fell` (with `detail` naming the exact term),
 * likewise `*_oob` -> `oob`.
 * recon/05:310-313 (the training vocabulary); harness.py:81-89; verdict.py:54-63
 */
export const TERMINALS = deepFreeze([
  'touchdown', 'touchdown_def', 'trunk_contact', 'fell', 'oob', 'time_out',
]);

// ---------------------------------------------------------------------------
// 9. THE SHIELD SEAM (not implemented here — see app/filter.js, filter agent)
// ---------------------------------------------------------------------------

/**
 * Where the QCBF certificate plugs in, and the constants it needs. Reproduced
 * here so the filter agent does not re-derive them; recon/03 is the spec.
 *
 * The shield does NOT wrap the final ctrl. It replaces the whole action TERM
 * (players.py:252-314 swaps `WalkAffineIncrementJointPositionAction` for
 * `GameQcbfShieldAction`), and inside it, it replaces `u_task` with `u_sel`
 * between the affine and the integrator (game_qcbf_action.py:274-293):
 *
 *     q_des  = 0.25 * a + q_default
 *     u_task = clamp((q_des - target) / 0.15, -1, +1)
 *     u_sel  = shield.step(obs62, u_task, q_des)      <-- THE SEAM
 *     target = applyIncrement(u_sel)                  <-- same v25 integrator
 *
 * `app/match.js` therefore exposes the seam as a whole action path
 * (`actionPaths.<seat>`), which is the faithful granularity.
 */
export const SHIELD = deepFreeze({
  obsDim: 62,   // recon/03:325; deploy.yaml obs_dim
  ctrlDim: 12,
  dstbDim: 6,
  kappa: 0.9,                 // deploy.yaml; recon/03:177
  cbfMaxIters: 20,            // deploy.yaml
  cbfTol: 0.01,               // deploy.yaml
  valueEps: -1.0e9,           // viability floor OFF
  pessimistic: 'max',
  /** 62-D split. recon/03:325; DESIGN.md §3.2 */
  slices: { proprio: [0, 48], wall: [48, 52], heading: [52, 54], opponentTail: [54, 62] },
  dVis: 1.5,       // field wall + collision directional horizon
  dMax: 3.0,       // collision tail distance clamp
  /**
   * The bundle's own field is 4.8 x 3.0. The asym presets inherit it; the SYM
   * preset passes (5.6, 3.0) @ (0,0) explicitly (sym_preset.py:118-129). Pass
   * GAMES[game].field / .center and never the bundle default.
   */
  fieldFromGame: true,
  /** Every shipped S2C policy is `filtered: true` upstream. recon/01 §5. */
  s2cShipsUnshielded: true,
});

// ---------------------------------------------------------------------------
// 10. MATCH / UI / CAMERA  (UX defaults — NOT derived from training code)
// ---------------------------------------------------------------------------

/**
 * Everything below is presentation. It is tuned for feel and may be changed
 * freely; nothing here is allowed to touch physics, observations or the
 * referee. Anything that would is in the sections above.
 */
export const MATCH = deepFreeze({
  /**
   * The countdown is DISPLAY ONLY: the sim is not stepped and the episode clock
   * does not run. Stepping during a countdown would let the robots sag out of
   * the z = 0.32 spawn the policies were trained to start from
   * (measured settle: 0.32 -> 0.2303 m, tools/export_scene.py keyframe report),
   * i.e. it would change the initial condition of every match.
   */
  countdownMs: 3000,
  countdownLabels: ['3', '2', '1', 'GO'],
  /** Freeze frame after the verdict before the result card takes over. */
  resultHoldMs: 1200,
  /** Player command smoothing: keys are 0/1, the policy dislikes step inputs. */
  cmdSlewPerSec: { vx: 6.0, vy: 6.0, wz: 10.0 },
  cmdDeadzone: 0.05,
  /**
   * Wall-clock pacing. The loop runs at most this many control steps per
   * animation frame so a hitch cannot fast-forward the match.
   */
  maxStepsPerFrame: 4,
});

/**
 * The camera.
 *
 * ⚠ app/render.js is the IMPLEMENTATION. It deliberately imports nothing but
 * three.js (tests/render_harness.html boots it with no app/config.js at all), so
 * the numbers below are a MIRROR of `MODE_CFG`, `CHASE`, `BROADCAST`, `FPV_EYE`
 * and the two orbit constants in app/render.js — not their source. Only
 * `default` and `modes` are read at runtime (app/ui.js:273, :339); everything
 * else is here so the camera can be read about, and changed, in one place.
 * Change a number here and you must change it in app/render.js too.
 *
 * `chase` geometry, in words: a boom `distance` long, `elevationDeg` above the
 * horizontal, pivoting `pivotZ` over the dog's base body. At the defaults that
 * is 3.13 m behind and 1.46 m above the pivot — the lens ~1.94 m off the floor,
 * looking down 26 deg. It replaced a 2.35 m / 20 deg boom (2.21 m back, 1.24 m
 * up) that sat low enough for the other robot to fill the frame.
 *
 * `omegaPos` / `omegaAim` are the natural frequencies of a critically damped
 * spring (settling ~5.8/w: 0.53 s and 0.36 s), not lerp factors: the eye and the
 * look-at each solve x'' = -2w x' - w^2 x exactly for the frame's dt, so the
 * shot is identical at 30 and 144 fps and can never overshoot.
 */
export const CAMERA = deepFreeze({
  default: 'chase',
  modes: ['chase', 'broadcast', 'fpv'],
  chase: {
    /** boom at zoom 1x; the wheel moves it inside [minDistance, maxDistance]. */
    distance: 3.45, minDistance: 1.6, maxDistance: 9.0,
    elevationDeg: 25, minElevationDeg: 7, maxElevationDeg: 78,
    pivotZ: 0.20, aimZ: 0.16, fov: 52, near: 0.05,
    /** critically damped follow, rad/s — eye and look-at. */
    omegaPos: 11.0, omegaAim: 16.0,
    /** velocity lead on the look-at: seconds ahead, capped in metres. */
    lead: 0.28, leadMax: 1.1,
    /** boom grows this fraction per m/s of dog speed, capped. */
    speedStretch: 0.10, speedStretchMax: 0.34,
    /** the look-at leans this fraction of the gap toward the other robot, */
    duelBias: 0.52,
    /** ... in full below duelNear m of separation, not at all above duelFar, */
    duelNear: 1.0, duelFar: 3.8,
    /** ... and never further than this many metres off the player. */
    duelAimMax: 0.95,
    /** an opponent this close to the camera->player line is eclipsing it, */
    guardRadius: 0.78,
    /** ... so the boom lifts this much and backs off this far, at full bite. */
    guardLiftDeg: 26, guardPush: 0.95,
    /** ground clearance: of the boom TARGET, and the camera's own backstop. */
    floorZ: 0.45, hardFloorZ: 0.32,
  },
  /**
   * az 90 / el -45 is the angle the 11th-meeting demo clips are rendered at
   * (Meeting/Video/material_sim/demo_sim_source/README.txt:19, via recon/06
   * B.3); lookatZ 0.25 from the same line and scripts/render_game_checkpoint.py
   * :206-212. The distance is SOLVED per frame so the whole pitch fits the lens
   * (render.js fitDistance), which is why refDistance is only a reference.
   */
  broadcast: {
    azimuthDeg: 90, elevationDeg: -45, refDistance: 5.1, lookatZ: 0.25,
    fov: 38, near: 0.08, omega: 9.0,
  },
  /**
   * Nose cam: 0.11 m ahead of the Go2's `base3_collision` primitive
   * (pos x = 0.293, assets/scene/asym/scene.xml:97) and a little above it.
   * First-order follow, NOT a spring — a spring lets the eye trail the skull.
   */
  fpv: { eye: [0.404, 0, 0.092], fov: 78, near: 0.015, kPos: 26.0, kQuat: 18.0 },
  /**
   * Mouse. `orbitRadPerPx` is the drag rate at sensitivity 1x and
   * `zoomPerWheelPx` the wheel's exponential rate; `sensitivityRange` /
   * `distanceRange` are the sliders app/ui.js ships, which reach
   * renderer.setMouseSensitivity() and renderer.setCameraDistance().
   */
  mouse: {
    orbitRadPerPx: 0.0055, zoomPerWheelPx: 0.0012,
    sensitivityRange: [0.2, 4.0], distanceRange: [0.4, 2.5],
  },
});

/**
 * Input. The key->axis tables here ARE the source of truth — app/input.js parses
 * them (`INPUT.schemes` -> `SCHEMES`) and invents nothing. The polarity is the
 * repo's own teleop mapping, not a choice: A is +vy, Q is +wz
 * (Project/unitree_rl_mjlab/scripts/teleop_fastwalk_record.py, recon/04:275).
 *
 * The RAMP is not here. Rise/fall seconds, the deadzone, the cruise/sprint split
 * and the pad deadzone live in app/input.js `FEEL`, in one block with the table
 * of what each of them measures out to. Two reasons they stay there: they are
 * per-instance (`createInput(target, { feel })` overrides them), and they are
 * meaningless without the ceiling they are tuned against — `MATCH.cmdSlewPerSec`
 * above, which app/match.js applies a second time on the control clock and which
 * is what actually caps the forward ramp (vx 6.0 m/s^2 = 0.5 s to the top of the
 * box). Raise that if W should reach cruise quicker than 0.3 s.
 */
export const INPUT = deepFreeze({
  schemes: {
    /** DESIGN.md §4: W/S -> vx, A/D -> vy, Q/E -> wz. */
    strafe: { KeyW: 'vx+', KeyS: 'vx-', KeyA: 'vy+', KeyD: 'vy-', KeyQ: 'wz+', KeyE: 'wz-' },
    /** "A/D turns" — for players used to racing controls. DESIGN.md §4. */
    turn: { KeyW: 'vx+', KeyS: 'vx-', KeyA: 'wz+', KeyD: 'wz-', KeyQ: 'vy+', KeyE: 'vy-' },
  },
  defaultScheme: 'strafe',
  sprintKey: 'ShiftLeft',
  pauseKey: 'KeyP',
  restartKey: 'KeyR',
  /**
   * The "Steering sensitivity" slider (app/ui.js), reaching
   * input.setSensitivity(). It scales the vy/wz target AND the vy/wz ramp rate;
   * it never widens CMD_BOX.
   */
  sensitivityRange: [0.4, 1.6],
});

// ---------------------------------------------------------------------------
// 11. ASSET LOCATIONS (relative — GitHub Pages, no build step)
// ---------------------------------------------------------------------------

export const ASSETS = deepFreeze({
  sceneDir: (game) => `assets/scene/${game}/`,
  sceneUrl: (game) => `assets/scene/${game}/scene.xml`,
  sceneJsonUrl: (game) => `assets/scene/${game}/scene.json`,
  policyDir: 'assets/policies/',
  policyManifest: 'assets/policies/manifest.json',
  policyJson: (name) => `assets/policies/${name}.json`,
});

// ---------------------------------------------------------------------------
// 12. HELPERS — the only sanctioned way to read the tables above
// ---------------------------------------------------------------------------

/** Accepts 'sym' | 'asym' | a GAMES entry. Throws on anything else. */
export function gameCfg(game) {
  if (typeof game === 'string') {
    const g = GAMES[game];
    if (!g) throw new Error(`unknown game ${game}; expected one of ${Object.keys(GAMES)}`);
    return g;
  }
  if (game && Array.isArray(game.seats) && Array.isArray(game.field)) return game;
  throw new Error('gameCfg: pass "sym" | "asym" or a GAMES entry');
}

/** 'sym' | 'asym' for a GAMES entry or a key. */
export function gameKey(game) {
  if (typeof game === 'string') { gameCfg(game); return game; }
  for (const [k, v] of Object.entries(GAMES)) if (v === game) return k;
  throw new Error('gameKey: not a GAMES entry');
}

export function seats(game) { return gameCfg(game).seats; }

export function otherSeat(game, seat) {
  const s = seats(game);
  assertSeat(game, seat);
  return s[0] === seat ? s[1] : s[0];
}

export function assertSeat(game, seat) {
  if (!seats(game).includes(seat)) {
    throw new Error(`seat ${seat} is not a seat of this game (${seats(game).join('/')})`);
  }
  return seat;
}

/** Seat -> the physics.js robot id ('a' | 'b'). */
export function seatRobot(game, seat) {
  return SEAT_ROBOT[gameKey(game)][assertSeat(game, seat)];
}

/** Robot id -> seat. */
export function robotSeat(game, robot) {
  const m = SEAT_ROBOT[gameKey(game)];
  for (const [s, r] of Object.entries(m)) if (r === robot) return s;
  throw new Error(`robot ${robot} is not in this game`);
}

/** The direction this seat SCORES in: +1, -1, or 0 (asym defender has no line). */
export function goalDir(game, seat) {
  return SEAT_GOAL_DIR[gameKey(game)][assertSeat(game, seat)];
}

/** The `dir` argument of `gameObs(...)`. See the SEAT_OBS_DIR comment. */
export function obsDir(game, seat) {
  return SEAT_OBS_DIR[gameKey(game)][assertSeat(game, seat)];
}

/**
 * Signed distance to this seat's own target line, positive = the line is ahead.
 * This is BOTH obs dim 54 and the HUD's "distance to line" — one expression, so
 * the number a player reads is the number the policy reads.
 *   asym: mdp.line_rel_x        = 1.9 - x            (observations.py:37-47)
 *   sym : sym.line_rel_x_signed = 1.9 - dir * x      (sym.py:131-145)
 */
export function lineDistance(game, seat, xEnv) {
  const g = gameCfg(game);
  return g.lineX - obsDir(game, seat) * xEnv;
}

/** Env-local x of a seat's scoring line, or null if it has none. */
export function lineXOf(game, seat) {
  const g = gameCfg(game);
  const d = goalDir(game, seat);
  return d === 0 ? null : d * g.lineX;
}

/**
 * ⚠ `GAMES[g].lineX` is the ENV-LOCAL line position (1.9 in both games,
 * `_LINE_ENV_X`, touchdown.py:52, asserted at :259). The scenario's OWN
 * `line_x` attribute is FIELD-LOCAL — it is 1.7 in asym, because the field is
 * centred at x = 0.2 and 0.2 + 1.7 = 1.9 (recon/05:222). `crossed_line` and
 * `crossed_line_signed` test `dir * (x - cx) > line_x_fieldlocal`, so the
 * referee needs the field-local magnitude, not the env-local one.
 *   asym: 1.9 - 0.2 = 1.7   sym: 1.9 - 0 = 1.9 (sym pins cx = 0, sym_touchdown.py:116)
 */
export function lineXLocal(game) {
  const g = gameCfg(game);
  return g.lineX - g.center[0];
}

/** The OOB rectangle: |x - cx| > halfX or |y - cy| > halfY. terminations.py:224-236 */
export function fieldBounds(game) {
  const g = gameCfg(game);
  return {
    cx: g.center[0], cy: g.center[1],
    halfX: g.field[0] / 2, halfY: g.field[1] / 2,
    xMin: g.center[0] - g.field[0] / 2, xMax: g.center[0] + g.field[0] / 2,
    yMin: g.center[1] - g.field[1] / 2, yMax: g.center[1] + g.field[1] / 2,
  };
}

/** Episode length in seconds. */
export function episodeSeconds(game) {
  return gameCfg(game).episodeSteps * PHYS.controlDt;
}

/**
 * The spawn spec to hand `sim.resetAll(...)`, keyed by physics.js robot id and
 * carrying the z and joint pose the training reset uses.
 * `variant`: 'default' (the game's opening) | 'training' (the nominal spawn).
 */
export function spawnSpec(game, variant = 'default') {
  const g = gameCfg(game);
  const src = variant === 'training' ? g.spawnTraining : g.spawn;
  const out = {};
  for (const seat of g.seats) {
    const p = src[seat];
    out[seatRobot(game, seat)] = {
      x: p.x, y: p.y, yaw: p.yaw, z: JOINTS.spawnZ,
      jointPos: DEFAULT_JOINT_POS.slice(),
    };
  }
  return out;
}

/**
 * The 4 corners of the canonical body rect, env-local, in the order
 * collide._corners produces them (++, +-, -+, --).
 * collide.py:137-157 (_planar_pose, incl. the forward offset) + :164-178
 */
export function hullCorners(x, y, yaw, half = HULL.half, fwd = HULL.fwdOffset) {
  const cos = Math.cos(yaw), sin = Math.sin(yaw);
  const cx = x + fwd * cos, cy = y + fwd * sin;
  const [hx, hy] = half;
  const signs = [[1, 1], [1, -1], [-1, 1], [-1, -1]];
  return signs.map(([sx, sy]) => {
    const lx = sx * hx, ly = sy * hy;
    return [cx + lx * cos - ly * sin, cy + lx * sin + ly * cos];
  });
}

/** Clamp a raw (vx, vy, wz) to the walker's trained command box. */
export function clampCmd(vx, vy, wz) {
  const c = (v, [lo, hi]) => (v < lo ? lo : v > hi ? hi : v);
  return { vx: c(vx, CMD_BOX.vx), vy: c(vy, CMD_BOX.vy), wz: c(wz, CMD_BOX.wz) };
}

export default {
  GAMES, PHYS, CMD_BOX, DEFAULT_JOINT_POS, JOINT_ORDER, JOINTS, GAINS, ACTION,
  POLICY_ROLES, OBS, SEAT_ROBOT, SEAT_GOAL_DIR, SEAT_OBS_DIR, REFEREE, HULL,
  TERMINALS, SHIELD, MATCH, CAMERA, INPUT, ASSETS,
};

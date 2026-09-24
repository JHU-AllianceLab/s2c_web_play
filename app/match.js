/**
 * app/match.js — the 50 Hz control loop, for both games.
 *
 * One `tick()` is one mjlab control step. The order below is
 * `ManagerBasedRlEnv.step` (mjlab/envs/manager_based_rl_env.py:339-418), quoted
 * in DESIGN.md §5, and the order is behaviour-bearing:
 *
 *     process_action(a)                # ONCE per control step
 *     repeat 4x: write ctrl -> mj_step
 *     episode_length_buf += 1
 *     terminations / referee           # reads the PRE-mj_forward frame,
 *                                      # one physics substep STALE
 *     rewards                          # -> our initiator EMA fold-in
 *     reset the done envs
 *     mj_forward()                     # exactly one
 *     observations                     # read the POST-forward frame
 *
 * So a tick here is: build obs (post-forward, from the previous tick's
 * `sim.forward()`) -> policies -> action paths -> `setCtrl` -> `sim.step(4)` ->
 * judge -> fold the EMA -> `sim.forward()`. `app/physics.js` deliberately does
 * NOT auto-forward inside `step()` for exactly this reason (its header note 2).
 *
 * Citation root: /home/ray/Disk_ext/Go2/Project/unitree_rl_mjlab/
 */

import {
  GAMES, PHYS, MATCH, JOINTS, DEFAULT_JOINT_POS, OBS, POLICY_ROLES, ACTION, GAINS,
  gameCfg, gameKey, otherSeat, assertSeat, seatRobot, obsDir, lineDistance,
  spawnSpec, clampCmd,
} from './config.js';
import {
  judge, createInitiatorEma, resetInitiatorEma, foldInitiatorEma, explain, timeLeft,
} from './referee.js';
import { walkObs, gameObs } from './obs.js';
import { WalkAffine, IncrementIntegrator } from './action.js';

/** Match phases. `countdown` and `over` do not advance the sim. */
export const PHASE = Object.freeze({
  COUNTDOWN: 'countdown',
  RUNNING: 'running',
  PAUSED: 'paused',
  OVER: 'over',
});

// ---------------------------------------------------------------------------
// action paths — and THE SEAM the safety filter plugs into
// ---------------------------------------------------------------------------

/**
 * An action path turns a policy's raw 12-vector into the 12 absolute joint
 * setpoints `sim.setCtrl` wants. It is a small object, not a function, because
 * the game path carries persistent state (`target`).
 *
 *   { reset(measured12), step(a12, ctx) -> Float64Array(12), info() }
 *
 * ── THE SAFETY-FILTER SEAM ─────────────────────────────────────────────────
 * The QCBF shield does NOT wrap the final ctrl, and it does NOT wrap the raw
 * action. Upstream it REPLACES the whole action term:
 * `CollisionFilteredDeploy.action_term` (src/tasks/game/players.py:252-314)
 * swaps `WalkAffineIncrementJointPositionAction` for `GameQcbfShieldAction`,
 * and inside that term the substitution is one line
 * (src/tasks/game/mdp/game_qcbf_action.py:274-293):
 *
 *     q_des  = 0.25 * a + q_default
 *     u_task = clamp((q_des - target) / 0.15, -1, +1)
 *     u_sel  = shield.step(obs62, u_task, q_des)      <-- the only change
 *     target = applyIncrement(u_sel)                  <-- the SAME v25 integrator
 *
 * So the seam here is a whole action path, which is the faithful granularity:
 * the shield needs `u_task` (not the ctrl), it keeps `prev_ctrl = u_sel` as its
 * own obs dims 36:48, and it rewrites this seat's PD gains while it intervenes
 * (`sim.setGains`, GAINS.safety) — none of which fits a "wrap the output" hook.
 *
 * To install it, pass `actionPaths: { <seat>: makeShieldPath({...}) }` to
 * `createMatch`. `ctx` gives the path everything it needs without importing
 * match.js: `{ sim, robot, seat, game, step, opponentRobot, measured12 }`.
 * Whatever `info()` returns is merged into `hud().filter[seat]`, so an
 * intervening filter can light up the HUD with no further plumbing.
 *
 * ⚠ The gain blend is OFF in the env at `num_envs == 1`
 * (game_qcbf_action.py:238) — which is exactly our case — so a faithful browser
 * shield leaves the gains at walk-soft 20/20/40 unless it deliberately opts in.
 */
// ---------------------------------------------------------------------------
// createMatch
// ---------------------------------------------------------------------------

/**
 * @param {Object}   opts
 * @param {Object}   opts.sim         from `createSim({sceneUrl})` — must be the
 *                                    scene of THIS game.
 * @param {'sym'|'asym'} opts.game
 * @param {string}   opts.playerSeat  'A' (sym) | 'attacker' | 'defender' (asym)
 * @param {Object}   [opts.opponent]  bookkeeping only: `{name, method, display}`
 *                                    from `assets/policies/manifest.json`. The
 *                                    weights arrive already loaded in `policies`.
 * @param {Object}   opts.policies    `{ walk, ai }`, each `{forward, obsDim, actDim}`
 *                                    from `loadPolicy()`. `opponent`/`player` are
 *                                    accepted as aliases for `ai`/`walk`.
 * @param {Object}   opts.input       `{ read(): {vx, vy, wz} }` from `createInput()`
 * @param {Object}   [opts.config]    overrides, merged shallowly over the
 *                                    `config.js` defaults. Only UX keys
 *                                    (`countdownMs`, `cmdSlewPerSec`, ...) are
 *                                    meant to be touched.
 * @param {Object}   [opts.actionPaths] `{ [seat]: path }` — the safety-filter seam.
 * @param {string}   [opts.spawnVariant] 'default' | 'training'
 * @param {Object}   [opts.deps]      test injection: `{walkObs, gameObs, WalkAffine,
 *                                    IncrementIntegrator, judge}`.
 * @returns {{tick, state, reset, pause, resume, hud, dispose, seatOf, info}}
 */
export function createMatch({
  sim, game, playerSeat, opponent = null, policies, input,
  config = null, actionPaths = null, spawnVariant = 'default', deps = null,
} = {}) {
  const g = gameCfg(game);
  const key = gameKey(g);
  const cfg = { ...MATCH, ...(config || {}) };
  const D = {
    walkObs, gameObs, WalkAffine, IncrementIntegrator, judge,
    ...(deps || {}),
  };

  // ---- seats ---------------------------------------------------------------
  if (!playerSeat) playerSeat = g.playerSeats[0];
  assertSeat(g, playerSeat);
  if (!g.playerSeats.includes(playerSeat)) {
    throw new Error(
      `${key}: the human cannot take seat ${playerSeat}. ` +
      `Playable seats are ${g.playerSeats.join('/')} — the shipped roster only ` +
      `has ${key === 'sym' ? 'seat-B' : 'both-seat'} AI builds ` +
      `(assets/policies/manifest.json "seating").`);
  }
  const aiSeat = otherSeat(g, playerSeat);
  const playerRobot = seatRobot(g, playerSeat);
  const aiRobot = seatRobot(g, aiSeat);
  const [seatA, seatB] = g.seats;

  // ---- policies ------------------------------------------------------------
  const walkPolicy = policies?.walk ?? policies?.player;
  const aiPolicy = policies?.ai ?? policies?.opponent;
  requirePolicy('player walk', walkPolicy, POLICY_ROLES.player.obsDim);
  requirePolicy('AI game', aiPolicy, POLICY_ROLES.ai.obsDim);
  if (!input || typeof input.read !== 'function') {
    throw new Error('createMatch: input must expose read() -> {vx, vy, wz}');
  }

  // ---- action paths --------------------------------------------------------
  const paths = {};
  for (const seat of g.seats) {
    const injected = actionPaths && actionPaths[seat];
    if (injected) { paths[seat] = injected; continue; }
    const kind = seat === playerSeat
      ? POLICY_ROLES.player.actionPath
      : POLICY_ROLES.ai.actionPath;
    paths[seat] = wrapPath(
      kind === 'walk_affine'
        // q_des = default + 0.25 * a, straight to the position actuators.
        // recon/04:206 (velocity_env_cfg.py:152-159)
        ? new D.WalkAffine(DEFAULT_JOINT_POS)
        // affine + the v25 increment integrator, clamped to the SOFT joint
        // limits. ctrl_action.py:179-191 (limits: go2_constants.py:126)
        : new D.IncrementIntegrator(DEFAULT_JOINT_POS, JOINTS.softLo, JOINTS.softHi),
      kind);
  }

  // ---- per-seat mutable state ---------------------------------------------
  const robotOf = { [seatA]: seatRobot(g, seatA), [seatB]: seatRobot(g, seatB) };
  const lastAction = {
    [seatA]: new Float32Array(OBS.actDim),
    [seatB]: new Float32Array(OBS.actDim),
  };
  const lastCtrl = { [seatA]: null, [seatB]: null };
  const filterInfo = { [seatA]: null, [seatB]: null };

  let phase = PHASE.COUNTDOWN;
  let stepCount = 0;              // = mjlab episode_length_buf (steps COMPLETED)
  let verdict = null;
  let countdownLeftMs = cfg.countdownMs;
  let episodeIndex = 0;
  let cmd = { vx: 0, vy: 0, wz: 0 };   // slew-limited player command
  let lastRawCmd = { vx: 0, vy: 0, wz: 0 };
  const ema = createInitiatorEma();
  let lastStepMs = 0;

  // ---- the state the referee reads ----------------------------------------
  const refState = {
    step: 0,
    contact: false,
    ema,
    [seatA]: { x: 0, y: 0, yaw: 0, tilt: 0, vx: 0, vy: 0 },
    [seatB]: { x: 0, y: 0, yaw: 0, tilt: 0, vx: 0, vy: 0 },
  };

  /**
   * Sample the sim into `refState`. Called on the PRE-`forward` frame, which is
   * where mjlab's TerminationManager reads (manager_based_rl_env.py:395-410) —
   * one physics substep stale, deliberately and consistently.
   *
   * The browser runs a single env whose origin is (0,0,0) (`env_origin_0` in
   * the shipped scene.xml), so world coordinates ARE the env-local coordinates
   * every `_local_xy` in terminations.py subtracts `env_origins` to get.
   */
  function sampleRefState() {
    for (const seat of g.seats) {
      const r = robotOf[seat];
      const base = sim.getBase(r);
      const s = refState[seat];
      s.x = base.pos[0];
      s.y = base.pos[1];
      s.yaw = sim.yaw(r);
      s.tilt = sim.tiltAngle(r);
      s.vx = base.linVelW[0];
      s.vy = base.linVelW[1];
    }
    refState.step = stepCount;
    refState.contact = sim.anyContactBetweenRobots();
    return refState;
  }

  // ---- reset ---------------------------------------------------------------
  /**
   * Put both robots on the opening pose exactly as a training reset does:
   * z = 0.32, default joints, zero velocity (`mdp.reset_root_state` +
   * `reset_joints` -> `sim.resetAll`), then re-seed every piece of persistent
   * state. `sim.resetAll` ends in a `mj_forward`, so the first observation of
   * the episode is already the post-forward frame mjlab would hand the policy.
   *
   * No jitter and no velocity kick. Training resets add both (touchdown.py:239,
   * recon/05:239) because they are a curriculum; a match wants a repeatable
   * opening, which is also what the demo renders use (a LOCKED spawn file,
   * recon/02:317).
   */
  function reset() {
    sim.resetAll(spawnSpec(g, spawnVariant));
    stepCount = 0;
    verdict = null;
    countdownLeftMs = Math.max(0, cfg.countdownMs);
    phase = countdownLeftMs > 0 ? PHASE.COUNTDOWN : PHASE.RUNNING;
    episodeIndex += 1;
    resetInitiatorEma(ema);
    cmd = { vx: 0, vy: 0, wz: 0 };
    lastRawCmd = { vx: 0, vy: 0, wz: 0 };
    for (const seat of g.seats) {
      lastAction[seat].fill(0);          // the obs `actions` block starts at 0
      // ctrl_action.py:107-122 — the integrator's target is seeded from the
      // POST-reset MEASURED joint pos, not from the default pose.
      if (typeof paths[seat].reset === 'function') {
        paths[seat].reset(sim.getJointPos(robotOf[seat]));
      }
      lastCtrl[seat] = sim.getCtrl(robotOf[seat]);
      filterInfo[seat] = null;
    }
    sampleRefState();
    return hud();
  }

  // ---- one control step ----------------------------------------------------
  /**
   * @param {number} [dtMs] wall-clock milliseconds since the last call. Only the
   *   countdown uses it; the match itself is driven in fixed 20 ms control steps
   *   so wall-clock jitter can never change the physics.
   * @returns {Object} hud()
   */
  function tick(dtMs = PHYS.controlDt * 1000) {
    if (phase === PHASE.COUNTDOWN) {
      // Display only: the sim is NOT stepped. Stepping here would let the legs
      // sag out of the z = 0.32 spawn every policy was trained to start from
      // (measured settle 0.32 -> 0.2303 m; tools/export_scene.py keyframe
      // report), i.e. it would silently change the initial condition.
      countdownLeftMs -= dtMs;
      if (countdownLeftMs <= 0) { countdownLeftMs = 0; phase = PHASE.RUNNING; }
      return hud();
    }
    if (phase !== PHASE.RUNNING) return hud();

    const t0 = now();

    // -- 1. observations (post-forward frame from the previous tick) ---------
    // `stepCount` here IS mjlab's episode_length_buf at the moment the obs are
    // built: 0 on the first tick, 499 on the last. The phase clock is
    // (episode_length_buf * 0.02) mod 0.6 (observations.py:301).
    const raw = input.read() || { vx: 0, vy: 0, wz: 0 };
    // Slewed by the CONTROL dt, not by wall time: the command is part of the
    // control step, so the same key-hold must accelerate identically on a
    // 30 fps laptop and a 144 Hz desktop.
    cmd = slewCommand(cmd, raw, PHYS.controlDt, cfg);
    lastRawCmd = raw;

    const obsPlayer = D.walkObs(sim, playerRobot, [cmd.vx, cmd.vy, cmd.wz], stepCount, lastAction[playerSeat]);
    const obsAi = D.gameObs(sim, aiRobot, playerRobot, obsDir(g, aiSeat), g.lineX, stepCount, lastAction[aiSeat]);
    assertDim('player walk obs', obsPlayer, OBS.walkDim);
    assertDim('AI game obs', obsAi, OBS.gameDim);

    // -- 2. policies (deterministic mean; no sampling, no clip) --------------
    const aPlayer = walkPolicy.forward(obsPlayer);
    const aAi = aiPolicy.forward(obsAi);
    assertDim('player action', aPlayer, OBS.actDim);
    assertDim('AI action', aAi, OBS.actDim);

    // -- 3. action paths -> absolute joint setpoints -------------------------
    const ctxPlayer = makeCtx(playerSeat, playerRobot, aiRobot);
    const ctxAi = makeCtx(aiSeat, aiRobot, playerRobot);
    const ctrlPlayer = paths[playerSeat].step(aPlayer, ctxPlayer);
    const ctrlAi = paths[aiSeat].step(aAi, ctxAi);
    assertDim('player ctrl', ctrlPlayer, OBS.actDim);
    assertDim('AI ctrl', ctrlAi, OBS.actDim);
    sim.setCtrl(playerRobot, ctrlPlayer);
    sim.setCtrl(aiRobot, ctrlAi);
    // Copy: both shipped action classes hand back a REUSED buffer
    // (app/action.js `_out`), and `state()` must not expose a view that the
    // next tick silently rewrites.
    lastCtrl[playerSeat] = Float64Array.from(ctrlPlayer);
    lastCtrl[aiSeat] = Float64Array.from(ctrlAi);
    filterInfo[playerSeat] = paths[playerSeat].info ? paths[playerSeat].info() : null;
    filterInfo[aiSeat] = paths[aiSeat].info ? paths[aiSeat].info() : null;

    // -- 4. four physics substeps, ctrl held --------------------------------
    sim.step(PHYS.decimation);

    // -- 5. the clock -------------------------------------------------------
    stepCount += 1;

    // -- 6. referee, on the PRE-forward frame -------------------------------
    const st = sampleRefState();
    verdict = D.judge(g, st);

    // -- 7. the reward layer: fold this step's closing speeds into the EMA.
    //       AFTER the verdict, unconditionally, every step (rewards.py:186-190).
    foldInitiatorEma(ema, st, g);

    // -- 8. one mj_forward, so the next tick's obs are fresh ----------------
    sim.forward();

    // -- 9. carry the RAW actions into the next obs -------------------------
    lastAction[playerSeat].set(aPlayer);
    lastAction[aiSeat].set(aAi);

    if (verdict) phase = PHASE.OVER;
    lastStepMs = now() - t0;
    return hud();
  }

  function makeCtx(seat, robot, oppRobot) {
    return {
      sim, game: key, seat, robot, opponentRobot: oppRobot,
      step: stepCount, controlDt: PHYS.controlDt,
      measured12: null,   // lazily read by a path that needs it
      get measuredJointPos() { return sim.getJointPos(robot); },
    };
  }

  // ---- controls ------------------------------------------------------------
  function pause() { if (phase === PHASE.RUNNING) phase = PHASE.PAUSED; return hud(); }
  function resume() { if (phase === PHASE.PAUSED) phase = PHASE.RUNNING; return hud(); }

  // ---- read-out ------------------------------------------------------------
  /** The full mutable state, for the renderer and for tests. */
  function state() {
    return {
      game: key, phase, step: stepCount, episodeIndex,
      playerSeat, aiSeat, playerRobot, aiRobot,
      verdict, cmd: { ...cmd }, rawCmd: { ...lastRawCmd },
      ema: { ...ema },
      seats: {
        [seatA]: { ...refState[seatA], robot: robotOf[seatA] },
        [seatB]: { ...refState[seatB], robot: robotOf[seatB] },
      },
      contact: refState.contact,
      lastAction: { [seatA]: lastAction[seatA], [seatB]: lastAction[seatB] },
      lastCtrl: { [seatA]: lastCtrl[seatA], [seatB]: lastCtrl[seatB] },
      lastStepMs,
    };
  }

  /** Everything the HUD draws. Cheap; safe to call every animation frame. */
  function hud() {
    const dist = {};
    for (const seat of g.seats) dist[seat] = lineDistance(g, seat, refState[seat].x);
    return {
      game: key,
      phase,
      step: stepCount,
      episodeSteps: g.episodeSteps,
      timeLeft: timeLeft(g, stepCount),
      countdown: phase === PHASE.COUNTDOWN
        ? { msLeft: countdownLeftMs, label: countdownLabel(countdownLeftMs, cfg) }
        : null,
      playerSeat, aiSeat,
      opponent: opponent ? { ...opponent } : null,
      /** Signed distance to each seat's own line — the SAME expression the
       *  policy reads as obs dim 54 (config.lineDistance). */
      lineDistance: dist,
      cmd: { ...cmd },
      contact: refState.contact,
      /** +1 = seat A is currently the one closing faster (would be at fault). */
      initiatorLean: ema.va >= ema.vd ? seatA : seatB,
      filter: { [seatA]: filterInfo[seatA], [seatB]: filterInfo[seatB] },
      verdict: verdict ? { ...verdict, text: explain(g, verdict) } : null,
      perf: { lastStepMs },
    };
  }

  function seatOf(robot) { return robot === robotOf[seatA] ? seatA : seatB; }

  function info() {
    return {
      game: key, seats: g.seats, playerSeat, aiSeat,
      robots: { ...robotOf },
      actionPaths: Object.fromEntries(g.seats.map((s) => [s, paths[s].kind])),
      obsDir: Object.fromEntries(g.seats.map((s) => [s, obsDir(g, s)])),
      gains: sim.getGains ? { [seatA]: sim.getGains(robotOf[seatA]) } : null,
      rules: g.rules,
      spawnVariant,
    };
  }

  function dispose() { /* the sim is owned by the caller */ }

  reset();
  return { tick, state, reset, pause, resume, hud, seatOf, info, dispose, PHASE };
}

// ---------------------------------------------------------------------------
// internals
// ---------------------------------------------------------------------------

/**
 * Adapt a frozen-interface action class (`WalkAffine` / `IncrementIntegrator`,
 * whose `step` takes only `a12`) to the action-path shape the loop uses. `ctx`
 * is forwarded anyway, harmlessly, so an injected shield path and a stock path
 * are called identically.
 */
function wrapPath(impl, kind) {
  return {
    kind,
    reset(measured12) {
      if (typeof impl.reset === 'function') impl.reset(measured12);
    },
    step(a12, ctx) { return impl.step(a12, ctx); },
    info() { return typeof impl.info === 'function' ? impl.info() : null; },
  };
}

function requirePolicy(what, p, dim) {
  if (!p || typeof p.forward !== 'function') {
    throw new Error(`createMatch: the ${what} policy is missing (need {forward, obsDim, actDim})`);
  }
  if (p.obsDim != null && p.obsDim !== dim) {
    throw new Error(`createMatch: the ${what} policy has obsDim ${p.obsDim}, expected ${dim}`);
  }
  if (p.actDim != null && p.actDim !== OBS.actDim) {
    throw new Error(`createMatch: the ${what} policy has actDim ${p.actDim}, expected ${OBS.actDim}`);
  }
}

function assertDim(what, v, n) {
  if (!v || v.length !== n) throw new Error(`${what}: expected ${n} values, got ${v && v.length}`);
}

/**
 * Rate-limit the keyboard's 0/1 steps into something the walker likes, then
 * clamp to its trained command box. Keys are bang-bang; feeding a step change
 * straight in makes the policy twitch (DESIGN.md §4).
 * Deadzone and slew rates are UX (config.MATCH); the BOX is not (config.CMD_BOX,
 * recon/04:271).
 */
function slewCommand(cur, target, dt, cfg) {
  const dz = cfg.cmdDeadzone;
  const t = clampCmd(
    Math.abs(target.vx ?? 0) < dz ? 0 : target.vx,
    Math.abs(target.vy ?? 0) < dz ? 0 : target.vy,
    Math.abs(target.wz ?? 0) < dz ? 0 : target.wz,
  );
  const s = cfg.cmdSlewPerSec;
  return {
    vx: approach(cur.vx, t.vx, s.vx * dt),
    vy: approach(cur.vy, t.vy, s.vy * dt),
    wz: approach(cur.wz, t.wz, s.wz * dt),
  };
}

function approach(a, b, maxDelta) {
  const d = b - a;
  if (d > maxDelta) return a + maxDelta;
  if (d < -maxDelta) return a - maxDelta;
  return b;
}

function countdownLabel(msLeft, cfg) {
  const labels = cfg.countdownLabels;
  if (msLeft <= 0) return labels[labels.length - 1];
  const nDigits = labels.length - 1;              // the last one is "GO"
  const per = cfg.countdownMs / nDigits;
  const i = Math.min(nDigits - 1, Math.floor((cfg.countdownMs - msLeft) / per));
  return labels[i];
}

const now = (typeof performance !== 'undefined' && performance.now)
  ? () => performance.now()
  : () => Date.now();

export { GAMES, PHYS, GAINS, ACTION };

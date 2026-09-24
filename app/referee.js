/**
 * app/referee.js — both rule sets, as pure functions over a plain state object.
 *
 * Nothing in here reads the sim. `app/match.js` samples the sim once per
 * control step (on the pre-`mj_forward` frame, exactly where mjlab evaluates
 * its TerminationManager) and hands the numbers over.
 *
 * ── THE RULE SET, WRITTEN DOWN ──────────────────────────────────────────────
 * Touchdown reference point: the BASE-LINK CENTRE (`mdp.crossed_line`,
 * terminations.py:37-61 / `sym.crossed_line_signed`, sym.py:47-66), not the
 * body hull. Past-line falls are STRICT: a topple is a topple. The full
 * argument, and why the eval harness's lenient "past-line referee" is the
 * wrong rule for this game, is the `REFEREE` block of `app/config.js`.
 * `config.REFEREE.hullTouchdown` flips to the hull rule (implemented and
 * tested below); `config.REFEREE.pastLineFall` accepts only 'strict' and this
 * module throws on anything else rather than half-applying the other referee.
 *
 * ── TERMINATION VOCABULARY ──────────────────────────────────────────────────
 * Exactly the live TerminationManager set (scene.json `rules.terminationTerms`):
 *   sym : touchdown, touchdown_def, trunk_contact,
 *         attacker_fell, attacker_oob, defender_fell, defender_oob, time_out
 *   asym: the same minus `touchdown_def`
 * `judge()` collapses the per-seat names to the six terminals the frozen
 * interface asks for and reports the exact term(s) in `detail`.
 *
 * Citation root: /home/ray/Disk_ext/Go2/Project/unitree_rl_mjlab/
 */

import {
  GAMES, PHYS, REFEREE, HULL, TERMINALS,
  gameCfg, gameKey, seats, goalDir, fieldBounds, hullCorners, lineXLocal,
} from './config.js';

// ---------------------------------------------------------------------------
// state shape
// ---------------------------------------------------------------------------

/**
 * @typedef {Object} SeatState
 * @property {number} x    env-local planar x of base_link's ORIGIN (world x; the
 *                         browser runs one env whose origin is (0,0,0) —
 *                         `env_origin_0` in the shipped scene.xml).
 *                         mjlab/entity/data.py:242 (`root_link_pos_w`)
 * @property {number} y    env-local planar y
 * @property {number} yaw  world yaw, atan2 of the body +x axis
 * @property {number} tilt acos(-projected_gravity_b[2]) in rad — the quantity
 *                         `mdp.bad_orientation` thresholds (terminations.py:24-32)
 * @property {number} vx   world-frame planar velocity of base_link, x
 * @property {number} vy   world-frame planar velocity of base_link, y
 */

/**
 * @typedef {Object} RefState
 * @property {number}  step     control steps COMPLETED (mjlab `episode_length_buf`
 *                              AFTER its increment). `time_out` fires at
 *                              step >= episodeSteps.
 * @property {boolean} contact  any geom of robot a touching any geom of robot b,
 *                              on the same pre-forward frame
 *                              (`sim.anyContactBetweenRobots()`; the v81 STRICT
 *                              rule, robots.py:194 + game_env_cfg.py:245-254)
 * @property {{va:number, vd:number}} [ema] the PRIOR steps' closing-speed EMA.
 *                              Required whenever `contact` can be true.
 * Plus one `SeatState` under each seat name of the game:
 *   sym : state.A, state.B      asym: state.attacker, state.defender
 */

// ---------------------------------------------------------------------------
// 1. the collision-initiator EMA  (stateful, but kept OUT of judge())
// ---------------------------------------------------------------------------

/**
 * `rewards.py:149-150, :174-196` (asym) and `sym.py:299-318` (sym) are the same
 * machinery, byte for byte:
 *
 *   n_hat = unit(pos_B - pos_A)                        planar, A -> B
 *   vA    = vel_A . n_hat        vB = vel_B . (-n_hat) closing speeds
 *   [verdict is read here, from the PRIOR EMA]
 *   emaX  = beta * emaX + (1 - beta) * vX              folded in AFTERWARDS
 *
 * beta = 0.5 ** (step_dt / 0.12) = 0.8908987181403393 at 50 Hz.
 * Reading the verdict before the fold-in is why a collision can never
 * contaminate its own attribution (`rewards.py` docstring at :146-150).
 */
export function createInitiatorEma() {
  return { va: 0, vd: 0 };
}

/** Reset at every episode start. rewards.py:251-255 / sym.py:357-361 */
export function resetInitiatorEma(ema) {
  ema.va = 0;
  ema.vd = 0;
  return ema;
}

/**
 * +1 = seat A drove the collision (A is the initiator and LOSES);
 * -1 = seat B drove it.
 * `ema_va >= ema_vd` -> +1, so an exact tie is charged to A
 * (rewards.py:174-176 uses `>=`; sym.py:299-301 likewise).
 * This is `env._contact_initiator_dir`.
 */
export function initiatorDir(ema) {
  if (!ema) throw new Error('initiatorDir: no EMA — trunk_contact cannot be attributed');
  return ema.va >= ema.vd ? 1 : -1;
}

/** The closing speeds of the current frame, before any EMA. */
export function closingSpeeds(a, b) {
  let nx = b.x - a.x, ny = b.y - a.y;
  const n = Math.hypot(nx, ny);
  // `n.norm().clamp_min(1e-6)` — rewards.py:180 / sym.py:308
  const d = n < 1e-6 ? 1e-6 : n;
  nx /= d; ny /= d;
  return {
    va: a.vx * nx + a.vy * ny,       // A closing on B
    vd: b.vx * -nx + b.vy * -ny,     // B closing on A
  };
}

/**
 * Fold this step's closing speeds in. Call AFTER `judge()`, every control step,
 * unconditionally — the reward term runs every step whether or not anything
 * terminated (rewards.py:186-190).
 */
export function foldInitiatorEma(ema, state, game) {
  const [sa, sb] = seats(game);
  const { va, vd } = closingSpeeds(state[sa], state[sb]);
  const beta = REFEREE.initiator.beta;
  ema.va = beta * ema.va + (1 - beta) * va;
  ema.vd = beta * ema.vd + (1 - beta) * vd;
  return ema;
}

// ---------------------------------------------------------------------------
// 2. the individual predicates
// ---------------------------------------------------------------------------

/** `mdp.bad_orientation`: acos(-g_b[2]) > 70 deg. terminations.py:24-32 */
export function fell(seatState, limitAngle = REFEREE.fallLimitAngle) {
  return seatState.tilt > limitAngle;
}

/** `mdp.base_oob`: the trunk CENTRE outside the rectangle. terminations.py:224-236 */
export function oob(game, seatState) {
  const b = fieldBounds(game);
  return Math.abs(seatState.x - b.cx) > b.halfX || Math.abs(seatState.y - b.cy) > b.halfY;
}

/**
 * Did this seat cross its own target line?
 *   centre rule: `dir * (x - cx) > line_x`, with `line_x` FIELD-local
 *     asym `mdp.crossed_line`        terminations.py:37-61 (dir fixed at +1,
 *                                    line_x 1.7 about cx 0.2, i.e. x > 1.9)
 *     sym  `sym.crossed_line_signed` sym.py:47-66 (line_x 1.9 about cx 0)
 *   hull rule (off by default): any corner of the canonical body rect past it
 *     asym `mdp.crossed_line_hull`        terminations.py:63-105
 *     sym  `sym.crossed_line_hull_signed` sym.py:69-90
 * `require_inside_half_y` is None in every shipped arm (touchdown.py:395,
 * sym_touchdown.py:213/:224), so there is no lateral gate.
 */
export function crossedLine(game, seat, seatState, useHull = REFEREE.hullTouchdown) {
  const g = gameCfg(game);
  const dir = goalDir(game, seat);
  if (dir === 0) return false;                 // this seat has no line (asym defender)
  const cx = g.center[0];
  const lx = lineXLocal(g);                    // field-local; 1.7 in asym, 1.9 in sym
  if (!useHull) return dir * (seatState.x - cx) > lx;
  const corners = hullCorners(seatState.x, seatState.y, seatState.yaw, HULL.half, HULL.fwdOffset);
  return corners.some(([qx]) => dir * (qx - cx) > lx);
}

/** `mdp.time_out`: episode_length_buf >= 500, a REAL terminal. touchdown.py:63 */
export function timedOut(game, step) {
  return step >= gameCfg(game).episodeSteps;
}

// ---------------------------------------------------------------------------
// 3. the full flag set — the live TerminationManager vocabulary
// ---------------------------------------------------------------------------

/**
 * Every termination term of this game, by its training name.
 * Keys are exactly `GAMES[game].terminationTerms`; nothing more, nothing less.
 */
export function terminationFlags(game, state) {
  assertRuleSet();
  const g = gameCfg(game);
  const key = gameKey(g);
  const [sa, sb] = g.seats;
  const A = requireSeat(state, sa), B = requireSeat(state, sb);

  const flags = {
    touchdown: crossedLine(g, sa, A),
    trunk_contact: REFEREE.trunkContactTerminal ? !!state.contact : false,
    attacker_fell: fell(A),
    attacker_oob: oob(g, A),
    defender_fell: fell(B),
    defender_oob: oob(g, B),
    time_out: timedOut(g, state.step),
  };
  if (key === 'sym') flags.touchdown_def = crossedLine(g, sb, B);

  // Contract check: the set we produce IS the set the sim builds.
  if (Object.keys(flags).length !== g.terminationTerms.length) {
    throw new Error(
      `referee produced ${Object.keys(flags).sort()} but the ${key} env builds ` +
      `${[...g.terminationTerms].sort()}`);
  }
  return flags;
}

// ---------------------------------------------------------------------------
// 4. the verdict
// ---------------------------------------------------------------------------

/**
 * SYMMETRIC payoff, seat-A perspective.
 * Transcribed from `sym.py:334-340` (and `verdict.classify_sym`, verdict.py:119-141):
 *
 *   score = 0
 *   score = where(bustA & ~bustB, -1, score)
 *   score = where(bustB & ~bustA, +1, score)
 *   score = where(tdA | tdB,       0, score)     // any crossing clears bust verdicts
 *   score = where(tdA & ~tdB,     +1, score)
 *   score = where(tdB & ~tdA,     -1, score)
 *   score = where(contact & ~anyTd, initiator_loses, score)
 *   // time_out never enters -> 0 = DRAW
 *
 * `decided = anyTd | bustA | bustB | trunk_contact | time_out` (verdict.py:141).
 */
function scoreSym(flags, ema) {
  const tdA = flags.touchdown, tdB = flags.touchdown_def;
  const bustA = flags.attacker_fell || flags.attacker_oob;
  const bustB = flags.defender_fell || flags.defender_oob;
  const anyTd = tdA || tdB;

  let score = 0;
  if (bustA && !bustB) score = -1;
  if (bustB && !bustA) score = 1;
  if (anyTd) score = 0;
  if (tdA && !tdB) score = 1;
  if (tdB && !tdA) score = -1;
  if (flags.trunk_contact && !anyTd) score = initiatorDir(ema) > 0 ? -1 : 1;

  const decided = anyTd || bustA || bustB || flags.trunk_contact || flags.time_out;
  return { score, decided };
}

/**
 * ASYMMETRIC payoff, attacker perspective.
 * Transcribed from `rewards.py:226-244` and `harness._classify` (harness.py:1044-1068):
 *
 *   win   = touchdown | defender_fell | defender_oob
 *   lose  = attacker_fell | attacker_oob | time_out
 *   score = where(lose, -1, 0); score = where(win, +1, score)   // win beats lose
 *   score = where(trunk_contact & ~touchdown, initiator_loses, score)
 *
 * Note the two deliberate asymmetries against the sym game:
 *   * `time_out` is a DEFENDER WIN, not a draw (touchdown.py:63).
 *   * a same-step double bust is an ATTACKER WIN, because `win` is applied last
 *     (rewards.py:140-141 spells this out: "win takes precedence on a same-step
 *     tie"). sym calls the same event a draw.
 */
function scoreAsym(flags, ema) {
  const td = flags.touchdown;
  const win = td || flags.defender_fell || flags.defender_oob;
  const lose = flags.attacker_fell || flags.attacker_oob || flags.time_out;

  let score = lose ? -1 : 0;
  if (win) score = 1;
  let decided = win || lose;
  if (flags.trunk_contact && !td) {
    score = initiatorDir(ema) > 0 ? -1 : 1;
    decided = true;
  }
  return { score, decided };
}

/**
 * Which terminal to NAME, when several fired on the same control step.
 * Precedence `config.REFEREE.terminalPrecedence`:
 *   touchdown > touchdown_def > trunk_contact > fell > oob > time_out
 * It is the SCORE precedence, so the label can never contradict the winner.
 */
function pickTerminal(game, flags) {
  const key = gameKey(game);
  const fired = [];
  // Both crossings are ONE bucket: a same-step double cross is a single event
  // (a DRAW, sym.py:337), so it must not be labelled as seat A's touchdown.
  const tdTerms = [];
  if (flags.touchdown) tdTerms.push('touchdown');
  if (key === 'sym' && flags.touchdown_def) tdTerms.push('touchdown_def');
  if (tdTerms.length) fired.push([tdTerms.length === 2 ? 'touchdown' : tdTerms[0], tdTerms]);
  if (flags.trunk_contact) fired.push(['trunk_contact', ['trunk_contact']]);
  const fellTerms = [];
  if (flags.attacker_fell) fellTerms.push('attacker_fell');
  if (flags.defender_fell) fellTerms.push('defender_fell');
  if (fellTerms.length) fired.push(['fell', fellTerms]);
  const oobTerms = [];
  if (flags.attacker_oob) oobTerms.push('attacker_oob');
  if (flags.defender_oob) oobTerms.push('defender_oob');
  if (oobTerms.length) fired.push(['oob', oobTerms]);
  if (flags.time_out) fired.push(['time_out', ['time_out']]);

  for (const name of REFEREE.terminalPrecedence) {
    const hit = fired.find(([n]) => n === name);
    if (hit) return { terminal: hit[0], detail: hit[1], fired: fired.map(([n]) => n) };
  }
  return null;
}

/**
 * THE REFEREE.
 *
 * @param {'sym'|'asym'|object} game
 * @param {RefState} state
 * @returns {null | {terminal:string, winner:'A'|'B'|'DRAW', score:number,
 *                   loser:'A'|'B'|null, winnerSeat:string|null, loserSeat:string|null,
 *                   detail:string[], fired:string[], initiatorDir:number|null,
 *                   flags:Object}}
 *   `null` while the episode is live.
 *   `winner` is 'A' | 'B' | 'DRAW' per the frozen interface; 'A' is seat A in
 *   sym and the ATTACKER in asym, 'B' is seat B / the DEFENDER.
 *   `winnerSeat`/`loserSeat` give the game's own seat names.
 */
export function judge(game, state) {
  const g = gameCfg(game);
  const key = gameKey(g);
  const flags = terminationFlags(g, state);
  const { score, decided } = key === 'sym'
    ? scoreSym(flags, state.ema)
    : scoreAsym(flags, state.ema);
  if (!decided) return null;

  const named = pickTerminal(g, flags);
  // `decided` is true, so at least one term fired.
  if (!named) throw new Error('referee: decided with no terminal — unreachable');

  const [sa, sb] = g.seats;
  const winner = score > 0 ? 'A' : score < 0 ? 'B' : 'DRAW';
  const loser = winner === 'DRAW' ? null : (winner === 'A' ? 'B' : 'A');
  return {
    terminal: named.terminal,
    winner,
    loser,
    winnerSeat: winner === 'DRAW' ? null : (winner === 'A' ? sa : sb),
    loserSeat: loser === null ? null : (loser === 'A' ? sa : sb),
    score,
    detail: named.detail,
    fired: named.fired,
    initiatorDir: flags.trunk_contact ? initiatorDir(state.ema) : null,
    flags,
  };
}

// ---------------------------------------------------------------------------
// 5. helpers for the HUD / the result card
// ---------------------------------------------------------------------------

/**
 * One line of plain English, seat names filled in. The terminal NAMES stay the
 * sim's own (`touchdown_def`, `trunk_contact`, ...) everywhere else so a web
 * verdict can be diffed against a training ledger.
 */
export function explain(game, verdict) {
  if (!verdict) return '';
  const g = gameCfg(game);
  const key = gameKey(g);
  const w = verdict.winnerSeat, l = verdict.loserSeat;
  switch (verdict.terminal) {
    case 'touchdown':
    case 'touchdown_def':
      return verdict.winner === 'DRAW'
        ? 'Both crossed on the same step — draw.'
        : `${w} crossed the line — touchdown.`;
    case 'trunk_contact': {
      const at = verdict.initiatorDir > 0 ? g.seats[0] : g.seats[1];
      return `Collision: ${at} was closing faster and is at fault. ${w} wins.`;
    }
    case 'fell':
      return verdict.winner === 'DRAW'
        ? 'Both robots went down — draw.'
        : `${l} fell. ${w} wins.`;
    case 'oob':
      return verdict.winner === 'DRAW'
        ? 'Both robots left the field — draw.'
        : `${l} left the field. ${w} wins.`;
    case 'time_out':
      return key === 'sym'
        ? 'Ten seconds, nobody crossed — draw.'
        : `Ten seconds. The defender held the line.`;
    default:
      return verdict.terminal;
  }
}

/** The clock the HUD shows, in seconds remaining. */
export function timeLeft(game, step) {
  return Math.max(0, (gameCfg(game).episodeSteps - step) * PHYS.controlDt);
}

// ---------------------------------------------------------------------------
// internals
// ---------------------------------------------------------------------------

function requireSeat(state, name) {
  const s = state[name];
  if (!s) throw new Error(`referee: state.${name} is missing`);
  for (const k of ['x', 'y', 'yaw', 'tilt', 'vx', 'vy']) {
    if (!Number.isFinite(s[k])) throw new Error(`referee: state.${name}.${k} is ${s[k]}`);
  }
  return s;
}

function assertRuleSet() {
  if (REFEREE.pastLineFall !== 'strict') {
    throw new Error(
      `config.REFEREE.pastLineFall = ${JSON.stringify(REFEREE.pastLineFall)}: ` +
      'only "strict" is implemented. The eval harness\'s void/amnesty rules ' +
      '(harness.py:264, terminations.py:107) apply to SHIELDED seats only and ' +
      'are deliberately not part of this game — see the REFEREE block of config.js.');
  }
}

export { TERMINALS, GAMES };

#!/usr/bin/env node
/**
 * tests/node_referee.mjs — the rules, the config tables and the control loop.
 *
 *   node tests/node_referee.mjs            # everything
 *   node tests/node_referee.mjs --quiet    # only failures + the summary
 *
 * Four sections:
 *   1. CONFIG      every table in app/config.js cross-checked against the
 *                  MEASURED assets/scene/<game>/scene.json and
 *                  assets/policies/manifest.json — so a drift between the
 *                  frozen interface and the exporters fails here, not in play.
 *   2. PREDICATES  crossed_line / base_oob / bad_orientation / time_out, plus
 *                  the collision-initiator EMA.
 *   3. VERDICTS    a table over EVERY terminal of BOTH games, including the
 *                  fault attribution, every draw case and the two places the
 *                  games deliberately disagree.
 *   4. MATCH LOOP  createMatch() driven by a scriptable fake sim: the mjlab step
 *                  order, the countdown, pause/resume/reset, the action-path
 *                  seam, and a synthetic 500-step episode ending in time_out.
 *
 * Section 4 needs app/obs.js and app/action.js. Until the runtime agent lands
 * them the test mirrors app/ into .cache/test-mirror/ and supplies RECORDING
 * STUBS, which is what section 4 actually wants to test anyway (the loop order
 * and the wiring, not the obs arithmetic). It says which mode it ran in.
 */

import { fileURLToPath } from 'node:url';
import path from 'node:path';
import fs from 'node:fs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, '..');
const QUIET = process.argv.includes('--quiet');

const CFG = await import(url(path.join(ROOT, 'app/config.js')));
const REF = await import(url(path.join(ROOT, 'app/referee.js')));

const {
  GAMES, PHYS, CMD_BOX, DEFAULT_JOINT_POS, JOINTS, GAINS, ACTION, OBS,
  SEAT_ROBOT, SEAT_GOAL_DIR, SEAT_OBS_DIR, REFEREE, HULL, TERMINALS,
  gameCfg, seatRobot, goalDir, obsDir, lineDistance, fieldBounds, spawnSpec,
  hullCorners, clampCmd, episodeSeconds, otherSeat,
} = CFG;
const {
  judge, terminationFlags, crossedLine, fell, oob, timedOut,
  createInitiatorEma, resetInitiatorEma, foldInitiatorEma, initiatorDir,
  closingSpeeds, explain, timeLeft,
} = REF;

// ---------------------------------------------------------------------------
// harness
// ---------------------------------------------------------------------------
let pass = 0, fail = 0, sections = 0;
const failures = [];

function section(name) { sections++; if (!QUIET) console.log(`\n=== ${name} ===`); }
function ok(name, cond, detail = '') {
  if (cond) { pass++; if (!QUIET) console.log(`  ok   ${name}${detail ? '  ' + detail : ''}`); }
  else { fail++; failures.push(name + (detail ? '  ' + detail : '')); console.log(`  FAIL ${name}${detail ? '  ' + detail : ''}`); }
}
function eq(name, got, want) {
  const same = JSON.stringify(got) === JSON.stringify(want);
  ok(name, same, same ? '' : `got ${JSON.stringify(got)} want ${JSON.stringify(want)}`);
}
function near(name, got, want, tol = 1e-9) {
  const d = Math.abs(got - want);
  ok(name, d <= tol, d <= tol ? '' : `got ${got} want ${want} (|d| ${d})`);
}
function arrNear(name, got, want, tol = 1e-6) {
  if (!got || got.length !== want.length) { ok(name, false, `length ${got && got.length} vs ${want.length}`); return; }
  let worst = 0, at = -1;
  for (let i = 0; i < want.length; i++) {
    const d = Math.abs(got[i] - want[i]);
    if (d > worst) { worst = d; at = i; }
  }
  ok(name, worst <= tol, worst <= tol ? `max|d| ${worst.toExponential(2)}` : `max|d| ${worst} at [${at}]`);
}
function throws(name, fn, needle = '') {
  try { fn(); ok(name, false, 'did not throw'); }
  catch (e) { ok(name, String(e.message).includes(needle), needle && !String(e.message).includes(needle) ? `message ${JSON.stringify(e.message)}` : ''); }
}
function url(p) { return 'file://' + p; }
function readJson(p) { return JSON.parse(fs.readFileSync(p, 'utf8')); }

// ---------------------------------------------------------------------------
// state builders
// ---------------------------------------------------------------------------
const SEAT0 = (o = {}) => ({ x: 0, y: 0, yaw: 0, tilt: 0, vx: 0, vy: 0, ...o });
const UPRIGHT = 0;
const TOPPLED = REFEREE.fallLimitAngle + 1e-6;     // just past 70 deg
const NEARLY = REFEREE.fallLimitAngle - 1e-6;      // just short of it

/** A live (nothing-fired) state for `game`, then patched. */
function st(game, patch = {}) {
  const g = gameCfg(game);
  const [a, b] = g.seats;
  const base = {
    step: 0,
    contact: false,
    ema: createInitiatorEma(),
    [a]: SEAT0({ x: g.spawnTraining[a].x, y: g.spawnTraining[a].y, yaw: g.spawnTraining[a].yaw }),
    [b]: SEAT0({ x: g.spawnTraining[b].x, y: g.spawnTraining[b].y, yaw: g.spawnTraining[b].yaw }),
  };
  const out = { ...base, ...patch };
  out[a] = { ...base[a], ...(patch[a] || {}) };
  out[b] = { ...base[b], ...(patch[b] || {}) };
  return out;
}
/** An EMA whose PRIOR value says `who` is closing faster. */
function emaFavouring(who) {
  return who === 'A' ? { va: 1.0, vd: 0.2 } : who === 'B' ? { va: 0.2, vd: 1.0 } : { va: 0.5, vd: 0.5 };
}

// ===========================================================================
// 1. CONFIG vs the shipped, MEASURED assets
// ===========================================================================
section('1. config.js vs the exported scene.json / manifest.json');

for (const game of ['sym', 'asym']) {
  const p = path.join(ROOT, `assets/scene/${game}/scene.json`);
  if (!fs.existsSync(p)) { ok(`${game}: scene.json present`, false, p); continue; }
  const s = readJson(p);
  const g = GAMES[game];

  eq(`${game}: field size`, g.field, s.field.size);
  eq(`${game}: field center`, g.center, s.field.center);
  near(`${game}: lineX`, g.lineX, s.field.lineX);
  near(`${game}: touchdownMargin`, g.touchdownMargin, s.field.touchdownMargin);
  eq(`${game}: episodeSteps`, g.episodeSteps, s.rules.episodeSteps);
  near(`${game}: episode seconds`, episodeSeconds(game), s.rules.episodeLengthS);

  // the OOB rectangle the referee uses IS the measured one
  const fb = fieldBounds(game);
  near(`${game}: oob halfX`, fb.halfX, s.field.oobHalfX);
  near(`${game}: oob halfY`, fb.halfY, s.field.oobHalfY);
  near(`${game}: oob xMin`, fb.xMin, s.field.oobX[0], 1e-9);
  near(`${game}: oob xMax`, fb.xMax, s.field.oobX[1], 1e-9);

  // the termination vocabulary is EXACTLY the live TerminationManager's
  eq(`${game}: terminationTerms`, [...g.terminationTerms].sort(), [...s.rules.terminationTerms].sort());

  // rule flags
  near(`${game}: fallLimitAngle`, REFEREE.fallLimitAngle, s.rules.fallLimitAngleRad);
  eq(`${game}: hullTouchdown`, REFEREE.hullTouchdown, s.rules.hullTouchdown);
  eq(`${game}: pastLineFall`, REFEREE.pastLineFall, s.rules.pastLineFall);
  eq(`${game}: trunkContactTerminal`, REFEREE.trunkContactTerminal, s.rules.trunkContactTerminal);
  eq(`${game}: safetyTerminal`, REFEREE.safetyTerminal, s.rules.safetyTerminal);
  eq(`${game}: terminalTimeout`, REFEREE.terminalTimeout, s.rules.terminalTimeout);
  near(`${game}: contactWinScale`, REFEREE.contactWinScale, s.rules.contactWinScale);

  // physics
  near(`${game}: dt`, PHYS.dt, s.physics.timestep);
  eq(`${game}: decimation`, PHYS.decimation, s.physics.decimation);
  near(`${game}: controlDt`, PHYS.controlDt, s.physics.controlDt);
  eq(`${game}: solver iterations`, PHYS.solver.iterations, s.physics.iterations);
  eq(`${game}: solver ls_iterations`, PHYS.solver.lsIterations, s.physics.lsIterations);
  eq(`${game}: solver ccd_iterations`, PHYS.solver.ccdIterations, s.physics.ccdIterations);
  eq(`${game}: integrator`, PHYS.solver.integrator, s.physics.integrator);
  eq(`${game}: gravity`, PHYS.solver.gravity, s.physics.gravity);

  // joints / gains
  eq(`${game}: joint order (scene.json spells it with the _joint suffix)`,
    CFG.JOINT_ORDER.map((j) => j + '_joint'), s.joints.order);
  arrNear(`${game}: defaultJointPos`, DEFAULT_JOINT_POS, s.joints.defaultJointPos, 0);
  arrNear(`${game}: soft limit lo`, JOINTS.softLo, s.joints.softLimitLo, 0);
  arrNear(`${game}: soft limit hi`, JOINTS.softHi, s.joints.softLimitHi, 0);
  arrNear(`${game}: hard limit lo`, JOINTS.hardLo, s.joints.hardLimitLo, 0);
  arrNear(`${game}: hard limit hi`, JOINTS.hardHi, s.joints.hardLimitHi, 0);
  near(`${game}: softLimitFactor`, JOINTS.softLimitFactor, s.joints.softLimitFactor);
  arrNear(`${game}: walk kp`, GAINS.walk.kp, s.joints.kp, 0);
  arrNear(`${game}: walk kd`, GAINS.walk.kd, s.joints.kd, 0);
  near(`${game}: spawn z`, JOINTS.spawnZ, s.joints.initialBaseHeight);

  // seat -> robot mapping
  for (const [r, row] of Object.entries(s.robots)) {
    eq(`${game}: robot ${r} is seat ${row.seat}`, seatRobot(game, uiSeat(game, row.seat)), r);
  }

  // the spawn the game opens from IS the one the exporter recorded
  for (const seat of g.seats) {
    const r = seatRobot(game, seat);
    const rec = s.spawn.config[r];
    near(`${game}: spawn ${seat}.x`, g.spawn[seat].x, rec.x, 1e-9);
    near(`${game}: spawn ${seat}.y`, g.spawn[seat].y, rec.y, 1e-9);
    near(`${game}: spawn ${seat}.yaw`, g.spawn[seat].yaw, rec.yaw, 1e-9);
  }
  // and the training nominal too
  for (const seat of g.seats) {
    const r = seatRobot(game, seat);
    const rec = s.spawn.trainingNominal[r];
    near(`${game}: training spawn ${seat}.x`, g.spawnTraining[seat].x, rec.x, 1e-9);
    near(`${game}: training spawn ${seat}.yaw`, g.spawnTraining[seat].yaw, rec.yaw, 1e-9);
  }

  // spawnSpec() hands physics.js robot-keyed poses at the training height
  const spec = spawnSpec(game);
  eq(`${game}: spawnSpec keys`, Object.keys(spec).sort(), ['a', 'b']);
  near(`${game}: spawnSpec z`, spec.a.z, s.joints.initialBaseHeight);
  arrNear(`${game}: spawnSpec joints`, spec.a.jointPos, s.joints.defaultJointPos, 0);

  // nobody is already past their line, or out, at the opening
  for (const seat of g.seats) {
    const s0 = SEAT0({ x: g.spawn[seat].x, y: g.spawn[seat].y, yaw: g.spawn[seat].yaw });
    ok(`${game}: ${seat} spawns short of its line`, !crossedLine(game, seat, s0));
    ok(`${game}: ${seat} spawns in bounds`, !oob(game, s0));
  }
  ok(`${game}: the opening is not already decided`, judge(game, st(game, {
    [g.seats[0]]: { x: g.spawn[g.seats[0]].x, y: g.spawn[g.seats[0]].y },
    [g.seats[1]]: { x: g.spawn[g.seats[1]].x, y: g.spawn[g.seats[1]].y },
  })) === null);
}

function uiSeat(game, scenarioSeat) {
  if (game === 'asym') return scenarioSeat;
  return scenarioSeat === 'A' || scenarioSeat === 'attacker' ? 'A' : 'B';
}

// manifest cross-check
{
  const mp = path.join(ROOT, 'assets/policies/manifest.json');
  if (!fs.existsSync(mp)) ok('manifest.json present', false, mp);
  else {
    const m = readJson(mp);
    near('manifest action_scale', ACTION.scale, m.action_scale);
    near('manifest increment scale', ACTION.increment.scale, m.increment_integrator.scale);
    near('manifest increment smoothing', ACTION.increment.smoothing, m.increment_integrator.smoothing);
    near('manifest increment inverse denom', ACTION.increment.inverseDenom, m.increment_integrator.inverse_denom);
    near('manifest norm eps', OBS.normEps, m.norm_eps);
    near('manifest control hz', PHYS.controlHz, m.control_hz);
    arrNear('manifest default joint pos', DEFAULT_JOINT_POS, m.default_joint_pos, 0);
    eq('manifest joint order', [...CFG.JOINT_ORDER], m.joint_order);
    eq('manifest walk obs dim', OBS.walkDim, m.player_walk.obs_dim);
    eq('manifest walk action path', 'walk_affine', m.player_walk.action_path);
    // every AI row is a 60-D increment-integrator policy
    const ai = m.policies.filter((p) => p.role === 'ai_opponent');
    ok('manifest: 15 AI policies', ai.length === 15, `got ${ai.length}`);
    ok('manifest: every AI policy is 60-D', ai.every((p) => p.obs_dim === OBS.gameDim));
    ok('manifest: every AI policy runs the increment integrator',
      ai.every((p) => p.action_path === 'increment_integrator'));
    // the sym roster is seat-B only -> the human can only take seat A
    eq('manifest: sym AI seats', Object.keys(m.games.sym.seats), [GAMES.sym.aiSeat]);
    eq('config: sym playable seats', GAMES.sym.playerSeats, ['A']);
    eq('manifest: asym AI seats', Object.keys(m.games.asym.seats).sort(), [...GAMES.asym.seats].sort());
    eq('config: asym playable seats', [...GAMES.asym.playerSeats].sort(), [...GAMES.asym.seats].sort());
  }
}

// ===========================================================================
// 2. PREDICATES
// ===========================================================================
section('2. predicates');

// --- the line observation, which is also the HUD number --------------------
// These four numbers are the ones recon/05:236 and sym_touchdown.py:146-147
// quote, and they are what proves SEAT_OBS_DIR (asym defender = +1, NOT -1).
near('asym attacker sees 3.3 m to the line at spawn', lineDistance('asym', 'attacker', -1.4), 3.3, 1e-12);
near('asym defender sees 1.15 m to the line at spawn', lineDistance('asym', 'defender', 0.75), 1.15, 1e-12);
near('sym A sees 3.3 m at the training spawn', lineDistance('sym', 'A', -1.4), 3.3, 1e-12);
near('sym B sees 3.3 m at the training spawn', lineDistance('sym', 'B', 1.4), 3.3, 1e-12);
eq('asym obsDir is +1 for BOTH seats', [obsDir('asym', 'attacker'), obsDir('asym', 'defender')], [1, 1]);
eq('sym obsDir is +1 / -1', [obsDir('sym', 'A'), obsDir('sym', 'B')], [1, -1]);
eq('asym goalDir: only the attacker scores', [goalDir('asym', 'attacker'), goalDir('asym', 'defender')], [1, 0]);
eq('sym goalDir is +1 / -1', [goalDir('sym', 'A'), goalDir('sym', 'B')], [1, -1]);

// --- crossed_line, base-link CENTRE, strict > -------------------------------
ok('sym A at x=1.91 has crossed', crossedLine('sym', 'A', SEAT0({ x: 1.91 })));
ok('sym A exactly ON the line has NOT crossed (strict >)', !crossedLine('sym', 'A', SEAT0({ x: 1.9 })));
ok('sym B at x=-1.91 has crossed', crossedLine('sym', 'B', SEAT0({ x: -1.91 })));
ok('sym B at x=+1.91 has NOT crossed (wrong end)', !crossedLine('sym', 'B', SEAT0({ x: 1.91 })));
ok('asym attacker at x=1.91 has crossed', crossedLine('asym', 'attacker', SEAT0({ x: 1.91 })));
ok('asym attacker at x=1.89 has not', !crossedLine('asym', 'attacker', SEAT0({ x: 1.89 })));
ok('asym defender never crosses (no line)', !crossedLine('asym', 'defender', SEAT0({ x: -2.3 })));

// --- the hull rule, implemented but OFF ------------------------------------
{
  // hull centre sits 0.05 m ahead of base_link; half extent 0.29 along body x,
  // so a yaw-0 robot's nose reaches x + 0.34.  collide.py:56-57
  const nose = SEAT0({ x: 1.9 - 0.34 + 1e-4, yaw: 0 });
  ok('hull: nose just over the line scores under the HULL rule',
    crossedLine('sym', 'A', nose, true));
  ok('hull: the same pose does NOT score under the CENTRE rule we ship',
    !crossedLine('sym', 'A', nose, false));
  ok('hull: and judge() uses the centre rule', judge('sym', st('sym', { A: nose })) === null);
  const c = hullCorners(0, 0, 0);
  arrNear('hull corner +x', c[0], [0.05 + 0.29, 0.193], 1e-12);
  arrNear('hull corner -x', c[3], [0.05 - 0.29, -0.193], 1e-12);
  eq('hull half extents', HULL.half, [0.29, 0.193]);
  near('hull forward offset', HULL.fwdOffset, 0.05);
}

// --- base_oob, trunk CENTRE, no walls --------------------------------------
ok('sym oob at |x| > 2.8', oob('sym', SEAT0({ x: 2.81 })));
ok('sym in bounds at |x| = 2.79', !oob('sym', SEAT0({ x: 2.79 })));
ok('sym oob at |y| > 1.5', oob('sym', SEAT0({ y: -1.51 })));
ok('asym oob at x > 2.8 (centre 0.2 + half 2.6)', oob('asym', SEAT0({ x: 2.81 })));
ok('asym in bounds at x = 2.79', !oob('asym', SEAT0({ x: 2.79 })));
ok('asym oob at x < -2.4', oob('asym', SEAT0({ x: -2.41 })));
ok('asym in bounds at x = -2.39', !oob('asym', SEAT0({ x: -2.39 })));

// --- bad_orientation --------------------------------------------------------
ok('fell at 70 deg + eps', fell(SEAT0({ tilt: TOPPLED })));
ok('upright at 70 deg - eps', !fell(SEAT0({ tilt: NEARLY })));
near('fall limit is exactly radians(70)', REFEREE.fallLimitAngle, (70 * Math.PI) / 180, 1e-15);

// --- time_out ---------------------------------------------------------------
ok('time_out at step 500', timedOut('sym', 500));
ok('no time_out at step 499', !timedOut('sym', 499));
near('10 s episode', episodeSeconds('asym'), 10.0, 1e-12);
near('timeLeft at step 0', timeLeft('sym', 0), 10.0, 1e-12);
near('timeLeft at step 250', timeLeft('sym', 250), 5.0, 1e-12);

// --- the collision-initiator EMA -------------------------------------------
{
  near('EMA beta = 0.5^(dt/halflife)', REFEREE.initiator.beta, Math.pow(0.5, 0.02 / 0.12), 0);
  near('EMA beta value', REFEREE.initiator.beta, 0.8908987181403393, 1e-15);

  // geometry: A at the origin driving +x, B parked 2 m ahead
  const cs = closingSpeeds(SEAT0({ x: 0, vx: 1 }), SEAT0({ x: 2 }));
  near('closing: A drives at B', cs.va, 1, 1e-12);
  near('closing: B is static', cs.vd, 0, 1e-12);
  // B retreating from A counts as NEGATIVE closing
  const cs2 = closingSpeeds(SEAT0({ x: 0 }), SEAT0({ x: 2, vx: 0.5 }));
  near('closing: B running away', cs2.vd, -0.5, 1e-12);
  // lateral motion at 90 deg closes nothing
  const cs3 = closingSpeeds(SEAT0({ x: 0, vy: 2 }), SEAT0({ x: 2 }));
  near('closing: pure lateral closes 0', cs3.va, 0, 1e-12);

  const e = createInitiatorEma();
  foldInitiatorEma(e, st('sym', { A: { x: 0, vx: 1 }, B: { x: 2 } }), 'sym');
  near('one fold from zero', e.va, 1 - REFEREE.initiator.beta, 1e-15);
  near('the other channel stays 0', e.vd, 0, 1e-15);
  resetInitiatorEma(e);
  eq('reset zeroes both channels', [e.va, e.vd], [0, 0]);

  eq('dir: A faster -> +1 (A at fault)', initiatorDir({ va: 1, vd: 0 }), 1);
  eq('dir: B faster -> -1', initiatorDir({ va: 0, vd: 1 }), -1);
  eq('dir: an exact tie is charged to A (>=)', initiatorDir({ va: 0.5, vd: 0.5 }), 1);
  throws('dir: no EMA is an error, not a guess', () => initiatorDir(null), 'cannot be attributed');
}

// --- the flag set is exactly the live one ----------------------------------
eq('sym flag names', Object.keys(terminationFlags('sym', st('sym'))).sort(),
  [...GAMES.sym.terminationTerms].sort());
eq('asym flag names', Object.keys(terminationFlags('asym', st('asym'))).sort(),
  [...GAMES.asym.terminationTerms].sort());
ok('asym has no touchdown_def', !('touchdown_def' in terminationFlags('asym', st('asym'))));

// --- guards ------------------------------------------------------------------
throws('judge rejects an unknown game', () => judge('freeform', st('sym')), 'unknown game');
throws('judge rejects a missing seat', () => judge('sym', { step: 0, contact: false, A: SEAT0() }), 'state.B is missing');
throws('judge rejects a NaN', () => judge('sym', st('sym', { A: { x: NaN } })), 'state.A.x is NaN');
throws('an unknown seat is an error', () => goalDir('sym', 'attacker'), 'not a seat');

// ===========================================================================
// 3. VERDICT TABLE — every terminal of both games
// ===========================================================================
section('3. verdicts');

const OUT = 3.0;          // |x| past every field's half-length
const PAST_A = 1.91;      // seat A / attacker just over its line
const PAST_B = -1.91;     // seat B just over its line

/** [name, game, patch, expected {terminal, winner} | null] */
const TABLE = [
  // ---- SYMMETRIC ----------------------------------------------------------
  ['sym live', 'sym', {}, null],
  ['sym A touchdown', 'sym', { A: { x: PAST_A } }, { terminal: 'touchdown', winner: 'A' }],
  ['sym B touchdown_def', 'sym', { B: { x: PAST_B } }, { terminal: 'touchdown_def', winner: 'B' }],
  ['sym double cross = DRAW', 'sym', { A: { x: PAST_A }, B: { x: PAST_B } }, { terminal: 'touchdown', winner: 'DRAW' }],
  ['sym A fell', 'sym', { A: { tilt: TOPPLED } }, { terminal: 'fell', winner: 'B' }],
  ['sym B fell', 'sym', { B: { tilt: TOPPLED } }, { terminal: 'fell', winner: 'A' }],
  ['sym double fall = DRAW', 'sym', { A: { tilt: TOPPLED }, B: { tilt: TOPPLED } }, { terminal: 'fell', winner: 'DRAW' }],
  ['sym A oob (x)', 'sym', { A: { x: -OUT } }, { terminal: 'oob', winner: 'B' }],
  ['sym A oob (y)', 'sym', { A: { y: 1.6 } }, { terminal: 'oob', winner: 'B' }],
  ['sym B oob', 'sym', { B: { x: OUT } }, { terminal: 'oob', winner: 'A' }],
  ['sym double oob = DRAW', 'sym', { A: { x: -OUT }, B: { x: OUT } }, { terminal: 'oob', winner: 'DRAW' }],
  ['sym mixed double bust = DRAW', 'sym', { A: { tilt: TOPPLED }, B: { x: OUT } }, { terminal: 'fell', winner: 'DRAW' }],
  ['sym touchdown clears a same-step bust', 'sym', { A: { x: PAST_A, tilt: TOPPLED } }, { terminal: 'touchdown', winner: 'A' }],
  ['sym time_out = DRAW', 'sym', { step: 500 }, { terminal: 'time_out', winner: 'DRAW' }],
  ['sym step 499 is still live', 'sym', { step: 499 }, null],
  ['sym a bust on the last step beats the clock', 'sym', { step: 500, A: { tilt: TOPPLED } }, { terminal: 'fell', winner: 'B' }],
  ['sym contact, A closing faster -> A at fault', 'sym', { contact: true, ema: emaFavouring('A') }, { terminal: 'trunk_contact', winner: 'B' }],
  ['sym contact, B closing faster -> B at fault', 'sym', { contact: true, ema: emaFavouring('B') }, { terminal: 'trunk_contact', winner: 'A' }],
  ['sym contact, exact tie -> A at fault', 'sym', { contact: true, ema: emaFavouring('tie') }, { terminal: 'trunk_contact', winner: 'B' }],
  ['sym touchdown beats contact', 'sym', { contact: true, ema: emaFavouring('A'), A: { x: PAST_A } }, { terminal: 'touchdown', winner: 'A' }],
  ['sym contact beats a fall', 'sym', { contact: true, ema: emaFavouring('B'), A: { tilt: TOPPLED } }, { terminal: 'trunk_contact', winner: 'A' }],
  ['sym contact beats the clock', 'sym', { step: 500, contact: true, ema: emaFavouring('A') }, { terminal: 'trunk_contact', winner: 'B' }],

  // ---- ASYMMETRIC ---------------------------------------------------------
  ['asym live', 'asym', {}, null],
  ['asym attacker touchdown', 'asym', { attacker: { x: PAST_A } }, { terminal: 'touchdown', winner: 'A' }],
  ['asym the defender cannot score', 'asym', { defender: { x: -1.91 } }, null],
  ['asym time_out = defender holds', 'asym', { step: 500 }, { terminal: 'time_out', winner: 'B' }],
  ['asym step 499 is still live', 'asym', { step: 499 }, null],
  ['asym attacker oob', 'asym', { attacker: { x: -OUT } }, { terminal: 'oob', winner: 'B' }],
  ['asym defender oob', 'asym', { defender: { x: OUT } }, { terminal: 'oob', winner: 'A' }],
  ['asym attacker fell', 'asym', { attacker: { tilt: TOPPLED } }, { terminal: 'fell', winner: 'B' }],
  ['asym defender fell', 'asym', { defender: { tilt: TOPPLED } }, { terminal: 'fell', winner: 'A' }],
  ['asym double fall = ATTACKER (win beats lose)', 'asym', { attacker: { tilt: TOPPLED }, defender: { tilt: TOPPLED } }, { terminal: 'fell', winner: 'A' }],
  ['asym touchdown clears a same-step fall', 'asym', { attacker: { x: PAST_A, tilt: TOPPLED } }, { terminal: 'touchdown', winner: 'A' }],
  ['asym contact, attacker closing faster', 'asym', { contact: true, ema: emaFavouring('A') }, { terminal: 'trunk_contact', winner: 'B' }],
  ['asym contact, defender closing faster', 'asym', { contact: true, ema: emaFavouring('B') }, { terminal: 'trunk_contact', winner: 'A' }],
  ['asym contact, exact tie -> attacker at fault', 'asym', { contact: true, ema: emaFavouring('tie') }, { terminal: 'trunk_contact', winner: 'B' }],
  ['asym touchdown beats contact', 'asym', { contact: true, ema: emaFavouring('A'), attacker: { x: PAST_A } }, { terminal: 'touchdown', winner: 'A' }],
  ['asym contact beats the clock', 'asym', { step: 500, contact: true, ema: emaFavouring('B') }, { terminal: 'trunk_contact', winner: 'A' }],
  ['asym contact beats an oob', 'asym', { contact: true, ema: emaFavouring('A'), defender: { x: OUT } }, { terminal: 'trunk_contact', winner: 'B' }],
];

for (const [name, game, patch, want] of TABLE) {
  const v = judge(game, st(game, patch));
  if (want === null) { ok(name, v === null, v ? `got ${v.terminal}/${v.winner}` : ''); continue; }
  const got = v ? { terminal: v.terminal, winner: v.winner } : null;
  eq(name, got, want);
}

// the two games really do disagree, on the same physical event
{
  const bust = { tilt: TOPPLED };
  const s = judge('sym', st('sym', { A: bust, B: bust }));
  const a = judge('asym', st('asym', { attacker: bust, defender: bust }));
  ok('double bust: sym DRAW vs asym ATTACKER', s.winner === 'DRAW' && a.winner === 'A',
    `sym ${s.winner}, asym ${a.winner}`);
  const so = judge('sym', st('sym', { step: 500 }));
  const ao = judge('asym', st('asym', { step: 500 }));
  ok('time_out: sym DRAW vs asym DEFENDER', so.winner === 'DRAW' && ao.winner === 'B',
    `sym ${so.winner}, asym ${ao.winner}`);
}

// the verdict carries the sim's own term names and the seat names
{
  const v = judge('sym', st('sym', { A: { tilt: TOPPLED } }));
  eq('detail names the training term', v.detail, ['attacker_fell']);
  eq('winnerSeat is a seat name', [v.winnerSeat, v.loserSeat], ['B', 'A']);
  near('score is the A-perspective payoff', v.score, -1);
  const v2 = judge('asym', st('asym', { defender: { tilt: TOPPLED } }));
  eq('asym detail', v2.detail, ['defender_fell']);
  eq('asym winnerSeat', [v2.winnerSeat, v2.loserSeat], ['attacker', 'defender']);
  const v3 = judge('sym', st('sym', { A: { x: PAST_A }, B: { x: PAST_B } }));
  eq('double cross detail names both terms', v3.detail, ['touchdown', 'touchdown_def']);
  const v4 = judge('sym', st('sym', { contact: true, ema: emaFavouring('A') }));
  eq('contact verdict exposes the initiator direction', v4.initiatorDir, 1);
  ok('explain() says who is at fault', explain('sym', v4).includes('A was closing faster'), explain('sym', v4));
  ok('explain() handles the double cross', explain('sym', v3).includes('draw'), explain('sym', v3));
  ok('explain() handles a sym timeout', explain('sym', so_()).includes('draw'));
  ok('explain() handles an asym timeout', explain('asym', ao_()).includes('defender held'));
  eq('every terminal we can emit is in TERMINALS',
    [...new Set(TABLE.map(([, g2, p]) => (judge(g2, st(g2, p)) || {}).terminal).filter(Boolean))]
      .filter((t) => !TERMINALS.includes(t)), []);
}
function so_() { return judge('sym', st('sym', { step: 500 })); }
function ao_() { return judge('asym', st('asym', { step: 500 })); }

// every terminal in the frozen interface is REACHABLE in at least one game
{
  const reached = new Set(TABLE.map(([, g2, p]) => (judge(g2, st(g2, p)) || {}).terminal).filter(Boolean));
  for (const t of TERMINALS) ok(`terminal '${t}' is covered by the table`, reached.has(t));
}

// the verdict is stable under the EMA fold ORDER (the trap rewards.py:146-150 warns about)
{
  // A is parked; B rams it at 3 m/s. If we folded BEFORE judging, this step's
  // own impact velocity would decide the verdict; folding after, the PRIOR
  // (all-zero) EMA decides it, and a zero tie is charged to A.
  const s = st('sym', { contact: true, A: { x: 0, vx: 0 }, B: { x: 0.4, vx: -3 } });
  const before = judge('sym', s);
  eq('prior-EMA verdict on the very first step is the tie rule', before.winner, 'B');
  foldInitiatorEma(s.ema, s, 'sym');
  const after = judge('sym', s);
  eq('once B\'s approach is in the EMA, B is at fault', after.winner, 'A');
}

// a synthetic 500-step episode, refereed step by step, ends in time_out
{
  const s = st('sym');
  let v = null, ended = -1;
  for (let k = 1; k <= 500; k++) {
    // both robots jog toward each other but nobody reaches a line
    s.A.x = -2.1431 + 0.002 * k;  s.A.vx = 0.1;
    s.B.x = 2.1775 - 0.002 * k;   s.B.vx = -0.1;
    s.step = k;
    v = judge('sym', s);
    foldInitiatorEma(s.ema, s, 'sym');
    if (v) { ended = k; break; }
  }
  ok('500-step episode: nothing fires early', ended === 500, `ended at ${ended}`);
  eq('500-step episode: terminal', v && v.terminal, 'time_out');
  eq('500-step episode: winner', v && v.winner, 'DRAW');
  ok('500-step episode: A travelled 1.0 m and is still short of the line',
    Math.abs(s.A.x - (-1.1431)) < 1e-9 && !crossedLine('sym', 'A', s.A));
}

// ===========================================================================
// 4. THE MATCH LOOP
// ===========================================================================
section('4. createMatch — the 50 Hz loop');

const REAL_DEPS = fs.existsSync(path.join(ROOT, 'app/obs.js'))
  && fs.existsSync(path.join(ROOT, 'app/action.js'));
const APP_DIR = REAL_DEPS ? path.join(ROOT, 'app') : mirrorAppWithStubs();
console.log(`  [match loop uses ${REAL_DEPS ? 'the REAL app/obs.js + app/action.js' : 'RECORDING STUBS for obs.js/action.js (not yet written) — mirror at ' + path.relative(ROOT, APP_DIR)}]`);

/**
 * Mirror app/{config,referee,match}.js into .cache/test-mirror/ next to stub
 * obs.js / action.js, so the loop can be exercised before the runtime agent
 * lands the real ones. The stubs RECORD their arguments; they compute nothing
 * that this section asserts about physics.
 */
function mirrorAppWithStubs() {
  const dir = path.join(ROOT, '.cache/test-mirror');
  fs.rmSync(dir, { recursive: true, force: true });
  fs.mkdirSync(dir, { recursive: true });
  for (const f of ['config.js', 'referee.js', 'match.js']) {
    fs.copyFileSync(path.join(ROOT, 'app', f), path.join(dir, f));
  }
  fs.writeFileSync(path.join(dir, 'obs.js'), `
// TEST STUB — records its arguments, returns a correctly sized vector.
export const calls = [];
export function walkObs(sim, robot, cmd3, stepCount, lastAction12) {
  calls.push({ fn: 'walkObs', robot, cmd3: Array.from(cmd3), stepCount,
               lastAction: Array.from(lastAction12), t: sim.__clock() });
  return new Float32Array(47);
}
export function gameObs(sim, me, opp, dir, lineX, stepCount, lastAction12) {
  calls.push({ fn: 'gameObs', me, opp, dir, lineX, stepCount,
               lastAction: Array.from(lastAction12), t: sim.__clock() });
  return new Float32Array(60);
}
`);
  fs.writeFileSync(path.join(dir, 'action.js'), `
// TEST STUB — records resets and returns a deterministic 12-vector.
export const resets = [];
export class WalkAffine {
  constructor(defaultJointPos) { this.def = Array.from(defaultJointPos); this.last = null; }
  reset(measured12) { resets.push({ kind: 'walk_affine', measured: measured12 && Array.from(measured12) }); }
  step(a12) {
    const out = new Float64Array(12);
    for (let i = 0; i < 12; i++) out[i] = this.def[i] + 0.25 * a12[i];
    this.last = out; return out;
  }
}
export class IncrementIntegrator {
  constructor(defaultJointPos, lo, hi) { this.def = Array.from(defaultJointPos); this.lo = lo; this.hi = hi; this.target = Array.from(defaultJointPos); }
  reset(measured12) { resets.push({ kind: 'increment_integrator', measured: measured12 && Array.from(measured12) });
    if (measured12) this.target = Array.from(measured12); }
  step(a12) {
    const out = new Float64Array(12);
    for (let i = 0; i < 12; i++) out[i] = this.target[i] + 0.001 * a12[i];
    this.target = Array.from(out); return out;
  }
}
`);
  return dir;
}

const MATCH = await import(url(path.join(APP_DIR, 'match.js')));
const OBS_MOD = await import(url(path.join(APP_DIR, 'obs.js')));
const ACT_MOD = await import(url(path.join(APP_DIR, 'action.js')));

/**
 * Recording wrappers around WHATEVER obs.js / action.js are in play (the real
 * ones, or the stubs above). Handed to createMatch through its `deps` hook, so
 * these assertions test the LOOP's wiring — which robot, which direction, which
 * stepCount, which lastAction — independently of the obs arithmetic.
 */
function recorder() {
  const calls = [], resets = [];
  const deps = {
    walkObs(sim, robot, cmd3, stepCount, lastAction12) {
      calls.push({ fn: 'walkObs', robot, cmd3: Array.from(cmd3), stepCount, lastAction: Array.from(lastAction12) });
      return OBS_MOD.walkObs(sim, robot, cmd3, stepCount, lastAction12);
    },
    gameObs(sim, me, opp, dir, lineX, stepCount, lastAction12) {
      calls.push({ fn: 'gameObs', me, opp, dir, lineX, stepCount, lastAction: Array.from(lastAction12) });
      return OBS_MOD.gameObs(sim, me, opp, dir, lineX, stepCount, lastAction12);
    },
    WalkAffine: class {
      constructor(...a) { this.i = new ACT_MOD.WalkAffine(...a); }
      reset(m) { resets.push({ kind: 'walk_affine', measured: m && Array.from(m) }); return this.i.reset(m); }
      step(a) { return this.i.step(a); }
    },
    IncrementIntegrator: class {
      constructor(...a) { this.i = new ACT_MOD.IncrementIntegrator(...a); this.ctor = a; }
      reset(m) { resets.push({ kind: 'increment_integrator', measured: m && Array.from(m) }); return this.i.reset(m); }
      step(a) { return this.i.step(a); }
    },
  };
  return { deps, calls, resets };
}

/**
 * A scriptable kinematic stand-in for app/physics.js: the exact surface
 * createMatch touches, plus a `script` hook so a test can place the robots.
 * Contact is whatever the script says. `log` records the call ORDER, which is
 * what section 4 is really testing.
 */
function fakeSim(game, script = null) {
  const g = GAMES[game];
  const rs = { a: mk(), b: mk() };
  const log = [];
  let clock = 0;
  function mk() {
    return {
      x: 0, y: 0, yaw: 0, tilt: 0, vx: 0, vy: 0, z: JOINTS.spawnZ,
      q: Float64Array.from(DEFAULT_JOINT_POS), ctrl: Float64Array.from(DEFAULT_JOINT_POS),
    };
  }
  let contact = false;
  const jointOverride = { a: null, b: null };
  const sim = {
    defaultJointPos: DEFAULT_JOINT_POS.slice(),
    /** Consistent with tiltAngle(): acos(-g[2]) == tilt, a pure roll. */
    projectedGravity(r) {
      const t = rs[r].tilt;
      return new Float64Array([0, Math.sin(t), -Math.cos(t)]);
    },
    __clock: () => clock,
    __log: log,
    __robots: rs,
    __setContact(v) { contact = v; },
    /** Force the post-reset MEASURED joint pose of one robot (a spawn jitter). */
    __setSpawnJoints(r, arr) { jointOverride[r] = Float64Array.from(arr); },
    resetAll(spec) {
      log.push('resetAll');
      for (const r of ['a', 'b']) {
        const p = spec[r];
        rs[r] = mk();
        rs[r].x = p.x; rs[r].y = p.y; rs[r].yaw = p.yaw; rs[r].z = p.z;
        rs[r].q = Float64Array.from(jointOverride[r] || p.jointPos);
        // physics.js seeds ctrl from the post-reset measured joint pos
        rs[r].ctrl = Float64Array.from(rs[r].q);
      }
      contact = false; clock = 0;
    },
    getJointPos(r) { return Float64Array.from(rs[r].q); },
    getJointVel(r) { return new Float64Array(12); },
    getCtrl(r) { return Float64Array.from(rs[r].ctrl); },
    setCtrl(r, arr) { log.push('setCtrl:' + r); rs[r].ctrl = Float64Array.from(arr); },
    getBase(r) {
      return { pos: [rs[r].x, rs[r].y, rs[r].z], quat: [1, 0, 0, 0],
               linVelW: [rs[r].vx, rs[r].vy, 0], angVelB: [0, 0, 0] };
    },
    yaw(r) { return rs[r].yaw; },
    tiltAngle(r) { return rs[r].tilt; },
    anyContactBetweenRobots() { log.push('contact?'); return contact; },
    step(n) {
      log.push('step:' + n);
      clock += 1;
      if (script) script(rs, clock, sim);
      for (const r of ['a', 'b']) { rs[r].x += rs[r].vx * PHYS.controlDt; rs[r].y += rs[r].vy * PHYS.controlDt; }
    },
    forward() { log.push('forward'); },
    getGains() { return { kp: GAINS.walk.kp.slice(), kd: GAINS.walk.kd.slice() }; },
  };
  return sim;
}

const constPolicy = (obsDim, value) => ({
  obsDim, actDim: 12, calls: 0,
  forward(obs) { this.calls++; return Float32Array.from({ length: 12 }, () => value); },
});
const constInput = (c = { vx: 0, vy: 0, wz: 0 }) => ({ read: () => c });

// --- 4a. the mjlab step order ----------------------------------------------
{
  const sim = fakeSim('sym');
  const m = MATCH.createMatch({
    sim, game: 'sym', playerSeat: 'A',
    policies: { walk: constPolicy(47, 0.1), ai: constPolicy(60, -0.1) },
    input: constInput(),
  });
  eq('reset() leaves the match in countdown', m.hud().phase, 'countdown');
  const beforeTicks = sim.__log.length;
  m.tick(3100);                       // burn the countdown, sim untouched
  eq('the countdown does not step the sim', sim.__log.length, beforeTicks);
  eq('countdown -> running', m.hud().phase, 'running');

  sim.__log.length = 0;
  m.tick();
  eq('one control step, in mjlab order',
    sim.__log,
    ['setCtrl:a', 'setCtrl:b', 'step:4', 'contact?', 'forward']);
  eq('exactly 4 physics substeps per control step', PHYS.decimation, 4);
  eq('the clock advanced one control step', m.hud().step, 1);
}

// --- 4b. what the obs and the action paths actually receive ----------------
{
  const rec = recorder();
  const sim = fakeSim('asym');
  const m = MATCH.createMatch({
    sim, game: 'asym', playerSeat: 'defender', deps: rec.deps,
    policies: { walk: constPolicy(47, 0), ai: constPolicy(60, 0) },
    input: constInput(),
  });
  eq('reset seeds BOTH action paths', rec.resets.map((r) => r.kind).sort(),
    ['increment_integrator', 'walk_affine']);
  arrNear('the integrator is seeded from the MEASURED joint pos, not a constant',
    rec.resets.find((r) => r.kind === 'increment_integrator').measured, DEFAULT_JOINT_POS, 0);

  rec.calls.length = 0;
  m.tick(3100); m.tick();
  const walk = rec.calls.find((c) => c.fn === 'walkObs');
  const gm = rec.calls.find((c) => c.fn === 'gameObs');
  eq('the human drives the seat they picked', walk.robot, seatRobot('asym', 'defender'));
  eq('the AI takes the other seat', gm.me, seatRobot('asym', 'attacker'));
  eq('gameObs gets the opponent robot too', gm.opp, seatRobot('asym', 'defender'));
  eq('gameObs dir for the asym attacker', gm.dir, 1);
  near('gameObs lineX is the ENV-local 1.9', gm.lineX, 1.9);
  eq('stepCount on the first tick is 0 (= episode_length_buf)', walk.stepCount, 0);
  eq('lastAction starts at zero', walk.lastAction, new Array(12).fill(0));

  rec.calls.length = 0;
  m.tick();
  eq('stepCount on the second tick is 1', rec.calls[0].stepCount, 1);
}
{
  // The ASYM DEFENDER's obs direction is +1 — the whole point of SEAT_OBS_DIR.
  const rec = recorder();
  const m = MATCH.createMatch({
    sim: fakeSim('asym'), game: 'asym', playerSeat: 'attacker', deps: rec.deps,
    policies: { walk: constPolicy(47, 0), ai: constPolicy(60, 0) },
    input: constInput(),
  });
  m.tick(3100); m.tick();
  eq('gameObs dir for the asym DEFENDER is +1, not -1',
    rec.calls.find((c) => c.fn === 'gameObs').dir, 1);
}
{
  // sym seat B is the one seat in either game that reads -1
  const rec = recorder();
  const m = MATCH.createMatch({
    sim: fakeSim('sym'), game: 'sym', playerSeat: 'A', deps: rec.deps,
    policies: { walk: constPolicy(47, 0), ai: constPolicy(60, 0) },
    input: constInput(),
  });
  m.tick(3100); m.tick();
  eq('gameObs dir for sym seat B is -1', rec.calls.find((c) => c.fn === 'gameObs').dir, -1);
}
{
  // the RAW action (pre-affine) is what gets carried into the next obs
  const rec = recorder();
  const m = MATCH.createMatch({
    sim: fakeSim('sym'), game: 'sym', playerSeat: 'A', deps: rec.deps,
    policies: { walk: constPolicy(47, 0.37), ai: constPolicy(60, -0.42) },
    input: constInput(),
  });
  m.tick(3100); m.tick(); rec.calls.length = 0; m.tick();
  arrNear('walkObs carries the previous RAW action',
    rec.calls.find((c) => c.fn === 'walkObs').lastAction, new Array(12).fill(0.37), 1e-7);
  arrNear('gameObs carries the previous RAW action',
    rec.calls.find((c) => c.fn === 'gameObs').lastAction, new Array(12).fill(-0.42), 1e-7);
}
{
  // The two action paths are genuinely different maths, and the AI's one is
  // the one people get wrong. Spawn the AI 0.2 rad off the default pose and
  // step ONE zero action through both.
  //
  //   walk affine (the human):  ctrl = default + 0.25*0 = default, instantly.
  //   integrator  (the AI):     target0 = the MEASURED pose = default + 0.2
  //                             q_des   = default
  //                             u       = clamp(-0.2/0.15, +/-1)  = -1
  //                             inc     = clamp(-0.5, +/-0.5)     = -0.5
  //                             clamped = target0 - 0.5 (inside the soft limits)
  //                             target1 = target0 + 0.3*(-0.5)    = target0 - 0.15
  //                             ctrl    = target1 = default + 0.05
  // ctrl_action.py:179-191 + the reset seeding at :107-122. An implementation
  // that wrote `ctrl = default + 0.25*a` would land on `default` here, which
  // is what "marches in place" looks like from the outside.
  const sim = fakeSim('sym');
  sim.__setSpawnJoints('b', DEFAULT_JOINT_POS.map((v) => v + 0.2));
  const m = MATCH.createMatch({
    sim, game: 'sym', playerSeat: 'A',
    policies: { walk: constPolicy(47, 0), ai: constPolicy(60, 0) },
    input: constInput(),
  });
  arrNear('the AI really did spawn off the default pose',
    sim.getJointPos('b'), DEFAULT_JOINT_POS.map((v) => v + 0.2), 1e-15);
  m.tick(3100); m.tick();
  arrNear('the integrator eases from the MEASURED pose by alpha*inc = 0.15 rad',
    sim.__robots.b.ctrl, DEFAULT_JOINT_POS.map((v) => v + 0.05), 1e-12);
  ok('and it is NOT the naive `default + 0.25*a`',
    Math.abs(sim.__robots.b.ctrl[0] - DEFAULT_JOINT_POS[0]) > 0.04);
  arrNear('the walk affine puts the human at the DEFAULT pose in one step',
    sim.__robots.a.ctrl, DEFAULT_JOINT_POS, 1e-12);
}

// --- 4c. the referee is wired to the loop ----------------------------------
{
  // scripted touchdown: seat A teleports past the line on control step 137
  const sim = fakeSim('sym', (rs, k) => { if (k === 137) rs.a.x = 1.95; });
  const m = MATCH.createMatch({
    sim, game: 'sym', playerSeat: 'A',
    policies: { walk: constPolicy(47, 0), ai: constPolicy(60, 0) },
    input: constInput(),
  });
  m.tick(3100);
  let k = 0;
  while (m.hud().phase === 'running' && k < 600) { m.tick(); k++; }
  eq('the loop ends the episode on the crossing step', m.hud().step, 137);
  eq('terminal', m.hud().verdict.terminal, 'touchdown');
  eq('winner', m.hud().verdict.winner, 'A');
  eq('phase', m.hud().phase, 'over');
  ok('hud carries the plain-English verdict', /touchdown/i.test(m.hud().verdict.text), m.hud().verdict.text);
  // a finished match ignores further ticks
  const s0 = m.hud().step;
  m.tick(); m.tick();
  eq('an ended match does not keep stepping', m.hud().step, s0);
}

{
  // scripted collision: B rams A, so B is at fault and A wins
  const sim = fakeSim('sym', (rs, k, s) => {
    rs.b.vx = -2.0;                       // B drives at A from the start
    if (k === 40) s.__setContact(true);
  });
  const m = MATCH.createMatch({
    sim, game: 'sym', playerSeat: 'A',
    policies: { walk: constPolicy(47, 0), ai: constPolicy(60, 0) },
    input: constInput(),
  });
  m.tick(3100);
  let k = 0;
  while (m.hud().phase === 'running' && k < 600) { m.tick(); k++; }
  eq('collision ends the episode on the contact step', m.hud().step, 40);
  eq('collision terminal', m.hud().verdict.terminal, 'trunk_contact');
  eq('the rammer loses', m.hud().verdict.winner, 'A');
  eq('initiator direction: seat B drove it', m.hud().verdict.initiatorDir, -1);
  ok('explain names the guilty seat', m.hud().verdict.text.includes('B was closing faster'),
    m.hud().verdict.text);
}

{
  // the full 500-step clock, both games
  for (const [game, seat, wantWinner] of [['sym', 'A', 'DRAW'], ['asym', 'attacker', 'B'], ['asym', 'defender', 'B']]) {
    const sim = fakeSim(game);
    const m = MATCH.createMatch({
      sim, game, playerSeat: seat,
      policies: { walk: constPolicy(47, 0), ai: constPolicy(60, 0) },
      input: constInput(),
    });
    m.tick(3100);
    let k = 0;
    while (m.hud().phase === 'running' && k < 700) { m.tick(); k++; }
    eq(`${game}/${seat}: 500 control steps`, m.hud().step, GAMES[game].episodeSteps);
    eq(`${game}/${seat}: terminal`, m.hud().verdict.terminal, 'time_out');
    eq(`${game}/${seat}: winner`, m.hud().verdict.winner, wantWinner);
    near(`${game}/${seat}: the clock ran out`, m.hud().timeLeft, 0, 1e-12);
    eq(`${game}/${seat}: sim stepped 500 times`, sim.__log.filter((l) => l === 'step:4').length, 500);
  }
}

// --- 4d. pause / resume / reset --------------------------------------------
{
  const sim = fakeSim('asym');
  const m = MATCH.createMatch({
    sim, game: 'asym', playerSeat: 'attacker',
    policies: { walk: constPolicy(47, 0), ai: constPolicy(60, 0) },
    input: constInput(),
  });
  m.tick(3100);
  for (let i = 0; i < 10; i++) m.tick();
  eq('10 steps in', m.hud().step, 10);
  m.pause();
  eq('paused', m.hud().phase, 'paused');
  const n = sim.__log.length;
  m.tick(); m.tick();
  eq('a paused match does not step the sim', sim.__log.length, n);
  eq('a paused match does not advance the clock', m.hud().step, 10);
  m.resume();
  m.tick();
  eq('resumed', m.hud().step, 11);
  m.reset();
  eq('reset zeroes the clock', m.hud().step, 0);
  eq('reset returns to the countdown', m.hud().phase, 'countdown');
  eq('reset clears the verdict', m.hud().verdict, null);
  eq('reset zeroes the initiator EMA', [m.state().ema.va, m.state().ema.vd], [0, 0]);
  eq('reset re-places the robots', [sim.__robots.a.x, sim.__robots.b.x],
    [GAMES.asym.spawn.attacker.x, GAMES.asym.spawn.defender.x]);
}

// --- 4e. HUD -----------------------------------------------------------------
{
  const sim = fakeSim('sym');
  const m = MATCH.createMatch({
    sim, game: 'sym', playerSeat: 'A',
    opponent: { name: 'sym_s2c_B', method: 's2c', display: 'S2C' },
    policies: { walk: constPolicy(47, 0), ai: constPolicy(60, 0) },
    input: constInput({ vx: 3.0, vy: 0, wz: 0 }),
  });
  const h0 = m.hud();
  eq('hud names the opponent', h0.opponent.display, 'S2C');
  near('hud: 10 s on the clock', h0.timeLeft, 10.0, 1e-12);
  eq('hud: countdown label', h0.countdown.label, '3');
  // the HUD distance IS the policy's own obs dim 54
  near('hud: seat A distance to its line at the opening',
    h0.lineDistance.A, lineDistance('sym', 'A', GAMES.sym.spawn.A.x), 1e-12);
  near('hud: seat B distance to its line at the opening',
    h0.lineDistance.B, lineDistance('sym', 'B', GAMES.sym.spawn.B.x), 1e-12);
  m.tick(3100);
  m.tick();
  const h1 = m.hud();
  ok('hud: the command is slew-limited, not a step', h1.cmd.vx > 0 && h1.cmd.vx < 3.0, `vx ${h1.cmd.vx}`);
  near('hud: exactly one control step of slew',
    h1.cmd.vx, CFG.MATCH.cmdSlewPerSec.vx * PHYS.controlDt, 1e-12);
  for (let i = 0; i < 60; i++) m.tick();
  near('hud: the command saturates at the trained box', m.hud().cmd.vx, CMD_BOX.vx[1], 1e-12);
  eq('clampCmd respects the box', clampCmd(99, -99, 99), { vx: 3.0, vy: -1.0, wz: 2.0 });
}

// --- 4f. the safety-filter seam ---------------------------------------------
{
  const seen = [];
  const shieldPath = {
    kind: 'qcbf_shield_stub',
    reset(measured12) { seen.push({ ev: 'reset', n: measured12 && measured12.length }); },
    step(a12, ctx) {
      seen.push({ ev: 'step', n: a12.length, seat: ctx.seat, robot: ctx.robot,
                  opp: ctx.opponentRobot, game: ctx.game, step: ctx.step });
      return Float64Array.from(DEFAULT_JOINT_POS);
    },
    info() { return { intervening: true, alpha: 0.42 }; },
  };
  const sim = fakeSim('sym');
  const m = MATCH.createMatch({
    sim, game: 'sym', playerSeat: 'A',
    policies: { walk: constPolicy(47, 0), ai: constPolicy(60, 0) },
    input: constInput(),
    actionPaths: { B: shieldPath },
  });
  m.tick(3100); m.tick();
  eq('the seam replaces the whole action path', m.info().actionPaths.B, 'qcbf_shield_stub');
  eq('the player keeps the walk affine', m.info().actionPaths.A, 'walk_affine');
  eq('the shield is reset with the measured joint pos', seen[0], { ev: 'reset', n: 12 });
  const s1 = seen.find((e) => e.ev === 'step');
  eq('the shield sees the raw 12-vector and its context',
    { n: s1.n, seat: s1.seat, robot: s1.robot, opp: s1.opp, game: s1.game },
    { n: 12, seat: 'B', robot: 'b', opp: 'a', game: 'sym' });
  arrNear('the shield ctrl reaches the sim', sim.__robots.b.ctrl, DEFAULT_JOINT_POS, 0);
  eq('the shield lights up the HUD', m.hud().filter.B, { intervening: true, alpha: 0.42 });
  eq('the unfiltered seat reports nothing', m.hud().filter.A, null);
}

// --- 4g. guards ---------------------------------------------------------------
{
  const sim = fakeSim('sym');
  const mk = (o) => MATCH.createMatch({
    sim, game: 'sym', playerSeat: 'A',
    policies: { walk: constPolicy(47, 0), ai: constPolicy(60, 0) },
    input: constInput(), ...o,
  });
  throws('the human cannot take sym seat B (no seat-A weights ship)',
    () => mk({ playerSeat: 'B' }), 'cannot take seat B');
  throws('a 60-D policy in the walk slot is refused',
    () => MATCH.createMatch({ sim, game: 'sym', playerSeat: 'A', input: constInput(),
      policies: { walk: constPolicy(60, 0), ai: constPolicy(60, 0) } }), 'obsDim 60, expected 47');
  throws('a 47-D policy in the AI slot is refused',
    () => MATCH.createMatch({ sim, game: 'sym', playerSeat: 'A', input: constInput(),
      policies: { walk: constPolicy(47, 0), ai: constPolicy(47, 0) } }), 'obsDim 47, expected 60');
  throws('no input is refused',
    () => MATCH.createMatch({ sim, game: 'sym', playerSeat: 'A',
      policies: { walk: constPolicy(47, 0), ai: constPolicy(60, 0) } }), 'input must expose read()');
}

// ===========================================================================
console.log(`\n${fail === 0 ? 'ALL' : ''} ${pass} checks passed, ${fail} failed, ${sections} sections`);
if (fail) {
  console.log('\nfailures:');
  for (const f of failures) console.log('  - ' + f);
}
process.exit(fail === 0 ? 0 : 1);

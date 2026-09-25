/**
 * tests/fall_study.mjs — why does the AI fall?
 *
 *   node tests/fall_study.mjs [episodes]
 *
 * Runs the same matchup with the safety filter ON and OFF and tallies the
 * referee's verdicts, the tilt the AI reaches, and the filter telemetry.
 * Training reference (recon/01): OURS_drw falls in 0.22 % of episodes. Anything
 * near that is healthy; percent-level falls are a bug.
 *
 * WHAT "TOO MANY INTERVENTIONS" MEANS, measured rather than guessed
 * -----------------------------------------------------------------
 * `Filter/intervention_rate` is a scalar the trainer logs every iteration
 * (game/rl/runner.py:3912), and it is the fraction of robot-steps with
 * `alpha > 0.05` (runner.py:3187). Pulled out of the tensorboard of the run
 * that produced the shipped attacker — 2026-08-17_22-23-27_v133fdrw, the
 * attacker writer, at the checkpoint we ship (game_6000, the final iteration):
 *
 *     Filter/intervention_rate   final 0.2723    last-200-it mean 0.3202
 *     Filter/mean_alpha          final 0.1568    last-200-it mean 0.1899
 *     Filter/mean_gain_alpha     final 0.1302    last-200-it mean 0.1605
 *     Filter/mean_value          final 0.0515    last-200-it mean 0.0477
 *
 * (v135fdrw, the shipped defender: 0.2550 / 0.1571 / 0.1346 / 0.0500.)
 *
 * So this certificate intervening on roughly a THIRD of control steps is the
 * regime the policy was trained inside, not a symptom. The "2-4 net evals"
 * figure in filtered_action.py:646 is the SECANT'S ITERATION COUNT on a step
 * that does intervene, not a claim about how often it intervenes.
 *
 * The numbers below are printed next to those four so the browser can be held
 * against the trainer directly.
 */
import { createSim } from '../app/physics.js';
import { loadManifest } from '../app/policy.js';
import { createMatch } from '../app/match.js';
import { loadFilter, makeShieldPath, DECISION_NAME } from '../app/filter.js';
import { seatRobot, otherSeat, gameCfg } from '../app/config.js';

const ROOT = new URL('..', import.meta.url).pathname.replace(/\/$/, '');
const argv = process.argv.slice(2);
const flag = (n, d) => { const i = argv.indexOf(n); return i >= 0 ? argv[i + 1] : d; };
const EPISODES = Number(argv.find((a) => /^\d+$/.test(a)) || 10);
/** Which game, which AI seat, which certificate fallback net, what the human does. */
const GAME = flag('--game', 'asym');
const OPPONENT = flag('--opponent', GAME === 'sym' ? 'sym_s2c_B' : 'asym_s2c_attacker');
const PLAYER_SEAT = flag('--seat', GAME === 'sym' ? 'A' : 'defender');
const CTRL = flag('--ctrl', '');                       // '', 'ctrl' or 'ctrl_s1'
const VX = Number(flag('--vx', GAME === 'sym' ? '2.0' : '0'));
/**
 * `--player erratic` drives the human seat the way a person actually does —
 * speed changes, turns, the odd sidestep — because a straight-line charge only
 * ever produces one contact geometry and the certificate's regime depends on
 * the geometry. Deterministic: episode index seeds the phase.
 */
let PLAYER_T = 0, PLAYER_EP = 0;
const ERRATIC = flag('--player', 'hold') === 'erratic';
const stillInput = {
  read: () => {
    if (!ERRATIC) return { vx: VX, vy: 0, wz: 0 };
    const t = PLAYER_T * 0.02, k = 1 + PLAYER_EP;
    return {
      vx: 1.6 + 1.2 * Math.sin(0.9 * k * t + k),
      vy: 0.6 * Math.sin(1.7 * t + 2 * k),
      wz: 1.1 * Math.sin(0.6 * k * t + 0.5 * k),
    };
  },
  bindings: {}, setScheme() {},
};

/** v133fdrw attacker, the run behind `asym_s2c_attacker`. See the header. */
const TRAINING = {
  interventionRate: [0.2723, 0.3202],   // [final iteration, last-200 mean]
  meanAlpha: [0.1568, 0.1899],
  meanGainAlpha: [0.1302, 0.1605],
  meanValue: [0.0515, 0.0477],
};

const mean = (a) => (a.length ? a.reduce((x, y) => x + y, 0) / a.length : 0);
const pct = (a, p) => {
  if (!a.length) return 0;
  const s = [...a].sort((x, y) => x - y);
  return s[Math.min(s.length - 1, Math.floor(p * s.length))];
};

async function run({ withFilter, episodes }) {
  const sim = await createSim({ sceneUrl: `${ROOT}/assets/scene/${GAME}/scene.xml` });
  const man = await loadManifest(`${ROOT}/assets/policies/manifest.json`);
  const walk = await man.load(man.playerWalk().name);
  const ai = await man.load(OPPONENT);

  const game = GAME;
  const g = gameCfg(game);
  const playerSeat = PLAYER_SEAT;
  const aiSeat = otherSeat(g, playerSeat);
  let actionPaths = null;
  let shield = null;
  if (withFilter) {
    const filter = await loadFilter({
      manifest: man.manifest, baseUrl: `${ROOT}/assets/policies/`, game,
      ...(CTRL ? { ctrlNet: CTRL } : {}),
    });
    shield = makeShieldPath({
      filter, sim, robot: seatRobot(g, aiSeat), opponentRobot: seatRobot(g, playerSeat),
    });
    actionPaths = { [aiSeat]: shield };
  }

  const match = createMatch({
    sim, game, playerSeat, opponent: man.entry(OPPONENT),
    policies: { walk, ai }, input: stillInput, actionPaths,
    config: { countdownMs: 0 },
  });

  const tally = {};
  const decisions = {};
  const tilts = [], heights = [];
  const alpha = [], gainAlpha = [], value = [], stepMs = [], qEvals = [];
  let maxTilt = 0, steps = 0, pgInfeasible = 0;
  for (let ep = 0; ep < episodes; ep++) {
    match.reset();
    PLAYER_EP = ep; PLAYER_T = 0;
    let hud = match.hud();
    while (!hud.verdict && hud.step < g.episodeSteps) {
      hud = match.tick(20);
      steps += 1; PLAYER_T += 1;
      const t = sim.tiltAngle(seatRobot(g, aiSeat));
      if (t > maxTilt) maxTilt = t;
      tilts.push(t);
      heights.push(sim.getBase(seatRobot(g, aiSeat)).pos[2]);
      const info = hud.filter && hud.filter[aiSeat];
      if (!info) continue;
      const name = DECISION_NAME[info.decision];
      decisions[name] = (decisions[name] || 0) + 1;
      alpha.push(info.alpha);
      gainAlpha.push(info.gainAlpha);
      value.push(info.value);
      stepMs.push(info.stepMs);
      qEvals.push(info.qEvals);
      pgInfeasible = info.pgInfeasible ?? 0;
    }
    const term = hud.verdict ? `${hud.verdict.terminal}/${hud.verdict.winner}` : 'no-verdict';
    tally[term] = (tally[term] || 0) + 1;
    process.stdout.write(`    ep${String(ep + 1).padStart(2)}  ${term.padEnd(20)} step ${hud.step}\n`);
  }
  const solver = shield ? shield.solver() : null;
  sim.dispose?.();
  return {
    tally,
    meanTiltDeg: (mean(tilts) * 180) / Math.PI,
    p95TiltDeg: (pct(tilts, 0.95) * 180) / Math.PI,
    meanHeight: mean(heights),
    sdHeight: Math.sqrt(mean(heights.map((h) => (h - mean(heights)) ** 2))),
    loHeight: pct(heights, 0.05),
    maxTiltDeg: (maxTilt * 180) / Math.PI,
    steps,
    solver,
    decisions,
    // The trainer's definition, so the two numbers are comparable.
    interventionRate: alpha.filter((a) => a > 0.05).length / Math.max(alpha.length, 1),
    meanAlpha: mean(alpha),
    meanGainAlpha: mean(gainAlpha),
    meanValue: mean(value),
    pgInfeasible,
    stepMs: { p50: pct(stepMs, 0.5), p95: pct(stepMs, 0.95), max: Math.max(0, ...stepMs) },
    qEvals: { p50: pct(qEvals, 0.5), max: Math.max(0, ...qEvals) },
  };
}

const cmp = (got, [fin, win]) =>
  `${got.toFixed(4)}   (training ${fin.toFixed(4)} final, ${win.toFixed(4)} last-200)`;

for (const withFilter of [false, true]) {
  console.log(`\n=== S2C attacker vs a standing human — filter ${withFilter ? 'ON' : 'OFF'} ===`);
  try {
    const r = await run({ withFilter, episodes: EPISODES });
    console.log(`  verdicts: ${JSON.stringify(r.tally)}`);
    console.log(`  AI tilt  mean ${r.meanTiltDeg.toFixed(1)}  p95 ${r.p95TiltDeg.toFixed(1)}  max ${r.maxTiltDeg.toFixed(1)} deg` +
      `   trunk height ${r.meanHeight.toFixed(3)} m  sd ${r.sdHeight.toFixed(3)}  p05 ${r.loHeight.toFixed(3)}`);
    if (!withFilter) continue;
    console.log(`  solver ${r.solver.intervention}  eta ${r.solver.pgStep} x ` +
      `${r.solver.pgMaxIters} steps, ${r.solver.pgBacktrackIters} backtracks`);
    console.log(`  decisions ${JSON.stringify(r.decisions)}   pg_infeasible ${r.pgInfeasible}`);
    console.log(`  intervention_rate (alpha>0.05) ${cmp(r.interventionRate, TRAINING.interventionRate)}`);
    console.log(`  mean_alpha                     ${cmp(r.meanAlpha, TRAINING.meanAlpha)}`);
    console.log(`  mean_gain_alpha                ${cmp(r.meanGainAlpha, TRAINING.meanGainAlpha)}`);
    console.log(`  mean_value                     ${cmp(r.meanValue, TRAINING.meanValue)}`);
    console.log(`  cost/step  p50 ${r.stepMs.p50.toFixed(2)} ms  p95 ${r.stepMs.p95.toFixed(2)} ms  ` +
      `max ${r.stepMs.max.toFixed(2)} ms   robust_q p50 ${r.qEvals.p50} max ${r.qEvals.max}`);
  } catch (err) {
    console.log(`  ERROR: ${err.stack || err.message}`);
  }
}

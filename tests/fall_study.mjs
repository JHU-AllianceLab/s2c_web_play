/**
 * tests/fall_study.mjs — why does the AI fall?
 *
 *   node tests/fall_study.mjs [episodes]
 *
 * Runs the same matchup with the safety filter ON and OFF and tallies the
 * referee's verdicts, the tilt the AI reaches, and how often the filter
 * intervenes. Training reference (recon/01): OURS_drw falls in 0.22 % of
 * episodes. Anything near that is healthy; percent-level falls are a bug.
 */
import { createSim } from '../app/physics.js';
import { loadManifest } from '../app/policy.js';
import { createMatch } from '../app/match.js';
import { loadFilter, makeShieldPath } from '../app/filter.js';
import { GAMES, seatRobot, otherSeat, gameCfg } from '../app/config.js';

const ROOT = new URL('..', import.meta.url).pathname.replace(/\/$/, '');
const EPISODES = Number(process.argv[2] || 10);
const stillInput = { read: () => ({ vx: 0, vy: 0, wz: 0 }), bindings: {}, setScheme() {} };

async function run({ withFilter, episodes }) {
  const sim = await createSim({ sceneUrl: `${ROOT}/assets/scene/asym/scene.xml` });
  const man = await loadManifest(`${ROOT}/assets/policies/manifest.json`);
  const walk = await man.load(man.playerWalk().name);
  const ai = await man.load('asym_s2c_attacker');

  const game = 'asym';
  const g = gameCfg(game);
  const playerSeat = 'defender';
  const aiSeat = otherSeat(g, playerSeat);
  let actionPaths = null;
  let shield = null;
  if (withFilter) {
    const filter = await loadFilter({ manifest: man.manifest, baseUrl: `${ROOT}/assets/policies/` });
    shield = makeShieldPath({
      filter, sim, robot: seatRobot(g, aiSeat), opponentRobot: seatRobot(g, playerSeat),
    });
    actionPaths = { [aiSeat]: shield };
  }

  const match = createMatch({
    sim, game, playerSeat, opponent: man.entry('asym_s2c_attacker'),
    policies: { walk, ai }, input: stillInput, actionPaths,
    config: { countdownMs: 0 },
  });

  const tally = {};
  let maxTilt = 0, interventions = 0, steps = 0;
  for (let ep = 0; ep < episodes; ep++) {
    match.reset();
    let hud = match.hud();
    while (!hud.verdict && hud.step < g.episodeSteps) {
      hud = match.tick(20);
      steps += 1;
      const t = sim.tiltAngle(seatRobot(g, aiSeat));
      if (t > maxTilt) maxTilt = t;
      const info = hud.filter && hud.filter[aiSeat];
      if (info && (info.active || info.intervened)) interventions += 1;
    }
    const term = hud.verdict ? `${hud.verdict.terminal}/${hud.verdict.winner}` : 'no-verdict';
    tally[term] = (tally[term] || 0) + 1;
    process.stdout.write(`    ep${String(ep + 1).padStart(2)}  ${term.padEnd(20)} step ${hud.step}\n`);
  }
  sim.dispose?.();
  return { tally, maxTiltDeg: (maxTilt * 180) / Math.PI, interventionRate: interventions / Math.max(steps, 1) };
}

for (const withFilter of [false, true]) {
  console.log(`\n=== S2C attacker vs a standing human — filter ${withFilter ? 'ON' : 'OFF'} ===`);
  try {
    const r = await run({ withFilter, episodes: EPISODES });
    console.log(`  verdicts: ${JSON.stringify(r.tally)}`);
    console.log(`  AI max tilt ${r.maxTiltDeg.toFixed(1)} deg (fall threshold 70)` +
      `   filter active on ${(r.interventionRate * 100).toFixed(1)} % of steps`);
  } catch (err) {
    console.log(`  ERROR: ${err.message}`);
  }
}

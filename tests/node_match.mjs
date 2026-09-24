/**
 * tests/node_match.mjs — the whole game, headless.
 *
 * Real scene, real checkpoint weights, real observation assembly, real action
 * integrator, real referee: exactly what the browser runs, minus the browser.
 * If this passes, the page is a front end over a working game; if it fails, no
 * amount of rendering hides it.
 *
 *   node tests/node_match.mjs            # sym + asym (both roles), S2C opponent
 *   node tests/node_match.mjs --all      # every opponent in the manifest too
 *
 * The "player" is a scripted hand: hold W (full forward). That is enough to
 * prove the loop is wired — the human's dog must actually cover ground and the
 * episode must end in a referee terminal, not in a crash or a NaN.
 */

import { createSim } from '../app/physics.js';
import { loadManifest } from '../app/policy.js';
import { createMatch } from '../app/match.js';
import { GAMES } from '../app/config.js';

const ROOT = new URL('..', import.meta.url).pathname.replace(/\/$/, '');
const sceneUrl = (game) => `${ROOT}/assets/scene/${game}/scene.xml`;
const manifestUrl = `${ROOT}/assets/policies/manifest.json`;

let failures = 0;
const ok = (cond, label, detail = '') => {
  console.log(`  ${cond ? 'ok  ' : 'FAIL'}  ${label}${detail ? `   ${detail}` : ''}`);
  if (!cond) failures += 1;
};

/** A scripted human: hold W, nothing else. */
function scriptedInput(cmd = { vx: 3.0, vy: 0, wz: 0 }) {
  return { read: () => ({ ...cmd }), bindings: {}, setScheme() {} };
}

const finite = (...xs) => xs.every((v) => Number.isFinite(v));

async function runOne({ game, playerSeat, opponentName, cmd, label }) {
  const sim = await createSim({ sceneUrl: sceneUrl(game) });
  const man = await loadManifest(manifestUrl);
  const walk = await man.load(man.playerWalk().name);
  const ai = await man.load(opponentName);

  const match = createMatch({
    sim,
    game,
    playerSeat,
    opponent: man.entry(opponentName),
    policies: { walk, ai },
    input: scriptedInput(cmd),
    config: { countdownMs: 0 },          // no countdown in a headless run
  });
  match.reset();

  const g = GAMES[game];
  const t0 = Date.now();
  let hud = match.hud();
  let steps = 0;
  let nan = false;

  while (!hud.verdict && steps < g.episodeSteps + 5) {
    hud = match.tick(20);
    steps += 1;
    const st = match.state();
    for (const seat of Object.keys(st.seats)) {
      const s = st.seats[seat];
      if (!finite(s.x, s.y, s.yaw)) nan = true;
    }
    if (nan) break;
  }

  const ms = Date.now() - t0;
  const st = match.state();
  const seats = Object.entries(st.seats)
    .map(([seat, s]) => `${seat}(${s.x.toFixed(2)}, ${s.y.toFixed(2)})`)
    .join('  ');

  console.log(`\n${label}`);
  console.log(`  terminal ${hud.verdict ? hud.verdict.terminal : '(none)'}` +
    `   winner ${hud.verdict ? hud.verdict.winner : '-'}` +
    `   steps ${steps}/${g.episodeSteps}   ${(ms / steps).toFixed(2)} ms/step` +
    `   ${(ms / (steps * 20)).toFixed(2)}x realtime`);
  console.log(`  final  ${seats}`);

  ok(!nan, 'no NaN anywhere in the episode');
  ok(!!hud.verdict, 'the episode reached a referee terminal', hud.verdict?.terminal);
  ok(steps <= g.episodeSteps + 1, 'it did not run past the episode length');
  ok(ms / steps < 20, 'a control step costs less than the 20 ms it simulates', `${(ms / steps).toFixed(2)} ms`);

  // The scripted hand holds W: the human's dog must have covered ground.
  const playerState = st.seats[st.playerSeat];
  const spawnX = GAMES[game].spawn[st.playerSeat].x;
  const moved = Math.abs(playerState.x - spawnX);
  ok(moved > 0.3, 'the player dog actually locomoted on a held W', `moved ${moved.toFixed(2)} m from spawn`);

  const out = {
    game, playerSeat, opponentName,
    terminal: hud.verdict?.terminal ?? null,
    winner: hud.verdict?.winner ?? null,
    steps, msPerStep: ms / steps, moved,
  };
  match.dispose?.();
  sim.dispose?.();
  return out;
}

const all = process.argv.includes('--all');

const cases = [
  { game: 'sym', playerSeat: 'A', opponentName: 'sym_s2c_B',
    label: 'SYM — you at seat A (score at +1.9) vs S2C at seat B' },
  { game: 'asym', playerSeat: 'attacker', opponentName: 'asym_s2c_defender',
    label: 'ASYM — you attack, S2C defends' },
  { game: 'asym', playerSeat: 'defender', opponentName: 'asym_s2c_attacker',
    // a defender holding W would run away from its own goal; hold position-ish
    cmd: { vx: 0.6, vy: 0, wz: 0 },
    label: 'ASYM — you defend, S2C attacks' },
];

if (all) {
  for (const name of ['sym_et_B', 'sym_nom_B', 'sym_cpo_B', 'sym_lag_B']) {
    cases.push({ game: 'sym', playerSeat: 'A', opponentName: name, label: `SYM — vs ${name}` });
  }
  for (const name of ['asym_et_defender', 'asym_nom_defender', 'asym_cpo_defender', 'asym_lag_defender']) {
    cases.push({ game: 'asym', playerSeat: 'attacker', opponentName: name, label: `ASYM — you attack, ${name}` });
  }
}

console.log('='.repeat(78));
console.log('FULL MATCH — real scene, real weights, real referee');
console.log('='.repeat(78));

const results = [];
for (const c of cases) results.push(await runOne(c));

console.log(`\n${'='.repeat(78)}`);
for (const r of results) {
  console.log(`  ${r.game.padEnd(5)} ${String(r.playerSeat).padEnd(9)} vs ${r.opponentName.padEnd(20)}` +
    ` -> ${String(r.terminal).padEnd(14)} winner ${String(r.winner).padEnd(9)}` +
    ` ${r.steps} steps, ${r.msPerStep.toFixed(2)} ms/step`);
}
console.log(failures ? `\n${failures} CHECK(S) FAILED` : '\nALL CHECKS PASSED');
console.log('='.repeat(78));
process.exit(failures ? 1 : 0);

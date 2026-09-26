/**
 * tests/head_on.mjs — walk straight at the shielded AI and count what happens to IT.
 *
 *   node tests/head_on.mjs --game sym  --vx 1.15 --lines 16
 *   node tests/head_on.mjs --game asym --vx 1.43
 *
 * The scripted opponents in fall_study.mjs steer; a person walking at a dog does
 * not, and that is the input that keeps the certificate's value negative for
 * many steps in a row. Each line is one episode on a slightly different
 * approach, so the battery covers the contact geometries a straight walk can
 * produce instead of the single one a fixed heading gives.
 *
 * What it prints is the AI's business only: whether the referee called it
 * fallen, how far it tipped, and what the certificate was doing.
 */
import { createSim } from '../app/physics.js';
import { createMatch } from '../app/match.js';
import { loadManifest, loadPolicy } from '../app/policy.js';
import { loadFilter, makeShieldPath, DECISION_NAME } from '../app/filter.js';
import { seatRobot, otherSeat, gameCfg, goalDir } from '../app/config.js';

const ROOT = new URL('..', import.meta.url).pathname.replace(/\/$/, '');
const argv = process.argv.slice(2);
const flag = (n, d) => { const i = argv.indexOf(n); return i >= 0 ? argv[i + 1] : d; };
const GAME = flag('--game', 'sym');
const VX = Number(flag('--vx', GAME === 'sym' ? '1.15' : '1.43'));
const LINES = Number(flag('--lines', '16'));
const OPP = flag('--opponent', GAME === 'sym' ? 'sym_s2c_B' : 'asym_s2c_attacker');
/** A candidate exported by tools/export_candidate.py, loaded by file name. */
const FILE = flag('--file', '');
const SHIELD = flag('--shield', 'on') !== 'off';
/**
 * `--park N` stops the human dead after N steps, in the AI's way. The AI must
 * work around a standing obstacle; a shielded dog that instead freezes in front
 * of it and runs the clock out is the failure this catches.
 */
const PARK = Number(flag('--park', '0'));

let nSteps = () => 0;
const g = gameCfg(GAME);
const PLAYER_SEAT = flag('--seat', GAME === 'sym' ? 'A' : 'defender');
const AI_SEAT = otherSeat(g, PLAYER_SEAT);

const seatOf = (rob) => (seatRobot(g, PLAYER_SEAT) === rob ? PLAYER_SEAT : AI_SEAT);
const man = await loadManifest(`${ROOT}/assets/policies/manifest.json`);
const walk = await man.load(man.playerWalk().name);

let falls = 0, over45 = 0, worst = 0, youWin = 0, draw = 0;
const tally = {}, dec = {};
for (let k = 0; k < LINES; k++) {
  const sim = await createSim({ sceneUrl: `${ROOT}/assets/scene/${GAME}/scene.xml` });
  const ai = FILE
    ? await loadPolicy(`${ROOT}/assets/policies/${FILE}.json`)
    : await man.load(OPP);
  let actionPaths = null;
  if (SHIELD) {
    const filter = await loadFilter({ manifest: man.manifest, game: GAME });
    actionPaths = {
      [AI_SEAT]: makeShieldPath({
        filter, sim, robot: seatRobot(g, AI_SEAT), opponentRobot: seatRobot(g, PLAYER_SEAT),
      }),
    };
  }
  const vy = (k - (LINES - 1) / 2) * (1.0 / LINES);
  const match = createMatch({
    sim, game: GAME, playerSeat: PLAYER_SEAT,
    opponent: FILE ? { name: FILE, display: FILE, method: 'candidate' } : man.entry(OPP),
    policies: { walk, ai },
    input: { read: () => (PARK && nSteps() > PARK ? { vx: 0, vy: 0, wz: 0 } : { vx: VX, vy, wz: 0 }) },
    actionPaths, config: { countdownMs: 0 },
  });
  match.reset();
  let hud = match.hud(), n = 0, tilt = 0;
  nSteps = () => n;
  while (!hud.verdict && n < g.episodeSteps) {
    hud = match.tick(20); n += 1;
    const t = sim.tiltAngle(seatRobot(g, AI_SEAT));
    if (t > tilt) tilt = t;
    const i = hud.filter && hud.filter[AI_SEAT];
    if (i) dec[DECISION_NAME[i.decision]] = (dec[DECISION_NAME[i.decision]] || 0) + 1;
  }
  const term = `${hud.verdict?.terminal}/${hud.verdict?.winner}`;
  const w = hud.verdict?.winnerSeat ?? hud.verdict?.winner;
  if (w === 'DRAW' || w === 'draw') draw += 1;
  else if (w === PLAYER_SEAT || w === (PLAYER_SEAT === seatOf('a') ? 'A' : 'B')) youWin += 1;
  tally[term] = (tally[term] || 0) + 1;
  const deg = (tilt * 180) / Math.PI;
  if (/(^|_)fell/.test(term)) falls += 1;
  if (deg > 45) over45 += 1;
  if (deg > worst) worst = deg;
  sim.dispose?.();
}
const pct = (x) => `${((100 * x) / LINES).toFixed(0)}%`;
console.log(
  `${GAME} / ${FILE || OPP} / player ${VX} m/s${SHIELD ? '' : ' / NO SHIELD'}: ` +
  `YOU win ${youWin}/${LINES} (${pct(youWin)})  draw ${draw}   ` +
  `AI fell ${falls}/${LINES} (${pct(falls)})   worst tilt ${worst.toFixed(1)} deg`);
console.log(`  verdicts ${JSON.stringify(tally)}`);
if (SHIELD) console.log(`  decisions ${JSON.stringify(dec)}`);

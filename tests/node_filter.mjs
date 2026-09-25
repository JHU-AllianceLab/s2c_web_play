/**
 * tests/node_filter.mjs — the S2C certificate, held against the training code.
 *
 *   node tests/node_filter.mjs          # everything
 *   node tests/node_filter.mjs --bench  # + a longer cost measurement
 *
 * Every reference number here came out of the PYTHON implementation, not out of
 * a transcription of it: tools/export_filter.py duck-types an env and calls
 * `GameQcbfShieldAction._assemble_obs` and `BatchedQcbfFilter.step` directly,
 * writing tests/filter_fixture.json.
 *
 *   1  NETS          the 4 exported .bin/.json load and have the right shapes
 *   2  GEOMETRY      rect_rect_distance / the 8-D opponent tail
 *   3  OBSERVATION   the 62-D vector, block by block, against python
 *   4  FILTER LAW    u_task / u_safe / V / q_task / u_sel / alpha / decision / target
 *   5  CLOSED LOOP   a real MuJoCo match with the filter on BOTH seats
 *   6  COST          measured us per filtered robot per control step
 */

import { readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { createSim } from '../app/physics.js';
import { loadManifest } from '../app/policy.js';
import { createMatch } from '../app/match.js';
import {
  loadFilter, makeShieldPath, createCertificate, createShield, filterObs,
  rectRectDistance, opponentTail, hullCentre, FILTER_OBS_LAYOUT, DECISION_NAME,
} from '../app/filter.js';
import { IncrementIntegrator, DEFAULT_JOINT_POS } from '../app/action.js';

const ROOT = new URL('..', import.meta.url).pathname.replace(/\/$/, '');
const BENCH = process.argv.includes('--bench');
const TOL = 1e-5;            // the parity budget, tests/parity.json `tolerance`
const TOL_U = 1e-4;          // line-search iterates: a root find under tol 0.01

let failures = 0;
const ok = (cond, label, detail = '') => {
  console.log(`  ${cond ? '\x1b[32mok\x1b[0m  ' : '\x1b[31mFAIL\x1b[0m'} ${label}${detail ? `   ${detail}` : ''}`);
  if (!cond) failures += 1;
};
const section = (t) => console.log(`\n\x1b[1m${t}\x1b[0m`);
const maxAbs = (a, b) => {
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
  return m;
};

const fx = JSON.parse(await readFile(join(ROOT, 'tests/filter_fixture.json'), 'utf8'));
const man = await loadManifest(join(ROOT, 'assets/policies/manifest.json'));
const filter = await loadFilter({ manifest: man.manifest });

// ===========================================================================
section('1. NETS — the exported certificate');
// ===========================================================================
{
  const e = filter.entry;
  ok(e.filter_nets_md5 === fx.filter_nets_md5,
    'the fixture and the manifest name the same filter_nets.pt', e.filter_nets_md5);
  ok(e.filter_nets_md5 === '96fcae4443cc1e51f421a608ea1780b5',
    'that md5 is the official v5prox agent_15000 bundle (recon/03:28)');
  const n = filter.nets;
  ok(n.ctrl.obsDim === 62 && n.ctrl.actDim === 12, 'ctrl  62 -> 12');
  ok(n.dstb.obsDim === 74 && n.dstb.actDim === 6, 'dstb  74 -> 6  (62 obs + 12 ctrl)');
  ok(n.q1.obsDim === 80 && n.q1.actDim === 1, 'q1    80 -> 1  (62 + 12 + 6)');
  ok(n.q2.obsDim === 80 && n.q2.actDim === 1, 'q2    80 -> 1');
  ok(n.ctrl.meta.activation === 'sin' && n.ctrl.meta.head_activation === 'tanh',
    'ctrl is sin-activated with a tanh head');
  ok(n.q1.meta.head_activation === null, 'the twin heads are bare linear');
  ok(n.ctrl.norm === null && n.dstb.norm === null && n.q1.norm === null,
    'no observation normalizer on any of them (filtered_action.py:296-322)');
  const p = filter.params;
  ok(p.kappa === 0.9, 'kappa 0.9', `${p.kappa}`);
  ok(p.cbfMaxIters === 20 && p.cbfTol === 0.01, 'line search 20 iters, tol 0.01');
  ok(p.incScale === 0.5 && p.incSmoothing === 0.3,
    'increment scale 0.5 / smoothing 0.3 -> inverse denominator 0.15');
  ok(filter.field.halfX === 2.4 && filter.field.halfY === 1.5,
    'the certificate sees the BUNDLE rectangle 4.8 x 3.0, not the 5.2 x 3.0 scenario',
    `hx ${filter.field.halfX}  hy ${filter.field.halfY}`);
}

// ===========================================================================
section('2. GEOMETRY — rect_rect_distance and the 8-D tail');
// ===========================================================================
{
  // Two axis-aligned hulls nose to nose: the gap is centre distance - 2*halfX.
  const g = rectRectDistance(0, 0, 0, 1.0, 0, 0);
  ok(Math.abs(g.d - (1.0 - 2 * 0.29)) < 1e-12, 'nose-to-nose gap = dx - 2*halfX',
    `${g.d.toFixed(6)}`);
  ok(Math.abs(g.nx - 1) < 1e-12 && Math.abs(g.ny) < 1e-12, 'normal points ego -> opp');
  // Symmetric in (A, B) for d.
  const h = rectRectDistance(1.0, 0, 0, 0, 0, 0);
  ok(Math.abs(h.d - g.d) < 1e-12, 'd is symmetric under swapping the two hulls');
  ok(Math.abs(h.nx + 1) < 1e-12, 'the normal flips with the swap');
  // Deep overlap: the centre-vs-box candidates must make d negative.
  const o = rectRectDistance(0, 0, 0, 0.05, 0, Math.PI / 2);
  ok(o.d < 0, 'a plus-shaped deep overlap reads a NEGATIVE d (the centre candidates)',
    `${o.d.toFixed(6)}`);
  // The +0.05 hull offset.
  const c = hullCentre(1, 2, Math.PI / 2);
  ok(Math.abs(c[0] - 1) < 1e-12 && Math.abs(c[1] - 2.05) < 1e-12,
    'the hull centre sits 0.05 m ahead along BODY x');
  // Split horizon.
  // Hull centres are 0.05 m ahead of base_link, so a head-on pair at base
  // separation dx has a gap of (dx - 0.1) - 2*0.29: dx 1.9 -> 1.22 (inside
  // d_vis), dx 2.3 -> 1.62 (outside).
  const near = opponentTail([0, 0, 0, 0, 0], [1.9, 0, Math.PI, -1, 0]);
  const far = opponentTail([0, 0, 0, 0, 0], [2.3, 0, Math.PI, -1, 0]);
  ok(near[0] < 1.5 && near[3] !== 0, 'inside d_vis the direction channels are live',
    `d ${near[0].toFixed(4)}  d_dot ${near[3].toFixed(4)}`);
  ok(far[0] > 1.5 && far[1] === 0 && far[3] === 0 && far[6] === 1 && far[7] === 0,
    'past d_vis only d survives; the other 7 blank to [0,0,0,0,0,1,0]',
    `d ${far[0].toFixed(4)}`);
  const veryFar = opponentTail([0, 0, 0, 0, 0], [20, 0, Math.PI, 0, 0]);
  ok(veryFar[0] === 3.0, 'd clamps at d_max = 3.0 (NOT at d_vis)');
}

// ===========================================================================
section('3. OBSERVATION — the 62-D vector, block by block, against python');
// ===========================================================================
//
// `filterObs` is driven by a stand-in Sim that returns exactly the state
// tools/export_filter.py handed the REAL `_assemble_obs`. That isolates the
// assembly maths: any difference here is this file's port, not MuJoCo.

function quatFromRpy(roll, pitch, yaw) {
  const cr = Math.cos(roll / 2), sr = Math.sin(roll / 2);
  const cp = Math.cos(pitch / 2), sp = Math.sin(pitch / 2);
  const cy = Math.cos(yaw / 2), sy = Math.sin(yaw / 2);
  return [
    cr * cp * cy + sr * sp * sy,
    sr * cp * cy - cr * sp * sy,
    cr * sp * cy + sr * cp * sy,
    cr * cp * sy - sr * sp * cy,
  ];
}
function fakeSim(s) {
  const base = {
    a: { pos: s.ego_pos, quat: quatFromRpy(...s.ego_rpy), linVelW: s.ego_lin_vel_w,
      angVelB: s.ego_ang_vel_b },
    b: { pos: s.opp_pos, quat: quatFromRpy(...s.opp_rpy), linVelW: s.opp_lin_vel_w,
      angVelB: [0, 0, 0] },
  };
  return {
    getBase: (r) => base[r],
    getJointPos: (r) => (r === 'a' ? s.joint_pos : DEFAULT_JOINT_POS),
    getJointVel: (r) => (r === 'a' ? s.joint_vel : new Array(12).fill(0)),
  };
}

{
  const perBlock = new Map(FILTER_OBS_LAYOUT.map((t) => [t.term, 0]));
  let worst = 0, worstState = '';
  for (let i = 0; i < fx.states.length; i++) {
    const s = fx.states[i];
    const ref = fx.obs[i];
    const got = filterObs(fakeSim(s), 'a', 'b', s.prev_ctrl, s.feet, filter.field);
    const m = maxAbs(got, ref);
    if (m > worst) { worst = m; worstState = s.label; }
    for (const t of FILTER_OBS_LAYOUT) {
      let bm = 0;
      for (let k = 0; k < t.dim; k++) bm = Math.max(bm, Math.abs(got[t.at + k] - ref[t.at + k]));
      perBlock.set(t.term, Math.max(perBlock.get(t.term), bm));
    }
  }
  for (const [term, m] of perBlock) {
    ok(m < TOL, `${term.padEnd(22)} over ${fx.states.length} python states`,
      `max|d| ${m.toExponential(3)}`);
  }
  console.log(`  worst dim over the whole 62-D vector: ${worst.toExponential(3)} (${worstState})`);

  // The split horizon has to be exercised by the fixture, not just by section 2.
  const nFar = fx.obs.filter((r) => r[55] === 0 && r[56] === 0 && r[60] === 1).length;
  ok(nFar > 0 && nFar < fx.obs.length,
    'the fixture covers BOTH sides of the d_vis split horizon',
    `${nFar}/${fx.obs.length} blanked`);
  const nNegMargin = fx.obs.filter((r) => Math.min(r[48], r[49], r[50], r[51]) < 0).length;
  ok(nNegMargin > 0, 'and at least one state outside the certificate rectangle',
    `${nNegMargin} with a negative wall margin`);
}

// ===========================================================================
section('4. FILTER LAW — the decision, against BatchedQcbfFilter');
// ===========================================================================
{
  const cert = createCertificate(filter.nets);
  const shield = createShield(cert, filter.params);
  const integ = new IncrementIntegrator(
    DEFAULT_JOINT_POS, filter.params.qLo, filter.params.qHi,
    { incrementScale: filter.params.incScale, smoothing: filter.params.incSmoothing },
  );

  let wV = 0, wQ = 0, wUt = 0, wUs = 0, wSel = 0, wA = 0, wTgt = 0;
  let decisionsOk = 0;
  const counts = {};
  for (const c of fx.cases) {
    const obs = Float32Array.from(c.obs);
    integ.reset(c.target_in);
    // q_des = 0.25*a + default, so the fixture's q_des pins the raw action.
    const a12 = c.q_des.map((v, i) => (v - DEFAULT_JOINT_POS[i]) / 0.25);
    const uTask = integ.taskIncrement(a12);
    const r = shield.step(obs, uTask);
    const target = integ.applyIncrement(r.u);

    wUt = Math.max(wUt, maxAbs(uTask, c.u_task));
    wV = Math.max(wV, Math.abs(r.V - c.V));
    wQ = Math.max(wQ, Math.abs(r.qTask - c.q_task));
    wSel = Math.max(wSel, maxAbs(r.u, c.u_sel));
    wA = Math.max(wA, Math.abs(r.alpha - c.alpha));
    wTgt = Math.max(wTgt, maxAbs(target, c.target_out));
    if (r.decision === c.decision) decisionsOk += 1;
    else {
      console.log(`     ${c.state}/${c.proposal}: decision ${DECISION_NAME[r.decision]} ` +
        `!= python ${DECISION_NAME[c.decision]}`);
    }
    counts[DECISION_NAME[c.decision]] = (counts[DECISION_NAME[c.decision]] || 0) + 1;
  }
  // u_safe depends only on obs, so check it once per state.
  for (let i = 0; i < fx.states.length; i++) {
    const obs = Float32Array.from(fx.obs[i]);
    const c = fx.cases.find((x) => x.state === fx.states[i].label);
    wUs = Math.max(wUs, maxAbs(cert.fallback(obs), c.u_safe));
  }

  ok(decisionsOk === fx.cases.length,
    `every decision matches python (${Object.entries(counts).map(([k, v]) => `${v} ${k}`).join(', ')})`,
    `${decisionsOk}/${fx.cases.length}`);
  ok(wUs < TOL, 'u_safe = pi_shield(obs)', `max|d| ${wUs.toExponential(3)}`);
  ok(wV < TOL, 'V = Qhat(obs, u_safe)', `max|d| ${wV.toExponential(3)}`);
  ok(wQ < TOL, 'q_task = Qhat(obs, u_task)', `max|d| ${wQ.toExponential(3)}`);
  ok(wUt < TOL, 'u_task = clamp((q_des - target)/0.15, -1, 1)', `max|d| ${wUt.toExponential(3)}`);
  ok(wSel < TOL_U, 'u_sel (task pass / fallback / line search)', `max|d| ${wSel.toExponential(3)}`);
  ok(wA < TOL_U, 'alpha (the projection onto [u_task, u_safe])', `max|d| ${wA.toExponential(3)}`);
  ok(wTgt < TOL_U, 'the v25 target the PD actually tracks', `max|d| ${wTgt.toExponential(3)}`);

  // The published anchors: recon/03:378-379 quotes these two V values.
  const stance = fx.cases.find((c) => c.state === 'stance_no_opponent');
  const wall = fx.cases.find((c) => c.state === 'near_wall_x2.3');
  ok(Math.abs(stance.V - 0.078329) < 1e-6,
    'stance V reproduces the published +0.078329 (recon/03:378)', `${stance.V.toFixed(6)}`);
  ok(Math.abs(wall.V - 0.043099) < 1e-6,
    '0.1 m from the +x wall reproduces +0.043099 (recon/03:379)', `${wall.V.toFixed(6)}`);
}

// ===========================================================================
section('5. CLOSED LOOP — a real match with the filter on BOTH seats');
// ===========================================================================
let costRows = [];
{
  const sim = await createSim({ sceneUrl: `${ROOT}/assets/scene/asym/scene.xml` });
  const walk = await man.load(man.playerWalk().name);
  const ai = await man.load('asym_s2c_defender');

  const paths = {
    attacker: makeShieldPath({ filter, sim, robot: 'a', opponentRobot: 'b' }),
    defender: makeShieldPath({ filter, sim, robot: 'b', opponentRobot: 'a' }),
  };
  const match = createMatch({
    sim, game: 'asym', playerSeat: 'attacker',
    opponent: man.entry('asym_s2c_defender'),
    policies: { walk, ai },
    // A human RAMMING the defender: straight at it, full speed. This is the
    // moment the demo exists to show.
    input: { read: () => ({ vx: 3.0, vy: 0, wz: 0 }) },
    config: { countdownMs: 0 },
    actionPaths: paths,
  });
  match.reset();

  const before = sim.getGains('a');
  let steps = 0, nan = false, interv = { attacker: 0, defender: 0 };
  let maxAlpha = { attacker: 0, defender: 0 };
  let hud = match.hud();
  const t0 = Date.now();
  while (!hud.verdict && steps < 500) {
    hud = match.tick(20);
    steps += 1;
    for (const seat of ['attacker', 'defender']) {
      const f = hud.filter[seat];
      if (!f) continue;
      if (!Number.isFinite(f.value) || !Number.isFinite(f.alpha)) nan = true;
      if (f.alpha > 0) interv[seat] += 1;
      maxAlpha[seat] = Math.max(maxAlpha[seat], f.alpha);
      costRows.push(f.stepMs);
    }
    const st = match.state();
    for (const s of Object.values(st.seats)) if (!Number.isFinite(s.x)) nan = true;
  }
  const ms = Date.now() - t0;
  const st = match.state();

  console.log(`  terminal ${hud.verdict ? hud.verdict.terminal : '(none)'}  ` +
    `winner ${hud.verdict ? hud.verdict.winner : '-'}  ${steps} steps  ` +
    `${(ms / steps).toFixed(2)} ms/step`);
  console.log(`  attacker (you, ramming): ${interv.attacker}/${steps} steps intervened, ` +
    `max alpha ${maxAlpha.attacker.toFixed(3)}`);
  console.log(`  defender (S2C):          ${interv.defender}/${steps} steps intervened, ` +
    `max alpha ${maxAlpha.defender.toFixed(3)}`);
  console.log(`  final  attacker x ${st.seats.attacker.x.toFixed(2)}  ` +
    `defender x ${st.seats.defender.x.toFixed(2)}`);

  ok(!nan, 'no NaN in the value, the alpha or either pose');
  ok(!!hud.verdict, 'the episode reached a referee terminal', hud.verdict?.terminal);
  ok(interv.attacker + interv.defender > 0,
    'the certificate actually intervened at least once',
    `${interv.attacker + interv.defender} robot-steps`);
  ok(Math.abs(st.seats.attacker.x - (-1.4)) > 0.3,
    'the shielded player still locomotes (the filter is not a freeze)',
    `moved ${Math.abs(st.seats.attacker.x - (-1.4)).toFixed(2)} m`);
  const after = sim.getGains('a');
  ok(maxAbs(before.kp, after.kp) === 0 && after.kp[0] === 20 && after.kp[2] === 40,
    'gainBlend defaults OFF, so the plant stays at walk-soft 20/20/40 ' +
    '(game_qcbf_action.py:238)');

  // ...and the blend, when it is asked for.
  const p = makeShieldPath({ filter, sim, robot: 'a', opponentRobot: 'b', gainBlend: true });
  p.reset(sim.getJointPos('a'));
  const g0 = sim.getGains('a');
  ok(g0.kp[0] === 20, 'with gainBlend on, reset() writes walk-soft');
  for (let i = 0; i < 12; i++) p.step(new Float32Array(12));
  const g1 = sim.getGains('a');
  ok(g1.kp[0] >= 20 && g1.kp[0] <= 100 && g1.kp[2] === 2 * g1.kp[0],
    'and a stint of stepping leaves kp inside [20, 100] with calf = 2x hip',
    `kp ${g1.kp[0].toFixed(1)} / ${g1.kp[2].toFixed(1)}, alpha ${p.info().gainAlpha.toFixed(3)}`);
  sim.setGains('a', [20, 20, 40, 20, 20, 40, 20, 20, 40, 20, 20, 40], [1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2]);

  match.dispose?.();
  sim.dispose?.();
}

// ===========================================================================
section('6. COST — us per filtered robot per control step');
// ===========================================================================
{
  const cert = createCertificate(filter.nets);
  const shield = createShield(cert, filter.params);
  const obs = Float32Array.from(fx.obs[0]);
  const u = new Float64Array(12);

  const time = (fn, n) => {
    for (let i = 0; i < 50; i++) fn();
    const t0 = process.hrtime.bigint();
    for (let i = 0; i < n; i++) fn();
    return Number(process.hrtime.bigint() - t0) / n / 1000;
  };
  const N = BENCH ? 4000 : 400;
  const usCtrl = time(() => cert.fallback(obs), N);
  const usQ = time(() => cert.robustQ(obs, u), N);
  const floor = usCtrl + 2 * usQ;

  // The two fixture cases that really run the search.
  const ls = fx.cases.filter((c) => c.decision === 2);
  let lsUs = 0, lsIters = 0;
  for (const c of ls) {
    const o = Float32Array.from(c.obs);
    const ut = Float64Array.from(c.u_task);
    lsUs += time(() => shield.step(o, ut), Math.max(50, N / 8));
    lsIters += shield.step(o, ut).iters;
  }
  lsUs /= Math.max(ls.length, 1);
  lsIters /= Math.max(ls.length, 1);

  console.log(`  pi_shield(obs)                 ${usCtrl.toFixed(1)} us`);
  console.log(`  one robust_q (dstb + twin Q)   ${usQ.toFixed(1)} us`);
  console.log(`  FLOOR   (task passes)          ${floor.toFixed(1)} us / robot`);
  console.log(`  TYPICAL (line search, ${lsIters.toFixed(1)} iters) ${lsUs.toFixed(1)} us / robot`);
  console.log(`  CEILING (20 iters)             ${(floor + 20 * usQ).toFixed(1)} us / robot`);
  console.log(`  two filtered robots, worst case: ${(2 * (floor + 20 * usQ) / 1000).toFixed(1)} ms ` +
    `of the 20 ms control step`);
  if (costRows.length) {
    costRows.sort((a, b) => a - b);
    const q = (p) => costRows[Math.min(costRows.length - 1, Math.floor(p * costRows.length))];
    console.log(`  measured in the section-5 match: median ${q(0.5).toFixed(2)} ms, ` +
      `p95 ${q(0.95).toFixed(2)} ms, max ${costRows[costRows.length - 1].toFixed(2)} ms ` +
      `(per filtered robot, ${costRows.length} samples)`);
  }
  ok(2 * (floor + 20 * usQ) / 1000 < 20,
    'even the 20-iteration ceiling for BOTH robots fits in one 20 ms control step');
}

console.log(`\n${failures ? `\x1b[31m${failures} CHECK(S) FAILED\x1b[0m` : '\x1b[32mALL CHECKS PASSED\x1b[0m'}`);
process.exit(failures ? 1 : 0);

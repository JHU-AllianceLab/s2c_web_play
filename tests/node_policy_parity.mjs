#!/usr/bin/env node
// tests/node_policy_parity.mjs — the acceptance battery for the policy runtime.
//
//   node tests/node_policy_parity.mjs            # everything
//   node tests/node_policy_parity.mjs --quick    # skip the live-MuJoCo section
//   node tests/node_policy_parity.mjs --bench    # + a longer timing run
//
// Sections
//   1  POLICY PARITY     every policy in tests/parity.json reproduced to < 1e-5
//   2  TIMING            us per forward, 47-D and 60-D, in node
//   3  FRAME UNITS       quat_apply_inverse / yaw / gait clock / seat direction
//   4  LIVE SIM          the same conventions on a real MuJoCo scene
//   5  OBS GROUND TRUTH  60-D x 2 seats x 2 games and 47-D, diffed against the
//                        LIVE mjlab envs (tests/obs_fixture.json)
//   6  ACTION GROUND TRUTH  WalkAffine and the v25 integrator, same source
//   7  INTEGRATOR UNITS  seeding, the slew-rate identity, the limit clamp
//   8  PLUMBING          manifest resolution, rotation flags, scene.json limits
//
// Exit code is non-zero if anything fails. The two ground-truth sections skip
// (loudly) when tests/obs_fixture.json is absent — regenerate it with
//   PYTHONPATH=/home/ray/Disk_ext/Go2/Project/unitree_rl_mjlab \
//     /home/ray/Disk_ext/Go2/envs/mjlab_venv/bin/python tools/export_obs_fixture.py

import { readFile, access } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

import { loadPolicy, loadManifest, buildPolicy } from '../app/policy.js';
import { GAMES, CMD_BOX } from '../app/config.js';
import {
  walkObs,
  gameObs,
  gaitPhase,
  quatApplyInverse,
  yawFromQuat,
  seatDirection,
  WALK_OBS_LAYOUT,
  GAME_OBS_LAYOUT,
  WALK_OBS_DIM,
  GAME_OBS_DIM,
} from '../app/obs.js';
import {
  WalkAffine,
  IncrementIntegrator,
  ACTION_SCALE,
  INCREMENT_SCALE,
  ACTION_SMOOTHING,
  SOFT_JOINT_POS_LIMIT_LO,
  SOFT_JOINT_POS_LIMIT_HI,
} from '../app/action.js';

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, '..');
const ARGV = process.argv.slice(2);
const QUICK = ARGV.includes('--quick');
const BENCH = ARGV.includes('--bench');

const PARITY_TOL = 1e-5; // tests/parity.json `tolerance`, and DESIGN.md section 8.1
const OBS_TOL = 1e-5; // float32 torch vs float64-accumulated JS
const ACT_TOL = 1e-5;

let pass = 0;
let fail = 0;
const failures = [];

function ok(cond, label, detail = '') {
  if (cond) {
    pass++;
    console.log(`  \x1b[32mok\x1b[0m   ${label}${detail ? '   ' + detail : ''}`);
  } else {
    fail++;
    failures.push(label);
    console.log(`  \x1b[31mFAIL\x1b[0m ${label}${detail ? '   ' + detail : ''}`);
  }
}

function near(a, b, tol, label) {
  const d = Math.abs(a - b);
  ok(d <= tol, label, `|d| = ${d.toExponential(3)} (tol ${tol.toExponential(0)})`);
  return d;
}

function maxAbsDiff(a, b) {
  let m = 0;
  let at = -1;
  for (let i = 0; i < a.length; i++) {
    const d = Math.abs(a[i] - b[i]);
    if (d > m) {
      m = d;
      at = i;
    }
  }
  return { m, at };
}

function section(title) {
  console.log(`\n\x1b[1m${title}\x1b[0m`);
}

async function exists(p) {
  try {
    await access(p);
    return true;
  } catch {
    return false;
  }
}

// ===========================================================================
// 1 + 2. policy parity and timing
// ===========================================================================

async function sectionPolicies() {
  section('1. POLICY PARITY — tests/parity.json, gate 1e-5');
  const parity = JSON.parse(await readFile(join(ROOT, 'tests/parity.json'), 'utf8'));
  const tol = parity.tolerance ?? PARITY_TOL;
  const names = Object.keys(parity.policies);

  const loaded = new Map();
  let worst = 0;
  let worstName = '';

  for (const name of names) {
    const e = parity.policies[name];
    const pol = await loadPolicy(join(ROOT, `assets/policies/${name}.json`));
    loaded.set(name, pol);

    let m = 0;
    let shapeOk = pol.obsDim === e.obs_dim && pol.actDim === e.act_dim;
    const out = new Float32Array(pol.actDim);
    for (let r = 0; r < e.obs.length; r++) {
      pol.forward(Float32Array.from(e.obs[r]), out);
      const d = maxAbsDiff(out, e.act[r]);
      if (d.m > m) m = d.m;
    }
    if (m > worst) {
      worst = m;
      worstName = name;
    }
    ok(
      shapeOk && m < tol,
      `${name.padEnd(30)} ${e.obs.length} vectors`,
      `max|d| ${m.toExponential(3)}${shapeOk ? '' : '  SHAPE MISMATCH'}`,
    );
    // The rotation flag must survive the round trip: a `rotation_baked` policy
    // is the one thing app/match.js must not rotate again.
    if (e.rotation_baked !== undefined) {
      ok(
        pol.rotationBaked === e.rotation_baked,
        `${name.padEnd(30)} rotation_baked flag`,
        `${pol.rotationBaked}`,
      );
    }
  }
  console.log(`  worst over all ${names.length} policies: ${worst.toExponential(3)} (${worstName})`);

  section('2. TIMING — us per deterministic forward (node, single thread)');
  const rows = [];
  for (const [name, pol] of loaded) {
    const e = parity.policies[name];
    const obs = Float32Array.from(e.obs[0]);
    const out = new Float32Array(pol.actDim);
    for (let i = 0; i < 500; i++) pol.forward(obs, out); // warm up the JIT
    const N = BENCH ? 20000 : 4000;
    const t0 = process.hrtime.bigint();
    for (let i = 0; i < N; i++) pol.forward(obs, out);
    const t1 = process.hrtime.bigint();
    rows.push({ name, dim: pol.obsDim, us: Number(t1 - t0) / N / 1000 });
  }
  const walkRow = rows.find((r) => r.dim === 47);
  const gameRows = rows.filter((r) => r.dim === 60);
  const gmin = Math.min(...gameRows.map((r) => r.us));
  const gmax = Math.max(...gameRows.map((r) => r.us));
  console.log(`  47-D walk   ${walkRow.us.toFixed(1)} us`);
  console.log(`  60-D game   ${gmin.toFixed(1)} - ${gmax.toFixed(1)} us  (${gameRows.length} policies)`);
  const perStep = walkRow.us + gmax;
  console.log(
    `  one control step = 1 walk + 1 opponent = ${perStep.toFixed(1)} us ` +
      `= ${((perStep / 20000) * 100).toFixed(2)} % of the 20 ms budget`,
  );
  ok(perStep < 20000 * 0.25, 'two forwards fit well inside one 50 Hz control step');

  return loaded;
}

// ===========================================================================
// 3. frame conventions — pure unit tests
// ===========================================================================

/** quaternion (w-first) for a rotation of `a` about +z */
const qz = (a) => [Math.cos(a / 2), 0, 0, Math.sin(a / 2)];
/** quaternion (w-first) for a rotation of `a` about +y */
const qy = (a) => [Math.cos(a / 2), 0, Math.sin(a / 2), 0];
/** quaternion (w-first) for a rotation of `a` about +x */
const qx = (a) => [Math.cos(a / 2), Math.sin(a / 2), 0, 0];

function sectionFrames() {
  section('3. FRAME CONVENTIONS — unit');

  // --- quat_apply_inverse: world -> body -----------------------------------
  // At yaw +90 deg the body +x axis points along world +y, so a world +x vector
  // must read as body -y.
  {
    const v = quatApplyInverse(qz(Math.PI / 2), [1, 0, 0]);
    ok(
      Math.abs(v[0]) < 1e-12 && Math.abs(v[1] + 1) < 1e-12 && Math.abs(v[2]) < 1e-12,
      'yaw +90: world (1,0,0) -> body (0,-1,0)',
      `(${v[0].toFixed(6)}, ${v[1].toFixed(6)}, ${v[2].toFixed(6)})`,
    );
  }
  {
    const v = quatApplyInverse(qz(Math.PI / 2), [0, 1, 0]);
    ok(
      Math.abs(v[0] - 1) < 1e-12 && Math.abs(v[1]) < 1e-12,
      'yaw +90: world (0,1,0) -> body (1,0,0)',
      `(${v[0].toFixed(6)}, ${v[1].toFixed(6)}, ${v[2].toFixed(6)})`,
    );
  }
  // Identity and a round trip through an arbitrary rotation.
  {
    const v = quatApplyInverse([1, 0, 0, 0], [3, -2, 7]);
    ok(v[0] === 3 && v[1] === -2 && v[2] === 7, 'identity quaternion is a no-op');
  }
  {
    // |R^T v| == |v| for a unit quaternion (rotation preserves length).
    const q = (() => {
      const a = [0.3, -0.7, 1.1];
      const c = Math.cos, s = Math.sin;
      // compose z*y*x
      const [cz, sz] = [c(a[2] / 2), s(a[2] / 2)];
      const [cy, sy] = [c(a[1] / 2), s(a[1] / 2)];
      const [cx, sx] = [c(a[0] / 2), s(a[0] / 2)];
      return [
        cz * cy * cx + sz * sy * sx,
        cz * cy * sx - sz * sy * cx,
        cz * sy * cx + sz * cy * sx,
        sz * cy * cx - cz * sy * sx,
      ];
    })();
    const v = [1.7, -0.4, 2.2];
    const b = quatApplyInverse(q, v);
    const n0 = Math.hypot(...v);
    const n1 = Math.hypot(b[0], b[1], b[2]);
    near(n1, n0, 1e-12, 'quat_apply_inverse preserves length');
  }

  // --- projected gravity ---------------------------------------------------
  {
    const g = quatApplyInverse([1, 0, 0, 0], [0, 0, -1]);
    ok(
      g[0] === 0 && g[1] === 0 && g[2] === -1,
      'level robot: projected_gravity == (0,0,-1) (UNIT, not 9.81)',
    );
  }
  {
    // pitch +90 deg (nose up): gravity should read along body +x.
    const g = quatApplyInverse(qy(Math.PI / 2), [0, 0, -1]);
    ok(
      Math.abs(g[0] - 1) < 1e-12 && Math.abs(g[2]) < 1e-12,
      'pitch +90: projected_gravity -> (1,0,0)',
      `(${g[0].toFixed(6)}, ${g[1].toFixed(6)}, ${g[2].toFixed(6)})`,
    );
    // ... and the training fall metric acos(-g_z) is then 90 deg.
    near((Math.acos(-g[2]) * 180) / Math.PI, 90, 1e-9, 'tilt from projected_gravity = 90 deg');
  }
  {
    // roll 70 deg is exactly the training fall threshold.
    const g = quatApplyInverse(qx((70 * Math.PI) / 180), [0, 0, -1]);
    near((Math.acos(-g[2]) * 180) / Math.PI, 70, 1e-9, 'roll 70 deg reads as 70 deg tilt');
  }

  // --- yaw -----------------------------------------------------------------
  for (const a of [0, 0.3, -1.2, Math.PI / 2, 3.1381, -3.0]) {
    const y = yawFromQuat(qz(a));
    if (Math.abs(y - a) > 1e-12) {
      ok(false, `yawFromQuat(qz(${a})) == ${a}`, `got ${y}`);
    }
  }
  ok(true, 'yawFromQuat inverts a pure yaw quaternion over 6 angles');
  {
    // yaw must be the euler-XYZ yaw even with roll and pitch present; check it
    // against the heading formula mjlab's heading_w uses (atan2 of body +x in
    // world), which app/physics.js `yaw()` implements.
    const q = (() => {
      const [r, p, y] = [0.4, -0.25, 1.9];
      const [cr, sr] = [Math.cos(r / 2), Math.sin(r / 2)];
      const [cp, sp] = [Math.cos(p / 2), Math.sin(p / 2)];
      const [cy, sy] = [Math.cos(y / 2), Math.sin(y / 2)];
      return [
        cy * cp * cr + sy * sp * sr,
        cy * cp * sr - sy * sp * cr,
        cy * sp * cr + sy * cp * sr,
        sy * cp * cr - cy * sp * sr,
      ];
    })();
    const heading = Math.atan2(
      2 * (q[1] * q[2] + q[0] * q[3]),
      1 - 2 * (q[2] * q[2] + q[3] * q[3]),
    );
    near(yawFromQuat(q), heading, 1e-15, 'euler-XYZ yaw == mjlab heading_w (roll+pitch present)');
  }

  // --- the trot clock ------------------------------------------------------
  {
    const p0 = gaitPhase(0);
    ok(
      Math.abs(p0[0]) < 1e-6 && Math.abs(p0[1] - 1) < 1e-6,
      'phase at episode_length_buf = 0 is (0, 1)',
      `(${p0[0].toExponential(2)}, ${p0[1].toFixed(6)})`,
    );
    const p15 = gaitPhase(15); // 0.30 s = half of the 0.6 s period
    ok(
      Math.abs(p15[0]) < 1e-5 && Math.abs(p15[1] + 1) < 1e-6,
      'phase at step 15 (0.30 s) is (0, -1)',
      `(${p15[0].toExponential(2)}, ${p15[1].toFixed(6)})`,
    );
    // No integer step lands on the quarter period (0.15 s = 7.5 steps), so
    // check step 8 against the closed form instead.
    const p8 = gaitPhase(8);
    const a8 = (2 * Math.PI * ((8 * 0.02) % 0.6)) / 0.6;
    ok(
      Math.abs(p8[0] - Math.sin(a8)) < 1e-6 && Math.abs(p8[1] - Math.cos(a8)) < 1e-6,
      'phase at step 8 matches sin/cos(2*pi * 0.16/0.6)',
      `(${p8[0].toFixed(6)}, ${p8[1].toFixed(6)})`,
    );
    const p30 = gaitPhase(30); // one full period
    ok(
      Math.abs(p30[0]) < 1e-5 && Math.abs(p30[1] - 1) < 1e-5,
      'phase wraps at step 30 (0.60 s) back to (0, 1)',
      `(${p30[0].toExponential(2)}, ${p30[1].toFixed(6)})`,
    );
    const p31 = gaitPhase(31);
    const p1 = gaitPhase(1);
    near(p31[0], p1[0], 1e-6, 'clock is periodic: step 31 == step 1 (sin)');
    ok(gaitPhase(7, true)[0] === 0 && gaitPhase(7, true)[1] === 0, 'stand gate zeroes the phase');
  }

  // --- seat direction ------------------------------------------------------
  ok(seatDirection('sym', 'A') === 1 && seatDirection('sym', 'B') === -1, 'sym direction +1 / -1');
  ok(
    seatDirection('asym', 'attacker') === 1 && seatDirection('asym', 'defender') === 1,
    'asym direction +1 for BOTH seats (one line, at env x = +1.9)',
  );

  // --- layout bookkeeping --------------------------------------------------
  {
    const wsum = WALK_OBS_LAYOUT.reduce((s, t) => s + t.dim, 0);
    const gsum = GAME_OBS_LAYOUT.reduce((s, t) => s + t.dim, 0);
    ok(wsum === WALK_OBS_DIM, 'walk layout sums to 47');
    ok(gsum === GAME_OBS_DIM, 'game layout sums to 60');
    let at = 0;
    let contiguous = true;
    for (const t of GAME_OBS_LAYOUT) {
      if (t.at !== at) contiguous = false;
      at += t.dim;
    }
    ok(contiguous, 'game layout offsets are contiguous and in order');
  }
}

// ===========================================================================
// 4. the same conventions on a live MuJoCo scene
// ===========================================================================

async function sectionLiveSim() {
  section('4. LIVE SIM — frame conventions against app/physics.js');
  const scenePath = join(ROOT, 'assets/scene/sym/scene.xml');
  if (!(await exists(scenePath))) {
    console.log('  \x1b[33mskip\x1b[0m  assets/scene/sym/scene.xml is missing');
    return;
  }
  const { createSim } = await import('../app/physics.js');
  const sim = await createSim({ sceneUrl: scenePath });

  const ZERO = new Float64Array(12);

  // Level, at the origin, facing +x.
  sim.setPose('a', { x: 0.5, y: -0.25, yaw: 0, z: 0.32 });
  sim.setPose('b', { x: 1.5, y: -0.25, yaw: Math.PI, z: 0.32 });
  sim.forward();
  {
    const g = sim.projectedGravity('a');
    ok(
      Math.abs(g[0]) < 1e-9 && Math.abs(g[1]) < 1e-9 && Math.abs(g[2] + 1) < 1e-9,
      'level robot on the real sim: projected_gravity == (0,0,-1)',
      `(${g[0].toExponential(1)}, ${g[1].toExponential(1)}, ${g[2].toFixed(9)})`,
    );
    near(yawFromQuat(sim.getBase('a').quat), sim.yaw('a'), 1e-12,
      'obs.js yawFromQuat == physics.js yaw()  (seat a)');
  }

  // Yaw seat a by +90 deg with the opponent 1 m along world +x.
  sim.setPose('a', { x: 0, y: 0, yaw: Math.PI / 2, z: 0.32 });
  sim.setPose('b', { x: 1, y: 0, yaw: 0, z: 0.32 });
  sim.setBaseVel('b', { linVelW: [2, 0, 0] });
  sim.forward();
  {
    const o = gameObs(sim, 'a', 'b', -1, 1.9, 0, ZERO);
    // arena_pose
    near(o[42], 0, 1e-9, 'arena_pose.x');
    near(o[43], 0, 1e-9, 'arena_pose.y');
    near(o[44], 0, 1e-7, 'arena_pose cos yaw at +90 deg == 0');
    near(o[45], 1, 1e-7, 'arena_pose sin yaw at +90 deg == 1');
    // rel_pos: opponent at world +x, I face world +y  =>  body -y
    near(o[46], 0, 1e-7, 'rel_pos.x (opponent abeam)');
    near(o[47], -1, 1e-7, 'rel_pos.y == -1 (opponent on my right)');
    // rel_yaw: yaw_opp - yaw_me = -90 deg
    near(o[49], 0, 1e-7, 'rel_yaw cos == 0');
    near(o[50], -1, 1e-7, 'rel_yaw sin == -1');
    // rel_vel: opponent moving world +x at 2 m/s  =>  body (0,-2,0)
    near(o[51], 0, 1e-6, 'rel_vel.x');
    near(o[52], -2, 1e-6, 'rel_vel.y == -2 (opponent world +x, me facing +y)');
    // line, for seat B (dir = -1) at x = 0:  1.9 - (-1)(0) = 1.9
    near(o[54], 1.9, 1e-6, 'line at x = 0, dir = -1');   // float32 storage: fround(1.9)
    ok(o[55] === 0 && o[56] === 0 && o[57] === 0, 'cmd slot is exactly zero');
    near(o[58], 0, 1e-6, 'phase sin at step 0');
    near(o[59], 1, 1e-6, 'phase cos at step 0');
    // proprio must agree with the sim reads it came from
    const jp = sim.getJointPos('a');
    const jv = sim.getJointVel('a');
    const d = sim.defaultJointPos;
    let pm = 0;
    for (let i = 0; i < 12; i++) {
      pm = Math.max(pm, Math.abs(o[6 + i] - (jp[i] - d[i])), Math.abs(o[18 + i] - jv[i]));
    }
    ok(pm < 1e-6, 'joint_pos-default and joint_vel blocks match the sim', `max|d| ${pm.toExponential(2)}`);
  }

  // The `line` term for both games, at a handful of x, straight off the sim.
  for (const [x, dir, want] of [
    [-1.4, 1, 3.3],   // asym attacker spawn: 1.9 - (-1.4) = 3.3
    [0.75, 1, 1.15],  // asym defender spawn
    [1.9, 1, 0],      // exactly on the line
    [2.1775, -1, 4.0775], // sym seat B at the locked demo spawn
    [-2.1431, 1, 4.0431], // sym seat A at the locked demo spawn
  ]) {
    sim.setPose('a', { x, y: 0, yaw: 0, z: 0.32 });
    sim.forward();
    const o = gameObs(sim, 'a', 'b', dir, 1.9, 0, ZERO);
    near(o[54], want, 1e-6, `line at x = ${x}, dir = ${dir > 0 ? '+1' : '-1'}`);
  }

  // Walk obs on the live sim: block placement + the stand gate.
  sim.setPose('a', { x: 0, y: 0, yaw: 0, z: 0.32 });
  sim.forward();
  {
    const cmd = [1.25, -0.5, 0.75];
    const la = Float64Array.from({ length: 12 }, (_, i) => 0.1 * (i + 1));
    const o = walkObs(sim, 'a', cmd, 3, la);
    ok(o.length === 47, 'walkObs returns 47 values');
    ok(o[6] === Math.fround(1.25) && o[7] === Math.fround(-0.5) && o[8] === Math.fround(0.75),
      'walk command block at 6:9');
    const g = sim.projectedGravity('a');
    ok(Math.abs(o[3] - g[0]) < 1e-7 && Math.abs(o[5] - g[2]) < 1e-7, 'walk projected_gravity at 3:6');
    let lm = 0;
    for (let i = 0; i < 12; i++) lm = Math.max(lm, Math.abs(o[35 + i] - la[i]));
    ok(lm < 1e-6, 'walk last_action block at 35:47', `max|d| ${lm.toExponential(2)}`);
    const ph = gaitPhase(3);
    near(o[9], ph[0], 0, 'walk phase sin matches the clock');
    const gated = walkObs(sim, 'a', [0.02, 0.01, 0.0], 3, la);
    ok(gated[9] === 0 && gated[10] === 0, 'walk phase is ZEROED when |cmd| < 0.1');
    const live = walkObs(sim, 'a', [0.0, 0.0, 0.11], 3, la);
    ok(live[9] !== 0 || live[10] !== 0, 'the gate uses the norm over all THREE dims (wz alone unzeroes it)');
  }

  sim.dispose();
}

// ===========================================================================
// 5 + 6. ground truth from the live mjlab envs
// ===========================================================================

function stubSim(robots, defaultJointPos) {
  const g = (r) => {
    const s = robots[r];
    if (!s) throw new Error(`fixture has no robot "${r}"`);
    return s;
  };
  return {
    defaultJointPos,
    getBase: (r) => {
      const s = g(r);
      return { pos: s.pos, quat: s.quat, linVelW: s.linVelW, angVelB: s.angVelB };
    },
    projectedGravity: (r) => g(r).projGrav,
    getJointPos: (r) => g(r).jointPos,
    getJointVel: (r) => g(r).jointVel,
  };
}

async function sectionGroundTruth() {
  const fx = join(ROOT, 'tests/obs_fixture.json');
  if (!(await exists(fx))) {
    section('5 + 6. GROUND TRUTH');
    console.log('  \x1b[33mskip\x1b[0m  tests/obs_fixture.json is missing — run tools/export_obs_fixture.py');
    return;
  }
  const F = JSON.parse(await readFile(fx, 'utf8'));

  section('5. OBSERVATION GROUND TRUTH — diffed against the LIVE mjlab envs');

  for (const game of ['asym', 'sym']) {
    const G = F.games[game];
    const meta = G.meta;
    // The env-local scoring line: field_center.x + the FIELD-local line_x.
    // asym 0.2 + 1.7, sym 0.0 + 1.9 — both 1.9, which is GAMES.<game>.lineX.
    const lineX = meta.fieldCenter[0] + meta.lineX;
    near(lineX, 1.9, 1e-9, `${game}: env-local scoring line is 1.9`);

    // The live term order must be the layout this module hardcodes.
    const wantOrder = GAME_OBS_LAYOUT.map((t) => t.term);
    const gotOrder = meta.obsTermOrder.attacker_actor.map((n) =>
      n.startsWith('rel_pos_') ? 'rel_pos_opp'
        : n.startsWith('rel_yaw_') ? 'rel_yaw_opp'
        : n.startsWith('rel_vel_') ? 'rel_vel_opp'
        : n.endsWith('_line') ? 'line'
        : n === 'actions' ? 'actions'
        : n,
    );
    ok(
      JSON.stringify(gotOrder) === JSON.stringify(wantOrder),
      `${game}: live actor term ORDER == GAME_OBS_LAYOUT`,
      gotOrder.join(','),
    );
    ok(
      JSON.stringify(meta.obsTermDims.attacker_actor) === JSON.stringify(GAME_OBS_LAYOUT.map((t) => t.dim)),
      `${game}: live actor term DIMS == GAME_OBS_LAYOUT`,
    );

    for (const [robot, opp] of [['a', 'b'], ['b', 'a']]) {
      const dir = meta.direction[robot];
      ok(
        dir === seatDirection(game, robot === 'a' ? (game === 'sym' ? 'A' : 'attacker') : (game === 'sym' ? 'B' : 'defender')),
        `${game}/${robot}: seatDirection() == the live scenario's direction`,
        `${dir > 0 ? '+1' : '-1'}`,
      );

      let worst = 0;
      let worstAt = -1;
      let worstStep = -1;
      const perTerm = new Map(GAME_OBS_LAYOUT.map((t) => [t.term, 0]));
      for (let k = 0; k < G.steps.length; k++) {
        const st = G.steps[k];
        const sim = stubSim(st.robots, meta.defaultJointPos);
        const got = gameObs(
          sim,
          robot,
          opp,
          dir,
          lineX,
          st.episodeLengthBuf,
          st.rawAction[robot],
          { envOrigin: meta.envOrigin },
        );
        const want = st.actor[robot];
        const d = maxAbsDiff(got, want);
        if (d.m > worst) {
          worst = d.m;
          worstAt = d.at;
          worstStep = k;
        }
        for (const t of GAME_OBS_LAYOUT) {
          let m = perTerm.get(t.term);
          for (let i = t.at; i < t.at + t.dim; i++) m = Math.max(m, Math.abs(got[i] - want[i]));
          perTerm.set(t.term, m);
        }
      }
      const worstTerm = GAME_OBS_LAYOUT.find((t) => worstAt >= t.at && worstAt < t.at + t.dim);
      ok(
        worst < OBS_TOL,
        `${game}/${robot}: 60-D obs over ${G.steps.length} live frames`,
        `max|d| ${worst.toExponential(3)} at dim ${worstAt} (${worstTerm ? worstTerm.term : '?'}), step ${worstStep}`,
      );
      if (worst >= OBS_TOL || BENCH) {
        for (const [t, m] of perTerm) console.log(`         ${t.padEnd(18)} ${m.toExponential(3)}`);
      }
    }
  }

  // --- the walk observation ------------------------------------------------
  {
    const W = F.walk;
    const wantOrder = WALK_OBS_LAYOUT.map((t) => t.term);
    const gotOrder = W.meta.obsTermOrder.map((n) => (n === 'actions' ? 'actions' : n));
    ok(
      JSON.stringify(gotOrder) === JSON.stringify(wantOrder),
      'walk: live actor term ORDER == WALK_OBS_LAYOUT',
      gotOrder.join(','),
    );
    ok(
      JSON.stringify(W.meta.obsTermDims) === JSON.stringify(WALK_OBS_LAYOUT.map((t) => t.dim)),
      'walk: live actor term DIMS == WALK_OBS_LAYOUT',
    );
    ok(
      W.meta.enableCorruption === false,
      'walk: the recorded actor group has observation noise OFF (env_cfgs.py:124)',
    );
    ok(
      JSON.stringify(W.meta.commandRanges.lin_vel_x) === JSON.stringify([-1.5, 3.0]) &&
        JSON.stringify(W.meta.commandRanges.lin_vel_y) === JSON.stringify([-1.0, 1.0]) &&
        JSON.stringify(W.meta.commandRanges.ang_vel_z) === JSON.stringify([-2.0, 2.0]),
      'walk: the live command box is CMD_BOX vx[-1.5,3] vy[-1,1] wz[-2,2]',
    );

    // The symmetric game narrows what the human may ask for (config GAMES.sym
    // playerCmd): both dogs race for a line there, and the walker the human
    // drives is faster than the policies. Narrowing only -- never wider than
    // the trained box, or the walker is extrapolating.
    const symCmd = GAMES.sym.playerCmd;
    ok(
      symCmd
        && symCmd.vx[1] <= CMD_BOX.vx[1] && symCmd.vx[0] >= CMD_BOX.vx[0]
        && symCmd.vy[1] <= CMD_BOX.vy[1] && symCmd.wz[1] <= CMD_BOX.wz[1],
      'sym: the player command limit is inside the trained box',
      `vx ${symCmd.vx.join('..')} cruise ${symCmd.cruiseFrac}`,
    );
    ok(
      Math.abs(symCmd.vx[1] * symCmd.cruiseFrac - 1.15) < 0.02,
      'sym: a held W asks for 1.15 m/s (the speed the opponents survive head-on)',
    );

    let worst = 0;
    let worstAt = -1;
    let worstStep = -1;
    let gatedFrames = 0;
    for (let k = 0; k < W.steps.length; k++) {
      const st = W.steps[k];
      const sim = stubSim({ a: st.robot }, W.meta.defaultJointPos);
      const got = walkObs(sim, 'a', st.cmd, st.episodeLengthBuf, st.rawAction);
      const want = st.actor;
      const d = maxAbsDiff(got, want);
      if (d.m > worst) {
        worst = d.m;
        worstAt = d.at;
        worstStep = k;
      }
      if (Math.hypot(st.cmd[0], st.cmd[1], st.cmd[2]) < 0.1) {
        gatedFrames++;
        if (want[9] !== 0 || want[10] !== 0) ok(false, `walk: live env did NOT gate the phase at step ${k}`);
      }
    }
    const worstTerm = WALK_OBS_LAYOUT.find((t) => worstAt >= t.at && worstAt < t.at + t.dim);
    ok(
      worst < OBS_TOL,
      `walk: 47-D obs over ${W.steps.length} live frames`,
      `max|d| ${worst.toExponential(3)} at dim ${worstAt} (${worstTerm ? worstTerm.term : '?'}), step ${worstStep}`,
    );
    ok(gatedFrames >= 2, `walk: the stand gate fired on ${gatedFrames} live frames (>= 2)`);
  }

  // =========================================================================
  section('6. ACTION GROUND TRUTH — the same live envs');

  for (const game of ['asym', 'sym']) {
    const G = F.games[game];
    const meta = G.meta;
    ok(
      meta.increment.termClass === 'WalkAffineIncrementJointPositionAction',
      `${game}: the live action term is WalkAffineIncrementJointPositionAction`,
    );
    ok(
      meta.increment.scale === ACTION_SCALE &&
        meta.increment.incrementScale === INCREMENT_SCALE &&
        meta.increment.smoothing === ACTION_SMOOTHING &&
        meta.increment.delayMaxSteps === 0,
      `${game}: live integrator constants == action.js`,
      `scale ${meta.increment.scale}, s ${meta.increment.incrementScale}, alpha ${meta.increment.smoothing}, delay ${meta.increment.delayMaxSteps}`,
    );
    {
      const dl = maxAbsDiff(meta.softLimitLo, SOFT_JOINT_POS_LIMIT_LO);
      const dh = maxAbsDiff(meta.softLimitHi, SOFT_JOINT_POS_LIMIT_HI);
      ok(
        dl.m < 1e-6 && dh.m < 1e-6,
        `${game}: live soft joint limits == action.js constants`,
        `max|d| ${Math.max(dl.m, dh.m).toExponential(2)}`,
      );
    }

    for (const robot of ['a', 'b']) {
      // The seeding rule: the post-reset target IS the measured joint_pos.
      const sc = G.seedCheck[robot];
      const sd = maxAbsDiff(sc.target, sc.measuredJointPos);
      ok(sd.m === 0, `${game}/${robot}: post-reset target == MEASURED joint_pos`, `max|d| ${sd.m}`);

      const it = new IncrementIntegrator(
        meta.defaultJointPos,
        meta.softLimitLo,
        meta.softLimitHi,
      );
      it.reset(sc.measuredJointPos);
      let worst = 0;
      let worstStep = -1;
      for (let k = 1; k < G.steps.length; k++) {
        const st = G.steps[k];
        // The env stores the raw action it was given; assert we replay the same one.
        const ra = maxAbsDiff(st.rawAction[robot], st.actionIn[robot]);
        if (ra.m > 1e-6) ok(false, `${game}/${robot}: raw_action != the action passed at step ${k}`);
        const got = it.step(st.actionIn[robot]);
        const d = maxAbsDiff(got, st.target[robot]);
        if (d.m > worst) {
          worst = d.m;
          worstStep = k;
        }
      }
      ok(
        worst < ACT_TOL,
        `${game}/${robot}: v25 target over ${G.steps.length - 1} live steps`,
        `max|d| ${worst.toExponential(3)} at step ${worstStep}`,
      );
    }
  }

  {
    const W = F.walk;
    ok(W.meta.termClass === 'JointPositionAction', 'walk: the live action term is a plain JointPositionAction');
    ok(W.meta.actionScale === ACTION_SCALE && W.meta.useDefaultOffset === true,
      'walk: scale 0.25 with use_default_offset');
    const wa = new WalkAffine(W.meta.defaultJointPos);
    let worst = 0;
    for (let k = 1; k < W.steps.length; k++) {
      const st = W.steps[k];
      const got = wa.step(st.actionIn);
      const d = maxAbsDiff(got, st.processed);
      if (d.m > worst) worst = d.m;
    }
    ok(worst < ACT_TOL, `walk: q_des over ${W.steps.length - 1} live steps`, `max|d| ${worst.toExponential(3)}`);
  }
}

// ===========================================================================
// 7. integrator unit tests
// ===========================================================================

function sectionIntegrator() {
  section('7. INTEGRATOR — unit');

  const DEF = [-0.1, 0.9, -1.8, 0.1, 0.9, -1.8, -0.1, 0.9, -1.8, 0.1, 0.9, -1.8];
  const LO = SOFT_JOINT_POS_LIMIT_LO;
  const HI = SOFT_JOINT_POS_LIMIT_HI;
  const A0 = new Float64Array(12);

  // --- WalkAffine ----------------------------------------------------------
  {
    const wa = new WalkAffine(DEF);
    const q = wa.step(A0);
    ok(maxAbsDiff(q, DEF).m === 0, 'WalkAffine: a = 0 gives exactly the default pose');
    const a = Float64Array.from({ length: 12 }, (_, i) => (i % 2 ? 2 : -2));
    const q2 = wa.step(a);
    let m = 0;
    for (let i = 0; i < 12; i++) m = Math.max(m, Math.abs(q2[i] - (DEF[i] + 0.25 * a[i])));
    ok(m === 0, 'WalkAffine: q_des = default + 0.25 a, exactly, with NO clipping');
  }

  // --- the seeding rule ----------------------------------------------------
  {
    const it = new IncrementIntegrator(DEF, LO, HI);
    let threw = false;
    try {
      it.step(A0);
    } catch {
      threw = true;
    }
    ok(threw, 'IncrementIntegrator.step() before reset() THROWS (the march-in-place trap)');
    const measured = DEF.map((v, i) => v + 0.05 * Math.sin(i));
    it.reset(measured);
    ok(maxAbsDiff(it.target, measured).m === 0, 'reset(measured) seeds the target from the measurement');
  }

  // --- the slew-rate identity ---------------------------------------------
  // Inside the band the alpha in the EMA exactly cancels the (s*alpha) in the
  // integrator inverse, so one step lands ON q_des.
  {
    const it = new IncrementIntegrator(DEF, LO, HI);
    const seed = DEF.slice();
    it.reset(seed);
    // ask for +0.10 rad on every joint: |q_des - target| = 0.10 <= 0.15
    const a = Float64Array.from({ length: 12 }, () => 0.4); // 0.25*0.4 = +0.10
    const t1 = Float64Array.from(it.step(a));
    let m = 0;
    for (let i = 0; i < 12; i++) m = Math.max(m, Math.abs(t1[i] - (DEF[i] + 0.1)));
    ok(m < 1e-12, 'inside the band: one step lands EXACTLY on q_des', `max|d| ${m.toExponential(2)}`);
  }
  {
    const it = new IncrementIntegrator(DEF, LO, HI);
    it.reset(DEF);
    // ask for +0.50 rad: saturated, so the target must move exactly 0.15 rad.
    const a = Float64Array.from({ length: 12 }, () => 2.0); // 0.25*2 = +0.50
    const t1 = Float64Array.from(it.step(a));
    let m = 0;
    for (let i = 0; i < 12; i++) m = Math.max(m, Math.abs(t1[i] - (DEF[i] + 0.15)));
    ok(m < 1e-12, 'saturated: the target moves exactly alpha*s = 0.15 rad/step', `max|d| ${m.toExponential(2)}`);
    const t2 = Float64Array.from(it.step(a));
    let m2 = 0;
    for (let i = 0; i < 12; i++) m2 = Math.max(m2, Math.abs(t2[i] - (DEF[i] + 0.30)));
    ok(m2 < 1e-12, 'saturated: the second step adds another 0.15 rad');
  }

  // --- a = 0 with the WALK AFFINE is NOT a hold ----------------------------
  // (the "zero action holds" line in DESIGN.md section 5 describes the PARENT
  //  term; the shipped game arms use the affine subclass)
  {
    const it = new IncrementIntegrator(DEF, LO, HI);
    const off = DEF.map((v) => v + 0.4);
    it.reset(off);
    const t1 = Float64Array.from(it.step(A0));
    let moved = 0;
    for (let i = 0; i < 12; i++) moved = Math.max(moved, Math.abs(t1[i] - off[i]));
    ok(
      Math.abs(moved - 0.15) < 1e-12,
      'affine mode: a = 0 slews the target toward the DEFAULT pose at 0.15 rad/step',
      `moved ${moved.toFixed(6)}`,
    );
    const parent = new IncrementIntegrator(DEF, LO, HI, { affine: false });
    parent.reset(off);
    const p1 = Float64Array.from(parent.step(A0));
    ok(maxAbsDiff(p1, off).m === 0, 'parent mode (affine:false): a = 0 HOLDS the target exactly');
  }

  // --- the soft-limit clamp ------------------------------------------------
  {
    const it = new IncrementIntegrator(DEF, LO, HI);
    it.reset(DEF);
    const a = Float64Array.from({ length: 12 }, () => 40); // far outside every limit
    let violated = false;
    for (let k = 0; k < 400; k++) {
      const t = it.step(a);
      for (let i = 0; i < 12; i++) {
        if (t[i] > HI[i] + 1e-12 || t[i] < LO[i] - 1e-12) violated = true;
      }
    }
    ok(!violated, 'the target never leaves the SOFT joint limits (400 saturating steps)');
    let atHi = 0;
    for (let i = 0; i < 12; i++) if (Math.abs(it.target[i] - HI[i]) < 1e-6) atHi++;
    ok(atHi === 12, 'a saturating +action drives every joint to its soft upper limit', `${atHi}/12`);

    const it2 = new IncrementIntegrator(DEF, LO, HI);
    it2.reset(DEF);
    const an = Float64Array.from({ length: 12 }, () => -40);
    for (let k = 0; k < 400; k++) it2.step(an);
    let atLo = 0;
    for (let i = 0; i < 12; i++) if (Math.abs(it2.target[i] - LO[i]) < 1e-6) atLo++;
    ok(atLo === 12, 'a saturating -action drives every joint to its soft lower limit', `${atLo}/12`);
  }

  // --- the two halves compose back into step() (the shield's entry points) --
  {
    const a = new IncrementIntegrator(DEF, LO, HI);
    const b = new IncrementIntegrator(DEF, LO, HI);
    const seed = DEF.map((v, i) => v + 0.3 * Math.cos(i));
    a.reset(seed);
    b.reset(seed);
    let worst = 0;
    for (let k = 0; k < 40; k++) {
      const act = Float64Array.from({ length: 12 }, (_, i) => 3 * Math.sin(0.7 * k + i));
      const viaStep = Float64Array.from(a.step(act));
      const viaHalves = Float64Array.from(b.applyIncrement(b.taskIncrement(act)));
      worst = Math.max(worst, maxAbsDiff(viaStep, viaHalves).m);
    }
    ok(
      worst === 0,
      'step(a) == applyIncrement(taskIncrement(a)) exactly (the shield reuses these)',
      `max|d| ${worst}`,
    );
    let threw = false;
    try {
      new IncrementIntegrator(DEF, LO, HI).taskIncrement(A0);
    } catch {
      threw = true;
    }
    ok(threw, 'taskIncrement also refuses to run before reset()');
  }

  // --- persistence ---------------------------------------------------------
  {
    const it = new IncrementIntegrator(DEF, LO, HI);
    it.reset(DEF);
    const a = Float64Array.from({ length: 12 }, () => 0.2); // small, inside the band
    const t1 = Float64Array.from(it.step(a));
    const t2 = Float64Array.from(it.step(a));
    ok(maxAbsDiff(t1, t2).m < 1e-15, 'a constant action reaches a FIXED POINT (the target persists)');
    it.reset(DEF.map((v) => v - 0.9));
    ok(
      Math.abs(it.target[0] - (DEF[0] - 0.9)) < 1e-15,
      'reset() re-seeds mid-episode (the browser must call it on every respawn)',
    );
  }
}

// ===========================================================================
// 8. plumbing
// ===========================================================================

async function sectionPlumbing(loaded) {
  section('8. PLUMBING — manifest, rotation flags, scene.json');

  const man = await loadManifest(join(ROOT, 'assets/policies/manifest.json'));
  ok(man.manifest.complete === true, 'manifest declares complete: true');
  ok(man.list().length === 16, `manifest lists 16 policies`, `${man.list().length}`);
  ok(man.opponents('sym', 'B').length === 5, 'sym seat B offers 5 opponents');
  ok(man.opponents('asym', 'attacker').length === 5, 'asym attacker offers 5 opponents');
  ok(man.opponents('asym', 'defender').length === 5, 'asym defender offers 5 opponents');
  ok(man.playerWalk().obs_dim === 47, 'the player walk row is 47-D');

  for (const row of man.list()) {
    const p = loaded.get(row.name);
    if (!p) continue;
    if (row.obs_dim !== p.obsDim || row.act_dim !== p.actDim) {
      ok(false, `${row.name}: manifest dims disagree with the weights`);
    }
  }
  ok(true, 'every manifest row agrees with its .bin/.json dimensions');

  // Loading through the manifest (no filename spelled by the caller).
  const viaManifest = await man.load(man.opponents('sym', 'B')[0]);
  ok(viaManifest.obsDim === 60 && viaManifest.actDim === 12, 'loadManifest().load() returns a runnable 60->12 policy');

  // Rotation: every A-half member seated at B must be flagged, every B-half one not.
  const rot = man.list().filter((r) => r.rotation_baked).map((r) => r.name).sort();
  ok(
    JSON.stringify(rot) === JSON.stringify(['sym_et_B', 'sym_nom_B', 'sym_s2c_B']),
    'exactly the A-half members carry the baked pi-rotation',
    rot.join(','),
  );

  // The action path each policy must be driven through.
  const walkPath = man.list().filter((r) => r.action_path === 'walk_affine').map((r) => r.name);
  const incPath = man.list().filter((r) => r.action_path === 'increment_integrator');
  ok(
    walkPath.length === 1 && walkPath[0] === 'player_walk_fastwalk_v3_9000',
    'only the player walk uses the plain walk affine',
  );
  ok(incPath.length === 15, 'all 15 game policies use the increment integrator', `${incPath.length}`);

  // scene.json must carry the soft limits action.js falls back to.
  for (const game of ['sym', 'asym']) {
    const p = join(ROOT, `assets/scene/${game}/scene.json`);
    if (!(await exists(p))) continue;
    const s = JSON.parse(await readFile(p, 'utf8'));
    const dl = maxAbsDiff(s.joints.softLimitLo, SOFT_JOINT_POS_LIMIT_LO);
    const dh = maxAbsDiff(s.joints.softLimitHi, SOFT_JOINT_POS_LIMIT_HI);
    const dd = maxAbsDiff(s.joints.defaultJointPos, [
      -0.1, 0.9, -1.8, 0.1, 0.9, -1.8, -0.1, 0.9, -1.8, 0.1, 0.9, -1.8,
    ]);
    ok(
      dl.m < 1e-6 && dh.m < 1e-6 && dd.m === 0,
      `scene/${game}/scene.json soft limits + default pose == action.js`,
      `max|d| ${Math.max(dl.m, dh.m).toExponential(2)}`,
    );
  }

  // buildPolicy must reject a mismatched blob rather than run garbage.
  {
    const meta = JSON.parse(await readFile(join(ROOT, 'assets/policies/sym_s2c_B.json'), 'utf8'));
    let threw = false;
    try {
      buildPolicy(meta, new Uint8Array(16));
    } catch {
      threw = true;
    }
    ok(threw, 'buildPolicy rejects a .bin whose size does not match layers[]');
  }
}

// ===========================================================================
// 9. closed loop — the three modules driving real physics
// ===========================================================================

async function sectionClosedLoop() {
  section('9. CLOSED LOOP — policy + obs + action against real MuJoCo');
  const scenePath = join(ROOT, 'assets/scene/asym/scene.xml');
  if (!(await exists(scenePath))) {
    console.log('  \x1b[33mskip\x1b[0m  assets/scene/asym/scene.xml is missing');
    return;
  }
  const { createSim } = await import('../app/physics.js');
  const man = await loadManifest(join(ROOT, 'assets/policies/manifest.json'));
  const sim = await createSim({ sceneUrl: scenePath });
  const scene = sim.sceneJson;
  const LO = scene.joints.softLimitLo;
  const HI = scene.joints.softLimitHi;
  const DEF = scene.joints.defaultJointPos;
  const SPAWN = scene.spawn.config; // == app/config.js GAMES.asym.spawn
  const LINE_X = scene.field.lineX; // 1.9, env-local

  /**
   * One 50 Hz control step, in the training order (recon/05:110-123):
   *   process_action -> 4x (write ctrl, mj_step) -> [referee] -> mj_forward -> obs
   */
  function rollout({ steps, seatA, seatB, cmdB }) {
    sim.resetAll(SPAWN);
    const drive = {};
    for (const [r, cfg] of Object.entries({ a: seatA, b: seatB })) {
      const d = { ...cfg, lastAction: new Float32Array(12) };
      if (cfg.kind === 'game') {
        d.act = new IncrementIntegrator(DEF, LO, HI);
        d.act.reset(sim.getJointPos(r));
        d.obs = new Float32Array(GAME_OBS_DIM);
      } else {
        d.act = new WalkAffine(DEF);
        d.obs = new Float32Array(WALK_OBS_DIM);
      }
      drive[r] = d;
    }
    const x0 = { a: sim.getBase('a').pos[0], b: sim.getBase('b').pos[0] };
    let nan = 0;
    let maxTilt = 0;
    let contact = 0;
    let n = 0;
    for (; n < steps; n++) {
      for (const [r, d] of Object.entries(drive)) {
        const o =
          d.kind === 'game'
            ? gameObs(sim, r, r === 'a' ? 'b' : 'a', d.dir, LINE_X, n, d.lastAction, { out: d.obs })
            : walkObs(sim, r, cmdB, n, d.lastAction, { out: d.obs });
        for (let i = 0; i < o.length; i++) if (!Number.isFinite(o[i])) nan++;
        const a = d.policy.forward(o);
        d.lastAction.set(a);
        sim.setCtrl(r, d.act.step(a));
      }
      sim.step(4);
      if (sim.anyContactBetweenRobots()) contact++;
      sim.forward();
      for (const r of ['a', 'b']) {
        maxTilt = Math.max(maxTilt, sim.tiltAngle(r));
        const p = sim.getBase(r).pos;
        if (!Number.isFinite(p[0]) || !Number.isFinite(p[2])) nan++;
      }
    }
    return {
      nan,
      contact,
      steps: n,
      maxTiltDeg: (maxTilt * 180) / Math.PI,
      dx: { a: sim.getBase('a').pos[0] - x0.a, b: sim.getBase('b').pos[0] - x0.b },
      z: { a: sim.getBase('a').pos[2], b: sim.getBase('b').pos[2] },
      fell: { a: sim.fallen('a'), b: sim.fallen('b') },
    };
  }

  // --- A: both seats on real game policies --------------------------------
  {
    const atk = await man.load('asym_s2c_attacker');
    const def = await man.load('asym_s2c_defender');
    const t0 = process.hrtime.bigint();
    const N = 250; // 5 s of game time
    const r = rollout({
      steps: N,
      seatA: { kind: 'game', policy: atk, dir: seatDirection('asym', 'attacker') },
      seatB: { kind: 'game', policy: def, dir: seatDirection('asym', 'defender') },
    });
    const ms = Number(process.hrtime.bigint() - t0) / 1e6;
    ok(r.nan === 0, 'S2C atk vs S2C def, 250 control steps: no NaN anywhere');
    ok(
      Math.abs(r.dx.a) > 0.25,
      'the attacker actually locomotes (it is not marching in place)',
      `dx = ${r.dx.a.toFixed(3)} m over ${(N * 0.02).toFixed(1)} s, base z ${r.z.a.toFixed(3)}`,
    );
    ok(
      r.dx.a > 0,
      'the attacker advances toward its line (+x)',
      `x moved ${r.dx.a.toFixed(3)} m, defender ${r.dx.b.toFixed(3)} m`,
    );
    console.log(
      `       wall clock ${ms.toFixed(0)} ms for ${N} steps = ${(ms / N).toFixed(2)} ms/control step ` +
        `(${((ms / N / 20) * 100).toFixed(1)} % of real time); max tilt ${r.maxTiltDeg.toFixed(1)} deg, ` +
        `${r.contact} contact steps`,
    );
    ok(ms / N < 20, 'a full control step (2 policies + obs + action + 4 physics) runs faster than real time');
  }

  // --- B: the HUMAN path — the walk policy on a constant forward command ---
  {
    const atk = await man.load('asym_s2c_attacker');
    const walk = await man.load(man.playerWalk().name);
    const r = rollout({
      steps: 250,
      seatA: { kind: 'game', policy: atk, dir: seatDirection('asym', 'attacker') },
      seatB: { kind: 'walk', policy: walk },
      // A plain "hold W" from the player: +2.0 m/s forward in the BODY frame.
      // Seat b spawns yawed pi, so forward is world -x.
      cmdB: [2.0, 0, 0],
    });
    ok(r.nan === 0, 'walk policy driving seat b, 250 steps: no NaN');
    const speed = -r.dx.b / (250 * 0.02); // forward for a pi-yawed seat is world -x
    ok(
      speed > 0.8 && !r.fell.b,
      'the walk policy WALKS on a held W (a wrong 47-D layout collapses here)',
      `commanded +2.00 m/s, achieved ${speed.toFixed(2)} m/s over 5.0 s, upright ${!r.fell.b}, base z ${r.z.b.toFixed(3)}`,
    );
  }

  sim.dispose();
}

// ===========================================================================
// 10. the browser fetch path
// ===========================================================================

async function sectionHttp() {
  section('10. HTTP — the same loaders over the browser fetch path');
  const http = await import('node:http');
  const { readFile: rf } = await import('node:fs/promises');
  const TYPES = { '.json': 'application/json', '.bin': 'application/octet-stream' };
  const server = http.createServer(async (req, res) => {
    try {
      const rel = decodeURIComponent(req.url.split('?')[0]).replace(/^\/+/, '');
      const body = await rf(join(ROOT, rel));
      const ext = rel.slice(rel.lastIndexOf('.'));
      res.writeHead(200, { 'content-type': TYPES[ext] ?? 'application/octet-stream' });
      res.end(body);
    } catch {
      res.writeHead(404);
      res.end('not found');
    }
  });
  await new Promise((r) => server.listen(0, '127.0.0.1', r));
  const base = `http://127.0.0.1:${server.address().port}/`;
  try {
    const parity = JSON.parse(await readFile(join(ROOT, 'tests/parity.json'), 'utf8'));
    const man = await loadManifest(`${base}assets/policies/manifest.json`);
    ok(man.list().length === 16, 'manifest loads over http');
    for (const name of ['player_walk_fastwalk_v3_9000', 'sym_et_B', 'asym_lag_defender']) {
      const pol = await man.load(name);
      const e = parity.policies[name];
      let m = 0;
      for (let r = 0; r < e.obs.length; r++) {
        const d = maxAbsDiff(pol.forward(Float32Array.from(e.obs[r])), e.act[r]);
        if (d.m > m) m = d.m;
      }
      ok(m < PARITY_TOL, `${name} over http reproduces parity.json`, `max|d| ${m.toExponential(3)}`);
    }
  } finally {
    await new Promise((r) => server.close(r));
  }
}

// ===========================================================================

async function main() {
  console.log('S2C web play — policy runtime / observations / actions');
  console.log(`node ${process.version}`);

  const loaded = await sectionPolicies();
  sectionFrames();
  if (!QUICK) await sectionLiveSim();
  await sectionGroundTruth();
  sectionIntegrator();
  await sectionPlumbing(loaded);
  if (!QUICK) await sectionClosedLoop();
  await sectionHttp();

  console.log(`\n${fail === 0 ? '\x1b[32m' : '\x1b[31m'}${pass} passed, ${fail} failed\x1b[0m`);
  if (fail) {
    console.log('failed checks:');
    for (const f of failures) console.log(`  - ${f}`);
  }
  process.exit(fail ? 1 : 0);
}

main().catch((e) => {
  console.error('\x1b[31mtest harness crashed\x1b[0m');
  console.error(e);
  process.exit(2);
});

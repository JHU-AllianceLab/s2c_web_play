// tests/node_physics_smoke.mjs — node smoke test for app/physics.js.
//
//   node tests/node_physics_smoke.mjs                 # every scene that exists
//   node tests/node_physics_smoke.mjs --scene <path>  # just this one
//   node tests/node_physics_smoke.mjs --fallback      # force the generated scene
//   node tests/node_physics_smoke.mjs --http          # also load one scene over
//                                                     # http:// — the BROWSER path
//
// Runs the whole battery once per scene (assets/scene/sym, assets/scene/asym):
//   1. structure — 24 actuators, joint/actuator maps, timestep, names
//   2. reset to the spawn, hold ctrl = the default stance for 2 s;
//      upright, no NaN, base heights printed
//   3. no NaN/Inf in qpos/qvel/ctrl/xpos/xquat/renderState
//   4. contactsBetweenRobots() — silent apart, fires overlapped, A/B ordered,
//      normal is unit and points from A to B
//   5. renderer feed — renderState reuses its buffer, describeGeoms/getMesh
//   6. FRAME CONVENTIONS — the spin test: write a known base velocity, read it
//      back through getBase, and prove angVelB is BODY frame / linVelW is WORLD
//   7. the training fall rule — 70 deg, tripped by tilting the robot past it
//   8. embind handle / heap stability across 40 000 contact scans
//   9. 1000 mj_step, timed -> us/step
//
// If no exported scene exists at all, a FALLBACK scene is generated into
// .cache/smoke/ (gitignored). The fallback is NOT authoritative — it is built
// here from the real go2.xml plus the measured mjlab plant settings so the
// timing number stays comparable. An exported scene always wins.

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  createSim, GO2_DEFAULT_JOINT_POS, GO2_SPAWN_Z, FALL_LIMIT_ANGLE,
} from '../app/physics.js';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO = path.resolve(HERE, '..');
const CACHE = path.join(REPO, '.cache/smoke');

// Go2 source tree — READ ONLY, used solely to build the fallback scene.
const GO2_XMLS = '/home/ray/Disk_ext/Go2/Project/unitree_rl_mjlab/src/assets/robots/unitree_go2/xmls';

const PHYSICS_DT = 0.005;   // recon/05:94   (game_env_cfg.py:50)
const DECIMATION = 4;       // recon/05:95   (game_env_cfg.py:51)

// The two games' spawns, from the frozen app/config.js block in the task
// contract. sym = the locked demo opening (recon/02:317); asym = the training
// nominal (touchdown.py:235-248, recon/05:234-236). Seat a = attacker,
// seat b = defender, matching tools/export_scene.py:84 RENAME.
const SCENES = [
  {
    key: 'sym',
    xml: path.join(REPO, 'assets/scene/sym/scene.xml'),
    spawn: {
      a: { x: -2.1431, y: -0.1409, yaw: 0.0979 },
      b: { x: 2.1775, y: -0.1215, yaw: 3.1381 },
    },
  },
  {
    key: 'asym',
    // S2C_ASYM_SCENE lets you point at an export before it is copied into
    // assets/ (the asym scene is produced by a different agent).
    xml: process.env.S2C_ASYM_SCENE
      ? path.resolve(process.env.S2C_ASYM_SCENE)
      : path.join(REPO, 'assets/scene/asym/scene.xml'),
    spawn: {
      attacker: { x: -1.4, y: 0, yaw: 0 },
      defender: { x: 0.75, y: 0, yaw: Math.PI },
    },
  },
];

// ---------------------------------------------------------------------------
// tiny assert harness
// ---------------------------------------------------------------------------
let failures = 0;
let checks = 0;
function check(ok, label, detail = '') {
  checks++;
  console.log(`${ok ? '  PASS' : '  FAIL'}  ${label}${detail ? `   ${detail}` : ''}`);
  if (!ok) failures++;
  return ok;
}
function finiteAll(arr, label) {
  for (let i = 0; i < arr.length; i++) {
    if (!Number.isFinite(arr[i])) return `${label}[${i}] = ${arr[i]}`;
  }
  return null;
}
const f3 = (v) => Array.from(v).map((x) => (+x).toFixed(3)).join(', ');

// ---------------------------------------------------------------------------
// fallback scene
// ---------------------------------------------------------------------------

/** kp / kd / effort / armature per joint group. go2_constants.py:41-67 (recon/05:133-139). */
const ACT = {
  hip: { kp: 20, kv: 1, force: 23.5, armature: 0.01 },
  thigh: { kp: 20, kv: 1, force: 23.5, armature: 0.01 },
  calf: { kp: 40, kv: 2, force: 45.0, armature: 0.02 },
};
const LEGS = ['FL', 'FR', 'RL', 'RR'];
const PARTS = ['hip', 'thigh', 'calf'];
const BASE_LINK_Z = 0.445; // go2.xml:41  <body name="base_link" pos="0 0 0.445">

function yawQuat(yaw) {
  const h = 0.5 * yaw;
  return `${Math.cos(h)} 0 0 ${Math.sin(h)}`;
}

function buildFallbackScene(spawn) {
  fs.mkdirSync(path.join(CACHE, 'assets'), { recursive: true });
  fs.copyFileSync(path.join(GO2_XMLS, 'go2.xml'), path.join(CACHE, 'go2.xml'));
  for (const f of fs.readdirSync(path.join(GO2_XMLS, 'assets'))) {
    fs.copyFileSync(path.join(GO2_XMLS, 'assets', f), path.join(CACHE, 'assets', f));
  }
  const actuators = [];
  for (const s of ['a', 'b']) {
    for (const leg of LEGS) {
      for (const part of PARTS) {
        const a = ACT[part];
        actuators.push(
          `    <position name="${s}/${leg}_${part}" joint="${s}/${leg}_${part}_joint" ` +
          `kp="${a.kp}" kv="${a.kv}" forcerange="${-a.force} ${a.force}" ctrllimited="false"/>`);
      }
    }
  }
  // <frame> offsets stack onto go2.xml's base_link pos="0 0 0.445", so shift by
  // (spawn z - 0.445) to make qpos0 stand at z = 0.32. resetAll() overwrites the
  // free-joint qpos anyway; this only makes the un-reset scene sane.
  const fz = GO2_SPAWN_Z - BASE_LINK_Z;
  const xml = `<mujoco model="s2c_smoke_fallback">
  <!-- FALLBACK TEST SCENE - generated by tests/node_physics_smoke.mjs.
       NOT the authoritative export; assets/scene/*/scene.xml supersedes it.
       opt block: recon/05_env_physics_contract.md:92-108 (game_env_cfg.py:50-51, :555-566) -->
  <compiler angle="radian"/>
  <option timestep="${PHYSICS_DT}" integrator="implicitfast" solver="Newton" iterations="10"
          ls_iterations="20" tolerance="1e-8" ls_tolerance="0.01" ccd_iterations="100"
          cone="pyramidal" impratio="1" gravity="0 0 -9.81"/>
  <asset>
    <model name="go2" file="go2.xml"/>
  </asset>
  <worldbody>
    <!-- TerrainEntityCfg(terrain_type="plane"), recon/05:71 -->
    <geom name="terrain" type="plane" size="0 0 0.01" contype="1" conaffinity="1" condim="3"
          friction="1 0.005 1e-4" solref="0.02 1" solimp="0.9 0.95 0.001 0.5 2"/>
    <frame pos="${spawn.a.x} ${spawn.a.y} ${fz}" quat="${yawQuat(spawn.a.yaw)}">
      <attach model="go2" prefix="a/"/>
    </frame>
    <frame pos="${spawn.b.x} ${spawn.b.y} ${fz}" quat="${yawQuat(spawn.b.yaw)}">
      <attach model="go2" prefix="b/"/>
    </frame>
  </worldbody>
  <actuator>
${actuators.join('\n')}
  </actuator>
</mujoco>
`;
  const xmlPath = path.join(CACHE, 'scene.xml');
  fs.writeFileSync(xmlPath, xml);
  return xmlPath;
}

/**
 * Apply the plant settings mjlab applies programmatically and that a plain
 * <attach> of go2.xml cannot express. Fallback scene only — a real export bakes
 * these into the XML.
 *   FULL_COLLISION   go2_constants.py:105-113  (recon/05:76-79)
 *   player_collision robots.py:220-244         (recon/05:75)
 *   armature         go2_constants.py:41-67    (recon/05:133-139)
 */
function patchFallbackPlant(sim) {
  const { model } = sim;
  const geoms = sim.describeGeoms();
  const condim = model.geom_condim, prio = model.geom_priority;
  const fric = model.geom_friction, solimp = model.geom_solimp;
  const contype = model.geom_contype, conaff = model.geom_conaffinity;
  let nColl = 0, nFoot = 0;
  for (const g of geoms) {
    if (!g.name || !g.name.endsWith('_collision')) continue;
    nColl++;
    const isFoot = /_foot_collision$/.test(g.name);
    const i = g.geomId;
    condim[i] = isFoot ? 3 : 1;
    prio[i] = isFoot ? 1 : 0;
    fric[3 * i] = isFoot ? 0.6 : 1.0;
    fric[3 * i + 1] = 0.005;
    fric[3 * i + 2] = 1e-4;
    solimp[5 * i] = 0.9; solimp[5 * i + 1] = 0.95;
    solimp[5 * i + 2] = isFoot ? 0.023 : 0.001;
    solimp[5 * i + 3] = 0.5; solimp[5 * i + 4] = 2;
    if (isFoot) nFoot++;
  }
  for (const g of geoms) {
    if (g.robot === 'a') { contype[g.geomId] = 1; conaff[g.geomId] = 2; }
    else if (g.robot === 'b') { contype[g.geomId] = 2; conaff[g.geomId] = 1; }
  }
  const arm = model.dof_armature;
  for (const s of ['a', 'b']) {
    const dofs = sim.seats[s].jointIds.map((j) => model.jnt_dofadr[j]);
    dofs.forEach((d, k) => { arm[d] = ACT[PARTS[k % 3]].armature; });
  }
  return { nColl, nFoot };
}

// ---------------------------------------------------------------------------
// the battery, once per scene
// ---------------------------------------------------------------------------

async function runScene({ key, xml, spawn, fallback = false }) {
  console.log(`\n${'#'.repeat(78)}`);
  console.log(`# scene "${key}"  ${xml}${fallback ? '   [FALLBACK — not the authoritative export]' : ''}`);
  console.log('#'.repeat(78));

  const t0 = performance.now();
  const sim = await createSim({ sceneUrl: xml });
  const tLoad = performance.now() - t0;

  console.log(`\n[load ] compiled in ${tLoad.toFixed(0)} ms; ${sim.assetNames.length} asset(s), ` +
    `${(sim.assetBytes / 1048576).toFixed(2)} MiB into the VFS`);
  console.log(`[model] nq ${sim.nq}  nv ${sim.nv}  nu ${sim.nu}  nbody ${sim.nbody}  ` +
    `ngeom ${sim.ngeom}  njnt ${sim.njnt}  nmesh ${sim.nmesh}  nkey ${sim.nkey}  ` +
    `timestep ${sim.timestep}`);
  console.log(`[seats] a: ${sim.seats.a.baseBody} (prefix "${sim.seats.a.prefix}")   ` +
    `b: ${sim.seats.b.baseBody} (prefix "${sim.seats.b.prefix}")`);
  console.log(`[seats] a joints    ${sim.seats.a.jointIds.join(',')}`);
  console.log(`[seats] a actuators ${sim.seats.a.actIds.join(',')}`);
  console.log(`[fall ] limit ${sim.fallLimitAngle.toFixed(10)} rad = ` +
    `${(sim.fallLimitAngle * 180 / Math.PI).toFixed(1)} deg   ` +
    `baseZMin ${sim.fallBaseZMin}`);
  for (const w of sim.warnings) console.log(`[warn ] ${w}`);

  if (fallback) {
    const p = patchFallbackPlant(sim);
    console.log(`[patch] ${p.nColl} collision geoms (${p.nFoot} feet), per-seat ` +
      'contype/conaffinity 1/2 and 2/1, dof armature 0.01/0.01/0.02');
  }

  // --- 1. structure --------------------------------------------------------
  console.log('\n--- 1. structure -------------------------------------------------');
  check(sim.nu >= 24, 'at least 24 actuators', `nu=${sim.nu}`);
  check(sim.seats.a.actIds.every((u) => u >= 0) && sim.seats.b.actIds.every((u) => u >= 0),
    'all 24 joints have an actuator');
  check(new Set([...sim.seats.a.jointIds, ...sim.seats.b.jointIds]).size === 24,
    '24 distinct joint ids across the two seats');
  check(Math.abs(sim.timestep - PHYSICS_DT) < 1e-12,
    `physics timestep == ${PHYSICS_DT}`, `got ${sim.timestep}`);
  {
    // Names come from model.names, never mj_id2name (which returns "emsc" for
    // unnamed objects in this build).
    const jn = sim.seats.a.jointNames;
    const expect = ['FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint', 'FR_hip_joint'];
    const ok = jn.length === 12 && expect.every((suf, i) => (jn[i] ?? '').endsWith(suf));
    check(ok, 'seat a joints are in JOINT order FL,FR,RL,RR x (hip,thigh,calf)',
      `${jn.slice(0, 4).join(' ')} ...`);
    const bad = sim.describeGeoms().filter((g) => g.name === 'emsc');
    check(bad.length === 0, 'no geom is named "emsc" (the mj_id2name junk string)',
      `${bad.length} bad`);
  }
  {
    const kp = sim.getGains('a').kp, kd = sim.getGains('a').kd;
    const want = [20, 20, 40, 20, 20, 40, 20, 20, 40, 20, 20, 40];
    const wantD = [1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2];
    check(want.every((v, i) => Math.abs(kp[i] - v) < 1e-9)
      && wantD.every((v, i) => Math.abs(kd[i] - v) < 1e-9),
      'walk-soft PD gains kp 20/20/40 kd 1/1/2 in JOINT order (recon/05:127-139)',
      `kp [${f3(kp.slice(0, 3))}]`);
  }

  // --- 2. reset + 2 s holding the default stance ---------------------------
  console.log('\n--- 2. reset + 2 s holding the default stance --------------------');
  sim.resetAll(spawn);
  const seatKeys = Object.keys(spawn);
  const wantA = spawn[seatKeys[0]], wantB = spawn[seatKeys[1]];
  const q0a = sim.getJointPos('a');
  const b0a = sim.getBase('a');
  const b0b = sim.getBase('b');
  console.log(`  after reset: a base (${f3(b0a.pos)})  b base (${f3(b0b.pos)})`);
  check(Math.abs(b0a.pos[0] - wantA.x) < 1e-9 && Math.abs(b0a.pos[1] - wantA.y) < 1e-9,
    'resetAll placed seat a at the spawn xy');
  check(Math.abs(b0b.pos[0] - wantB.x) < 1e-9 && Math.abs(b0b.pos[1] - wantB.y) < 1e-9,
    'resetAll placed seat b at the spawn xy');
  check(Math.abs(b0a.pos[2] - sim.spawnZ) < 1e-12 && Math.abs(b0b.pos[2] - sim.spawnZ) < 1e-12,
    `resetAll used the scene spawn height z = ${sim.spawnZ}`);
  check(Math.abs(sim.yaw('b') - Math.atan2(Math.sin(wantB.yaw), Math.cos(wantB.yaw))) < 1e-9,
    'resetAll yaw -> quaternion round-trips', `yaw(b) = ${sim.yaw('b').toFixed(6)}`);
  check(GO2_DEFAULT_JOINT_POS.every((v, i) => Math.abs(q0a[i] - sim.defaultJointPos[i]) < 1e-12),
    'reset joint angles == the scene default pose', `[${f3(q0a.slice(0, 3))} ...]`);
  check(GO2_DEFAULT_JOINT_POS.every((v, i) => Math.abs(sim.defaultJointPos[i] - v) < 1e-12),
    'the scene default pose == go2_constants.py INIT_STATE');

  const nCtrl = Math.round(2.0 / (PHYSICS_DT * DECIMATION)); // 100 control steps = 2.0 s
  const hold = sim.defaultJointPos;
  const heights = [];
  for (let k = 0; k < nCtrl; k++) {
    sim.setCtrl('a', hold);
    sim.setCtrl('b', hold);
    sim.step(DECIMATION);
    sim.forward();               // the control step's single forward
    if (k % 10 === 9 || k === 0) {
      heights.push([(k + 1) * PHYSICS_DT * DECIMATION,
        sim.getBase('a').pos[2], sim.getBase('b').pos[2]]);
    }
  }
  console.log('      t(s)    a.z      b.z');
  for (const [t, za, zb] of heights) {
    console.log(`    ${t.toFixed(2).padStart(6)}  ${za.toFixed(4)}  ${zb.toFixed(4)}`);
  }
  const za = sim.getBase('a').pos[2], zb = sim.getBase('b').pos[2];
  check(za > 0.20 && za < 0.45 && zb > 0.20 && zb < 0.45,
    'both bases settle in [0.20, 0.45] m', `a=${za.toFixed(4)} b=${zb.toFixed(4)}`);
  check(!sim.fallen('a') && !sim.fallen('b'),
    'neither robot is "fallen" after 2 s of standing',
    `tilt a=${(sim.tiltAngle('a') * 180 / Math.PI).toFixed(1)} deg ` +
    `b=${(sim.tiltAngle('b') * 180 / Math.PI).toFixed(1)} deg`);

  // --- 3. no NaN / Inf -----------------------------------------------------
  console.log('\n--- 3. no NaN / Inf ----------------------------------------------');
  {
    const bad = finiteAll(sim.data.qpos, 'qpos') || finiteAll(sim.data.qvel, 'qvel')
      || finiteAll(sim.data.ctrl, 'ctrl') || finiteAll(sim.data.xpos, 'xpos')
      || finiteAll(sim.data.xquat, 'xquat') || finiteAll(sim.renderState(), 'renderState');
    check(bad === null, 'qpos/qvel/ctrl/xpos/xquat/renderState all finite', bad ?? '');
  }
  {
    const g = sim.projectedGravity('a');
    const n = Math.hypot(g[0], g[1], g[2]);
    check(Math.abs(n - 1) < 1e-12, 'projected gravity is a UNIT vector (not 9.81-scaled)',
      `|g| = ${n}`);
  }
  {
    const base = sim.getBase('a');
    const qn = Math.hypot(...base.quat);
    check(Math.abs(qn - 1) < 1e-9, 'base quaternion is normalized', `|q| = ${qn}`);
    check(base.linVelW.every(Number.isFinite) && base.angVelB.every(Number.isFinite),
      'getBase velocities finite', `|v| = ${Math.hypot(...base.linVelW).toFixed(4)} m/s`);
  }

  // --- 4. a<->b contacts ---------------------------------------------------
  console.log('\n--- 4. contactsBetweenRobots() -----------------------------------');
  sim.resetAll(spawn);
  sim.step(DECIMATION);
  check(sim.anyContactBetweenRobots() === false && sim.contactsBetweenRobots().length === 0,
    'silent at the spawn (robots metres apart)', `ncon=${sim.ncon}`);
  sim.setPose('a', { x: 0, y: 0, yaw: 0, jointPos: hold });
  sim.setPose('b', { x: 0.02, y: 0, yaw: Math.PI, jointPos: hold });
  sim.forward();
  sim.step(1);
  {
    const cross = sim.contactsBetweenRobots();
    check(sim.anyContactBetweenRobots() === true && cross.length > 0,
      'fires when the two robots are overlapped',
      `ncon=${sim.ncon}, a<->b pairs=${cross.length}`);
    const geoms = sim.describeGeoms();
    const aOk = cross.every((c) => geoms[c.geomA].robot === 'a');
    const bOk = cross.every((c) => geoms[c.geomB].robot === 'b');
    check(aOk && bOk, 'geomA always belongs to seat a, geomB to seat b');
    const unit = cross.every((c) => Math.abs(Math.hypot(...c.normal) - 1) < 1e-9);
    check(unit, 'every contact normal is a unit vector');
  }
  // Direction, on a SHALLOW head-on touch. The trunk box half-extent along x is
  // 0.1881 (go2.xml base1_collision), so the boxes meet at dx = 0.3762; 0.370
  // leaves 6.2 mm of penetration on the x axis, which is then unambiguously the
  // minimum-penetration axis. (At dx = 0.02 the boxes are near-coincident and
  // MuJoCo picks the narrow y axis instead — a degenerate case, not a bug.)
  for (const sgn of [+1, -1]) {
    sim.setPose('a', { x: sgn < 0 ? 0.370 : 0, y: 0, z: 0.32, yaw: 0, jointPos: hold });
    sim.setPose('b', { x: sgn < 0 ? 0 : 0.370, y: 0, z: 0.32, yaw: Math.PI, jointPos: hold });
    sim.forward();
    sim.step(1);
    const head = sim.contactsBetweenRobots().find(
      (c) => /base1_collision$/.test(c.nameA ?? '') && /base1_collision$/.test(c.nameB ?? ''));
    check(head != null && Math.abs(head.normal[0] - sgn) < 1e-6,
      `normal points from A towards B (b at ${sgn > 0 ? '+' : '-'}x of a)`,
      head ? `n = [${f3(head.normal)}], dist = ${head.dist.toExponential(2)}` : 'no trunk pair');
  }
  sim.resetAll(spawn);
  sim.step(DECIMATION);
  check(sim.anyContactBetweenRobots() === false, 'clears after resetAll');

  // --- 5. renderer feed ----------------------------------------------------
  console.log('\n--- 5. renderer feed ---------------------------------------------');
  {
    const rs = sim.renderState();
    check(rs.length === sim.nbody * 7, 'renderState length == nbody*7', `${rs.length}`);
    check(rs === sim.renderState() && rs instanceof Float32Array,
      'renderState reuses ONE Float32Array (no per-frame allocation)');
    const geoms = sim.describeGeoms();
    check(geoms.length === sim.ngeom, 'describeGeoms length == ngeom');
    const byType = {};
    for (const g of geoms) byType[g.typeName] = (byType[g.typeName] ?? 0) + 1;
    console.log(`    geom types: ${Object.entries(byType).map(([k, v]) => `${k}x${v}`).join(' ')}`);
    const perSeat = { a: 0, b: 0, none: 0 };
    for (const g of geoms) perSeat[g.robot ?? 'none']++;
    console.log(`    geoms per seat: a=${perSeat.a} b=${perSeat.b} world=${perSeat.none}`);
    check(perSeat.a === perSeat.b && perSeat.a > 0, 'the two seats own the same geom count');
    const coll = geoms.filter((g) => g.robot === 'a' && g.group === 3);
    check(coll.length === 23 || fallback,
      'seat a has the 23 collision primitives (robots.py:119-134)', `${coll.length}`);
    check(geoms.every((g) => g.rgba.length === 4 && g.rgba.every(Number.isFinite)),
      'every geom carries a finite rgba');

    const meshGeoms = geoms.filter((g) => g.meshId >= 0);
    if (meshGeoms.length) {
      let verts = 0, tris = 0;
      const ids = [...new Set(meshGeoms.map((g) => g.meshId))];
      for (const id of ids) {
        const m = sim.getMesh(id);
        verts += m.vertices.length / 3;
        tris += m.indices.length / 3;
        const maxIdx = m.indices.reduce((x, y) => (y > x ? y : x), 0);
        if (m.vertices.length % 3 || m.normals.length !== m.vertices.length
          || m.indices.length % 3 || maxIdx >= m.vertices.length / 3
          || finiteAll(m.vertices, 'v') !== null || finiteAll(m.normals, 'n') !== null) {
          check(false, `getMesh(${id}) "${m.name}" is well-formed`);
          break;
        }
      }
      check(true, `all ${ids.length} meshes give consistent vertex/normal/index arrays`,
        `${verts} verts, ${tris} tris total`);
      const named = meshGeoms.filter((g) => g.meshName).length;
      check(named === meshGeoms.length, 'every mesh geom resolves a mesh name',
        `${named}/${meshGeoms.length}`);
    } else {
      console.log('    (scene carries no mesh geoms — getMesh not exercised)');
    }
    const bodies = sim.describeBodies();
    check(bodies.length === sim.nbody && bodies[0].name === 'world',
      'describeBodies covers every body, world first');
    const mats = sim.describeMaterials();
    check(mats.every((m) => m.rgba.length === 4), `describeMaterials: ${mats.length} materials`);
  }

  // --- 6. FRAME CONVENTIONS — the spin test --------------------------------
  //
  // Write a KNOWN base velocity through the free joint and read it back through
  // getBase(). Done at a non-zero yaw so world and body axes differ: a
  // convention mix-up cannot hide.
  console.log('\n--- 6. frame conventions (spin test) -----------------------------');
  {
    const YAW = Math.PI / 2;                 // body +x == world +y
    const WB = [0.0, 0.0, 3.0];              // body-frame angular velocity
    const VW = [1.0, 0.0, 0.0];              // world-frame linear velocity
    sim.resetAll(spawn);
    sim.setPose('a', { x: 0, y: 0, z: 2.0, yaw: YAW, jointPos: hold });
    sim.setBaseVel('a', { linVelW: VW, angVelB: WB });
    sim.forward();
    let base = sim.getBase('a');
    check(WB.every((v, i) => Math.abs(base.angVelB[i] - v) < 1e-9),
      'angVelB reads back the BODY-frame angular velocity that was written',
      `wrote [${f3(WB)}] read [${f3(base.angVelB)}]`);
    check(VW.every((v, i) => Math.abs(base.linVelW[i] - v) < 1e-9),
      'linVelW reads back the WORLD-frame linear velocity that was written',
      `wrote [${f3(VW)}] read [${f3(base.linVelW)}]`);

    // Now a body-x spin at yaw=90: the WORLD angular velocity must land on +y.
    sim.setBaseVel('a', { linVelW: [0, 0, 0], angVelB: [2.0, 0, 0] });
    sim.forward();
    base = sim.getBase('a');
    check(Math.abs(base.angVelB[0] - 2) < 1e-9 && Math.abs(base.angVelW[1] - 2) < 1e-9
      && Math.abs(base.angVelW[0]) < 1e-9,
      'body +x spin at yaw=90 deg shows up on WORLD +y — the two frames differ',
      `angVelB [${f3(base.angVelB)}] angVelW [${f3(base.angVelW)}]`);

    // linVelW is base_link's velocity, NOT the subtree COM's: a pure spin about
    // the body origin must read exactly zero linear velocity even though the
    // COM (0.0211 m ahead of the origin) is moving.
    sim.setBaseVel('a', { linVelW: [0, 0, 0], angVelB: [0, 0, 5.0] });
    sim.forward();
    base = sim.getBase('a');
    const comSpeed = Math.hypot(sim.data.cvel[6 * sim.seats.a.baseBodyId + 3],
      sim.data.cvel[6 * sim.seats.a.baseBodyId + 4]);
    check(Math.hypot(...base.linVelW) < 1e-12 && comSpeed > 1e-3,
      'linVelW is base_link, not the subtree COM (cvel offset removed)',
      `|v_link| = ${Math.hypot(...base.linVelW).toExponential(1)}, ` +
      `|v_com| = ${comSpeed.toFixed(4)}`);

    // Free-fall for one substep: linVelW must pick up exactly g*dt on z.
    sim.setBaseVel('a', { linVelW: [0, 0, 0], angVelB: [0, 0, 0] });
    sim.forward();
    sim.step(1);
    sim.forward();
    const vz = sim.getBase('a').linVelW[2];
    const g = sim.model.opt.gravity[2];
    check(Math.abs(vz - g * sim.timestep) < 1e-9,
      'one free-fall substep gives linVelW.z == g*dt',
      `${vz.toExponential(6)} vs ${(g * sim.timestep).toExponential(6)}`);

    // A yaw-only pose leaves projected gravity at exactly (0, 0, -1).
    sim.setPose('a', { x: 0, y: 0, z: 2.0, yaw: 1.234, jointPos: hold });
    sim.forward();
    const pg = sim.projectedGravity('a');
    check(Math.abs(pg[0]) < 1e-12 && Math.abs(pg[1]) < 1e-12 && Math.abs(pg[2] + 1) < 1e-12,
      'yaw-only pose -> projected gravity (0, 0, -1)', `[${f3(pg)}]`);
  }

  // --- 7. the training fall rule -------------------------------------------
  console.log('\n--- 7. the training fall rule (70 deg) ---------------------------');
  {
    check(Math.abs(sim.fallLimitAngle - FALL_LIMIT_ANGLE) < 1e-15,
      'fall limit == math.radians(70) (touchdown.py:123)',
      `${sim.fallLimitAngle}`);
    // Roll the base by a known angle and check the rule trips exactly at 70 deg.
    const rollTo = (deg) => {
      const r = 0.5 * deg * Math.PI / 180;
      sim.setPose('a', { x: 0, y: 0, z: 2.0, quat: [Math.cos(r), Math.sin(r), 0, 0], jointPos: hold });
      sim.forward();
    };
    rollTo(69.0);
    const at69 = { tilt: sim.tiltAngle('a') * 180 / Math.PI, fell: sim.fallen('a') };
    rollTo(71.0);
    const at71 = { tilt: sim.tiltAngle('a') * 180 / Math.PI, fell: sim.fallen('a') };
    check(Math.abs(at69.tilt - 69) < 1e-6 && Math.abs(at71.tilt - 71) < 1e-6,
      'tiltAngle == acos(-projected_gravity_b[2]) reproduces the roll angle',
      `${at69.tilt.toFixed(4)} / ${at71.tilt.toFixed(4)} deg`);
    check(at69.fell === false && at71.fell === true,
      'fallen() is false at 69 deg and true at 71 deg');
  }

  // --- 8. embind handle / heap stability -----------------------------------
  console.log('\n--- 8. embind handle / heap stability ----------------------------');
  {
    sim.resetAll(spawn);
    sim.setPose('a', { x: 0, y: 0, yaw: 0, jointPos: hold });
    sim.setPose('b', { x: 0.02, y: 0, yaw: Math.PI, jointPos: hold });
    sim.forward();
    sim.step(1);
    const before = sim.heapBytes();
    for (let i = 0; i < 20000; i++) sim.anyContactBetweenRobots();
    for (let i = 0; i < 20000; i++) sim.contactsBetweenRobots();
    const after = sim.heapBytes();
    check(after === before, '40 000 contact scans did not grow the wasm heap',
      `${before} -> ${after} B`);
  }

  // --- 9. scene.json consumption -------------------------------------------
  //
  // tools/export_scene.py writes `s2c_web_play/scene@1`. Feed a scene@1 object
  // built from this very model back in and check (a) it resolves to exactly the
  // same maps, and (b) a wrong entry is reported on sim.warnings instead of
  // silently corrupting the seat map.
  console.log('\n--- 9. scene.json (schema s2c_web_play/scene@1) -------------------');
  {
    const mk = (mutate) => {
      const j = {
        schema: 's2c_web_play/scene@1',
        files: { xml: path.basename(xml), meshDir: 'assets' },
        robots: {},
        joints: { defaultJointPos: sim.defaultJointPos.slice() },
        rules: { fallLimitAngleRad: FALL_LIMIT_ANGLE },
      };
      for (const r of ['a', 'b']) {
        j.robots[r] = {
          prefix: sim.seats[r].prefix,
          baseBodyName: sim.seats[r].baseBody,
          jointNames: sim.seats[r].jointNames.slice(),
          jointIds: sim.seats[r].jointIds.slice(),
          actuatorNames: sim.seats[r].actNames.slice(),
          actuatorIdsJointOrder: sim.seats[r].actIds.slice(),
        };
      }
      if (mutate) mutate(j);
      return j;
    };
    const good = await createSim({ sceneUrl: xml, sceneJson: mk() });
    check(good.warnings.length === 0,
      'a faithful scene@1 resolves with zero warnings',
      good.warnings.join(' | ') || 'clean');
    check(good.seats.a.jointIds.join() === sim.seats.a.jointIds.join()
      && good.seats.b.actIds.join() === sim.seats.b.actIds.join(),
      'scene.json-driven maps == model-derived maps');
    check(Math.abs(good.fallLimitAngle - FALL_LIMIT_ANGLE) < 1e-15,
      'rules.fallLimitAngleRad reaches fallen()');
    good.dispose();

    // Swap two actuator entries: the transmission cross-check must catch it.
    const bad = await createSim({
      sceneUrl: xml,
      sceneJson: mk((j) => {
        const a = j.robots.a.actuatorNames;
        [a[0], a[1]] = [a[1], a[0]];
        const i = j.robots.a.actuatorIdsJointOrder;
        [i[0], i[1]] = [i[1], i[0]];
      }),
    });
    check(bad.warnings.some((w) => /actuator_trnid/.test(w)),
      'a scrambled actuator list is caught by the actuator_trnid cross-check',
      bad.warnings.find((w) => /actuator_trnid/.test(w)) ?? 'NOT CAUGHT');
    check(bad.seats.a.actIds.join() === sim.seats.a.actIds.join(),
      'and the transmission map wins, so setCtrl still hits the right actuators');
    bad.dispose();

    // The real export ships the geom -> robot map both as a top-level
    // `geomRobot` array (nulls for the unnamed visual meshes) and as
    // robots.<r>.geomIds. Give a collision geom to the wrong seat in BOTH and
    // check the disagreement is reported rather than swallowed.
    const real = fs.existsSync(path.join(path.dirname(xml), 'scene.json'))
      ? JSON.parse(fs.readFileSync(path.join(path.dirname(xml), 'scene.json'), 'utf8'))
      : null;
    if (real) {
      check(real.robots && real.robots.a && real.robots.b,
        `the exported scene.json is consumed (schema ${real.schema ?? '?'})`,
        `${Object.keys(real).length} top-level keys`);
      const stolen = real.robots.a.geomIds[0];
      const wrecked = JSON.parse(JSON.stringify(real));
      if (Array.isArray(wrecked.geomRobot)) wrecked.geomRobot[stolen] = 'b';
      wrecked.robots.b.geomIds = [...wrecked.robots.b.geomIds, stolen];
      const w = await createSim({ sceneUrl: xml, sceneJson: wrecked });
      check(w.warnings.some((s) => /disagrees|not owned/.test(s)),
        'a geom handed to the wrong seat is reported, not swallowed',
        w.warnings[0] ?? 'NOT CAUGHT');
      w.dispose();
    } else {
      console.log('    (no scene.json beside this scene — export cross-check skipped)');
    }
  }

  // --- 10. timing ----------------------------------------------------------
  console.log('\n--- 10. timing ---------------------------------------------------');
  let usPerStep = NaN;
  {
    sim.resetAll(spawn);
    sim.setCtrl('a', hold); sim.setCtrl('b', hold);
    for (let i = 0; i < 400; i++) sim.step(1);           // warm up / settle
    const nconSettled = sim.ncon;
    const N = 1000;
    const t = performance.now();
    sim.step(N);
    const dt = performance.now() - t;
    usPerStep = (dt * 1000) / N;
    console.log(`  ${N} mj_step in ${dt.toFixed(1)} ms  ->  ${usPerStep.toFixed(1)} us/step   ` +
      `(ncon ${nconSettled})`);
    console.log(`  one 50 Hz control step (x${DECIMATION}) = ` +
      `${(usPerStep * DECIMATION / 1000).toFixed(3)} ms; realtime cost = ` +
      `${(usPerStep * 200 / 1000).toFixed(1)} ms per wall-second ` +
      `(${(usPerStep * 200 / 10000).toFixed(2)} % of one core)`);
    check(Number.isFinite(usPerStep) && usPerStep > 0 && usPerStep < 1000,
      'step cost is sane', `${usPerStep.toFixed(1)} us`);
    check(finiteAll(sim.data.qpos, 'qpos') === null, 'still finite after 1400 more steps');
  }

  sim.dispose();
  return { key, usPerStep };
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

const argv = process.argv.slice(2);
const forceFallback = argv.includes('--fallback');
const sceneArgIdx = argv.indexOf('--scene');

let todo = [];
if (sceneArgIdx >= 0) {
  todo = [{ key: 'cli', xml: path.resolve(argv[sceneArgIdx + 1]), spawn: SCENES[0].spawn }];
} else if (!forceFallback) {
  todo = SCENES.filter((s) => fs.existsSync(s.xml));
}
if (!todo.length) {
  if (!fs.existsSync(path.join(GO2_XMLS, 'go2.xml'))) {
    console.error(`no exported scene under ${REPO}/assets/scene and no go2.xml at ` +
      `${GO2_XMLS} — cannot build a fallback`);
    process.exit(2);
  }
  todo = [{ key: 'fallback', xml: buildFallbackScene(SCENES[0].spawn),
    spawn: SCENES[0].spawn, fallback: true }];
}

console.log('='.repeat(78));
console.log('s2c_web_play — node physics smoke test');
console.log(`node   : ${process.version}`);
console.log(`scenes : ${todo.map((s) => s.key).join(', ')}`);
for (const s of SCENES) {
  if (!todo.some((t) => t.xml === s.xml)) console.log(`         (skipped "${s.key}": ${s.xml} absent)`);
}
console.log('='.repeat(78));

const results = [];
for (const s of todo) results.push(await runScene(s));

// ---------------------------------------------------------------------------
// the browser path: serve the repo and load a scene over http://
//
// createSim() branches on the URL scheme, so an http:// sceneUrl takes exactly
// the code path a browser takes — global fetch for the MJCF, for scene.json and
// for every mesh — instead of node:fs. The only piece this cannot cover is
// emscripten's own `new URL('mujoco.wasm', import.meta.url)` resolution inside
// the vendored glue (vendor/mujoco/mujoco.js:648-655), which is upstream code.
// ---------------------------------------------------------------------------
if (argv.includes('--http') && todo.length && !todo[0].fallback) {
  const { createServer } = await import('node:http');
  const MIME = {
    '.xml': 'application/xml', '.json': 'application/json', '.obj': 'text/plain',
    '.js': 'text/javascript', '.wasm': 'application/wasm', '.html': 'text/html',
  };
  let served = 0;
  const server = createServer((req, res) => {
    const rel = decodeURIComponent(req.url.split('?')[0]).replace(/^\/+/, '');
    const file = path.join(REPO, rel);
    if (!file.startsWith(REPO) || !fs.existsSync(file) || fs.statSync(file).isDirectory()) {
      res.writeHead(404).end('no');
      return;
    }
    served++;
    res.writeHead(200, { 'content-type': MIME[path.extname(file)] ?? 'application/octet-stream' });
    fs.createReadStream(file).pipe(res);
  });
  await new Promise((r) => server.listen(0, '127.0.0.1', r));
  const port = server.address().port;
  const rel = path.relative(REPO, todo[0].xml).split(path.sep).join('/');
  const httpUrl = `http://127.0.0.1:${port}/${rel}`;

  console.log(`\n${'#'.repeat(78)}`);
  console.log(`# browser path — same module, http:// URLs   ${httpUrl}`);
  console.log('#'.repeat(78));
  try {
    const t = performance.now();
    const hsim = await createSim({ sceneUrl: httpUrl });
    const ms = performance.now() - t;
    check(hsim.ngeom > 0 && hsim.nu >= 24,
      'the scene compiles when every asset arrives over fetch()',
      `${served} HTTP request(s), ${(hsim.assetBytes / 1048576).toFixed(2)} MiB, ${ms.toFixed(0)} ms`);
    hsim.resetAll(todo[0].spawn);
    for (let k = 0; k < 25; k++) {
      hsim.setCtrl('a', hsim.defaultJointPos);
      hsim.setCtrl('b', hsim.defaultJointPos);
      hsim.step(DECIMATION);
      hsim.forward();
    }
    const hz = hsim.getBase('a').pos[2];
    check(Number.isFinite(hz) && hz > 0.2 && hz < 0.45,
      'half a second of standing over the fetch path', `a.z = ${hz.toFixed(4)}`);
    hsim.dispose();
  } catch (e) {
    check(false, 'browser path (http:// fetch)', String(e && e.message ? e.message : e));
  } finally {
    server.close();
  }
}

console.log(`\n${'='.repeat(78)}`);
for (const r of results) {
  console.log(`  ${r.key.padEnd(10)} ${r.usPerStep.toFixed(1)} us/step  ` +
    `(${(r.usPerStep * DECIMATION / 1000).toFixed(3)} ms per 50 Hz control step)`);
}
console.log(failures === 0
  ? `ALL ${checks} CHECKS PASSED across ${results.length} scene(s)`
  : `${failures} of ${checks} CHECK(S) FAILED`);
console.log('='.repeat(78));
process.exit(failures === 0 ? 0 : 1);

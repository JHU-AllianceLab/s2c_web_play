// app/physics.js — MuJoCo WASM harness for the two-Go2 touchdown game.
//
// One dependency-free ES module, unmodified in the browser AND in node. The
// only environment-dependent thing is *reading bytes*, handled by `readBytes()`
// below (fetch in a browser / for http(s)|data|blob URLs, node:fs otherwise).
// The @mujoco/mujoco 3.14.0 glue is itself dual-environment — it branches on
// ENVIRONMENT_IS_NODE and resolves the wasm with
// `new URL('mujoco.wasm', import.meta.url)`, an HTTP fetch in a browser and an
// fs read in node (vendor/mujoco/mujoco.js:648-655) — so there is ONE import
// path, no bundler, no CDN, all paths relative.
//
// =============================================================================
// CONVENTIONS THIS FILE COMMITS TO (all verified, see vendor/mujoco/VERSION.md)
// =============================================================================
//
// 1. FRAMES — `getBase()` reproduces mjlab's entity data byte for byte:
//      pos      = data.xpos[base_link]        the BODY-FRAME ORIGIN, not the COM
//                 (mjlab/entity/data.py:242 root_link_pose_w)
//      quat     = data.xquat[base_link]       w-FIRST (w, x, y, z)
//      linVelW  = cvel[3:6] - cvel[0:3] x (subtree_com - xpos)     WORLD frame
//                 (compute_velocity_from_cvel, mjlab/entity/data.py:19-30,
//                  root_link_vel_w :247-257)  — the cvel correction moves the
//                 reference point from the subtree COM back to base_link.
//      angVelB  = quat_apply_inverse(quat, cvel[0:3])              BODY frame
//                 (root_link_ang_vel_w = cvel[0:3], data.py:427-428;
//                  root_link_ang_vel_b, data.py:564-566)
//    projectedGravity() = quat_apply_inverse(quat, (0,0,-1)) — a UNIT vector,
//    not 9.81-scaled (gravity_vec_w, mjlab/entity/entity.py:697;
//    projected_gravity_b, mjlab/entity/data.py:548-550).
//
//    FREE-JOINT qvel, measured on this exact wasm build (see setBaseVel):
//      qvel[dofadr+0..2] = linear velocity of base_link in the WORLD frame
//      qvel[dofadr+3..5] = angular velocity in the BODY frame
//    (Proof: yaw=pi/2 with qvel_ang=(2,0,0) reads back cvel_ang_w=(0,2,0) and
//     angVelB=(2,0,0); yaw=pi/2 with qvel_lin=(1,0,0) reads back linVelW=
//     (1,0,0), i.e. world +x, not body +x. tests/node_physics_smoke.mjs §6.)
//
// 2. STALENESS — deliberate, and it matches training. MuJoCo runs forward
//    kinematics BEFORE integrating, so right after `step()` the derived arrays
//    (xpos, xquat, cvel, contact) lag qpos/qvel by one physics substep. mjlab
//    relies on exactly this: terminations/referee read the stale frame, then a
//    single sim.forward() runs, then the observations read the fresh frame
//    (mjlab/envs/manager_based_rl_env.py:339-345 and :395-418). So:
//
//      setCtrl(a,...); setCtrl(b,...); step(4);   // 1 control step @ 50 Hz
//      judge(...)        <- referee reads HERE (stale by one substep) = training
//      forward();                                  // the single forward
//      gameObs(...)      <- observations read HERE (fresh)            = training
//
//    This module NEVER auto-forwards after step(); doing so would silently
//    change the referee's frame. It DOES auto-forward after setPose(), because
//    a teleport with no forward leaves xpos describing the old pose.
//
// 3. FALL — `fallen()` is the training termination, not an invented threshold:
//    `mdp.bad_orientation` with limit_angle = fall_limit_angle, i.e.
//    `acos(-projected_gravity_b[2]) > limit`
//    (mjlab/envs/mdp/terminations.py:24-32; wired at
//     src/tasks/game/scenarios/touchdown.py:419-437 and
//     src/tasks/sym_game/scenarios/sym_touchdown.py:243-262).
//    limit = math.radians(70) = 1.2217304763960306 rad
//    (touchdown.py:123 `fall_limit_angle: float = math.radians(70.0)`; the sym
//     scenario inherits the same field, sym_touchdown.py:246). There is NO base
//    height test in training — `root_height_below_minimum` exists in mjlab
//    (terminations.py:35-42) but no game scenario installs it, so baseZMin
//    defaults to -Infinity here.
//    The past-line variants (`bad_orientation_void_past_line`) are a REFEREE
//    policy layered on top; app/referee.js owns that choice and can read
//    `tiltAngle()` directly.
//
// 4. NAMES — `mj_id2name` in this build returns the junk string "emsc" for
//    UNNAMED objects instead of null (verified: geom 1 of the sym scene is an
//    unnamed visual mesh, mj_id2name says "emsc", model.names says nothing).
//    Every name in this file therefore comes from the `model.names` buffer via
//    `model.name_<obj>adr`. Other agents: do not call mj_id2name.
//
// -----------------------------------------------------------------------------
// The only physical numbers hardcoded here are fallbacks, used solely when the
// scene carries neither a keyframe nor a scene.json and the caller passes
// nothing: the Go2 nominal stance and spawn height (go2_constants.py:73-83),
// the 12 joint-name suffixes in JOINT order (mjlab/entity/entity.py:459-481,
// recon/05_env_physics_contract.md:146-150) and the 70-degree fall limit.
// Timestep, solver, gains, friction, collision bitmasks and field geometry all
// come from the scene XML — this module never sets them.
// -----------------------------------------------------------------------------

import loadMujoco from '../vendor/mujoco/mujoco.js';

/** Go2 nominal standing pose, JOINT order FL,FR,RL,RR x (hip,thigh,calf).
 *  go2_constants.py:73-83 (INIT_STATE.joint_pos). Fallback only. */
export const GO2_DEFAULT_JOINT_POS = Object.freeze([
  -0.1, 0.9, -1.8, 0.1, 0.9, -1.8, -0.1, 0.9, -1.8, 0.1, 0.9, -1.8,
]);

/** Go2 nominal spawn height. go2_constants.py:74 (INIT_STATE.pos). Fallback only. */
export const GO2_SPAWN_Z = 0.32;

/** Joint-name suffixes in the JOINT order the policies and actions use.
 *  mjlab/entity/entity.py:459-481 / recon/05:146-150.
 *  NB the compiled mjModel *actuator* array is grouped hip x4 / thigh x4 /
 *  calf x4 — never index actuators positionally, always via actuator_trnid. */
export const JOINT_SUFFIXES = Object.freeze([
  'FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint',
  'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint',
  'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint',
  'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint',
]);

/** math.radians(70.0) — src/tasks/game/scenarios/touchdown.py:123.
 *  The threshold `fallen()` uses unless the scene or the caller overrides it. */
export const FALL_LIMIT_ANGLE = 1.2217304763960306;

const MJ_TRN_JOINT = 0;   // mjtTrn.mjTRN_JOINT
const MJ_JNT_FREE = 0;    // mjtJoint.mjJNT_FREE
const MJ_JNT_HINGE = 3;   // mjtJoint.mjJNT_HINGE
const MJ_GEOM_MESH = 7;   // mjtGeom.mjGEOM_MESH

const GEOM_TYPE_NAMES = [
  'plane', 'hfield', 'sphere', 'capsule', 'ellipsoid',
  'cylinder', 'box', 'mesh', 'sdf',
];

// --------------------------------------------------------------------------
// environment-neutral byte reading
// --------------------------------------------------------------------------

const IS_NODE =
  typeof process !== 'undefined' &&
  process.versions != null &&
  process.versions.node != null &&
  typeof window === 'undefined';

let nodeFs = null;
async function getNodeFs() {
  if (!nodeFs) nodeFs = await import('node:fs/promises');
  return nodeFs;
}

function isNetUrl(u) {
  return /^(https?|data|blob):/i.test(u);
}

/** Read a URL or path as bytes. Browser: fetch. Node: fetch for http(s), fs otherwise. */
async function readBytes(url) {
  if (!IS_NODE || isNetUrl(url)) {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`fetch ${url} -> HTTP ${res.status}`);
    return new Uint8Array(await res.arrayBuffer());
  }
  const fs = await getNodeFs();
  const path = url.startsWith('file:') ? new URL(url).pathname : url;
  return new Uint8Array(await fs.readFile(path));
}

async function readText(url) {
  return new TextDecoder().decode(await readBytes(url));
}

/** Join a base ("dir/" or "dir") with a relative path. Leaves absolute URLs alone. */
function joinUrl(base, rel) {
  if (!base) return rel;
  if (isNetUrl(rel) || rel.startsWith('/')) return rel;
  return base.endsWith('/') ? base + rel : `${base}/${rel}`;
}

function dirOf(url) {
  const i = url.lastIndexOf('/');
  return i < 0 ? '' : url.slice(0, i + 1);
}

// --------------------------------------------------------------------------
// module singleton
// --------------------------------------------------------------------------

let modulePromise = null;
let moduleWasmUrl = null;

/**
 * Instantiate (once per page / per process) the MuJoCo WASM module.
 * Compiling the 4.5 MB module takes ~200 ms, so every sim shares one instance.
 * @param {string} [wasmUrl] override the location of mujoco.wasm.
 */
export function loadMujocoModule(wasmUrl) {
  if (modulePromise && wasmUrl && wasmUrl !== moduleWasmUrl) {
    throw new Error(
      `loadMujocoModule: already instantiated from "${moduleWasmUrl}", cannot ` +
      `switch to "${wasmUrl}" (the module is a per-process singleton)`);
  }
  if (!modulePromise) {
    moduleWasmUrl = wasmUrl ?? null;
    const opts = wasmUrl ? { locateFile: (p) => (p.endsWith('.wasm') ? wasmUrl : p) } : undefined;
    modulePromise = loadMujoco(opts);
  }
  return modulePromise;
}

// --------------------------------------------------------------------------
// MJCF asset discovery
// --------------------------------------------------------------------------

function attrOf(tagText, name) {
  const m = new RegExp(`\\b${name}\\s*=\\s*"([^"]*)"`).exec(tagText);
  return m ? m[1] : null;
}

/**
 * Scan an MJCF document for the asset files it references, returning
 * meshdir/texturedir-resolved relative names — exactly the names MuJoCo will
 * look up. Handles <mesh>, <texture>, <hfield>, <skin>, <model> and <include>,
 * and recurses into nested <model file> / <include file> documents.
 *
 * (A regex scan suffices: MJCF has no CDATA and no namespaces, node has no
 * DOMParser, and an XML parser dependency would break "no bundler, no CDN".)
 */
function scanAssets(xmlText, docDir, out, seen) {
  const compiler = /<compiler\b[^>]*>/i.exec(xmlText);
  const assetdir = compiler ? attrOf(compiler[0], 'assetdir') : null;
  const meshdir = (compiler ? attrOf(compiler[0], 'meshdir') : null) ?? assetdir ?? '';
  const texdir = (compiler ? attrOf(compiler[0], 'texturedir') : null) ?? assetdir ?? '';

  const tagRe = /<(mesh|texture|hfield|skin|model|include)\b[^>]*?>/gi;
  const nested = [];
  let m;
  while ((m = tagRe.exec(xmlText)) !== null) {
    const tag = m[1].toLowerCase();
    const file = attrOf(m[0], 'file');
    if (!file) continue;
    if (tag === 'model' || tag === 'include') {
      // Nested MJCF: the path is relative to the including document.
      if (!seen.has(file)) {
        seen.add(file);
        nested.push({ vfsName: file, fetchRel: joinUrl(docDir, file) });
      }
      continue;
    }
    const dir = tag === 'texture' ? texdir : tag === 'hfield' ? (assetdir ?? '') : meshdir;
    const vfsName = dir ? `${dir.replace(/\/$/, '')}/${file}` : file;
    if (!seen.has(vfsName)) {
      seen.add(vfsName);
      out.push({ vfsName, fetchRel: vfsName });
    }
  }
  return nested;
}

// --------------------------------------------------------------------------
// quaternion helpers (w, x, y, z)
// --------------------------------------------------------------------------

/** R^T v — rotate a world vector into the body frame. */
function quatApplyInverse(q, v, out) {
  const w = q[0], x = q[1], y = q[2], z = q[3];
  const vx = v[0], vy = v[1], vz = v[2];
  // t = 2 * (q_vec x v);  R v = v + w t + q_vec x t;  R^T v = v - w t + q_vec x t
  const tx = 2 * (y * vz - z * vy);
  const ty = 2 * (z * vx - x * vz);
  const tz = 2 * (x * vy - y * vx);
  out[0] = vx - w * tx + (y * tz - z * ty);
  out[1] = vy - w * ty + (z * tx - x * tz);
  out[2] = vz - w * tz + (x * ty - y * tx);
  return out;
}

function yawToQuat(yaw, out) {
  const h = 0.5 * yaw;
  out[0] = Math.cos(h);
  out[1] = 0;
  out[2] = 0;
  out[3] = Math.sin(h);
  return out;
}

// --------------------------------------------------------------------------
// misc
// --------------------------------------------------------------------------

/**
 * Turn whatever MuJoCo threw into a string.
 *
 * The build uses `-fexceptions -s DISABLE_EXCEPTION_CATCHING=0`, so a C++ throw
 * arrives as a raw pointer (a number) or a WebAssembly.Exception, and only then
 * does `getExceptionMessage` apply. Handing it an ordinary JS Error makes it
 * read out of bounds and abort the whole module — which is how a plain "bad
 * MJCF" turns into `RuntimeError: memory access out of bounds`. Never call it
 * unguarded.
 */
function describeThrow(mj, e) {
  if (typeof mj.getExceptionMessage === 'function' &&
      (typeof e === 'number' ||
       (typeof WebAssembly.Exception === 'function' && e instanceof WebAssembly.Exception))) {
    try {
      const m = mj.getExceptionMessage(e);
      if (m) return Array.isArray(m) ? m.filter(Boolean).join(': ') : String(m);
    } catch { /* fall through to plain stringification */ }
  }
  return e && e.message ? e.message : String(e);
}

/**
 * Name lookup straight out of `model.names`.
 * `mj_id2name` returns the junk string "emsc" for unnamed objects in this
 * build, so it is not used anywhere in this file (see header note 4).
 */
function makeNamer(model) {
  const names = model.names; // Int8Array over the wasm heap
  const dec = new TextDecoder();
  const cache = new Map();
  return function nameAt(adrArray, id) {
    if (!adrArray || id < 0 || id >= adrArray.length) return null;
    const adr = adrArray[id];
    if (adr < 0 || adr >= names.length) return null;
    const key = adr;
    if (cache.has(key)) return cache.get(key);
    let end = adr;
    while (end < names.length && names[end] !== 0) end++;
    const out = end === adr ? null : dec.decode(names.subarray(adr, end));
    cache.set(key, out);
    return out;
  };
}

/** Canonicalise a seat key: A/attacker -> 'a', B/defender -> 'b'. */
function canonSeat(k) {
  const s = String(k).toLowerCase();
  if (s === 'a' || s === 'attacker') return 'a';
  if (s === 'b' || s === 'defender') return 'b';
  return null;
}

// --------------------------------------------------------------------------
// seat / index resolution
// --------------------------------------------------------------------------

/**
 * Work out which bodies / geoms / joints / actuators belong to seat 'a' and
 * seat 'b'.
 *
 * The compiled model is the primary source: exactly two free-joint root bodies,
 * with ownership of every geom resolved through `body_rootid`. That map is
 * exact and prefix-free, and it is precisely what mjlab's referee sensors
 * encode (`<contact geom1="a_*_collision" subtree2="b_base_link">`,
 * assets/scene/sym/scene.xml:433+).
 *
 * `scene.json` (schema `s2c_web_play/scene@1`, written by tools/export_scene.py)
 * is consumed when present: explicit joint / actuator / body / geom lists are
 * used, and every overlapping signal is cross-checked against the derived map,
 * with any disagreement pushed onto `sim.warnings`.
 *
 * Accepted scene.json shapes (all optional, first match wins). Written against
 * `s2c_web_play/scene@1` and `@2`; the schema string itself is not checked, only
 * the keys:
 *   robots.<a|b>.baseBodyName | baseBody | prefix
 *   robots.<a|b>.jointNames | jointIds | joints              (12, JOINT order)
 *   robots.<a|b>.actuatorNames | actuatorIdsJointOrder | actuators
 *   robots.<a|b>.geomIds | geomNames                         (the geom -> robot map)
 *   robots.<a|b>.bodyIds | bodyNames | collisionGeoms        (cross-checks)
 *   geomRobot: {"<geom name>": "a"|"b"} | ["a"|"b"|null, ...ngeom]
 */
function resolveSeats(model, sceneJson, warnings, nameOfBody, nameOfJoint,
                      nameOfActuator, nameOfGeom) {
  const jntType = model.jnt_type;
  const jntBody = model.jnt_bodyid;
  const bodyRootid = model.body_rootid;
  const geomBodyid = model.geom_bodyid;

  const byName = (adr, n, want) => {
    if (want == null) return -1;
    for (let i = 0; i < n; i++) if (adr(i) === want) return i;
    return -1;
  };
  const bodyId = (nm) => byName(nameOfBody, model.nbody, nm);
  const jointId = (nm) => byName(nameOfJoint, model.njnt, nm);
  const actId = (nm) => byName(nameOfActuator, model.nu, nm);
  const geomId = (nm) => byName(nameOfGeom, model.ngeom, nm);

  // --- 1. the two free-floating roots --------------------------------------
  const freeRoots = [];
  for (let j = 0; j < model.njnt; j++) {
    if (jntType[j] === MJ_JNT_FREE) freeRoots.push({ jointId: j, bodyId: jntBody[j] });
  }
  if (freeRoots.length !== 2) {
    throw new Error(
      `scene must contain exactly 2 free-joint robots, found ${freeRoots.length}`);
  }

  const explicit = sceneJson && sceneJson.robots ? sceneJson.robots : null;
  const seatOf = {};
  for (const seat of ['a', 'b']) {
    const cfg = explicit ? explicit[seat] : null;
    const wanted = cfg && (cfg.baseBodyName || cfg.baseBody ||
      (cfg.prefix != null ? `${cfg.prefix}base_link` : null));
    if (!wanted) continue;
    const bid = bodyId(wanted);
    if (bid < 0) throw new Error(`scene.json names body "${wanted}" which the model does not have`);
    const hit = freeRoots.find((r) => r.bodyId === bid);
    if (!hit) throw new Error(`scene.json body "${wanted}" is not a free-joint root`);
    seatOf[seat] = hit;
  }
  if (!seatOf.a || !seatOf.b) {
    const tagged = freeRoots.map((r) => {
      const nm = nameOfBody(r.bodyId) ?? '';
      const prefix = nm.endsWith('base_link') ? nm.slice(0, -'base_link'.length) : `${nm}/`;
      return { ...r, name: nm, prefix };
    });
    const pick = (re) => tagged.find((t) => re.test(t.prefix));
    let a = pick(/^a[/_-]?$/i) || pick(/^attacker[/_-]?$/i);
    let b = pick(/^b[/_-]?$/i) || pick(/^defender[/_-]?$/i);
    if (!a && !b) {
      const sorted = [...tagged].sort((p, q) => p.bodyId - q.bodyId);
      [a, b] = sorted;
      warnings.push(
        `seat assignment fell back to body order: a="${a.name}", b="${b.name}" ` +
        '(no scene.json robots map and no a/b or attacker/defender prefix)');
    } else if (!a) {
      a = tagged.find((t) => t !== b);
    } else if (!b) {
      b = tagged.find((t) => t !== a);
    }
    seatOf.a = seatOf.a || a;
    seatOf.b = seatOf.b || b;
  }
  if (seatOf.a.bodyId === seatOf.b.bodyId) throw new Error('seats a and b resolved to the same body');

  // --- 2. joint -> actuator map from the transmission (never by position) ---
  const jointToAct = new Int32Array(model.njnt).fill(-1);
  const trnType = model.actuator_trntype;
  const trnId = model.actuator_trnid;
  for (let u = 0; u < model.nu; u++) {
    if (trnType[u] !== MJ_TRN_JOINT) continue;
    const j = trnId[2 * u];
    if (j >= 0 && j < model.njnt) jointToAct[j] = u;
  }

  // --- 3. per-seat joint list in JOINT order -------------------------------
  const seats = {};
  for (const seat of ['a', 'b']) {
    const root = seatOf[seat];
    const bodyName = nameOfBody(root.bodyId) ?? `body${root.bodyId}`;
    const prefix = bodyName.endsWith('base_link')
      ? bodyName.slice(0, -'base_link'.length)
      : '';
    const cfg = explicit ? explicit[seat] : null;

    // Names beat ids: a scene.json written against a different MuJoCo build
    // could carry stale indices, but a joint name is a joint name.
    let jointIds = null;
    const jn = cfg && (cfg.jointNames || (Array.isArray(cfg.joints) &&
      typeof cfg.joints[0] === 'string' ? cfg.joints : null));
    const ji = cfg && (cfg.jointIds || (Array.isArray(cfg.joints) &&
      typeof cfg.joints[0] === 'number' ? cfg.joints : null));
    if (jn && jn.length === 12) {
      jointIds = jn.map(jointId);
      if (ji && ji.length === 12 && jointIds.some((v, i) => v !== ji[i])) {
        warnings.push(`seat ${seat}: scene.json jointIds disagree with jointNames; using the names`);
      }
    } else if (ji && ji.length === 12) {
      jointIds = ji.slice();
    } else {
      jointIds = JOINT_SUFFIXES.map((suf) => jointId(prefix + suf));
    }
    if (jointIds.some((j) => j < 0 || j >= model.njnt)) {
      // last resort: the seat's hinge joints in model order
      const hinge = [];
      for (let j = 0; j < model.njnt; j++) {
        if (jntType[j] === MJ_JNT_HINGE && bodyRootid[jntBody[j]] === root.bodyId) hinge.push(j);
      }
      if (hinge.length !== 12) {
        throw new Error(
          `seat ${seat}: could not resolve 12 joints (prefix "${prefix}", found ` +
          `${hinge.length} hinges)`);
      }
      jointIds = hinge;
      warnings.push(
        `seat ${seat}: joint names did not match ${JOINT_SUFFIXES[0]}-style suffixes; ` +
        'fell back to model hinge order, which may NOT be FL,FR,RL,RR x (hip,thigh,calf)');
    }
    for (const j of jointIds) {
      if (bodyRootid[jntBody[j]] !== root.bodyId) {
        warnings.push(
          `seat ${seat}: joint "${nameOfJoint(j)}" does not belong to ${bodyName}`);
        break;
      }
    }

    let actIds;
    const an = cfg && cfg.actuatorNames;
    const ai = cfg && (cfg.actuatorIdsJointOrder ||
      (Array.isArray(cfg.actuators) && typeof cfg.actuators[0] === 'number' ? cfg.actuators : null));
    if (an && an.length === 12) {
      actIds = an.map(actId);
    } else if (ai && ai.length === 12) {
      actIds = ai.slice();
    } else {
      actIds = jointIds.map((j) => jointToAct[j]);
    }
    // Always cross-check against the transmission: this is the hip x4 /
    // thigh x4 / calf x4 vs FL,FR,RL,RR trap, and it is silent when wrong.
    const fromTrn = jointIds.map((j) => jointToAct[j]);
    if (actIds.some((u, i) => u !== fromTrn[i])) {
      warnings.push(
        `seat ${seat}: scene.json actuator list disagrees with actuator_trnid; ` +
        'using actuator_trnid (the transmission cannot lie)');
      actIds = fromTrn;
    }
    if (actIds.some((u) => u < 0)) {
      warnings.push(
        `seat ${seat}: ${actIds.filter((u) => u < 0).length} of 12 joints have no ` +
        'position actuator; setCtrl will skip them');
    }

    seats[seat] = {
      key: seat,
      prefix,
      baseBody: root.bodyId,
      baseBodyName: bodyName,
      freeJoint: root.jointId,
      freeQposAdr: model.jnt_qposadr[root.jointId],
      freeDofAdr: model.jnt_dofadr[root.jointId],
      jointIds: Int32Array.from(jointIds),
      jointNames: jointIds.map((j) => nameOfJoint(j)),
      qposAdr: Int32Array.from(jointIds.map((j) => model.jnt_qposadr[j])),
      dofAdr: Int32Array.from(jointIds.map((j) => model.jnt_dofadr[j])),
      actIds: Int32Array.from(actIds),
      actNames: actIds.map((u) => (u >= 0 ? nameOfActuator(u) : null)),
    };
  }

  // --- 4. geom -> seat, derived from body_rootid ---------------------------
  const derived = new Array(model.ngeom).fill(null);
  for (let g = 0; g < model.ngeom; g++) {
    const root = bodyRootid[geomBodyid[g]];
    if (root === seats.a.baseBody) derived[g] = 'a';
    else if (root === seats.b.baseBody) derived[g] = 'b';
  }

  let geomSeat = derived;

  // scene.json's own geom -> robot information, folded into one sparse map.
  // `s2c_web_play/scene@2` carries it as robots.<r>.geomIds / .geomNames, which
  // list only the 23 NAMED collision geoms per seat — the 66 visual mesh geoms
  // are unnamed and absent. So it is a PARTIAL map: it is cross-checked against
  // the body_rootid map entry by entry, and body_rootid stays the source of
  // truth for the geoms it does not mention (the renderer needs the visual
  // meshes attributed too). Only a COMPLETE top-level `geomRobot` overrides.
  const fromJson = new Array(model.ngeom).fill(null);
  let jsonEntries = 0;
  const putJson = (gid, v) => {
    const s = canonSeat(v);
    if (gid >= 0 && gid < model.ngeom && s) { fromJson[gid] = s; jsonEntries++; }
  };
  const top = sceneJson && (sceneJson.geomRobot || sceneJson.geomSeat);
  if (Array.isArray(top)) {
    for (let g = 0; g < Math.min(top.length, model.ngeom); g++) putJson(g, top[g]);
  } else if (top) {
    for (const [k, v] of Object.entries(top)) {
      putJson(/^\d+$/.test(k) ? Number(k) : geomId(k), v);
    }
  } else if (explicit) {
    for (const s of ['a', 'b']) {
      for (const g of ((explicit[s] || {}).geomIds || [])) putJson(g, s);
      for (const nm of ((explicit[s] || {}).geomNames || [])) putJson(geomId(nm), s);
    }
  }
  if (jsonEntries > 0) {
    let mismatch = 0;
    for (let g = 0; g < model.ngeom; g++) {
      if (fromJson[g] !== null && fromJson[g] !== derived[g]) mismatch++;
    }
    if (mismatch > 0) {
      warnings.push(
        `scene.json assigns ${mismatch} of its ${jsonEntries} geom entries to a seat ` +
        'the compiled model disagrees with; using scene.json for those — check the export');
      geomSeat = derived.map((d, g) => (fromJson[g] !== null ? fromJson[g] : d));
    }
    // A map that covers every geom replaces the derived one outright.
    if (jsonEntries === model.ngeom) geomSeat = fromJson;
  }

  // Cross-check the remaining scene.json lists against the derived ownership.
  if (explicit) {
    for (const s of ['a', 'b']) {
      const cfg = explicit[s] || {};
      let bad = 0, n = 0;
      for (const g of (cfg.collisionGeoms || []).concat(cfg.footGeoms || [])) {
        n++;
        const gid = geomId(g);
        if (gid < 0 || derived[gid] !== s) bad++;
      }
      for (const b of (cfg.bodyNames || [])) {
        n++;
        const bid = bodyId(b);
        if (bid < 0 || bodyRootid[bid] !== seats[s].baseBody) bad++;
      }
      for (const b of (cfg.bodyIds || [])) {
        n++;
        if (!(b >= 0 && b < model.nbody) || bodyRootid[b] !== seats[s].baseBody) bad++;
      }
      if (bad > 0) {
        warnings.push(
          `seat ${s}: ${bad}/${n} scene.json geom/body entries are not owned by ` +
          `${seats[s].baseBodyName} in the compiled model`);
      }
    }
  }

  // Independent sanity check on the bitmasks: mjlab gives seat a
  // contype=1/conaffinity=2 and seat b 2/1 (robots.py:220-244, recon/05:75),
  // which is what makes robot-vs-robot collide and self-collision not.
  {
    const ct = model.geom_contype, ca = model.geom_conaffinity;
    let crossable = 0;
    for (let g = 0; g < model.ngeom; g++) {
      if (geomSeat[g] !== 'a' || (ct[g] | ca[g]) === 0) continue;
      for (let h = 0; h < model.ngeom; h++) {
        if (geomSeat[h] !== 'b') continue;
        if ((ct[g] & ca[h]) || (ct[h] & ca[g])) { crossable++; break; }
      }
      if (crossable) break;
    }
    if (!crossable) {
      warnings.push(
        'no geom of seat a can collide with any geom of seat b under the scene ' +
        'contype/conaffinity masks — the two robots will pass through each other');
    }
  }

  // Int8 lookup for the hot contact scan: 1 = a, 2 = b, 0 = neither.
  const geomSeatCode = new Int8Array(model.ngeom);
  for (let g = 0; g < model.ngeom; g++) {
    geomSeatCode[g] = geomSeat[g] === 'a' ? 1 : geomSeat[g] === 'b' ? 2 : 0;
  }

  return { seats, geomSeat, geomSeatCode };
}

// --------------------------------------------------------------------------
// createSim
// --------------------------------------------------------------------------

/**
 * Build a Sim for one scene.
 *
 * @param {object} opts
 * @param {string} opts.sceneUrl        URL/path of the scene MJCF (alias: xmlUrl).
 * @param {string} [opts.assetsBase]    base for mesh/texture files (default: sceneUrl's directory).
 * @param {string} [opts.sceneJsonUrl]  default: scene.json next to sceneUrl (absent is fine).
 * @param {object} [opts.sceneJson]     pre-parsed scene.json (skips the fetch).
 * @param {string} [opts.wasmUrl]       override the mujoco.wasm location.
 * @param {number[]} [opts.defaultJointPos] 12, JOINT order. Default: scene.json,
 *                                      else the scene keyframe, else Go2 nominal.
 * @param {number} [opts.spawnZ]        default: scene keyframe, else 0.32.
 * @param {number} [opts.fallLimitAngle] rad; default scene.json rules.fallLimitAngleRad,
 *                                      else FALL_LIMIT_ANGLE (70 deg, training).
 * @param {number} [opts.fallBaseZMin]  extra "base too low" test; default -Infinity
 *                                      (training installs no height termination).
 * @returns {Promise<Sim>}
 */
export async function createSim({
  sceneUrl,
  xmlUrl,
  assetsBase,
  sceneJsonUrl,
  sceneJson,
  wasmUrl,
  defaultJointPos,
  spawnZ,
  fallLimitAngle,
  fallBaseZMin,
} = {}) {
  const url = sceneUrl ?? xmlUrl;
  if (!url) throw new Error('createSim: sceneUrl is required');
  const docDir = dirOf(url);
  const base = assetsBase == null ? docDir : assetsBase;
  const warnings = [];

  const mj = await loadMujocoModule(wasmUrl);

  // ---- scene.json (optional) ---------------------------------------------
  let scene = sceneJson ?? null;
  if (!scene) {
    const jsonUrl = sceneJsonUrl ?? joinUrl(docDir, 'scene.json');
    try {
      scene = JSON.parse(await readText(jsonUrl));
    } catch {
      scene = null;
      warnings.push(`no scene.json at ${jsonUrl}; seat/geom maps derived from the model`);
    }
  }

  // ---- XML + assets into the VFS -----------------------------------------
  //
  // THE INCANTATION (copy this):
  //   const vfs = new mj.MjVFS();
  //   vfs.addBuffer('scene.xml', new TextEncoder().encode(xmlText));
  //   for (const f of meshes) vfs.addBuffer(f.name, f.bytes);   // meshdir-relative
  //   const model = mj.MjModel.mj_loadXML('scene.xml', vfs);
  //   const data  = new mj.MjData(model);
  // addBuffer takes (name: string, bytes: Uint8Array). The VFS name must be the
  // string MuJoCo will look up, i.e. the compiler's meshdir prefix included
  // ("assets/base_0.obj" for `<compiler meshdir="assets">` + `file="base_0.obj"`).
  const xmlText = await readText(url);
  const vfs = new mj.MjVFS();
  const ROOT = 'scene.xml';
  vfs.addBuffer(ROOT, new TextEncoder().encode(xmlText));

  const assets = [];
  const seen = new Set([ROOT]);
  let queue = scanAssets(xmlText, '', assets, seen);
  while (queue.length) {
    const next = [];
    for (const doc of queue) {
      const text = await readText(joinUrl(base, doc.fetchRel));
      vfs.addBuffer(doc.vfsName, new TextEncoder().encode(text));
      next.push(...scanAssets(text, dirOf(doc.vfsName), assets, seen));
    }
    queue = next;
  }
  // scene.json may list assets the regex cannot see (e.g. a lite mesh dir).
  const declared = (scene && scene.files && scene.files.meshFiles) || (scene && scene.assets);
  const meshDir = (scene && scene.files && scene.files.meshDir) || '';
  if (Array.isArray(declared)) {
    for (const f of declared) {
      const vfsName = f.includes('/') || !meshDir ? f : `${meshDir.replace(/\/$/, '')}/${f}`;
      if (!seen.has(vfsName)) { seen.add(vfsName); assets.push({ vfsName, fetchRel: vfsName }); }
    }
  }
  const fetched = await Promise.all(assets.map((a) => readBytes(joinUrl(base, a.fetchRel))));
  let assetBytes = 0;
  for (let i = 0; i < assets.length; i++) {
    vfs.addBuffer(assets[i].vfsName, fetched[i]);
    assetBytes += fetched[i].byteLength;
  }

  // ---- compile ------------------------------------------------------------
  let model;
  try {
    model = mj.MjModel.mj_loadXML(ROOT, vfs);
  } catch (e) {
    try { vfs.delete(); } catch { /* ignore */ }
    throw new Error(`MuJoCo failed to compile ${url}: ${describeThrow(mj, e)}`);
  }
  const data = new mj.MjData(model);

  const nameAt = makeNamer(model);
  const nameOfBody = (i) => nameAt(model.name_bodyadr, i);
  const nameOfJoint = (i) => nameAt(model.name_jntadr, i);
  const nameOfActuator = (i) => nameAt(model.name_actuatoradr, i);
  const nameOfGeom = (i) => nameAt(model.name_geomadr, i);
  const nameOfMesh = (i) => nameAt(model.name_meshadr, i);
  const nameOfMat = (i) => nameAt(model.name_matadr, i);

  const { seats, geomSeat, geomSeatCode } = resolveSeats(
    model, scene, warnings, nameOfBody, nameOfJoint, nameOfActuator, nameOfGeom);

  // ---- reset defaults -----------------------------------------------------
  let defJoint = defaultJointPos ? Float64Array.from(defaultJointPos) : null;
  let defZ = spawnZ;
  if (!defJoint && scene && scene.joints && Array.isArray(scene.joints.defaultJointPos)
      && scene.joints.defaultJointPos.length === 12) {
    defJoint = Float64Array.from(scene.joints.defaultJointPos);
  }
  if ((!defJoint || defZ == null) && model.nkey > 0) {
    const kq = model.key_qpos;
    const a = seats.a;
    if (!defJoint) {
      defJoint = new Float64Array(12);
      for (let i = 0; i < 12; i++) defJoint[i] = kq[a.qposAdr[i]];
    }
    if (defZ == null) defZ = kq[a.freeQposAdr + 2];
  }
  if (!defJoint) defJoint = Float64Array.from(GO2_DEFAULT_JOINT_POS);
  if (defZ == null) defZ = GO2_SPAWN_Z;
  if (defJoint.length !== 12) throw new Error('defaultJointPos must have 12 entries');

  const fallLimit = fallLimitAngle
    ?? (scene && scene.rules && typeof scene.rules.fallLimitAngleRad === 'number'
      ? scene.rules.fallLimitAngleRad
      : FALL_LIMIT_ANGLE);
  const fallZMin = fallBaseZMin ?? -Infinity;

  // ---- reused buffers -----------------------------------------------------
  const renderBuf = new Float32Array(model.nbody * 7);
  const quatScratch = new Float64Array(4);
  const vec3Scratch = new Float64Array(3);
  const DOWN = [0, 0, -1];

  let poseDirty = false; // set by setPose/setBaseVel; cleared by forward()/step()

  const seat = (r) => {
    const k = canonSeat(r);
    const s = k ? seats[k] : null;
    if (!s) throw new Error(`unknown robot "${r}" (expected 'a' or 'b')`);
    return s;
  };

  function ensureFresh() {
    if (poseDirty) { mj.mj_forward(model, data); poseDirty = false; }
  }

  function readQuat(bodyId, out) {
    const xq = data.xquat;
    const o = 4 * bodyId;
    out[0] = xq[o]; out[1] = xq[o + 1]; out[2] = xq[o + 2]; out[3] = xq[o + 3];
    return out;
  }

  const sim = {
    // ---- raw handles, for the renderer and the safety filter ---------------
    mj, model, data, vfs,

    nq: model.nq,
    nv: model.nv,
    nu: model.nu,
    nbody: model.nbody,
    ngeom: model.ngeom,
    njnt: model.njnt,
    nmesh: model.nmesh,
    nkey: model.nkey,

    sceneUrl: url,
    assetsBase: base,
    assetBytes,
    assetNames: assets.map((a) => a.vfsName),
    sceneJson: scene,
    warnings,
    seats: {
      a: {
        prefix: seats.a.prefix, baseBody: seats.a.baseBodyName,
        baseBodyId: seats.a.baseBody,
        jointIds: Array.from(seats.a.jointIds), jointNames: seats.a.jointNames,
        actIds: Array.from(seats.a.actIds), actNames: seats.a.actNames,
      },
      b: {
        prefix: seats.b.prefix, baseBody: seats.b.baseBodyName,
        baseBodyId: seats.b.baseBody,
        jointIds: Array.from(seats.b.jointIds), jointNames: seats.b.jointNames,
        actIds: Array.from(seats.b.actIds), actNames: seats.b.actNames,
      },
    },
    defaultJointPos: Array.from(defJoint),
    spawnZ: defZ,
    /** rad; `fallen()` trips above this tilt. Training value: 70 deg. */
    fallLimitAngle: fallLimit,
    fallBaseZMin: fallZMin,
    /** physics timestep as compiled into the scene (never set here). */
    get timestep() { return model.opt.timestep; },
    get time() { return data.time; },
    get ncon() { return data.ncon; },

    // ---- stepping -----------------------------------------------------------
    /**
     * Advance `n` physics substeps. Leaves xpos/xquat/cvel/contact lagging
     * qpos/qvel by one substep — the frame the training referee reads. Call
     * `forward()` before building observations. See header note 2.
     */
    step(n = 1) {
      ensureFresh();
      for (let i = 0; i < n; i++) mj.mj_step(model, data);
      poseDirty = false;
    },
    /** The single mj_forward of the control step: derived arrays catch up. */
    forward() {
      mj.mj_forward(model, data);
      poseDirty = false;
    },

    // ---- control ------------------------------------------------------------
    /** @param {'a'|'b'} robot @param {ArrayLike<number>} arr12 JOINT order, absolute setpoints. */
    setCtrl(robot, arr12) {
      const s = seat(robot);
      if (arr12.length !== 12) throw new Error('setCtrl expects 12 values');
      const ctrl = data.ctrl;
      for (let i = 0; i < 12; i++) {
        const u = s.actIds[i];
        if (u >= 0) ctrl[u] = arr12[i];
      }
    },
    getCtrl(robot) {
      const s = seat(robot);
      const out = new Float64Array(12);
      const ctrl = data.ctrl;
      for (let i = 0; i < 12; i++) out[i] = s.actIds[i] >= 0 ? ctrl[s.actIds[i]] : 0;
      return out;
    },
    /**
     * Rewrite this seat's position-actuator gains, in JOINT order.
     * MuJoCo position actuators are gaintype=FIXED / biastype=AFFINE, i.e.
     * force = kp*(ctrl - q) - kd*qdot encoded as gainprm[0]=kp,
     * biasprm[1]=-kp, biasprm[2]=-kd (recon/05:127-139). This is the hook the
     * shield's 20/20/40 -> 100/100/200 gain blend needs
     * (safety_collision/rl/touchdown_driver.py:42-44, :117-126); nothing else
     * should call it.
     */
    setGains(robot, kp12, kd12) {
      const s = seat(robot);
      const gp = model.actuator_gainprm, bp = model.actuator_biasprm;
      const NG = gp.length / model.nu, NB = bp.length / model.nu;
      for (let i = 0; i < 12; i++) {
        const u = s.actIds[i];
        if (u < 0) continue;
        gp[NG * u] = kp12[i];
        bp[NB * u + 1] = -kp12[i];
        bp[NB * u + 2] = -kd12[i];
      }
    },
    getGains(robot) {
      const s = seat(robot);
      const gp = model.actuator_gainprm, bp = model.actuator_biasprm;
      const NG = gp.length / model.nu, NB = bp.length / model.nu;
      const kp = new Float64Array(12), kd = new Float64Array(12);
      for (let i = 0; i < 12; i++) {
        const u = s.actIds[i];
        if (u < 0) continue;
        kp[i] = gp[NG * u];
        kd[i] = -bp[NB * u + 2];
      }
      return { kp, kd };
    },

    // ---- state reads --------------------------------------------------------
    /** Absolute joint angles, JOINT order. Fresh array each call. */
    getJointPos(robot) {
      const s = seat(robot);
      ensureFresh();
      const q = data.qpos;
      const out = new Float64Array(12);
      for (let i = 0; i < 12; i++) out[i] = q[s.qposAdr[i]];
      return out;
    },
    getJointVel(robot) {
      const s = seat(robot);
      ensureFresh();
      const v = data.qvel;
      const out = new Float64Array(12);
      for (let i = 0; i < 12; i++) out[i] = v[s.dofAdr[i]];
      return out;
    },
    /**
     * Base-link state — the exact quantities the 60-D and 47-D obs need.
     *   pos     [3] world position of base_link's ORIGIN (not the COM)
     *   quat    [4] world orientation, w-FIRST
     *   linVelW [3] base_link's WORLD linear velocity
     *   angVelW [3] world angular velocity (= cvel[0:3])
     *   angVelB [3] BODY-frame angular velocity — the obs term `base_ang_vel`
     * See header note 1 for the mjlab citations and the proof.
     */
    getBase(robot) {
      const s = seat(robot);
      ensureFresh();
      const b = s.baseBody;
      const xpos = data.xpos, cvel = data.cvel, com = data.subtree_com;
      const px = xpos[3 * b], py = xpos[3 * b + 1], pz = xpos[3 * b + 2];
      const wx = cvel[6 * b], wy = cvel[6 * b + 1], wz = cvel[6 * b + 2];
      const lx = cvel[6 * b + 3], ly = cvel[6 * b + 4], lz = cvel[6 * b + 5];
      const ox = com[3 * b] - px, oy = com[3 * b + 1] - py, oz = com[3 * b + 2] - pz;
      const q = readQuat(b, quatScratch);
      const angB = quatApplyInverse(q, [wx, wy, wz], new Float64Array(3));
      return {
        pos: [px, py, pz],
        quat: [q[0], q[1], q[2], q[3]],
        linVelW: [lx - (wy * oz - wz * oy), ly - (wz * ox - wx * oz), lz - (wx * oy - wy * ox)],
        angVelW: [wx, wy, wz],
        angVelB: [angB[0], angB[1], angB[2]],
      };
    },
    /** projected_gravity_b = quat_apply_inverse(xquat, (0,0,-1)); UNIT vector. */
    projectedGravity(robot) {
      const s = seat(robot);
      ensureFresh();
      return quatApplyInverse(readQuat(s.baseBody, quatScratch), DOWN, new Float64Array(3));
    },
    /** yaw = atan2 of the body +x axis in world — mjlab's heading_w (data.py:553-556). */
    yaw(robot) {
      const s = seat(robot);
      ensureFresh();
      const q = readQuat(s.baseBody, quatScratch);
      const w = q[0], x = q[1], y = q[2], z = q[3];
      // forward_w = R * (1,0,0)
      const fx = 1 - 2 * (y * y + z * z);
      const fy = 2 * (x * y + w * z);
      return Math.atan2(fy, fx);
    },
    /** acos(-projected_gravity_b[2]) — the quantity training thresholds at 70 deg. */
    tiltAngle(robot) {
      const s = seat(robot);
      ensureFresh();
      const g = quatApplyInverse(readQuat(s.baseBody, quatScratch), DOWN, vec3Scratch);
      return Math.acos(Math.min(1, Math.max(-1, -g[2])));
    },

    // ---- pose ---------------------------------------------------------------
    /**
     * Teleport a robot and zero its velocities.
     *
     * `z` defaults to the scene spawn height (0.32); `yaw` becomes the free
     * joint's quaternion (w-first, about world +z). qvel and qacc_warmstart for
     * this robot's 6 base DOFs and 12 joint DOFs are zeroed, so a reset never
     * inherits momentum from the previous episode. The next state read (or
     * step()/forward()) refreshes the derived arrays.
     *
     * @param {'a'|'b'} robot
     * @param {{x?:number,y?:number,yaw?:number,z?:number,quat?:number[],jointPos?:ArrayLike<number>}} pose
     */
    setPose(robot, { x, y, yaw, z, quat, jointPos } = {}) {
      const s = seat(robot);
      const q = data.qpos, v = data.qvel, aw = data.qacc_warmstart;
      const fq = s.freeQposAdr, fd = s.freeDofAdr;
      if (x != null) q[fq] = x;
      if (y != null) q[fq + 1] = y;
      q[fq + 2] = z ?? defZ;
      if (quat) {
        for (let i = 0; i < 4; i++) q[fq + 3 + i] = quat[i];
      } else if (yaw != null) {
        yawToQuat(yaw, quatScratch);
        for (let i = 0; i < 4; i++) q[fq + 3 + i] = quatScratch[i];
      }
      for (let i = 0; i < 6; i++) { v[fd + i] = 0; if (aw) aw[fd + i] = 0; }
      if (jointPos) {
        if (jointPos.length !== 12) throw new Error('setPose jointPos expects 12 values');
        for (let i = 0; i < 12; i++) q[s.qposAdr[i]] = jointPos[i];
      }
      for (let i = 0; i < 12; i++) { v[s.dofAdr[i]] = 0; if (aw) aw[s.dofAdr[i]] = 0; }
      poseDirty = true;
    },

    /**
     * Write the base velocity. `linVelW` is WORLD frame, `angVelB` is BODY
     * frame — the free joint's own convention, measured on this build (header
     * note 1, proven by tests/node_physics_smoke.mjs §6). Unused by the game
     * loop; it exists so the spin test can prove the convention and so a future
     * reset-kick event can reproduce training's `velocity kick`.
     */
    setBaseVel(robot, { linVelW, angVelB } = {}) {
      const s = seat(robot);
      const v = data.qvel, fd = s.freeDofAdr;
      if (linVelW) for (let i = 0; i < 3; i++) v[fd + i] = linVelW[i];
      if (angVelB) for (let i = 0; i < 3; i++) v[fd + 3 + i] = angVelB[i];
      poseDirty = true;
    },

    /**
     * Full reset: mj_resetData, both seats to `spawn`, joints to the default
     * pose, ctrl seeded to that same pose, then one mj_forward.
     *
     * Seeding ctrl = the measured joint_pos mirrors the increment integrator's
     * reset rule ("target re-seeded from the post-reset measured joint_pos",
     * safety/mdp/ctrl_action.py:107-123; recon/05:178), so a step taken before
     * the first setCtrl holds the stance instead of snapping to 0 rad.
     *
     * Seat keys are tolerant: {a,b}, {A,B} and {attacker,defender} all work, so
     * GAMES.sym.spawn and GAMES.asym.spawn from app/config.js can be passed
     * straight through once app/match.js has fixed the seat->robot mapping.
     */
    resetAll(spawn) {
      mj.mj_resetData(model, data);
      const byRobot = { a: null, b: null };
      for (const [k, val] of Object.entries(spawn || {})) {
        const r = canonSeat(k);
        if (r) byRobot[r] = val;
      }
      for (const r of ['a', 'b']) {
        const p = byRobot[r] || {};
        const jp = p.jointPos || defJoint;
        sim.setPose(r, { x: p.x, y: p.y, yaw: p.yaw, z: p.z, quat: p.quat, jointPos: jp });
        sim.setCtrl(r, jp);
      }
      mj.mj_forward(model, data);
      poseDirty = false;
    },

    // ---- referee inputs -----------------------------------------------------
    /**
     * True as soon as ANY geom of seat a touches ANY geom of seat b.
     * This is the live v81 STRICT rule that the `a_vs_b_trunk_*` /
     * `b_vs_a_trunk_*` sensors encode (game_env_cfg.py:245-254; recon/05:82,
     * :252). Cheap: it stops at the first hit.
     */
    anyContactBetweenRobots() {
      const n = data.ncon;
      if (n === 0) return false;
      // ⚠ `data.contact` is NOT a view: every property access COPIES the whole
      // std::vector<mjContact> onto the wasm heap, and the copy must be
      // .delete()d. Measured: 5 000 un-deleted accesses at ncon=72 grew the
      // heap by 186 MiB and 40 000 hit the 2 GiB ALLOW_MEMORY_GROWTH ceiling.
      // Read it ONCE per call, always free it.
      const vec = data.contact;
      let hit = false;
      try {
        for (let i = 0; i < n; i++) {
          const c = vec.get(i);
          if (c === undefined) continue;
          if (c.exclude === 0) {
            const s1 = geomSeatCode[c.geom1];
            const s2 = geomSeatCode[c.geom2];
            if (s1 !== 0 && s2 !== 0 && s1 !== s2) hit = true;
          }
          c.delete();
          if (hit) break;
        }
      } finally {
        vec.delete();
      }
      return hit;
    },
    /** Back-compat alias for the boolean above. */
    trunkContact() { return sim.anyContactBetweenRobots(); },

    /**
     * Every seat-a <-> seat-b contact of the current frame.
     *
     * @returns {{geomA:number, geomB:number, nameA:string|null, nameB:string|null,
     *            pos:number[], normal:number[], dist:number, geom1:number, geom2:number}[]}
     *   geomA is always the seat-a geom and geomB the seat-b geom (geom1/geom2
     *   keep MuJoCo's own ordering). `normal` is the unit contact normal
     *   ORIENTED FROM A TOWARDS B: MuJoCo's contact frame row 0 points from
     *   geom1 to geom2 (measured on this build: a at x=0, b at x=+0.37, the
     *   head-on trunk pair reports geom1=a_base1_collision and normal=(1,0,0)),
     *   so it is negated whenever geom1 belonged to seat b.
     *   `dist` < 0 is penetration depth. `pos` is the world contact point.
     *
     * Reads the same stale-by-one-substep frame the training referee reads
     * (header note 2). The geom -> robot map comes from scene.json when it
     * carries one, otherwise from body_rootid (see resolveSeats).
     */
    contactsBetweenRobots() {
      const n = data.ncon;
      const out = [];
      if (n === 0) return out;
      const vec = data.contact;   // heap copy — see anyContactBetweenRobots
      try {
        for (let i = 0; i < n; i++) {
          const c = vec.get(i);
          if (c === undefined) continue;
          const g1 = c.geom1, g2 = c.geom2;
          const s1 = geomSeatCode[g1], s2 = geomSeatCode[g2];
          if (c.exclude === 0 && s1 !== 0 && s2 !== 0 && s1 !== s2) {
            const aFirst = s1 === 1;
            const f = c.frame;            // 9 floats, rows = (normal, tan1, tan2)
            const p = c.pos;
            const sgn = aFirst ? 1 : -1;
            out.push({
              geomA: aFirst ? g1 : g2,
              geomB: aFirst ? g2 : g1,
              nameA: nameOfGeom(aFirst ? g1 : g2),
              nameB: nameOfGeom(aFirst ? g2 : g1),
              pos: [p[0], p[1], p[2]],
              normal: [sgn * f[0], sgn * f[1], sgn * f[2]],
              dist: c.dist,
              geom1: g1,
              geom2: g2,
            });
          }
          c.delete();
        }
      } finally {
        vec.delete();
      }
      return out;
    },

    /**
     * The training fall termination, `mdp.bad_orientation`:
     *   acos(-projected_gravity_b[2]) > fallLimitAngle     (70 deg by default)
     * mjlab/envs/mdp/terminations.py:24-32, installed at touchdown.py:419-437 /
     * sym_touchdown.py:243-262 with fall_limit_angle = math.radians(70)
     * (touchdown.py:123). `fallBaseZMin` is off by default because training
     * installs no height termination; see header note 3.
     */
    fallen(robot) {
      const s = seat(robot);
      ensureFresh();
      if (sim.tiltAngle(robot) > fallLimit) return true;
      return data.xpos[3 * s.baseBody + 2] < fallZMin;
    },

    // ---- renderer feed ------------------------------------------------------
    /**
     * nbody*7 floats: [x, y, z, qw, qx, qy, qz] per body, world frame.
     * REUSED buffer — copy it if you intend to keep it past the next call.
     * Compose with describeGeoms()[g].pos/.quat (body-LOCAL) to place a geom.
     */
    renderState() {
      ensureFresh();
      const xpos = data.xpos, xquat = data.xquat;
      for (let b = 0; b < model.nbody; b++) {
        const o = 7 * b;
        renderBuf[o] = xpos[3 * b];
        renderBuf[o + 1] = xpos[3 * b + 1];
        renderBuf[o + 2] = xpos[3 * b + 2];
        renderBuf[o + 3] = xquat[4 * b];
        renderBuf[o + 4] = xquat[4 * b + 1];
        renderBuf[o + 5] = xquat[4 * b + 2];
        renderBuf[o + 6] = xquat[4 * b + 3];
      }
      return renderBuf;
    },

    /** Static body table: names, parents, and the seat each belongs to. */
    describeBodies() {
      const parent = model.body_parentid, root = model.body_rootid;
      const out = new Array(model.nbody);
      for (let b = 0; b < model.nbody; b++) {
        out[b] = {
          bodyId: b,
          name: nameOfBody(b),
          parentId: parent[b],
          rootId: root[b],
          robot: root[b] === seats.a.baseBody ? 'a' : root[b] === seats.b.baseBody ? 'b' : null,
          mass: model.body_mass ? model.body_mass[b] : 0,
        };
      }
      return out;
    },

    /** Static material table (rgba + the PBR-ish scalars MuJoCo carries). */
    describeMaterials() {
      const out = new Array(model.nmat);
      for (let m = 0; m < model.nmat; m++) {
        out[m] = {
          matId: m,
          name: nameOfMat(m),
          rgba: [model.mat_rgba[4 * m], model.mat_rgba[4 * m + 1],
            model.mat_rgba[4 * m + 2], model.mat_rgba[4 * m + 3]],
          emission: model.mat_emission[m],
          specular: model.mat_specular[m],
          shininess: model.mat_shininess[m],
          reflectance: model.mat_reflectance[m],
        };
      }
      return out;
    },

    /**
     * Static geom table — everything the renderer needs to build the scene once.
     * `pos`/`quat` are body-LOCAL: world = renderState()[body] composed with them.
     * `rgba` is the effective colour: MuJoCo's abstract visualizer takes the
     * material's rgba when matid >= 0 and the geom's own otherwise, so both are
     * reported (`matRgba`, `geomRgba`) and `rgba` is the resolved one.
     * `group` 2 = the visual meshes, 3 = the 23 collision primitives per robot.
     * `meshId >= 0` -> call getMesh(meshId).
     */
    describeGeoms() {
      const gType = model.geom_type, gBody = model.geom_bodyid, gSize = model.geom_size;
      const gPos = model.geom_pos, gQuat = model.geom_quat, gRgba = model.geom_rgba;
      const gData = model.geom_dataid, gGroup = model.geom_group, gMat = model.geom_matid;
      const matRgba = model.mat_rgba;
      const out = new Array(model.ngeom);
      for (let g = 0; g < model.ngeom; g++) {
        const matid = gMat ? gMat[g] : -1;
        const geomRgba = [gRgba[4 * g], gRgba[4 * g + 1], gRgba[4 * g + 2], gRgba[4 * g + 3]];
        const mrgba = matid >= 0 && matRgba
          ? [matRgba[4 * matid], matRgba[4 * matid + 1],
            matRgba[4 * matid + 2], matRgba[4 * matid + 3]]
          : null;
        out[g] = {
          geomId: g,
          name: nameOfGeom(g),
          bodyId: gBody[g],
          bodyName: nameOfBody(gBody[g]),
          robot: geomSeat[g],
          type: gType[g],
          typeName: GEOM_TYPE_NAMES[gType[g]] ?? String(gType[g]),
          group: gGroup[g],
          size: [gSize[3 * g], gSize[3 * g + 1], gSize[3 * g + 2]],
          pos: [gPos[3 * g], gPos[3 * g + 1], gPos[3 * g + 2]],
          quat: [gQuat[4 * g], gQuat[4 * g + 1], gQuat[4 * g + 2], gQuat[4 * g + 3]],
          rgba: mrgba ?? geomRgba,
          geomRgba,
          matRgba: mrgba,
          matId: matid,
          matName: matid >= 0 ? nameOfMat(matid) : null,
          meshId: gType[g] === MJ_GEOM_MESH ? gData[g] : -1,
          meshName: gType[g] === MJ_GEOM_MESH ? nameOfMesh(gData[g]) : null,
          contype: model.geom_contype[g],
          conaffinity: model.geom_conaffinity[g],
          condim: model.geom_condim[g],
        };
      }
      return out;
    },

    /**
     * Triangle soup for one mesh, copied out of the wasm heap into plain
     * Float32/Uint32 arrays (safe to hand straight to a THREE.BufferGeometry;
     * the wasm heap can be reallocated under you, these copies cannot).
     * Mesh scale is already baked into mesh_vert by the compiler.
     * @returns {{meshId, name, vertices:Float32Array, normals:Float32Array,
     *            indices:Uint32Array, texcoords:Float32Array|null, indexed:boolean}}
     */
    getMesh(meshId) {
      if (!(meshId >= 0 && meshId < model.nmesh)) throw new Error(`no mesh ${meshId}`);
      const vAdr = model.mesh_vertadr[meshId], vNum = model.mesh_vertnum[meshId];
      const fAdr = model.mesh_faceadr[meshId], fNum = model.mesh_facenum[meshId];
      const nAdr = model.mesh_normaladr ? model.mesh_normaladr[meshId] : vAdr;
      const nNum = model.mesh_normalnum ? model.mesh_normalnum[meshId] : vNum;
      const mv = model.mesh_vert, mn = model.mesh_normal, mf = model.mesh_face;
      const mfn = model.mesh_facenormal;
      const tAdr = model.mesh_texcoordadr ? model.mesh_texcoordadr[meshId] : -1;
      const tNum = model.mesh_texcoordnum ? model.mesh_texcoordnum[meshId] : 0;

      // The indexed fast path is only valid when every triangle corner uses the
      // same index for its normal as for its vertex. Checked, never assumed.
      let sharedIndexing = nNum === vNum;
      if (sharedIndexing && mfn) {
        for (let i = 0, n = 3 * fNum; i < n; i++) {
          if (mfn[3 * fAdr + i] !== mf[3 * fAdr + i]) { sharedIndexing = false; break; }
        }
      }
      const name = nameOfMesh(meshId);
      if (sharedIndexing) {
        const vertices = new Float32Array(3 * vNum);
        const normals = new Float32Array(3 * vNum);
        for (let i = 0; i < 3 * vNum; i++) vertices[i] = mv[3 * vAdr + i];
        for (let i = 0; i < 3 * nNum; i++) normals[i] = mn[3 * nAdr + i];
        const indices = new Uint32Array(3 * fNum);
        for (let i = 0; i < 3 * fNum; i++) indices[i] = mf[3 * fAdr + i];
        let texcoords = null;
        if (tAdr >= 0 && tNum === vNum && model.mesh_texcoord) {
          texcoords = new Float32Array(2 * vNum);
          for (let i = 0; i < 2 * vNum; i++) texcoords[i] = model.mesh_texcoord[2 * tAdr + i];
        }
        return { meshId, name, vertices, normals, indices, texcoords, indexed: true };
      }
      // Distinct normal indexing: expand to one vertex per triangle corner.
      const vertices = new Float32Array(9 * fNum);
      const normals = new Float32Array(9 * fNum);
      const indices = new Uint32Array(3 * fNum);
      for (let f = 0; f < fNum; f++) {
        for (let k = 0; k < 3; k++) {
          const vi = mf[3 * (fAdr + f) + k];
          const ni = mfn ? mfn[3 * (fAdr + f) + k] : vi;
          const o = 9 * f + 3 * k;
          vertices[o] = mv[3 * (vAdr + vi)];
          vertices[o + 1] = mv[3 * (vAdr + vi) + 1];
          vertices[o + 2] = mv[3 * (vAdr + vi) + 2];
          normals[o] = mn[3 * (nAdr + ni)];
          normals[o + 1] = mn[3 * (nAdr + ni) + 1];
          normals[o + 2] = mn[3 * (nAdr + ni) + 2];
          indices[3 * f + k] = 3 * f + k;
        }
      }
      return { meshId, name, vertices, normals, indices, texcoords: null, indexed: false };
    },

    /**
     * Current size of the WASM linear memory, in bytes. The build exports
     * neither HEAPU8 nor wasmMemory, so read it off any live heap view (every
     * typed array the bindings hand back is a view onto that memory).
     * Diagnostics only — the smoke test uses it to catch embind handle leaks.
     */
    heapBytes() {
      return data.qpos.buffer.byteLength;
    },

    /** Free the wasm-side objects. The module itself stays loaded. */
    dispose() {
      try { data.delete(); } catch { /* already gone */ }
      try { model.delete(); } catch { /* already gone */ }
      try { vfs.delete(); } catch { /* already gone */ }
    },
  };

  mj.mj_forward(model, data);
  return sim;
}

export default { createSim, loadMujocoModule };

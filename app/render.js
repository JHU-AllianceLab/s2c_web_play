/**
 * app/render.js — three.js renderer for S2C Web Play.
 *
 * Builds the whole visible scene from what app/physics.js already exposes:
 *   sim.describeGeoms()  -> static per-geom table (type / size / body-local pos+quat / rgba / group / robot)
 *   sim.getMesh(id)      -> heap-copied triangle soup for the 32 Go2 visual meshes
 *   sim.renderState()    -> nbody*7 [x y z qw qx qy qz], world frame, REUSED buffer
 * and drives it every animation frame from renderState() alone. Nothing here
 * touches collision geometry, ctrl, qpos or the referee: the renderer is a
 * pure consumer.
 *
 * FRAME CONVENTION. MuJoCo is z-up. Rather than re-parent the world under a
 * -90 deg X rotation, the scene stays in MuJoCo coordinates and every camera
 * gets `camera.up = (0,0,1)` — the same choice DeepMind's own bridge makes
 * (`google-deepmind/mujoco:wasm/demo_app/app.ts:150`, recon/06 B.3). Field x/y,
 * lineX and the spawn poses are therefore usable verbatim.
 *
 * ---------------------------------------------------------------------------
 * CONSTANTS TAKEN FROM THE SCENE / THE CODE (nothing here is invented)
 *
 *  haze / horizon   (0.15, 0.25, 0.35)      assets/scene/{sym,asym}/scene.xml:10  <rgba haze="...">
 *  headlight ambient 0.3                    assets/scene/<game>/scene.xml:8  <headlight ambient="0.3 0.3 0.3">
 *  headlight diffuse 0.6                    assets/scene/<game>/scene.xml:8
 *  ground rgb1/rgb2 (0.2,0.3,0.4)/(0.1,0.2,0.3), markrgb (0.8,0.8,0.8), mark="edge"
 *                                           assets/scene/<game>/scene.xml:38  (builtin checker texture)
 *                                           = mjlab/terrains/terrain_entity.py:23-24
 *  ground checker period 0.5 m, cell 0.25 m MEASURED by rendering exactly that
 *                                           texture+material string through mujoco 3.5.0's own
 *                                           renderer top-down and reading the transition
 *                                           spacing off the image: 0.2444 m/cell at 0.00414 m/px,
 *                                           markrgb edge every second boundary. See
 *                                           tests/render_shots.mjs --selftest notes.
 *  Go2 material rgba: metal (0.9,0.95,0.95) / black (0,0,0) / white (1,1,1)
 *                     gray (0.671705,0.692426,0.77427)
 *                                           assets/scene/<game>/scene.xml:40-47
 *  broadcast az 90 / el -45                 the angle the 11th-meeting demo clips use,
 *                                           Meeting/Video/material_sim/demo_sim_source/README.txt:19
 *                                           (via recon/06 B.3); reference distance 5.1,
 *                                           lookat z 0.25 from the same line and from
 *                                           scripts/render_game_checkpoint.py:206-212.
 *  MuJoCo free-camera convention            forward = (cos el cos az, cos el sin az, sin el),
 *                                           pos = lookat - dist*forward. VERIFIED numerically
 *                                           against mujoco 3.5.0 mjv_updateScene for
 *                                           (az 90, el -45), (az 90, el -55), (az 135, el -25):
 *                                           max |d forward| = 1.4e-08.
 *  head offset for FPV  x = +0.293          `a_base3_collision` pos, assets/scene/<game>/scene.xml:97
 *                                           (the Go2 "nose" primitive on base_link).
 *
 * Team colours are a presentation choice, declared once in TEAM below:
 * player teal #1F6F8B is the paper's S2C colour, opponent is a warm grey.
 * ---------------------------------------------------------------------------
 *
 * THE VENUE (DESIGN.md section 12)
 *
 * The scene above is the plant. Everything that makes it read as a *place* —
 * the matte hall floor, the walls, the spectator bowl and its crowd, the barrier
 * boards, the ceiling rig, the contact shadows and the bloom — lives in
 * app/venue.js and is added to this same scene graph. It is visual-only: no geom,
 * no constant and no file MuJoCo reads is touched by it, and scene.xml's md5 is
 * unchanged. `createRenderer(..., {venue: false})` skips it and restores the
 * previous look (the mjlab checker disc and the gradient sky dome), which is what
 * the venue/no-venue A/B comparison in tests/shots is shot with.
 *
 * With the venue on, the two constants above that described the *old* backdrop —
 * GROUND_TEX (the mjlab checker) and the sky dome's haze — are unused: an indoor
 * hall has neither an infinite checkerboard nor a sky. They are kept because the
 * {venue:false} path still draws them.
 *
 * ---------------------------------------------------------------------------
 *
 * PERFORMANCE CONTRACT
 *  - Scene is built ONCE. describeGeoms()/getMesh() are never called per frame.
 *  - update() allocates nothing: every Vector3/Quaternion/Matrix4/Color it needs
 *    is preallocated in the closure (see the `_v3a … _up` block).
 *  - Render rate is decoupled from the 50 Hz control loop: update() measures its
 *    own wall-clock dt, all smoothing is `1 - exp(-k*dt)` (frame-rate
 *    independent), and body poses are interpolated between the last two distinct
 *    physics frames so a 60/120/144 Hz display is smooth over a 50 Hz sim.
 */

import * as THREE from '../vendor/three/three.module.js';
import { buildVenue, createContactShadows, createBloom } from './venue.js';

// ---------------------------------------------------------------------------
// MuJoCo enums / scene constants
// ---------------------------------------------------------------------------

/** mjtGeom, matching scene.json.model.geomTypeEnum. */
const MJ_PLANE = 0, MJ_HFIELD = 1, MJ_SPHERE = 2, MJ_CAPSULE = 3,
  MJ_ELLIPSOID = 4, MJ_CYLINDER = 5, MJ_BOX = 6, MJ_MESH = 7;

/** scene.xml:10  <rgba haze="0.15 0.25 0.35 1"> — horizon / fog / sky base. */
const HAZE = [0.15, 0.25, 0.35];
/** scene.xml:8  <headlight ambient="0.3 0.3 0.3" diffuse="0.6 0.6 0.6"> */
const HEADLIGHT = { ambient: 0.3, diffuse: 0.6 };
/** scene.xml:38  builtin="checker" rgb1/rgb2/markrgb, and the measured period. */
const GROUND_TEX = {
  rgb1: [0.2, 0.3, 0.4],
  rgb2: [0.1, 0.2, 0.3],
  markrgb: [0.8, 0.8, 0.8],
  /** metres per full texture tile (2x2 checker cells) — measured, see header. */
  periodM: 0.5,
  /** markrgb edge width as a fraction of the tile — measured ~12 mm / 500 mm. */
  markFrac: 12 / 500,
};
/** Radius of the visible ground disc. Purely cosmetic; the MJCF plane is infinite. */
const GROUND_RADIUS = 80;

/** The fixed broadcast shot. az/el are MuJoCo free-camera angles, degrees. */
const BROADCAST = { azimuthDeg: 90, elevationDeg: -45, refDistance: 5.1, lookatZ: 0.25 };

/**
 * FPV eye: 0.11 m ahead of the Go2's nose primitive (`base3_collision` pos
 * x = 0.293, scene.xml:97) and a little above it. Sitting exactly ON the nose
 * puts base_4 (the head mesh) through the near plane, so the shot is taken from
 * just in front of it — a nose cam, not an inside-the-skull cam.
 */
const FPV_EYE = { x: 0.404, y: 0, z: 0.092 };

/** Team ramps: dark -> mid -> light, indexed by the source material's luminance. */
const TEAM = {
  player: { key: 'player', name: 'You', dark: 0x0c242d, mid: 0x1f6f8b, light: 0x7fd2e8, ring: 0x3fc9e8 },
  ai: { key: 'ai', name: 'Opponent', dark: 0x241f1b, mid: 0x8a7458, light: 0xe8dccb, ring: 0xd8a05a },
};

export const CAMERA_MODES = ['chase', 'fpv', 'broadcast'];

const DEG = Math.PI / 180;

/**
 * Per-mode lens + smoothing.
 *   w*   natural frequency of a CRITICALLY DAMPED spring, rad/s (chase, broadcast)
 *   k*   rate of a first-order exponential follow, 1/s          (fpv, lens easing)
 * Both are frame-rate independent; the difference is the shape. A first-order
 * follow jumps to its maximum speed on frame 1 and then trails an exponential
 * tail, which is exactly the "icy" feel. The critically damped spring starts at
 * zero velocity, accelerates, and arrives without overshoot — the boom has
 * weight but always catches up. FPV stays first-order on purpose: the eye is
 * bolted to the skull and must not be allowed to lag it (see stepCamera).
 * Settling time for critical damping is ~5.8/w: chase 0.53 s, broadcast 0.64 s.
 */
const MODE_CFG = {
  chase: { fov: 52, near: 0.05, wPos: 11.0, wTgt: 16.0 },
  fpv: { fov: 78, near: 0.015, kPos: 26.0, kQuat: 18.0 },
  broadcast: { fov: 38, near: 0.08, wPos: 9.0, wTgt: 9.0 },
};

/**
 * The chase boom. Everything here is FEEL — no training constant is involved —
 * but the numbers are chosen against the pitch this game is played on
 * (5.2 x 3.0 m, app/config.js GAMES.asym.field) and against the 52 deg vertical
 * lens above, which at 16:9 is 81 deg horizontal.
 *
 *  dist/elev   3.45 m at 25 deg = 3.13 m behind and 1.46 m above the boom pivot,
 *              so the lens sits ~1.94 m off the floor: high enough to look over
 *              the far dog instead of through it, close enough that the player's
 *              own dog still reads. (It was 2.35 m at 20 deg = 2.21 m back and
 *              1.24 m up, which is what let the opponent fill the foreground.)
 *  the duel    the aim point leans off the player toward the opponent when the
 *              two are close, so the shot frames the contest and not a backside.
 *              At the 3.13 m boom the frame is +-2.66 m wide at the dog's range;
 *              an opponent within ~2.5 m of the player is in shot without help.
 *  the guard   when the opponent crosses the line of sight the boom lifts and
 *              backs off instead of letting it eclipse the player.
 *  floorZ      the boom TARGET never goes under this, so the spring is never
 *              chasing a point inside the floor (the hard clamp in stepCamera is
 *              only the backstop).
 */
const CHASE = {
  dist: 3.45, minDist: 1.6, maxDist: 9.0,
  elev: 25 * DEG, minElev: 7 * DEG, maxElev: 78 * DEG,
  pivotZ: 0.20,                 // boom pivot, above the dog's base body
  aimZ: 0.16,                   // look-at, above the dog's base body
  lead: 0.28, leadMax: 1.1,     // velocity lead on the aim, m and metres cap
  speedStretch: 0.10,           // boom grows this fraction per m/s of dog speed
  speedStretchMax: 0.34,        // ... capped here (3.45 -> 4.62 m flat out)
  duelBias: 0.52,               // aim leans at most this fraction of the gap
  duelNear: 1.0, duelFar: 3.8,  // full lean at/below near, none at/above far
  duelAimMax: 0.95,             // ... and never further than this, metres
  guardRadius: 0.78,            // opponent this close to the sight line is in the way
  guardLift: 26 * DEG,          // elevation added at full intrusion
  guardPush: 0.95,              // boom added at full intrusion, metres
  floorZ: 0.45,                 // minimum ground clearance of the boom target
  hardFloorZ: 0.32,             // ... and of the camera itself, every mode
};

/** Mouse orbit, radians per pixel at sensitivity 1, and the wheel's zoom decade. */
const ORBIT_RAD_PER_PX = 0.0055;
const ZOOM_PER_WHEEL_PX = 0.0012;

const clamp = (x, lo, hi) => (x < lo ? lo : x > hi ? hi : x);
/** Shortest signed difference a-b, wrapped to (-pi, pi]. */
function angDelta(a, b) {
  let d = (a - b) % (2 * Math.PI);
  if (d > Math.PI) d -= 2 * Math.PI;
  if (d <= -Math.PI) d += 2 * Math.PI;
  return d;
}
/** Frame-rate independent exponential follow weight. */
const follow = (k, dt) => 1 - Math.exp(-k * dt);
/** Hermite smoothstep on an already-clamped [0,1] input. */
const smoothstep = (t) => t * t * (3 - 2 * t);

/**
 * One EXACT step of a critically damped spring, applied per axis to a
 * THREE.Vector3 pair (pos carries the state, vel carries its derivative).
 *
 * Solving x'' = -2w x' - w^2 x in closed form rather than integrating it means
 * the result is identical at 30, 60 and 144 fps, cannot overshoot, and cannot
 * blow up at a long frame — all three of which a fixed-factor lerp gets wrong.
 * With e = exp(-w dt), d0 = pos - tgt and c = vel + w d0:
 *      d(dt) = (d0 + c dt) e          v(dt) = (c (1 - w dt) - w d0) e
 * Steady-state lag behind a target moving at speed s is 2 s / w.
 */
function critDamp3(pos, vel, tgt, w, dt) {
  const e = Math.exp(-w * dt);
  let d = pos.x - tgt.x, c = vel.x + w * d;
  pos.x = tgt.x + (d + c * dt) * e;
  vel.x = (c * (1 - w * dt) - w * d) * e;
  d = pos.y - tgt.y; c = vel.y + w * d;
  pos.y = tgt.y + (d + c * dt) * e;
  vel.y = (c * (1 - w * dt) - w * d) * e;
  d = pos.z - tgt.z; c = vel.z + w * d;
  pos.z = tgt.z + (d + c * dt) * e;
  vel.z = (c * (1 - w * dt) - w * d) * e;
}

function canonRobot(r) {
  if (r == null) return null;
  const s = String(r).toLowerCase();
  if (s === 'a' || s === 'attacker') return 'a';
  if (s === 'b' || s === 'defender') return 'b';
  return null;
}

/** MuJoCo rgba are display-referred, so feed them to three as sRGB. */
function mjColor(out, r, g, b) { return out.setRGB(r, g, b, THREE.SRGBColorSpace); }
const REC709 = (r, g, b) => 0.2126 * r + 0.7152 * g + 0.0722 * b;

// ---------------------------------------------------------------------------
// geometry factories — MuJoCo geom size semantics -> three geometry
// ---------------------------------------------------------------------------
//  sphere    size[0] = radius
//  capsule   size[0] = radius, size[1] = half-length along LOCAL Z
//  ellipsoid size[0..2] = semi-axes
//  cylinder  size[0] = radius, size[1] = half-height along LOCAL Z
//  box       size[0..2] = half-extents
//  plane     size[0..1] = half-extents (0 = infinite), size[2] = grid spacing
// three's Capsule/Cylinder run along Y, so both are rotated +90 deg about X.

function makePrimitive(type, sx, sy, sz, quality) {
  const seg = quality === 'low' ? 10 : quality === 'medium' ? 16 : 24;
  let g;
  switch (type) {
    case MJ_SPHERE:
      g = new THREE.SphereGeometry(sx, seg, Math.max(6, seg >> 1));
      break;
    case MJ_CAPSULE:
      g = new THREE.CapsuleGeometry(sx, 2 * sy, Math.max(3, seg >> 2), seg);
      g.rotateX(Math.PI / 2);
      break;
    case MJ_ELLIPSOID:
      g = new THREE.SphereGeometry(1, seg, Math.max(6, seg >> 1));
      g.scale(sx, sy, sz);
      break;
    case MJ_CYLINDER:
      g = new THREE.CylinderGeometry(sx, sx, 2 * sy, seg, 1);
      g.rotateX(Math.PI / 2);
      break;
    case MJ_BOX:
      g = new THREE.BoxGeometry(2 * sx, 2 * sy, 2 * sz);
      break;
    case MJ_PLANE: {
      const hx = sx > 0 ? sx : GROUND_RADIUS;
      const hy = sy > 0 ? sy : GROUND_RADIUS;
      g = new THREE.PlaneGeometry(2 * hx, 2 * hy, 1, 1); // already in XY, +z normal
      break;
    }
    default:
      g = new THREE.BoxGeometry(2 * (sx || 0.02), 2 * (sy || 0.02), 2 * (sz || 0.02));
  }
  return g;
}

function meshGeometry(m) {
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(m.vertices, 3));
  if (m.normals && m.normals.length === m.vertices.length) {
    g.setAttribute('normal', new THREE.BufferAttribute(m.normals, 3));
  }
  if (m.indexed) g.setIndex(new THREE.BufferAttribute(m.indices, 1));
  if (!g.getAttribute('normal')) g.computeVertexNormals();
  g.computeBoundingSphere();
  return g;
}

/**
 * The mjlab ground checker, rebuilt byte-exactly from the MJCF's own
 * rgb1/rgb2/markrgb: one texture tile = one 2x2 checker + a markrgb edge.
 */
function groundTexture() {
  const N = 256, half = N >> 1;
  const mark = Math.max(1, Math.round(N * GROUND_TEX.markFrac));
  const px = new Uint8Array(N * N * 4);
  const b = (v) => Math.round(clamp(v, 0, 1) * 255);
  const c1 = [b(GROUND_TEX.rgb1[0]), b(GROUND_TEX.rgb1[1]), b(GROUND_TEX.rgb1[2])];
  const c2 = [b(GROUND_TEX.rgb2[0]), b(GROUND_TEX.rgb2[1]), b(GROUND_TEX.rgb2[2])];
  const cm = [b(GROUND_TEX.markrgb[0]), b(GROUND_TEX.markrgb[1]), b(GROUND_TEX.markrgb[2])];
  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      const edge = x < mark || y < mark || x >= N - mark || y >= N - mark;
      const c = edge ? cm : ((x < half) === (y < half) ? c1 : c2);
      const o = 4 * (y * N + x);
      px[o] = c[0]; px[o + 1] = c[1]; px[o + 2] = c[2]; px[o + 3] = 255;
    }
  }
  const t = new THREE.DataTexture(px, N, N, THREE.RGBAFormat);
  t.colorSpace = THREE.SRGBColorSpace;
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.magFilter = THREE.LinearFilter;
  t.minFilter = THREE.LinearMipmapLinearFilter;
  t.generateMipmaps = true;
  const reps = (2 * GROUND_RADIUS) / GROUND_TEX.periodM;
  t.repeat.set(reps, reps);
  t.needsUpdate = true;
  return t;
}

/** Gradient sky dome, z-up, with a soft sun bloom toward the key light. */
function makeSky(hazeColor) {
  const zenith = new THREE.Color().copy(hazeColor).multiplyScalar(0.42).offsetHSL(0, 0.10, 0.03);
  const mat = new THREE.ShaderMaterial({
    side: THREE.BackSide,
    depthWrite: false,
    fog: false,
    uniforms: {
      uHorizon: { value: new THREE.Color().copy(hazeColor) },
      uZenith: { value: zenith },
      uSun: { value: new THREE.Vector3(0, 0, 1) },
      uSunColor: { value: new THREE.Color(0xfff2d8) },
    },
    vertexShader: /* glsl */`
      varying vec3 vDir;
      void main() {
        vDir = normalize(position);
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }`,
    // The chunks below are what keeps a raw ShaderMaterial consistent with the
    // renderer's tone mapping + output colour space (three does not add them
    // to custom shaders automatically).
    fragmentShader: /* glsl */`
      uniform vec3 uHorizon; uniform vec3 uZenith; uniform vec3 uSun; uniform vec3 uSunColor;
      varying vec3 vDir;
      void main() {
        float h = clamp(vDir.z, -1.0, 1.0);
        float t = pow(clamp(h, 0.0, 1.0), 0.55);
        vec3 col = mix(uHorizon, uZenith, t);
        col = mix(col * 0.82, col, smoothstep(-0.25, 0.02, h));      // slight dip below horizon
        float s = max(dot(normalize(vDir), normalize(uSun)), 0.0);
        col += uSunColor * (pow(s, 48.0) * 0.55 + pow(s, 6.0) * 0.07);
        gl_FragColor = vec4(col, 1.0);
        #include <tonemapping_fragment>
        #include <colorspace_fragment>
      }`,
  });
  const sky = new THREE.Mesh(new THREE.SphereGeometry(200, 32, 16), mat);
  sky.frustumCulled = false;
  sky.renderOrder = -1000;
  sky.matrixAutoUpdate = false;
  return sky;
}

// ---------------------------------------------------------------------------
// createRenderer
// ---------------------------------------------------------------------------

/**
 * @param {HTMLCanvasElement} canvas
 * @param {object} sim      an app/physics.js Sim
 * @param {object|string} game  'sym' | 'asym', or the app/config.js GAMES entry
 *                              ({field:[w,h], center:[x,y], lineX, ...}); when it
 *                              carries neither, sim.sceneJson.field is used.
 * @param {object} [options]
 *   quality               'low' | 'medium' | 'high'   (default 'high')
 *   playerRobot           'a' | 'b'                   (default 'a'; also settable later)
 *   interpolate           boolean                     (default true)
 *   controls              attach mouse orbit/zoom     (default true)
 *   cameraDistance        chase boom multiplier       (default 1; = setCameraDistance)
 *   mouseSensitivity      orbit rate multiplier       (default 1; = setMouseSensitivity)
 *   preserveDrawingBuffer for screenshot harnesses     (default false)
 *   antialias             default true
 *   maxPixelRatio         default 2
 * @returns renderer handle — update / setCamera / resize are the frozen surface.
 */
export function createRenderer(canvas, sim, game, options = {}) {
  if (!canvas) throw new Error('createRenderer: canvas is required');
  if (!sim || typeof sim.describeGeoms !== 'function') {
    throw new Error('createRenderer: sim must be an app/physics.js Sim');
  }

  const warnings = [];
  /**
   * Both ends live (the symmetric game) vs one (asymmetric). It decides whether
   * an end zone is neutral turf or is owned by one of the two dogs.
   */
  const bothScore =
    typeof game === 'string' ? game === 'sym' : Boolean(game && game.rules && game.rules.bothSeatsScore);
  const opt = {
    quality: 'high',
    interpolate: true,
    controls: true,
    preserveDrawingBuffer: false,
    antialias: true,
    maxPixelRatio: 2,
    /** build app/venue.js around the pitch; false restores the pre-venue look. */
    venue: true,
    ...options,
  };

  // ---- field geometry -----------------------------------------------------
  const field = resolveField(game, sim, warnings);

  // ---- three core ---------------------------------------------------------
  const renderer = new THREE.WebGLRenderer({
    canvas,
    antialias: opt.antialias,
    alpha: false,
    powerPreference: 'high-performance',
    preserveDrawingBuffer: opt.preserveDrawingBuffer,
  });
  renderer.setClearColor(0x000000, 1);
  // ACES, per DESIGN.md section 12. It rolls the ceiling rig's highlights off
  // instead of clipping them (which is what makes the bloom read as light rather
  // than as a white patch) at the cost of a little saturation in the endzone
  // green and the touchdown red; the exposure below is set to put the pitch back
  // where Neutral had it. The bloom composite in app/venue.js goes through the
  // renderer's own tone-mapping include, so both paths share this one curve.
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = opt.venue ? 1.32 : 1.0;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;

  const hazeColor = mjColor(new THREE.Color(), HAZE[0], HAZE[1], HAZE[2]);
  const scene = new THREE.Scene();
  // Exponential, not linear: from a 1.2 m chase camera most of the visible floor
  // sits well inside any sensible linear `far`, so only exp fog actually grades it.
  // Indoors the density is much higher — the far wall is 11 m away, not 80 — and
  // it is what hides the seam where the bowl meets the hall wall.
  scene.fog = new THREE.FogExp2(hazeColor.clone(), opt.venue ? 0.055 : 0.019);

  const camera = new THREE.PerspectiveCamera(MODE_CFG.chase.fov, 1.6, MODE_CFG.chase.near, 420);
  camera.up.set(0, 0, 1); // MuJoCo is z-up
  camera.position.set(-3.2, -3.6, 2.4);
  camera.lookAt(field.cx, field.cy, 0.3);

  // ---- lighting -----------------------------------------------------------
  // MJCF headlight ambient 0.3 / diffuse 0.6 (scene.xml:8) is the reference level
  // for the OUTDOOR ({venue:false}) look: a hemisphere term plus a key/fill pair,
  // so the dogs get form and cast a shadow.
  //
  // Indoors it becomes proper three-point lighting, and the reference level goes
  // down with it: a hall is lit by its rig, not by a sky, so the ambient and
  // hemisphere terms drop to a bounce and the three directionals do the work.
  //   key   warm, high and behind the near touchline — the only shadow caster
  //   fill  cool, opposite and low, opening up the shadow side of both dogs
  //   rim   cool-white from behind the far touchline, drawing an edge on the dogs
  //         so they separate from the dark bowl instead of sinking into it
  const V = opt.venue;
  const ambient = new THREE.AmbientLight(0xffffff, V ? 0.14 : HEADLIGHT.ambient * 0.62);
  const hemi = new THREE.HemisphereLight(
    V ? 0x4a6d92 : 0xbcd8ee, V ? 0x10161f : 0x2a3440, V ? 0.62 : 0.70);
  const key = new THREE.DirectionalLight(V ? 0xfff1d8 : 0xfff4e2, V ? 2.55 : HEADLIGHT.diffuse * 3.1);
  key.position.set(-3.4, -4.6, V ? 7.4 : 6.6);
  key.target.position.set(field.cx, field.cy, 0);
  key.castShadow = true;
  const shadowHalf = Math.max(field.halfX, field.halfY) + 1.6;
  key.shadow.camera.left = -shadowHalf;
  key.shadow.camera.right = shadowHalf;
  key.shadow.camera.top = shadowHalf;
  key.shadow.camera.bottom = -shadowHalf;
  // Snug near/far, not 0.5..24. The only casters are the two dogs, and the
  // receivers that matter are 1.5 mm decor slabs stacked 1.5 mm apart; over a
  // 23.5 m depth range the map could not tell them apart and the endzone striped
  // with acne (visible in tests/shots/render_asym_chase_far.png before this).
  // Halving the range and raising the bias clears it without detaching the
  // dogs' shadows from their feet.
  key.shadow.camera.near = V ? 4.0 : 0.5;
  key.shadow.camera.far = V ? 16 : 24;
  key.shadow.bias = V ? -0.005 : -0.0008;
  key.shadow.normalBias = V ? 0.02 : 0.012;
  const fill = new THREE.DirectionalLight(V ? 0xa8c8f0 : 0xc8dcf0, V ? 0.62 : 0.55);
  fill.position.set(4.2, 3.6, V ? 2.6 : 3.0);
  fill.target.position.set(field.cx, field.cy, 0);
  scene.add(ambient, hemi, key, key.target, fill, fill.target);

  const rim = V ? new THREE.DirectionalLight(0xdbe9ff, 1.05) : null;
  if (rim) {
    rim.position.set(field.cx + 1.6, field.cy + 6.4, 2.3);
    rim.target.position.set(field.cx, field.cy, 0.22);
    scene.add(rim, rim.target);
  }

  // ---- backdrop -----------------------------------------------------------
  // Outdoors: a gradient sky dome over a finite checker disc (the MJCF terrain
  // plane is infinite; the disc stands in for it and fades into the haze).
  // Indoors: neither exists — app/venue.js puts a hall there instead, and the
  // scene background is the fog colour so every unfilled pixel is hall air.
  const sky = makeSky(hazeColor);
  sky.material.uniforms.uSun.value.copy(key.position).normalize();
  const groundTex = groundTexture();
  groundTex.anisotropy = renderer.capabilities.getMaxAnisotropy();
  const ground = new THREE.Mesh(
    new THREE.CircleGeometry(GROUND_RADIUS, 96),
    new THREE.MeshStandardMaterial({
      map: groundTex, roughness: 0.96, metalness: 0.0,
      polygonOffset: true, polygonOffsetFactor: 2, polygonOffsetUnits: 2,
    }),
  );
  ground.receiveShadow = true;
  ground.matrixAutoUpdate = false;

  let venue = null;
  if (V) {
    try {
      venue = buildVenue(scene, renderer, field, { quality: opt.quality });
      scene.fog.color.copy(venue.fogColor);
      scene.background = venue.fogColor.clone();
    } catch (e) {
      warnings.push(`venue build failed (${e && e.message}); falling back to the open-air scene`);
      venue = null;
    }
  }
  if (!venue) { scene.add(sky); scene.add(ground); }

  // ---- build the model scene ---------------------------------------------
  const geoms = sim.describeGeoms();
  const nbody = sim.nbody;

  const geomCache = new Map();   // primitive/mesh geometry, shared
  const matCache = new Map();    // world-geom materials, shared
  /** per-robot material tables, re-tinted by applyTeam() */
  const robotMats = { a: new Map(), b: new Map() };

  const dynamic = [];            // {mesh, bodyId, local} — robot subtrees, re-posed every frame
  const statics = [];            // {mesh, bodyId, local} — world-welded props, posed once
  const staticBodies = new Set();
  const collisionMeshes = [];
  const decorMeshes = [];
  const bodyUsed = new Uint8Array(nbody);   // bodies whose pose is read each frame

  const _q = new THREE.Quaternion();
  const _p = new THREE.Vector3();
  const _s = new THREE.Vector3(1, 1, 1);
  const _m = new THREE.Matrix4();

  let nMeshGeoms = 0, nPrimGeoms = 0, nSkipped = 0;

  for (let i = 0; i < geoms.length; i++) {
    const g = geoms[i];
    if (g.type === MJ_PLANE) continue;   // the terrain disc above stands in for it
    if (g.type === MJ_HFIELD) { nSkipped++; continue; }

    let geo;
    try {
      if (venue && g.name === 'decor_centre_circle') {
        // The MJCF draws the centre circle as a FILLED 0.45 m disc at 30 % alpha
        // (scene.xml decor_centre_circle, tools/export_scene.py:407); on a real
        // pitch it is a painted line, and as a filled patch it reads as a grey
        // smudge under whichever dog is standing on it. The geom is
        // contype=0 conaffinity=0 density=0 and is never read by physics.js,
        // obs.js or referee.js, so the renderer draws the line the disc was
        // standing in for. Same centre, same radius, same slab height.
        const ck = `ring|${g.size[0]}`;
        geo = geomCache.get(ck);
        if (!geo) {
          geo = new THREE.RingGeometry(g.size[0] - 0.045, g.size[0], 72);
          geomCache.set(ck, geo);
        }
        nPrimGeoms++;
      } else if (g.type === MJ_MESH) {
        const ck = `m${g.meshId}`;
        geo = geomCache.get(ck);
        if (!geo) { geo = meshGeometry(sim.getMesh(g.meshId)); geomCache.set(ck, geo); }
        nMeshGeoms++;
      } else {
        const ck = `${g.type}|${g.size[0]}|${g.size[1]}|${g.size[2]}`;
        geo = geomCache.get(ck);
        if (!geo) {
          geo = makePrimitive(g.type, g.size[0], g.size[1], g.size[2], opt.quality);
          geomCache.set(ck, geo);
        }
        nPrimGeoms++;
      }
    } catch (e) {
      warnings.push(`geom ${g.geomId} (${g.name || g.typeName}): ${e && e.message}`);
      nSkipped++;
      continue;
    }

    const rob = canonRobot(g.robot);
    let mat = rob ? robotMaterial(rob, g) : worldMaterial(g);
    // the ring above is a line, not a wash: it carries the slab's colour at the
    // alpha the other painted markings use (decor_touchline_*, 0.95), not the
    // 0.30 the filled disc needed to stay subtle.
    if (venue && g.name === 'decor_centre_circle') mat = markingMaterial(g);
    // When both ends score, a neutral green on both of them tells the player
    // nothing. Each end zone wears the colour of the dog that scores in it, so
    // "run at the cyan end" is readable from the chase camera.
    if (venue && bothScore && g.name && g.name.startsWith('decor_endzone')) {
      const mine = Math.sign(g.pos[0]) === (opt.playerRobot === 'b' ? -1 : 1);
      mat = mat.clone();
      mat.color.setHex(mine ? TEAM.player.ring : TEAM.ai.ring);
    }

    const mesh = new THREE.Mesh(geo, mat);
    mesh.name = g.name || `${g.typeName}_${g.geomId}`;
    mesh.matrixAutoUpdate = false;

    // body-local placement, baked once
    const local = new THREE.Matrix4().compose(
      _p.set(g.pos[0], g.pos[1], g.pos[2]),
      _q.set(g.quat[1], g.quat[2], g.quat[3], g.quat[0]), // MuJoCo is w-first
      _s,
    );

    if (rob) {
      mesh.castShadow = g.group !== 3;
      mesh.receiveShadow = false;
    } else {
      mesh.castShadow = false;
      mesh.receiveShadow = true;
    }
    if (g.group === 3) { mesh.visible = false; collisionMeshes.push(mesh); }
    if (g.name && g.name.startsWith('decor_')) decorMeshes.push(mesh);

    // Only the two robots move. The arena decoration and mjlab's field markers
    // live on their own massless world-welded bodies (NOT body 0), so "bodyId
    // === 0" is not the right static test — `robot == null` is. Their matrices
    // are baked on the first frame and then never recomputed; settleStatics()
    // re-checks them for a few frames and promotes anything that does move, so
    // the optimisation cannot silently freeze a future animated prop.
    if (rob) {
      bodyUsed[g.bodyId] = 1;
      dynamic.push({ mesh, bodyId: g.bodyId, local });
    } else {
      // painter order for the stacked 1.5 mm decor slabs
      if (mat.transparent) mesh.renderOrder = 1 + Math.round(g.pos[2] * 1000);
      if (g.bodyId === 0) {
        mesh.matrix.copy(local);
        mesh.matrixWorldNeedsUpdate = true;
      } else {
        statics.push({ mesh, bodyId: g.bodyId, local });
        staticBodies.add(g.bodyId);
      }
    }
    scene.add(mesh);
  }

  function worldMaterial(g) {
    const [r, gr, b, a] = g.rgba;
    const ck = `w|${r}|${gr}|${b}|${a}`;
    let m = matCache.get(ck);
    if (m) return m;
    const transparent = a < 0.999;
    m = new THREE.MeshStandardMaterial({
      color: mjColor(new THREE.Color(), r, gr, b),
      roughness: 0.82,
      metalness: 0.03,
      transparent,
      opacity: a,
      depthWrite: !transparent,
      side: THREE.DoubleSide,
    });
    matCache.set(ck, m);
    return m;
  }

  /**
   * A painted pitch marking: the geom's own rgb, at the alpha the touch lines
   * use (scene.json field.decor.palette.marking = 0.92 0.92 0.88 0.95). Only the
   * centre circle needs it, and only because the renderer draws that one as a
   * ring rather than as the disc the MJCF ships.
   */
  function markingMaterial(g) {
    const [r, gr, b] = g.rgba;
    const ck = `mark|${r}|${gr}|${b}`;
    let m = matCache.get(ck);
    if (m) return m;
    m = new THREE.MeshStandardMaterial({
      color: mjColor(new THREE.Color(), r, gr, b),
      roughness: 0.82, metalness: 0.03,
      transparent: true, opacity: 0.95, depthWrite: false,
      side: THREE.DoubleSide,
    });
    matCache.set(ck, m);
    return m;
  }

  /** Go2 part materials, keyed by the MJCF material name (metal/black/white/gray). */
  function robotMaterial(rob, g) {
    const base = (g.matName || '').replace(/^[ab]_/, '') || `rgba${g.rgba.join(',')}`;
    const ck = `${base}|${g.group}`;
    let m = robotMats[rob].get(ck);
    if (m) return m;
    const pbr = base === 'metal' ? { metalness: 0.75, roughness: 0.30 }
      : base === 'black' ? { metalness: 0.18, roughness: 0.52 }
        : base === 'white' ? { metalness: 0.06, roughness: 0.44 }
          : base === 'gray' ? { metalness: 0.38, roughness: 0.40 }
            : { metalness: 0.20, roughness: 0.60 };
    m = new THREE.MeshStandardMaterial({
      color: new THREE.Color(0x888888),
      ...pbr,
      transparent: g.group === 3,
      opacity: g.group === 3 ? 0.35 : 1,
      wireframe: g.group === 3,
    });
    // remember the source luminance so applyTeam() can place it on the ramp
    m.userData.lum = REC709(g.rgba[0], g.rgba[1], g.rgba[2]);
    m.userData.group = g.group;
    robotMats[rob].set(ck, m);
    return m;
  }

  // ---- team tinting -------------------------------------------------------
  const _cA = new THREE.Color(), _cB = new THREE.Color(), _cC = new THREE.Color();

  function applyTeam(rob, team) {
    _cA.setHex(team.dark, THREE.SRGBColorSpace);
    _cB.setHex(team.mid, THREE.SRGBColorSpace);
    _cC.setHex(team.light, THREE.SRGBColorSpace);
    for (const m of robotMats[rob].values()) {
      const L = m.userData.lum;
      if (m.userData.group === 3) { m.color.copy(_cC); continue; }
      if (L < 0.5) m.color.copy(_cA).lerp(_cB, L * 2);
      else m.color.copy(_cB).lerp(_cC, (L - 0.5) * 2);
      m.needsUpdate = false;
    }
  }

  let playerRobot = canonRobot(opt.playerRobot) || canonRobot(game && game.playerRobot)
    || canonRobot(game && game.playerSeat) || 'a';
  let aiRobot = playerRobot === 'a' ? 'b' : 'a';
  function setPlayerRobot(r) {
    const c = canonRobot(r);
    if (!c || c === playerRobot) return;
    playerRobot = c; aiRobot = c === 'a' ? 'b' : 'a';
    applyTeam(playerRobot, TEAM.player);
    applyTeam(aiRobot, TEAM.ai);
  }
  applyTeam(playerRobot, TEAM.player);
  applyTeam(aiRobot, TEAM.ai);

  // ---- player ground ring -------------------------------------------------
  // Sits at z = 0.02: above every decor slab (top 0.008) and above mjlab's
  // touchdown_region slab (top 0.016), so it can never z-fight the pitch.
  const RING_Z = 0.02;
  const ringGroup = new THREE.Group();
  ringGroup.matrixAutoUpdate = false;
  {
    const ringMat = new THREE.MeshBasicMaterial({
      color: new THREE.Color().setHex(TEAM.player.ring, THREE.SRGBColorSpace),
      transparent: true, opacity: 0.55, depthWrite: false, side: THREE.DoubleSide,
      toneMapped: false, fog: false,
    });
    const ring = new THREE.Mesh(new THREE.RingGeometry(0.34, 0.40, 56), ringMat);
    ring.renderOrder = 900;
    const halo = new THREE.Mesh(
      new THREE.RingGeometry(0.10, 0.34, 40),
      new THREE.MeshBasicMaterial({
        color: ringMat.color, transparent: true, opacity: 0.11, depthWrite: false,
        side: THREE.DoubleSide, toneMapped: false, fog: false,
      }),
    );
    halo.renderOrder = 899;
    // heading chevron, so the player can see which way the dog faces
    const chev = new THREE.BufferGeometry();
    chev.setAttribute('position', new THREE.Float32BufferAttribute(
      [0.62, 0, 0, 0.42, 0.13, 0, 0.42, 0.05, 0, 0.62, 0, 0, 0.42, -0.05, 0, 0.42, -0.13, 0], 3));
    const chevron = new THREE.Mesh(chev, new THREE.MeshBasicMaterial({
      color: ringMat.color, transparent: true, opacity: 0.7, depthWrite: false,
      side: THREE.DoubleSide, toneMapped: false, fog: false,
    }));
    chevron.renderOrder = 901;
    ringGroup.add(halo, ring, chevron);
    ringGroup.userData.mats = [ringMat, halo.material, chevron.material];
  }
  scene.add(ringGroup);

  // ---- contact shadows ----------------------------------------------------
  // One soft blob under each dog, so they are planted on the floor even at
  // quality 'low' (no shadow map at all) and under the belly, where a single
  // directional shadow never reaches. See app/venue.js:createContactShadows.
  const contact = venue ? createContactShadows(scene) : null;

  // ---- pose buffers / interpolation --------------------------------------
  const prevBuf = new Float32Array(nbody * 7);
  const nextBuf = new Float32Array(nbody * 7);
  const drawBuf = new Float32Array(nbody * 7);
  const bodyMats = new Array(nbody);
  for (let b = 0; b < nbody; b++) bodyMats[b] = new THREE.Matrix4();

  let haveState = false;
  let frameInterval = 0.02;         // EMA of the physics frame period, s
  let lastFrameT = 0;               // performance.now()/1000 when nextBuf arrived
  let lastUpdateT = 0;

  const baseBodyOf = {
    a: (sim.seats && sim.seats.a && sim.seats.a.baseBodyId) | 0,
    b: (sim.seats && sim.seats.b && sim.seats.b.baseBodyId) | 0,
  };
  if (!baseBodyOf.a || !baseBodyOf.b) {
    warnings.push('sim.seats.*.baseBodyId missing; camera focus falls back to body 1/2');
    baseBodyOf.a = baseBodyOf.a || 1;
    baseBodyOf.b = baseBodyOf.b || 2;
  }

  // focus state, smoothed; velocity is differentiated from the pose buffers so
  // the camera leads exactly what is on screen (no extra sim call, no alloc).
  const focusPos = new THREE.Vector3();
  const focusVel = new THREE.Vector3();
  const focusQuat = new THREE.Quaternion();
  let focusYaw = 0;

  // ---- camera state -------------------------------------------------------
  let mode = 'chase';
  const camPos = new THREE.Vector3().copy(camera.position);
  const camTgt = new THREE.Vector3(field.cx, field.cy, 0.3);
  /** Spring state for camPos / camTgt. Zeroed whenever the camera snaps. */
  const camVel = new THREE.Vector3();
  const camTgtVel = new THREE.Vector3();
  const camQuat = new THREE.Quaternion();
  let camInit = false;

  /**
   * `chase.yaw/elev/dist` is the PLAYER's boom: the mouse writes it, the
   * auto-align nudges its yaw, and nothing else may touch it. Every automatic
   * framing term (speed stretch, the sight-line guard) is computed on top of it
   * per frame and thrown away, so an orbit is never silently overwritten.
   * `yawSynced` is false until the boom has been dropped behind the dog's own
   * heading — without it the camera opens every match pointing down world +x,
   * which on the asym pitch parks it right on top of the far robot.
   */
  const chase = {
    yaw: 0, elev: CHASE.elev, dist: CHASE.dist,
    minDist: CHASE.minDist, maxDist: CHASE.maxDist,
  };
  let yawSynced = false;
  /**
   * Settings-panel multipliers, also accepted up front so a session can open
   * with the player's saved sliders instead of snapping to them on first drag.
   */
  let distScale = 1;
  let orbitSens = 1;
  if (Number.isFinite(opt.cameraDistance)) distScale = clamp(opt.cameraDistance, 0.4, 2.5);
  if (Number.isFinite(opt.mouseSensitivity)) orbitSens = clamp(opt.mouseSensitivity, 0.2, 4.0);
  const fpv = { yawOff: 0, pitchOff: 0 };
  const bcast = {
    az: BROADCAST.azimuthDeg * DEG, el: BROADCAST.elevationDeg * DEG, distScale: 1,
  };
  let lastDragT = -1e9;

  // scratch (allocated once; update() must not allocate)
  const _v3a = new THREE.Vector3(), _v3b = new THREE.Vector3();
  const _fwd = new THREE.Vector3(), _bx = new THREE.Vector3();
  const _by = new THREE.Vector3(), _bz = new THREE.Vector3();
  const _foe = new THREE.Vector3();
  const _qa = new THREE.Quaternion(), _qb = new THREE.Quaternion();
  const _up = new THREE.Vector3(0, 0, 1);

  // mutable view state (declared before update() runs)
  let lastFocus = null;
  let userSetPlayer = !!opt.playerRobot;
  let ringVisible = true;
  let frames = 0, frameMs = 0;
  /** body -> camera basis: cam looks down -z, we want -z = body +x and +y = body +z. */
  const FPV_BASIS = new THREE.Quaternion().setFromRotationMatrix(
    new THREE.Matrix4().makeBasis(
      new THREE.Vector3(0, -1, 0), new THREE.Vector3(0, 0, 1), new THREE.Vector3(-1, 0, 0)),
  );

  // ---- controls -----------------------------------------------------------
  let dragging = false, dragId = -1, lastX = 0, lastY = 0, controlsOn = false;

  function onPointerDown(e) {
    if (e.button !== 0 && e.button !== 2) return;
    dragging = true; dragId = e.pointerId; lastX = e.clientX; lastY = e.clientY;
    lastDragT = performance.now() / 1000;
    try { canvas.setPointerCapture(e.pointerId); } catch { /* not captured; fine */ }
  }
  function onPointerMove(e) {
    if (!dragging || e.pointerId !== dragId) return;
    const dx = e.clientX - lastX, dy = e.clientY - lastY;
    lastX = e.clientX; lastY = e.clientY;
    lastDragT = performance.now() / 1000;
    const s = ORBIT_RAD_PER_PX * orbitSens;
    if (mode === 'chase') {
      // Orbiting counts as taking the wheel: the boom stops auto-aligning for a
      // moment (lastDragT) but keeps the yaw the player just chose.
      yawSynced = true;
      chase.yaw -= dx * s;
      chase.elev = clamp(chase.elev + dy * s, CHASE.minElev, CHASE.maxElev);
    } else if (mode === 'fpv') {
      fpv.yawOff = clamp(fpv.yawOff - dx * s, -100 * DEG, 100 * DEG);
      fpv.pitchOff = clamp(fpv.pitchOff + dy * s, -55 * DEG, 45 * DEG);
    } else {
      bcast.az -= dx * s;
      bcast.el = clamp(bcast.el + dy * s, -85 * DEG, -6 * DEG);
    }
    e.preventDefault();
  }
  function onPointerUp(e) {
    if (e.pointerId !== dragId) return;
    dragging = false; dragId = -1;
    try { canvas.releasePointerCapture(e.pointerId); } catch { /* fine */ }
  }
  function onWheel(e) {
    const f = Math.exp(e.deltaY * (e.deltaMode === 1 ? 25 * ZOOM_PER_WHEEL_PX : ZOOM_PER_WHEEL_PX));
    if (mode === 'broadcast') bcast.distScale = clamp(bcast.distScale * f, 0.45, 2.2);
    else chase.dist = clamp(chase.dist * f, chase.minDist, chase.maxDist);
    e.preventDefault();
  }
  function onContextMenu(e) { if (dragging) e.preventDefault(); }

  function attachControls() {
    if (controlsOn) return;
    canvas.addEventListener('pointerdown', onPointerDown);
    canvas.addEventListener('pointermove', onPointerMove, { passive: false });
    canvas.addEventListener('pointerup', onPointerUp);
    canvas.addEventListener('pointercancel', onPointerUp);
    canvas.addEventListener('wheel', onWheel, { passive: false });
    canvas.addEventListener('contextmenu', onContextMenu);
    controlsOn = true;
  }
  function detachControls() {
    if (!controlsOn) return;
    canvas.removeEventListener('pointerdown', onPointerDown);
    canvas.removeEventListener('pointermove', onPointerMove);
    canvas.removeEventListener('pointerup', onPointerUp);
    canvas.removeEventListener('pointercancel', onPointerUp);
    canvas.removeEventListener('wheel', onWheel);
    canvas.removeEventListener('contextmenu', onContextMenu);
    controlsOn = false;
  }
  if (opt.controls) attachControls();

  // ---- sizing -------------------------------------------------------------
  let vw = 0, vh = 0;
  const _dbSize = new THREE.Vector2();
  function resize() {
    const w = Math.max(1, Math.round(canvas.clientWidth || canvas.width || 1));
    const h = Math.max(1, Math.round(canvas.clientHeight || canvas.height || 1));
    const dpr = Math.min(
      typeof devicePixelRatio === 'number' ? devicePixelRatio : 1,
      opt.quality === 'low' ? 1 : opt.quality === 'medium' ? 1.5 : opt.maxPixelRatio,
    );
    if (w === vw && h === vh && renderer.getPixelRatio() === dpr) return;
    vw = w; vh = h;
    renderer.setPixelRatio(dpr);
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    // the bloom chain works in DRAWING-BUFFER pixels, not CSS pixels
    if (bloom) { renderer.getDrawingBufferSize(_dbSize); bloom.setSize(_dbSize.x, _dbSize.y); }
  }

  // ---- bloom --------------------------------------------------------------
  // High only: the chain costs one full-resolution half-float target plus five
  // fullscreen passes, which is nothing on a GPU and everything on a software
  // rasteriser. Medium and low render straight to the canvas, through the same
  // ACES curve (app/venue.js:createBloom explains why the two paths agree).
  let bloom = null;
  function setBloom(on) {
    if (on && !bloom) {
      try {
        bloom = createBloom(renderer, scene, camera, { samples: opt.antialias ? 4 : 0 });
        renderer.getDrawingBufferSize(_dbSize);
        bloom.setSize(_dbSize.x, _dbSize.y);
      } catch (e) {
        warnings.push(`bloom unavailable (${e && e.message}); rendering direct`);
        bloom = null;
      }
    } else if (!on && bloom) {
      bloom.dispose();
      bloom = null;
      renderer.setRenderTarget(null);
    }
  }

  resize();

  function setQuality(q) {
    if (!['low', 'medium', 'high'].includes(q)) return;
    opt.quality = q;
    renderer.shadowMap.enabled = q !== 'low';
    key.castShadow = q !== 'low';
    key.shadow.mapSize.setScalar(q === 'high' ? 2048 : 1024);
    if (key.shadow.map) { key.shadow.map.dispose(); key.shadow.map = null; }
    if (venue) venue.setQuality(q);
    setBloom(!!venue && q === 'high');
    vw = 0; // force resize() to re-apply the pixel ratio
    resize();
  }
  setQuality(opt.quality);

  // ---- camera solvers -----------------------------------------------------

  /** MuJoCo free-camera direction: (cos el cos az, cos el sin az, sin el). */
  function mjForward(out, az, el) {
    const ce = Math.cos(el);
    return out.set(ce * Math.cos(az), ce * Math.sin(az), Math.sin(el));
  }

  /**
   * Distance at which the whole field box fits the current lens, solved exactly
   * (not via a bounding sphere, which over-pads an elongated pitch seen from 45 deg):
   * for every corner q (relative to the lookat), d >= |q.right|/tan(fovX/2) - q.fwd
   * and d >= |q.up|/tan(fovY/2) - q.fwd.
   */
  function fitDistance(fwd, lookZ, fovDeg, aspect) {
    const ty = Math.tan(fovDeg * DEG / 2);
    const tx = ty * Math.max(0.2, aspect);
    // camera basis from fwd and world up (dedicated scratch: fwd may alias a caller's vector)
    _bz.copy(fwd).negate();                       // camera z axis
    _bx.copy(_up).cross(_bz).normalize();         // camera x axis (right)
    _by.copy(_bz).cross(_bx).normalize();         // camera y axis (up)
    const m = 0.20; // margin, metres
    let d = 0;
    for (let i = 0; i < 8; i++) {
      const qx = (i & 1 ? field.halfX + m : -(field.halfX + m));
      const qy = (i & 2 ? field.halfY + m : -(field.halfY + m));
      const qz = (i & 4 ? 0.55 : 0.0) - lookZ;
      const f = qx * fwd.x + qy * fwd.y + qz * fwd.z;
      const r = qx * _bx.x + qy * _bx.y + qz * _bx.z;
      const u = qx * _by.x + qy * _by.y + qz * _by.z;
      d = Math.max(d, Math.abs(r) / tx - f, Math.abs(u) / ty - f);
    }
    return Math.max(d, 1.0);
  }

  /**
   * Where the chase boom WANTS to be this frame. The spring in stepCamera is
   * what actually gets there; nothing below is smoothed, so every term can be
   * read as a static rule.
   *
   * Order matters: yaw first (it defines the sight line), then the aim, then the
   * sight-line guard (which needs the aim), then the boom itself.
   */
  function solveChase(dt, desiredPos, desiredTgt) {
    const speed = Math.hypot(focusVel.x, focusVel.y);

    // -- 1. yaw. On the first frame the boom is dropped straight behind the dog;
    //    after that it is the player's, nudged back in line when they are moving
    //    and have left the mouse alone (GTA-style auto-align).
    if (!yawSynced && haveState) { chase.yaw = focusYaw; yawSynced = true; }
    const sinceDrag = performance.now() / 1000 - lastDragT;
    if (!dragging && sinceDrag > 1.1 && speed > 0.25) {
      const w = follow(clamp(2.4 * (speed - 0.2), 0, 3.4), dt);
      chase.yaw += angDelta(focusYaw, chase.yaw) * w;
    }
    const cy = Math.cos(chase.yaw), sy = Math.sin(chase.yaw);

    // -- 2. the other dog, in the same interpolated frame as the one we follow.
    const oo = 7 * (playerFocus() === 'a' ? baseBodyOf.b : baseBodyOf.a);
    _foe.set(drawBuf[oo], drawBuf[oo + 1], drawBuf[oo + 2]);
    const gx = _foe.x - focusPos.x, gy = _foe.y - focusPos.y;
    const sep = Math.hypot(gx, gy);

    // -- 3. aim. Lean off the player toward the opponent as the gap closes, so a
    //    close-quarters duel is centred instead of happening over the dog's
    //    shoulder. The lean is killed when the opponent is behind the lens
    //    (`face`): leaning at something behind the camera would flip the shot.
    let lean = CHASE.duelBias
      * smoothstep(clamp((CHASE.duelFar - sep) / (CHASE.duelFar - CHASE.duelNear), 0, 1));
    if (sep > 1e-3) {
      const face = clamp(((gx * cy + gy * sy) / sep) * 1.6 + 0.5, 0, 1);
      lean *= face;
      if (sep * lean > CHASE.duelAimMax) lean = CHASE.duelAimMax / sep;
    } else {
      lean = 0;
    }
    desiredTgt.set(
      focusPos.x + gx * lean + clamp(focusVel.x * CHASE.lead, -CHASE.leadMax, CHASE.leadMax),
      focusPos.y + gy * lean + clamp(focusVel.y * CHASE.lead, -CHASE.leadMax, CHASE.leadMax),
      focusPos.z + CHASE.aimZ,
    );

    // -- 4. boom length: the player's zoom, times the settings slider, eased out
    //    with speed so a sprint opens the shot instead of tailgating.
    let d = clamp(chase.dist * distScale, CHASE.minDist, CHASE.maxDist * 1.25);
    d *= 1 + Math.min(CHASE.speedStretch * speed, CHASE.speedStretchMax);

    // -- 5. sight-line guard. Project the opponent onto the camera->player line
    //    in plan view; if it is inside guardRadius of that line AND genuinely
    //    between the two (t away from both ends), lift the boom over it and back
    //    off, in proportion to how badly it is in the way.
    let elev = chase.elev;
    const reach = d * Math.cos(elev);
    if (reach > 0.2) {
      // opponent relative to the nominal lens, in plan view
      const wx = gx + reach * cy, wy = gy + reach * sy;
      const t = (wx * cy + wy * sy) / reach;
      const perp = Math.abs(wy * cy - wx * sy);
      if (t > 0 && t < 1 && perp < CHASE.guardRadius) {
        const taper = smoothstep(clamp(Math.min(t / 0.10, (1 - t) / 0.25), 0, 1));
        const bite = smoothstep(clamp(1 - perp / CHASE.guardRadius, 0, 1)) * taper;
        elev = clamp(elev + CHASE.guardLift * bite, CHASE.minElev, CHASE.maxElev);
        d += CHASE.guardPush * bite;
      }
    }

    // -- 6. the boom, pivoted above the dog's base so a low orbit still clears
    //    the floor; the target is floored outright so the spring never chases a
    //    point underground.
    const ce = Math.cos(elev), se = Math.sin(elev);
    desiredPos.set(
      focusPos.x - d * ce * cy,
      focusPos.y - d * ce * sy,
      focusPos.z + CHASE.pivotZ + d * se,
    );
    if (desiredPos.z < CHASE.floorZ) desiredPos.z = CHASE.floorZ;
  }

  function solveBroadcast(desiredPos, desiredTgt) {
    mjForward(_fwd, bcast.az, bcast.el);
    // fit against the mode's TARGET lens, not the mid-transition one
    const fit = fitDistance(_fwd, BROADCAST.lookatZ, MODE_CFG.broadcast.fov, camera.aspect);
    const d = fit * bcast.distScale;
    desiredTgt.set(field.cx, field.cy, BROADCAST.lookatZ);
    desiredPos.copy(desiredTgt).addScaledVector(_fwd, -d);
  }

  // ---- update -------------------------------------------------------------

  /**
   * Draw one frame.
   * @param {Float32Array} [renderState] sim.renderState() — the reused nbody*7
   *        buffer is fine, its contents are copied. Omit to redraw the last state
   *        (paused game: the camera keeps easing, the dogs hold still).
   * @param {'a'|'b'|'attacker'|'defender'} [focusRobot] who the camera follows.
   *        The first robot seen here is treated as the player and gets the teal
   *        tint + the ground ring unless setPlayerRobot() said otherwise.
   */
  function update(renderState, focusRobot) {
    const now = performance.now() / 1000;
    let dt = lastUpdateT ? now - lastUpdateT : 1 / 60;
    lastUpdateT = now;
    if (!(dt > 0)) dt = 1 / 60;
    if (dt > 0.25) dt = 0.25;             // tab was backgrounded

    // -- ingest a new physics frame (detected by content, so the caller can keep
    //    handing us the same reused buffer) ---------------------------------
    if (renderState && renderState.length >= nbody * 7) {
      if (!haveState) {
        prevBuf.set(renderState); nextBuf.set(renderState);
        haveState = true; lastFrameT = now;
      } else if (!sameDynamic(nextBuf, renderState)) {
        prevBuf.set(nextBuf); nextBuf.set(renderState);
        const gap = now - lastFrameT;
        if (gap > 1e-4 && gap < 0.25) frameInterval += (gap - frameInterval) * 0.2;
        lastFrameT = now;
      }
    }

    const fr = canonRobot(focusRobot);
    if (fr && fr !== playerRobot && !opt.playerRobot && !userSetPlayer) setPlayerRobot(fr);
    const focus = fr || playerRobot;

    if (haveState) {
      const alpha = opt.interpolate
        ? clamp((now - lastFrameT) / Math.max(frameInterval, 1e-3), 0, 1)
        : 1;
      writeDrawBuf(alpha);
      if (staticCheck >= 0) settleStatics();
      poseBodies();
      updateFocus(focus, dt);
      updateRing(focus);
      if (contact) updateContact();
    }

    stepCamera(dt);

    if (!venue) {
      sky.position.copy(camera.position);
      sky.matrix.makeTranslation(camera.position.x, camera.position.y, camera.position.z);
      sky.matrixWorldNeedsUpdate = true;
    }

    if (bloom) bloom.render();
    else renderer.render(scene, camera);
    frames++;
    frameMs += (performance.now() / 1000 - now) * 1000;
  }

  /** True when every dynamic body's 7 floats are unchanged. */
  function sameDynamic(a, b) {
    for (let i = 0; i < dynamic.length; i++) {
      const o = 7 * dynamic[i].bodyId;
      if (a[o] !== b[o] || a[o + 1] !== b[o + 1] || a[o + 2] !== b[o + 2]
        || a[o + 3] !== b[o + 3] || a[o + 4] !== b[o + 4]
        || a[o + 5] !== b[o + 5] || a[o + 6] !== b[o + 6]) return false;
    }
    return true;
  }

  /**
   * Pose the world-welded props once, then spend a few frames proving they
   * really are static. Anything that moves is promoted to the per-frame list
   * and reported on `warnings`, so the fast path can never silently freeze a
   * prop a future scene decides to animate.
   */
  let staticCheck = 0;                       // -1 once settled
  const staticRef = new Float32Array(nbody * 7);
  function settleStatics() {
    if (staticCheck === 0) {
      staticRef.set(nextBuf);
      for (let i = 0; i < statics.length; i++) {
        const st = statics[i];
        const o = 7 * st.bodyId;
        _p.set(nextBuf[o], nextBuf[o + 1], nextBuf[o + 2]);
        _q.set(nextBuf[o + 4], nextBuf[o + 5], nextBuf[o + 6], nextBuf[o + 3]);
        bodyMats[st.bodyId].compose(_p, _q, _s);
        st.mesh.matrix.multiplyMatrices(bodyMats[st.bodyId], st.local);
        st.mesh.matrixWorldNeedsUpdate = true;
      }
      staticCheck = 1;
      return;
    }
    let moved = false;
    for (const b of staticBodies) {
      const o = 7 * b;
      for (let k = 0; k < 7; k++) {
        if (Math.abs(nextBuf[o + k] - staticRef[o + k]) > 1e-6) { moved = true; break; }
      }
      if (moved) break;
    }
    if (moved) {
      for (const st of statics) { bodyUsed[st.bodyId] = 1; dynamic.push(st); }
      warnings.push(`${statics.length} world-welded geoms moved; promoted to the per-frame path`); // alloc-ok: fires at most once, on the frame a world-welded prop is first seen to move
      statics.length = 0; staticBodies.clear();
      staticCheck = -1;
      return;
    }
    if (++staticCheck > 5) staticCheck = -1;
  }

  function writeDrawBuf(alpha) {
    if (alpha >= 1) { drawBuf.set(nextBuf); return; }
    for (let b = 0; b < nbody; b++) {
      if (!bodyUsed[b] && b !== baseBodyOf.a && b !== baseBodyOf.b) continue;
      const o = 7 * b;
      drawBuf[o] = prevBuf[o] + (nextBuf[o] - prevBuf[o]) * alpha;
      drawBuf[o + 1] = prevBuf[o + 1] + (nextBuf[o + 1] - prevBuf[o + 1]) * alpha;
      drawBuf[o + 2] = prevBuf[o + 2] + (nextBuf[o + 2] - prevBuf[o + 2]) * alpha;
      _qa.set(prevBuf[o + 4], prevBuf[o + 5], prevBuf[o + 6], prevBuf[o + 3]);
      _qb.set(nextBuf[o + 4], nextBuf[o + 5], nextBuf[o + 6], nextBuf[o + 3]);
      _qa.slerp(_qb, alpha);
      drawBuf[o + 3] = _qa.w; drawBuf[o + 4] = _qa.x;
      drawBuf[o + 5] = _qa.y; drawBuf[o + 6] = _qa.z;
    }
  }

  function poseBodies() {
    for (let b = 0; b < nbody; b++) {
      if (!bodyUsed[b]) continue;
      const o = 7 * b;
      _p.set(drawBuf[o], drawBuf[o + 1], drawBuf[o + 2]);
      _q.set(drawBuf[o + 4], drawBuf[o + 5], drawBuf[o + 6], drawBuf[o + 3]);
      bodyMats[b].compose(_p, _q, _s);
    }
    for (let i = 0; i < dynamic.length; i++) {
      const d = dynamic[i];
      d.mesh.matrix.multiplyMatrices(bodyMats[d.bodyId], d.local);
      d.mesh.matrixWorldNeedsUpdate = true;
    }
  }

  function updateFocus(rob, dt) {
    const o = 7 * baseBodyOf[rob];
    _v3a.set(drawBuf[o], drawBuf[o + 1], drawBuf[o + 2]);
    focusQuat.set(drawBuf[o + 4], drawBuf[o + 5], drawBuf[o + 6], drawBuf[o + 3]);
    // velocity straight off the two physics frames -> exactly what is on screen
    const iv = 1 / Math.max(frameInterval, 1e-3);
    _v3b.set(
      (nextBuf[o] - prevBuf[o]) * iv,
      (nextBuf[o + 1] - prevBuf[o + 1]) * iv,
      (nextBuf[o + 2] - prevBuf[o + 2]) * iv,
    );
    if (!camInit) { focusPos.copy(_v3a); focusVel.copy(_v3b); } else {
      focusPos.lerp(_v3a, follow(30, dt));
      focusVel.lerp(_v3b, follow(6, dt));
    }
    // yaw only — never inherit the base's pitch/roll or the shot shakes every footfall
    const qw = focusQuat.w, qx = focusQuat.x, qy = focusQuat.y, qz = focusQuat.z;
    focusYaw = Math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz));
  }

  function updateRing(rob) {
    const o = 7 * baseBodyOf[rob];
    const show = ringVisible && mode !== 'fpv';   // you cannot see your own marker
    ringGroup.visible = show;
    if (!show) return;
    _qa.setFromAxisAngle(_up, focusYaw);
    ringGroup.matrix.compose(_v3a.set(drawBuf[o], drawBuf[o + 1], RING_Z), _qa, _s);
    ringGroup.matrixWorldNeedsUpdate = true;
  }

  /** Slide both contact blobs under their dogs. Two matrix writes, no alloc. */
  function updateContact() {
    const a = 7 * baseBodyOf.a, b = 7 * baseBodyOf.b;
    contact.place('a', drawBuf[a], drawBuf[a + 1], drawBuf[a + 2]);
    contact.place('b', drawBuf[b], drawBuf[b + 1], drawBuf[b + 2]);
  }

  /**
   * Move the real camera one frame toward whatever the mode's solver asked for.
   *
   * chase/broadcast run a critically damped spring (critDamp3) on the eye and
   * on the look-at separately — the look-at is the stiffer of the two, so the
   * shot keeps pointing at the action while the boom still has some weight.
   * FPV keeps its first-order follow: a spring would make the eye trail the
   * skull by 2 v / w metres at speed, which on a head cam reads as the camera
   * sinking into the dog.
   */
  function stepCamera(dt) {
    const cfg = MODE_CFG[mode];
    if (mode === 'fpv') {
      // eye in the head, orientation = yaw + damped pitch/roll + mouse look
      const o = 7 * baseBodyOf[playerFocus()];
      _qa.set(drawBuf[o + 4], drawBuf[o + 5], drawBuf[o + 6], drawBuf[o + 3]);
      _qb.setFromAxisAngle(_up, focusYaw);
      _qb.slerp(_qa, 0.5);                       // half the body's pitch/roll
      _v3a.set(FPV_EYE.x, FPV_EYE.y, FPV_EYE.z).applyQuaternion(_qa);
      _v3a.set(drawBuf[o] + _v3a.x, drawBuf[o + 1] + _v3a.y, drawBuf[o + 2] + _v3a.z);
      if (!camInit) { camPos.copy(_v3a); camVel.set(0, 0, 0); } else {
        camPos.lerp(_v3a, follow(cfg.kPos, dt));
      }
      camPos.z = Math.max(camPos.z, 0.06);
      _qb.multiply(FPV_BASIS);
      _qa.setFromAxisAngle(_up, fpv.yawOff);     // mouse look, applied in world yaw
      _qb.premultiply(_qa);
      _qa.setFromAxisAngle(_v3b.set(1, 0, 0), fpv.pitchOff);
      _qb.multiply(_qa);
      if (!camInit) camQuat.copy(_qb); else camQuat.slerp(_qb, follow(cfg.kQuat, dt));
      camera.position.copy(camPos);
      camera.quaternion.copy(camQuat);
    } else {
      if (mode === 'chase') solveChase(dt, _v3a, _v3b);
      else solveBroadcast(_v3a, _v3b);
      if (!camInit) {
        camPos.copy(_v3a); camTgt.copy(_v3b);
        camVel.set(0, 0, 0); camTgtVel.set(0, 0, 0);
      } else {
        critDamp3(camPos, camVel, _v3a, cfg.wPos, dt);
        critDamp3(camTgt, camTgtVel, _v3b, cfg.wTgt, dt);
      }
      // Backstop: the solver already floors its target, so this only ever fires
      // on a snap from a stale position. Kill the downward velocity with it, or
      // the spring keeps pushing into the clamp and the shot sticks.
      if (camPos.z < CHASE.hardFloorZ) {
        camPos.z = CHASE.hardFloorZ;
        if (camVel.z < 0) camVel.z = 0;
      }
      camera.position.copy(camPos);
      camera.up.copy(_up);
      camera.lookAt(camTgt);
    }
    // lens easing on a mode switch
    const wantFov = cfg.fov;
    if (Math.abs(camera.fov - wantFov) > 0.01 || Math.abs(camera.near - cfg.near) > 1e-6) {
      camera.fov += (wantFov - camera.fov) * (camInit ? follow(6, dt) : 1);
      camera.near = cfg.near;
      camera.updateProjectionMatrix();
    }
    // Only arm the easing once there is a pose to ease FROM. Before the first
    // physics frame the focus is still the origin, and letting the spring start
    // there would fly the shot in from the middle of the pitch.
    if (haveState) camInit = true;
  }

  function playerFocus() { return lastFocus || playerRobot; }

  // ---- public surface -----------------------------------------------------
  const api = {
    // --- frozen interface ---
    update(renderState, focusRobot) {
      const c = canonRobot(focusRobot);
      if (c) lastFocus = c;
      update(renderState, c);
    },
    /** 'chase' | 'fpv' | 'broadcast'. snap=true skips the fly-over. */
    setCamera(m, snap = false) {
      if (!CAMERA_MODES.includes(m)) throw new Error(`setCamera: unknown mode "${m}"`);
      if (m === mode) { if (snap) { camInit = false; if (m === 'chase') yawSynced = false; } return; }
      mode = m;
      if (m === 'broadcast') {
        bcast.az = BROADCAST.azimuthDeg * DEG;
        bcast.el = BROADCAST.elevationDeg * DEG;
        bcast.distScale = 1;
      } else if (m === 'fpv') { fpv.yawOff = 0; fpv.pitchOff = 0; }
      // Coming back to chase, drop the boom behind the dog again rather than
      // resuming whatever yaw it held three camera modes ago.
      if (m === 'chase') yawSynced = false;
      if (snap) camInit = false;
    },
    resize,

    // --- extras the shell may use ---
    get camera() { return camera; },
    get scene() { return scene; },
    get renderer() { return renderer; },
    get three() { return THREE; },
    get cameraMode() { return mode; },
    get playerRobot() { return playerRobot; },
    warnings,
    /** Team colours, so the HUD can match the dogs. */
    teamColors: {
      player: '#' + TEAM.player.mid.toString(16).padStart(6, '0'),
      ai: '#' + TEAM.ai.mid.toString(16).padStart(6, '0'),
      playerRing: '#' + TEAM.player.ring.toString(16).padStart(6, '0'),
      aiAccent: '#' + TEAM.ai.ring.toString(16).padStart(6, '0'),
    },
    setPlayerRobot(r) { userSetPlayer = true; setPlayerRobot(r); },
    setQuality,
    setInterpolation(on) { opt.interpolate = !!on; },
    /** Debug: show the 23 collision primitives per dog as wireframe. */
    setShowCollision(on) { for (const m of collisionMeshes) m.visible = !!on; },
    setShowDecor(on) { for (const m of decorMeshes) m.visible = !!on; },
    setPlayerRing(on) { ringVisible = !!on; ringGroup.visible = !!on; },
    attachControls,
    detachControls,
    /**
     * Settings panel, "Camera distance" (ui.js ships it at 0.6..1.8x). Multiplies
     * the chase boom on top of whatever the wheel has zoomed to; the product is
     * still clamped to the boom's own range, so the slider can never put the lens
     * inside the dog or out past the arena.
     */
    setCameraDistance(v) {
      const n = Number(v);
      distScale = clamp(Number.isFinite(n) && n > 0 ? n : 1, 0.4, 2.5);
    },
    getCameraDistance: () => distScale,
    /**
     * Settings panel hook for the mouse. Separate from input.setSensitivity(),
     * which scales the velocity command, not the orbit — a player who wants a
     * twitchy camera does not necessarily want a twitchy robot.
     */
    setMouseSensitivity(v) {
      const n = Number(v);
      orbitSens = clamp(Number.isFinite(n) && n > 0 ? n : 1, 0.2, 4.0);
    },
    getMouseSensitivity: () => orbitSens,
    /** Drop the camera on its target immediately (no easing) on the next frame. */
    snapCamera() { camInit = false; yawSynced = false; },
    /** Mean ms spent inside update() since the last call, and the frame count. */
    stats() {
      const s = { frames, meanMs: frames ? frameMs / frames : 0 };
      frames = 0; frameMs = 0;
      return s;
    },
    /** Hide the whole venue without rebuilding it — for an A/B screenshot. */
    setVenueVisible(on) { if (venue) venue.group.visible = !!on; },
    info: {
      three: THREE.REVISION,
      field,
      geoms: geoms.length,
      meshGeoms: nMeshGeoms,
      primitiveGeoms: nPrimGeoms,
      skipped: nSkipped,
      get dynamicGeoms() { return dynamic.length; },
      get staticGeoms() { return statics.length; },
      worldGeoms: geoms.length - nSkipped,
      uniqueGeometries: geomCache.size,
      venue: venue ? venue.info : null,
      get bloom() { return !!bloom; },
      toneMapping: 'ACESFilmic',
      drawCalls: () => renderer.info.render.calls,
      triangles: () => renderer.info.render.triangles,
    },
    dispose() {
      detachControls();
      if (bloom) { bloom.dispose(); bloom = null; }
      if (contact) contact.dispose();
      if (venue) venue.dispose();
      for (const g of geomCache.values()) g.dispose();
      for (const m of matCache.values()) m.dispose();
      for (const t of [robotMats.a, robotMats.b]) for (const m of t.values()) m.dispose();
      for (const m of ringGroup.userData.mats) m.dispose();
      ground.geometry.dispose(); ground.material.dispose(); groundTex.dispose();
      sky.geometry.dispose(); sky.material.dispose();
      renderer.dispose();
    },
  };
  return api;
}

// ---------------------------------------------------------------------------
// field resolution
// ---------------------------------------------------------------------------

/**
 * Field rectangle for the broadcast fit and the shadow camera. Preference order:
 *   1. the app/config.js GAMES entry handed in as `game`
 *   2. sim.sceneJson.field (written by tools/export_scene.py off the live scenario)
 *   3. the bounding box of the arena decor / mjlab marker geoms
 * Never a literal: a wrong rectangle silently mis-frames both games.
 */
function resolveField(game, sim, warnings) {
  const sj = sim.sceneJson || null;
  const cfg = (game && typeof game === 'object') ? game : null;
  const id = typeof game === 'string' ? game : (cfg && (cfg.id || cfg.game)) || (sj && sj.game) || null;

  let size = null, center = null, lineX = null, lineXB = null, source = null;
  if (cfg && Array.isArray(cfg.field) && cfg.field.length === 2) {
    size = cfg.field.slice(0, 2);
    center = Array.isArray(cfg.center) ? cfg.center.slice(0, 2) : [0, 0];
    lineX = typeof cfg.lineX === 'number' ? cfg.lineX : null;
    source = 'game config';
  } else if (sj && sj.field && Array.isArray(sj.field.size)) {
    size = sj.field.size.slice(0, 2);
    center = Array.isArray(sj.field.center) ? sj.field.center.slice(0, 2) : [0, 0];
    lineX = typeof sj.field.lineX === 'number' ? sj.field.lineX : null;
    lineXB = typeof sj.field.lineXB === 'number' ? sj.field.lineXB : null;
    source = 'scene.json';
  } else {
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity, n = 0;
    for (const g of sim.describeGeoms()) {
      if (g.bodyId !== 0 || g.type === 0) continue;
      x0 = Math.min(x0, g.pos[0] - g.size[0]); x1 = Math.max(x1, g.pos[0] + g.size[0]);
      y0 = Math.min(y0, g.pos[1] - g.size[1]); y1 = Math.max(y1, g.pos[1] + g.size[1]);
      n++;
    }
    if (n) {
      size = [x1 - x0, y1 - y0]; center = [(x0 + x1) / 2, (y0 + y1) / 2];
      source = 'world-geom bbox';
    } else {
      size = [5.6, 3.0]; center = [0, 0]; source = 'fallback';
    }
    warnings.push(`field geometry taken from the ${source}; pass GAMES.<id> or ship scene.json`);
  }
  // scene.json's field.size is the PLAY rectangle; the out-of-bounds box is what
  // the camera must cover, and it is wider on the asym pitch (touchdown_margin).
  let halfX = size[0] / 2, halfY = size[1] / 2;
  if (sj && sj.field && Array.isArray(sj.field.oobX) && Array.isArray(sj.field.oobY)) {
    halfX = Math.max(halfX, (sj.field.oobX[1] - sj.field.oobX[0]) / 2);
    halfY = Math.max(halfY, (sj.field.oobY[1] - sj.field.oobY[0]) / 2);
    center = [(sj.field.oobX[0] + sj.field.oobX[1]) / 2, (sj.field.oobY[0] + sj.field.oobY[1]) / 2];
  }
  return {
    id, source,
    size, center,
    cx: center[0], cy: center[1],
    halfX, halfY,
    lineX, lineXB,
  };
}

export default { createRenderer, CAMERA_MODES };

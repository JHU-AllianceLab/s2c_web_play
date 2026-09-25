/**
 * app/venue.js — the arena AROUND the physics.
 *
 * DESIGN.md section 12 is the brief: no MuJoCo default look. No grey checkerboard
 * infinite plane, no grey void, no default headlight. The reference
 * (https://ai-coaching-drone-racing.github.io/demo/) keeps physics and visuals in
 * two separate worlds and ships a *venue* for the visual one. This module is that
 * venue: an indoor hall with a matte floor, a stepped spectator bowl, a crowd,
 * barrier branding, ceiling fixtures, contact shadows, and the bloom that makes
 * the fixtures glow.
 *
 * ---------------------------------------------------------------------------
 * ★ NOTHING HERE TOUCHES THE PHYSICS.
 *
 * Not one geom, constant or file MuJoCo reads is modified by this module:
 *   - assets/scene/asym/scene.xml is NOT edited (md5 b68c9ea498a4156421f9437b7b53018a)
 *   - tools/export_scene.py is NOT re-run
 *   - nothing built here is ever seen by app/physics.js, app/obs.js or app/referee.js
 * It only adds three.js meshes to the scene graph app/render.js already owns. The
 * arena decoration that *is* in the MJCF (`decor_*`, every one of them
 * contype=0 conaffinity=0 density=0, appended by tools/export_scene.py:add_arena)
 * stays exactly as exported; this module reads its published dimensions so the
 * venue sits flush against it.
 * ---------------------------------------------------------------------------
 *
 * WHERE THE NUMBERS COME FROM
 *
 * Anything that has to line up with something real is quoted from the shipped
 * scene, not guessed. From `assets/scene/asym/scene.json` -> field.decor.geoms
 * (written by tools/export_scene.py:add_arena lines 378-423, and identical to the
 * geom table in assets/scene/asym/scene.xml):
 *
 *   decor_apron          pos (0.2, 0, 0.00125)   half (3.30, 2.20, 0.00075)
 *   decor_barrier_y_min  pos (0.2, -1.82, 0.11)  half (2.92, 0.05, 0.11)
 *   decor_barrier_y_max  pos (0.2,  1.82, 0.11)  half (2.92, 0.05, 0.11)
 *   decor_barrier_x_min  pos (-2.72, 0, 0.11)    half (0.05, 1.82, 0.11)
 *   decor_barrier_x_max  pos ( 3.12, 0, 0.11)    half (0.05, 1.82, 0.11)
 *   decor_barrier_*_top  a 0.012 half slab on top -> the ring's top is z = 0.244
 *   decor_line           pos z 0.00725  half z 0.00075 -> topmost flat slab at 0.008
 *
 * so the barrier ring encloses x in [-2.77, 3.17], y in [-1.87, 1.87] and its four
 * INNER faces are at x = -2.67 / +3.07 and y = -1.77 / +1.77. Those faces are where
 * the banners go: flush, 1 mm proud, never intersecting anything.
 *
 * The hall itself — bowl radii, tier rise, ceiling height, palette — has nothing to
 * be faithful to. It is a presentation choice, declared once in LAYOUT / PALETTE.
 *
 * BYTES ON THE WIRE: zero. Every texture here is painted into a <canvas> at boot
 * (concrete mottle, wall panelling, the wordmark, the shadow blob). Nothing is
 * fetched, so the venue costs 0 B of the 12 MB first-load budget.
 *
 * PER-FRAME COST: zero, except the two contact-shadow quads (two matrix writes)
 * and, on `high`, the bloom composite. Every other mesh is built once with
 * matrixAutoUpdate = false; the crowd is two InstancedMeshes, so ~500 people cost
 * two draw calls and are never re-posed, re-tinted or re-uploaded.
 */

import * as THREE from '../vendor/three/three.module.js';

// ---------------------------------------------------------------------------
// the shipped arena, quoted (see the header for the source of every line)
// ---------------------------------------------------------------------------

/** assets/scene/asym/scene.json -> field.decor.geoms. Metres. */
const DECOR = {
  /** the two x-barrier centre planes; also the y-barriers' x extent. */
  barrierX: [-2.72, 3.12],
  /** the two y-barrier centre planes; also the x-barriers' y extent. */
  barrierY: [-1.82, 1.82],
  /** every barrier box is 0.05 half-thick. */
  barrierThickHalf: 0.05,
  /** decor_barrier_* half-height 0.11 -> its painted face spans z in [0, 0.22]. */
  barrierFaceTopZ: 0.22,
  /** the topmost flat decor slab (decor_line) has its upper face here. */
  topSlabZ: 0.008,
};

// ---------------------------------------------------------------------------
// presentation: the hall
// ---------------------------------------------------------------------------

const LAYOUT = {
  /** hall interior, half-extents about the field centre, plus the ceiling height. */
  hallHalf: [11.2, 9.6],
  ceilingZ: 8.5,
  /**
   * The spectator bowl: front-row offset, step depth, step rise, step count.
   * The front row stands 5.8 m from the centre in x and 4.3 m in y — i.e. 2.6 m
   * and 2.4 m outside the barrier ring. Closer than that and a 1.7 m human leans
   * over a 0.35 m robot: the crowd stops being a backdrop and becomes the subject.
   */
  bowlInner: [5.8, 4.3],
  tierDepth: 1.00,
  tierRise: 0.45,
  /** tier 0 stands on a solid front wall this tall, so the pitch reads sunken. */
  tierFirstZ: 1.05,
  tiers: 4,
  /** seat rows per tier and the gap between people along a row. */
  rowsPerTier: 2,
  seatPitch: 0.64,
  /** the ceiling rig: rows across y, bars along x, and its height. */
  rigRows: 3,
  rigBars: 4,
  rigZ: 7.9,
};

const PALETTE = {
  /** the hall air: fog, background, and what every far surface fades into. */
  haze: 0x0a1018,
  floor: 0x161d27,
  wall: 0x0c1119,
  wallTop: 0x18202d,
  wallRib: 0x070b11,
  ceiling: 0x05080d,
  tierA: 0x151d29,
  tierB: 0x101722,
  tierNose: 0x25313f,
  rig: 0x1a222e,
  /** JHU blue #002D72 and its on-dark lift #5AA0FF (styles.css:11-13). */
  brand: 0x002d72,
  brandLift: 0x5aa0ff,
  bannerInk: 0xe8eef7,
};

/**
 * Clothing and skin ramps for the crowd. Both are deliberately dark: a stand
 * full of bright faces pulls the eye straight off the pitch, and the crowd's job
 * here is only to say "someone is watching".
 */
const CROWD_WEAR = [
  0x141b26, 0x1a2029, 0x0f141c, 0x221d1a, 0x172227,
  0x261b16, 0x1b2330, 0x13171e, 0x1f2531, 0x11191f,
  0x2a221c, 0x18505f, 0x2c231b, 0x1e2634, 0x0d1218,
];
const CROWD_SKIN = [0x5a4434, 0x6f5743, 0x453024, 0x7d6249, 0x342419, 0x604c3a];

/** Tile sizes, in metres of surface per texture repeat. */
const FLOOR_TILE = 3.0;
const WALL_TILE = 3.2;
const BANNER_TILE = 1.95;

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

/** Deterministic LCG — the crowd must be identical in every screenshot. */
function lcg(seed) {
  let s = seed >>> 0;
  return () => ((s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 4294967296);
}

const srgb = (hex) => new THREE.Color().setHex(hex, THREE.SRGBColorSpace);
const hex6 = (n) => '#' + n.toString(16).padStart(6, '0');

/**
 * A 2D canvas, or null when there is none (a node harness importing this file).
 * Every texture builder falls back to a flat DataTexture in that case, so the
 * module imports cleanly outside a browser.
 */
function makeCanvas(w, h) {
  if (typeof document !== 'undefined' && typeof document.createElement === 'function') {
    const c = document.createElement('canvas');
    c.width = w; c.height = h;
    return c;
  }
  if (typeof OffscreenCanvas !== 'undefined') return new OffscreenCanvas(w, h);
  return null;
}

function flatTexture(hex) {
  const c = srgb(hex);
  const px = new Uint8Array([
    Math.round(c.r * 255), Math.round(c.g * 255), Math.round(c.b * 255), 255]);
  const t = new THREE.DataTexture(px, 1, 1, THREE.RGBAFormat);
  t.colorSpace = THREE.SRGBColorSpace;
  t.needsUpdate = true;
  return t;
}

/** One CanvasTexture per (canvas, repeat) pair; the canvas itself is shared. */
function texFrom(canvas, fallbackHex, repeatX, repeatY, aniso) {
  if (!canvas) return flatTexture(fallbackHex);
  const t = new THREE.CanvasTexture(canvas);
  t.colorSpace = THREE.SRGBColorSpace;
  t.wrapS = THREE.RepeatWrapping;
  t.wrapT = repeatY === 1 ? THREE.ClampToEdgeWrapping : THREE.RepeatWrapping;
  t.repeat.set(repeatX, repeatY);
  t.anisotropy = aniso;
  t.needsUpdate = true;
  return t;
}

/**
 * Stand a PlaneGeometry (which lies in XY facing +z) up on its edge and turn it
 * to face `yaw`. Composed as qZ(yaw) * qX(90deg), so the normal starts at -Y and
 * is then swung round; the texture never mirrors, because the plane's local +x
 * lands on the viewer's right for every yaw.
 */
const _AX = new THREE.Vector3(1, 0, 0);
const _AZ = new THREE.Vector3(0, 0, 1);
function standUp(mesh, x, y, z, yaw) {
  const qx = new THREE.Quaternion().setFromAxisAngle(_AX, Math.PI / 2);
  const qz = new THREE.Quaternion().setFromAxisAngle(_AZ, yaw);
  mesh.quaternion.copy(qz).multiply(qx);
  mesh.position.set(x, y, z);
  mesh.matrixAutoUpdate = false;
  mesh.updateMatrix();
  return mesh;
}

// ---------------------------------------------------------------------------
// procedural texture canvases (painted at boot; 0 bytes on the wire)
// ---------------------------------------------------------------------------

/**
 * Matte poured concrete: a low-frequency mottle, a few trowel sweeps and a fine
 * aggregate speckle. At chase height the eye reads texture and never a repeat.
 */
function floorCanvas(size = 512) {
  const cv = makeCanvas(size, size);
  if (!cv) return null;
  const g = cv.getContext('2d');
  g.fillStyle = hex6(PALETTE.floor);
  g.fillRect(0, 0, size, size);

  const rnd = lcg(0x00C0FFEE);
  g.globalCompositeOperation = 'lighter';
  for (let i = 0; i < 90; i++) {
    const x = rnd() * size, y = rnd() * size, r = 24 + rnd() * 120;
    const grd = g.createRadialGradient(x, y, 0, x, y, r);
    const a = 0.026 + rnd() * 0.055;
    grd.addColorStop(0, 'rgba(140,170,205,' + a.toFixed(4) + ')');
    grd.addColorStop(1, 'rgba(140,170,205,0)');
    g.fillStyle = grd;
    g.fillRect(x - r, y - r, 2 * r, 2 * r);
  }
  for (let i = 0; i < 5; i++) {
    g.strokeStyle = 'rgba(150,180,215,' + (0.010 + rnd() * 0.013).toFixed(4) + ')';
    g.lineWidth = 30 + rnd() * 70;
    g.beginPath();
    const y0 = rnd() * size;
    g.moveTo(-40, y0);
    g.bezierCurveTo(size * 0.3, y0 + (rnd() - 0.5) * 180,
      size * 0.7, y0 + (rnd() - 0.5) * 180, size + 40, y0 + (rnd() - 0.5) * 120);
    g.stroke();
  }
  g.globalCompositeOperation = 'source-over';

  const img = g.getImageData(0, 0, size, size);
  const d = img.data;
  for (let i = 0; i < d.length; i += 4) {
    const n = (rnd() - 0.5) * 15;
    d[i] = Math.max(0, Math.min(255, d[i] + n));
    d[i + 1] = Math.max(0, Math.min(255, d[i + 1] + n));
    d[i + 2] = Math.max(0, Math.min(255, d[i + 2] + n));
  }
  g.putImageData(img, 0, 0);
  return cv;
}

/**
 * Hall wall: dark ribbed panelling with a service band, lifting toward the top
 * where the ceiling rig spills onto it. One tile is WALL_TILE metres wide and
 * covers the full wall height (no vertical repeat).
 */
function wallCanvas(w = 256, h = 512) {
  const cv = makeCanvas(w, h);
  if (!cv) return null;
  const g = cv.getContext('2d');
  const grd = g.createLinearGradient(0, h, 0, 0);
  grd.addColorStop(0.00, hex6(PALETTE.wall));
  grd.addColorStop(0.55, hex6(PALETTE.wall));
  grd.addColorStop(1.00, hex6(PALETTE.wallTop));
  g.fillStyle = grd;
  g.fillRect(0, 0, w, h);

  g.strokeStyle = hex6(PALETTE.wallRib);
  g.lineWidth = 3;
  for (let i = 1; i < 8; i++) {
    const x = (i / 8) * w;
    g.beginPath(); g.moveTo(x, 0); g.lineTo(x, h); g.stroke();
  }
  g.fillStyle = 'rgba(90,160,255,0.06)';
  g.fillRect(0, h * 0.60, w, h * 0.035);
  g.fillStyle = 'rgba(0,0,0,0.38)';
  g.fillRect(0, h * 0.635, w, h * 0.012);
  return cv;
}

/**
 * The barrier board. One tile is BANNER_TILE metres and carries
 * "JHU ALLIANCE LAB" and "S2C" once each, separated by a rule: small type, a lot
 * of dark, one accent. Restrained, the way a real board is.
 */
function bannerCanvas(w = 1024, h = 96) {
  const cv = makeCanvas(w, h);
  if (!cv) return null;
  const g = cv.getContext('2d');
  g.fillStyle = '#0a1220';
  g.fillRect(0, 0, w, h);
  const grd = g.createLinearGradient(0, 0, w, 0);
  grd.addColorStop(0.00, 'rgba(0,45,114,0.88)');
  grd.addColorStop(0.60, 'rgba(0,45,114,0.28)');
  grd.addColorStop(0.64, 'rgba(0,45,114,0)');
  g.fillStyle = grd;
  g.fillRect(0, 0, w, h);
  g.fillStyle = 'rgba(232,238,247,0.18)';
  g.fillRect(0, 0, w, 2);
  g.fillRect(0, h - 2, w, 2);

  g.textBaseline = 'middle';
  // letterspacing by hand: ctx.letterSpacing is not everywhere yet
  const run = (text, x, colour, font, extra) => {
    g.font = font;
    g.fillStyle = colour;
    let cur = x;
    for (const ch of text) {
      g.fillText(ch, cur, h / 2 + 1);
      cur += g.measureText(ch).width + extra;
    }
    return cur;
  };
  const SANS = '"Work Sans", "Inter", "Segoe UI", system-ui, sans-serif';
  run('JHU ALLIANCE LAB', 44, hex6(PALETTE.bannerInk), '600 42px ' + SANS, 4.5);
  g.fillStyle = 'rgba(232,238,247,0.26)';
  g.fillRect(Math.round(w * 0.655), Math.round(h * 0.22), 2, Math.round(h * 0.56));
  run('S2C', Math.round(w * 0.72), hex6(PALETTE.brandLift), '700 46px ' + SANS, 6);
  return cv;
}

/**
 * The contact-shadow mask: white in the middle, black at the rim, fully opaque.
 * It is used as an `alphaMap`, and three reads an alphaMap's GREEN channel — a
 * gradient made of transparent black would read as zero everywhere.
 */
function blobTexture(size = 128) {
  const cv = makeCanvas(size, size);
  if (!cv) return flatTexture(0x000000);
  const g = cv.getContext('2d');
  const r = size / 2;
  const grd = g.createRadialGradient(r, r, 0, r, r, r);
  grd.addColorStop(0.00, 'rgb(255,255,255)');
  grd.addColorStop(0.42, 'rgb(190,190,190)');
  grd.addColorStop(0.76, 'rgb(52,52,52)');
  grd.addColorStop(1.00, 'rgb(0,0,0)');
  g.fillStyle = grd;
  g.fillRect(0, 0, size, size);
  const t = new THREE.CanvasTexture(cv);
  t.colorSpace = THREE.NoColorSpace;      // a mask, not a colour
  t.needsUpdate = true;
  return t;
}

// ---------------------------------------------------------------------------
// geometry helpers
// ---------------------------------------------------------------------------

/** A rectangular band (a ring) extruded from z = 0 to z = h, centred on (cx, cy). */
function bandGeometry(cx, cy, innerX, innerY, outerX, outerY, h) {
  const shape = new THREE.Shape();
  shape.moveTo(cx - outerX, cy - outerY);
  shape.lineTo(cx + outerX, cy - outerY);
  shape.lineTo(cx + outerX, cy + outerY);
  shape.lineTo(cx - outerX, cy + outerY);
  shape.closePath();
  const hole = new THREE.Path();
  hole.moveTo(cx - innerX, cy - innerY);
  hole.lineTo(cx - innerX, cy + innerY);
  hole.lineTo(cx + innerX, cy + innerY);
  hole.lineTo(cx + innerX, cy - innerY);
  hole.closePath();
  shape.holes.push(hole);
  const g = new THREE.ExtrudeGeometry(shape, { depth: h, bevelEnabled: false, curveSegments: 1 });
  g.computeVertexNormals();
  return g;
}

/** One seated spectator: a stubby capsule plus a head, both stood up along +z. */
function crowdBodyGeometry(seg) {
  const g = new THREE.CapsuleGeometry(0.15, 0.30, 2, seg);
  g.rotateX(Math.PI / 2);
  g.translate(0, 0, 0.30);
  return g;
}
function crowdHeadGeometry(seg) {
  const g = new THREE.SphereGeometry(0.105, seg, Math.max(3, seg >> 1));
  g.translate(0, 0, 0.60);
  return g;
}

// ---------------------------------------------------------------------------
// buildVenue
// ---------------------------------------------------------------------------

/**
 * @param {THREE.Scene} scene
 * @param {THREE.WebGLRenderer} renderer
 * @param {{cx:number, cy:number, halfX:number, halfY:number}} field
 *        the resolved rectangle from app/render.js:resolveField
 * @param {{quality?: 'low'|'medium'|'high'}} [opts]
 */
export function buildVenue(scene, renderer, field, opts = {}) {
  const quality = opts.quality || 'high';
  const cx = field.cx, cy = field.cy;
  const [hx, hy] = LAYOUT.hallHalf;
  const maxAniso = renderer.capabilities.getMaxAnisotropy
    ? renderer.capabilities.getMaxAnisotropy() : 1;

  const group = new THREE.Group();
  group.name = 'venue';
  group.matrixAutoUpdate = false;
  const junk = [];
  const keep = (x) => { junk.push(x); return x; };
  const texs = [];
  const addTex = (t) => { texs.push(t); junk.push(t); return t; };

  // ---- the floor ----------------------------------------------------------
  // One matte slab over the whole hall, at z = 0 (the MJCF terrain plane's own
  // height) with the same polygon offset the checker disc it replaces used, so
  // the 1.5 mm decor slabs stacked above it cannot z-fight.
  const floorCv = floorCanvas(512);
  const floorTex = addTex(texFrom(floorCv, PALETTE.floor,
    (2 * hx) / FLOOR_TILE, (2 * hy) / FLOOR_TILE, maxAniso));
  floorTex.minFilter = THREE.LinearMipmapLinearFilter;
  floorTex.magFilter = THREE.LinearFilter;
  const floorMat = keep(new THREE.MeshStandardMaterial({
    map: floorTex, roughness: 0.95, metalness: 0.04,
    polygonOffset: true, polygonOffsetFactor: 2, polygonOffsetUnits: 2,
  }));
  const floor = new THREE.Mesh(keep(new THREE.PlaneGeometry(2 * hx, 2 * hy, 1, 1)), floorMat);
  floor.position.set(cx, cy, 0);
  floor.receiveShadow = true;
  floor.matrixAutoUpdate = false;
  floor.updateMatrix();
  group.add(floor);

  // ---- the shell: four walls and a ceiling --------------------------------
  // DoubleSide on purpose: broadcast can be zoomed out past the ceiling, and a
  // dark surface seen from outside beats seeing the void through a culled wall.
  const wallCv = wallCanvas();
  const wallTexY = addTex(texFrom(wallCv, PALETTE.wall, (2 * hx) / WALL_TILE, 1, maxAniso));
  const wallTexX = addTex(texFrom(wallCv, PALETTE.wall, (2 * hy) / WALL_TILE, 1, maxAniso));
  const wallMatY = keep(new THREE.MeshStandardMaterial({
    map: wallTexY, roughness: 0.95, metalness: 0.02, side: THREE.DoubleSide }));
  const wallMatX = keep(new THREE.MeshStandardMaterial({
    map: wallTexX, roughness: 0.95, metalness: 0.02, side: THREE.DoubleSide }));
  const wallGeoY = keep(new THREE.PlaneGeometry(2 * hx, LAYOUT.ceilingZ, 1, 1));
  const wallGeoX = keep(new THREE.PlaneGeometry(2 * hy, LAYOUT.ceilingZ, 1, 1));
  const zc = LAYOUT.ceilingZ / 2;
  // yaw is chosen so each wall's front face looks INTO the hall.
  group.add(standUp(new THREE.Mesh(wallGeoY, wallMatY), cx, cy - hy, zc, Math.PI));
  group.add(standUp(new THREE.Mesh(wallGeoY, wallMatY), cx, cy + hy, zc, 0));
  group.add(standUp(new THREE.Mesh(wallGeoX, wallMatX), cx - hx, cy, zc, Math.PI / 2));
  group.add(standUp(new THREE.Mesh(wallGeoX, wallMatX), cx + hx, cy, zc, -Math.PI / 2));

  const ceiling = new THREE.Mesh(
    keep(new THREE.PlaneGeometry(2 * hx, 2 * hy, 1, 1)),
    keep(new THREE.MeshStandardMaterial({
      color: srgb(PALETTE.ceiling), roughness: 1.0, metalness: 0.0, side: THREE.DoubleSide })),
  );
  ceiling.position.set(cx, cy, LAYOUT.ceilingZ);
  ceiling.rotation.x = Math.PI;                 // normal points down, into the hall
  ceiling.matrixAutoUpdate = false;
  ceiling.updateMatrix();
  group.add(ceiling);

  // ---- the bowl -----------------------------------------------------------
  const tierMats = [
    keep(new THREE.MeshStandardMaterial({ color: srgb(PALETTE.tierA), roughness: 0.93, metalness: 0.03 })),
    keep(new THREE.MeshStandardMaterial({ color: srgb(PALETTE.tierB), roughness: 0.93, metalness: 0.03 })),
  ];
  const tierRects = [];
  for (let i = 0; i < LAYOUT.tiers; i++) {
    const innerX = LAYOUT.bowlInner[0] + i * LAYOUT.tierDepth;
    const innerY = LAYOUT.bowlInner[1] + i * LAYOUT.tierDepth;
    const outerX = innerX + LAYOUT.tierDepth;
    const outerY = innerY + LAYOUT.tierDepth;
    const topZ = LAYOUT.tierFirstZ + i * LAYOUT.tierRise;
    tierRects.push([innerX, innerY, outerX, outerY, topZ]);
    const m = new THREE.Mesh(
      keep(bandGeometry(cx, cy, innerX, innerY, outerX, outerY, topZ)), tierMats[i & 1]);
    m.matrixAutoUpdate = false;
    m.updateMatrix();
    group.add(m);
  }
  {
    const [ix, iy] = LAYOUT.bowlInner;
    // The front wall is the single biggest surface the chase camera ever sees
    // that is not the pitch. Left as raw extruded sides it reads as one black
    // band across the top of the frame, so it gets the panelling the hall walls
    // have, on four planes sunk 2 mm into the bowl so they cannot z-fight it.
    const fw = LAYOUT.tierFirstZ;
    const frontTexY = addTex(texFrom(wallCv, PALETTE.wall, (2 * ix) / WALL_TILE, 1, maxAniso));
    const frontTexX = addTex(texFrom(wallCv, PALETTE.wall, (2 * iy) / WALL_TILE, 1, maxAniso));
    const frontMatY = keep(new THREE.MeshStandardMaterial({
      map: frontTexY, roughness: 0.93, metalness: 0.03 }));
    const frontMatX = keep(new THREE.MeshStandardMaterial({
      map: frontTexX, roughness: 0.93, metalness: 0.03 }));
    const gY = keep(new THREE.PlaneGeometry(2 * ix, fw, 1, 1));
    const gX = keep(new THREE.PlaneGeometry(2 * iy, fw, 1, 1));
    const d = 0.002;   // yaw as for the banners: the front face looks at the pitch
    group.add(standUp(new THREE.Mesh(gY, frontMatY), cx, cy - iy + d, fw / 2, Math.PI));
    group.add(standUp(new THREE.Mesh(gY, frontMatY), cx, cy + iy - d, fw / 2, 0));
    group.add(standUp(new THREE.Mesh(gX, frontMatX), cx - ix + d, cy, fw / 2, Math.PI / 2));
    group.add(standUp(new THREE.Mesh(gX, frontMatX), cx + ix - d, cy, fw / 2, -Math.PI / 2));

    // An LED trim lip along the top of that wall: the one thing in the venue
    // that is above 1.0 in linear light at pitch level, so the bloom has
    // something to do in shots where the ceiling rig is out of frame.
    const m = new THREE.Mesh(
      keep(bandGeometry(cx, cy, ix, iy, ix + 0.05, iy + 0.05, fw + 0.025)),
      keep(new THREE.MeshBasicMaterial({
        color: new THREE.Color().setRGB(0.55, 0.72, 1.05, THREE.LinearSRGBColorSpace) })),
    );
    m.matrixAutoUpdate = false;
    m.updateMatrix();
    group.add(m);
  }

  // ---- the crowd ----------------------------------------------------------
  // Two InstancedMeshes = two draw calls for the whole audience. Placement is a
  // deterministic LCG, then a deterministic shuffle, so lowering `count` for a
  // lower quality tier thins the crowd evenly instead of deleting a whole side.
  const seats = [];
  {
    const rnd = lcg(0x5A2C01);
    for (let t = 0; t < tierRects.length; t++) {
      const [ix, iy, ox, oy, topZ] = tierRects[t];
      for (let r = 0; r < LAYOUT.rowsPerTier; r++) {
        const f = (r + 0.62) / (LAYOUT.rowsPerTier + 0.2);
        const rx = ix + (ox - ix) * f;
        const ry = iy + (oy - iy) * f;
        const nX = Math.max(2, Math.round((2 * rx) / LAYOUT.seatPitch));
        const nY = Math.max(2, Math.round((2 * ry) / LAYOUT.seatPitch));
        const push = (x, y) => {
          if (rnd() < 0.16) return;                       // empty seats
          const px = cx + x + (rnd() - 0.5) * 0.14;
          const py = cy + y + (rnd() - 0.5) * 0.14;
          seats.push({
            x: px, y: py, z: topZ,
            yaw: Math.atan2(cy - py, cx - px),
            s: 0.88 + rnd() * 0.26,
            wear: CROWD_WEAR[(rnd() * CROWD_WEAR.length) | 0],
            skin: CROWD_SKIN[(rnd() * CROWD_SKIN.length) | 0],
          });
        };
        for (let i = 0; i <= nX; i++) {
          const x = -rx + (2 * rx) * (i / nX);
          push(x, -ry); push(x, ry);
        }
        for (let i = 1; i < nY; i++) {
          const y = -ry + (2 * ry) * (i / nY);
          push(-rx, y); push(rx, y);
        }
      }
    }
    for (let i = seats.length - 1; i > 0; i--) {
      const j = (rnd() * (i + 1)) | 0;
      const tmp = seats[i]; seats[i] = seats[j]; seats[j] = tmp;
    }
  }

  const seg = quality === 'high' ? 7 : 5;
  const crowdBodyGeo = keep(crowdBodyGeometry(seg));
  const crowdHeadGeo = keep(crowdHeadGeometry(seg));
  const bodies = new THREE.InstancedMesh(
    crowdBodyGeo,
    keep(new THREE.MeshStandardMaterial({ roughness: 0.96, metalness: 0.0 })),
    Math.max(1, seats.length));
  const heads = new THREE.InstancedMesh(
    crowdHeadGeo,
    keep(new THREE.MeshStandardMaterial({ roughness: 0.90, metalness: 0.0 })),
    Math.max(1, seats.length));
  {
    const m4 = new THREE.Matrix4();
    const pos = new THREE.Vector3();
    const qt = new THREE.Quaternion();
    const sc = new THREE.Vector3();
    const col = new THREE.Color();
    for (let i = 0; i < seats.length; i++) {
      const s = seats[i];
      pos.set(s.x, s.y, s.z);
      qt.setFromAxisAngle(_AZ, s.yaw);
      sc.set(s.s, s.s, s.s);
      m4.compose(pos, qt, sc);
      bodies.setMatrixAt(i, m4);
      heads.setMatrixAt(i, m4);
      bodies.setColorAt(i, col.setHex(s.wear, THREE.SRGBColorSpace));
      heads.setColorAt(i, col.setHex(s.skin, THREE.SRGBColorSpace));
    }
    bodies.instanceMatrix.needsUpdate = true;
    heads.instanceMatrix.needsUpdate = true;
    if (bodies.instanceColor) bodies.instanceColor.needsUpdate = true;
    if (heads.instanceColor) heads.instanceColor.needsUpdate = true;
  }
  for (const m of [bodies, heads]) {
    m.castShadow = false; m.receiveShadow = false;
    m.matrixAutoUpdate = false;
    m.computeBoundingSphere();
    group.add(m);
  }

  // ---- barrier branding ---------------------------------------------------
  // Flush on the INNER face of each MJCF barrier box, 1 mm proud, sized to the
  // face it covers so it can never poke into the neighbouring barrier.
  const bannerCv = bannerCanvas();
  const [bxMin, bxMax] = DECOR.barrierX;
  const [byMin, byMax] = DECOR.barrierY;
  const th = DECOR.barrierThickHalf;
  const eps = 0.001;
  const BANNER_Z0 = 0.022, BANNER_Z1 = 0.206;    // inside the 0 .. 0.22 barrier face
  const bannerH = BANNER_Z1 - BANNER_Z0;
  const bannerZ = (BANNER_Z0 + BANNER_Z1) / 2;
  const lenX = bxMax - bxMin;                    // the y-barriers' own length
  const lenY = (byMax - th) - (byMin + th);      // the x-barriers' visible length
  const bannerMats = [];
  for (const [w, px, py, yaw] of [
    [lenX, (bxMin + bxMax) / 2, byMin + th + eps, Math.PI],
    [lenX, (bxMin + bxMax) / 2, byMax - th - eps, 0],
    [lenY, bxMin + th + eps, (byMin + byMax) / 2, Math.PI / 2],
    [lenY, bxMax - th - eps, (byMin + byMax) / 2, -Math.PI / 2],
  ]) {
    const tex = addTex(texFrom(bannerCv, PALETTE.brand,
      Math.max(1, Math.round(w / BANNER_TILE)), 1, maxAniso));
    const mat = keep(new THREE.MeshStandardMaterial({ map: tex, roughness: 0.58, metalness: 0.06 }));
    bannerMats.push(mat);
    const m = new THREE.Mesh(keep(new THREE.PlaneGeometry(w, bannerH, 1, 1)), mat);
    m.receiveShadow = true;
    group.add(standUp(m, px, py, bannerZ, yaw));
  }

  // ---- the ceiling rig ----------------------------------------------------
  // Housings (dull metal) plus lamp faces set above 1.0 in LINEAR light, so the
  // bright pass has real headroom to bleed from instead of clipped white.
  const rigCount = LAYOUT.rigRows * LAYOUT.rigBars;
  const barLen = (2 * hx) / (LAYOUT.rigBars + 1.1);
  const housings = new THREE.InstancedMesh(
    keep(new THREE.BoxGeometry(barLen, 0.30, 0.16)),
    keep(new THREE.MeshStandardMaterial({
      color: srgb(PALETTE.rig), roughness: 0.6, metalness: 0.5, side: THREE.DoubleSide })),
    rigCount);
  const lamps = new THREE.InstancedMesh(
    keep(new THREE.BoxGeometry(barLen * 0.94, 0.20, 0.03)),
    keep(new THREE.MeshBasicMaterial({
      color: new THREE.Color().setRGB(2.6, 2.9, 3.4, THREE.LinearSRGBColorSpace),
      fog: false, side: THREE.DoubleSide })),
    rigCount);
  {
    const m4 = new THREE.Matrix4();
    const pos = new THREE.Vector3();
    const qt = new THREE.Quaternion();
    const sc = new THREE.Vector3(1, 1, 1);
    let k = 0;
    for (let r = 0; r < LAYOUT.rigRows; r++) {
      const y = cy + (2 * hy) * ((r + 1) / (LAYOUT.rigRows + 1) - 0.5) * 0.62;
      for (let b = 0; b < LAYOUT.rigBars; b++) {
        const x = cx + (2 * hx) * ((b + 1) / (LAYOUT.rigBars + 1) - 0.5) * 0.86;
        pos.set(x, y, LAYOUT.rigZ);
        m4.compose(pos, qt, sc);
        housings.setMatrixAt(k, m4);
        pos.set(x, y, LAYOUT.rigZ - 0.09);
        m4.compose(pos, qt, sc);
        lamps.setMatrixAt(k, m4);
        k++;
      }
    }
    housings.instanceMatrix.needsUpdate = true;
    lamps.instanceMatrix.needsUpdate = true;
  }
  for (const m of [housings, lamps]) {
    m.castShadow = false; m.receiveShadow = false;
    m.matrixAutoUpdate = false;
    m.computeBoundingSphere();
    group.add(m);
  }

  scene.add(group);

  // ---- quality ------------------------------------------------------------
  function setQuality(q) {
    const showCrowd = q !== 'low';
    const n = q === 'high' ? seats.length : Math.round(seats.length * 0.55);
    bodies.visible = heads.visible = showCrowd;
    bodies.count = heads.count = showCrowd ? n : 0;
    const aniso = q === 'low' ? 1 : maxAniso;
    for (const t of texs) {
      if (t.anisotropy !== aniso) { t.anisotropy = aniso; t.needsUpdate = true; }
    }
  }
  setQuality(quality);

  return {
    group,
    fogColor: srgb(PALETTE.haze),
    setQuality,
    dispose() {
      scene.remove(group);
      for (const d of junk) { if (d && d.dispose) d.dispose(); }
      bodies.dispose(); heads.dispose(); housings.dispose(); lamps.dispose();
    },
    info: {
      hall: [2 * hx, 2 * hy, LAYOUT.ceilingZ],
      tiers: LAYOUT.tiers,
      seats: seats.length,
      get spectatorsDrawn() { return bodies.visible ? bodies.count : 0; },
      rigLamps: rigCount,
      textureBytesOnWire: 0,
      drawCalls: 5 /* floor+walls+ceiling */ + LAYOUT.tiers + 1 + 2 /* crowd */ + 4 /* banners */ + 2,
    },
  };
}

// ---------------------------------------------------------------------------
// contact shadows
// ---------------------------------------------------------------------------

/**
 * A soft dark blob under each dog. The key light's shadow map grounds them at
 * `medium` and `high`, but at `low` there is no shadow map at all, and even with
 * one, a directional shadow never darkens the few centimetres directly under the
 * belly. Two quads, two matrix writes a frame, no allocation.
 *
 * z = 0.0095 clears every flat decor slab (the topmost, decor_line, has its upper
 * face at 0.008) and sits below the player ring at 0.02.
 */
export function createContactShadows(scene) {
  const tex = blobTexture();
  const geo = new THREE.PlaneGeometry(1, 1, 1, 1);
  const mats = [];
  const mk = () => {
    const mat = new THREE.MeshBasicMaterial({
      color: 0x000000, alphaMap: tex, transparent: true, opacity: 0.55,
      depthWrite: false, toneMapped: false, fog: false,
    });
    mats.push(mat);
    const m = new THREE.Mesh(geo, mat);
    m.renderOrder = 800;
    m.matrixAutoUpdate = false;
    m.frustumCulled = false;
    scene.add(m);
    return m;
  };
  const meshes = { a: mk(), b: mk() };
  const _p = new THREE.Vector3();
  const _q = new THREE.Quaternion();
  const _s = new THREE.Vector3();
  return {
    /** @param {'a'|'b'} rob @param {number} x @param {number} y @param {number} z trunk height */
    place(rob, x, y, z) {
      const m = meshes[rob];
      if (!m) return;
      // wider and fainter the higher the trunk rides — a cheap soft shadow
      const k = 1 + 1.4 * (z > 0.05 ? z : 0.05);
      _p.set(x, y, 0.0095);
      _s.set(0.95 * k, 0.95 * k, 1);
      m.matrix.compose(_p, _q, _s);
      m.matrixWorldNeedsUpdate = true;
      m.material.opacity = 0.66 / k;
    },
    setVisible(on) { meshes.a.visible = !!on; meshes.b.visible = !!on; },
    dispose() {
      scene.remove(meshes.a); scene.remove(meshes.b);
      for (const m of mats) m.dispose();
      geo.dispose(); tex.dispose();
    },
  };
}

// ---------------------------------------------------------------------------
// bloom
// ---------------------------------------------------------------------------

/**
 * A small bloom with no new dependency. The vendored three build is the core
 * only (no examples/jsm), so EffectComposer/UnrealBloomPass do not exist here and
 * are not worth vendoring for one effect.
 *
 *   1. The scene is rendered into a half-float target. three turns tone mapping
 *      and the sRGB encode OFF whenever the destination is a render target
 *      (WebGLProgram's prefix only injects them when `currentRenderTarget === null`;
 *      the output colour space for any other target is the working space), so what
 *      lands there is raw linear HDR — exactly what a bright pass needs.
 *   2. A soft-knee bright pass at quarter resolution, then two separable Gaussian
 *      blurs ping-ponged at that size (5 taps each, linear-sampled 9-wide kernel).
 *   3. One fullscreen composite to the canvas: scene + strength*bloom, then
 *      `#include <tonemapping_fragment>` and `#include <colorspace_fragment>`.
 *      Because the composite IS rendered to the canvas, three injects its own
 *      `toneMapping()` — whatever `renderer.toneMapping` is set to, with the
 *      renderer's own exposure. The bloom path and the plain path therefore share
 *      one tone curve by construction, and differ only by the glow.
 */
export function createBloom(renderer, scene, camera, opts = {}) {
  const strength = opts.strength ?? 0.55;
  const threshold = opts.threshold ?? 0.85;
  const knee = opts.knee ?? 0.35;
  const radius = opts.radius ?? 1.3;
  const samples = opts.samples ?? 4;

  const base = {
    type: THREE.HalfFloatType,
    minFilter: THREE.LinearFilter,
    magFilter: THREE.LinearFilter,
    stencilBuffer: false,
  };
  const rtScene = new THREE.WebGLRenderTarget(1, 1, { ...base, depthBuffer: true, samples });
  const rtA = new THREE.WebGLRenderTarget(1, 1, { ...base, depthBuffer: false });
  const rtB = new THREE.WebGLRenderTarget(1, 1, { ...base, depthBuffer: false });
  for (const rt of [rtScene, rtA, rtB]) rt.texture.colorSpace = THREE.NoColorSpace;

  const QUAD_VERT = /* glsl */`
    varying vec2 vUv;
    void main() { vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }`;

  // Unity/KinoBloom soft knee: the glow fades in instead of popping on.
  const brightMat = new THREE.ShaderMaterial({
    uniforms: {
      tSrc: { value: rtScene.texture },
      uThreshold: { value: threshold },
      uKnee: { value: knee },
    },
    vertexShader: QUAD_VERT,
    fragmentShader: /* glsl */`
      uniform sampler2D tSrc; uniform float uThreshold; uniform float uKnee;
      varying vec2 vUv;
      void main() {
        vec3 c = texture2D(tSrc, vUv).rgb;
        float br = max(c.r, max(c.g, c.b));
        float rq = clamp(br - uThreshold + uKnee, 0.0, 2.0 * uKnee);
        rq = rq * rq / (4.0 * uKnee + 1e-4);
        float w = max(rq, br - uThreshold) / max(br, 1e-4);
        gl_FragColor = vec4(c * clamp(w, 0.0, 1.0), 1.0);
      }`,
    depthTest: false, depthWrite: false,
  });

  const blurMat = new THREE.ShaderMaterial({
    uniforms: { tSrc: { value: null }, uStep: { value: new THREE.Vector2() } },
    vertexShader: QUAD_VERT,
    fragmentShader: /* glsl */`
      uniform sampler2D tSrc; uniform vec2 uStep;
      varying vec2 vUv;
      void main() {
        const float O1 = 1.3846153846, O2 = 3.2307692308;
        const float W0 = 0.2270270270, W1 = 0.3162162162, W2 = 0.0702702703;
        vec3 c = texture2D(tSrc, vUv).rgb * W0;
        c += texture2D(tSrc, vUv + uStep * O1).rgb * W1;
        c += texture2D(tSrc, vUv - uStep * O1).rgb * W1;
        c += texture2D(tSrc, vUv + uStep * O2).rgb * W2;
        c += texture2D(tSrc, vUv - uStep * O2).rgb * W2;
        gl_FragColor = vec4(c, 1.0);
      }`,
    depthTest: false, depthWrite: false,
  });

  const compositeMat = new THREE.ShaderMaterial({
    uniforms: {
      tScene: { value: rtScene.texture },
      tBloom: { value: rtA.texture },
      uStrength: { value: strength },
    },
    vertexShader: QUAD_VERT,
    fragmentShader: /* glsl */`
      uniform sampler2D tScene; uniform sampler2D tBloom; uniform float uStrength;
      varying vec2 vUv;
      void main() {
        vec3 c = texture2D(tScene, vUv).rgb + texture2D(tBloom, vUv).rgb * uStrength;
        gl_FragColor = vec4(c, 1.0);
        #include <tonemapping_fragment>
        #include <colorspace_fragment>
      }`,
    depthTest: false, depthWrite: false,
  });

  const quadGeo = new THREE.PlaneGeometry(2, 2);
  const quadCam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
  const quadScene = new THREE.Scene();
  const quad = new THREE.Mesh(quadGeo, brightMat);
  quad.frustumCulled = false;
  quadScene.add(quad);

  let bw = 1, bh = 1;
  function setSize(width, height) {
    const w = Math.max(1, Math.round(width));
    const h = Math.max(1, Math.round(height));
    bw = Math.max(1, w >> 2);
    bh = Math.max(1, h >> 2);
    rtScene.setSize(w, h);
    rtA.setSize(bw, bh);
    rtB.setSize(bw, bh);
  }

  const _step = blurMat.uniforms.uStep.value;
  function pass(mat, target) {
    quad.material = mat;
    renderer.setRenderTarget(target);
    renderer.render(quadScene, quadCam);
  }
  function blur(src, dst, dx, dy) {
    blurMat.uniforms.tSrc.value = src.texture;
    _step.set(dx, dy);
    pass(blurMat, dst);
  }

  function render() {
    renderer.setRenderTarget(rtScene);
    renderer.render(scene, camera);
    pass(brightMat, rtA);
    blur(rtA, rtB, radius / bw, 0);
    blur(rtB, rtA, 0, radius / bh);
    blur(rtA, rtB, radius * 2.2 / bw, 0);
    blur(rtB, rtA, 0, radius * 2.2 / bh);
    compositeMat.uniforms.tBloom.value = rtA.texture;
    renderer.setRenderTarget(null);
    pass(compositeMat, null);
  }

  return {
    render,
    setSize,
    setStrength(s) { compositeMat.uniforms.uStrength.value = s; },
    dispose() {
      rtScene.dispose(); rtA.dispose(); rtB.dispose();
      quadGeo.dispose();
      brightMat.dispose(); blurMat.dispose(); compositeMat.dispose();
    },
  };
}

export default { buildVenue, createContactShadows, createBloom };

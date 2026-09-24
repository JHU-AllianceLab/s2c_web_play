/**
 * app/main.js — the shell. Boots the UI, and on Start builds a session:
 *
 *     scene.xml  -> app/physics.js  createSim()      (MuJoCo WASM, real plant)
 *     manifest   -> app/policy.js   loadPolicy()     (our exported checkpoint weights)
 *     canvas     -> app/render.js   createRenderer() (three.js, MuJoCo z-up frame)
 *     the above  -> app/match.js    createMatch()    (the 50 Hz control loop + referee)
 *
 * The loop below owns only WALL CLOCK. Physics advances in fixed 20 ms control
 * steps through `match.tick()`; rendering happens once per animation frame from
 * the sim's pose buffer. A slow frame therefore costs frames, never physics:
 * the accumulator catches up, bounded by MAX_CATCHUP_STEPS so a backgrounded
 * tab cannot come back and simulate a minute in one frame.
 *
 * Nothing about the game lives here — no rules, no geometry, no policy paths.
 * This file is wiring.
 */

import { createUI } from './ui.js';
import { createSim } from './physics.js';
import { createRenderer } from './render.js';
import { createMatch, PHASE } from './match.js';
import { loadManifest } from './policy.js';
import * as CONFIG from './config.js';

/** Never simulate more than this many control steps in one animation frame. */
const MAX_CATCHUP_STEPS = 6;

/** The camera modes app/render.js implements. */
const CAMERA_MODES = ['chase', 'fpv', 'broadcast'];

const SCENE_URL = (game) => `assets/scene/${game}/scene.xml`;

/** Games this build offers. Add 'sym' back here (and un-ignore its assets) to ship it. */
const PUBLISHED_GAMES = ['asym'];

let session = null;   // the live match, or null on the title screen
let ui = null;

/* --------------------------------------------------------------------------- utilities */

const nowMs = () =>
  (typeof performance !== 'undefined' && performance.now ? performance.now() : Date.now());

function logError(err, where) {
  // Mirror into the page as well: the headless screenshot harness reads #log.
  console.error(`[s2c] ${where}:`, err);
  const pre = document.getElementById('log');
  if (pre) pre.textContent += `${where}: ${(err && err.stack) || err}\n`;
}

/* --------------------------------------------------------------------------- session */

async function startSession(setup) {
  await teardown();

  const game = setup.game;
  const steps = [
    { id: 'scene', label: 'Arena and physics' },
    { id: 'walk', label: 'Your locomotion policy' },
    { id: 'ai', label: `Opponent policy — ${setup.opponent.display || setup.opponent.method}` },
    { id: 'render', label: 'Renderer' },
  ];
  ui.loading.begin(steps);

  try {
    // ---- 1. physics ---------------------------------------------------------
    ui.loading.update('scene', { state: 'active' });
    const sim = await createSim({ sceneUrl: SCENE_URL(game) });
    ui.loading.update('scene', { state: 'done', detail: `${sim.nbody} bodies, ${sim.nu} actuators` });
    ui.loading.progress(0.35, 'Arena ready');

    // ---- 2. policies --------------------------------------------------------
    const man = await loadManifest();
    ui.loading.update('walk', { state: 'active' });
    const walk = await man.load(setup.playerWalk.name);
    ui.loading.update('walk', { state: 'done', detail: `${walk.obsDim}→${walk.actDim}` });
    ui.loading.progress(0.6, 'Walk policy ready');

    ui.loading.update('ai', { state: 'active' });
    const ai = await man.load(setup.opponent.name);
    ui.loading.update('ai', { state: 'done', detail: `${ai.obsDim}→${ai.actDim}` });
    ui.loading.progress(0.8, 'Opponent ready');

    // ---- 3. match -----------------------------------------------------------
    const match = createMatch({
      sim,
      game,
      playerSeat: setup.playerSeat,
      opponent: setup.opponent,
      policies: { walk, ai },
      input: ui.input,
    });
    const playerRobot = match.info?.playerRobot ?? match.state().playerRobot;

    // ---- 4. renderer --------------------------------------------------------
    ui.loading.update('render', { state: 'active' });
    const renderer = createRenderer(ui.canvas, sim, game, {
      playerRobot,
      quality: ui.settings?.quality,
      preserveDrawingBuffer: true,   // so the screenshot harness captures pixels
    });
    for (const w of renderer.warnings || []) console.warn('[s2c][render]', w);
    ui.loading.update('render', { state: 'done' });
    ui.loading.progress(1, 'Ready');

    // ---- 5. go --------------------------------------------------------------
    match.reset();
    ui.startMatch({ game, playerSeat: setup.playerSeat, opponent: setup.opponent });
    const camera = ui.settings?.camera || 'chase';
    if (CAMERA_MODES.includes(camera)) renderer.setCamera(camera, true);

    session = { sim, match, renderer, game, playerRobot, raf: 0, last: nowMs(), acc: 0, fps: 60, resultShown: false };
    session.raf = requestAnimationFrame(frame);
    globalThis.__s2c = { ...(globalThis.__s2c || {}), session, sim, match, renderer };
    return session;
  } catch (err) {
    logError(err, 'startSession');
    ui.loading.fail(String((err && err.message) || err));
    throw err;
  }
}

async function teardown() {
  if (!session) return;
  cancelAnimationFrame(session.raf);
  try { session.renderer?.dispose?.(); } catch (err) { logError(err, 'renderer.dispose'); }
  try { session.match?.dispose?.(); } catch (err) { logError(err, 'match.dispose'); }
  try { session.sim?.dispose?.(); } catch (err) { logError(err, 'sim.dispose'); }
  session = null;
}

/* --------------------------------------------------------------------------- the loop */

function frame() {
  const s = session;
  if (!s) return;
  s.raf = requestAnimationFrame(frame);

  const t = nowMs();
  let dt = t - s.last;
  s.last = t;
  if (!(dt >= 0) || dt > 1000) dt = CONFIG.PHYS.controlDt * 1000;   // tab wake-up
  s.fps += ((1000 / Math.max(dt, 1)) - s.fps) * 0.1;

  const controlMs = CONFIG.PHYS.controlDt * 1000;
  let hud = null;

  if (ui.isPaused()) {
    s.acc = 0;
    hud = s.match.hud();
  } else {
    s.acc += dt;
    let n = 0;
    while (s.acc >= controlMs && n < MAX_CATCHUP_STEPS) {
      hud = s.match.tick(controlMs);
      s.acc -= controlMs;
      n += 1;
      if (hud.verdict) { s.acc = 0; break; }
    }
    if (s.acc > controlMs * MAX_CATCHUP_STEPS) s.acc = 0;
    if (!hud) hud = s.match.hud();
  }

  // Render from the sim's own pose buffer (reused; never allocated per frame).
  s.renderer.update(s.sim.renderState(), s.playerRobot);

  ui.updateHud({ ...hud, fps: s.fps, camera: s.renderer.cameraMode });

  if (hud.verdict && !s.resultShown) {
    s.resultShown = true;
    ui.showResult(hud.verdict);
  }
}

/* --------------------------------------------------------------------------- boot */

export async function boot() {
  ui = await createUI({
    config: CONFIG,
    capabilities: { filter: false },   // the QCBF seam exists in match.js; no weights wired yet
    cameraModes: CAMERA_MODES,
    // What this build ships. The engine runs both games (tests/node_match.mjs
    // covers sym too); only the asymmetric one is published for now.
    games: PUBLISHED_GAMES,

    onStart: (setup) => { startSession(setup).catch(() => {}); },

    onRestart: () => {
      if (!session) return;
      session.match.reset();
      session.resultShown = false;
      session.acc = 0;
      session.last = nowMs();
      ui.hideResult();
    },

    onChangeSetup: () => { teardown(); },

    onPause: () => { session?.match.pause(); },
    onResume: () => { session?.match.resume(); session && (session.last = nowMs()); },

    onCamera: (mode) => {
      if (session && CAMERA_MODES.includes(mode)) session.renderer.setCamera(mode);
    },

    onSetting: (key, value) => {
      if (!session) return;
      if (key === 'quality') session.renderer.setQuality?.(value);
      if (key === 'cameraDistance') session.renderer.setCameraDistance?.(value);
      if (key === 'interpolate') session.renderer.setInterpolation?.(value);
    },

    onResize: () => { session?.renderer.resize(); },
  });

  globalThis.__s2c = { ui, CONFIG };
  return ui;
}

export default { boot };

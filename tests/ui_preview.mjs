/**
 * tests/ui_preview.mjs — drives app/ui.js with stub state, so the interface can be built,
 * reviewed and screenshotted without the wasm engine, the scene or the policy weights.
 *
 * Reached from the real page with `index.html?uipreview=1`, and by tests/shoot_ui.mjs.
 * It loads NO wasm and NO weights. Everything it shows in the HUD is fake and labelled as such
 * on the canvas.
 *
 * It uses the real app/config.js when that file exists. If it does not exist yet (the rules agent
 * owns it), it falls back to FROZEN_CONFIG below, which is the frozen app/config.js contract from
 * the build brief copied verbatim — not a guess, and not a second source of truth: the moment
 * app/config.js lands, that file wins.
 */

/* eslint-disable no-console */

/* Console capture, so tests/shoot_ui.mjs can assert the interface logs nothing. Installed before
 * anything else in the preview path runs. */
const captured = (globalThis.__s2cPreviewErrors = []);
for (const level of ['error', 'warn']) {
  const original = console[level].bind(console);
  console[level] = (...args) => {
    captured.push({ level, text: args.map((a) => (a && a.stack) || String(a)).join(' ') });
    original(...args);
  };
}
globalThis.addEventListener('error', (ev) =>
  captured.push({ level: 'error', text: String((ev.error && ev.error.stack) || ev.message) })
);
globalThis.addEventListener('unhandledrejection', (ev) =>
  captured.push({ level: 'error', text: 'unhandled rejection: ' + String(ev.reason) })
);

const FROZEN_CONFIG = {
  GAMES: {
    sym: {
      field: [5.6, 3.0],
      center: [0, 0],
      lineX: 1.9,
      seats: ['A', 'B'],
      episodeSteps: 500,
      spawn: {
        A: { x: -2.1431, y: -0.1409, yaw: 0.0979 },
        B: { x: 2.1775, y: -0.1215, yaw: 3.1381 },
      },
    },
    asym: {
      field: [5.2, 3.0],
      center: [0.2, 0],
      lineX: 1.9,
      seats: ['attacker', 'defender'],
      episodeSteps: 500,
      spawn: {
        attacker: { x: -1.4, y: 0, yaw: 0 },
        defender: { x: 0.75, y: 0, yaw: Math.PI },
      },
    },
  },
  PHYS: { dt: 0.005, decimation: 4, controlDt: 0.02 },
  CMD_BOX: { vx: [-1.5, 3.0], vy: [-1.0, 1.0], wz: [-2.0, 2.0] },
  DEFAULT_JOINT_POS: [-0.1, 0.9, -1.8, 0.1, 0.9, -1.8, -0.1, 0.9, -1.8, 0.1, 0.9, -1.8],
  __source: 'tests/ui_preview.mjs FROZEN_CONFIG (app/config.js was not found)',
};

async function resolveConfig() {
  const url = new URL('../app/config.js', import.meta.url);
  let exists = false;
  try {
    const res = await fetch(url, { method: 'HEAD' });
    exists = res.ok;
  } catch {
    exists = false;
  }
  if (exists) {
    try {
      const mod = await import(url.href);
      if (mod && mod.GAMES && mod.CMD_BOX) return { config: mod, real: true };
    } catch (err) {
      console.warn('[ui-preview] app/config.js exists but did not import:', err);
    }
  }
  return { config: FROZEN_CONFIG, real: false };
}

function stageNote(canvas, text) {
  const dpr = Math.min(globalThis.devicePixelRatio || 1, 2);
  const w = canvas.clientWidth || 1200;
  const hh = canvas.clientHeight || 700;
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(hh * dpr);
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const g = ctx.createLinearGradient(0, 0, 0, hh);
  g.addColorStop(0, '#0b1220');
  g.addColorStop(1, '#070a10');
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, w, hh);
  ctx.fillStyle = 'rgba(174,187,207,0.35)';
  ctx.font = '500 13px system-ui, sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText(text, w / 2, hh / 2);
}

export async function boot() {
  const { config, real } = await resolveConfig();
  globalThis.S2C_CONFIG_OVERRIDE = config;

  const { createUI } = await import(new URL('../app/ui.js', import.meta.url).href);

  const NOTE = real
    ? 'UI preview — app/render.js is not loaded'
    : 'UI preview — app/render.js and app/config.js are not loaded (frozen contract in use)';

  const params = new URLSearchParams(location.search);
  const ui = await createUI({
    config,
    capabilities: { filter: params.get('filtercap') === '1' },
    games: ['asym', 'sym'],          // the order app/main.js publishes
    onStart: (setup) => runFakeMatch(setup),
    onRestart: () => runFakeMatch(lastSetup),
    onChangeSetup: () => stop(),
    onPause: () => {
      paused = true;
    },
    onResume: () => {
      paused = false;
    },
    onCamera: (mode) => {
      camera = mode;
    },
    onResize: () => stageNote(ui.canvas, NOTE),
  });

  let lastSetup = null;
  let timer = null;
  let paused = false;
  let camera = 'chase';
  let step = 0;
  let t0 = 0;

  function stop() {
    if (timer) clearInterval(timer);
    timer = null;
  }

  /** A believable HUD state: both dogs walk toward their lines with a wobble. */
  function fakeState(setup, stepNow, phase) {
    const g = config.GAMES[setup.game];
    const seats = g.seats;
    const dirOf = (seat) => (seat === seats[0] ? +1 : -1);
    const spawn = g.spawn || {};
    const prog = Math.min(1, stepNow / 260);
    const mk = (seat, lead) => {
      const sp = spawn[seat] || { x: 0, y: 0, yaw: 0 };
      const d = dirOf(seat);
      const defends = setup.game === 'asym' && seat === 'defender';
      const travel = defends ? 0.5 * prog : (2.0 + lead) * prog;
      const x = sp.x + d * travel;
      const y = sp.y + Math.sin(stepNow / 41 + (d > 0 ? 0 : 2)) * 0.22;
      return {
        seat,
        role: seat,
        dir: d,
        x,
        y,
        yaw: (sp.yaw || 0) + Math.sin(stepNow / 60) * 0.12,
        lineRemaining: g.lineX - d * x,
        speed: defends ? 0.6 : 1.6 + 0.3 * Math.sin(stepNow / 30),
        fallen: false,
        oob: false,
        filterActive: setup.filter && setup.filter.ai && seat === setup.aiSeat && stepNow % 90 < 26,
      };
    };
    return {
      phase,
      step: stepNow,
      stepsTotal: g.episodeSteps,
      timeLeftS: Math.max(0, (g.episodeSteps - stepNow) * config.PHYS.controlDt),
      countdownS: phase === 'countdown' ? Math.max(0, 3 - (performance.now() - t0) / 1000) : null,
      cmd: ui.input.read(),
      camera,
      fps: 60,
      stepMs: 3.4,
      you: mk(setup.playerSeat, 0.35),
      ai: mk(setup.aiSeat, 0),
    };
  }

  function runFakeMatch(setup) {
    if (!setup) return;
    lastSetup = setup;
    stop();
    ui.loading.begin([
      { id: 'wasm', label: 'MuJoCo WebAssembly' },
      { id: 'scene', label: `Scene · ${setup.game}` },
      { id: 'walk', label: 'Player walk policy' },
      { id: 'opp', label: `Opponent · ${setup.opponent ? setup.opponent.display : '?'}` },
      { id: 'render', label: 'Renderer' },
    ]);
    const tasks = ['wasm', 'scene', 'walk', 'opp', 'render'];
    const details = {
      wasm: '1.26 MB gz',
      scene: 'meshes + keyframe',
      walk: setup.playerWalk ? setup.playerWalk.name : 'walk',
      opp: setup.opponent ? setup.opponent.name : '',
      render: 'arena',
    };
    let i = 0;
    const tick = () => {
      if (i > 0) ui.loading.update(tasks[i - 1], { state: 'done', detail: details[tasks[i - 1]] });
      if (i >= tasks.length) return begin();
      ui.loading.update(tasks[i], { state: 'active' });
      ui.loading.progress(i / tasks.length, `Loading ${details[tasks[i]] || tasks[i]}…`);
      i += 1;
      setTimeout(tick, 220);
    };
    tick();

    function begin() {
      ui.startMatch({ game: setup.game, playerSeat: setup.playerSeat, opponent: setup.opponent });
      stageNote(ui.canvas, NOTE);
      step = 0;
      t0 = performance.now();
      paused = false;
      timer = setInterval(() => {
        if (paused) return;
        const phase = performance.now() - t0 < 3000 ? 'countdown' : 'running';
        if (phase === 'running') step += 1;
        ui.updateHud(fakeState(setup, step, phase));
        if (step >= 260) {
          stop();
          ui.showResult({
            terminal: setup.game === 'sym' ? 'touchdown' : 'touchdown',
            winner: 'A',
            stats: { steps: step, timeS: step * config.PHYS.controlDt, closest: 0.0 },
          });
        }
      }, 1000 / 60);
    }
  }

  /* ---- hooks for tests/shoot_ui.mjs (WebDriver executeScript) ---- */
  globalThis.__s2cPreview = {
    ui,
    config,
    usingRealConfig: real,
    note: NOTE,
    /** Jump straight to the game view with a frozen HUD frame. */
    showGame(opts = {}) {
      stop();
      const setup = { ...ui.getSetup(), ...opts };
      lastSetup = setup;
      ui.startMatch({ game: setup.game, playerSeat: setup.playerSeat, opponent: setup.opponent });
      stageNote(ui.canvas, NOTE);
      t0 = performance.now() - 10000;
      const s = fakeState(setup, opts.step == null ? 132 : opts.step, opts.phase || 'running');
      if (opts.cmd) s.cmd = opts.cmd;
      if (opts.countdownS != null) {
        s.phase = 'countdown';
        s.countdownS = opts.countdownS;
      }
      if (opts.filterActive) s.ai.filterActive = true;
      ui.updateHud(s);
      return s;
    },
    showLoading() {
      const setup = ui.getSetup();
      ui.loading.begin([
        { id: 'wasm', label: 'MuJoCo WebAssembly' },
        { id: 'scene', label: `Scene · ${setup.game}` },
        { id: 'walk', label: 'Player walk policy' },
        { id: 'opp', label: `Opponent · ${setup.opponent ? setup.opponent.display : '?'}` },
        { id: 'render', label: 'Renderer' },
      ]);
      ui.loading.update('wasm', { state: 'done', detail: '1.26 MB gz' });
      ui.loading.update('scene', { state: 'done', detail: '430 KB gz' });
      ui.loading.update('walk', { state: 'done', detail: '761 KB' });
      ui.loading.update('opp', { state: 'active', detail: '788 KB' });
      ui.loading.progress(0.62, 'Loading opponent weights…');
    },
    showResult(result) {
      ui.showResult(result);
    },
    stop,
  };

  stageNote(ui.canvas, NOTE);
  console.info('[ui-preview] ready. window.__s2cPreview exposes showGame/showLoading/showResult.');
  return ui;
}

export default boot;

/**
 * app/ui.js — the whole interface: title/setup, loading, in-game HUD, pause, result, settings,
 * about. No framework, no build step, no runtime fetch outside this origin.
 *
 * =============================================================================================
 *  THE API app/main.js USES
 * =============================================================================================
 *
 *   import { createUI } from './ui.js';
 *
 *   const ui = await createUI({
 *     mount:        document.getElementById('app'),   // default: #app, else <body>
 *     config:       configModule,                     // default: await import('./config.js')
 *     manifestUrl:  'assets/policies/manifest.json',  // default
 *     capabilities: { filter: false },                // what the engine can actually do today
 *     cameraModes:  ['chase', 'orbit', 'broadcast'],  // owned by app/render.js
 *
 *     onStart:       (setup) => {},   // the player pressed Start. See SETUP below.
 *     onRestart:     () => {},        // R, or Restart on the result/pause card
 *     onChangeSetup: () => {},        // back to the title screen; tear the match down
 *     onPause:       () => {},        // Esc while running
 *     onResume:      () => {},        // Esc while paused, or Resume
 *     onCamera:      (mode) => {},    // C, or the camera control
 *     onSetting:     (key, value, all) => {},  // any settings-panel change
 *     onResize:      () => {},        // the stage element changed size
 *   });
 *
 *   ui.canvas         // <canvas> for createRenderer(canvas, sim, game) — UI owns the DOM
 *   ui.input          // the createInput() instance, already wired to the settings panel
 *   ui.settings       // live settings object (scheme, sensitivity, cameraDistance, quality, ...)
 *
 *   ui.loading.begin([{id,label}, ...])          // show the loading screen with a checklist
 *   ui.loading.update(id, {state, detail, value})// state: pending | active | done | fail
 *   ui.loading.progress(0..1, 'label')           // the overall bar
 *   ui.loading.fail('message')                   // stop, show the error, offer Back
 *
 *   ui.startMatch({ game, playerSeat, opponent })// switch to the game view + arm the HUD
 *   ui.updateHud(state)                          // every frame; shape documented in app/hud.js
 *   ui.showResult(result)                        // { terminal, winner, ... } from app/referee.js
 *   ui.hideResult(); ui.setPaused(bool); ui.setCamera(mode); ui.toast(msg); ui.fatal(msg)
 *   ui.showTitle()                               // back to setup
 *
 * SETUP (what onStart receives):
 *   {
 *     game: 'sym' | 'asym',
 *     playerSeat: 'A' | 'attacker' | 'defender',
 *     aiSeat:     the other seat,
 *     opponent: { method, name, display, bin, json, obs_dim, ... },   // the manifest row
 *     playerWalk: { name, bin, json, ... },                           // the manifest walk row
 *     filter: { ai: bool, you: bool },
 *     settings: { ...ui.settings }
 *   }
 *   Nothing here is a hardcoded checkpoint path: every policy field comes from
 *   assets/policies/manifest.json (DESIGN.md section 3.5 forbids hardcoded paths).
 *
 * DEEP LINKS (for testing):  ?game=sym&role=A&opponent=s2c&autostart=1
 *   also: &filter=ai|you|both|off  &camera=chase  &scheme=strafe|steer  &uipreview=1
 *
 * =============================================================================================
 *  WHAT THIS FILE IS AND IS NOT ALLOWED TO KNOW
 * =============================================================================================
 *   Field geometry, line position, episode length and the command box come from app/config.js.
 *   The opponent roster comes from assets/policies/manifest.json.
 *   The verdict vocabulary is the live TerminationManager set — touchdown, touchdown_def,
 *   trunk_contact, fell/attacker_fell/defender_fell, oob/..._oob, time_out
 *   (recon/05_env_physics_contract.md:250-254 asym, section 6 sym; assets/scene/<game>/scene.json
 *   rules.terminationTerms). This file translates those tokens into English and never invents a
 *   new one; the raw token is always shown next to the plain-words verdict.
 *   Who wins on which token is the payoff in src/tasks/game/mdp/rewards.py:226-227 (asym) and
 *   src/tasks/sym_game/mdp/sym.py:334-340 (sym): asym time_out = the defender wins, sym
 *   time_out = a draw; the collision initiator loses (0.12 s EMA of closing speed,
 *   rewards.py:149-196).
 */

import { createHud, h, SIDE_COLORS, FIELD_COLORS } from './hud.js';
import { createInput, SCHEMES, SCHEME_ALIAS } from './input.js';

/* ------------------------------------------------------------------ config resolution */

const OVERRIDE = globalThis.S2C_CONFIG_OVERRIDE || null;

async function resolveConfig(injected) {
  if (injected) return injected;
  if (OVERRIDE) return OVERRIDE;
  return await import('./config.js');
}

/* ------------------------------------------------------------------ static copy (ours) */

/** One line per method. Descriptive only — no numbers claimed that are not in recon/. */
const METHOD_BLURB = {
  s2c: 'Ours — runs behind the safety certificate.',
  et: 'Baseline — early termination.',
  nom: 'Baseline — safety penalty.',
  cpo: 'Baseline — constrained policy optimisation.',
  lag: 'Baseline — Lagrangian.',
};

const METHOD_ORDER = ['s2c', 'et', 'nom', 'cpo', 'lag'];

const GAME_COPY = {
  sym: {
    name: 'Symmetric',
    tag: 'race',
    line: 'Both dogs attack. First reference point past its own line wins.',
    detail:
      'You take seat A and score at x = +1.9 m; the AI takes seat B and scores at x = −1.9 m. ' +
      'Time out is a draw.',
  },
  asym: {
    name: 'Asymmetric',
    tag: 'attack / defend',
    line: 'One attacker, one defender, ten seconds on the clock.',
    detail:
      'The attacker must put its trunk centre past x = +1.9 m before the clock runs out. ' +
      'The defender wins on time out.',
  },
};

const ROLE_COPY = {
  attacker: { name: 'Attacker', line: 'Get past the line before the clock dies.' },
  defender: { name: 'Defender', line: 'Survive ten seconds without letting it through.' },
};

const SETTINGS_KEY = 's2c.web_play.settings.v1';

const DEFAULT_SETTINGS = {
  scheme: 'strafe',
  sensitivity: 1.0,
  cameraDistance: 1.0,
  quality: 'high',
  gamepad: true,
  touchControls: 'auto',
};

/* ---------------------------------------------------------------------------- helpers */

function loadSettings() {
  const out = { ...DEFAULT_SETTINGS };
  try {
    const raw = globalThis.localStorage && globalThis.localStorage.getItem(SETTINGS_KEY);
    if (raw) Object.assign(out, JSON.parse(raw));
  } catch {
    /* private mode / blocked storage: defaults are fine */
  }
  out.scheme = SCHEME_ALIAS[out.scheme] || out.scheme;
  if (!SCHEMES[out.scheme]) out.scheme = SCHEMES[DEFAULT_SETTINGS.scheme] ? DEFAULT_SETTINGS.scheme : Object.keys(SCHEMES)[0];
  return out;
}

function saveSettings(s) {
  try {
    if (globalThis.localStorage) globalThis.localStorage.setItem(SETTINGS_KEY, JSON.stringify(s));
  } catch {
    /* ignore */
  }
}

function seatLabel(game, seat) {
  if (game === 'sym') return `seat ${seat}`;
  return seat;
}

/**
 * app/referee.js reports `winner` as 'A' | 'B' | 'DRAW' (the frozen interface) and also
 * publishes `winnerSeat` / `loserSeat` in the game's own seat names. Normalise either to a seat.
 */
function asSeat(config, game, who) {
  if (!who) return null;
  const seats = config.GAMES[game].seats;
  if (who === 'DRAW' || who === 'draw') return 'DRAW';
  if (seats.includes(who)) return who;
  if (who === 'A') return seats[0];
  if (who === 'B') return seats[1];
  return null;
}

/* --------------------------------------------------------------------------- verdicts */

/**
 * Plain words for a referee terminal, from the player's point of view.
 * `result` is app/referee.js's `{ terminal, winner }` plus anything else it chooses to publish
 * (`loser`, `culprit`, `initiator` are used when present and never required).
 */
export function verdictText(config, game, playerSeat, aiSeat, result) {
  const term = (result && result.terminal) || 'unknown';
  const w = asSeat(config, game, result && (result.winnerSeat || result.winner));
  const youWon = w === playerSeat;
  const draw = w === 'DRAW';
  const tone = draw ? 'draw' : youWon ? 'win' : w ? 'lose' : 'draw';

  // Which side the token is about, when the token names one. app/referee.js collapses the
  // per-seat term names (attacker_fell/defender_oob/...) into the six frozen terminals and
  // publishes the side separately as loserSeat, so both spellings are handled.
  let subject = asSeat(
    config,
    game,
    result && (result.culprit || result.loserSeat || result.loser || result.subject)
  );
  if (!subject) {
    if (/^attacker_/.test(term)) subject = game === 'sym' ? config.GAMES[game].seats[0] : 'attacker';
    else if (/^defender_/.test(term)) subject = game === 'sym' ? config.GAMES[game].seats[1] : 'defender';
  }
  const base = term.replace(/^(attacker|defender)_/, '');
  const subjectIsYou = subject && subject !== 'DRAW' ? subject === playerSeat : !youWon;
  const they = 'the AI';

  let headline;
  let detail;

  switch (base) {
    case 'touchdown':
    case 'touchdown_def': {
      const scorer =
        base === 'touchdown_def' ? config.GAMES[game].seats[1] : config.GAMES[game].seats[0];
      const scorerIsYou = game === 'asym' ? playerSeat === 'attacker' : scorer === playerSeat;
      headline = scorerIsYou ? 'Touchdown. You scored.' : `Touchdown. ${they} scored.`;
      detail =
        game === 'sym'
          ? 'Trunk centre past its own scoring line first.'
          : 'The attacker put its trunk centre past x = +1.9 m before the clock ran out.';
      break;
    }
    case 'trunk_contact':
      headline = youWon
        ? `${they} initiated the collision — you win.`
        : draw
          ? 'Collision.'
          : 'You initiated the collision — you lose.';
      detail =
        'Any robot-on-robot contact ends the episode; the side closing faster at the moment of ' +
        'contact is at fault (0.12 s EMA of the closing speed).';
      break;
    case 'fell':
      headline = subjectIsYou ? 'You fell over — you lose.' : `${they} fell over. You win.`;
      detail = 'Trunk tilted more than 70° from upright.';
      break;
    case 'oob':
      headline = subjectIsYou
        ? 'You left the field — you lose.'
        : `${they} left the field. You win.`;
      detail = 'The trunk centre crossed the field rectangle. There are no walls to stop you.';
      break;
    case 'time_out':
      if (game === 'sym') {
        headline = 'Time out. Draw.';
        detail = 'Nobody crossed in ten seconds, so the symmetric game scores it a draw.';
      } else {
        headline = youWon ? 'Time out. You held the line.' : 'Time out. The defender held.';
        detail = 'Ten seconds, 500 control steps. In the asymmetric game the defender wins the clock.';
      }
      break;
    default:
      headline = draw ? 'Draw.' : youWon ? 'You win.' : 'You lose.';
      detail = 'Episode ended.';
  }
  return { headline, detail, tone, term, winner: w, youWon, draw };
}

/* ============================================================================== createUI */

export async function createUI(options = {}) {
  const config = await resolveConfig(options.config);
  if (!config || !config.GAMES || !config.CMD_BOX) {
    throw new Error('app/ui.js: app/config.js must export GAMES, PHYS and CMD_BOX.');
  }
  const GAMES = config.GAMES;
  const mount =
    options.mount || document.getElementById('app') || document.body;
  const params = new URLSearchParams(globalThis.location ? globalThis.location.search : '');
  const capabilities = { filter: false, ...(options.capabilities || {}) };
  const cameraModes =
    options.cameraModes || (config.CAMERA && config.CAMERA.modes) || ['chase', 'broadcast', 'fpv'];
  /** Which games this build offers. The engine supports both; the shell decides
   *  what ships (app/main.js passes `games`). Unknown ids are ignored. */
  const offeredGames = (options.games || Object.keys(GAMES)).filter((id) => GAMES[id]);
  if (!offeredGames.length) throw new Error('app/ui.js: no offered game exists in config.GAMES');
  const cb = (name, ...args) => {
    const fn = options[name];
    if (typeof fn === 'function') {
      try {
        return fn(...args);
      } catch (err) {
        console.error(`app/ui.js: ${name} threw`, err);
      }
    }
    return undefined;
  };

  /* -------------------------------------------------------------- manifest */

  let manifest = options.manifest || null;
  let manifestError = null;
  if (!manifest) {
    const url = options.manifestUrl || 'assets/policies/manifest.json';
    try {
      let res = null;
      for (let i = 0; i < 4 && !(res && res.ok); i++) {
        if (i) await new Promise((r) => setTimeout(r, [200, 600, 1500][Math.min(i - 1, 2)]));
        try { res = await fetch(url, { cache: 'no-cache' }); } catch { res = null; }
        if (res && !res.ok && res.status < 500 && res.status !== 429) break;
      }
      if (!res) throw new Error(`fetch ${url} failed`);
      if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
      manifest = await res.json();
    } catch (err) {
      manifestError = err;
      manifest = null;
    }
  }
  const policyByName = new Map();
  if (manifest && Array.isArray(manifest.policies)) {
    for (const p of manifest.policies) policyByName.set(p.name, p);
  }

  // The certificate is a capability of the BUILD (app/filter.js implements the
  // law) *and* of the ASSETS (tools/export_filter.py has to have written the
  // weights). The shell claims the first; only the manifest can confirm the
  // second, so a build whose assets/policies/ has no `filter` block falls back
  // to the greyed-out checkboxes rather than failing on Start.
  if (capabilities.filter && manifest?.filter?.available !== true) {
    capabilities.filter = false;
  }

  /** The roster for one game + seat, ordered by METHOD_ORDER, as manifest rows. */
  function roster(game, aiSeat) {
    if (!manifest || !manifest.games || !manifest.games[game]) return [];
    const names = (manifest.games[game].seats || {})[aiSeat] || [];
    const rows = names.map((n) => policyByName.get(n)).filter(Boolean);
    rows.sort((a, b) => METHOD_ORDER.indexOf(a.method) - METHOD_ORDER.indexOf(b.method));
    return rows;
  }

  /* -------------------------------------------------------------- settings + input */

  const settings = loadSettings();
  const input = createInput(globalThis, {
    cmdBox: config.CMD_BOX,
    scheme: settings.scheme,
    sensitivity: settings.sensitivity,
    gamepad: settings.gamepad,
  });
  input.setEnabled(false); // menus first

  /* -------------------------------------------------------------- selection state */

  const state = {
    phase: 'setup', // setup | loading | game
    game: offeredGames[0],
    playerSeat: defaultSeat(offeredGames[0]),
    method: 's2c',
    filter: { ai: false, you: false },
    paused: false,
    camera: (config.CAMERA && config.CAMERA.default) || cameraModes[0],
    result: null,
  };

  function aiSeatFor(game, playerSeat) {
    return GAMES[game].seats.find((s) => s !== playerSeat);
  }
  function defaultSeat(game) {
    return game === 'sym' ? 'A' : 'attacker';
  }

  /* ============================================================ DOM: shell */

  const canvas = h('canvas.stage-canvas', { id: 'stage' });
  const stage = h('div.stage', null, [canvas]);

  const hud = createHud({ config, cmdBox: config.CMD_BOX });
  stage.appendChild(hud.el);

  const overlays = h('div.overlays');
  const root = h('div.ui', null, [stage, overlays]);
  mount.appendChild(root);

  /* ============================================================ DOM: title / setup */

  const gameCards = {};
  const roleRow = h('div.choice-row');
  const roleSection = h('section.setup-step', null, [
    h('h2.step-title', null, [h('span.step-n', { text: '2' }), 'Pick your side']),
    roleRow,
  ]);
  const oppGrid = h('div.opp-grid');
  const oppNote = h('p.step-note', { text: '' });


  const startBtn = h('button.btn.btn-primary.btn-start', { type: 'button' }, ['Start match']);
  const startHint = h('p.start-hint', { text: '' });

  function gameCard(id) {
    const copy = GAME_COPY[id];
    const g = GAMES[id];
    const card = h('button.choice.choice-game', { type: 'button', 'data-game': id }, [
      h('div.choice-head', null, [
        h('span.choice-name', { text: copy.name }),
        h('span.choice-tag', { text: copy.tag }),
      ]),
      fieldDiagram(id),
      h('p.choice-line', { text: copy.line }),
      h('div.choice-facts', null, [
        h('span', { text: `${g.field[0]} × ${g.field[1]} m` }),
        h('span', { text: `line x = ${g.lineX > 0 ? '+' : ''}${g.lineX} m` }),
        h('span', { text: `${g.episodeSteps} steps · ${(g.episodeSteps * config.PHYS.controlDt).toFixed(0)} s` }),
      ]),
    ]);
    card.addEventListener('click', () => selectGame(id));
    gameCards[id] = card;
    return card;
  }

  /** A to-scale SVG of the field: the rectangle, the scoring line(s), the end zone(s), spawns. */
  function fieldDiagram(id) {
    const g = GAMES[id];
    const [fw, fh] = g.field;
    const [cx, cy] = g.center;
    const pad = 0.25;
    const vbW = fw + pad * 2;
    const vbH = fh + pad * 2;
    const NS = 'http://www.w3.org/2000/svg';
    const svg = document.createElementNS(NS, 'svg');
    svg.setAttribute('viewBox', `${cx - fw / 2 - pad} ${-cy - fh / 2 - pad} ${vbW} ${vbH}`);
    svg.setAttribute('class', 'field-svg');
    svg.setAttribute('role', 'img');
    svg.setAttribute('aria-label', `${GAME_COPY[id].name} field, ${fw} by ${fh} metres`);
    const mk = (tag, attrs) => {
      const e = document.createElementNS(NS, tag);
      for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, String(v));
      svg.appendChild(e);
      return e;
    };
    mk('rect', {
      x: cx - fw / 2, y: -cy - fh / 2, width: fw, height: fh,
      fill: FIELD_COLORS.surface, stroke: FIELD_COLORS.boundary, 'stroke-width': 0.035,
    });
    const lines = id === 'sym' ? [{ x: g.lineX, dir: 1 }, { x: -g.lineX, dir: -1 }] : [{ x: g.lineX, dir: 1 }];
    for (const ln of lines) {
      const edge = ln.dir > 0 ? cx + fw / 2 : cx - fw / 2;
      mk('rect', {
        x: Math.min(ln.x, edge), y: -cy - fh / 2, width: Math.abs(edge - ln.x), height: fh,
        fill: FIELD_COLORS.zone,
      });
      mk('line', {
        x1: ln.x, y1: -cy - fh / 2, x2: ln.x, y2: -cy + fh / 2,
        stroke: FIELD_COLORS.line, 'stroke-width': 0.05,
      });
    }
    mk('line', {
      x1: cx, y1: -cy - fh / 2, x2: cx, y2: -cy + fh / 2,
      stroke: FIELD_COLORS.midline, 'stroke-width': 0.02,
    });
    const seats = GAMES[id].seats;
    const spawn = GAMES[id].spawn || {};
    seats.forEach((seat, i) => {
      const sp = spawn[seat];
      if (!sp) return;
      const color = i === 0 ? SIDE_COLORS.player.bright : SIDE_COLORS.ai.bright;
      mk('circle', { cx: sp.x, cy: -sp.y, r: 0.11, fill: color, opacity: 0.95 });
      mk('line', {
        x1: sp.x, y1: -sp.y,
        x2: sp.x + Math.cos(sp.yaw || 0) * 0.34, y2: -(sp.y + Math.sin(sp.yaw || 0) * 0.34),
        stroke: color, 'stroke-width': 0.05, opacity: 0.95,
      });
    });
    return svg;
  }

  function roleCard(seat) {
    const copy = ROLE_COPY[seat] || { name: seat, line: '' };
    const card = h('button.choice.choice-role', { type: 'button', 'data-role': seat }, [
      h('div.choice-head', null, [h('span.choice-name', { text: copy.name })]),
      h('p.choice-line', { text: copy.line }),
    ]);
    card.addEventListener('click', () => {
      state.playerSeat = seat;
      syncSetup();
    });
    return card;
  }

  function oppCard(row) {
    const isOurs = row.method === 's2c';
    const card = h('button.choice.choice-opp', { type: 'button', 'data-method': row.method }, [
      h('div.choice-head', null, [
        h('span.choice-name', { text: row.display || row.method }),
        isOurs ? h('span.choice-tag.tag-ours', { text: 'ours' }) : null,
      ]),
      h('p.choice-line', { text: METHOD_BLURB[row.method] || '' }),
      h('div.choice-facts', null, [
        h('span.mono', { text: row.name }),
        h('span.mono', { text: `${row.obs_dim}→${row.act_dim}` }),
      ]),
    ]);
    card.addEventListener('click', () => {
      state.method = row.method;
      syncSetup();
    });
    return card;
  }

  // built before setupScreen because controlsCard() fills it in
  const controlsList = h('ul.controls-list');

  const setupScreen = h('section.screen.screen-setup', null, [
    h('header.hero', null, [
      h('div.hero-mark', null, [h('span.hero-dot'), h('span.hero-rule')]),
      h('h1.hero-title', null, ['S2C ', h('span.hero-thin', { text: 'Web Play' })]),
      h('p.hero-sub', {
        text:
          'Drive a Unitree Go2 against one of our trained policies. Real MuJoCo physics in the ' +
          'browser, the real checkpoint weights, the same referee as the simulator.',
      }),
      h('p.hero-org', null, [
        'JHU Alliance Lab',
        h('span.dot-sep', { text: '·' }),
        h('a.link', { href: 'https://github.com/JHU-AllianceLab', target: '_blank', rel: 'noopener' }, ['GitHub']),
      ]),
    ]),
    h('div.setup-grid', null, [
      h('div.setup-main', null, [
        h('section.setup-step', null, [
          h('h2.step-title', null, [h('span.step-n', { text: '1' }), 'Pick the game']),
          h('div.choice-row.choice-row-games', null, offeredGames.map(gameCard)),
        ]),
        roleSection,
        h('section.setup-step', null, [
          h('h2.step-title', null, [h('span.step-n', { text: '3' }), 'Pick the opponent']),
          oppGrid,
          oppNote,
        ]),
        h('div.start-row', null, [startBtn, startHint]),
      ]),
      h('aside.setup-side', null, [controlsCard()]),
    ]),
  ]);

  function keycap(t) {
    return h('kbd.key', { text: t });
  }

  function controlsCard() {
    return h('section.card.card-controls', null, [
      h('h3.card-title', { text: 'Controls' }),
      controlsList,
      h('div.controls-extra', null, [
        h('div.control-row', null, [
          h('span.control-keys', null, [keycap('Esc')]),
          h('span.control-what', { text: 'pause' }),
        ]),
        h('div.control-row', null, [
          h('span.control-keys', null, [keycap('R')]),
          h('span.control-what', { text: 'restart' }),
        ]),
        h('div.control-row', null, [
          h('span.control-keys', null, [keycap('C')]),
          h('span.control-what', { text: 'camera' }),
        ]),
      ]),
    ]);
  }

  function renderControlsList() {
    controlsList.textContent = '';
    for (const row of input.bindings.rows) {
      const keys = h('span.control-keys', null, row.keys.map(keycap));
      const range = row.range
        ? h('span.control-range', { text: `${row.range[0]} … ${row.range[1]} ${row.unit}` })
        : null;
      controlsList.appendChild(
        h('li.control-row', null, [keys, h('span.control-what', { text: row.label }), range])
      );
    }
  }

  function aboutCard() {
    return h('section.card.card-about', null, [
      h('h3.card-title', { text: 'What is real here' }),
      h('ul.fact-list', null, [
        h('li', { text: 'MuJoCo 3 compiled to WebAssembly, 0.005 s steps, decimation 4, 50 Hz control — the training solver settings.' }),
        h('li', { text: 'The opponent is the exported checkpoint, not a scripted bot.' }),
        h('li', { text: 'The referee reproduces the training termination set, including the collision-initiator rule.' }),
        h('li', { text: 'The field has no walls. Out of bounds is a rule, not a fence.' }),
      ]),
      h('button.btn.btn-ghost.btn-small', { type: 'button', onclick: () => openPanel(aboutPanel) }, ['Read the differences']),
    ]);
  }

  /* ============================================================ DOM: loading */

  const loadList = h('ul.load-list');
  const loadBar = h('i.load-bar-fill');
  const loadLabel = h('p.load-label', { text: 'Loading…' });
  const loadErr = h('div.load-error');
  loadErr.hidden = true;
  const loadingScreen = h('section.screen.screen-loading', null, [
    h('div.load-card', null, [
      h('h2.load-title', { text: 'Building the match' }),
      loadLabel,
      h('div.load-bar', null, [loadBar]),
      loadList,
      loadErr,
      h('p.load-note', {
        text:
          'First load pulls the physics engine and the policy weights — a few megabytes. ' +
          'They are cached after that.',
      }),
    ]),
  ]);
  const loadRows = new Map();

  /* ============================================================ DOM: overlays */

  function overlayCard(cls, children) {
    const card = h(`div.overlay-card.${cls}`, null, children);
    const wrap = h('div.overlay', null, [card]);
    wrap.hidden = true;
    overlays.appendChild(wrap);
    return wrap;
  }

  // --- pause
  const pauseOverlay = overlayCard('card-pause', [
    h('h2.overlay-title', { text: 'Paused' }),
    h('p.overlay-sub', { text: 'The simulation is frozen. Nothing is being stepped.' }),
    h('div.overlay-actions', null, [
      h('button.btn.btn-primary', { type: 'button', onclick: () => setPaused(false) }, ['Resume']),
      h('button.btn', { type: 'button', onclick: () => doRestart() }, ['Restart']),
      h('button.btn.btn-ghost', { type: 'button', onclick: () => openPanel(settingsPanel) }, ['Settings']),
      h('button.btn.btn-ghost', { type: 'button', onclick: () => backToSetup() }, ['Change setup']),
    ]),
  ]);

  // --- result
  const resultTone = h('div.result-tone');
  const resultHead = h('h2.overlay-title.result-head', { text: '' });
  const resultSub = h('p.overlay-sub', { text: '' });
  const resultTerm = h('code.result-term', { text: '' });
  const resultStats = h('div.result-stats');
  const resultOverlay = overlayCard('card-result', [
    resultTone,
    resultHead,
    resultSub,
    h('div.result-meta', null, [h('span.result-meta-label', { text: 'referee' }), resultTerm]),
    resultStats,
    h('div.overlay-actions', null, [
      h('button.btn.btn-primary', { type: 'button', onclick: () => doRestart() }, ['Restart', h('kbd.key.key-inline', { text: 'R' })]),
      h('button.btn.btn-ghost', { type: 'button', onclick: () => backToSetup() }, ['Change setup']),
    ]),
  ]);

  // --- fatal
  const fatalMsg = h('p.overlay-sub.fatal-msg', { text: '' });
  const fatalOverlay = overlayCard('card-fatal', [
    h('h2.overlay-title', { text: 'That did not load' }),
    fatalMsg,
    h('div.overlay-actions', null, [
      h('button.btn.btn-ghost', { type: 'button', onclick: () => backToSetup() }, ['Back to setup']),
    ]),
  ]);

  /* ---- settings panel ---- */

  function segmented(name, items, get, set) {
    const wrap = h('div.segmented', { role: 'radiogroup', 'aria-label': name });
    const buttons = [];
    for (const it of items) {
      const b = h('button.seg', { type: 'button', role: 'radio', 'data-value': it.id, title: it.hint || '' }, [it.label]);
      b.addEventListener('click', () => {
        set(it.id);
        sync();
      });
      buttons.push(b);
      wrap.appendChild(b);
    }
    function sync() {
      const v = get();
      for (const b of buttons) {
        const on = b.dataset.value === String(v);
        b.classList.toggle('on', on);
        b.setAttribute('aria-checked', on ? 'true' : 'false');
      }
    }
    sync();
    return { el: wrap, sync };
  }

  function slider(label, min, max, step, get, set, format) {
    const out = h('span.range-val', { text: format(get()) });
    const inp = h('input.range', { type: 'range', min, max, step, value: get() });
    inp.addEventListener('input', () => {
      set(Number(inp.value));
      out.textContent = format(Number(inp.value));
    });
    const row = h('div.field', null, [
      h('div.field-head', null, [h('label.field-label', { text: label }), out]),
      inp,
    ]);
    return { el: row, sync: () => { inp.value = get(); out.textContent = format(get()); } };
  }

  const schemeSeg = segmented(
    'Control scheme',
    Object.keys(SCHEMES).map((id) => ({ id, label: SCHEMES[id].label, hint: SCHEMES[id].hint })),
    () => settings.scheme,
    (v) => {
      settings.scheme = v;
      input.setScheme(v);
      renderControlsList();
      persist('scheme', v);
    }
  );
  const schemeHint = h('p.field-hint', { text: SCHEMES[settings.scheme].hint });

  const sensSlider = slider('Steering sensitivity', 0.4, 1.6, 0.05,
    () => settings.sensitivity,
    (v) => { settings.sensitivity = v; input.setSensitivity(v); persist('sensitivity', v); },
    (v) => `${v.toFixed(2)}×`);

  const camSlider = slider('Camera distance', 0.6, 1.8, 0.05,
    () => settings.cameraDistance,
    (v) => { settings.cameraDistance = v; persist('cameraDistance', v); },
    (v) => `${v.toFixed(2)}×`);

  const qualitySeg = segmented(
    'Quality',
    [
      { id: 'low', label: 'Low', hint: 'no shadows, no bloom' },
      { id: 'medium', label: 'Medium' },
      { id: 'high', label: 'High' },
    ],
    () => settings.quality,
    (v) => { settings.quality = v; persist('quality', v); }
  );

  const padToggle = h('input', { type: 'checkbox', id: 'opt-gamepad' });
  padToggle.checked = settings.gamepad;
  padToggle.addEventListener('change', () => {
    settings.gamepad = padToggle.checked;
    input.setGamepadEnabled(padToggle.checked);
    persist('gamepad', padToggle.checked);
  });

  const settingsPanel = panel('Settings', [
    h('div.field', null, [
      h('div.field-head', null, [h('label.field-label', { text: 'Control scheme' })]),
      schemeSeg.el,
      schemeHint,
    ]),
    sensSlider.el,
    camSlider.el,
    h('div.field', null, [
      h('div.field-head', null, [h('label.field-label', { text: 'Quality' })]),
      qualitySeg.el,
      h('p.field-hint', { text: 'Visual only. Changing it must not change the physics.' }),
    ]),
    h('label.opt', { for: 'opt-gamepad' }, [padToggle, h('span', { text: 'Use a gamepad when one is connected' })]),
    h('p.field-hint', {
      text: `Commands are always clamped to the trained box: vx ${config.CMD_BOX.vx[0]} to ${config.CMD_BOX.vx[1]} m/s, ` +
        `vy ±${config.CMD_BOX.vy[1]} m/s, ωz ±${config.CMD_BOX.wz[1]} rad/s.`,
    }),
  ]);

  const aboutPanel = panel('Honest differences', [
    h('p.panel-lead', {
      text:
        'Everything below is a real difference between this page and the simulator the paper ' +
        'reports. None of it is hidden behind a "close enough" implementation.',
    }),
    h('ul.fact-list', null, [
      h('li', { text: 'Training ran on GPU MuJoCo; this is the CPU C implementation compiled to WebAssembly. The contact solve is not bit-identical, so the same start diverges over a few seconds.' }),
      h('li', { text: 'The visual meshes are decimated for download size. Collision is 23 primitives per robot and is untouched — the exporter proves qpos is bit-identical with and without the decoration.' }),
      h('li', { text: 'The opponent trained against another policy, not against a human. You will find behaviour it never saw.' }),
      h('li', { text: 'S2C without the certificate is not S2C. The 60-D network is only half of it; with the shield off you are driving the unshielded policy.' }),
      h('li', { text: 'The field has no walls, in the simulator and here. Leaving the rectangle is a referee decision about the trunk centre.' }),
    ]),
  ]);

  function panel(title, children) {
    const body = h('div.panel-body', null, children);
    const card = h('div.panel-card', null, [
      h('div.panel-head', null, [
        h('h2.panel-title', { text: title }),
        h('button.btn-close', { type: 'button', 'aria-label': 'Close', onclick: () => closePanel() }, ['×']),
      ]),
      body,
    ]);
    const wrap = h('div.panel', null, [card]);
    wrap.hidden = true;
    overlays.appendChild(wrap);
    return wrap;
  }

  let openPanelEl = null;
  function openPanel(p) {
    if (openPanelEl) openPanelEl.hidden = true;
    openPanelEl = p;
    p.hidden = false;
  }
  function closePanel() {
    if (openPanelEl) openPanelEl.hidden = true;
    openPanelEl = null;
  }

  function persist(key, value) {
    saveSettings(settings);
    cb('onSetting', key, value, { ...settings });
  }

  /* ---- toasts ---- */
  const toastWrap = h('div.toasts');
  overlays.appendChild(toastWrap);
  function toast(msg, kind = 'info', ms = 2600) {
    const t = h(`div.toast.toast-${kind}`, { text: msg });
    toastWrap.appendChild(t);
    setTimeout(() => {
      t.classList.add('out');
      setTimeout(() => t.remove(), 300);
    }, ms);
  }

  /* ---- in-game top-right buttons ---- */
  const gameBar = h('div.game-bar', null, [
    h('button.btn.btn-icon', { type: 'button', title: 'Camera (C)', onclick: () => cycleCamera() }, ['▣']),
    h('button.btn.btn-icon', { type: 'button', title: 'Settings', onclick: () => openPanel(settingsPanel) }, ['⚙']),
    h('button.btn.btn-icon', { type: 'button', title: 'Pause (Esc)', onclick: () => setPaused(!state.paused) }, ['⏸']),
  ]);
  stage.appendChild(gameBar);

  /* ---- touch controls (bonus) ---- */
  const touchPad = h('div.touch-pad', null, [h('i.touch-knob')]);
  const touchTurnL = h('button.touch-turn.turn-l', { type: 'button', 'aria-label': 'turn left' }, ['↺']);
  const touchTurnR = h('button.touch-turn.turn-r', { type: 'button', 'aria-label': 'turn right' }, ['↻']);
  const touchWrap = h('div.touch-controls', null, [touchPad, h('div.touch-turns', null, [touchTurnL, touchTurnR])]);
  touchWrap.hidden = true;
  stage.appendChild(touchWrap);
  wireTouch();

  function wireTouch() {
    const knob = touchPad.querySelector('.touch-knob');
    let padId = null;
    let axes = { vx: 0, vy: 0, wz: 0 };
    const push = () => input.setExternalAxes(axes.vx || axes.vy || axes.wz ? { ...axes } : null);
    const setFromPoint = (ev) => {
      const r = touchPad.getBoundingClientRect();
      const dx = (ev.clientX - (r.left + r.width / 2)) / (r.width / 2);
      const dy = (ev.clientY - (r.top + r.height / 2)) / (r.height / 2);
      const nx = Math.max(-1, Math.min(1, dx));
      const ny = Math.max(-1, Math.min(1, dy));
      axes.vx = -ny;
      axes.vy = -nx; // +vy is left
      knob.style.transform = `translate(${nx * 34}px, ${ny * 34}px)`;
      push();
    };
    touchPad.addEventListener('pointerdown', (ev) => {
      padId = ev.pointerId;
      touchPad.setPointerCapture(padId);
      setFromPoint(ev);
    });
    touchPad.addEventListener('pointermove', (ev) => {
      if (ev.pointerId === padId) setFromPoint(ev);
    });
    const end = (ev) => {
      if (ev.pointerId !== padId) return;
      padId = null;
      axes.vx = 0;
      axes.vy = 0;
      knob.style.transform = '';
      push();
    };
    touchPad.addEventListener('pointerup', end);
    touchPad.addEventListener('pointercancel', end);
    const turn = (btn, sign) => {
      btn.addEventListener('pointerdown', () => { axes.wz = sign; push(); });
      const stop = () => { axes.wz = 0; push(); };
      btn.addEventListener('pointerup', stop);
      btn.addEventListener('pointerleave', stop);
      btn.addEventListener('pointercancel', stop);
    };
    turn(touchTurnL, +1); // +wz is a left turn
    turn(touchTurnR, -1);
  }

  function touchWanted() {
    if (settings.touchControls === 'on') return true;
    if (settings.touchControls === 'off') return false;
    try {
      return globalThis.matchMedia && globalThis.matchMedia('(pointer: coarse)').matches;
    } catch {
      return false;
    }
  }

  /* ============================================================ screens */

  const screens = h('div.screens', null, [setupScreen, loadingScreen]);
  root.insertBefore(screens, overlays);
  loadingScreen.hidden = true;

  function setPhase(p) {
    state.phase = p;
    setupScreen.hidden = p !== 'setup';
    loadingScreen.hidden = p !== 'loading';
    screens.hidden = p === 'game';
    stage.classList.toggle('stage-live', p === 'game');
    hud.setVisible(p === 'game');
    gameBar.hidden = p !== 'game';
    touchWrap.hidden = !(p === 'game' && touchWanted());
    input.setEnabled(p === 'game' && !state.paused);
    root.dataset.phase = p;
  }

  /* ============================================================ setup logic */

  function selectGame(id) {
    if (!GAMES[id]) return;
    state.game = id;
    state.playerSeat = defaultSeat(id);
    syncSetup();
  }

  function syncSetup() {
    for (const [id, card] of Object.entries(gameCards)) {
      card.classList.toggle('on', id === state.game);
      card.setAttribute('aria-pressed', id === state.game ? 'true' : 'false');
    }

    // roles
    const seats = GAMES[state.game].seats;
    roleRow.textContent = '';
    if (state.game === 'asym') {
      roleSection.hidden = false;
      for (const seat of seats) roleRow.appendChild(roleCard(seat));
      for (const card of roleRow.children) {
        const on = card.dataset.role === state.playerSeat;
        card.classList.toggle('on', on);
        card.setAttribute('aria-pressed', on ? 'true' : 'false');
      }
    } else {
      roleSection.hidden = false;
      state.playerSeat = 'A';
      roleRow.appendChild(
        h('div.fixed-seat', null, [
          h('span.fixed-seat-name', { text: 'You are seat A' }),
          h('p.choice-line', {
            text:
              'Both dogs attack, so the seats are interchangeable: you score at x = +1.9 m, the ' +
              'AI at x = −1.9 m. The shipped pool members are all built for seat B.',
          }),
        ])
      );
    }

    // opponents
    const aiSeat = aiSeatFor(state.game, state.playerSeat);
    const rows = roster(state.game, aiSeat);
    oppGrid.textContent = '';
    if (!rows.length) {
      oppGrid.appendChild(
        h('p.empty', {
          text: manifestError
            ? `Could not read assets/policies/manifest.json (${manifestError.message}).`
            : `No opponent is listed for ${state.game} / ${aiSeat} in the manifest.`,
        })
      );
      state.method = null;
    } else {
      if (!rows.some((r) => r.method === state.method)) state.method = rows[0].method;
      for (const row of rows) oppGrid.appendChild(oppCard(row));
      for (const card of oppGrid.children) {
        const on = card.dataset && card.dataset.method === state.method;
        card.classList.toggle('on', Boolean(on));
        if (card.setAttribute) card.setAttribute('aria-pressed', on ? 'true' : 'false');
      }
      const sel = rows.find((r) => r.method === state.method);
      oppNote.textContent = sel
        ? `${sel.display} plays ${seatLabel(state.game, aiSeat)}. Weights: ${sel.name} (${sel.obs_dim}→${sel.act_dim}).`
        : '';
    }

    // filter — S2C IS its certificate: it always runs behind the Q-CBF stack, and no
    // baseline ever does. There is nothing for the player to toggle (Ray, 2026-09-25).
    state.filter = { ai: capabilities.filter === true && state.method === 's2c', you: false };

    // start button
    const ready = Boolean(state.method);
    startBtn.disabled = !ready;
    startHint.textContent = ready
      ? `${GAME_COPY[state.game].name} · you play ${seatLabel(state.game, state.playerSeat)} · ` +
        `against ${(rows.find((r) => r.method === state.method) || {}).display || ''}`
      : 'Pick an opponent to start.';
  }

  startBtn.addEventListener('click', () => start());

  function currentSetup() {
    const aiSeat = aiSeatFor(state.game, state.playerSeat);
    const rows = roster(state.game, aiSeat);
    const opponent = rows.find((r) => r.method === state.method) || null;
    return {
      game: state.game,
      playerSeat: state.playerSeat,
      aiSeat,
      opponent,
      playerWalk: (manifest && manifest.player_walk) || null,
      filter: { ...state.filter },
      settings: { ...settings },
    };
  }

  function start() {
    const setup = currentSetup();
    if (!setup.opponent) {
      toast('No opponent selected', 'warn');
      return;
    }
    state.result = null;
    hideResult();
    cb('onStart', setup);
  }

  function doRestart() {
    hideResult();
    setPaused(false);
    cb('onRestart');
  }

  function backToSetup() {
    hideResult();
    closePanel();
    pauseOverlay.hidden = true;
    state.paused = false;
    setPhase('setup');
    syncSetup();
    cb('onChangeSetup');
  }

  /* ============================================================ loading */

  const loading = {
    begin(tasks) {
      loadRows.clear();
      loadList.textContent = '';
      loadErr.hidden = true;
      loadBar.style.width = '0%';
      loadLabel.textContent = 'Loading…';
      for (const t of tasks || []) {
        const dot = h('i.load-dot');
        const detail = h('span.load-detail', { text: '' });
        const li = h('li.load-row', { 'data-state': 'pending' }, [
          dot,
          h('span.load-name', { text: t.label || t.id }),
          detail,
        ]);
        loadRows.set(t.id, { li, detail });
        loadList.appendChild(li);
      }
      setPhase('loading');
    },
    update(id, patch = {}) {
      const row = loadRows.get(id);
      if (!row) return;
      if (patch.state) row.li.dataset.state = patch.state;
      if (patch.detail != null) row.detail.textContent = patch.detail;
      const rows = Array.from(loadRows.values());
      const done = rows.filter((r) => r.li.dataset.state === 'done').length;
      loadBar.style.width = ((done / Math.max(1, rows.length)) * 100).toFixed(1) + '%';
    },
    progress(value, label) {
      if (Number.isFinite(value)) loadBar.style.width = (Math.max(0, Math.min(1, value)) * 100).toFixed(1) + '%';
      if (label) loadLabel.textContent = label;
    },
    fail(message) {
      loadErr.hidden = false;
      loadErr.textContent = '';
      loadErr.appendChild(h('p.load-error-msg', { text: String(message) }));
      loadErr.appendChild(
        h('button.btn.btn-ghost.btn-small', { type: 'button', onclick: () => backToSetup() }, ['Back to setup'])
      );
    },
  };

  /* ============================================================ game view */

  function startMatch(info) {
    const game = info && info.game ? info.game : state.game;
    const playerSeat = info && info.playerSeat ? info.playerSeat : state.playerSeat;
    state.game = game;
    state.playerSeat = playerSeat;
    hud.setMatch({
      game,
      playerSeat,
      opponent: (info && info.opponent) || currentSetup().opponent,
    });
    state.result = null;
    hideResult();
    setPaused(false);
    setPhase('game');
    cb('onResize');
  }

  /**
   * app/match.js publishes its own `hud()` shape. Rather than make app/main.js translate, this
   * accepts either that shape or the one documented at the top of app/hud.js.
   *
   *   ui.updateHud(match.hud())                  // positions come from the state if it has them
   *   ui.updateHud(match.hud(), match.state())   // preferred: state() carries x / y / yaw
   *
   * match.hud() fields used here: phase, step, episodeSteps, timeLeft, countdown.msLeft,
   * playerSeat, aiSeat, opponent, lineDistance[seat], cmd, filter[seat], perf.lastStepMs.
   * match.state().seats[seat] supplies x, y, yaw, tilt, vx, vy.
   */
  function adaptHud(raw, extra) {
    if (!raw) return raw;
    if (raw.you || raw.ai) return raw; // already the app/hud.js shape
    if (!raw.playerSeat || !raw.aiSeat) return raw;

    const game = raw.game && GAMES[raw.game] ? raw.game : state.game;
    const seatsState = (extra && extra.seats) || raw.seats || null;
    const fallAngle = (config.REFEREE && config.REFEREE.fallLimitAngle) || Infinity;
    const bounds = typeof config.fieldBounds === 'function' ? config.fieldBounds(game) : null;
    const dist = raw.lineDistance || {};
    const filters = raw.filter || {};

    const side = (seat) => {
      const st = (seatsState && seatsState[seat]) || {};
      const f = filters[seat] || null;
      const speed =
        Number.isFinite(st.vx) && Number.isFinite(st.vy) ? Math.hypot(st.vx, st.vy) : undefined;
      let outOfBounds;
      if (bounds && Number.isFinite(st.x) && Number.isFinite(st.y)) {
        outOfBounds =
          Math.abs(st.x - bounds.cx) > bounds.halfX || Math.abs(st.y - bounds.cy) > bounds.halfY;
      }
      return {
        seat,
        role: game === 'asym' ? seat : (GAMES[game].seatRole || {})[seat] || seat,
        dir: typeof config.goalDir === 'function' ? config.goalDir(game, seat) : undefined,
        x: st.x,
        y: st.y,
        yaw: st.yaw,
        speed,
        lineRemaining: dist[seat],
        fallen: Number.isFinite(st.tilt) ? st.tilt > fallAngle : undefined,
        oob: outOfBounds,
        filterActive: Boolean(f && (f.active || f.intervening || f.alpha > 0)),
        filterAlpha: f ? f.alpha : undefined,
      };
    };

    return {
      phase: raw.phase,
      step: raw.step,
      stepsTotal: raw.episodeSteps,
      timeLeftS: raw.timeLeft,
      countdownS: raw.countdown ? raw.countdown.msLeft / 1000 : null,
      cmd: raw.cmd,
      camera: raw.camera || state.camera,
      fps: raw.fps,
      stepMs: raw.perf ? raw.perf.lastStepMs : raw.stepMs,
      you: side(raw.playerSeat),
      ai: { ...side(raw.aiSeat), label: raw.opponent ? raw.opponent.display : undefined },
      result: raw.verdict || null,
    };
  }

  function updateHud(s, extra) {
    const adapted = adaptHud(s, extra);
    hud.update(adapted);
    if (adapted && adapted.camera && adapted.camera !== state.camera) state.camera = adapted.camera;
  }

  function showResult(result) {
    state.result = result || null;
    const aiSeat = aiSeatFor(state.game, state.playerSeat);
    const v = verdictText(config, state.game, state.playerSeat, aiSeat, result || {});
    resultHead.textContent = v.headline;
    resultSub.textContent = v.detail;
    resultTerm.textContent = v.term;
    resultTone.className = `result-tone tone-${v.tone}`;
    resultTone.textContent = v.tone === 'win' ? 'WIN' : v.tone === 'lose' ? 'LOSS' : 'DRAW';
    resultStats.textContent = '';
    const st = (result && result.stats) || {};
    const bits = [];
    if (Number.isFinite(st.steps)) bits.push(['steps', `${st.steps} / ${GAMES[state.game].episodeSteps}`]);
    if (Number.isFinite(st.timeS)) bits.push(['elapsed', `${st.timeS.toFixed(2)} s`]);
    if (Number.isFinite(st.closest)) bits.push(['closest to the line', `${st.closest.toFixed(2)} m`]);
    for (const [k, val] of bits) {
      resultStats.appendChild(h('div.result-stat', null, [h('span', { text: k }), h('b', { text: val })]));
    }
    input.setEnabled(false);
    resultOverlay.hidden = false;
  }

  function hideResult() {
    resultOverlay.hidden = true;
    if (state.phase === 'game' && !state.paused) input.setEnabled(true);
  }

  function setPaused(p) {
    const next = Boolean(p);
    if (next === state.paused) {
      pauseOverlay.hidden = !next;
      return;
    }
    state.paused = next;
    pauseOverlay.hidden = !next;
    input.setEnabled(state.phase === 'game' && !next && resultOverlay.hidden);
    cb(next ? 'onPause' : 'onResume');
  }

  function setCamera(mode) {
    state.camera = mode;
    cb('onCamera', mode);
  }

  function cycleCamera() {
    const i = cameraModes.indexOf(state.camera);
    setCamera(cameraModes[(i + 1) % cameraModes.length]);
    toast(`Camera: ${state.camera}`, 'info', 1200);
  }

  function fatal(message) {
    fatalMsg.textContent = String(message);
    fatalOverlay.hidden = false;
  }

  /* ============================================================ hotkeys */

  function onKey(ev) {
    if (ev.ctrlKey || ev.metaKey || ev.altKey) return;
    const t = ev.target;
    if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.isContentEditable)) return;
    const pauseKey = (config.INPUT && config.INPUT.pauseKey) || 'KeyP';
    if (ev.code === 'Escape' || ev.code === pauseKey) {
      if (openPanelEl) {
        closePanel();
        ev.preventDefault();
        return;
      }
      if (state.phase === 'game' && resultOverlay.hidden) {
        setPaused(!state.paused);
        ev.preventDefault();
      }
      return;
    }
    if (state.phase !== 'game') {
      if (ev.code === 'Enter' && state.phase === 'setup' && !startBtn.disabled) {
        start();
        ev.preventDefault();
      }
      return;
    }
    const restartKey = (config.INPUT && config.INPUT.restartKey) || 'KeyR';
    if (ev.code === restartKey) {
      doRestart();
      ev.preventDefault();
    } else if (ev.code === 'KeyC') {
      cycleCamera();
      ev.preventDefault();
    }
  }
  globalThis.addEventListener('keydown', onKey);

  const onWinResize = () => cb('onResize');
  globalThis.addEventListener('resize', onWinResize);

  /* ============================================================ boot */

  renderControlsList();
  syncSetup();
  setPhase('setup');

  // deep links
  const deep = {
    game: params.get('game'),
    role: params.get('role'),
    opponent: params.get('opponent'),
    filter: params.get('filter'),
    camera: params.get('camera'),
    scheme: params.get('scheme'),
    autostart: params.get('autostart'),
  };
  const deepScheme = deep.scheme && (SCHEME_ALIAS[deep.scheme] || deep.scheme);
  if (deepScheme && SCHEMES[deepScheme]) {
    settings.scheme = deepScheme;
    input.setScheme(deepScheme);
    schemeSeg.sync();
    renderControlsList();
  }
  if (deep.game && GAMES[deep.game] && offeredGames.includes(deep.game)) {
    state.game = deep.game;
    state.playerSeat = defaultSeat(deep.game);
  }
  if (deep.role) {
    const seats = GAMES[state.game].seats;
    if (seats.includes(deep.role)) state.playerSeat = deep.role;
  }
  if (deep.opponent) state.method = deep.opponent;
  if (deep.filter && capabilities.filter) {
    state.filter.ai = deep.filter === 'ai' || deep.filter === 'both';
    state.filter.you = deep.filter === 'you' || deep.filter === 'both';
  }
  if (deep.camera && cameraModes.includes(deep.camera)) state.camera = deep.camera;
  syncSetup();


  if (manifestError) {
    toast('Could not load the policy manifest', 'warn', 6000);
  }

  const ui = {
    el: root,
    stage,
    canvas,
    hud,
    input,
    settings,
    config,
    manifest,
    capabilities,
    loading,
    startMatch,
    updateHud,
    showResult,
    hideResult,
    setPaused,
    isPaused: () => state.paused,
    setCamera,
    cycleCamera,
    toast,
    fatal,
    showTitle: backToSetup,
    getSetup: currentSetup,
    setCapabilities(next) {
      Object.assign(capabilities, next || {});
      syncSetup();
    },
    verdictText: (result) =>
      verdictText(config, state.game, state.playerSeat, aiSeatFor(state.game, state.playerSeat), result),
    get phase() {
      return state.phase;
    },
    destroy() {
      globalThis.removeEventListener('keydown', onKey);
      globalThis.removeEventListener('resize', onWinResize);
      input.destroy();
      hud.destroy();
      root.remove();
    },
  };

  if (deep.autostart === '1' || deep.autostart === 'true') {
    // let the caller wire its handlers first
    setTimeout(() => {
      if (!startBtn.disabled) start();
    }, 0);
  }

  return ui;
}

export default createUI;

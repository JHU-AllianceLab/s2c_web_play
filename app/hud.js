/**
 * app/hud.js — the in-game overlay: clock, per-side distance to the scoring line, the field
 * minimap, the command stick, the countdown, the camera/perf readout and the who-is-who legend.
 *
 *   import { createHud } from './hud.js';
 *   const hud = createHud({ config });          // config = app/config.js namespace object
 *   parent.appendChild(hud.el);
 *   hud.setMatch({ game:'sym', playerSeat:'A', opponent:{...} });
 *   hud.update(stateFromMatchHud);              // every animation frame is fine
 *
 * THE STATE OBJECT (what app/main.js feeds in, from match.hud()) — every field is optional and a
 * missing one renders as a dash rather than a zero, so a half-wired engine never lies on screen:
 *
 *   phase        'countdown' | 'running' | 'paused' | 'over'
 *   step         int, control steps elapsed          stepsTotal int (GAMES[game].episodeSteps)
 *   timeLeftS    float seconds                       countdownS float, counts down to 0
 *   cmd          { vx, vy, wz } actually sent to the walk policy this step
 *   camera       string, the live camera mode
 *   fps          float                               stepMs float, wall ms per control step
 *   you / ai     {
 *                  seat, role, label, method,
 *                  x, y, yaw,                        // trunk centre, env frame, metres / rad
 *                  dir,                              // +1 scores at +lineX, -1 at -lineX
 *                  lineRemaining,                    // optional; else derived as lineX - dir*x
 *                  fallen, oob, filterActive, filterAlpha, speed
 *                }
 *   result       { terminal, winner, ... } once phase === 'over' (the overlay lives in ui.js)
 *
 * GEOMETRY AND COLOUR ARE NOT INVENTED HERE.
 *   Field rectangle, centre and scoring line come from `config.GAMES[game]`.
 *   The out-of-bounds rectangle IS the field rectangle: asym |x-0.2| > 2.6 and |y| > 1.5 for a
 *   5.2 x 3.0 field centred at (0.2, 0), sym |x| > 2.8 and |y| > 1.5 for 5.6 x 3.0 at (0,0)
 *   (recon/05_env_physics_contract.md:254 and section 6; assets/scene/<game>/scene.json
 *   field.oobX / field.oobY agree to the digit). There are no walls — out of bounds is a referee
 *   test on the trunk centre (recon/05 trap 2).
 *   `lineRemaining = lineX - dir * x` is the training observation `{seat}_line`
 *   (src/tasks/sym_game/mdp/sym.py:145, quoted at recon/05 section 6), so the number on screen is
 *   the number the policy reads.
 *   Field colours are mjlab's own play-marker palette, so the minimap matches the 3D arena:
 *   boundary (0.92,0.92,0.88), touchdown line (1.0,0.28,0.14), end zone (0.1,0.8,0.35)
 *   — src/tasks/game/game_env_cfg.py:98-100, re-exported in scene.json field.mjlabMarkers.
 */

/* --------------------------------------------------------------------------- DOM helper */

/** Tiny hyperscript. `h('div.card', {id:'x'}, [child, 'text'])`. Shared with app/ui.js. */
export function h(spec, attrs, children) {
  const m = /^([a-zA-Z0-9-]+)?((?:[.#][^.#]+)*)$/.exec(spec) || [];
  const tag = m[1] || 'div';
  const el = document.createElement(tag);
  const rest = m[2] || '';
  const classes = [];
  for (const token of rest.split(/(?=[.#])/)) {
    if (!token) continue;
    if (token[0] === '.') classes.push(token.slice(1));
    else if (token[0] === '#') el.id = token.slice(1);
  }
  if (classes.length) el.className = classes.join(' ');
  if (attrs) {
    for (const [k, v] of Object.entries(attrs)) {
      if (v == null || v === false) continue;
      if (k === 'class') el.className = (el.className ? el.className + ' ' : '') + v;
      else if (k === 'text') el.textContent = String(v);
      else if (k === 'html') el.innerHTML = v;
      else if (k === 'style' && typeof v === 'object') Object.assign(el.style, v);
      else if (k.startsWith('on') && typeof v === 'function') el.addEventListener(k.slice(2), v);
      else if (k === 'dataset') Object.assign(el.dataset, v);
      else el.setAttribute(k, v === true ? '' : String(v));
    }
  }
  if (children != null) {
    for (const c of Array.isArray(children) ? children : [children]) {
      if (c == null || c === false) continue;
      el.appendChild(typeof c === 'string' || typeof c === 'number' ? document.createTextNode(String(c)) : c);
    }
  }
  return el;
}

/* ------------------------------------------------------------------------------ palette */

/**
 * The single source for "who is who" colours. app/render.js should read these so the 3D dogs,
 * the minimap and the legend agree. Player teal and a warm grey opponent are DESIGN.md section 12.
 */
export const SIDE_COLORS = {
  player: { base: '#1F6F8B', bright: '#4FC3E8', name: 'teal' },
  ai: { base: '#B9AFA2', bright: '#E4D9C8', name: 'warm grey' },
};

/** mjlab play-marker palette (game_env_cfg.py:98-100), as CSS. */
export const FIELD_COLORS = {
  apron: '#141F2B',
  surface: '#293D54',
  boundary: 'rgba(235,235,225,0.72)',
  line: '#FF4724',
  zone: 'rgba(26,204,89,0.20)',
  zoneEdge: 'rgba(26,204,89,0.55)',
  /** Both ends live: each one takes the colour of the dog that scores in it. */
  zoneYou: 'rgba(79,195,232,0.26)',
  zoneAi: 'rgba(216,160,90,0.30)',
  midline: 'rgba(235,235,225,0.20)',
};

/* -------------------------------------------------------------------------------- utils */

const NUM_DASH = '—'; // em dash for "no data"

function fmt(v, digits = 2, unit = '') {
  if (!Number.isFinite(v)) return NUM_DASH;
  return v.toFixed(digits) + unit;
}

function clamp01(v) {
  return v < 0 ? 0 : v > 1 ? 1 : v;
}

function mmss(seconds) {
  if (!Number.isFinite(seconds)) return NUM_DASH;
  const s = Math.max(0, seconds);
  const whole = Math.floor(s);
  const tenths = Math.floor((s - whole) * 10);
  return `${whole}.${tenths}`;
}

/* ---------------------------------------------------------------------------- the HUD */

export function createHud(options = {}) {
  const config = options.config || null;
  if (!config || !config.GAMES) {
    throw new Error('app/hud.js: createHud needs { config } (the app/config.js namespace).');
  }
  const GAMES = config.GAMES;
  const controlDt = (config.PHYS && config.PHYS.controlDt) || null;

  let game = null;
  let spec = null;
  let playerSeat = null;
  let opponent = null;
  let lastState = null;

  /* ---- structure ---- */
  const youChip = h('span.side-chip', { style: { background: SIDE_COLORS.player.base } });
  const aiChip = h('span.side-chip', { style: { background: SIDE_COLORS.ai.base } });

  function sideCard(which) {
    const chip = which === 'you' ? youChip : aiChip;
    const name = h('span.side-name', { text: which === 'you' ? 'YOU' : 'AI' });
    const seat = h('span.side-seat', { text: NUM_DASH });
    const method = h('span.side-method', { text: '' });
    const shield = h('span.chip.chip-shield', { text: 'SHIELD', title: 'the QCBF certificate is overriding this side right now' });
    shield.hidden = true;
    const distNum = h('span.dist-num', { text: NUM_DASH });
    const distUnit = h('span.dist-unit', { text: 'm to line' });
    const barFill = h('i.bar-fill');
    const bar = h('div.bar', null, [barFill]);
    const flag = h('span.side-flag', { text: '' });
    const el = h(`div.side-card.side-${which}`, null, [
      h('div.side-head', null, [chip, name, seat, method, shield]),
      h('div.side-dist', null, [distNum, distUnit]),
      bar,
      flag,
    ]);
    return { el, seat, method, shield, distNum, distUnit, barFill, flag, name };
  }

  const you = sideCard('you');
  const ai = sideCard('ai');

  const clockNum = h('div.clock-num', { text: NUM_DASH });
  const clockSub = h('div.clock-sub', { text: 'seconds left' });
  const stepNum = h('div.clock-steps', { text: NUM_DASH });
  const clockBar = h('i.clock-bar-fill');
  const clock = h('div.hud-clock', null, [
    clockNum,
    clockSub,
    h('div.clock-bar', null, [clockBar]),
    stepNum,
  ]);

  const legend = h('div.hud-legend', { text: '' });

  const top = h('div.hud-top', null, [you.el, clock, ai.el]);

  const bigText = h('div.big-text', { text: '' });
  const bigSub = h('div.big-sub', { text: '' });
  const center = h('div.hud-center', null, [bigText, bigSub]);
  center.hidden = true;

  const mapCanvas = h('canvas.hud-map');
  const mapWrap = h('div.hud-panel.hud-map-wrap', null, [
    h('div.panel-title', { text: 'FIELD' }),
    mapCanvas,
    h('div.map-scale', { text: '' }),
  ]);
  const mapScale = mapWrap.querySelector('.map-scale');

  const stickCanvas = h('canvas.hud-stick');
  const cmdVx = h('b', { text: NUM_DASH });
  const cmdVy = h('b', { text: NUM_DASH });
  const cmdWz = h('b', { text: NUM_DASH });
  const stickWrap = h('div.hud-panel.hud-stick-wrap', null, [
    h('div.panel-title', { text: 'COMMAND' }),
    stickCanvas,
    h('div.cmd-nums', null, [
      h('span', null, ['vx ', cmdVx]),
      h('span', null, ['vy ', cmdVy]),
      h('span', null, ['ωz ', cmdWz]),
    ]),
  ]);

  const camText = h('span.meta-cam', { text: NUM_DASH });
  const perfText = h('span.meta-perf', { text: '' });
  const meta = h('div.hud-meta', null, [
    h('span.meta-key', { text: 'C' }),
    camText,
    h('span.meta-sep', { text: '·' }),
    perfText,
  ]);

  const bottom = h('div.hud-bottom', null, [mapWrap, h('div.hud-bottom-right', null, [meta, stickWrap])]);

  const el = h('div.hud', { 'aria-hidden': 'true' }, [top, legend, center, bottom]);
  el.hidden = true;

  /* ---- minimap drawing ---- */

  let mapDpr = 1;
  function sizeCanvas(canvas, cssW, cssH) {
    const dpr = Math.min(globalThis.devicePixelRatio || 1, 2);
    const w = Math.max(1, Math.round(cssW * dpr));
    const hh = Math.max(1, Math.round(cssH * dpr));
    if (canvas.width !== w || canvas.height !== hh) {
      canvas.width = w;
      canvas.height = hh;
    }
    canvas.style.width = cssW + 'px';
    canvas.style.height = cssH + 'px';
    return dpr;
  }

  function drawDog(ctx, px, py, yaw, color, scale, isPlayer) {
    ctx.save();
    ctx.translate(px, py);
    ctx.rotate(-yaw); // canvas y is down; env yaw is CCW about +z
    const L = 7 * scale;
    const W = 4.6 * scale;
    ctx.beginPath();
    ctx.moveTo(L, 0);
    ctx.lineTo(-L * 0.72, W);
    ctx.lineTo(-L * 0.42, 0);
    ctx.lineTo(-L * 0.72, -W);
    ctx.closePath();
    ctx.fillStyle = color;
    ctx.fill();
    ctx.lineWidth = 1 * scale;
    ctx.strokeStyle = 'rgba(8,12,18,0.85)';
    ctx.stroke();
    if (isPlayer) {
      ctx.beginPath();
      ctx.arc(0, 0, L * 1.35, 0, Math.PI * 2);
      ctx.strokeStyle = SIDE_COLORS.player.bright;
      ctx.globalAlpha = 0.55;
      ctx.lineWidth = 1.2 * scale;
      ctx.stroke();
    }
    ctx.restore();
  }

  function drawMap(state) {
    if (!spec) return;
    const cssW = mapCanvas.clientWidth || 208;
    const cssH = mapCanvas.clientHeight || Math.round((208 * spec.field[1]) / spec.field[0]);
    mapDpr = sizeCanvas(mapCanvas, cssW, cssH);
    const ctx = mapCanvas.getContext('2d');
    if (!ctx) return;
    ctx.setTransform(mapDpr, 0, 0, mapDpr, 0, 0);
    ctx.clearRect(0, 0, cssW, cssH);

    const pad = 8;
    const [fw, fh] = spec.field;
    const [cx, cy] = spec.center;
    const sc = Math.min((cssW - pad * 2) / fw, (cssH - pad * 2) / fh);
    const originX = cssW / 2;
    const originY = cssH / 2;
    // env (x,y) -> canvas: +x right, +y up
    const X = (x) => originX + (x - cx) * sc;
    const Y = (y) => originY - (y - cy) * sc;

    const left = X(cx - fw / 2);
    const right = X(cx + fw / 2);
    const topY = Y(cy + fh / 2);
    const botY = Y(cy - fh / 2);

    ctx.fillStyle = FIELD_COLORS.apron;
    ctx.fillRect(0, 0, cssW, cssH);
    ctx.fillStyle = FIELD_COLORS.surface;
    ctx.fillRect(left, topY, right - left, botY - topY);

    // end zones: from the scoring line to the field edge on that side.
    for (const line of spec.lines) {
      const lx = X(line.x);
      const edge = line.dir > 0 ? right : left;
      ctx.fillStyle =
        line.mine === undefined
          ? FIELD_COLORS.zone
          : line.mine
            ? FIELD_COLORS.zoneYou
            : FIELD_COLORS.zoneAi;
      ctx.fillRect(Math.min(lx, edge), topY, Math.abs(edge - lx), botY - topY);
    }
    // halfway line
    ctx.strokeStyle = FIELD_COLORS.midline;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(X(cx), topY);
    ctx.lineTo(X(cx), botY);
    ctx.stroke();

    // boundary == the out-of-bounds rectangle
    ctx.strokeStyle = FIELD_COLORS.boundary;
    ctx.lineWidth = 1.5;
    ctx.strokeRect(left + 0.5, topY + 0.5, right - left - 1, botY - topY - 1);

    // scoring lines
    for (const line of spec.lines) {
      ctx.strokeStyle = FIELD_COLORS.line;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(X(line.x), topY);
      ctx.lineTo(X(line.x), botY);
      ctx.stroke();
    }

    const dogScale = Math.max(0.75, Math.min(1.35, sc / 36));
    const sides = [
      { s: state && state.ai, color: SIDE_COLORS.ai.bright, player: false },
      { s: state && state.you, color: SIDE_COLORS.player.bright, player: true },
    ];
    for (const side of sides) {
      if (!side.s || !Number.isFinite(side.s.x) || !Number.isFinite(side.s.y)) continue;
      drawDog(ctx, X(side.s.x), Y(side.s.y), side.s.yaw || 0, side.color, dogScale, side.player);
    }
  }

  /* ---- command stick drawing ---- */

  function drawStick(cmd, box) {
    const cssW = stickCanvas.clientWidth || 104;
    const cssH = stickCanvas.clientHeight || 104;
    const dpr = sizeCanvas(stickCanvas, cssW, cssH);
    const ctx = stickCanvas.getContext('2d');
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cssW, cssH);

    const pad = 10;
    const wzH = 10;
    const boxW = cssW - pad * 2;
    const boxH = cssH - pad * 2 - wzH - 6;
    const cx = pad + boxW / 2;
    const cy = pad + boxH / 2;

    ctx.strokeStyle = 'rgba(200,215,235,0.22)';
    ctx.lineWidth = 1;
    ctx.strokeRect(pad + 0.5, pad + 0.5, boxW - 1, boxH - 1);
    ctx.beginPath();
    ctx.moveTo(cx, pad + 2);
    ctx.lineTo(cx, pad + boxH - 2);
    ctx.moveTo(pad + 2, cy);
    ctx.lineTo(pad + boxW - 2, cy);
    ctx.strokeStyle = 'rgba(200,215,235,0.12)';
    ctx.stroke();

    const vx = Number.isFinite(cmd && cmd.vx) ? cmd.vx : 0;
    const vy = Number.isFinite(cmd && cmd.vy) ? cmd.vy : 0;
    const wz = Number.isFinite(cmd && cmd.wz) ? cmd.wz : 0;
    // forward is up; +vy is LEFT (body frame, recon/04:275), so screen x uses -vy.
    const nx = vy >= 0 ? vy / Math.abs(box.vy[1]) : -vy / Math.abs(box.vy[0]);
    const ny = vx >= 0 ? vx / Math.abs(box.vx[1]) : -vx / Math.abs(box.vx[0]);
    const px = cx - clamp01(Math.abs(nx)) * Math.sign(nx) * (boxW / 2 - 8);
    const py = cy - clamp01(Math.abs(ny)) * Math.sign(ny) * (boxH / 2 - 8);

    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(px, py);
    ctx.strokeStyle = SIDE_COLORS.player.bright;
    ctx.globalAlpha = 0.6;
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.globalAlpha = 1;
    ctx.beginPath();
    ctx.arc(px, py, 5, 0, Math.PI * 2);
    ctx.fillStyle = SIDE_COLORS.player.bright;
    ctx.fill();

    // yaw bar
    const barY = pad + boxH + 6;
    ctx.fillStyle = 'rgba(200,215,235,0.10)';
    ctx.fillRect(pad, barY, boxW, wzH);
    const nwz = wz >= 0 ? wz / Math.abs(box.wz[1]) : -wz / Math.abs(box.wz[0]);
    const half = boxW / 2;
    const w = clamp01(Math.abs(nwz)) * half;
    ctx.fillStyle = SIDE_COLORS.player.bright;
    // +wz is a LEFT turn, so it fills to the left of centre.
    if (nwz >= 0) ctx.fillRect(pad + half - w, barY, w, wzH);
    else ctx.fillRect(pad + half, barY, w, wzH);
    ctx.fillStyle = 'rgba(200,215,235,0.45)';
    ctx.fillRect(pad + half - 0.5, barY, 1, wzH);
  }

  /* ---- match description ---- */

  function seatDirection(gameId, seat) {
    // config.goalDir is the authority (SEAT_GOAL_DIR: sym A/+1 B/-1, asym attacker/+1 and
    // defender/0 = "no scoring line"). The fallback keeps the same convention for a config
    // that predates the helper.
    if (typeof config.goalDir === 'function') {
      const d = config.goalDir(gameId, seat);
      if (d !== 0) return d;
      return -1; // the asym defender: no line of its own, only a side of the field
    }
    const seats = GAMES[gameId].seats;
    return seat === seats[0] ? +1 : -1;
  }

  function setMatch(m) {
    game = m.game;
    playerSeat = m.playerSeat;
    opponent = m.opponent || null;
    const g = GAMES[game];
    if (!g) throw new Error(`app/hud.js: unknown game "${game}"`);
    const aiSeat = g.seats.find((s) => s !== playerSeat);
    const lines = [];
    if (game === 'sym') {
      // `mine` paints the minimap end zones the same way app/render.js paints
      // the pitch: the one you run at carries your colour.
      const youDir = seatDirection(game, playerSeat);
      lines.push(
        { x: g.lineX, dir: +1, mine: youDir > 0 },
        { x: -g.lineX, dir: -1, mine: youDir < 0 },
      );
    } else {
      lines.push({ x: g.lineX, dir: +1 });
    }
    spec = {
      field: g.field,
      center: g.center,
      lineX: g.lineX,
      lines,
      aiSeat,
      steps: g.episodeSteps,
      maxRemaining: g.lineX + g.field[0] / 2,
    };
    const pDir = seatDirection(game, playerSeat);
    const aDir = seatDirection(game, aiSeat);
    spec.dir = { you: pDir, ai: aDir };

    you.seat.textContent = seatLabel(game, playerSeat);
    ai.seat.textContent = seatLabel(game, aiSeat);
    ai.method.textContent = opponent && opponent.display ? opponent.display : '';
    you.method.textContent = '';
    // Plain words, not coordinates: the player is looking down the pitch, not
    // reading the scenario file.
    const goalOf = (seat, dir, third) =>
      game === 'asym' && seat === 'defender'
        ? `hold${third ? 's' : ''} the line`
        : `run${third ? 's' : ''} ${dir > 0 ? 'right' : 'left'}`;
    const youGoal = goalOf(playerSeat, pDir, false);
    const aiGoal = goalOf(aiSeat, aDir, true);
    legend.textContent =
      `You ${youGoal}    |    ` +
      `${opponent && opponent.display ? opponent.display : 'the AI'} ${aiGoal}`;
    mapScale.textContent = `${g.field[0].toFixed(1)} × ${g.field[1].toFixed(1)} m`;
    if (controlDt) {
      clockSub.textContent = `s left of ${(g.episodeSteps * controlDt).toFixed(0)}`;
    }
    drawMap(null);
    drawStick({ vx: 0, vy: 0, wz: 0 }, (options.cmdBox || config.CMD_BOX));
  }

  function seatLabel(gameId, seat) {
    // In the symmetric game the seat is bookkeeping — both dogs do the same
    // thing — so the chip carries only "You" and the opponent's name.
    if (gameId === 'sym') return '';
    return seat === 'attacker' ? 'attacker' : 'defender';
  }

  /* ---- per-frame update ---- */

  function sideUpdate(ui, s, which) {
    if (!s) {
      ui.distNum.textContent = NUM_DASH;
      ui.barFill.style.width = '0%';
      ui.flag.textContent = '';
      ui.shield.hidden = true;
      return;
    }
    const dir = Number.isFinite(s.dir) ? s.dir : spec.dir[which];
    let remaining = s.lineRemaining;
    if (!Number.isFinite(remaining) && Number.isFinite(s.x)) {
      remaining = spec.lineX - dir * s.x; // sym.py:145
    }
    const defends = game === 'asym' && s.role === 'defender';
    if (defends) {
      ui.distNum.textContent = NUM_DASH;
      ui.distUnit.textContent = 'defending';
      ui.barFill.style.width = '0%';
    } else {
      ui.distUnit.textContent = 'm to line';
      ui.distNum.textContent = fmt(remaining, 2);
      const frac = Number.isFinite(remaining) ? 1 - clamp01(remaining / spec.maxRemaining) : 0;
      ui.barFill.style.width = (frac * 100).toFixed(1) + '%';
    }
    ui.shield.hidden = !s.filterActive;
    let flag = '';
    if (s.fallen) flag = 'DOWN';
    else if (s.oob) flag = 'OUT';
    else if (Number.isFinite(s.speed)) flag = fmt(s.speed, 1, ' m/s');
    ui.flag.textContent = flag;
    ui.flag.classList.toggle('flag-bad', Boolean(s.fallen || s.oob));
  }

  function update(state) {
    lastState = state || null;
    if (!spec) return;
    const s = state || {};
    const total = Number.isFinite(s.stepsTotal) ? s.stepsTotal : spec.steps;
    let timeLeft = s.timeLeftS;
    if (!Number.isFinite(timeLeft) && Number.isFinite(s.step) && controlDt) {
      timeLeft = Math.max(0, (total - s.step) * controlDt);
    }
    clockNum.textContent = mmss(timeLeft);
    stepNum.textContent = Number.isFinite(s.step) ? `step ${s.step} / ${total}` : NUM_DASH;
    const frac = Number.isFinite(s.step) ? clamp01(1 - s.step / total) : 1;
    clockBar.style.width = (frac * 100).toFixed(1) + '%';
    clockBar.classList.toggle('clock-low', Number.isFinite(timeLeft) && timeLeft <= 3);

    sideUpdate(you, s.you, 'you');
    sideUpdate(ai, s.ai, 'ai');

    camText.textContent = s.camera || NUM_DASH;
    const bits = [];
    if (Number.isFinite(s.fps)) bits.push(`${Math.round(s.fps)} fps`);
    if (Number.isFinite(s.stepMs)) bits.push(`${s.stepMs.toFixed(1)} ms/step`);
    perfText.textContent = bits.join('  ');

    drawMap(s);
    drawStick(s.cmd || (s.you && s.you.cmd) || { vx: 0, vy: 0, wz: 0 }, options.cmdBox || config.CMD_BOX);

    if (s.phase === 'countdown') {
      const c = Number.isFinite(s.countdownS) ? Math.ceil(s.countdownS) : null;
      center.hidden = false;
      bigText.textContent = c == null ? 'GET READY' : c <= 0 ? 'GO' : String(c);
      bigText.className = 'big-text' + (c === 0 || c == null ? ' big-go' : '');
      bigSub.textContent = c == null ? '' : 'the clock starts when the countdown ends';
    } else if (s.phase === 'paused') {
      center.hidden = true; // the pause card is ui.js's overlay
    } else {
      center.hidden = true;
      bigText.className = 'big-text';
    }
  }

  function setVisible(v) {
    el.hidden = !v;
    el.setAttribute('aria-hidden', v ? 'false' : 'true');
  }

  const onResize = () => {
    if (spec) {
      drawMap(lastState);
      drawStick(
        (lastState && lastState.cmd) || { vx: 0, vy: 0, wz: 0 },
        options.cmdBox || config.CMD_BOX
      );
    }
  };
  globalThis.addEventListener('resize', onResize);

  return {
    el,
    setMatch,
    update,
    setVisible,
    redraw: onResize,
    get state() {
      return lastState;
    },
    destroy() {
      globalThis.removeEventListener('resize', onResize);
      el.remove();
    },
  };
}

export default createHud;

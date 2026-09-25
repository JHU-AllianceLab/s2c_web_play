/**
 * app/input.js — keyboard / gamepad / touch -> the walk policy's velocity command.
 *
 * FROZEN INTERFACE
 *   export function createInput(target): { read(): {vx,vy,wz}, bindings, setScheme(name) }
 * Everything else on the returned object is additive.
 *
 * WHAT THIS MODULE IS ALLOWED TO DECIDE, AND WHAT IT IS NOT
 *   NOT ours: the command box. `vx in [-1.5, 3.0], vy in [-1.0, 1.0], wz in [-2.0, 2.0]` is the box
 *   the fastwalk_v3 walker was trained on and is read from app/config.js `CMD_BOX`
 *   (recon/04_player_walk_policy.md:20 and :271; the box itself is
 *   src/tasks/velocity/config/go2/env_cfgs.py, `Unitree-Go2-Flat-Fast`).
 *   Nothing here may hand the policy a command outside it.
 *
 *   NOT ours: key polarity. A = +vy (left), D = -vy (right), Q = +wz (left turn), E = -wz
 *   (right turn) is the repo's own teleop mapping,
 *   Project/unitree_rl_mjlab/scripts/teleop_fastwalk_record.py (quoted at recon/04:275:
 *   "W/S -> +-vx, A/D -> +-vy (A is +vy, D is -vy), Q/E -> +-wz (Q is +wz)"), and X zeroes the
 *   command there too. The signs are body frame, matching the 47-D obs slot [6:9]
 *   (recon/04:159).
 *
 *   OURS (feel constants, invented here, marked FEEL below): the ramp rates, the deadzone, the
 *   cruise/sprint split, the gamepad deadzone. They exist because a keyboard produces 0/1 and a
 *   step command makes the policy judder (DESIGN.md section 4: "must be slope-limited, deadzone
 *   0.05, clamp to the command box"). They are not training constants and are not cited as such.
 *   Every one of them is an option on createInput().
 *
 * USAGE
 *   const input = createInput(window);              // uses CMD_BOX from ./config.js
 *   const input = createInput(window, { cmdBox });  // or inject it
 *   const cmd = input.read();                       // call once per control step (50 Hz)
 *
 * `read()` is time-based: it ramps toward the key target using the wall time since the previous
 * read, so the caller's tick rate does not change the feel. Call it exactly once per control step.
 */

/** Test seam. Production resolves ./config.js; the UI preview driver sets this before import. */
const OVERRIDE = globalThis.S2C_CONFIG_OVERRIDE || null;

let CONFIG_MODULE = OVERRIDE;
if (!CONFIG_MODULE) {
  try {
    CONFIG_MODULE = await import('./config.js');
  } catch (err) {
    CONFIG_MODULE = null; // caller must inject opts.cmdBox
  }
}
const CMD_BOX_DEFAULT = CONFIG_MODULE ? CONFIG_MODULE.CMD_BOX : null;

/* ------------------------------------------------------------------ FEEL constants (ours) */

/**
 * ⚠ THERE IS A SECOND SLEW LIMITER DOWNSTREAM. app/match.js runs every command
 * through `slewCommand()` with `config.MATCH.cmdSlewPerSec`
 * (vx 6.0 m/s^2, vy 6.0 m/s^2, wz 10.0 rad/s^2, config.js:687) on the CONTROL
 * clock, in both directions. The rate a player actually feels is therefore
 * `min(span/seconds, cmdSlewPerSec)` per axis, and nothing tuned here can go
 * above that ceiling. The numbers below are set just under it so this file is
 * the one that decides — with one exception, vx, which is pinned to the ceiling.
 *
 * Measured against the trained box (CMD_BOX: vx 3.0, vy 1.0, wz 2.0):
 *
 *   axis   rate         hold to cruise        hold to the box       release
 *   vx     6.00 m/s^2   1.8 m/s in 0.300 s    3.0 m/s in 0.500 s    0.300 s
 *   vy     5.56 m/s^2   -                     1.0 m/s in 0.180 s    0.170 s
 *   wz     9.52 rad/s^2 -                     2.0 rad/s in 0.210 s  0.200 s
 *
 * vy was 3.33 m/s^2 (1.0 m/s in 0.300 s), which made A/D feel like the dog was
 * thinking about it; it is the one axis with real headroom under the ceiling.
 * vx cannot be made quicker from here — raising it needs MATCH.cmdSlewPerSec.vx.
 */
export const FEEL = {
  /** Seconds to ramp an axis across its full range when a key is held. */
  riseSeconds: { vx: 0.50, vy: 0.18, wz: 0.21 },
  /**
   * Seconds to decay an axis back to zero when the key is released. Faster than
   * the rise wherever the downstream ceiling leaves room — on vx it does not,
   * so the two are equal there and a release takes as long as a press.
   */
  fallSeconds: { vx: 0.50, vy: 0.17, wz: 0.20 },
  /** |v| below this is snapped to 0 (DESIGN.md section 4 asks for 0.05). */
  deadzone: 0.05,
  /** Without Shift the forward/backward target is this fraction of the box; Shift = the full box. */
  cruiseFrac: 0.6,
  /** Analog sticks: ignore below this, then rescale so the usable travel still reaches 1.0. */
  padDeadzone: 0.12,
  /** dt is clamped to this many seconds so a backgrounded tab cannot slam the command. */
  maxDt: 0.1,
};

/* ------------------------------------------------------------------------------- schemes */

/**
 * Two schemes, both keeping W/S on vx.
 *  strafe (default): A/D strafe (vy), Q/E turn (wz)   <- the repo's teleop mapping
 *  turn:             A/D turn  (wz), Q/E strafe (vy)  <- for people used to driving games
 *
 * The key->axis tables are NOT defined here when app/config.js is available: `INPUT.schemes`
 * there is the single source of truth and this module only parses it ('vx+' -> ['vx', +1]).
 * The labels and hints below are UI copy and are ours. `steer` is accepted as an alias of
 * `turn` so a deep link written either way works.
 */
const SCHEME_COPY = {
  strafe: {
    label: 'Strafe (default)',
    hint: 'A/D slide sideways, Q/E turn. Matches the lab teleop script.',
  },
  turn: {
    label: 'Steer',
    hint: 'A/D turn, Q/E slide sideways. Car-style.',
  },
};

export const SCHEME_ALIAS = { steer: 'turn', strafe: 'strafe', turn: 'turn' };

const FALLBACK_SCHEMES = {
  strafe: { KeyW: 'vx+', KeyS: 'vx-', KeyA: 'vy+', KeyD: 'vy-', KeyQ: 'wz+', KeyE: 'wz-' },
  turn: { KeyW: 'vx+', KeyS: 'vx-', KeyA: 'wz+', KeyD: 'wz-', KeyQ: 'vy+', KeyE: 'vy-' },
};

function parseSchemes(source) {
  const out = {};
  for (const [id, keys] of Object.entries(source)) {
    const map = {};
    for (const [code, token] of Object.entries(keys)) {
      const axis = String(token).slice(0, 2);
      if (!['vx', 'vy', 'wz'].includes(axis)) continue;
      map[code] = [axis, String(token).endsWith('-') ? -1 : +1];
    }
    out[id] = {
      label: (SCHEME_COPY[id] && SCHEME_COPY[id].label) || id,
      hint: (SCHEME_COPY[id] && SCHEME_COPY[id].hint) || '',
      map,
    };
  }
  return out;
}

export const SCHEMES = parseSchemes(
  (CONFIG_MODULE && CONFIG_MODULE.INPUT && CONFIG_MODULE.INPUT.schemes) || FALLBACK_SCHEMES
);

const DEFAULT_SCHEME =
  (CONFIG_MODULE && CONFIG_MODULE.INPUT && CONFIG_MODULE.INPUT.defaultScheme) || 'strafe';

const COMMON_MAP = {
  ArrowUp: ['vx', +1],
  ArrowDown: ['vx', -1],
};

/** Arrow-key aliases for the horizontal pair, resolved per scheme. */
const ARROW_ALIAS = { ArrowLeft: 'KeyA', ArrowRight: 'KeyD' };

const AXES = ['vx', 'vy', 'wz'];

function clamp(v, lo, hi) {
  return v < lo ? lo : v > hi ? hi : v;
}

function approach(current, target, rate, dt) {
  if (current === target) return target;
  const step = rate * dt;
  if (target > current) return Math.min(target, current + step);
  return Math.max(target, current - step);
}

function applyPadDeadzone(v, dz) {
  const a = Math.abs(v);
  if (a <= dz) return 0;
  return Math.sign(v) * ((a - dz) / (1 - dz));
}

/* ------------------------------------------------------------------------------ the thing */

/**
 * @param {EventTarget} target  where key listeners go (usually `window`).
 * @param {object} [opts]
 *   cmdBox     {vx:[lo,hi], vy:[lo,hi], wz:[lo,hi]} — required if ./config.js is unavailable
 *   scheme     'strafe' | 'steer'                   (default 'strafe')
 *   sensitivity number                              (default 1; scales the vy/wz targets only)
 *   gamepad    boolean                              (default true)
 *   feel       partial override of FEEL
 *   onChange   (state) => void, fired when the scheme/sensitivity/source changes
 */
export function createInput(target = globalThis, opts = {}) {
  const box = opts.cmdBox || CMD_BOX_DEFAULT;
  if (!box || !box.vx || !box.vy || !box.wz) {
    throw new Error(
      'app/input.js: no command box. app/config.js must export CMD_BOX, or pass ' +
        'createInput(target, { cmdBox }). Refusing to invent one — it is the trained box ' +
        '(recon/04:271).'
    );
  }
  const feel = {
    ...FEEL,
    ...(opts.feel || {}),
    riseSeconds: { ...FEEL.riseSeconds, ...((opts.feel || {}).riseSeconds || {}) },
    fallSeconds: { ...FEEL.fallSeconds, ...((opts.feel || {}).fallSeconds || {}) },
  };

  const span = {
    vx: Math.max(Math.abs(box.vx[0]), Math.abs(box.vx[1])),
    vy: Math.max(Math.abs(box.vy[0]), Math.abs(box.vy[1])),
    wz: Math.max(Math.abs(box.wz[0]), Math.abs(box.wz[1])),
  };

  const resolveScheme = (name) => {
    const id = SCHEME_ALIAS[name] || name;
    return SCHEMES[id] ? id : null;
  };
  let scheme = resolveScheme(opts.scheme) || resolveScheme(DEFAULT_SCHEME) || Object.keys(SCHEMES)[0];
  let sensitivity = Number.isFinite(opts.sensitivity) ? opts.sensitivity : 1;
  let gamepadEnabled = opts.gamepad !== false;
  let enabled = true;

  const held = new Set(); // KeyboardEvent.code
  let sprint = false;
  let zeroLatch = false; // X pressed: force a full stop until every key is released
  const value = { vx: 0, vy: 0, wz: 0 };
  const out = { vx: 0, vy: 0, wz: 0 };
  let external = null; // touch stick / programmatic override, in normalised [-1,1] axis units
  let lastSource = 'keyboard';
  let lastTime = null;
  let padIndex = null;

  const listeners = [];
  function on(el, type, fn, capture) {
    if (!el || !el.addEventListener) return;
    el.addEventListener(type, fn, capture || false);
    listeners.push([el, type, fn, capture || false]);
  }

  function isTypingTarget(el) {
    if (!el) return false;
    const tag = el.tagName;
    return (
      el.isContentEditable === true ||
      tag === 'INPUT' ||
      tag === 'TEXTAREA' ||
      tag === 'SELECT' ||
      tag === 'OPTION'
    );
  }

  function resolveCode(code) {
    if (ARROW_ALIAS[code]) return ARROW_ALIAS[code];
    return code;
  }

  function isBound(code) {
    const c = resolveCode(code);
    return Boolean(COMMON_MAP[c] || SCHEMES[scheme].map[c]);
  }

  function onKeyDown(ev) {
    if (!enabled || ev.ctrlKey || ev.metaKey || ev.altKey) return;
    if (isTypingTarget(ev.target)) return;
    if (ev.code === 'ShiftLeft' || ev.code === 'ShiftRight') {
      sprint = true;
      return;
    }
    if (ev.code === 'KeyX') {
      zeroLatch = true;
      held.clear();
      ev.preventDefault();
      return;
    }
    const code = resolveCode(ev.code);
    if (!isBound(code)) return;
    if (!ev.repeat) held.add(code);
    zeroLatch = false;
    lastSource = 'keyboard';
    ev.preventDefault(); // stop WASD/arrows scrolling the page
  }

  function onKeyUp(ev) {
    if (ev.code === 'ShiftLeft' || ev.code === 'ShiftRight') {
      sprint = false;
      return;
    }
    const code = resolveCode(ev.code);
    held.delete(code);
    if (held.size === 0) zeroLatch = false;
    if (isBound(code)) ev.preventDefault();
  }

  function releaseAll() {
    held.clear();
    sprint = false;
    zeroLatch = false;
    external = null;
    // Drop the wall clock too. read() is not called while the match is paused or
    // counting down (app/match.js returns before it), so without this the first
    // read afterwards would bill the whole gap — clamped to feel.maxDt, but that
    // is still a fifth of a second of ramp in one control step.
    lastTime = null;
  }

  on(target, 'keydown', onKeyDown);
  on(target, 'keyup', onKeyUp);
  // Losing focus must drop every key, or the dog runs off on its own (DESIGN.md section 4).
  on(globalThis, 'blur', releaseAll);
  on(globalThis.document, 'visibilitychange', () => {
    if (globalThis.document && globalThis.document.hidden) releaseAll();
  });
  on(globalThis, 'gamepadconnected', (ev) => {
    if (gamepadEnabled && ev && ev.gamepad) padIndex = ev.gamepad.index;
  });
  on(globalThis, 'gamepaddisconnected', (ev) => {
    if (ev && ev.gamepad && ev.gamepad.index === padIndex) padIndex = null;
  });

  /** Raw normalised targets in [-1,1] per axis, before the box and the cruise/sprint split. */
  function keyTargets() {
    const t = { vx: 0, vy: 0, wz: 0 };
    const map = { ...COMMON_MAP, ...SCHEMES[scheme].map };
    for (const code of held) {
      const bind = map[code];
      if (!bind) continue;
      t[bind[0]] += bind[1];
    }
    for (const a of AXES) t[a] = clamp(t[a], -1, 1);
    return t;
  }

  function padTargets() {
    if (!gamepadEnabled || !navigator.getGamepads) return null;
    let pads;
    try {
      pads = navigator.getGamepads();
    } catch {
      return null;
    }
    if (!pads) return null;
    let pad = padIndex != null ? pads[padIndex] : null;
    if (!pad || !pad.connected) {
      pad = null;
      for (const p of pads) {
        if (p && p.connected) {
          pad = p;
          padIndex = p.index;
          break;
        }
      }
    }
    if (!pad || !pad.axes || pad.axes.length < 2) return null;
    const dz = feel.padDeadzone;
    // Left stick: up = forward, left = +vy. Right stick X (axes[2]) or shoulder pair: turn.
    const vx = applyPadDeadzone(-(pad.axes[1] || 0), dz);
    const vy = applyPadDeadzone(-(pad.axes[0] || 0), dz);
    let wz = applyPadDeadzone(-(pad.axes[2] || 0), dz);
    if (wz === 0 && pad.buttons && pad.buttons.length > 5) {
      const l = pad.buttons[4] && pad.buttons[4].pressed ? 1 : 0;
      const r = pad.buttons[5] && pad.buttons[5].pressed ? 1 : 0;
      wz = l - r;
    }
    const padSprint = Boolean(
      (pad.buttons && pad.buttons[7] && pad.buttons[7].value > 0.5) ||
        (pad.buttons && pad.buttons[0] && pad.buttons[0].pressed)
    );
    if (vx === 0 && vy === 0 && wz === 0 && !padSprint) return null;
    return { vx, vy, wz, sprint: padSprint };
  }

  function read(nowMs) {
    const now = Number.isFinite(nowMs)
      ? nowMs
      : typeof performance !== 'undefined'
        ? performance.now()
        : Date.now();
    let dt = lastTime == null ? 0 : (now - lastTime) / 1000;
    lastTime = now;
    if (!(dt > 0)) dt = 0;
    dt = Math.min(dt, feel.maxDt);

    let norm;
    let sprinting = sprint;
    if (!enabled || zeroLatch) {
      norm = { vx: 0, vy: 0, wz: 0 };
      sprinting = false;
    } else if (external) {
      norm = {
        vx: clamp(external.vx || 0, -1, 1),
        vy: clamp(external.vy || 0, -1, 1),
        wz: clamp(external.wz || 0, -1, 1),
      };
      sprinting = sprint || Boolean(external.sprint);
      lastSource = external.source || 'touch';
    } else {
      const pad = padTargets();
      if (pad) {
        norm = pad;
        sprinting = sprint || pad.sprint;
        lastSource = 'gamepad';
      } else {
        norm = keyTargets();
        if (norm.vx || norm.vy || norm.wz) lastSource = 'keyboard';
      }
    }

    // normalised -> physical target, inside the trained box.
    const forwardFrac = sprinting ? 1 : feel.cruiseFrac;
    const targets = {
      vx: norm.vx >= 0 ? norm.vx * box.vx[1] * forwardFrac : -norm.vx * box.vx[0] * forwardFrac,
      vy: (norm.vy >= 0 ? norm.vy * box.vy[1] : -norm.vy * box.vy[0]) * sensitivity,
      wz: (norm.wz >= 0 ? norm.wz * box.wz[1] : -norm.wz * box.wz[0]) * sensitivity,
    };

    for (const a of AXES) {
      const tgt = clamp(targets[a], box[a][0], box[a][1]);
      const decaying = Math.abs(tgt) < Math.abs(value[a]) || tgt * value[a] < 0;
      const seconds = decaying ? feel.fallSeconds[a] : feel.riseSeconds[a];
      // Sensitivity scales the RAMP as well as the target. Scaling only the
      // target made the top half of the slider a no-op on a keyboard: a held key
      // asks for the full box either way, and the box clamp above throws the
      // rest away. Scaling the rate too is what makes 1.6x feel different from
      // 1.0x (turn onset 0.21 s -> 0.13 s, and the same key reaches the same
      // trained maximum — the box is never widened, only approached sooner).
      const rate = (span[a] / Math.max(seconds, 1e-3)) * (a === 'vx' ? 1 : sensitivity);
      value[a] = approach(value[a], tgt, rate, dt);
      let v = value[a];
      if (Math.abs(v) < feel.deadzone) v = 0;
      out[a] = clamp(v, box[a][0], box[a][1]);
    }
    return out;
  }

  function bindingRows() {
    const s = SCHEMES[scheme];
    const axisName = { vx: 'forward / back', vy: 'strafe', wz: 'turn' };
    const rows = [
      { keys: ['W', 'S'], axis: 'vx', label: axisName.vx },
      {
        keys: ['A', 'D'],
        axis: s.map.KeyA[0],
        label: axisName[s.map.KeyA[0]],
      },
      {
        keys: ['Q', 'E'],
        axis: s.map.KeyQ[0],
        label: axisName[s.map.KeyQ[0]],
      },
      { keys: ['Shift'], axis: 'vx', label: 'sprint (full forward box)' },
      { keys: ['X'], axis: null, label: 'stop' },
    ];
    for (const r of rows) {
      if (!r.axis) continue;
      r.range = box[r.axis];
      r.unit = r.axis === 'wz' ? 'rad/s' : 'm/s';
    }
    return rows;
  }

  const api = {
    /** FROZEN: the command for this control step. */
    read,
    /** FROZEN: a description of the live key map, for the UI to render. */
    get bindings() {
      return {
        scheme,
        label: SCHEMES[scheme].label,
        hint: SCHEMES[scheme].hint,
        rows: bindingRows(),
        cmdBox: box,
      };
    },
    /** FROZEN. Accepts a scheme id from config.INPUT.schemes, or the alias "steer". */
    setScheme(name) {
      const id = resolveScheme(name);
      if (!id) {
        throw new Error(
          `app/input.js: unknown scheme "${name}" (have ${Object.keys(SCHEMES).join(', ')})`
        );
      }
      if (id === scheme) return;
      scheme = id;
      held.clear();
      if (opts.onChange) opts.onChange(api.state());
    },
    getScheme: () => scheme,
    schemes: () => Object.keys(SCHEMES).map((k) => ({ id: k, ...SCHEMES[k] })),
    setSensitivity(v) {
      sensitivity = clamp(Number(v) || 1, 0.2, 2.0);
      if (opts.onChange) opts.onChange(api.state());
    },
    getSensitivity: () => sensitivity,
    setGamepadEnabled(v) {
      gamepadEnabled = Boolean(v);
    },
    /** Pause input without tearing the listeners down (menus, countdown, result overlay). */
    setEnabled(v) {
      const was = enabled;
      enabled = Boolean(v);
      if (!enabled) releaseAll();
      else if (!was) lastTime = null;   // resume: do not bill the paused seconds
    },
    isEnabled: () => enabled,
    /** Touch stick / scripted driving: normalised axes in [-1,1], or null to hand control back. */
    setExternalAxes(axes) {
      external = axes ? { source: 'touch', ...axes } : null;
    },
    releaseAll,
    /** Everything the HUD wants to draw, including the raw key set. */
    state() {
      return {
        scheme,
        sensitivity,
        sprint,
        source: lastSource,
        enabled,
        keys: Array.from(held),
        cmd: { ...out },
        cmdBox: box,
        gamepad: padIndex != null,
      };
    },
    cmdBox: box,
    feel,
    destroy() {
      for (const [el, type, fn, capture] of listeners) el.removeEventListener(type, fn, capture);
      listeners.length = 0;
      releaseAll();
    },
  };
  return api;
}

export default createInput;

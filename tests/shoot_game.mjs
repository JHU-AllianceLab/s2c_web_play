/**
 * tests/shoot_game.mjs — the REAL game in a real browser, headless.
 *
 *   node tests/shoot_game.mjs            # tests/shots/game_*.png + a pass/fail summary
 *   node tests/shoot_game.mjs --keep     # leave the server up afterwards
 *
 * Unlike tests/shoot_ui.mjs (which boots the UI preview with stub state) this one
 * boots app/main.js: MuJoCo WASM, the exported checkpoint weights, the renderer and
 * the referee, exactly what a player gets. It checks, for both asymmetric roles:
 *   - the match actually advances control steps in the browser
 *   - the canvas draws something other than one flat colour
 *   - the page reports no console error (index.html mirrors them into #log)
 *   - the browser's own control-step cost and frame rate
 */

import { createServer } from 'node:http';
import { spawn } from 'node:child_process';
import { readFile, mkdir, writeFile } from 'node:fs/promises';
import { extname, join, normalize, resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(HERE, '..');
const SHOTS = join(HERE, 'shots');
const argv = process.argv.slice(2);
const flag = (n, d) => { const i = argv.indexOf(n); return i >= 0 ? argv[i + 1] : d; };
const PORT = Number(flag('--port', '8791'));
const GD_PORT = Number(flag('--driver-port', '4481'));
const KEEP = argv.includes('--keep');
/** Point the harness at a deployed site instead of the local server. */
const BASE = flag('--base', '');

const MIME = {
  '.html': 'text/html; charset=utf-8', '.css': 'text/css; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8', '.mjs': 'text/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8', '.wasm': 'application/wasm',
  '.bin': 'application/octet-stream', '.obj': 'text/plain; charset=utf-8',
  '.xml': 'application/xml; charset=utf-8', '.png': 'image/png',
};
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function serve() {
  const server = createServer(async (req, res) => {
    const url = new URL(req.url, 'http://x');
    const rel = normalize(decodeURIComponent(url.pathname)).replace(/^(\.\.[/\\])+/, '');
    const abs = join(ROOT, rel === '/' ? 'index.html' : rel);
    try {
      const body = await readFile(abs);
      res.writeHead(200, { 'content-type': MIME[extname(abs)] || 'application/octet-stream' });
      res.end(body);
    } catch {
      res.writeHead(404).end('not found');
    }
  });
  return new Promise((ok) => server.listen(PORT, '127.0.0.1', () => ok(server)));
}

class Driver {
  constructor(port) { this.base = `http://127.0.0.1:${port}`; this.sid = null; }
  async call(method, path, body) {
    const res = await fetch(this.base + path, {
      method, headers: { 'content-type': 'application/json' },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
    const json = await res.json();
    if (json && json.value && json.value.error) throw new Error(`${json.value.error}: ${json.value.message}`);
    return json.value;
  }
  async start() {
    for (let i = 0; i < 120; i++) {
      try { await fetch(this.base + '/status'); break; } catch { await sleep(150); }
    }
    const s = await this.call('POST', '/session', {
      capabilities: { alwaysMatch: { 'moz:firefoxOptions': {
        args: ['-headless'],
        prefs: { 'browser.tabs.remote.autostart': false, 'devtools.console.stdout.content': true,
                 'webgl.force-enabled': true, 'layers.acceleration.force-enabled': true },
      } } },
    });
    this.sid = s.sessionId;
  }
  go(url) { return this.call('POST', `/session/${this.sid}/url`, { url }); }
  script(fn, args = []) {
    return this.call('POST', `/session/${this.sid}/execute/sync`, { script: `return (${fn}).apply(null, arguments);`, args });
  }
  setRect(width, height) { return this.call('POST', `/session/${this.sid}/window/rect`, { width, height, x: 0, y: 0 }); }
  async shot(name) {
    const b64 = await this.call('GET', `/session/${this.sid}/screenshot`);
    const buf = Buffer.from(b64, 'base64');
    await writeFile(join(SHOTS, name), buf);
    console.log(`  shot ${name}  ${buf.length.toLocaleString()} B`);
    return buf.length;
  }
  quit() { return this.call('DELETE', `/session/${this.sid}`).catch(() => {}); }
}

let failures = 0;
const ok = (cond, label, detail = '') => {
  console.log(`  ${cond ? 'ok  ' : 'FAIL'}  ${label}${detail ? `   ${detail}` : ''}`);
  if (!cond) failures += 1;
};

/** Poll a predicate in the page. */
async function waitFor(drv, expr, { timeout = 90000, every = 400, label = 'condition' } = {}) {
  const t0 = Date.now();
  while (Date.now() - t0 < timeout) {
    const v = await drv.script(`() => { try { return !!(${expr}); } catch (e) { return false; } }`);
    if (v) return true;
    await sleep(every);
  }
  throw new Error(`timed out waiting for ${label} (${Math.round((Date.now() - t0) / 1000)} s)`);
}

async function runCase(drv, { game = 'asym', role, opponent = 's2c' }) {
  const url = `${BASE || `http://127.0.0.1:${PORT}`}/index.html?game=${game}&role=${role}&opponent=${opponent}&autostart=1`;
  console.log(`\n--- ${game} / you ${role} vs ${opponent} -------------------------------`);
  await drv.go(url);

  await waitFor(drv, 'window.__s2c && window.__s2c.session', { label: 'the session to boot' });
  await waitFor(drv, 'window.__s2c.session.match.hud().step > 40', { label: '40 control steps' });
  await sleep(300);
  const bytes = await drv.shot(`game_${game}_${role}.png`);

  const info = await drv.script(`() => {
    const s = window.__s2c.session;
    const h = s.match.hud();
    const st = s.match.state();
    const cv = s.renderer.renderer.domElement;
    const g = cv.getContext('webgl2') || cv.getContext('webgl');
    // sample the framebuffer: a dead renderer paints one flat colour
    const px = new Uint8Array(4 * 16);
    let uniq = new Set();
    if (g) {
      for (let i = 0; i < 16; i++) {
        const x = Math.floor((i % 4 + 0.5) * cv.width / 4);
        const y = Math.floor((Math.floor(i / 4) + 0.5) * cv.height / 4);
        const one = new Uint8Array(4);
        g.readPixels(x, y, 1, 1, g.RGBA, g.UNSIGNED_BYTE, one);
        uniq.add(one.join(','));
      }
    }
    return {
      step: h.step, phase: h.phase, fps: s.fps,
      stepMs: h.perf && h.perf.lastStepMs,
      timeLeft: h.timeLeft,
      verdict: h.verdict ? h.verdict.terminal : null,
      you: st.seats[st.playerSeat], ai: st.seats[st.aiSeat],
      canvas: [cv.width, cv.height],
      distinctPixels: uniq.size,
      // A hidden element that an author display rule keeps on screen is
      // invisible to a hidden-attribute check, so ask the browser what it paints.
      countdownVisible: (() => {
        const c = document.querySelector('.hud-center');
        if (!c) return '';
        const vis = getComputedStyle(c).display !== 'none';
        return vis ? 'showing ' + (c.textContent || '').trim().slice(0, 24) : '';
      })(),
      log: (document.getElementById('log') || {}).textContent || '',
    };
  }`);

  console.log(`  step ${info.step}  phase ${info.phase}  fps ${info.fps?.toFixed(0)}  ` +
    `step ${info.stepMs?.toFixed(2)} ms  canvas ${info.canvas.join('x')}  ` +
    `you(${info.you.x.toFixed(2)}, ${info.you.y.toFixed(2)})  ai(${info.ai.x.toFixed(2)}, ${info.ai.y.toFixed(2)})`);

  ok(info.step > 40, 'the browser advanced control steps', `step ${info.step}`);
  ok(!info.countdownVisible, 'the countdown overlay is gone once the match runs', info.countdownVisible || '');
  ok(Number.isFinite(info.you.x) && Number.isFinite(info.ai.x), 'positions are finite');
  ok(info.distinctPixels > 1, 'the canvas is drawing a scene, not one flat colour', `${info.distinctPixels}/16 distinct samples`);
  ok(bytes > 40000, 'the screenshot has real content', `${bytes} B`);
  ok(!info.log.trim(), 'no console error mirrored into #log', info.log.trim().slice(0, 300));
  ok((info.stepMs ?? 99) < 20, 'a control step fits in its 20 ms budget in the browser', `${info.stepMs?.toFixed(2)} ms`);

  // let it run to a verdict (or the clock), then shoot the result card
  await waitFor(drv, 'window.__s2c.session.match.hud().verdict || window.__s2c.session.match.hud().step >= 499',
    { timeout: 120000, label: 'a verdict' });
  await sleep(700);
  await drv.shot(`game_${game}_${role}_result.png`);
  const end = await drv.script(`() => { const h = window.__s2c.session.match.hud();
    return { terminal: h.verdict && h.verdict.terminal, winner: h.verdict && h.verdict.winner, step: h.step,
             log: (document.getElementById('log')||{}).textContent || '' }; }`);
  console.log(`  verdict ${end.terminal || '(clock)'} winner ${end.winner || '-'} at step ${end.step}`);
  ok(!!end.terminal, 'the episode ended with a referee terminal', end.terminal || '');
  ok(!end.log.trim(), 'still no console error at the end', end.log.trim().slice(0, 300));
  return { game, role, ...end, fps: info.fps, stepMs: info.stepMs };
}

async function main() {
  await mkdir(SHOTS, { recursive: true });
  const server = await serve();
  console.log(`http://127.0.0.1:${PORT}/  (root ${ROOT})`);
  const gd = spawn('geckodriver', ['--port', String(GD_PORT), '--host', '127.0.0.1'], { stdio: ['ignore', 'pipe', 'pipe'] });
  let gdlog = ''; gd.stdout.on('data', (d) => (gdlog += d)); gd.stderr.on('data', (d) => (gdlog += d));
  const drv = new Driver(GD_PORT);
  const out = [];
  try {
    await drv.start();
    await drv.setRect(1440, 900);

    console.log('\n--- title screen ------------------------------------------------');
    await drv.go(`${BASE || `http://127.0.0.1:${PORT}`}/index.html`);
    await waitFor(drv, 'document.querySelector(".choice-game")', { label: 'the title screen' });
    await sleep(400);
    await drv.shot('game_title.png');
    const offered = await drv.script(`() => [...document.querySelectorAll('[data-game]')].map(e => e.dataset.game)`);
    console.log(`  games offered: ${offered.join(', ')}`);
    ok(offered.includes('asym'), 'the asymmetric game is offered', offered.join(','));
    ok(offered.includes('sym'), 'the symmetric game is offered', offered.join(','));

    for (const role of ['attacker', 'defender']) out.push(await runCase(drv, { role }));
    out.push(await runCase(drv, { game: 'sym', role: 'A' }));
  } catch (err) {
    failures += 1;
    console.error('\nHARNESS ERROR:', err.message);
    console.error(gdlog.split('\n').slice(-12).join('\n'));
  } finally {
    await drv.quit();
    gd.kill('SIGTERM');
    if (!KEEP) server.close();
  }

  console.log(`\n${'='.repeat(70)}`);
  for (const r of out) {
    console.log(`  ${r.game.padEnd(4)} ${r.role.padEnd(9)} -> ${String(r.terminal).padEnd(14)} winner ${String(r.winner).padEnd(9)} ` +
      `${r.fps?.toFixed(0)} fps, ${r.stepMs?.toFixed(2)} ms/step`);
  }
  console.log(failures ? `${failures} CHECK(S) FAILED` : 'ALL BROWSER CHECKS PASSED');
  console.log('='.repeat(70));
  if (!KEEP) process.exit(failures ? 1 : 0);
}

main();

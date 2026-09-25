/**
 * tests/shoot_filter.mjs — the S2C certificate in a REAL browser, headless.
 *
 *   node tests/shoot_filter.mjs          # tests/shots/filter_*.png + a pass/fail summary
 *   node tests/shoot_filter.mjs --keep   # leave the server up afterwards
 *
 * tests/shoot_game.mjs proves the page runs. This one proves the FILTER runs in
 * the page, and measures what it costs there — the one number that cannot be
 * taken from node, because the browser's JIT, its float32 arrays and its
 * animation-frame budget are not node's.
 *
 * It boots app/main.js with `?filter=both`, holds W so the human's dog charges
 * the opponent, and records:
 *   - that both seats report a live certificate (alpha, V, decision)
 *   - that the player is VISIBLY held back: the certificate intervenes, the two
 *     hulls never touch, and the SHIELD chip lights up in the HUD
 *   - the measured per-robot filter cost and the whole control step, in ms
 *
 * Same server + geckodriver machinery as tests/shoot_game.mjs; nothing has to be
 * running beforehand.
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
const PORT = Number(flag('--port', '8793'));
const GD_PORT = Number(flag('--driver-port', '4483'));
const KEEP = argv.includes('--keep');

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

async function waitFor(drv, expr, { timeout = 90000, every = 400, label = 'condition' } = {}) {
  const t0 = Date.now();
  while (Date.now() - t0 < timeout) {
    const v = await drv.script(`() => { try { return !!(${expr}); } catch (e) { return false; } }`);
    if (v) return true;
    await sleep(every);
  }
  throw new Error(`timed out waiting for ${label} (${Math.round((Date.now() - t0) / 1000)} s)`);
}

/** Install a recorder + a scripted hand on the live page. Returns nothing. */
const ARM = `(holdAt) => {
  const S = window.__s2c.session;
  const rec = { rows: [], contact: 0, frozen: null };
  window.__rec = rec;
  // Hold W through the REAL input path, so the command goes through the same
  // slew limiter and command box a player's keyboard does.
  const down = () => window.dispatchEvent(new KeyboardEvent('keydown', { code: 'KeyW', bubbles: true }));
  down();
  rec.timer = setInterval(down, 200);   // Firefox drops synthetic key state on blur
  const tick = S.match.tick;
  S.match.tick = function (dt) {
    if (rec.frozen) return rec.frozen;            // hold the frame for a screenshot
    const h = tick.call(this, dt);
    const fa = h.filter.attacker, fd = h.filter.defender;
    if (fa || fd) {
      rec.rows.push({
        step: h.step,
        stepMs: h.perf.lastStepMs,
        a: fa ? { alpha: fa.alpha, v: fa.value, dec: fa.decision, it: fa.iters, q: fa.qEvals, ms: fa.stepMs } : null,
        d: fd ? { alpha: fd.alpha, v: fd.value, dec: fd.decision, it: fd.iters, q: fd.qEvals, ms: fd.stepMs } : null,
      });
    }
    if (h.contact) rec.contact += 1;
    // Freeze on the money frame: the human's certificate hard at work.
    if (holdAt && fa && fa.alpha > holdAt) rec.frozen = h;
    return h;
  };
}`;

async function main() {
  await mkdir(SHOTS, { recursive: true });
  const server = await serve();
  console.log(`http://127.0.0.1:${PORT}/  (root ${ROOT})`);
  const gd = spawn('geckodriver', ['--port', String(GD_PORT), '--host', '127.0.0.1'], { stdio: ['ignore', 'pipe', 'pipe'] });
  let gdlog = ''; gd.stdout.on('data', (d) => (gdlog += d)); gd.stderr.on('data', (d) => (gdlog += d));
  const drv = new Driver(GD_PORT);
  let summary = null;

  try {
    await drv.start();
    await drv.setRect(1440, 900);

    // ---- setup screen: the two checkboxes must be live -----------------------
    console.log('\n--- setup screen: the certificate is offered ---------------------');
    await drv.go(`http://127.0.0.1:${PORT}/index.html`);
    await waitFor(drv, 'document.querySelector("#opt-filter-ai")', { label: 'the setup screen' });
    const boxes = await drv.script(`() => {
      const a = document.getElementById('opt-filter-ai');
      const y = document.getElementById('opt-filter-you');
      return { ai: !a.disabled, you: !y.disabled, cap: !!window.__s2c.ui.capabilities.filter,
               note: (document.querySelector('.step-note') ? '' : '') };
    }`);
    ok(boxes.cap, 'capabilities.filter is true in the shipped page');
    ok(boxes.ai && boxes.you, 'both filter checkboxes are enabled, not greyed out');
    await drv.shot('filter_setup.png');

    // ---- the run: both seats shielded, the human ramming ---------------------
    console.log('\n--- asym / you attack, BOTH seats behind the certificate ---------');
    await drv.go(`http://127.0.0.1:${PORT}/index.html?game=asym&role=attacker&opponent=s2c&filter=both&autostart=1`);
    await waitFor(drv, 'window.__s2c && window.__s2c.session', { timeout: 120000, label: 'the session to boot' });

    const wired = await drv.script(`() => {
      const i = window.__s2c.session.match.info();
      return { paths: i.actionPaths, gains: i.gains };
    }`);
    console.log(`  action paths: ${JSON.stringify(wired.paths)}`);
    ok(wired.paths.attacker === 'qcbf_shield' && wired.paths.defender === 'qcbf_shield',
      'both seats run the qcbf_shield action path');

    await drv.script(ARM, [0.6]);
    // The loop FREEZES itself on the first frame where the human's certificate
    // is past alpha 0.6, so the screenshot lands on that frame instead of
    // whatever the shutter happens to catch 200 ms later.
    await waitFor(drv, 'window.__rec.frozen', { timeout: 90000, every: 120,
      label: 'the player being held back' });
    await sleep(400);
    await drv.shot('filter_intervening.png');
    const atMoment = await drv.script(`() => {
      const h = window.__s2c.session.match.hud();
      const chips = [...document.querySelectorAll('.chip-shield')].map(e => !e.hidden);
      const out = { step: h.step, chips,
        a: h.filter.attacker, d: h.filter.defender,
        fps: window.__s2c.session.fps };
      window.__rec.frozen = null;          // ... and run on
      return out;
    }`);
    console.log(`  at step ${atMoment.step}: you alpha ${atMoment.a.alpha.toFixed(3)} ` +
      `(${atMoment.a.decisionName}, V ${atMoment.a.value.toFixed(5)})  |  ` +
      `AI alpha ${atMoment.d.alpha.toFixed(3)} (${atMoment.d.decisionName}, V ${atMoment.d.value.toFixed(5)})`);
    ok(atMoment.chips.length === 2 && atMoment.chips[0] && atMoment.chips[1],
      'both HUD SHIELD chips are lit on the frame the screenshot caught',
      `chips ${JSON.stringify(atMoment.chips)}`);

    await waitFor(drv, 'window.__s2c.session.match.hud().verdict || window.__s2c.session.match.hud().step >= 499',
      { timeout: 180000, label: 'a verdict' });
    await sleep(600);
    await drv.shot('filter_result.png');

    summary = await drv.script(`() => {
      clearInterval(window.__rec.timer);
      const rec = window.__rec, h = window.__s2c.session.match.hud();
      const num = (xs) => { xs = xs.slice().sort((a,b)=>a-b);
        return { n: xs.length, med: xs[Math.floor(xs.length/2)] ?? 0,
                 p95: xs[Math.floor(xs.length*0.95)] ?? 0, max: xs[xs.length-1] ?? 0,
                 mean: xs.reduce((s,v)=>s+v,0)/Math.max(xs.length,1) }; };
      const side = (k) => {
        const rows = rec.rows.filter(r => r[k]);
        const dec = [0,0,0,0,0,0];
        for (const r of rows) dec[r[k].dec] += 1;
        return { ms: num(rows.map(r=>r[k].ms)), qEvals: num(rows.map(r=>r[k].q)),
                 iters: num(rows.filter(r=>r[k].dec===2).map(r=>r[k].it)),
                 dec, interv: rows.filter(r=>r[k].alpha>0).length,
                 maxAlpha: Math.max(...rows.map(r=>r[k].alpha)),
                 minV: Math.min(...rows.map(r=>r[k].v)) };
      };
      return {
        steps: rec.rows.length, contactSteps: rec.contact,
        verdict: h.verdict ? h.verdict.terminal : null,
        winner: h.verdict ? h.verdict.winner : null,
        fps: window.__s2c.session.fps,
        stepMs: num(rec.rows.map(r=>r.stepMs)),
        you: side('a'), ai: side('d'),
        log: (document.getElementById('log') || {}).textContent || '',
      };
    }`);

    const D = ['task_pass','fallback','line_search','guard','handback','qp'];
    const show = (t, s) => {
      console.log(`  ${t}`);
      console.log(`     filter cost  median ${s.ms.med.toFixed(2)} ms  p95 ${s.ms.p95.toFixed(2)}  max ${s.ms.max.toFixed(2)}`);
      console.log(`     robust_q evals per step  median ${s.qEvals.med}  max ${s.qEvals.max}` +
        (s.iters.n ? `   (line search ${s.iters.n} steps, median ${s.iters.med} iters, max ${s.iters.max})` : ''));
      console.log(`     decisions  ${s.dec.map((v,i)=>v?`${v} ${D[i]}`:null).filter(Boolean).join(', ')}`);
      console.log(`     intervened ${s.interv}/${summary.steps} steps, max alpha ${s.maxAlpha.toFixed(3)}, min V ${s.minV.toFixed(5)}`);
    };
    console.log(`\n  verdict ${summary.verdict} (winner ${summary.winner}) after ${summary.steps} steps, ${summary.fps.toFixed(0)} fps`);
    show('YOU (walk policy behind the certificate, holding W):', summary.you);
    show('AI  (S2C defender behind the certificate):', summary.ai);
    console.log(`  WHOLE control step (2 policies + 2 filters + obs + 4 physics substeps):`);
    console.log(`     median ${summary.stepMs.med.toFixed(2)} ms   p95 ${summary.stepMs.p95.toFixed(2)} ms   ` +
      `max ${summary.stepMs.max.toFixed(2)} ms   of a 20 ms budget`);

    ok(!summary.log.trim(), 'no console error mirrored into #log', summary.log.trim().slice(0, 300));
    ok(summary.you.interv > 0 && summary.ai.interv > 0,
      'the certificate intervened on BOTH seats',
      `you ${summary.you.interv}, AI ${summary.ai.interv}`);
    ok(summary.contactSteps === 0,
      'the ramming player never reached the opponent (no robot-robot contact frame)',
      `${summary.contactSteps} contact steps`);
    ok(summary.stepMs.p95 < 20,
      'p95 of the WHOLE control step fits in the 20 ms it simulates',
      `${summary.stepMs.p95.toFixed(2)} ms`);
    ok(summary.stepMs.max < 20,
      'so does the worst step seen',
      `${summary.stepMs.max.toFixed(2)} ms`);
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
  console.log(failures ? `${failures} CHECK(S) FAILED` : 'ALL BROWSER FILTER CHECKS PASSED');
  console.log('='.repeat(70));
  if (!KEEP) process.exit(failures ? 1 : 0);
}

main();

#!/usr/bin/env node
/**
 * tests/render_shots.mjs — headless-Firefox screenshots of app/render.js.
 *
 *   node tests/render_shots.mjs                    # every shot, both scenes
 *   node tests/render_shots.mjs --shot chase       # one shot
 *   node tests/render_shots.mjs --scene asym
 *   node tests/render_shots.mjs --bench            # + a 300-frame timing run
 *   node tests/render_shots.mjs --keep             # keep the served page alive for a look
 *
 * WHY A SERVER AND A GATE. `firefox --screenshot` fires as soon as the page's
 * load event does, which for a WASM + mesh + WebGL page is far too early. The
 * harness therefore requests `/__gate` from an <img>; this server holds that
 * request open until the page reports `/__ready`, so the load event — and the
 * screenshot with it — cannot happen before the renderer has drawn its frames.
 * The page also posts its log back through /__ready, so the shot and its numbers
 * are captured in the same run.
 *
 * SNAP CONFINEMENT (this box). The packaged Firefox cannot read or write /tmp,
 * and it refuses a profile outside the user's home, so both the profile and the
 * PNG are written under $HOME/snap/firefox/common/ and the PNG is then moved
 * into tests/shots/.
 */

import http from 'node:http';
import fs from 'node:fs';
import fsp from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, '..');
const SHOT_DIR = path.join(ROOT, 'tests', 'shots');
/** snap-writable staging area; see the header. */
const SNAP = path.join(os.homedir(), 'snap', 'firefox', 'common');
const STAGE = path.join(SNAP, 's2c_render_shots');
const PROFILE = path.join(SNAP, 's2c_render_profile');

const MIME = {
  '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8', '.json': 'application/json',
  '.wasm': 'application/wasm', '.xml': 'application/xml', '.obj': 'text/plain',
  '.png': 'image/png', '.css': 'text/css', '.bin': 'application/octet-stream',
  '.map': 'application/json',
};

const argv = process.argv.slice(2);
const flag = (n) => argv.includes(n);
const val = (n, d) => { const i = argv.indexOf(n); return i >= 0 ? argv[i + 1] : d; };

const ALL_SHOTS = ['broadcast', 'chase', 'chase_far', 'fpv', 'orbit', 'swap', 'collision', 'spawn'];
const shots = val('--shot') ? [val('--shot')] : ALL_SHOTS;
const scenes = val('--scene') ? [val('--scene')] : ['sym', 'asym'];
const BENCH = flag('--bench');
const W = +(val('--w', '1600')), H = +(val('--h', '1000'));

// ---------------------------------------------------------------- server ---
const gates = new Map();     // id -> {res, resolve}
const results = new Map();   // id -> harness result JSON
const pngs = new Map();      // id -> PNG bytes read back from the WebGL canvas

function serve() {
  const server = http.createServer(async (req, res) => {
    const u = new URL(req.url, 'http://127.0.0.1');
    if (u.pathname === '/__gate') {
      const id = u.searchParams.get('id') || 'x';
      const g = gates.get(id);
      if (g) g.res = res; else gates.set(id, { res });
      return; // held open on purpose
    }
    if (u.pathname === '/__ready') {
      const id = u.searchParams.get('id') || 'x';
      const g = gates.get(id);
      if (g && g.res) {
        // 1x1 transparent PNG closes the <img> and releases the load event
        const px = Buffer.from(
          'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==',
          'base64');
        g.res.writeHead(200, { 'content-type': 'image/png', 'content-length': px.length });
        g.res.end(px);
        g.res = null;
      }
      if (g && g.resolve) g.resolve();
      res.writeHead(204).end();
      return;
    }
    if (u.pathname === '/__png' && req.method === 'POST') {
      let body = '';
      for await (const c of req) body += c;
      const m = /^data:image\/png;base64,(.*)$/s.exec(body.trim());
      if (m) pngs.set(u.searchParams.get('id') || 'x', Buffer.from(m[1], 'base64'));
      res.writeHead(204).end();
      return;
    }
    if (u.pathname === '/__result' && req.method === 'POST') {
      let body = '';
      for await (const c of req) body += c;
      results.set(u.searchParams.get('id') || 'x', body);
      res.writeHead(204).end();
      return;
    }
    // static
    const rel = decodeURIComponent(u.pathname).replace(/^\/+/, '') || 'index.html';
    const file = path.resolve(ROOT, rel);
    if (!file.startsWith(ROOT)) { res.writeHead(403).end('no'); return; }
    try {
      const st = await fsp.stat(file);
      if (st.isDirectory()) { res.writeHead(404).end('dir'); return; }
      res.writeHead(200, {
        'content-type': MIME[path.extname(file)] || 'application/octet-stream',
        'content-length': st.size,
        'cache-control': 'no-store',
      });
      fs.createReadStream(file).pipe(res);
    } catch {
      res.writeHead(404).end('not found: ' + rel);
    }
  });
  return new Promise((ok) => server.listen(0, '127.0.0.1', () => ok(server)));
}

// --------------------------------------------------------------- firefox ---
function firefox(url, png) {
  return new Promise((resolve) => {
    const args = ['--headless', '--new-instance', '--profile', PROFILE,
      `--window-size=${W},${H}`, '--screenshot', png, url];
    const p = spawn('firefox', args, {
      env: { ...process.env, MOZ_NO_REMOTE: '1' },
      stdio: ['ignore', 'pipe', 'pipe'],
    });
    let err = '';
    p.stderr.on('data', (d) => { err += d; });
    const t = setTimeout(() => { p.kill('SIGTERM'); }, 180000);
    p.on('close', (code) => { clearTimeout(t); resolve({ code, err }); });
  });
}

// ------------------------------------------------------------------ main ---
const server = await serve();
const port = server.address().port;
await fsp.mkdir(SHOT_DIR, { recursive: true });
await fsp.rm(STAGE, { recursive: true, force: true });
await fsp.mkdir(STAGE, { recursive: true });
await fsp.rm(PROFILE, { recursive: true, force: true });
await fsp.mkdir(PROFILE, { recursive: true });

console.log(`serving ${ROOT} on http://127.0.0.1:${port}`);
let fails = 0;
const report = [];

for (const scene of scenes) {
  for (const shot of shots) {
    const id = `${scene}_${shot}`;
    const stage = path.join(STAGE, `${id}.png`);
    const out = path.join(SHOT_DIR, `render_${id}.png`);
    const url = `http://127.0.0.1:${port}/tests/render_harness.html`
      + `?scene=${scene}&shot=${shot}&w=${W}&h=${H}&gate=1&gateid=${id}`
      + (BENCH ? '&bench=1' : '');
    const t0 = Date.now();
    const { code, err } = await firefox(url, stage);
    const buf = pngs.get(id);
    let ok = !!buf && buf.length > 5000, size = buf ? buf.length : 0;
    if (ok) await fsp.writeFile(out, buf);
    await fsp.rm(stage, { force: true });
    const secs = ((Date.now() - t0) / 1000).toFixed(1);
    let res = null;
    try { res = JSON.parse(results.get(id)); } catch { /* none */ }
    if (res && res.ok === false) ok = false;
    console.log(`${ok ? 'ok  ' : 'FAIL'}  ${id.padEnd(20)} ${String(size).padStart(8)} B  ${secs}s`);
    for (const l of (res && res.lines) || []) console.log('        | ' + l);
    if (res && res.error) console.log('        ! ' + res.error.split('\n').slice(0, 4).join('\n        ! '));
    if (!ok && !res) console.log(`        ! firefox exit=${code} ${err.split('\n').filter(Boolean).slice(-2).join(' | ')}`);
    report.push({ id, ok, bytes: size, seconds: +secs, png: ok ? out : null, result: res });
    if (!ok) fails++;
  }
}

server.close();
await fsp.rm(STAGE, { recursive: true, force: true });
await fsp.rm(PROFILE, { recursive: true, force: true });
console.log(`\n${report.filter((r) => r.ok).length}/${report.length} shots written to ${SHOT_DIR}`);
process.exit(fails ? 1 : 0);

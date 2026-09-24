/**
 * tests/shoot_ui.mjs — headless Firefox screenshots of the interface.
 *
 *   node tests/shoot_ui.mjs               # writes tests/shots/ui_*.png
 *   node tests/shoot_ui.mjs --keep        # leave the http server up for a manual look
 *   node tests/shoot_ui.mjs --port 8788
 *
 * No npm dependencies: a small static server from node:http, geckodriver over plain WebDriver
 * HTTP, screenshots returned as base64 and written here. Firefox is a snap on this box and cannot
 * read file:// paths outside its own sandbox, so everything goes over 127.0.0.1.
 *
 * The page is loaded as index.html?uipreview=1 — the documented test hook that boots
 * tests/ui_preview.mjs (no wasm, no weights) instead of app/main.js.
 */

import { createServer } from 'node:http';
import { spawn } from 'node:child_process';
import { readFile, mkdir, writeFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { extname, join, normalize, resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(HERE, '..');
const SHOTS = join(HERE, 'shots');

const argv = process.argv.slice(2);
const flag = (name, dflt) => {
  const i = argv.indexOf(name);
  return i >= 0 ? argv[i + 1] : dflt;
};
const PORT = Number(flag('--port', '8787'));
const GD_PORT = Number(flag('--driver-port', '4477'));
const KEEP = argv.includes('--keep');

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.wasm': 'application/wasm',
  '.png': 'image/png',
  '.svg': 'image/svg+xml',
  '.obj': 'text/plain; charset=utf-8',
  '.bin': 'application/octet-stream',
  '.xml': 'application/xml',
};

function serve() {
  const server = createServer(async (req, res) => {
    try {
      const url = new URL(req.url, 'http://127.0.0.1');
      let p = decodeURIComponent(url.pathname);
      if (p.endsWith('/')) p += 'index.html';
      const abs = join(ROOT, normalize(p).replace(/^(\.\.[/\\])+/, ''));
      if (!abs.startsWith(ROOT)) {
        res.writeHead(403).end('no');
        return;
      }
      const body = await readFile(abs);
      res.writeHead(200, {
        'content-type': MIME[extname(abs)] || 'application/octet-stream',
        'cache-control': 'no-store',
      });
      res.end(req.method === 'HEAD' ? undefined : body);
    } catch {
      res.writeHead(404, { 'content-type': 'text/plain' }).end('404');
    }
  });
  return new Promise((ok) => server.listen(PORT, '127.0.0.1', () => ok(server)));
}

/* --------------------------------------------------------------- WebDriver */

class Driver {
  constructor(port) {
    this.base = `http://127.0.0.1:${port}`;
    this.sid = null;
  }
  async call(method, path, body) {
    const res = await fetch(this.base + path, {
      method,
      headers: { 'content-type': 'application/json' },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
    const json = await res.json();
    if (json && json.value && json.value.error) {
      throw new Error(`${json.value.error}: ${json.value.message}`);
    }
    return json.value;
  }
  async start() {
    for (let i = 0; i < 100; i++) {
      try {
        await fetch(this.base + '/status');
        break;
      } catch {
        await sleep(150);
      }
    }
    const s = await this.call('POST', '/session', {
      capabilities: {
        alwaysMatch: {
          'moz:firefoxOptions': {
            args: ['-headless'],
            prefs: {
              'browser.tabs.remote.autostart': false,
              'devtools.console.stdout.content': true,
            },
          },
        },
      },
    });
    this.sid = s.sessionId;
  }
  go(url) {
    return this.call('POST', `/session/${this.sid}/url`, { url });
  }
  script(fn, args = []) {
    return this.call('POST', `/session/${this.sid}/execute/sync`, {
      script: `return (${fn}).apply(null, arguments);`,
      args,
    });
  }
  setRect(width, height) {
    return this.call('POST', `/session/${this.sid}/window/rect`, { width, height, x: 0, y: 0 });
  }
  async fitViewport(w, hh) {
    await this.setRect(w, hh);
    const got = await this.script('() => [window.innerWidth, window.innerHeight]');
    if (!got) return;
    const dw = w - got[0];
    const dh = hh - got[1];
    if (dw || dh) await this.setRect(w + dw, hh + dh);
  }
  async shot(name) {
    const b64 = await this.call('GET', `/session/${this.sid}/screenshot`);
    const file = join(SHOTS, name);
    await writeFile(file, Buffer.from(b64, 'base64'));
    const bytes = Buffer.from(b64, 'base64').length;
    console.log(`  shot ${name}  ${bytes.toLocaleString()} B`);
    return file;
  }
  quit() {
    return this.call('DELETE', `/session/${this.sid}`).catch(() => {});
  }
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/* --------------------------------------------------------------- the run */

const DESKTOP = [1440, 900];
const MOBILE = [390, 844];

async function main() {
  await mkdir(SHOTS, { recursive: true });
  if (!existsSync(join(ROOT, 'index.html'))) throw new Error('run me from the repo');

  const server = await serve();
  console.log(`http://127.0.0.1:${PORT}/  (root ${ROOT})`);

  const gd = spawn('geckodriver', ['--port', String(GD_PORT), '--host', '127.0.0.1'], {
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  let gdlog = '';
  gd.stdout.on('data', (d) => (gdlog += d));
  gd.stderr.on('data', (d) => (gdlog += d));

  const drv = new Driver(GD_PORT);
  const problems = [];
  try {
    await drv.start();
    const url = (q) => `http://127.0.0.1:${PORT}/index.html?uipreview=1${q ? '&' + q : ''}`;

    await drv.fitViewport(...DESKTOP);

    /* 1 — title / setup, symmetric (the default) */
    await drv.go(url());
    await waitReady(drv);
    await drv.shot('ui_01_title_sym.png');

    /* 2 — asymmetric picked: the role step appears */
    await drv.script(`() => document.querySelector('[data-game="asym"]').click()`);
    await sleep(160);
    await drv.shot('ui_02_setup_asym_roles.png');

    /* 3 — defender role + a different opponent, scrolled to the roster */
    await drv.script(`() => document.querySelector('[data-role="defender"]').click()`);
    await sleep(120);
    await drv.script(`() => document.querySelector('[data-method="cpo"]').click()`);
    await sleep(120);
    await drv.script(
      `() => { const g = document.querySelector('.opp-grid'); g.scrollIntoView({block:'center'}); }`
    );
    await sleep(200);
    await drv.shot('ui_03_opponents_asym_defender.png');

    /* 4 — loading */
    await drv.script(`() => window.__s2cPreview.showLoading()`);
    await sleep(250);
    await drv.shot('ui_04_loading.png');

    /* 5 — in-game HUD, symmetric, mid-episode with a live command */
    await drv.go(url('game=sym&opponent=s2c'));
    await waitReady(drv);
    await drv.script(
      `() => window.__s2cPreview.showGame({ step: 148, cmd: { vx: 2.1, vy: 0.42, wz: -0.9 } })`
    );
    await sleep(250);
    await drv.shot('ui_05_hud_sym.png');

    /* 6 — countdown, asymmetric, as the attacker, with the AI shield chip lit */
    await drv.go(url('game=asym&role=attacker&opponent=nom'));
    await waitReady(drv);
    await drv.script(
      `() => window.__s2cPreview.showGame({ step: 0, countdownS: 2.4, filterActive: true })`
    );
    await sleep(250);
    await drv.shot('ui_06_hud_asym_countdown.png');

    /* 7 — result: you scored */
    await drv.script(
      `() => window.__s2cPreview.showResult({ terminal: 'touchdown', winner: 'A',
         stats: { steps: 212, timeS: 4.24, closest: 0.0 } })`
    );
    await sleep(250);
    await drv.shot('ui_07_result_touchdown.png');

    /* 8 — result: you initiated the collision */
    await drv.script(
      `() => { window.__s2cPreview.ui.hideResult();
               window.__s2cPreview.showResult({ terminal: 'trunk_contact', winner: 'B',
                 stats: { steps: 96, timeS: 1.92 } }); }`
    );
    await sleep(250);
    await drv.shot('ui_08_result_collision.png');

    /* 9 — pause */
    await drv.script(
      `() => { window.__s2cPreview.ui.hideResult(); window.__s2cPreview.ui.setPaused(true); }`
    );
    await sleep(200);
    await drv.shot('ui_09_pause.png');

    /* 10 — settings panel */
    await drv.script(
      `() => { window.__s2cPreview.ui.setPaused(false);
               document.querySelectorAll('.game-bar .btn-icon')[1].click(); }`
    );
    await sleep(220);
    await drv.shot('ui_10_settings.png');

    /* 11 — the honest-differences panel, from the title screen */
    await drv.go(url());
    await waitReady(drv);
    await drv.script(`() => document.querySelector('.card-about .btn').click()`);
    await sleep(220);
    await drv.shot('ui_11_about.png');

    /* 12 — deep link + autostart runs the whole flow unattended */
    await drv.go(url('game=sym&role=A&opponent=lag&autostart=1'));
    await waitReady(drv);
    await sleep(1800);
    const phase = await drv.script(`() => document.querySelector('.ui').dataset.phase`);
    if (phase !== 'game' && phase !== 'loading') {
      problems.push(`autostart deep link ended in phase "${phase}" (expected loading|game)`);
    }
    await drv.shot('ui_12_deeplink_autostart.png');

    /* 13 / 14 — mobile */
    await drv.fitViewport(...MOBILE);
    await drv.go(url());
    await waitReady(drv);
    await sleep(200);
    const overflow = await drv.script(
      `() => document.documentElement.scrollWidth - document.documentElement.clientWidth`
    );
    if (overflow > 0) problems.push(`mobile title overflows horizontally by ${overflow}px`);
    await drv.shot('ui_13_mobile_title.png');

    await drv.script(`() => window.__s2cPreview.showGame({ step: 120 })`);
    await sleep(250);
    const overflow2 = await drv.script(
      `() => document.documentElement.scrollWidth - document.documentElement.clientWidth`
    );
    if (overflow2 > 0) problems.push(`mobile HUD overflows horizontally by ${overflow2}px`);
    await drv.shot('ui_14_mobile_hud.png');

    /* console hygiene */
    const errs = await drv.script(`() => (window.__s2cPreviewErrors || []).map(e => e.level + ': ' + e.text)`);
    for (const e of errs || []) problems.push('console ' + e);
  } finally {
    await drv.quit();
    gd.kill();
    if (!KEEP) server.close();
  }

  if (problems.length) {
    console.log('\nPROBLEMS');
    for (const p of problems) console.log('  ! ' + p);
    process.exitCode = 1;
  } else {
    console.log('\nOK — every shot taken, no console output, no horizontal overflow.');
  }
  if (KEEP) console.log(`server still up on http://127.0.0.1:${PORT}/index.html?uipreview=1`);
}

async function waitReady(drv) {
  for (let i = 0; i < 80; i++) {
    const ok = await drv.script(
      `() => Boolean(window.__s2cPreview) || Boolean(document.querySelector('.boot-error'))`
    );
    if (ok) break;
    await sleep(100);
  }
  const boot = await drv.script(
    `() => { const e = document.querySelector('.boot-error'); return e ? e.textContent : null; }`
  );
  if (boot) throw new Error('page failed to boot: ' + boot.slice(0, 400));
  await sleep(120);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});

#!/usr/bin/env node
/**
 * tests/render_alloc_lint.mjs — prove the renderer's per-frame path allocates nothing.
 *
 *   node tests/render_alloc_lint.mjs
 *
 * "60 FPS on a laptop" is only true if update() does not hand the GC work every
 * frame. Browsers give no allocation counter we can assert on (performance.memory
 * is Chrome-only and coarse), so this checks the thing that actually causes the
 * garbage: the source of every function reachable from update() must contain no
 * allocating construct.
 *
 * A line may opt out with a trailing `// alloc-ok: <reason>` marker, which makes
 * each exception explicit and reviewable instead of invisible.
 */

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const SRC = fs.readFileSync(path.join(ROOT, 'app', 'render.js'), 'utf8');

/** Everything update() can reach on a normal frame. */
const HOT = [
  'update', 'sameDynamic', 'writeDrawBuf', 'poseBodies', 'updateFocus',
  'updateRing', 'stepCamera', 'solveChase', 'solveBroadcast', 'fitDistance',
  'mjForward', 'settleStatics', 'playerFocus',
];

/** Constructs that allocate on the JS heap. */
const BAD = [
  [/\bnew\s+[A-Z]/, 'constructor call'],
  [/\bArray\.from\b/, 'Array.from'],
  [/\bObject\.(keys|values|entries|assign)\b/, 'Object.* allocates'],
  [/\.(map|filter|slice|concat|split|flat|reduce)\s*\(/, 'allocating array method'],
  [/=>\s*[^;)]*\{[^}]/, 'inline closure'],
  [/`/, 'template literal'],
  [/(^|[^\w.$])\[\s*[^\]\s]/, 'array literal'],
  [/(=|return|\(|,)\s*\{\s*\w+\s*:/, 'object literal'],
  [/\.toArray\s*\(/, '.toArray()'],
  [/JSON\./, 'JSON'],
  [/\.padStart|\.padEnd|\.toFixed|\.toString\s*\(/, 'string building'],
];

/**
 * Extract a function body by brace matching from `function <name>(`.
 * Good enough for this file: every hot function is a plain declaration.
 */
function bodyOf(name) {
  const re = new RegExp(`\\bfunction\\s+${name}\\s*\\(`);
  const m = re.exec(SRC);
  if (!m) return null;
  let i = SRC.indexOf('{', m.index + m[0].length - 1);
  if (i < 0) return null;
  const start = i;
  let depth = 0, inStr = null, inCmt = null;
  for (; i < SRC.length; i++) {
    const c = SRC[i], n = SRC[i + 1];
    if (inCmt === '//') { if (c === '\n') inCmt = null; continue; }
    if (inCmt === '/*') { if (c === '*' && n === '/') { inCmt = null; i++; } continue; }
    if (inStr) {
      if (c === '\\') { i++; continue; }
      if (c === inStr) inStr = null;
      continue;
    }
    if (c === '/' && n === '/') { inCmt = '//'; i++; continue; }
    if (c === '/' && n === '*') { inCmt = '/*'; i++; continue; }
    if (c === '"' || c === "'" || c === '`') { inStr = c; continue; }
    if (c === '{') depth++;
    else if (c === '}') { depth--; if (depth === 0) return { start, end: i + 1, text: SRC.slice(start, i + 1) }; }
  }
  return null;
}

const lineOf = (idx) => SRC.slice(0, idx).split('\n').length;

let problems = 0, checked = 0, exempt = 0, missing = [];
for (const fn of HOT) {
  const b = bodyOf(fn);
  if (!b) { missing.push(fn); continue; }
  checked++;
  const base = lineOf(b.start);
  const lines = b.text.split('\n');
  for (let i = 0; i < lines.length; i++) {
    let line = lines[i];
    if (/\/\/\s*alloc-ok/.test(line)) { exempt++; continue; }
    // strip comments and string bodies before matching
    const code = line.replace(/\/\/.*$/, '').replace(/\/\*.*?\*\//g, '');
    if (!code.trim()) continue;
    for (const [re, why] of BAD) {
      if (re.test(code)) {
        console.log(`ALLOC  ${fn}()  app/render.js:${base + i}  ${why}\n       ${code.trim()}`);
        problems++;
        break;
      }
    }
  }
}

if (missing.length) {
  console.log(`\nNOT FOUND (rename? then update HOT): ${missing.join(', ')}`);
}
console.log(`\n${checked}/${HOT.length} hot functions scanned, ${exempt} lines exempted, ${problems} allocating lines`);
if (missing.length || problems) {
  console.log('FAIL — the per-frame path allocates');
  process.exit(1);
}
console.log('PASS — the per-frame path allocates nothing');

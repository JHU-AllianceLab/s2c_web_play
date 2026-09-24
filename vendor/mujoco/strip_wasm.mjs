#!/usr/bin/env node
// Strip every custom section (section id 0) from a WebAssembly binary.
//
// @mujoco/mujoco 3.14.0 is built with `-g -gsource-map -s ASSERTIONS=1`
// (google-deepmind/mujoco:wasm/CMakeLists.txt), so 56 % of mujoco.wasm is
// DWARF + the `name` section. None of it is needed to instantiate or run the
// module; dropping it halves the download and measurably changes nothing
// (see vendor/mujoco/VERSION.md for the before/after numbers).
//
// Usage: node strip_wasm.mjs <in.wasm> <out.wasm>
//
// Binary format: magic(4) version(4) then a sequence of
//   section_id : u8
//   payload_len: LEB128 u32
//   payload    : payload_len bytes
// Custom sections are id 0; every other id is a standard section we keep.

import fs from 'node:fs';

function readLEB(buf, off) {
  let result = 0;
  let shift = 0;
  let pos = off;
  for (;;) {
    const byte = buf[pos++];
    if (byte === undefined) throw new Error(`truncated LEB128 at ${off}`);
    result |= (byte & 0x7f) << shift;
    if ((byte & 0x80) === 0) break;
    shift += 7;
    if (shift > 35) throw new Error(`LEB128 too long at ${off}`);
  }
  return { value: result >>> 0, next: pos };
}

export function stripCustomSections(buf) {
  if (buf.length < 8 || buf.readUInt32LE(0) !== 0x6d736100) {
    throw new Error('not a wasm binary (bad magic)');
  }
  const kept = [buf.subarray(0, 8)];
  const dropped = [];
  let off = 8;
  while (off < buf.length) {
    const start = off;
    const id = buf[off++];
    const { value: len, next } = readLEB(buf, off);
    off = next;
    const payloadStart = off;
    const payloadEnd = off + len;
    if (payloadEnd > buf.length) throw new Error(`section ${id} overruns file`);
    if (id === 0) {
      // Custom section: payload starts with a LEB-prefixed name.
      const { value: nameLen, next: nameStart } = readLEB(buf, payloadStart);
      const name = buf.toString('utf8', nameStart, nameStart + nameLen);
      dropped.push({ name, bytes: payloadEnd - start });
    } else {
      kept.push(buf.subarray(start, payloadEnd));
    }
    off = payloadEnd;
  }
  return { out: Buffer.concat(kept), dropped };
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const [, , inPath, outPath] = process.argv;
  if (!inPath || !outPath) {
    console.error('usage: node strip_wasm.mjs <in.wasm> <out.wasm>');
    process.exit(2);
  }
  const src = fs.readFileSync(inPath);
  const { out, dropped } = stripCustomSections(src);
  fs.writeFileSync(outPath, out);
  for (const d of dropped) console.log(`  dropped custom section "${d.name}" ${d.bytes} B`);
  console.log(`${inPath} ${src.length} B -> ${outPath} ${out.length} B ` +
              `(${(100 * (1 - out.length / src.length)).toFixed(1)} % smaller)`);
}

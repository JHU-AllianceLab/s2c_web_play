# vendor/mujoco — @mujoco/mujoco 3.14.0, single-thread, debug-stripped

Everything in this directory is DeepMind's official WASM build, vendored so the
site has no bundler, no npm install and no CDN. `app/physics.js` is the only
consumer; other agents should go through it rather than importing the glue.

## What is here

| file | bytes | gzip -9 | md5 | origin |
|---|---:|---:|---|---|
| `mujoco.js` | 293,092 | 67,994 | `4f5c1dc471f282f66a1fd89c3148d38d` | npm `@mujoco/mujoco@3.14.0`, **byte-identical** |
| `mujoco.wasm` | 4,519,987 | 1,261,674 | `7ab801a037c6facf7dbcb64359840330` | the npm wasm with every custom section stripped |
| `mujoco.d.ts` | 143,441 | 24,664 | `9969bbcac01fc15ae623ba27438fbfa0` | npm, byte-identical — reference only, never fetched |
| `upstream-package.json` | 1,291 | — | — | the npm manifest, kept for provenance |
| `strip_wasm.mjs` | 2,740 | — | — | the stripper, so the step is reproducible |

**Runtime download: 4,813,079 B raw / 1,329,668 B gzip** (`mujoco.js` +
`mujoco.wasm`). `mujoco.d.ts` and `upstream-package.json` are never requested by
the page. GitHub Pages gzips on the fly, so no `.gz` sibling is needed.

Upstream `mujoco.wasm` is 10,313,475 B / 2,524,774 B gz (md5
`de65be4d63617aabdaf8c4ba3b3a827a`). 56 % of it is DWARF + the `name` section
(`-g -gsource-map -s ASSERTIONS=1` in google-deepmind/mujoco:wasm/CMakeLists.txt).
Stripping the custom sections leaves only type/import/function/table/memory/
global/export/elem/datacount/code/data. **−5,793,488 B raw, −1,263,100 B gz, no
measured change in step cost.** It also removes the `sourceMappingURL` section,
so the browser never 404s on `mujoco.wasm.map`.

Not vendored: the `./mt` (pthreads + SharedArrayBuffer) build. GitHub Pages
cannot set COOP/COEP, so SharedArrayBuffer is unavailable and the multithreaded
build cannot be used. Do not add `coi-serviceworker`.

## Re-vendoring

```sh
npm pack @mujoco/mujoco@3.14.0 && tar xf mujoco-mujoco-3.14.0.tgz
cp package/mujoco.js package/mujoco.d.ts package/package.json vendor/mujoco/
node vendor/mujoco/strip_wasm.mjs package/mujoco.wasm vendor/mujoco/mujoco.wasm
node tests/node_physics_smoke.mjs --http        # must stay green
```

## Properties measured on THIS build

- `mj_versionString()` reports **3014000**. Compiling the two-robot scenes gives
  the same counts as native MuJoCo on the same XML: sym `nq 38 nv 36 nu 24
  nbody 30 ngeom 149 nmesh 32`, asym the same with `ngeom 145`.
- **No SIMD** — no `-msimd128` in the build flags and zero `v128` entries in the
  type section. The solver is scalar.
- **No `mjr_*`** (the OpenGL renderer is not bound at all). `mjv_*` and
  `MjvScene`/`MjvGeom` are bound, but `app/render.js` drives three.js from
  `renderState()` + `describeGeoms()` instead.
- **No MuJoCo plugins are linked**, so a `<plugin>`-using MJCF will not load.
- `obj_decoder` and `stl_decoder` are linked, so mesh assets work through the VFS.
- Step cost, settled double stance (ncon 12), node v24.13.0 on a 13th-gen
  i5-1340P, one thread: **55.9 – 65.4 µs per physics step**, i.e. 0.22 – 0.26 ms
  per 50 Hz control step (4 substeps) = ~1.2 % of one core at real time.
- Start-up, against the current export (16 decimated OBJs, 1.22 MiB):
  module instantiate 50 ms, `MjVFS.addBuffer` x17 ~20 ms, `mj_loadXML` ~350 ms,
  `new MjData` <1 ms — **~0.4–0.6 s to a playable sim**, and 0.6 s over
  loopback HTTP including the 18 fetches. `mj_loadXML` on a mesh-free variant of
  the same scene is 10 ms, so essentially all of the compile time is OBJ parsing
  plus the convex hull of the 32 visual meshes. The earlier 27.1 MiB full-detail
  mesh set cost **8.4 s** in `mj_loadXML` — if mesh detail is ever raised again,
  that is the number that will bite, not the download.

## API incantations — copy these

### Load a model from an in-memory XML string plus assets

```js
import loadMujoco from '../vendor/mujoco/mujoco.js';   // ESM default export

const mj  = await loadMujoco();            // ~200 ms; do this ONCE per page
const vfs = new mj.MjVFS();
vfs.addBuffer('scene.xml', new TextEncoder().encode(xmlText));   // (name, Uint8Array)
for (const m of meshes) vfs.addBuffer(m.name, m.bytes);          // meshdir-RELATIVE name
const model = mj.MjModel.mj_loadXML('scene.xml', vfs);           // throws a C++ exception
const data  = new mj.MjData(model);
// ... mj.mj_step(model, data); mj.mj_forward(model, data); ...
data.delete(); model.delete(); vfs.delete();
```

The VFS key must be the string MuJoCo will look up: with
`<compiler meshdir="assets">` and `<mesh file="base_0.obj">` that is
`"assets/base_0.obj"`. `app/physics.js` derives the whole list by scanning the
MJCF for `<mesh|texture|hfield|skin|model|include file="...">`.

The glue resolves the wasm with `new URL('mujoco.wasm', import.meta.url)`
(`mujoco.js:648-655`), so the same import works in a browser (HTTP fetch) and in
node (fs read). Override with `loadMujoco({ locateFile: p => myUrl })`.

### Five traps that cost real debugging time here

1. **`data.contact` is a COPY, not a view.** Every property access allocates a
   fresh `std::vector<mjContact>` on the wasm heap, and the copy must be
   `.delete()`d. Measured: 5,000 un-deleted accesses at `ncon = 72` grew the
   heap by 186 MiB; 40,000 hit the 2 GiB `ALLOW_MEMORY_GROWTH` ceiling. Read it
   **once** per scan, wrap the loop in `try { … } finally { vec.delete(); }`,
   and `.delete()` each `vec.get(i)` handle too.

2. **`mj_id2name` returns the junk string `"emsc"` for unnamed objects**, not
   `null` and not `""`. The sym scene's 66 visual mesh geoms are unnamed, so a
   renderer that labels geoms with `mj_id2name` gets 66 geoms called "emsc".
   Read `model.names` (an `Int8Array`) at `model.name_<obj>adr[id]` up to the
   NUL instead; `app/physics.js` exposes the resolved names on
   `describeGeoms()`, `describeBodies()` and `sim.seats.*.jointNames`.

3. **`mjtByte` model arrays throw on access.** `model.mat_texuniform`,
   `model.jnt_limited`, `model.actuator_ctrllimited`, `model.eq_active0` all
   raise `BindingError: _emval_take_value has unknown type
   N10emscripten11memory_viewIbEE`, because `memory_view<bool>` is not
   registered. `int`/`double`/`mjtNum` arrays are fine, and some byte arrays
   (`geom_sameframe`, `body_simple`, `tex_type`) happen to work. Guard anything
   you have not tried.

4. **Never hand a plain JS `Error` to `getExceptionMessage`.** The build uses
   `-fexceptions -s DISABLE_EXCEPTION_CATCHING=0`, so a C++ throw arrives as a
   raw pointer or a `WebAssembly.Exception`; calling `getExceptionMessage` on
   anything else reads out of bounds and aborts the module, turning "bad MJCF"
   into `RuntimeError: memory access out of bounds`. See `describeThrow()`.

5. **`mj_step` runs forward kinematics BEFORE integrating.** Right after
   `mj_step`, `xpos`/`xquat`/`cvel`/`contact` lag `qpos`/`qvel` by one substep.
   This is not a bug to paper over: mjlab depends on it, its referee reads the
   stale frame and its observations read the frame after the single
   `mj_forward` (`mjlab/envs/manager_based_rl_env.py:339-345`, `:395-418`).
   `app/physics.js` reproduces it exactly and never auto-forwards after `step()`.

### Free-joint velocity convention (measured, not assumed)

`qvel[jnt_dofadr + 0..2]` is the **world-frame** linear velocity of the body
frame origin; `qvel[jnt_dofadr + 3..5]` is the **body-frame** angular velocity.
Proof, `tests/node_physics_smoke.mjs` §6: at yaw = 90°, writing angular
`(2,0,0)` reads back `cvel[0:3] = (0,2,0)` in world and `(2,0,0)` in body;
writing linear `(1,0,0)` reads back world `(1,0,0)`, not body `(1,0,0)`.

### Contact normal convention (measured)

`contact.frame` is 9 numbers, rows `(normal, tangent1, tangent2)`, and row 0
points **from `geom1` towards `geom2`**. Verified with a shallow head-on trunk
touch: seat a at `x = 0`, seat b at `x = +0.370` gives
`geom1 = a_base1_collision`, `geom2 = b_base1_collision`, `normal = (1, 0, 0)`.
`sim.contactsBetweenRobots()` re-orients every normal to point from the seat-a
geom towards the seat-b geom regardless of MuJoCo's own ordering.

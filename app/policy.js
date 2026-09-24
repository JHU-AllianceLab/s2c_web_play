// app/policy.js — the policy runtime: load <name>.json + <name>.bin, run the
// MLP by hand in plain JS typed arrays. No onnxruntime, no wasm, no bundler.
//
// =============================================================================
// WHAT THIS REPRODUCES, AND WHERE IT COMES FROM
// =============================================================================
//
// Every shipped policy is an rsl_rl `MLPModel` actor evaluated at its
// DETERMINISTIC mean. The whole forward pass is four lines:
//
//     z = (obs - norm.mean) / (norm.std + norm.eps)      // EmpiricalNormalization
//     h = ELU(L0 z);  h = ELU(L1 h);  h = ELU(L2 h)
//     a = L3 h                                           // BARE linear head
//
//   * normalizer  `(x - _mean) / (_std + eps)`   rsl_rl/modules/normalization.py:46-48
//   * eps = 1e-2                                 rsl_rl/modules/normalization.py:18
//                                                (constructor default; MLPModel never
//                                                 overrides it)
//   * hidden dims (512, 256, 128), activation "elu"
//                                                src/tasks/game/rl/cfg.py:22-32
//                                                rsl_rl/modules/mlp.py:52-63
//   * ELU alpha = 1.0                            resolve_nn_activation("elu") is
//                                                torch.nn.ELU(), rsl_rl/utils/utils.py:48
//   * head has NO activation                     rsl_rl/modules/mlp.py (trailing
//                                                activation popped); recon/04:32
//   * deterministic output = the mean, identity  rsl_rl/modules/distribution.py:182-184
//     (no tanh, no clip). `distribution.std_param` is exploration noise and is
//     not exported.
//
// The 47-D walker (`Unitree-Go2-Flat-Fast`) has exactly the same shape with a
// 47-wide input (recon/04:24-32).
//
// NOTHING is hardcoded here. Widths, activation, normalizer statistics, the
// float32 blob layout and the eps all come out of `<name>.json`, which
// tools/export_policy.py writes next to the weights. `assets/policies/manifest.json`
// is the only place a policy NAME is resolved to a file.
//
// =============================================================================
// NUMERICS — why this lands under the 1e-5 parity gate
// =============================================================================
//
// The reference in tests/parity.json is torch's FLOAT32 output. torch runs a
// blocked float32 GEMM; we accumulate each dot product in JS float64 and round
// only when the result is written into the next layer's Float32Array. So every
// value that torch stores in float32 we also store in float32 (input, z, h0,
// h1, h2, action) and only the intra-layer summation order differs. That is the
// same situation as the exporter's own numpy-float32 replay, which measured a
// worst case of 5.603e-06 over all 16 policies (gate 1e-5).
//
// =============================================================================
// FILE FORMAT (written by tools/export_policy.py, described in every json)
// =============================================================================
//
//   json.layers          [{in, out}, ...] in forward order
//   json.activation      "elu" | "tanh" | "relu" | "sin" | "none"   (between layers)
//   json.activation_alpha  ELU alpha (1.0)
//   json.head_activation null | one of the above  (applied AFTER the last layer)
//   json.norm            {mean: [obsDim], std: [obsDim], eps} | null | absent
//   json.bin             the sibling .bin filename
//   json.bin_layout      "little-endian float32; for each Linear in order:
//                         weight (row-major, out x in) then bias (out)"
//
// A policy with `rotation_baked: true` has the pi-rotation already folded into
// `norm.mean[42:46]` and `L0.weight[:, 42:46]`. THE CALLER MUST NOT ROTATE THE
// OBSERVATION AGAIN — feed a plain seat-B 60-D obs. loadPolicy() surfaces this
// as `.rotationBaked` so app/match.js can assert it once.

/** rsl_rl EmpiricalNormalization eps, used only when a json omits `norm.eps`.
 *  rsl_rl/modules/normalization.py:18 */
export const NORM_EPS_DEFAULT = 0.01;

const IS_NODE =
  typeof process !== 'undefined' && process.versions != null && process.versions.node != null;

const isNetUrl = (u) => /^(https?:|data:|blob:)/.test(u);

let nodeFs = null;
async function getNodeFs() {
  if (!nodeFs) nodeFs = await import('node:fs/promises');
  return nodeFs;
}

/** Read bytes in either environment (same rule as app/physics.js:143-152). */
async function readBytes(url) {
  if (!IS_NODE || isNetUrl(url)) {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`fetch ${url} -> HTTP ${res.status}`);
    return new Uint8Array(await res.arrayBuffer());
  }
  const fs = await getNodeFs();
  const path = url.startsWith('file:') ? new URL(url).pathname : url;
  return new Uint8Array(await fs.readFile(path));
}

async function readText(url) {
  return new TextDecoder().decode(await readBytes(url));
}

function dirOf(url) {
  const i = url.lastIndexOf('/');
  return i < 0 ? '' : url.slice(0, i + 1);
}

function joinUrl(base, rel) {
  if (!base) return rel;
  if (isNetUrl(rel) || rel.startsWith('/')) return rel;
  return base.endsWith('/') ? base + rel : `${base}/${rel}`;
}

// ---------------------------------------------------------------------------
// activations
// ---------------------------------------------------------------------------
//
// Applied in place on a Float32Array, so the value the next layer reads is the
// float32 torch would have stored.

/** torch.nn.ELU: x if x > 0 else alpha*(exp(x) - 1). alpha is 1.0 for every
 *  shipped policy (rsl_rl/utils/utils.py:48). */
function eluInPlace(v, alpha) {
  for (let i = 0; i < v.length; i++) {
    const x = v[i];
    if (x <= 0) v[i] = alpha * (Math.exp(x) - 1);
  }
}

function tanhInPlace(v) {
  for (let i = 0; i < v.length; i++) v[i] = Math.tanh(v[i]);
}

function reluInPlace(v) {
  for (let i = 0; i < v.length; i++) if (v[i] < 0) v[i] = 0;
}

function sinInPlace(v) {
  for (let i = 0; i < v.length; i++) v[i] = Math.sin(v[i]);
}

/** name -> in-place kernel. `null`/"none"/"identity"/"linear" -> no-op. */
export function resolveActivation(name, alpha = 1.0) {
  const key = name == null ? 'none' : String(name).toLowerCase();
  switch (key) {
    case 'none':
    case 'null':
    case 'identity':
    case 'linear':
      return null;
    case 'elu':
      return (v) => eluInPlace(v, alpha);
    case 'tanh':
      return tanhInPlace;
    case 'relu':
      return reluInPlace;
    case 'sin':
      return sinInPlace;
    default:
      throw new Error(`policy.js: unsupported activation "${name}"`);
  }
}

// ---------------------------------------------------------------------------
// the net
// ---------------------------------------------------------------------------

/**
 * A loaded, ready-to-run MLP. Construct via `loadPolicy()` (fetches) or
 * `buildPolicy()` (bytes already in hand — used by the tests).
 */
class MlpPolicy {
  /**
   * @param {object} meta   the parsed <name>.json
   * @param {ArrayBuffer|Uint8Array} binBytes  the raw <name>.bin
   */
  constructor(meta, binBytes) {
    const layers = meta.layers;
    if (!Array.isArray(layers) || layers.length === 0) {
      throw new Error(`policy.js: ${meta.name ?? '<policy>'} has no layers[]`);
    }

    this.meta = meta;
    this.name = meta.name ?? null;
    this.obsDim = layers[0].in;
    this.actDim = layers[layers.length - 1].out;
    this.rotationBaked = meta.rotation_baked === true;
    this.actionPath = meta.action_path ?? null;
    this.game = meta.game ?? null;
    this.seat = meta.seat ?? null;

    if (meta.obs_dim != null && meta.obs_dim !== this.obsDim) {
      throw new Error(
        `policy.js: ${this.name} declares obs_dim ${meta.obs_dim} but layer 0 takes ${this.obsDim}`,
      );
    }
    if (meta.act_dim != null && meta.act_dim !== this.actDim) {
      throw new Error(
        `policy.js: ${this.name} declares act_dim ${meta.act_dim} but the head emits ${this.actDim}`,
      );
    }

    // --- weights ----------------------------------------------------------
    // Little-endian float32. Per Linear, in forward order: weight (row-major,
    // out x in) then bias (out). Every browser this ships to is little-endian
    // (wasm mandates it), and node on x86/arm is too; a Float32Array view over
    // the buffer is therefore the exact bytes.
    const u8 = binBytes instanceof Uint8Array ? binBytes : new Uint8Array(binBytes);
    let need = 0;
    for (const L of layers) need += L.out * L.in + L.out;
    if (u8.byteLength !== need * 4) {
      throw new Error(
        `policy.js: ${this.name} .bin is ${u8.byteLength} B, the layer list needs ${need * 4} B`,
      );
    }
    // Copy into an aligned buffer: a Uint8Array from fs/fetch may sit at a
    // non-multiple-of-4 byteOffset, which a Float32Array view cannot address.
    const aligned = new Float32Array(need);
    new Uint8Array(aligned.buffer).set(u8);

    this.layers = [];
    let off = 0;
    for (const L of layers) {
      const w = aligned.subarray(off, off + L.out * L.in);
      off += L.out * L.in;
      const b = aligned.subarray(off, off + L.out);
      off += L.out;
      this.layers.push({ in: L.in, out: L.out, w, b, y: new Float32Array(L.out) });
    }

    // --- normalizer -------------------------------------------------------
    const n = meta.norm;
    if (n && Array.isArray(n.mean) && Array.isArray(n.std)) {
      if (n.mean.length !== this.obsDim || n.std.length !== this.obsDim) {
        throw new Error(
          `policy.js: ${this.name} norm has ${n.mean.length}/${n.std.length} entries, obsDim is ${this.obsDim}`,
        );
      }
      this.norm = {
        // Float32Array so the stored constants are bit-identical to the
        // checkpoint tensors the exporter read.
        mean: Float32Array.from(n.mean),
        std: Float32Array.from(n.std),
        eps: n.eps == null ? NORM_EPS_DEFAULT : n.eps,
      };
      // (std + eps) is computed once here exactly as torch computes it, in
      // float32, then stored; the per-step work is one divide.
      this.norm.denom = new Float32Array(this.obsDim);
      for (let i = 0; i < this.obsDim; i++) {
        this.norm.denom[i] = this.norm.std[i] + this.norm.eps;
      }
    } else {
      this.norm = null; // normalizer folded into the graph, or none at all
    }

    this._z = new Float32Array(this.obsDim);
    this._act = resolveActivation(meta.activation ?? 'elu', meta.activation_alpha ?? 1.0);
    this._headAct = resolveActivation(meta.head_activation ?? null, meta.activation_alpha ?? 1.0);
  }

  /**
   * Deterministic action for one observation.
   * @param {ArrayLike<number>} obs  length obsDim
   * @param {Float32Array} [out]     optional destination (length actDim)
   * @returns {Float32Array}         length actDim
   */
  forward(obs, out) {
    if (obs.length !== this.obsDim) {
      throw new Error(
        `policy.js: ${this.name} expects a ${this.obsDim}-D obs, got ${obs.length}`,
      );
    }

    // ---- normalize -------------------------------------------------------
    let x;
    if (this.norm) {
      const { mean, denom } = this.norm;
      const z = this._z;
      for (let i = 0; i < this.obsDim; i++) z[i] = (obs[i] - mean[i]) / denom[i];
      x = z;
    } else if (obs instanceof Float32Array) {
      x = obs;
    } else {
      const z = this._z;
      for (let i = 0; i < this.obsDim; i++) z[i] = obs[i];
      x = z;
    }

    // ---- layers ----------------------------------------------------------
    const L = this.layers;
    const last = L.length - 1;
    for (let li = 0; li <= last; li++) {
      const { in: nIn, out: nOut, w, b, y } = L[li];
      const n4 = nIn - (nIn & 3);
      for (let r = 0, base = 0; r < nOut; r++, base += nIn) {
        // Four float64 accumulators (~15 % faster than one, and the partial
        // sums are a shade more accurate than a single serial chain). The store
        // into `y` (a Float32Array) rounds to the float32 torch would keep.
        let s0 = 0, s1 = 0, s2 = 0, s3 = 0, c = 0;
        for (; c < n4; c += 4) {
          s0 += w[base + c] * x[c];
          s1 += w[base + c + 1] * x[c + 1];
          s2 += w[base + c + 2] * x[c + 2];
          s3 += w[base + c + 3] * x[c + 3];
        }
        let s = s0 + s1 + s2 + s3;
        for (; c < nIn; c++) s += w[base + c] * x[c];
        y[r] = s + b[r];
      }
      if (li < last) {
        if (this._act) this._act(y);
      } else if (this._headAct) {
        this._headAct(y);
      }
      x = y;
    }

    if (out) {
      if (out.length !== this.actDim) {
        throw new Error(`policy.js: out must have ${this.actDim} entries`);
      }
      out.set(x);
      return out;
    }
    return Float32Array.from(x);
  }
}

/**
 * Build a policy from bytes already in hand (no I/O). Same object `loadPolicy`
 * returns.
 * @param {object} meta                        parsed <name>.json
 * @param {ArrayBuffer|Uint8Array} binBytes    the matching .bin
 */
export function buildPolicy(meta, binBytes) {
  return new MlpPolicy(meta, binBytes);
}

/**
 * Load `<name>.json` + its `.bin` and return a runnable policy.
 *
 * @param {string} jsonUrl  path/URL of the json (the .bin is resolved relative
 *                          to it from `json.bin`, falling back to the same
 *                          basename with a .bin extension).
 * @param {object} [opts]
 * @param {string} [opts.binUrl]  override the .bin location
 * @param {object} [opts.meta]    pre-parsed json (skips the fetch)
 * @returns {Promise<{forward(obs:ArrayLike<number>, out?:Float32Array):Float32Array,
 *                    obsDim:number, actDim:number, name:string|null,
 *                    rotationBaked:boolean, actionPath:string|null, norm:object|null,
 *                    meta:object}>}
 */
export async function loadPolicy(jsonUrl, opts = {}) {
  const meta = opts.meta ?? JSON.parse(await readText(jsonUrl));
  const binName = meta.bin ?? `${jsonUrl.replace(/\.json$/, '')}.bin`.split('/').pop();
  const binUrl = opts.binUrl ?? joinUrl(dirOf(jsonUrl), binName);
  const bytes = await readBytes(binUrl);
  const p = new MlpPolicy(meta, bytes);
  p.jsonUrl = jsonUrl;
  p.binUrl = binUrl;
  return p;
}

/**
 * Load `assets/policies/manifest.json` and hand back a small resolver so no
 * caller ever spells a policy filename (DESIGN.md section 3.5: "code must not
 * hardcode checkpoint paths -- everything goes through manifest.json").
 *
 * @param {string} [manifestUrl]
 */
export async function loadManifest(manifestUrl = 'assets/policies/manifest.json') {
  const manifest = JSON.parse(await readText(manifestUrl));
  const dir = dirOf(manifestUrl);
  const byName = new Map();
  for (const row of manifest.policies ?? []) byName.set(row.name, row);

  return {
    manifest,
    dir,
    /** Every policy row, in manifest order. */
    list: () => manifest.policies ?? [],
    /** The row for one policy name. */
    entry(name) {
      const row = byName.get(name);
      if (!row) throw new Error(`policy.js: "${name}" is not in ${manifestUrl}`);
      return row;
    },
    /** The opponent names offered for one game/seat, e.g. ('sym', 'B'). */
    opponents(game, seat) {
      const g = manifest.games?.[game];
      if (!g) throw new Error(`policy.js: no game "${game}" in the manifest`);
      const names = g.seats?.[seat];
      if (!names) throw new Error(`policy.js: no seat "${seat}" for game "${game}"`);
      return names.slice();
    },
    /** The human's walk policy row. */
    playerWalk: () => manifest.player_walk,
    /** Load one policy by manifest name. */
    load(name) {
      const row = this.entry(name);
      return loadPolicy(joinUrl(dir, row.json));
    },
  };
}

export default { loadPolicy, loadManifest, buildPolicy, resolveActivation, NORM_EPS_DEFAULT };

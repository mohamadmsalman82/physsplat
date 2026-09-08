/**
 * GpuNet: the PhysSplat graph network on raw WebGPU.
 *
 * Why not ONNX Runtime Web: its WebGPU provider dispatches ~200 ops with
 * ~1 ms overhead each (measured 190-490 ms/step) and cannot graph-capture
 * this model. The network is only dense layers, gathers, layer-norms and
 * segment sums, so a few WGSL kernels recorded into ONE command buffer per
 * step run it in tens of milliseconds.
 *
 * Numerics mirror src/physsplat/model/gnn.py + export/to_onnx.py exactly:
 *   mlp(din,dout): Linear -> ReLU -> Linear -> ReLU -> Linear [-> LayerNorm]
 *   block: e += edge_mlp([e, h[s], h[r]]); agg = segsum(e); h += node_mlp([h, agg])
 *   body head: mean-pool h by body, concat scalars, body_dec (no LN)
 * Segment sums use the sorted-edge contract (deterministic, race-free).
 * Verified against a Python-captured forward in web/public/test_gpu.html.
 */

const WG = 64;          // threads per workgroup for row-parallel kernels
// GEMM: 16x16 threads per workgroup, each owning a 4x4 output block, so one
// workgroup covers a 64x64 tile of Y with a K-slice of 16 staged in shared
// memory. Register tiling raised the dense layers from ~200 GFLOP/s to a
// usable fraction of the GPU; the 16x16 one-output-per-thread version cost
// ~50 ms per step on a 5-pencil scene.
const BT = 16, RT = 4, BM = BT * RT, KT = 16;

const SHADERS = {
  // Y[r, c] = sum_k X[r,k] * W[c,k] + b[c]; optional ReLU. W is torch [out,in].
  gemm: `
struct Dims { rows: u32, kdim: u32, cols: u32, relu: u32 };
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> W: array<f32>;
@group(0) @binding(2) var<storage, read> Bv: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<uniform> d: Dims;
var<workgroup> xs: array<f32, ${BM * KT}>;   // [row][k]
var<workgroup> ws: array<f32, ${KT * BM}>;   // [k][col]
@compute @workgroup_size(${BT}, ${BT})
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
  let row0 = wid.y * ${BM}u + lid.y * ${RT}u;
  let col0 = wid.x * ${BM}u + lid.x * ${RT}u;
  let tid = lid.y * ${BT}u + lid.x;
  var acc: array<array<f32, ${RT}>, ${RT}>;
  for (var i = 0u; i < ${RT}u; i++) { for (var j = 0u; j < ${RT}u; j++) { acc[i][j] = 0.0; } }
  let ntiles = (d.kdim + ${KT}u - 1u) / ${KT}u;
  for (var t = 0u; t < ntiles; t++) {
    let k0 = t * ${KT}u;
    // stage X[64 rows x 16 k] and W^T[16 k x 64 cols]: 1024 elements each,
    // 4 per thread
    for (var q = 0u; q < ${RT}u; q++) {
      let idx = tid + q * ${BT * BT}u;
      let xr = idx / ${KT}u; let xk = idx % ${KT}u;
      let gr = wid.y * ${BM}u + xr; let gk = k0 + xk;
      xs[idx] = select(0.0, X[gr * d.kdim + gk], gr < d.rows && gk < d.kdim);
      let wk = idx / ${BM}u; let wc = idx % ${BM}u;
      let gc = wid.x * ${BM}u + wc; let gk2 = k0 + wk;
      ws[idx] = select(0.0, W[gc * d.kdim + gk2], gc < d.cols && gk2 < d.kdim);
    }
    workgroupBarrier();
    for (var k = 0u; k < ${KT}u; k++) {
      var a: array<f32, ${RT}>;
      var b: array<f32, ${RT}>;
      for (var i = 0u; i < ${RT}u; i++) { a[i] = xs[(lid.y * ${RT}u + i) * ${KT}u + k]; }
      for (var j = 0u; j < ${RT}u; j++) { b[j] = ws[k * ${BM}u + lid.x * ${RT}u + j]; }
      for (var i = 0u; i < ${RT}u; i++) {
        for (var j = 0u; j < ${RT}u; j++) { acc[i][j] += a[i] * b[j]; }
      }
    }
    workgroupBarrier();
  }
  for (var i = 0u; i < ${RT}u; i++) {
    let r = row0 + i;
    if (r >= d.rows) { continue; }
    for (var j = 0u; j < ${RT}u; j++) {
      let c = col0 + j;
      if (c >= d.cols) { continue; }
      var v = acc[i][j] + Bv[c];
      if (d.relu == 1u) { v = max(v, 0.0); }
      Y[r * d.cols + c] = v;
    }
  }
}`,
  // out[r] = base[r] + LayerNorm(Y[r]) (residual add), or just LN when add=0
  layernorm: `
struct Dims { rows: u32, cols: u32, add: u32, pad: u32 };
@group(0) @binding(0) var<storage, read> Y: array<f32>;
@group(0) @binding(1) var<storage, read> G: array<f32>;
@group(0) @binding(2) var<storage, read> Bt: array<f32>;
@group(0) @binding(3) var<storage, read_write> O: array<f32>;
@group(0) @binding(4) var<uniform> d: Dims;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let r = gid.x; if (r >= d.rows) { return; }
  let n = f32(d.cols);
  var mean = 0.0;
  for (var c = 0u; c < d.cols; c++) { mean += Y[r * d.cols + c]; }
  mean = mean / n;
  var v = 0.0;
  for (var c = 0u; c < d.cols; c++) { let x = Y[r * d.cols + c] - mean; v += x * x; }
  let inv = inverseSqrt(v / n + 1e-5);
  for (var c = 0u; c < d.cols; c++) {
    let ln = (Y[r * d.cols + c] - mean) * inv * G[c] + Bt[c];
    let idx = r * d.cols + c;
    O[idx] = select(ln, O[idx] + ln, d.add == 1u);
  }
}`,
  // X_e = [e, h[s], h[r]] (E x 3d)
  gather_edges: `
struct Dims { E: u32, d: u32, pad0: u32, pad1: u32 };
@group(0) @binding(0) var<storage, read> Ein: array<f32>;
@group(0) @binding(1) var<storage, read> H: array<f32>;
@group(0) @binding(2) var<storage, read> S: array<u32>;
@group(0) @binding(3) var<storage, read> R: array<u32>;
@group(0) @binding(4) var<storage, read_write> X: array<f32>;
@group(0) @binding(5) var<uniform> dm: Dims;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let e = gid.x; if (e >= dm.E) { return; }
  let s = S[e]; let r = R[e]; let d = dm.d;
  for (var k = 0u; k < d; k++) {
    X[e * 3u * d + k] = Ein[e * d + k];
    X[e * 3u * d + d + k] = H[s * d + k];
    X[e * 3u * d + 2u * d + k] = H[r * d + k];
  }
}`,
  // X_n = [h, segsum(e over ptr[i]..ptr[i+1])] (N x 2d); rows sorted by
  // receiver. Dummy nodes (i >= nReal) get zero aggregates: the padded
  // self-loop edges all land on the first dummy, and a single thread
  // walking ~1000 x 128 dependent reads was the entire 200 ms bottleneck.
  segsum_concat: `
struct Dims { N: u32, d: u32, nReal: u32, pad1: u32 };
@group(0) @binding(0) var<storage, read> H: array<f32>;
@group(0) @binding(1) var<storage, read> Ev: array<f32>;
@group(0) @binding(2) var<storage, read> P: array<u32>;
@group(0) @binding(3) var<storage, read_write> X: array<f32>;
@group(0) @binding(4) var<uniform> dm: Dims;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x; if (i >= dm.N) { return; }
  let d = dm.d;
  let real = i < dm.nReal;
  let a = select(0u, P[i], real); let b = select(0u, P[i + 1u], real);
  for (var k = 0u; k < d; k++) {
    var acc = 0.0;
    for (var e = a; e < b; e++) { acc += Ev[e * d + k]; }
    X[i * 2u * d + k] = H[i * d + k];
    X[i * 2u * d + d + k] = acc;
  }
}`,
  // pooled[b] = mean of h rows in [ptr[b], ptr[b+1]) concat scalars[b]
  // (B x (d+4)); one thread per (body, channel) so ~200-row sums run in
  // parallel instead of one thread per body walking 200 x 128 reads (3 ms)
  pool: `
struct Dims { B: u32, d: u32, pad0: u32, pad1: u32 };
@group(0) @binding(0) var<storage, read> H: array<f32>;
@group(0) @binding(1) var<storage, read> P: array<u32>;
@group(0) @binding(2) var<storage, read> Sc: array<f32>;
@group(0) @binding(3) var<storage, read_write> X: array<f32>;
@group(0) @binding(4) var<uniform> dm: Dims;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let d = dm.d; let w = d + 4u;
  let b = gid.x / w; let k = gid.x % w;
  if (b >= dm.B) { return; }
  if (k >= d) { X[b * w + k] = Sc[b * 4u + (k - d)]; return; }
  let a = P[b]; let z = P[b + 1u];
  var acc = 0.0;
  for (var i = a; i < z; i++) { acc += H[i * d + k]; }
  X[b * w + k] = acc / max(f32(z - a), 1.0);
}`,
};

export class GpuNet {
  static uid = 0;   // buffer identity for the bind-group cache

  static async create(manifestUrl, binUrl) {
    if (!navigator.gpu) throw new Error("WebGPU unavailable");
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error("no WebGPU adapter");
    // per-kernel GPU timing (?prof=1) needs the timestamp-query feature;
    // absent it the profile simply stays empty
    const wantProf = new URLSearchParams(location.search).get("prof") === "1";
    const canProf = wantProf && adapter.features.has("timestamp-query");
    const device = await adapter.requestDevice(
      canProf ? { requiredFeatures: ["timestamp-query"] } : {});
    const manifest = await (await fetch(manifestUrl)).json();
    const bin = new Float32Array(await (await fetch(binUrl)).arrayBuffer());
    const net = new GpuNet(device, manifest, bin);
    net.profile = canProf;
    return net;
  }

  constructor(device, manifest, bin) {
    this.dev = device;
    this.d = manifest.latent;
    this.L = manifest.layers;
    this.w = {};
    for (const [name, m] of Object.entries(manifest.tensors)) {
      const n = m.shape.reduce((a, b) => a * b, 1);
      this.w[name] = this.#buf(bin.subarray(m.offset, m.offset + n), GPUBufferUsage.STORAGE);
      this.w[name].shape = m.shape;
    }
    this.pipes = {};
    for (const [k, code] of Object.entries(SHADERS)) {
      this.pipes[k] = device.createComputePipeline({
        layout: "auto",
        compute: { module: device.createShaderModule({ code }), entryPoint: "main" },
      });
    }
    this.scratch = new Map();
  }

  #buf(data, usage, extra = 0) {
    const b = this.dev.createBuffer({
      size: Math.max(16, data.byteLength + extra), usage: usage | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true });
    new (data.constructor)(b.getMappedRange()).set(data);
    b.unmap();
    b.uid = ++GpuNet.uid;
    return b;
  }

  #scratchBuf(key, bytes) {
    const need = Math.max(16, bytes);
    let b = this.scratch.get(key);
    if (!b || b.size < need) {
      b?.destroy();
      b = this.dev.createBuffer({ size: need, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
      b.uid = ++GpuNet.uid;
      this.scratch.set(key, b);
    }
    return b;
  }

  /** Persistent, grow-on-demand input buffer written with queue.writeBuffer:
   * creating and destroying seven mapped buffers per step was measurable. */
  #inputBuf(key, data) {
    const b = this.#scratchBuf(`in_${key}`, data.byteLength);
    this.dev.queue.writeBuffer(b, 0, data.buffer, data.byteOffset, data.byteLength);
    return b;
  }

  #bindGroup(pipe, name, buffers) {
    this.bindGroups ??= new Map();
    const key = name + ":" + buffers.map((b) => b.uid).join(",");
    let bg = this.bindGroups.get(key);
    if (!bg) {
      if (this.bindGroups.size > 2048) this.bindGroups.clear();
      bg = this.dev.createBindGroup({
        layout: pipe.getBindGroupLayout(0),
        entries: buffers.map((b, i) => ({ binding: i, resource: { buffer: b } })),
      });
      this.bindGroups.set(key, bg);
    }
    return bg;
  }

  #uniform(vals) {
    // uniform contents are fixed per scene shape; cache instead of creating
    // ~110 GPU buffers per step (they leaked and bogged the tab down)
    this.uniforms ??= new Map();
    const key = vals.join(",");
    let b = this.uniforms.get(key);
    if (!b) {
      b = this.#buf(new Uint32Array(vals), GPUBufferUsage.UNIFORM);
      this.uniforms.set(key, b);
    }
    return b;
  }

  /**
   * All dispatches of a step share ONE compute pass (WebGPU orders them and
   * makes storage writes visible to later dispatches); ~120 separate passes
   * cost ~10 ms of encoder/driver overhead. Profiling mode (?prof=1) uses a
   * pass per dispatch because timestamps are written per pass.
   */
  #dispatch(enc, pipe, buffers, wgX, wgY = 1, tag = "other") {
    const bg = this.#bindGroup(pipe, tag.split("_")[0], buffers);
    let pass;
    if (this.prof) {
      const i = this.prof.tags.length;
      this.prof.tags.push(tag);
      pass = enc.beginComputePass({ timestampWrites: { querySet: this.prof.qs,
        beginningOfPassWriteIndex: 2 * i, endOfPassWriteIndex: 2 * i + 1 } });
    } else {
      pass = this.pass ??= enc.beginComputePass();
    }
    pass.setPipeline(pipe);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(wgX, wgY);
    if (this.prof) pass.end();
  }

  /** Resolve the timestamp queries recorded this step into {tag: ms}. */
  async #resolveProfile(enc) {
    const p = this.prof, n = p.tags.length;
    const res = this.dev.createBuffer({ size: n * 16,
      usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
    const rd = this.dev.createBuffer({ size: n * 16,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    enc.resolveQuerySet(p.qs, 0, 2 * n, res, 0);
    enc.copyBufferToBuffer(res, 0, rd, 0, n * 16);
    return async () => {
      await rd.mapAsync(GPUMapMode.READ);
      const ts = new BigInt64Array(rd.getMappedRange().slice(0));
      rd.unmap(); res.destroy(); rd.destroy();
      const out = {};
      for (let i = 0; i < n; i++)
        out[p.tags[i]] = (out[p.tags[i]] ?? 0) + Number(ts[2 * i + 1] - ts[2 * i]) / 1e6;
      out.total_ms = Object.values(out).reduce((a, b) => a + b, 0);
      this.lastProfile = out;
    };
  }

  /** mlp(prefix, X[rows x din]) -> out buffer [rows x dout]. With a
   * LayerNorm present the result goes to `resid`: added into it when
   * `add` (residual block), or overwriting it (encoder). */
  #mlp(enc, prefix, X, rows, din, tmp, resid = null, add = true) {
    const d = this.d;
    const layers = [[`${prefix}.0`, din, d, 1], [`${prefix}.2`, d, d, 1]];
    const w4 = this.w[`${prefix}.4.weight`];
    const dout = w4.shape[0];
    layers.push([`${prefix}.4`, d, dout, 0]);
    let cur = X;
    let out;
    layers.forEach(([nm, kin, kout, relu], i) => {
      out = this.#scratchBuf(`${tmp}_${i}`, rows * kout * 4);
      const u = this.#uniform([rows, kin, kout, relu]);
      this.#dispatch(enc, this.pipes.gemm,
        [cur, this.w[`${nm}.weight`], this.w[`${nm}.bias`], out, u],
        Math.ceil(kout / BM), Math.ceil(rows / BM), `gemm_${tmp}${i}`);
      cur = out;
    });
    const g = this.w[`${prefix}.5.weight`];
    if (g) {   // LayerNorm present
      const target = resid ?? this.#scratchBuf(`${tmp}_ln`, rows * dout * 4);
      const u = this.#uniform([rows, dout, resid && add ? 1 : 0, 0]);
      this.#dispatch(enc, this.pipes.layernorm,
        [cur, g, this.w[`${prefix}.5.bias`], target, u], Math.ceil(rows / WG), 1,
        `ln_${tmp}`);
      return target;
    }
    return cur;
  }

  /**
   * Inputs are typed arrays following the export contract (edges sorted by
   * receiver, nodes by body). Returns Float32Array (B*6) normalized residuals.
   */
  async forward({ nodeFeats, nodeDim, edgeFeats, senders, receivers, segPtr,
                  bodyPtr, bodyScalars, N, E, B, nReal = N }) {
    const d = this.d, dev = this.dev;
    const nf = this.#inputBuf("nf", nodeFeats);
    const ef = this.#inputBuf("ef", edgeFeats);
    const sB = this.#inputBuf("s", senders);
    const rB = this.#inputBuf("r", receivers);
    const pB = this.#inputBuf("p", segPtr);
    const bpB = this.#inputBuf("bp", bodyPtr);
    const scB = this.#inputBuf("sc", bodyScalars);
    const enc = dev.createCommandEncoder();
    this.pass = null;
    if (this.profile) {
      // 2 timestamps per pass; ~120 passes per step
      this.prof = { qs: dev.createQuerySet({ type: "timestamp", count: 512 }), tags: [] };
    }

    // encoders (LayerNorm, no residual) write straight into the residual
    // streams h (N x d), e (E x d)
    const h = this.#scratchBuf("h", N * d * 4);
    const e = this.#scratchBuf("e", E * d * 4);
    this.#mlp(enc, "node_enc", nf, N, nodeDim, "ne", h, false);
    this.#mlp(enc, "edge_enc", ef, E, 5, "ee", e, false);

    const xe = this.#scratchBuf("xe", E * 3 * d * 4);
    const xn = this.#scratchBuf("xn", N * 2 * d * 4);
    const uE = this.#uniform([E, d, 0, 0]);
    const uN = this.#uniform([N, d, nReal, 0]);
    for (let l = 0; l < this.L; l++) {
      this.#dispatch(enc, this.pipes.gather_edges, [e, h, sB, rB, xe, uE],
        Math.ceil(E / WG), 1, "gather");
      this.#mlp(enc, `blocks.${l}.edge_mlp`, xe, E, 3 * d, "em", e);      // e += LN(mlp)
      this.#dispatch(enc, this.pipes.segsum_concat, [h, e, pB, xn, uN],
        Math.ceil(N / WG), 1, "segsum");
      this.#mlp(enc, `blocks.${l}.node_mlp`, xn, N, 2 * d, "nm", h);      // h += LN(mlp)
    }
    const xp = this.#scratchBuf("xp", B * (d + 4) * 4);
    const uB = this.#uniform([B, d, 0, 0]);
    this.#dispatch(enc, this.pipes.pool, [h, bpB, scB, xp, uB],
      Math.ceil(B * (d + 4) / WG), 1, "pool");
    const out = this.#mlp(enc, "body_dec", xp, B, d + 4, "bd");
    if (this.pass) { this.pass.end(); this.pass = null; }
    const read = dev.createBuffer({ size: Math.max(16, B * 6 * 4),
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    enc.copyBufferToBuffer(out, 0, read, 0, B * 6 * 4);
    const finishProfile = this.prof ? await this.#resolveProfile(enc) : null;
    dev.queue.submit([enc.finish()]);
    await read.mapAsync(GPUMapMode.READ);
    const result = new Float32Array(read.getMappedRange().slice(0));
    read.unmap();
    read.destroy();
    if (finishProfile) { await finishProfile(); this.prof.qs.destroy(); this.prof = null; }
    if (this.uniforms && this.uniforms.size > 4096) {   // scene changed a lot
      for (const b of this.uniforms.values()) b.destroy();
      this.uniforms.clear();
      this.bindGroups?.clear();
    }
    return result;
  }
}

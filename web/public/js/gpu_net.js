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
const TILE = 16;        // GEMM tile

const SHADERS = {
  // Y[r, c] = sum_k X[r,k] * W[c,k] + b[c]; optional ReLU. W is torch [out,in].
  gemm: `
struct Dims { rows: u32, kdim: u32, cols: u32, relu: u32 };
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> W: array<f32>;
@group(0) @binding(2) var<storage, read> Bv: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<uniform> d: Dims;
var<workgroup> xs: array<f32, ${TILE * TILE}>;
var<workgroup> ws: array<f32, ${TILE * TILE}>;
@compute @workgroup_size(${TILE}, ${TILE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
  let r = gid.y; let c = gid.x;
  var acc = 0.0;
  let ntiles = (d.kdim + ${TILE}u - 1u) / ${TILE}u;
  for (var t = 0u; t < ntiles; t++) {
    let k0 = t * ${TILE}u;
    let kx = k0 + lid.x; let ky = k0 + lid.y;
    xs[lid.y * ${TILE}u + lid.x] = select(0.0, X[r * d.kdim + kx], r < d.rows && kx < d.kdim);
    ws[lid.y * ${TILE}u + lid.x] = select(0.0, W[c * d.kdim + ky], c < d.cols && ky < d.kdim);
    workgroupBarrier();
    for (var k = 0u; k < ${TILE}u; k++) {
      acc += xs[lid.y * ${TILE}u + k] * ws[k * ${TILE}u + lid.x];
    }
    workgroupBarrier();
  }
  if (r < d.rows && c < d.cols) {
    var v = acc + Bv[c];
    if (d.relu == 1u) { v = max(v, 0.0); }
    Y[r * d.cols + c] = v;
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
  // pooled[b] = mean of h rows in [ptr[b], ptr[b+1]) concat scalars[b] (B x (d+4))
  pool: `
struct Dims { B: u32, d: u32, pad0: u32, pad1: u32 };
@group(0) @binding(0) var<storage, read> H: array<f32>;
@group(0) @binding(1) var<storage, read> P: array<u32>;
@group(0) @binding(2) var<storage, read> Sc: array<f32>;
@group(0) @binding(3) var<storage, read_write> X: array<f32>;
@group(0) @binding(4) var<uniform> dm: Dims;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let b = gid.x; if (b >= dm.B) { return; }
  let d = dm.d; let a = P[b]; let z = P[b + 1u];
  let cnt = max(f32(z - a), 1.0);
  for (var k = 0u; k < d; k++) {
    var acc = 0.0;
    for (var i = a; i < z; i++) { acc += H[i * d + k]; }
    X[b * (d + 4u) + k] = acc / cnt;
  }
  for (var k = 0u; k < 4u; k++) { X[b * (d + 4u) + d + k] = Sc[b * 4u + k]; }
}`,
};

export class GpuNet {
  static async create(manifestUrl, binUrl) {
    if (!navigator.gpu) throw new Error("WebGPU unavailable");
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error("no WebGPU adapter");
    const device = await adapter.requestDevice();
    const manifest = await (await fetch(manifestUrl)).json();
    const bin = new Float32Array(await (await fetch(binUrl)).arrayBuffer());
    return new GpuNet(device, manifest, bin);
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
    return b;
  }

  #scratchBuf(key, bytes) {
    const need = Math.max(16, bytes);
    let b = this.scratch.get(key);
    if (!b || b.size < need) {
      b?.destroy();
      b = this.dev.createBuffer({ size: need, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
      this.scratch.set(key, b);
    }
    return b;
  }

  #uniform(vals) {
    return this.#buf(new Uint32Array(vals), GPUBufferUsage.UNIFORM);
  }

  #dispatch(enc, pipe, buffers, wgX, wgY = 1) {
    const bg = this.dev.createBindGroup({
      layout: pipe.getBindGroupLayout(0),
      entries: buffers.map((b, i) => ({ binding: i, resource: { buffer: b } })),
    });
    const pass = enc.beginComputePass();
    pass.setPipeline(pipe);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(wgX, wgY);
    pass.end();
  }

  /** mlp(prefix, X[rows x din]) -> out buffer [rows x dout]; ln=true applies LayerNorm(+residual into resid) */
  #mlp(enc, prefix, X, rows, din, tmp, resid = null) {
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
        Math.ceil(kout / TILE), Math.ceil(rows / TILE));
      cur = out;
    });
    const g = this.w[`${prefix}.5.weight`];
    if (g) {   // LayerNorm present
      const target = resid ?? this.#scratchBuf(`${tmp}_ln`, rows * dout * 4);
      const u = this.#uniform([rows, dout, resid ? 1 : 0, 0]);
      this.#dispatch(enc, this.pipes.layernorm,
        [cur, g, this.w[`${prefix}.5.bias`], target, u], Math.ceil(rows / WG));
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
    const nf = this.#buf(nodeFeats, GPUBufferUsage.STORAGE);
    const ef = this.#buf(edgeFeats, GPUBufferUsage.STORAGE);
    const sB = this.#buf(senders, GPUBufferUsage.STORAGE);
    const rB = this.#buf(receivers, GPUBufferUsage.STORAGE);
    const pB = this.#buf(segPtr, GPUBufferUsage.STORAGE);
    const bpB = this.#buf(bodyPtr, GPUBufferUsage.STORAGE);
    const scB = this.#buf(bodyScalars, GPUBufferUsage.STORAGE);
    const enc = dev.createCommandEncoder();

    // encoders (LayerNorm, no residual): h (N x d), e (E x d)
    const h = this.#scratchBuf("h", N * d * 4);
    const e = this.#scratchBuf("e", E * d * 4);
    const h0 = this.#mlp(enc, "node_enc", nf, N, nodeDim, "ne");
    const e0 = this.#mlp(enc, "edge_enc", ef, E, 5, "ee");
    enc.copyBufferToBuffer(h0, 0, h, 0, N * d * 4);
    enc.copyBufferToBuffer(e0, 0, e, 0, E * d * 4);

    const xe = this.#scratchBuf("xe", E * 3 * d * 4);
    const xn = this.#scratchBuf("xn", N * 2 * d * 4);
    const uE = this.#uniform([E, d, 0, 0]);
    const uN = this.#uniform([N, d, nReal, 0]);
    for (let l = 0; l < this.L; l++) {
      this.#dispatch(enc, this.pipes.gather_edges, [e, h, sB, rB, xe, uE], Math.ceil(E / WG));
      this.#mlp(enc, `blocks.${l}.edge_mlp`, xe, E, 3 * d, "em", e);      // e += LN(mlp)
      this.#dispatch(enc, this.pipes.segsum_concat, [h, e, pB, xn, uN], Math.ceil(N / WG));
      this.#mlp(enc, `blocks.${l}.node_mlp`, xn, N, 2 * d, "nm", h);      // h += LN(mlp)
    }
    const xp = this.#scratchBuf("xp", B * (d + 4) * 4);
    const uB = this.#uniform([B, d, 0, 0]);
    this.#dispatch(enc, this.pipes.pool, [h, bpB, scB, xp, uB], Math.ceil(B / WG));
    const out = this.#mlp(enc, "body_dec", xp, B, d + 4, "bd");
    const read = dev.createBuffer({ size: Math.max(16, B * 6 * 4),
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    enc.copyBufferToBuffer(out, 0, read, 0, B * 6 * 4);
    dev.queue.submit([enc.finish()]);
    await read.mapAsync(GPUMapMode.READ);
    const result = new Float32Array(read.getMappedRange().slice(0));
    read.unmap();
    for (const b of [nf, ef, sB, rB, pB, bpB, scB, read]) b.destroy();
    return result;
  }
}

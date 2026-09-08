/**
 * PhysSim: the browser twin of src/physsplat/model/live.py.
 * Holds body state + velocity history; step() builds the graph, runs the
 * ONNX network, and integrates. Everything numeric comes from runtime.json.
 */
import {
  actionFeature, buildEdges, edgeFeatures, externalAccels, matVec,
  quatToMatrix, stepBodies,
} from "./physics.js";

export class PhysSim {
  constructor(ort, session, runtime, packet) {
    this.ort = ort;
    this.session = session;
    this.rt = runtime;
    this.packet = packet;
    this.reset();
  }

  reset() {
    const bs = this.packet.bodies;
    this.B = bs.length;
    this.mass = bs.map((b) => b.mass);
    this.inertia = bs.map((b) => b.inertia);
    this.offsets = bs.map((b) => b.offsets);
    this.counts = bs.map((b) => b.offsets.length);
    this.N = this.counts.reduce((a, c) => a + c, 0);
    this.bodyIds = new Int32Array(this.N + 1);
    let k = 0;
    bs.forEach((b, i) => { for (const _ of b.offsets) this.bodyIds[k++] = i; });
    this.bodyIds[this.N] = this.B; // dummy
    this.state = {
      pos: bs.map((b) => [...b.pos]),
      quat: bs.map((b) => [...b.quat]),
      linvel: bs.map(() => [0, 0, 0]),
      angvel: bs.map(() => [0, 0, 0]),
    };
    const H = this.rt.history;
    this.linHist = Array.from({ length: H }, () => bs.map(() => [0, 0, 0]));
    this.angHist = Array.from({ length: H }, () => bs.map(() => [0, 0, 0]));
    this.quatHist = Array.from({ length: H }, () => bs.map((b) => [...b.quat]));
    this.bodyScalars = this.#bodyScalars();
  }

  #bodyScalars() {
    const n = this.rt.normalize;
    const out = new Float32Array((this.B + 1) * 4);
    for (let b = 0; b <= this.B; b++) {
      const m = b < this.B ? this.mass[b] : 0.01;
      const I = b < this.B ? this.inertia[b] : [1e-6, 1e-6, 1e-6];
      out[4 * b] = (Math.log10(m) + n.mass_log_shift) / n.mass_log_scale;
      for (let k = 0; k < 3; k++)
        out[4 * b + 1 + k] =
          (Math.log10(Math.max(I[k], 1e-12)) + n.inertia_log_shift) /
          n.inertia_log_scale;
    }
    return out;
  }

  particlesWorld() {
    const parts = new Float64Array(this.N * 3);
    let k = 0;
    for (let b = 0; b < this.B; b++) {
      const R = quatToMatrix(this.state.quat[b]);
      const p = this.state.pos[b];
      for (const o of this.offsets[b]) {
        const w = matVec(R, o);
        parts[k++] = w[0] + p[0];
        parts[k++] = w[1] + p[1];
        parts[k++] = w[2] + p[2];
      }
    }
    return parts;
  }

  /** per-particle velocity at history slot h (derived from body state) */
  #particleVels(h) {
    const v = new Float64Array(this.N * 3);
    let k = 0;
    for (let b = 0; b < this.B; b++) {
      const R = quatToMatrix(this.quatHist[h][b]);
      const lv = this.linHist[h][b], av = this.angHist[h][b];
      for (const o of this.offsets[b]) {
        const r = matVec(R, o);
        v[k++] = lv[0] + av[1] * r[2] - av[2] * r[1];
        v[k++] = lv[1] + av[2] * r[0] - av[0] * r[2];
        v[k++] = lv[2] + av[0] * r[1] - av[1] * r[0];
      }
    }
    return v;
  }

  async step(actBody = -1, actPoint = null, actForce = null) {
    const rt = this.rt, n = rt.normalize, H = rt.history;
    const parts = this.particlesWorld();
    const vels = Array.from({ length: H }, (_, h) => this.#particleVels(h));

    // ---- node features (N+1 with dummy), float32 ----
    const NODE_DIM = H * 3 + 1 + 1 + 3 + 3;
    const nf = new Float32Array((this.N + 1) * NODE_DIM);
    let aext = null;
    if (actBody >= 0) {
      const sel = new Uint8Array(this.N);
      for (let i = 0; i < this.N; i++) sel[i] = this.bodyIds[i] === actBody ? 1 : 0;
      aext = actionFeature(parts, sel, actPoint, actForce,
        this.mass[actBody], rt.falloff_sigma);
    }
    for (let i = 0; i < this.N; i++) {
      const row = i * NODE_DIM;
      for (let h = 0; h < H; h++)
        for (let k = 0; k < 3; k++)
          nf[row + h * 3 + k] =
            (vels[h][3 * i + k] - n.vel_mean[k]) / n.vel_std[k];
      nf[row + H * 3] =
        Math.min(Math.max(parts[3 * i + 2], 0), rt.contact_radius) /
        rt.contact_radius;
      const b = this.bodyIds[i];
      nf[row + H * 3 + 1] = this.bodyScalars[4 * b]; // log-mass (same normalizer)
      for (let k = 0; k < 3; k++)
        nf[row + H * 3 + 2 + k] = this.bodyScalars[4 * b + 1 + k];
      if (aext && aext.has(i)) {
        const a = aext.get(i);
        for (let k = 0; k < 3; k++)
          nf[row + H * 3 + 5 + k] = a[k] / n.a_ext_scale;
      }
    }

    // ---- edges (bucketed to dummy node) ----
    const lastVel = vels[H - 1];
    const { senders, receivers } = buildEdges(parts, lastVel,
      rt.contact_radius, rt.dt);
    const E = senders.length;
    const Epad = Math.max(1, Math.ceil(E / rt.edge_bucket)) * rt.edge_bucket;
    const s64 = new BigInt64Array(Epad), r64 = new BigInt64Array(Epad);
    const ef = new Float32Array(Epad * 5);
    const efReal = edgeFeatures(parts, senders, receivers, this.bodyIds,
      rt.contact_radius);
    ef.set(efReal);
    for (let e = 0; e < E; e++) {
      s64[e] = BigInt(senders[e]);
      r64[e] = BigInt(receivers[e]);
    }
    for (let e = E; e < Epad; e++) { s64[e] = BigInt(this.N); r64[e] = BigInt(this.N); }
    const bids64 = new BigInt64Array(this.N + 1);
    for (let i = 0; i <= this.N; i++) bids64[i] = BigInt(this.bodyIds[i]);

    const T = this.ort.Tensor;
    const out = await this.session.run({
      node_feats: new T("float32", nf, [this.N + 1, NODE_DIM]),
      edge_feats: new T("float32", ef, [Epad, 5]),
      senders: new T("int64", s64, [Epad]),
      receivers: new T("int64", r64, [Epad]),
      body_ids: new T("int64", bids64, [this.N + 1]),
      body_scalars: new T("float32", this.bodyScalars, [this.B + 1, 4]),
    });
    const pred = out.residual_norm.data; // (B+1, 6) normalized

    const residual = new Float64Array(this.B * 6);
    for (let b = 0; b < this.B; b++)
      for (let k = 0; k < 6; k++)
        residual[6 * b + k] =
          pred[6 * b + k] * n.target_std[k] + n.target_mean[k];

    const ext = externalAccels(this.state.pos, this.state.quat, this.mass,
      this.inertia, actBody, actPoint ?? [0, 0, 0], actForce ?? [0, 0, 0],
      this.B);
    stepBodies(this.state, residual, ext, rt.dt, rt.gravity);
    this.linHist.shift(); this.linHist.push(this.state.linvel.map((v) => [...v]));
    this.angHist.shift(); this.angHist.push(this.state.angvel.map((v) => [...v]));
    this.quatHist.shift(); this.quatHist.push(this.state.quat.map((q) => [...q]));
    return this.state;
  }
}

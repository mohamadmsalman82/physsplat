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
  /**
   * backend: { kind: "gpu", net: GpuNet } (custom WebGPU, ~30 ms/step) or
   *          { kind: "ort", ort, session } (ONNX Runtime Web fallback).
   */
  constructor(backend, runtime, packet, { groundGuard = true } = {}) {
    this.backend = backend;
    this.rt = runtime;
    this.packet = packet;
    this.groundGuard = groundGuard;   // off in parity tests (Python eval is unguarded)
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
    this.restCount = null;
    // Reconstructed poses overlap by up to ~1 cm (single-view depth error).
    // Resolve that before the first frame instead of letting the guards
    // jolt the pile apart in front of the viewer.
    if (this.groundGuard && bs.length) {
      for (let it = 0; it < 30; it++) { this.#capsuleGuard(); this.#groundGuard(); }
      this.state.linvel = bs.map(() => [0, 0, 0]);
    }
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

  /**
   * Analytic ground guard (design doc, non-penetration section): the learned
   * model has no hard constraint and its ground contact residual can run a
   * little weak on out-of-distribution reconstructed bodies (measured: a
   * slow ~1.5 mm/s sink on photo scenes). If any particle dips below z=0,
   * lift the body out and cancel its downward velocity. Millimetre-scale
   * cleanup, never applied during evaluation.
   */
  /** A body is "lying" when its center of mass is within ~2.5 radii of the
   * floor: a pencil on its side. Standing on its tip is NOT lying, and the
   * guards deliberately leave that case to gravity + the model, otherwise
   * they pin the tip and the pencil balances upright forever (a blind test
   * caught exactly that). */
  #lying(b) {
    const cap = this.packet.bodies[b].capsule;
    const r = cap ? cap.radius : 0.01;
    return this.state.pos[b][2] < 2.5 * r;
  }

  #groundGuard() {
    for (let b = 0; b < this.B; b++) {
      const R = quatToMatrix(this.state.quat[b]);
      let minz = Infinity;
      for (const o of this.offsets[b]) {
        const z = R[6] * o[0] + R[7] * o[1] + R[8] * o[2] + this.state.pos[b][2];
        if (z < minz) minz = z;
      }
      if (minz < 0) {
        this.state.pos[b][2] -= minz;
        if (this.state.linvel[b][2] < 0 && this.#lying(b)) this.state.linvel[b][2] = 0;
      }
    }
  }

  /**
   * Capsule-capsule guard (design doc: analytic non-penetration cleanup).
   * Every demo body carries a capsule proxy (axis, half length, radius).
   * If two capsules overlap, push them apart along the closest-point
   * direction (mass-weighted) and cancel the approaching velocity. The
   * learned model does the contact physics; this only removes the residual
   * overlap it leaves behind so pencils never visibly pass through each other.
   */
  #capsuleGuard() {
    const bs = this.packet.bodies;
    if (!bs.length || !bs[0].capsule) return;
    const seg = (b) => {
      const R = quatToMatrix(this.state.quat[b]);
      const a = matVec(R, bs[b].capsule.axis);
      const h = bs[b].capsule.half, p = this.state.pos[b];
      return { p, a, h, r: bs[b].capsule.radius };
    };
    const segs = bs.map((_, b) => seg(b));
    for (let i = 0; i < this.B; i++) for (let j = i + 1; j < this.B; j++) {
      const A = segs[i], Bc = segs[j];
      // closest points between segments p +/- h*a (clamped, standard form)
      const r = [A.p[0] - Bc.p[0], A.p[1] - Bc.p[1], A.p[2] - Bc.p[2]];
      const dot = (u, v) => u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
      const aa = dot(A.a, A.a), ee = dot(Bc.a, Bc.a), bb = dot(A.a, Bc.a);
      const cc = dot(A.a, r), ff = dot(Bc.a, r);
      const den = aa * ee - bb * bb;
      let s = den > 1e-12 ? (bb * ff - cc * ee) / den : 0;
      s = Math.max(-A.h, Math.min(A.h, s));
      let t = (bb * s + ff) / ee;
      t = Math.max(-Bc.h, Math.min(Bc.h, t));
      s = Math.max(-A.h, Math.min(A.h, (bb * t - cc) / aa));
      const ca = [A.p[0] + s * A.a[0], A.p[1] + s * A.a[1], A.p[2] + s * A.a[2]];
      const cb = [Bc.p[0] + t * Bc.a[0], Bc.p[1] + t * Bc.a[1], Bc.p[2] + t * Bc.a[2]];
      const d = [ca[0] - cb[0], ca[1] - cb[1], ca[2] - cb[2]];
      const dist = Math.hypot(...d);
      const pen = (A.r + Bc.r) - dist;
      if (pen <= 0 || dist < 1e-9) continue;
      const n = d.map((x) => x / dist);         // from j toward i
      const mi = this.mass[i], mj = this.mass[j];
      const wi = mj / (mi + mj), wj = mi / (mi + mj);
      for (let k = 0; k < 3; k++) {
        this.state.pos[i][k] += n[k] * pen * wi;
        this.state.pos[j][k] -= n[k] * pen * wj;
      }
      // cancel approaching relative velocity along the normal
      const vi = this.state.linvel[i], vj = this.state.linvel[j];
      const vrel = dot([vi[0] - vj[0], vi[1] - vj[1], vi[2] - vj[2]], n);
      if (vrel < 0) for (let k = 0; k < 3; k++) {
        vi[k] -= n[k] * vrel * wi; vj[k] += n[k] * vrel * wj;
      }
    }
  }

  /**
   * Settle damping (design doc: demo hygiene). A body that has been nearly
   * at rest for a while gets its residual velocities zeroed, so the learned
   * model's slow creep cannot accumulate on an untouched scene. Any real
   * motion (a poke, a collision, a grab) immediately clears the counter.
   */
  #settle() {
    this.restCount ??= new Int32Array(this.B);
    for (let b = 0; b < this.B; b++) {
      const v = this.state.linvel[b], w = this.state.angvel[b];
      // only a body lying on the floor can settle; a standing or lifted
      // body must keep obeying gravity
      const slow = this.#lying(b) && Math.hypot(...v) < 0.012 && Math.hypot(...w) < 0.35;
      this.restCount[b] = slow ? this.restCount[b] + 1 : 0;
      if (this.restCount[b] >= 12) {
        v[0] = v[1] = v[2] = 0; w[0] = w[1] = w[2] = 0;
      }
    }
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
    const T0 = performance.now();
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
    const T1 = performance.now();
    // Export contract (race-free segmented aggregation): edges sorted by
    // receiver with segment pointers; nodes ordered by body with body
    // pointers. No edge padding needed here (that was for MPS kernels).
    const Ereal = senders.length;
    const efReal = edgeFeatures(parts, senders, receivers, this.bodyIds,
      rt.contact_radius);
    const order = Array.from({ length: Ereal }, (_, i) => i)
      .sort((a, b) => receivers[a] - receivers[b] || a - b);
    // Pad the edge count to a bucket so tensor shapes are static per scene
    // (lets ORT capture and replay the GPU command stream). Padded edges
    // are self-loops on the dummy node (last index), which sorts last and
    // lands in the dummy's own segment: zero effect on real nodes.
    const Nn = this.N + 1;                       // nodes incl. one dummy
    const E = Math.max(1, Math.ceil(Ereal / rt.edge_bucket)) * rt.edge_bucket;
    const s64 = new BigInt64Array(E), r64 = new BigInt64Array(E);
    const ef = new Float32Array(E * 5);
    order.forEach((o, e) => {
      s64[e] = BigInt(senders[o]); r64[e] = BigInt(receivers[o]);
      for (let k = 0; k < 5; k++) ef[5 * e + k] = efReal[5 * o + k];
    });
    for (let e = Ereal; e < E; e++) { s64[e] = BigInt(this.N); r64[e] = BigInt(this.N); }
    const segPtr = new BigInt64Array(Nn + 1);
    let ei = 0;
    for (let i = 0; i <= Nn; i++) {
      while (ei < E && Number(r64[ei]) < i) ei++;
      segPtr[i] = BigInt(ei);
    }
    const bids64 = new BigInt64Array(Nn);
    for (let i = 0; i < Nn; i++) bids64[i] = BigInt(this.bodyIds[i]);
    const bodyPtr = new BigInt64Array(this.B + 2);
    let ni = 0;
    for (let b = 0; b <= this.B + 1; b++) {
      while (ni < Nn && this.bodyIds[ni] < b) ni++;
      bodyPtr[b] = BigInt(ni);
    }

    const T2 = performance.now();
    let pred;   // (B+1, 6) normalized residuals
    if (this.backend.kind === "gpu") {
      const u32 = (a) => Uint32Array.from(a, (x) => Number(x));
      pred = await this.backend.net.forward({
        nodeFeats: nf, nodeDim: NODE_DIM, edgeFeats: ef,
        senders: u32(s64), receivers: u32(r64), segPtr: u32(segPtr),
        bodyPtr: u32(bodyPtr), bodyScalars: this.bodyScalars,
        N: Nn, E, B: this.B + 1, nReal: this.N,
      });
    } else {
      const T = this.backend.ort.Tensor;
      const out = await this.backend.session.run({
        node_feats: new T("float32", nf, [Nn, NODE_DIM]),
        edge_feats: new T("float32", ef, [E, 5]),
        senders: new T("int64", s64, [E]),
        receivers: new T("int64", r64, [E]),
        seg_ptr: new T("int64", segPtr, [Nn + 1]),
        body_ids: new T("int64", bids64, [Nn]),
        body_ptr: new T("int64", bodyPtr, [this.B + 2]),
        body_scalars: new T("float32", this.bodyScalars, [this.B + 1, 4]),
      });
      pred = out.residual_norm.location === "gpu-buffer"
        ? await out.residual_norm.getData(true) : out.residual_norm.data;
    }
    const T3 = performance.now();
    this.timing = { features_ms: T1 - T0, graph_ms: T2 - T1, net_ms: T3 - T2,
      E: Ereal, N: this.N, backend: this.backend.kind };

    const residual = new Float64Array(this.B * 6);
    for (let b = 0; b < this.B; b++)
      for (let k = 0; k < 6; k++)
        residual[6 * b + k] =
          pred[6 * b + k] * n.target_std[k] + n.target_mean[k];

    const ext = externalAccels(this.state.pos, this.state.quat, this.mass,
      this.inertia, actBody, actPoint ?? [0, 0, 0], actForce ?? [0, 0, 0],
      this.B);
    stepBodies(this.state, residual, ext, rt.dt, rt.gravity);
    if (this.groundGuard) { this.#groundGuard(); this.#capsuleGuard(); this.#settle(); }
    this.linHist.shift(); this.linHist.push(this.state.linvel.map((v) => [...v]));
    this.angHist.shift(); this.angHist.push(this.state.angvel.map((v) => [...v]));
    this.quatHist.shift(); this.quatHist.push(this.state.quat.map((q) => [...q]));
    return this.state;
  }
}

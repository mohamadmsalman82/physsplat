/**
 * PhysSim: the browser twin of src/physsplat/model/live.py.
 * Holds body state + velocity history; step() builds the graph, runs the
 * ONNX network, and integrates. Everything numeric comes from runtime.json.
 */
import {
  actionFeature, buildEdges, capsuleClosest, capsuleWorld, edgeFeatures,
  externalAccels, quatToMatrix, stepBodies,
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
    this.stepCount = 0;
    this.last = null;                 // per-step diagnostics (see #newLast)
    // Reconstructed poses overlap by up to ~1 cm (single-view depth error).
    // Resolve that before the first frame instead of letting the guards
    // jolt the pile apart in front of the viewer.
    if (this.groundGuard && bs.length) {
      this.last = this.#newLast();
      for (let it = 0; it < 30; it++) { this.#capsuleGuard(); this.#groundGuard(); }
      this.state.linvel = bs.map(() => [0, 0, 0]);
      this.loadCorrection = {                       // what the pre-settle moved
        ground_mm: Array.from(this.last.guard.ground, (x) => x * 1e3),
        capsule_mm: Array.from(this.last.guard.capsule, (x) => x * 1e3),
      };
      this.last = null;
    }
  }

  /**
   * Everything a step decided, kept for diagnostics: the model's residual
   * accelerations (SI, after de-normalization), the analytic external
   * accelerations, the action, and how much each guard intervened.
   */
  #newLast() {
    const B = this.B;
    return {
      residual: new Float64Array(B * 6),
      ext: { lin: new Float64Array(B * 3), ang: new Float64Array(B * 3) },
      action: null,
      guard: {
        ground: new Float64Array(B),       // metres lifted out of the floor
        capsule: new Float64Array(B),      // metres pushed out of other bodies
        pairs: [],                         // {i, j, pen} overlaps corrected
        settled: new Uint8Array(B),        // 1 if settle held the body still
        freeFlight: new Uint8Array(B),     // 1 if the model residual was zeroed
      },
    };
  }

  /**
   * Free-flight rule. A rigid body touching nothing feels only gravity and
   * the applied force, both integrated analytically, so the learned
   * residual must be zero for it. The network never saw a body hanging
   * motionless in mid-air (training bodies at rest are always supported)
   * and answers "+g, hold still" for one: a lifted pencil released from
   * the cursor hovered forever (diagnostics probe, 53 mm above the floor).
   * "Touching nothing" is read straight off the contact graph: no edge to
   * another body's particle and no particle within the contact radius of
   * the floor.
   */
  #freeFlight(parts, senders, receivers) {
    const touched = new Uint8Array(this.B);
    for (let e = 0; e < senders.length; e++) {
      const bs = this.bodyIds[senders[e]], br = this.bodyIds[receivers[e]];
      if (bs !== br) { touched[bs] = 1; touched[br] = 1; }
    }
    const rc = this.rt.contact_radius;
    for (let i = 0; i < this.N; i++)
      if (parts[3 * i + 2] < rc) touched[this.bodyIds[i]] = 1;
    for (let b = 0; b < this.B; b++) if (!touched[b]) {
      for (let k = 0; k < 6; k++) this.last.residual[6 * b + k] = 0;
      this.last.guard.freeFlight[b] = 1;
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
        if (this.last) this.last.guard.ground[b] += -minz;
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
    const segs = bs.map((b, i) => capsuleWorld(this.state.pos[i], this.state.quat[i], b.capsule));
    for (let i = 0; i < this.B; i++) for (let j = i + 1; j < this.B; j++) {
      const { pen, dist, n } = capsuleClosest(segs[i], segs[j]);   // n: j -> i
      if (pen <= 0 || dist < 1e-9) continue;
      const mi = this.mass[i], mj = this.mass[j];
      const wi = mj / (mi + mj), wj = mi / (mi + mj);
      for (let k = 0; k < 3; k++) {
        this.state.pos[i][k] += n[k] * pen * wi;
        this.state.pos[j][k] -= n[k] * pen * wj;
      }
      if (this.last) {
        this.last.guard.capsule[i] += pen * wi;
        this.last.guard.capsule[j] += pen * wj;
        this.last.guard.pairs.push({ i, j, pen });
      }
      // cancel approaching relative velocity along the normal
      const vi = this.state.linvel[i], vj = this.state.linvel[j];
      const vrel = (vi[0] - vj[0]) * n[0] + (vi[1] - vj[1]) * n[1] + (vi[2] - vj[2]) * n[2];
      if (vrel < 0) for (let k = 0; k < 3; k++) {
        vi[k] -= n[k] * vrel * wi; vj[k] += n[k] * vrel * wj;
      }
    }
  }

  /**
   * Settle (design doc: demo hygiene). A body that has been nearly at rest
   * for a while is held exactly still: velocities zeroed AND the pose
   * restored to what it was before this step. Zeroing velocity alone was
   * not enough: the model's residual left a ~0.15 m/s^2 net sag, the
   * capsule guard pushed the body back out along the contact normal, and
   * that position-only push crept the pile sideways at ~2 mm/s with zero
   * velocity (diagnostics: "creep", 9 mm in 4 s). Any real motion (a poke,
   * a collision, a grab) clears the counter and the body moves again.
   */
  #settle(prePos, preQuat, actBody) {
    this.restCount ??= new Int32Array(this.B);
    const bs = this.packet.bodies;
    // a body may settle only when something holds it up: the floor under a
    // lying body, or another capsule within a contact gap. A body in the air
    // stays with gravity (free-flight rule), and a pencil balanced on end
    // (elevation > 45 deg) is left to the model so it can fall over.
    const segs = bs.map((b, i) => capsuleWorld(this.state.pos[i], this.state.quat[i], b.capsule));
    const supported = new Uint8Array(this.B);
    for (let i = 0; i < this.B; i++) {
      if (this.#lying(i)) supported[i] = 1;
      for (let j = i + 1; j < this.B; j++)
        if (capsuleClosest(segs[i], segs[j]).pen > -1.5e-3) { supported[i] = 1; supported[j] = 1; }
    }
    for (let b = 0; b < this.B; b++) {
      const v = this.state.linvel[b], w = this.state.angvel[b];
      const standing = Math.abs(segs[b].a[2]) > 0.7071;
      const slow = b !== actBody && supported[b] && !standing &&
        Math.hypot(...v) < 0.012 && Math.hypot(...w) < 0.35;
      this.restCount[b] = slow ? this.restCount[b] + 1 : 0;
      if (this.restCount[b] >= 12) {
        v[0] = v[1] = v[2] = 0; w[0] = w[1] = w[2] = 0;
        this.state.pos[b] = [...prePos[b]];
        this.state.quat[b] = [...preQuat[b]];
        if (this.last) this.last.guard.settled[b] = 1;
      }
    }
  }

  particlesWorld() {
    const parts = new Float64Array(this.N * 3);
    let k = 0;
    for (let b = 0; b < this.B; b++) {
      const R = quatToMatrix(this.state.quat[b]);
      const p = this.state.pos[b], os = this.offsets[b];
      for (let n = 0; n < os.length; n++) {
        const o = os[n];
        parts[k++] = R[0] * o[0] + R[1] * o[1] + R[2] * o[2] + p[0];
        parts[k++] = R[3] * o[0] + R[4] * o[1] + R[5] * o[2] + p[1];
        parts[k++] = R[6] * o[0] + R[7] * o[1] + R[8] * o[2] + p[2];
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
      const os = this.offsets[b];
      for (let n = 0; n < os.length; n++) {
        const o = os[n];
        const r0 = R[0] * o[0] + R[1] * o[1] + R[2] * o[2];
        const r1 = R[3] * o[0] + R[4] * o[1] + R[5] * o[2];
        const r2 = R[6] * o[0] + R[7] * o[1] + R[8] * o[2];
        v[k++] = lv[0] + av[1] * r2 - av[2] * r1;
        v[k++] = lv[1] + av[2] * r0 - av[0] * r2;
        v[k++] = lv[2] + av[0] * r1 - av[1] * r0;
      }
    }
    return v;
  }

  async step(actBody = -1, actPoint = null, actForce = null) {
    const rt = this.rt, n = rt.normalize, H = rt.history;
    const T0 = performance.now();
    this.last = this.#newLast();
    if (actBody >= 0) this.last.action = { body: actBody, point: [...actPoint], force: [...actForce] };
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
    // counting sort by receiver (stable): edges arrive in sender-major order
    // and a comparison sort on ~5k edges was a measurable slice of the step
    const order = new Int32Array(Ereal);
    {
      const cnt = new Int32Array(this.N + 2);
      for (let e = 0; e < Ereal; e++) cnt[receivers[e] + 1]++;
      for (let i = 1; i <= this.N + 1; i++) cnt[i] += cnt[i - 1];
      for (let e = 0; e < Ereal; e++) order[cnt[receivers[e]]++] = e;
    }
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

    const residual = this.last.residual;
    for (let b = 0; b < this.B; b++)
      for (let k = 0; k < 6; k++)
        residual[6 * b + k] =
          pred[6 * b + k] * n.target_std[k] + n.target_mean[k];

    const ext = externalAccels(this.state.pos, this.state.quat, this.mass,
      this.inertia, actBody, actPoint ?? [0, 0, 0], actForce ?? [0, 0, 0],
      this.B);
    this.last.ext = ext;
    if (this.groundGuard) this.#freeFlight(parts, senders, receivers);
    const prePos = this.state.pos.map((p) => [...p]);
    const preQuat = this.state.quat.map((q) => [...q]);
    stepBodies(this.state, residual, ext, rt.dt, rt.gravity);
    if (this.groundGuard) {
      // capsules first: their push can move a body into the floor, and the
      // floor is the hard constraint (diagnostics caught 1 mm "sinking"
      // episodes from the reverse order)
      this.#capsuleGuard(); this.#groundGuard(); this.#settle(prePos, preQuat, actBody);
    }
    this.linHist.shift(); this.linHist.push(this.state.linvel.map((v) => [...v]));
    this.angHist.shift(); this.angHist.push(this.state.angvel.map((v) => [...v]));
    this.quatHist.shift(); this.quatHist.push(this.state.quat.map((q) => [...q]));
    this.stepCount++;
    return this.state;
  }
}

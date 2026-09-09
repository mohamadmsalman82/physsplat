/**
 * PhysSim: the browser twin of src/physsplat/model/live.py.
 * Holds body state + velocity history; step() builds the graph, runs the
 * ONNX network, and integrates. Everything numeric comes from runtime.json.
 */
import {
  actionFeature, buildEdges, capsuleClosest, capsuleWorld, edgeFeatures,
  externalAccels, quatFromRotvec, quatMul, quatToMatrix, stepBodies,
  supportAnalysis, supportPoints,
} from "./physics.js";

const SETTLE_STEPS = 15;   // consecutive slow steps before a body is held still

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
    this.gen = (this.gen ?? 0) + 1;   // a step in flight across a reset aborts
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
    // Resolve that before the first frame by lifting the upper body of each
    // overlapping pair straight up: the photo's x/y arrangement is what the
    // reconstruction gets right and heights are what it gets wrong. Pushing
    // along contact normals instead moved pencils up to 6 cm sideways
    // (blind test round 3: "the loader rewrites the photo").
    if (this.groundGuard && bs.length) {
      const lift = new Float64Array(this.B);
      for (let it = 0; it < 60; it++) {
        let moved = false;
        const segs = bs.map((b, i) => capsuleWorld(this.state.pos[i], this.state.quat[i], b.capsule));
        for (let i = 0; i < this.B; i++) for (let j = i + 1; j < this.B; j++) {
          const { pen } = capsuleClosest(segs[i], segs[j]);
          if (pen <= 3e-4) continue;
          const up = this.state.pos[i][2] >= this.state.pos[j][2] ? i : j;
          this.state.pos[up][2] += pen; lift[up] += pen; moved = true;
        }
        for (let b = 0; b < this.B; b++) {
          const R = quatToMatrix(this.state.quat[b]);
          let minz = Infinity;
          for (const o of this.offsets[b])
            minz = Math.min(minz, R[6] * o[0] + R[7] * o[1] + R[8] * o[2] + this.state.pos[b][2]);
          if (minz < 0) { this.state.pos[b][2] -= minz; lift[b] -= minz; moved = true; }
        }
        if (!moved) break;
      }
      this.loadCorrection = { lift_mm: Array.from(lift, (x) => x * 1e3) };
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
        pivot: new Uint8Array(B),          // 1 if the pivot rule drove the body
        energyScale: 1,                    // <1 if the residual was scaled for energy
        energyGain_uJ: 0,                  // what the unscaled residual would have added
      },
    };
  }

  /** Total mechanical energy of a state (J): kinetic + gravitational. */
  #energy(st) {
    const g = this.rt.gravity;
    let E = 0;
    for (let b = 0; b < this.B; b++) {
      const v = st.linvel[b], w = st.angvel[b], m = this.mass[b], I = this.inertia[b];
      const R = quatToMatrix(st.quat[b]);
      const wb = [R[0] * w[0] + R[3] * w[1] + R[6] * w[2],
        R[1] * w[0] + R[4] * w[1] + R[7] * w[2], R[2] * w[0] + R[5] * w[1] + R[8] * w[2]];
      E += 0.5 * m * (v[0] ** 2 + v[1] ** 2 + v[2] ** 2) +
        0.5 * (I[0] * wb[0] ** 2 + I[1] * wb[1] ** 2 + I[2] * wb[2] ** 2) + m * g * st.pos[b][2];
    }
    return E;
  }

  /**
   * No free energy. Contact forces from static things cannot add
   * mechanical energy to a pile; only an applied force can, and only as
   * much work as it does. The network's residual on a reconstructed pile
   * did not respect that: a pencil at rest reared up to 89 degrees on its
   * own, twice, and balanced on its end (diagnostics track, IMG_8596).
   * Trial-integrate the step; if the energy rises by more than the
   * action's work (plus a small allowance for pushing out of overlaps),
   * scale the residual down until it does not.
   */
  #energyRule(residual, ext, actBody, actPoint, actForce) {
    const rt = this.rt, st = this.state;
    const E0 = this.#energy(st);
    let W = 0;
    if (actBody >= 0) {
      const v = st.linvel[actBody], w = st.angvel[actBody], p = st.pos[actBody];
      const r = [actPoint[0] - p[0], actPoint[1] - p[1], actPoint[2] - p[2]];
      const vp = [v[0] + w[1] * r[2] - w[2] * r[1], v[1] + w[2] * r[0] - w[0] * r[2],
        v[2] + w[0] * r[1] - w[1] * r[0]];
      W = (actForce[0] * vp[0] + actForce[1] * vp[1] + actForce[2] * vp[2]) * rt.dt;
    }
    const allow = Math.max(W, 0) + 5e-7;          // J per step; 5e-7 ~ 0.5 mm/s of lift
    const trial = (scale) => {
      const copy = { pos: st.pos.map((x) => [...x]), quat: st.quat.map((x) => [...x]),
        linvel: st.linvel.map((x) => [...x]), angvel: st.angvel.map((x) => [...x]) };
      const res = scale === 1 ? residual : residual.map((x) => x * scale);
      stepBodies(copy, res, ext, rt.dt, rt.gravity);
      return this.#energy(copy) - E0;
    };
    const gain = trial(1);
    this.last.guard.energyGain_uJ = (gain - Math.max(W, 0)) * 1e6;
    if (gain <= allow) return;
    // bisection on the residual scale (5 rounds is 3% resolution)
    let lo = 0, hi = 1;
    if (trial(0) > allow) { lo = 0; hi = 0; }   // gravity alone exceeds it: drop the residual
    for (let it = 0; it < 5 && hi > lo; it++) {
      const mid = (lo + hi) / 2;
      if (trial(mid) > allow) hi = mid; else lo = mid;
    }
    for (let i = 0; i < residual.length; i++) residual[i] *= lo;
    this.last.guard.energyScale = lo;
  }

  /**
   * Pivot rule. A body whose centre of mass is not over its support cannot
   * be at rest: gravity's torque about the support must rotate it. The
   * network holds such a body still (a pencil that landed on its end came
   * to rest 9 degrees up with the far end in the air; the round-1 tester
   * saw pencils balanced on their tips). The body is integrated
   * analytically as a pendulum about the hinge, the point of the support
   * polygon's boundary nearest the centre of mass (an edge between two
   * contacts, or a lone contact): alpha = (r x m g) / I_hinge, the model
   * residual is dropped, and the centre of mass moves with omega x r so
   * the hinge stays put. A body whose centre of mass is over its support
   * is left to the model, and so is anything the cursor holds.
   */
  #pivotRule(parts, ext, actBody) {
    const bs = this.packet.bodies, rt = this.rt, g = rt.gravity;
    const segs = bs.map((b, i) => capsuleWorld(this.state.pos[i], this.state.quat[i], b.capsule));
    const out = [];
    let k = 0;
    // support verdicts for this step, computed once on the state the
    // diagnostics recorded last step, and shared with settle so the two
    // can never disagree about whether a body may be held still
    this.balancedNow ??= new Uint8Array(this.B);
    this.balancedNow.fill(0);
    this.pivoting ??= new Uint8Array(this.B);
    for (let b = 0; b < this.B; b++) {
      const n = this.counts[b], start = k; k += n;
      if (this.last.guard.freeFlight[b]) continue;
      const rc = rt.contact_radius;
      const sp = supportPoints(parts, start, n, segs, b, rc, rc);
      const an = supportAnalysis(this.state.pos[b], sp.points);
      this.balancedNow[b] = an.n && an.balanced ? 1 : 0;
      if (b === actBody) continue;
      if (!an.n) {
        // touching things, but nothing from below (only pencils on its
        // back, or a neighbour beside it): it falls like a free body, and
        // whatever rests on it comes down with it
        for (let q = 0; q < 6; q++) this.last.residual[6 * b + q] = 0;
        this.last.guard.freeFlight[b] = 1;
        this.pivoting[b] = 0;
        continue;
      }
      // hysteresis: start tipping only when clearly off the support, keep
      // tipping until clearly over it. Flip-flopping at the boundary (pivot
      // one step, model the next) crept bodies at ~1 mm/s.
      const off = an.dist > (this.pivoting[b] ? 4e-3 : 7e-3);
      this.pivoting[b] = off ? 1 : 0;
      if (!off) continue;
      const P = an.hinge;
      const c = this.state.pos[b], m = this.mass[b];
      const r = [c[0] - P[0], c[1] - P[1], c[2] - P[2]];
      // torque of gravity about P: r x (0, 0, -m g)
      const tq = [-r[1] * m * g, r[0] * m * g, 0];
      const tn = Math.hypot(tq[0], tq[1], tq[2]);
      if (tn < 1e-12) continue;
      const u = tq.map((x) => x / tn);
      // inertia about the pivot along u: u^T (R I R^T) u + m |r|^2
      const R = quatToMatrix(this.state.quat[b]), I = this.inertia[b];
      const ub = [R[0] * u[0] + R[3] * u[1] + R[6] * u[2],
        R[1] * u[0] + R[4] * u[1] + R[7] * u[2],
        R[2] * u[0] + R[5] * u[1] + R[8] * u[2]];              // u in body frame
      const Ip = I[0] * ub[0] ** 2 + I[1] * ub[1] ** 2 + I[2] * ub[2] ** 2 +
        m * (r[0] ** 2 + r[1] ** 2 + r[2] ** 2);
      const alpha = tn / Ip;
      for (let q = 0; q < 6; q++) this.last.residual[6 * b + q] = 0;
      for (let q = 0; q < 3; q++) ext.ang[3 * b + q] += alpha * u[q];
      // keep only the spin about the pivot axis: a body arriving with an
      // unrelated spin would otherwise drift its contact point
      const w = this.state.angvel[b];
      const wu = w[0] * u[0] + w[1] * u[1] + w[2] * u[2];
      for (let q = 0; q < 3; q++) w[q] = wu * u[q];
      this.last.guard.pivot[b] = 1;
      out.push({ b, P, start, n });
    }
    return out;
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
    const gap = new Float64Array(this.B).fill(Infinity);   // to nearest capsule
    // "supported" is the same 5 mm contact zone the pivot rule uses
    // (supportPoints): a tighter 1.5 mm test left bodies the model holds
    // 2-4 mm above their neighbours un-settled, creeping at ~1 mm/s and
    // ringing for a second after every landing
    const parts = this.particlesWorld();
    const spAll = [];
    let k = 0;
    for (let i = 0; i < this.B; i++) {
      const n = this.counts[i], start = k; k += n;
      // supported AND balanced, as judged at the start of this step by the
      // pivot rule (same state the diagnostics recorded): a body whose
      // centre of mass is off its support must keep moving, whatever its
      // speed. Judging it here, after the guards moved things, let a body
      // be held still 16 mm off its support while the record said so.
      spAll.push(supportPoints(parts, start, n, segs, i, this.rt.contact_radius, this.rt.contact_radius));
      supported[i] = this.balancedNow?.[i] ?? 0;
      for (let j = i + 1; j < this.B; j++) {
        const g = -capsuleClosest(segs[i], segs[j]).pen;
        gap[i] = Math.min(gap[i], g); gap[j] = Math.min(gap[j], g);
      }
    }
    for (let b = 0; b < this.B; b++) {
      const v = this.state.linvel[b], w = this.state.angvel[b];
      const standing = Math.abs(segs[b].a[2]) > 0.7071;
      // Slow motion of a supported body dies quickly in reality (friction
      // decelerates a sliding pencil at ~5 m/s^2, so 8 cm/s is gone in
      // 16 ms); the model instead rings for ~10 steps after every landing
      // (diagnostics: "jitter"). Damp the tail.
      if (b !== actBody && supported[b] && !this.pivoting?.[b] &&
          Math.hypot(...v) < 0.05 && Math.hypot(...w) < 2)
        for (let q = 0; q < 3; q++) { v[q] *= 0.5; w[q] *= 0.5; }
      // the model's contact response rings at 1-3 cm/s amplitude for a
      // second after a landing; a body that slow on a support is at rest
      const slow = b !== actBody && supported[b] && !standing && !this.pivoting?.[b] &&
        Math.hypot(...v) < 0.03 && Math.hypot(...w) < 0.6;
      this.restCount[b] = slow ? this.restCount[b] + 1 : 0;
      if (this.restCount[b] === SETTLE_STEPS) {
        // The model's contact response equilibrates 2-5 mm above whatever
        // it landed on (particle contact radius 6 mm), so a pencil that
        // just came to rest hovers slightly. Close that gap once, when it
        // settles. A pencil with one end on the floor and its body above a
        // neighbour closes it by rotating about the floor end; anything
        // else drops by the smaller of the floor gap and the capsule gap.
        // The capsule guard resolves any resulting overlap.
        const R = quatToMatrix(this.state.quat[b]);
        let lowest = Infinity;
        for (const o of this.offsets[b])
          lowest = Math.min(lowest, R[6] * o[0] + R[7] * o[1] + R[8] * o[2] + this.state.pos[b][2]);
        const sp = spAll[b];
        if (lowest < 1e-3 && sp.floor > 0 && sp.capsule.length && gap[b] > 3e-4 && gap[b] < 6e-3) {
          // hinge at the floor cluster; the widest capsule gap sets the angle
          const P = [0, 0, 0];
          for (let q = 0; q < sp.floor; q++) for (let c = 0; c < 3; c++) P[c] += sp.points[q][c] / sp.floor;
          let Q = null, g = 0;
          for (let q = 0; q < sp.capsule.length; q++) {
            const cc = capsuleClosest(segs[b], segs[sp.capsule[q]]);
            if (-cc.pen > g) { g = -cc.pen; Q = cc.ca; }
          }
          const r = [Q[0] - P[0], Q[1] - P[1], Q[2] - P[2]];
          const d = Math.hypot(...r);
          if (d > 0.02 && g > 3e-4) {
            const theta = g / d;
            // axis: horizontal, perpendicular to r, oriented so Q moves down
            let u = [r[1], -r[0], 0];
            const un = Math.hypot(u[0], u[1]);
            if (un > 1e-9) {
              u = u.map((x) => x / un);
              const dq = quatFromRotvec([u[0] * theta, u[1] * theta, 0]);
              const Rq = quatToMatrix(dq);
              const rot = (v) => [Rq[0] * v[0] + Rq[1] * v[1] + Rq[2] * v[2],
                Rq[3] * v[0] + Rq[4] * v[1] + Rq[5] * v[2], Rq[6] * v[0] + Rq[7] * v[1] + Rq[8] * v[2]];
              if (rot(r)[2] > r[2]) { u = u.map((x) => -x); }
              const dq2 = quatFromRotvec([u[0] * theta, u[1] * theta, 0]);
              const R2 = quatToMatrix(dq2);
              const rot2 = (v) => [R2[0] * v[0] + R2[1] * v[1] + R2[2] * v[2],
                R2[3] * v[0] + R2[4] * v[1] + R2[5] * v[2], R2[6] * v[0] + R2[7] * v[1] + R2[8] * v[2]];
              const c = this.state.pos[b];
              const nc = rot2([c[0] - P[0], c[1] - P[1], c[2] - P[2]]);
              this.state.pos[b] = [P[0] + nc[0], P[1] + nc[1], P[2] + nc[2]];
              this.state.quat[b] = quatMul(dq2, this.state.quat[b]);
              const qn = Math.hypot(...this.state.quat[b]);
              this.state.quat[b] = this.state.quat[b].map((x) => x / qn);
              prePos[b] = [...this.state.pos[b]]; preQuat[b] = [...this.state.quat[b]];
              if (this.last) this.last.guard.ground[b] -= g;   // recorded as a negative lift
            }
          }
        } else {
          const drop = Math.max(0, Math.min(lowest, gap[b]));
          if (drop > 3e-4 && drop < 6e-3) {
            this.state.pos[b][2] -= drop;
            prePos[b][2] -= drop;
            if (this.last) this.last.guard.ground[b] -= drop;   // recorded as a negative lift
          }
        }
      }
      if (this.restCount[b] >= SETTLE_STEPS) {
        v[0] = v[1] = v[2] = 0; w[0] = w[1] = w[2] = 0;
        this.state.pos[b] = [...prePos[b]];
        this.state.quat[b] = [...preQuat[b]];
        if (this.last) {
          this.last.guard.settled[b] = 1;
          // the guards' pushes on this body were undone by the restore, so
          // report no correction (a tester read the raw pushes as a
          // "fight" at rest)
          this.last.guard.ground[b] = 0; this.last.guard.capsule[b] = 0;
        }
      }
    }
    // hygiene caps: nothing in a pencil pile moves faster than this, and a
    // runaway (a spring at a bad lever arm, a bad contact step) must not
    // fling bodies off the table
    for (let b = 0; b < this.B; b++) {
      const v = this.state.linvel[b], w = this.state.angvel[b];
      const sv = Math.hypot(...v), sw = Math.hypot(...w);
      if (sv > 3) for (let q = 0; q < 3; q++) v[q] *= 3 / sv;
      if (sw > 60) for (let q = 0; q < 3; q++) w[q] *= 60 / sw;
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
    const gen = this.gen;
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
    // the scene was reset (or replaced) while the network ran: this step's
    // inputs describe a state that no longer exists, so drop it
    if (gen !== this.gen || !this.last) return this.state;

    const residual = this.last.residual;
    for (let b = 0; b < this.B; b++)
      for (let k = 0; k < 6; k++)
        residual[6 * b + k] =
          pred[6 * b + k] * n.target_std[k] + n.target_mean[k];

    const ext = externalAccels(this.state.pos, this.state.quat, this.mass,
      this.inertia, actBody, actPoint ?? [0, 0, 0], actForce ?? [0, 0, 0],
      this.B);
    this.last.ext = ext;
    let pivots = null;
    if (this.groundGuard) {
      this.#freeFlight(parts, senders, receivers);
      pivots = this.#pivotRule(parts, ext, actBody);
      this.#energyRule(residual, ext, actBody, actPoint ?? [0, 0, 0], actForce ?? [0, 0, 0]);
    }
    const prePos = this.state.pos.map((p) => [...p]);
    const preQuat = this.state.quat.map((q) => [...q]);
    stepBodies(this.state, residual, ext, rt.dt, rt.gravity);
    const bs = this.packet.bodies;
    if (pivots) for (const { b, P, start, n } of pivots) {
      // rotation about the fixed contact point: v_com = omega x (com - P),
      // replacing the free integration of the centre of mass
      const w = this.state.angvel[b], c = prePos[b];
      const r = [c[0] - P[0], c[1] - P[1], c[2] - P[2]];
      const v = [w[1] * r[2] - w[2] * r[1], w[2] * r[0] - w[0] * r[2], w[0] * r[1] - w[1] * r[0]];
      this.state.linvel[b] = v;
      this.state.pos[b] = [c[0] + v[0] * rt.dt, c[1] + v[1] * rt.dt, c[2] + v[2] * rt.dt];
      // the swing ending on a new support is an impact: a pencil's end on
      // a table keeps little of its speed (restitution ~0.2), and without
      // this the swing carried through and the pencil bounced 12 degrees
      // back up. "New support" = the centre of mass is now over the support.
      const partsNow = this.particlesWorld();
      const segsNow = bs.map((bb, i) => capsuleWorld(this.state.pos[i], this.state.quat[i], bb.capsule));
      const spNow = supportPoints(partsNow, start, n, segsNow, b, rt.contact_radius, rt.contact_radius);
      const anNow = supportAnalysis(this.state.pos[b], spNow.points);
      if (anNow.n && anNow.dist < 4e-3) {
        for (let q = 0; q < 3; q++) { this.state.linvel[b][q] *= 0.2; this.state.angvel[b][q] *= 0.2; }
        this.pivoting[b] = 0;
      }
    }
    if (this.groundGuard) {
      // capsules first: their push can move a body into the floor, and the
      // floor is the hard constraint (diagnostics caught 1 mm "sinking"
      // episodes from the reverse order). Two passes: a pencil dragged out
      // from under two others is a chain, and one pass left 4-7 mm.
      this.#capsuleGuard(); this.#groundGuard();
      this.#capsuleGuard(); this.#groundGuard();
      this.#settle(prePos, preQuat, actBody);
      // settle restores poses, which undoes the guards' pushes on held
      // bodies: two settled pencils that were nudged into each other stayed
      // 3.8 mm overlapped for two seconds. Separate them after the restore.
      this.#capsuleGuard(); this.#groundGuard();
    }
    this.linHist.shift(); this.linHist.push(this.state.linvel.map((v) => [...v]));
    this.angHist.shift(); this.angHist.push(this.state.angvel.map((v) => [...v]));
    this.quatHist.shift(); this.quatHist.push(this.state.quat.map((q) => [...q]));
    this.stepCount++;
    return this.state;
  }
}

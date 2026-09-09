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
// "sitting still" for the energy rule: slower than this and nothing that
// happens to it can be an impact
const QUIESCENT_V = 0.05, QUIESCENT_W = 1.0;
// A pinch resists rotation: 1/s, so held spin decays with a 0.08 s time
// constant, about what two fingertips on a pencil feel like.
const PINCH_DAMPING = 12;
// A body the guards keep pushing out while it goes nowhere is in a limit
// cycle, not in motion: 0.5 s of that and settle may claim it.
const GUARD_HELD_V = 0.06, GUARD_HELD_W = 1.5, GUARD_HELD_STEPS = 30;
// A body may sleep only once its pose is resolved: within a third of a
// millimetre of the floor or a neighbour.
const SLEEP_GAP = 3e-4;
// Floor stiction: 20 mm/s horizontal is 0.3 mm a step, well under a
// pencil's static friction; the vertical and angular gates keep falling
// and toppling bodies out of it.
const STICTION_V = 0.02, STICTION_VZ = 0.01, STICTION_W = 0.6, STICTION_STEPS = 10;
// The furthest a non-penetration correction may move a body in one step.
// 1.5 mm is 90 mm/s, fast enough to clear an overlap in a few frames and
// slow enough that the correction is never itself a jump through
// something. The guard runs three times a step, so up to 4.5 mm.
const MAX_CORRECTION = 1.5e-3;
// The rest of a correction is delivered as speed: enough to clear an
// overlap in about three steps, never faster than a pencil is pushed.
const SEPARATION_BETA = 0.3, SEPARATION_V_MAX = 0.3;
// TARGET_SPEED_MAX from data generation: the speed a grab was trained at.
const HELD_SPEED_MAX = 0.25;
// how many bounded correction passes a step may take to clear an overlap
const GUARD_PASSES = 12;

export class PhysSim {
  /**
   * backend: { kind: "gpu", net: GpuNet } (custom WebGPU, ~30 ms/step) or
   *          { kind: "ort", ort, session } (ONNX Runtime Web fallback).
   */
  constructor(backend, runtime, packet,
              { groundGuard = true, energyRule = false, smooth = false,
                swept = true } = {}) {
    this.swept = swept;             // off only to measure what it prevents
    this.backend = backend;
    this.rt = runtime;
    this.packet = packet;
    this.groundGuard = groundGuard;   // off in parity tests (Python eval is unguarded)
    // Both measured on the synthetic scorecard and both rejected: the
    // energy rule scored 48.1 and the two-tap residual mean 34.2, against
    // 62.3 without them (stability 0.83 -> 0.42 and 0.21, photo drift 9 mm
    // -> 17 and 30 mm). Kept behind flags because the measurements are
    // worth being able to reproduce.
    this.energyRule = energyRule;
    this.smooth = smooth;
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
    this.prevResidual = null;
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

  /** Mechanical energy (J) of the listed bodies: kinetic + gravitational. */
  #energy(st, which) {
    const g = this.rt.gravity;
    let E = 0;
    for (const b of which) {
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
   * No free energy, for bodies that are sitting still: a quiescent body's
   * residual is scaled down until its mechanical energy stops rising.
   *
   * OFF by default, and kept only because the measurement is worth
   * keeping. It was written for the rearing failure (a resting pencil
   * rotating up to 89 degrees on its own) and it does stop it, but the
   * scorecard says it is the wrong tool: policing the whole scene scored
   * 42.4 against 62.3 without it, and restricting it to quiescent bodies
   * only reached 48.1 (stability 0.83 -> 0.42, photo drift 9 -> 17 mm),
   * because scaling a resting body's contact response biases it downward
   * and the body sinks. The rearing turned out to be the same
   * alternating-residual ratchet as the lift ringing, and the two-tap mean
   * in step() removes it: 30 s of rest on all four photo scenes moves
   * nothing, with this rule off (web/test/rest.mjs).
   */
  #energyRule(residual, ext, actBody) {
    const rt = this.rt, st = this.state;
    const quiet = [];
    for (let b = 0; b < this.B; b++) {
      if (b === actBody) continue;
      if (Math.hypot(...st.linvel[b]) < QUIESCENT_V && Math.hypot(...st.angvel[b]) < QUIESCENT_W)
        quiet.push(b);
    }
    if (!quiet.length) return;
    // a neighbour arriving at speed can legitimately lift this body
    for (let b = 0; b < this.B; b++) {
      if (quiet.includes(b) || b === actBody) continue;
      const fast = Math.hypot(...st.linvel[b]) > QUIESCENT_V;
      if (!fast) continue;
      const segs = this.packet.bodies.map((x, i) =>
        capsuleWorld(st.pos[i], st.quat[i], x.capsule));
      for (let q = quiet.length - 1; q >= 0; q--)
        if (-capsuleClosest(segs[quiet[q]], segs[b]).pen < 3 * rt.contact_radius)
          quiet.splice(q, 1);
      break;
    }
    if (!quiet.length) return;
    const E0 = this.#energy(st, quiet);
    const allow = 5e-7 * quiet.length;      // J/step; 5e-7 ~ 0.5 mm/s of lift
    const trial = (scale) => {
      const copy = { pos: st.pos.map((x) => [...x]), quat: st.quat.map((x) => [...x]),
        linvel: st.linvel.map((x) => [...x]), angvel: st.angvel.map((x) => [...x]) };
      const res = scale === 1 ? residual : Float64Array.from(residual);
      if (scale !== 1) for (const b of quiet)
        for (let k = 0; k < 6; k++) res[6 * b + k] *= scale;
      stepBodies(copy, res, ext, rt.dt, rt.gravity);
      return this.#energy(copy, quiet) - E0;
    };
    const gain = trial(1);
    this.last.guard.energyGain_uJ = gain * 1e6;
    if (gain <= allow) return;
    // bisection on the residual scale (5 rounds is 3% resolution)
    let lo = 0, hi = 1;
    if (trial(0) > allow) { lo = 0; hi = 0; }   // gravity alone exceeds it: drop the residual
    for (let it = 0; it < 5 && hi > lo; it++) {
      const mid = (lo + hi) / 2;
      if (trial(mid) > allow) hi = mid; else lo = mid;
    }
    for (const b of quiet) for (let k = 0; k < 6; k++) residual[6 * b + k] *= lo;
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
   * Contact torque must go to zero as a body separates: full at zero gap,
   * nothing at the contact radius, where the free-flight rule takes over.
   * The model's does not, so the handoff is a cliff and a body lifted out
   * of a pile keeps receiving hundreds of rad/s^2 across gaps of several
   * millimetres. Only the angular part is faded: the linear part holds the
   * pile up, and weakening it makes bodies sink.
   */
  #fadeAngular(parts) {
    const rc = this.rt.contact_radius, bs = this.packet.bodies, res = this.last.residual;
    const segs = bs.map((b, i) => capsuleWorld(this.state.pos[i], this.state.quat[i], b.capsule));
    let k = 0;
    for (let b = 0; b < this.B; b++) {
      const n = this.counts[b], start = k; k += n;
      let gap = Infinity;
      for (let i = start; i < start + n; i++) gap = Math.min(gap, parts[3 * i + 2]);
      for (let j = 0; j < this.B; j++)
        if (j !== b) gap = Math.min(gap, -capsuleClosest(segs[b], segs[j]).pen);
      const fade = Math.max(0, Math.min(1, 1 - gap / rc));
      if (fade < 1) for (let q = 3; q < 6; q++) res[6 * b + q] *= fade;
    }
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
        // capped like the capsule guard: rotating a 150 mm pencil a few
        // degrees plunges an end far below the floor, and lifting the whole
        // body out in one step was a 13 mm teleport that carried it through
        // its neighbours. It comes out over a few steps instead.
        const lift = Math.min(-minz, MAX_CORRECTION);
        this.state.pos[b][2] += lift;
        if (this.last) this.last.guard.ground[b] += lift;
        if (this.state.linvel[b][2] < 0 && this.#lying(b)) this.state.linvel[b][2] = 0;
        // the rest of the way out through velocity, so the body rises
        // over a few steps instead of jumping
        this.state.linvel[b][2] +=
          Math.min((-minz - lift) * SEPARATION_BETA / this.rt.dt, SEPARATION_V_MAX);
      }
    }
  }

  /**
   * Swept collision. The guards below look at where bodies ARE, so they
   * cannot see a body that was on one side of a pencil at the start of a
   * step and the other side at the end: at 60 Hz a pencil moving 0.6 m/s
   * covers 10 mm, more than its own radius, and one yanked out from under
   * a pile or dropped from a height goes straight through its neighbours.
   *
   * Walk the path each body took this step, in slices no longer than half
   * the smallest radius. At the first slice where a pair overlaps, put
   * both bodies back to that moment and cancel the speed at which they
   * were closing: that is the impact, and the ordinary guards then resolve
   * the contact from a pose where it is visible.
   */
  #sweptCollisions(prePos, preQuat) {
    const bs = this.packet.bodies;
    if (this.B < 2) return;
    let maxDisp = 0;
    for (let b = 0; b < this.B; b++) {
      const p = this.state.pos[b], q = prePos[b];
      maxDisp = Math.max(maxDisp, Math.hypot(p[0] - q[0], p[1] - q[1], p[2] - q[2]));
    }
    const minR = Math.min(...bs.map((x) => x.capsule.radius));
    this.sweptInfo = { maxDisp, minR, K: 0, hits: 0 };
    if (maxDisp < 0.5 * minR) return;              // too slow to tunnel
    // slices no longer than a quarter radius, so an impact is caught while
    // it is still shallow enough for the guards to resolve outward
    const K = Math.min(32, Math.ceil(maxDisp / (0.25 * minR)));
    this.sweptInfo.K = K;
    const qa = [0, 0, 0, 0];
    const poseAt = (b, t) => {
      const p0 = prePos[b], p1 = this.state.pos[b];
      const pos = [p0[0] + (p1[0] - p0[0]) * t, p0[1] + (p1[1] - p0[1]) * t,
        p0[2] + (p1[2] - p0[2]) * t];
      // nlerp is enough over one step's rotation
      const a = preQuat[b], c = this.state.quat[b];
      const sign = (a[0] * c[0] + a[1] * c[1] + a[2] * c[2] + a[3] * c[3]) < 0 ? -1 : 1;
      let n = 0;
      for (let k = 0; k < 4; k++) { qa[k] = a[k] + (sign * c[k] - a[k]) * t; n += qa[k] * qa[k]; }
      n = Math.sqrt(n) || 1;
      return { pos, quat: [qa[0] / n, qa[1] / n, qa[2] / n, qa[3] / n] };
    };
    // Baseline: how much each pair already overlapped before the step. A
    // resting contact sits at about zero and is the ordinary guard's job;
    // what matters here is overlap that DEEPENS along the path. Excluding
    // touching pairs outright, as a first version did, switched the check
    // off exactly where it is needed, since a pencil being pulled out from
    // under a pile is touching everything it might tunnel through.
    const startSegs = bs.map((x, b) => capsuleWorld(prePos[b], preQuat[b], x.capsule));
    const pen0 = new Float64Array(this.B * this.B);
    for (let i = 0; i < this.B; i++) for (let j = i + 1; j < this.B; j++)
      pen0[i * this.B + j] = Math.max(0, capsuleClosest(startSegs[i], startSegs[j]).pen);

    for (let k = 1; k <= K; k++) {
      const t = k / K;
      const poses = bs.map((_, b) => poseAt(b, t));
      const segs = bs.map((x, b) => capsuleWorld(poses[b].pos, poses[b].quat, x.capsule));
      const hits = [];
      for (let i = 0; i < this.B; i++) for (let j = i + 1; j < this.B; j++) {
        const c = capsuleClosest(segs[i], segs[j]);
        if (c.pen > pen0[i * this.B + j] + 3e-4) hits.push({ i, j, n: c.n });
      }
      if (!hits.length) continue;
      // Rewind the WHOLE scene to this moment, so what happens next is
      // time-consistent, then take the speed out of every pair that has
      // just met. A five-pencil pile passes the impact along a chain, and
      // resolving only the first pair left the rest to tunnel.
      for (let b = 0; b < this.B; b++) {
        this.state.pos[b] = [...poses[b].pos];
        this.state.quat[b] = [...poses[b].quat];
      }
      for (const { i, j, n } of hits) {
        const vi = this.state.linvel[i], vj = this.state.linvel[j];
        const vrel = (vi[0] - vj[0]) * n[0] + (vi[1] - vj[1]) * n[1] + (vi[2] - vj[2]) * n[2];
        if (vrel >= 0) continue;
        const mi = this.mass[i], mj = this.mass[j];
        const wi = mj / (mi + mj), wj = mi / (mi + mj);
        for (let q = 0; q < 3; q++) {
          vi[q] -= n[q] * vrel * wi; vj[q] += n[q] * vrel * wj;
        }
      }
      if (this.last) this.last.guard.swept = (this.last.guard.swept ?? 0) + hits.length;
      this.sweptInfo.hits = hits.length;
      this.sweptInfo.t = t;
      return;
    }
  }

  /** Deepest capsule overlap in the scene right now (m). */
  #worstOverlap() {
    const bs = this.packet.bodies;
    if (!bs.length || !bs[0].capsule) return 0;
    const segs = bs.map((b, i) => capsuleWorld(this.state.pos[i], this.state.quat[i], b.capsule));
    let w = 0;
    for (let i = 0; i < this.B; i++) for (let j = i + 1; j < this.B; j++)
      w = Math.max(w, capsuleClosest(segs[i], segs[j]).pen);
    return w;
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
      // Separate through VELOCITY, not by teleporting. Moving a body out
      // of an overlap in one step was a jump of up to 14 mm when a pencil
      // was yanked from under a pile, and the jump itself carried it
      // across its neighbour: the pass-through a player sees. The bodies
      // are given a speed that clears the overlap over the next few
      // steps, plus a small positional nudge for numerical stability, so
      // no correction is ever larger than a fraction of a millimetre.
      const push = Math.min(pen, MAX_CORRECTION);
      const sep = Math.min(pen * SEPARATION_BETA / this.rt.dt, SEPARATION_V_MAX);
      const vi2 = this.state.linvel[i], vj2 = this.state.linvel[j];
      for (let k = 0; k < 3; k++) {
        this.state.pos[i][k] += n[k] * push * wi;
        this.state.pos[j][k] -= n[k] * push * wj;
        vi2[k] += n[k] * sep * wi;
        vj2[k] -= n[k] * sep * wj;
      }
      if (this.last) {
        this.last.guard.capsule[i] += push * wi;
        this.last.guard.capsule[j] += push * wj;
        this.last.guard.pairs.push({ i, j, pen });
        // remember which way each body was pushed out, so a grab can stop
        // pressing that way without losing the rest of its pull
        (this.last.guard.normal ??= {})[i] = n;
        this.last.guard.normal[j] = n.map((x) => -x);
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
    const spAll = [], lowestOf = new Float64Array(this.B);
    let k = 0;
    for (let i = 0; i < this.B; i++) {
      const n = this.counts[i], start = k; k += n;
      let lo = Infinity;
      for (let q = start; q < start + n; q++) lo = Math.min(lo, parts[3 * q + 2]);
      lowestOf[i] = lo;
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
      // Guard-held bodies. A body the guards have to push out on step
      // after step, while it is going nowhere, is not in motion: it is in
      // a limit cycle, riding the positional correction. One rode it
      // 102 mm across the table in 77 s (blind test round 5, IMG_8626;
      // the CPU backend does not reproduce it, the cycle needs the GPU
      // backend's slightly different numbers to stay just above the
      // settle threshold). Count those steps and let settle claim the
      // body. A body that is falling, being pulled or genuinely sliding
      // is not slow, so it never qualifies.
      this.guardHeld ??= new Int32Array(this.B);
      const pushed = this.last &&
        (this.last.guard.ground[b] > 1e-5 || this.last.guard.capsule[b] > 1e-5);
      this.guardHeld[b] = (pushed && b !== actBody &&
        Math.hypot(...v) < GUARD_HELD_V && Math.hypot(...w) < GUARD_HELD_W)
        ? this.guardHeld[b] + 1 : 0;

      // Floor stiction. A pencil lying on a table does not slide at
      // 1.9 mm/s: friction holds it, and the learned model does not. One
      // walked 50 mm in 27 s under the guards' corrections, never slow
      // enough to sleep because it was, precisely, moving. Gated on being
      // on the floor, barely moving in every direction, and untouched, so
      // a falling body (large vertical speed) and a pulled one are exempt.
      this.stickCount ??= new Int32Array(this.B);
      // touching anything, not just the floor: the pencil that walked was
      // sliding across its neighbours 6 mm up, so a floor-only gate left
      // it exactly as it was
      const onSomething = Math.min(gap[b], lowestOf[b]) < 1e-3;
      const stickable = b !== actBody && onSomething &&
        Math.abs(v[2]) < STICTION_VZ && Math.hypot(v[0], v[1]) < STICTION_V &&
        Math.hypot(...w) < STICTION_W;
      // ten consecutive slow steps, so a body that has just lost its
      // support and is starting to fall is never caught by stiction
      this.stickCount[b] = stickable ? this.stickCount[b] + 1 : 0;
      if (this.stickCount[b] >= STICTION_STEPS) { v[0] = 0; v[1] = 0; }
      // Slow motion of a supported body dies quickly in reality (friction
      // decelerates a sliding pencil at ~5 m/s^2, so 8 cm/s is gone in
      // 16 ms); the model instead rings for ~10 steps after every landing
      // (diagnostics: "jitter"). Damp the tail.
      if (b !== actBody && supported[b] && !this.pivoting?.[b] &&
          Math.hypot(...v) < 0.05 && Math.hypot(...w) < 2)
        for (let q = 0; q < 3; q++) { v[q] *= 0.5; w[q] *= 0.5; }
      // the model's contact response rings at 1-3 cm/s amplitude for a
      // second after a landing; a body that slow on a support is at rest
      // Sleep only once the body is really where it belongs. Settling on
      // "supported" alone froze scenes in their loaded pose: a blind
      // tester measured a two-pencil cluster asleep 2.3 and 14.8 mm in the
      // air, with other barrels 0.5-3.8 mm inside the tabletop, and proved
      // the solver could fix it (one 5 s drag brought every body within
      // 0.9 mm of the table). The pose has to be resolved first: touching
      // the floor or a neighbour within a third of a millimetre.
      const touching = Math.min(gap[b], lowestOf[b]) < SLEEP_GAP;
      const slow = (b !== actBody && supported[b] && touching && !standing &&
        !this.pivoting?.[b] &&
        Math.hypot(...v) < 0.03 && Math.hypot(...w) < 0.6) ||
        this.guardHeld[b] >= GUARD_HELD_STEPS;
      this.restCount[b] = slow ? this.restCount[b] + 1 : 0;
      // every settled step, not only the first: a body that settles during
      // the load pre-roll and only later ends up with a gap under it would
      // otherwise keep it forever (a blind tester measured resting gaps of
      // 1.8-3.0 mm and said, correctly, that nothing was touching anything)
      if (this.restCount[b] >= SETTLE_STEPS) {
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

  /**
   * actPinch: the action is a held grab rather than a push. A single-point
   * spring lets a pencil swing freely about the grab point, and with the
   * grab at the centre of mass nothing at all resists rotation, so any spin
   * picked up while separating from the pile persists and the pencil ends up
   * hanging at 50-78 degrees. Fingers do not do that: a pinch is two contact
   * patches and resists rotation. This adds that resistance as an angular
   * damper on the held body, in the interaction model where it belongs, not
   * in the physics.
   */
  async step(actBody = -1, actPoint = null, actForce = null, actPinch = false) {
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
    if (actPinch && actBody >= 0) {
      const w = this.state.angvel[actBody];
      for (let q = 0; q < 3; q++) ext.ang[3 * actBody + q] -= PINCH_DAMPING * w[q];
    }
    this.last.ext = ext;
    // Two-tap mean of the residual: cancels the step-alternating ringing
    // the model produces on reconstructed piles. Off by default (it costs
    // 28 composite points on the synthetic scorecard); the angular contact
    // fade below addresses the same failure without that cost.
    if (this.smooth) {
      const prev = this.prevResidual;
      if (prev && prev.length === residual.length) {
        for (let i = 0; i < residual.length; i++) {
          const cur = residual[i];
          residual[i] = 0.5 * (cur + prev[i]);
          prev[i] = cur;
        }
      } else this.prevResidual = Float64Array.from(residual);
    }

    if (this.groundGuard) this.#fadeAngular(parts);

    let pivots = null;
    if (this.groundGuard) {
      this.#freeFlight(parts, senders, receivers);
      pivots = this.#pivotRule(parts, ext, actBody);
      if (this.energyRule) this.#energyRule(residual, ext, actBody);
    }
    const prePos = this.state.pos.map((p) => [...p]);
    const preQuat = this.state.quat.map((q) => [...q]);
    stepBodies(this.state, residual, ext, rt.dt, rt.gravity);
    // Hold a grabbed body to the speed a grab was generated at. Above it
    // the learned contact response has never seen the situation and stops
    // resisting, so a pencil yanked from under a pile drove 3-5 mm into
    // its neighbours faster than the guards could push it out and the
    // overlap grew every step: what "it phases through" looks like. A hand
    // moving a pencil out of a pile is not faster than this anyway. The
    // clamp has to be here, after integration; applying it beforehand, as
    // a first attempt did, clamps last step's velocity and does nothing.
    if (actPinch && actBody >= 0) {
      const v = this.state.linvel[actBody];
      const sp = Math.hypot(v[0], v[1], v[2]);
      if (sp > HELD_SPEED_MAX) for (let q = 0; q < 3; q++) v[q] *= HELD_SPEED_MAX / sp;
    }

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
      // before anything else, catch a body that crossed another during the
      // step rather than ending up overlapping it
      if (this.swept) this.#sweptCollisions(prePos, preQuat);
      // capsules first: their push can move a body into the floor, and the
      // floor is the hard constraint (diagnostics caught 1 mm "sinking"
      // episodes from the reverse order). Two passes: a pencil dragged out
      // from under two others is a chain, and one pass left 4-7 mm.
      // Iterate to convergence rather than a fixed number of passes, each
      // pass still bounded by MAX_CORRECTION so no single one is a jump.
      // With three fixed passes the grab could drive a pencil into its
      // neighbour faster than the guard pushed it out and the overlap grew
      // every step, which is what a player sees as phasing through.
      for (let it = 0; it < GUARD_PASSES; it++) {
        this.#capsuleGuard(); this.#groundGuard();
        if (this.#worstOverlap() < 3e-4) break;
      }
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

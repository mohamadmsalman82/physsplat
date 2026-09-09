/**
 * Diagnostics: a transparent, machine-readable view of the simulation, so
 * a tester (human or agent) reads what the pencils are doing instead of
 * inferring it from pixels.
 *
 *   physsplat.diag.snapshot()        current state + derived quantities
 *   physsplat.diag.history(n)        last n per-step frames
 *   physsplat.diag.track(body, n)    time series for one body
 *   physsplat.diag.check()           anomalies active right now
 *   physsplat.diag.events(n)         event log (actions, anomalies, scene)
 *   physsplat.diag.summary(n)        aggregates over the last n steps
 *   physsplat.diag.export()          everything as a JSON string
 *   physsplat.probe.*                scripted experiments with measured reports
 *
 * Units are SI unless a field name ends in _mm, _deg, _ms.
 * All quantities come from the physics state, the capsule proxies, and the
 * per-step record the simulator keeps (model residual, external
 * accelerations, guard corrections). Nothing here is read from the screen.
 */
import { capsuleClosest, capsuleWorld, quatToMatrix, supportAnalysis, supportPoints } from "./physics.js";

const G = 9.81;
// The model resolves contact at particle level with a 6 mm contact radius
// and settles stacked bodies with a 1-3 mm gap between capsule surfaces,
// so "in contact" for the capsule proxies is a 3 mm tolerance. The resting
// gap itself is reported (summary.minGap_mm) because it is a model-quality
// number: PyBullet's is zero.
const CONTACT_GAP = 3e-3;
const NEAR_GAP = 5e-3;
const REST_V = 0.01, REST_W = 0.3;   // resting thresholds (m/s, rad/s)

const hyp = (v) => Math.hypot(v[0], v[1], v[2]);
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const r3 = (x, n = 4) => (Array.isArray(x) ? x.map((v) => r3(v, n)) : +(+x).toFixed(n));

/** Angle (deg) between two unit vectors, sign-insensitive (a pencil has no
 * front end as far as its axis is concerned). */
function axisAngleDeg(a, b) {
  const d = Math.abs(a[0] * b[0] + a[1] * b[1] + a[2] * b[2]);
  return Math.acos(Math.min(1, d)) * 180 / Math.PI;
}

export class Diagnostics {
  constructor(sim, { capacity = 3600 } = {}) {
    this.sim = sim;
    this.capacity = capacity;   // 3600 steps = 60 s of sim time
    this.reset("init");
  }

  reset(reason = "reset") {
    this.frames = new Array(this.capacity);
    this.head = 0; this.count = 0;
    this.eventsLog = [];
    this.waiters = [];
    this.restStart = [];        // per body: {pos, step} when it came to rest
    this.episodes = new Map();  // active anomaly episodes
    this.actionFreeSteps = 0;
    this.lastKE = null;
    this.event("scene", { reason, name: this.sim.packet?.name, bodies: this.sim.B,
      load_correction: this.sim.loadCorrection ?? null });
  }

  // ------------------------------------------------------------ recording

  event(type, data = {}) {
    const e = { step: this.sim.stepCount, t: r3(this.sim.stepCount * this.sim.rt.dt, 3),
      wall: new Date().toISOString(), type, ...data };
    this.eventsLog.push(e);
    if (this.eventsLog.length > 2000) this.eventsLog.shift();
    return e;
  }

  frame(i) {            // i = 0 newest, 1 previous, ...
    if (i >= this.count) return null;
    return this.frames[(this.head - 1 - i + this.capacity) % this.capacity];
  }

  /** Called after every physics step. `action` is what the loop applied. */
  record(action = null) {
    const sim = this.sim, st = sim.state, bs = sim.packet.bodies, B = sim.B;
    const step = sim.stepCount, t = step * sim.rt.dt;
    const caps = bs.map((b, i) => capsuleWorld(st.pos[i], st.quat[i], b.capsule));
    const last = sim.last;

    // pairwise capsule geometry
    const pairs = [];
    for (let i = 0; i < B; i++) for (let j = i + 1; j < B; j++) {
      const c = capsuleClosest(caps[i], caps[j]);
      pairs.push({ i, j, gap: -c.pen, n: c.n,
        point: [(c.ca[0] + c.cb[0]) / 2, (c.ca[1] + c.cb[1]) / 2, (c.ca[2] + c.cb[2]) / 2] });
    }
    const pairsOf = (i) => pairs.filter((p) => p.i === i || p.j === i)
      .map((p) => ({ other: p.i === i ? p.j : p.i, gap: p.gap, point: p.point }));

    const bodies = [];
    // world particles, computed here so the recorder only needs state,
    // offsets and capsules (the unit test drives it with a mock simulator)
    const nAll = sim.offsets.reduce((a, o) => a + o.length, 0);
    const partsAll = new Float64Array(nAll * 3);
    {
      let q = 0;
      for (let i = 0; i < B; i++) {
        const R = quatToMatrix(st.quat[i]), p = st.pos[i];
        for (const o of sim.offsets[i]) {
          partsAll[q++] = R[0] * o[0] + R[1] * o[1] + R[2] * o[2] + p[0];
          partsAll[q++] = R[3] * o[0] + R[4] * o[1] + R[5] * o[2] + p[1];
          partsAll[q++] = R[6] * o[0] + R[7] * o[1] + R[8] * o[2] + p[2];
        }
      }
    }
    let pStart = 0;
    for (let i = 0; i < B; i++) {
      const cap = caps[i], v = st.linvel[i], w = st.angvel[i], p = st.pos[i];
      const speed = hyp(v), angSpeed = hyp(w);
      // lowest / highest surface points from the physics particles (what the
      // model and the ground guard see), not the fitted capsule
      const R = quatToMatrix(st.quat[i]);
      const nP = sim.offsets[i].length, start = pStart; pStart += nP;
      let lowest = Infinity, highest = -Infinity;
      for (let q = start; q < start + nP; q++) {
        const z = partsAll[3 * q + 2];
        if (z < lowest) lowest = z;
        if (z > highest) highest = z;
      }
      // static equilibrium: is the centre of mass over the support?
      const rc = sim.rt.contact_radius ?? 6e-3;
      const sp = supportPoints(partsAll, start, nP, caps, i, rc, rc);
      const an = supportAnalysis(p, sp.points);
      const elevation = Math.asin(Math.min(1, Math.abs(cap.a[2]))) * 180 / Math.PI;
      const mine = pairsOf(i);
      const contacts = mine.filter((c) => c.gap < CONTACT_GAP);
      const near = mine.filter((c) => c.gap < NEAR_GAP);
      const penetration = Math.max(0, ...mine.map((c) => -c.gap));
      const supportedBy = contacts.filter((c) => c.point[2] < p[2] - 1e-3).map((c) => c.other);
      const supports = contacts.filter((c) => c.point[2] > p[2] + 1e-3).map((c) => c.other);
      // rotational KE in the body frame: w_b = R^T w
      const wb = [R[0] * w[0] + R[3] * w[1] + R[6] * w[2],
        R[1] * w[0] + R[4] * w[1] + R[7] * w[2],
        R[2] * w[0] + R[5] * w[1] + R[8] * w[2]];
      const I = sim.inertia[i], m = sim.mass[i];
      const KE = 0.5 * m * speed * speed +
        0.5 * (I[0] * wb[0] ** 2 + I[1] * wb[1] ** 2 + I[2] * wb[2] ** 2);
      // "resting" is the simulator's own verdict (settle held the body
      // still) when the guard record exists; the speed test alone counted
      // slow settling motion as rest and reported it as creep
      const resting = last ? !!last.guard.settled[i] : (speed < REST_V && angSpeed < REST_W);
      const snapped = last && last.guard.ground[i] < 0;   // settle closed a gap on purpose
      if (resting && !snapped) {
        if (!this.restStart[i]) this.restStart[i] = { pos: [...p], axis: [...cap.a], step };
      } else this.restStart[i] = null;
      const rs = this.restStart[i];
      // slow by speed alone (a standing pencil never settles, so the
      // tip-balance detector must not depend on the settle verdict)
      this.slowSteps ??= [];
      const slowNow = speed < REST_V && angSpeed < REST_W;
      this.slowSteps[i] = slowNow ? (this.slowSteps[i] ?? 0) + 1 : 0;
      const minGap = Math.min(lowest, ...mine.map((c) => c.gap));
      const b = {
        id: i, mass: m,
        pos: [...p], quat: [...st.quat[i]], linvel: [...v], angvel: [...w],
        axis: [...cap.a], speed, angSpeed,
        height: p[2], lowest, highest, elevation_deg: elevation,
        minGap_mm: minGap * 1e3,          // to the floor or the nearest capsule
        groundContact: lowest < CONTACT_GAP,
        groundPen_mm: Math.max(0, -lowest) * 1e3,
        contacts: contacts.map((c) => ({ other: c.other, gap_mm: c.gap * 1e3 })),
        near: near.map((c) => c.other),
        penetration_mm: penetration * 1e3,
        supportedBy, supports,
        unsupported: an.n === 0 && lowest >= CONTACT_GAP,   // nothing below it
        support: { n: an.n, floor: sp.floor, capsule: sp.capsule, balanced: an.balanced,
          dist_mm: an.dist === Infinity ? null : an.dist * 1e3, spread_mm: an.spread * 1e3 },
        resting, restingSteps: rs ? step - rs.step : 0,
        slowSteps: this.slowSteps[i],
        driftSinceRest_mm: rs ? hyp(sub(p, rs.pos)) * 1e3 : 0,
        rotSinceRest_deg: rs ? axisAngleDeg(cap.a, rs.axis) : 0,
        KE, PE: m * G * p[2],
        model: last ? {
          lin: [last.residual[6 * i], last.residual[6 * i + 1], last.residual[6 * i + 2]],
          ang: [last.residual[6 * i + 3], last.residual[6 * i + 4], last.residual[6 * i + 5]],
        } : null,
        ext: last ? {
          lin: [last.ext.lin[3 * i], last.ext.lin[3 * i + 1], last.ext.lin[3 * i + 2]],
          ang: [last.ext.ang[3 * i], last.ext.ang[3 * i + 1], last.ext.ang[3 * i + 2]],
        } : null,
        guard: last ? {
          ground_mm: last.guard.ground[i] * 1e3,
          capsule_mm: last.guard.capsule[i] * 1e3,
          settled: !!last.guard.settled[i],
          freeFlight: !!last.guard.freeFlight[i],
          pivot: !!last.guard.pivot?.[i],
          energyScale: last.guard.energyScale ?? 1,
        } : null,
      };
      bodies.push(b);
    }
    const totalKE = bodies.reduce((a, b) => a + b.KE, 0);
    this.actionFreeSteps = action ? 0 : this.actionFreeSteps + 1;
    this.lastActionStep ??= [];
    if (action) this.lastActionStep[action.body] = step;
    for (const b of bodies) b.stepsSinceAction = step - (this.lastActionStep[b.id] ?? -1e9);
    const frame = {
      step, t, action: action ? { ...action } : null,
      actionFreeSteps: this.actionFreeSteps,
      bodies, pairs: pairs.map((p) => ({ i: p.i, j: p.j, gap_mm: p.gap * 1e3 })),
      totals: {
        KE: totalKE, PE: bodies.reduce((a, b) => a + b.PE, 0),
        maxPenetration_mm: Math.max(0, ...bodies.map((b) => b.penetration_mm)),
        maxGroundPen_mm: Math.max(0, ...bodies.map((b) => b.groundPen_mm)),
        maxSpeed: Math.max(0, ...bodies.map((b) => b.speed)),
        resting: bodies.filter((b) => b.resting).length,
        guardPairs: last ? last.guard.pairs.length : 0,
        energyScale: last ? (last.guard.energyScale ?? 1) : 1,
        energyGain_uJ: last ? (last.guard.energyGain_uJ ?? 0) : 0,
      },
      timing: sim.timing ? { ...sim.timing } : null,
      anomalies: [],
    };
    this.frames[this.head] = frame;
    this.head = (this.head + 1) % this.capacity;
    this.count = Math.min(this.count + 1, this.capacity);
    this.#detect(frame);
    this.lastKE = totalKE;
    // wake probes waiting on steps
    const w = this.waiters; this.waiters = [];
    for (const x of w) { if (--x.left <= 0) x.resolve(frame); else this.waiters.push(x); }
    return frame;
  }

  waitSteps(n = 1) {
    return new Promise((resolve) => this.waiters.push({ left: n, resolve }));
  }

  // ------------------------------------------------------------ detectors

  /** Anomaly detectors run on every frame. Each returns null or
   * {type, key, severity, value, message}. An episode starts when a
   * detector first fires and ends when it stops; both are logged. */
  #detect(frame) {
    const found = [];
    const add = (type, key, severity, value, message, extra = {}) =>
      found.push({ type, key: `${type}:${key}`, severity, value, message, ...extra });
    for (const b of frame.bodies) {
      const id = b.id;
      if (b.speed > 3 || b.angSpeed > 100)
        add("explosion", id, "error", b.speed, `body ${id} speed ${b.speed.toFixed(2)} m/s, ${b.angSpeed.toFixed(0)} rad/s`, { body: id });
      if (b.groundPen_mm > 1)
        add("sinking", id, b.groundPen_mm > 3 ? "error" : "warn", b.groundPen_mm,
          `body ${id} is ${b.groundPen_mm.toFixed(1)} mm below the floor`, { body: id });
      if (b.unsupported && b.lowest > 6e-3 && b.restingSteps >= 20 && b.stepsSinceAction > 20)
        add("floating", id, "error", b.lowest * 1e3,
          `body ${id} rests ${(b.lowest * 1e3).toFixed(1)} mm above the floor with no contact`, { body: id });
      if (b.elevation_deg > 60 && b.lowest < 3e-3 && b.slowSteps >= 30)
        add("tip_balance", id, "error", b.elevation_deg,
          `body ${id} balanced on its end at ${b.elevation_deg.toFixed(0)} deg for ${b.slowSteps} steps`, { body: id });
      if (b.elevation_deg > 45 && b.stepsSinceAction > 30 && b.rotSinceRest_deg === 0 && b.angSpeed > 0.5 && b.lowest < 3e-3)
        add("rearing", id, "error", b.elevation_deg,
          `body ${id} is rotating upward on its own (${b.elevation_deg.toFixed(0)} deg, ${b.angSpeed.toFixed(1)} rad/s) with no action`, { body: id });
      else if (b.elevation_deg > 4 && b.lowest < 3e-3 && !b.contacts.length && b.restingSteps >= 30)
        add("tilt_hold", id, "error", b.elevation_deg,
          `body ${id} rests tilted ${b.elevation_deg.toFixed(1)} deg with one end on the floor and nothing under the other`, { body: id });
      if (b.support.n && !b.support.balanced && b.restingSteps >= 30 && b.stepsSinceAction > 30)
        add("unbalanced_rest", id, "error", b.support.dist_mm,
          `body ${id} rests with its centre of mass ${b.support.dist_mm.toFixed(1)} mm outside its support (${b.support.n} contact points, spread ${b.support.spread_mm.toFixed(0)} mm)`, { body: id });
      if (b.restingSteps >= 60 && b.driftSinceRest_mm > 2)
        add("creep", id, "warn", b.driftSinceRest_mm,
          `body ${id} drifted ${b.driftSinceRest_mm.toFixed(1)} mm while classified at rest`, { body: id });
      if (b.restingSteps >= 60 && b.rotSinceRest_deg > 3)
        add("creep_rot", id, "warn", b.rotSinceRest_deg,
          `body ${id} rotated ${b.rotSinceRest_deg.toFixed(1)} deg while at rest`, { body: id });
    }
    for (const p of frame.pairs) {
      if (p.gap_mm < -2)
        add("penetration", `${p.i}-${p.j}`, p.gap_mm < -5 ? "error" : "warn", -p.gap_mm,
          `bodies ${p.i} and ${p.j} overlap by ${(-p.gap_mm).toFixed(1)} mm`, { bodies: [p.i, p.j] });
    }
    // jitter: vertical velocity flipping sign at rest-scale amplitudes
    if (this.count >= 30) {
      for (let i = 0; i < frame.bodies.length; i++) {
        let flips = 0, prev = 0;
        for (let k = 29; k >= 0; k--) {
          const vz = this.frame(k).bodies[i].linvel[2];
          // 15 mm/s is 0.25 mm per step: the smallest tremble a viewer can
          // see; the model's residual noise alone flips sign at 5 mm/s
          if (Math.abs(vz) < 15e-3) continue;
          if (prev && Math.sign(vz) !== prev) flips++;
          prev = Math.sign(vz);
        }
        if (flips > 8) add("jitter", i, "warn", flips,
          `body ${i} vertical velocity flipped sign ${flips} times in 30 steps`, { body: i });
      }
    }
    // spontaneous motion: kinetic energy rising long after the last action
    if (frame.actionFreeSteps > 90 && this.count >= 11) {
      const past = this.frame(10).totals.KE;
      if (frame.totals.KE > past + 2e-6 && frame.totals.KE > 5e-6)
        add("spontaneous_motion", "scene", "warn", frame.totals.KE,
          `kinetic energy rose from ${past.toExponential(2)} to ${frame.totals.KE.toExponential(2)} J with no action for ${frame.actionFreeSteps} steps`);
    }
    frame.anomalies = found;
    // episodes
    const seen = new Set();
    for (const a of found) {
      seen.add(a.key);
      const ep = this.episodes.get(a.key);
      if (!ep) {
        this.episodes.set(a.key, { ...a, startStep: frame.step, peak: a.value });
        this.event("anomaly_start", { key: a.key, severity: a.severity, message: a.message });
      } else if (a.value > ep.peak) { ep.peak = a.value; ep.message = a.message; }
    }
    for (const [key, ep] of [...this.episodes]) {
      if (!seen.has(key)) {
        this.episodes.delete(key);
        this.event("anomaly_end", { key, steps: frame.step - ep.startStep, peak: ep.peak });
      }
    }
  }

  // ------------------------------------------------------------ queries

  /** Current frame with rounded numbers (safe to print). */
  snapshot() {
    const f = this.frame(0);
    if (!f) return null;
    return {
      step: f.step, t: r3(f.t, 3), scene: this.sim.packet?.name,
      action: f.action ? { body: f.action.body, force_N: r3(hyp(f.action.force), 4) } : null,
      bodies: f.bodies.map((b) => ({
        id: b.id, mass_g: r3(b.mass * 1e3, 1),
        pos: r3(b.pos), axis: r3(b.axis, 3), speed: r3(b.speed), angSpeed: r3(b.angSpeed, 2),
        height_mm: r3(b.height * 1e3, 1), lowest_mm: r3(b.lowest * 1e3, 1),
        elevation_deg: r3(b.elevation_deg, 1), minGap_mm: r3(b.minGap_mm, 1),
        groundContact: b.groundContact, contacts: b.contacts.map((c) => ({ other: c.other, gap_mm: r3(c.gap_mm, 1) })),
        supportedBy: b.supportedBy, supports: b.supports, unsupported: b.unsupported,
        support: { n: b.support.n, floor: b.support.floor, capsule: b.support.capsule,
          balanced: b.support.balanced, dist_mm: b.support.dist_mm == null ? null : r3(b.support.dist_mm, 1),
          spread_mm: r3(b.support.spread_mm, 0) },
        penetration_mm: r3(b.penetration_mm, 1), groundPen_mm: r3(b.groundPen_mm, 1),
        resting: b.resting, restingSteps: b.restingSteps,
        driftSinceRest_mm: r3(b.driftSinceRest_mm, 1), rotSinceRest_deg: r3(b.rotSinceRest_deg, 1),
        KE_uJ: r3(b.KE * 1e6, 2),
        model_accel: b.model ? { lin: r3(b.model.lin, 3), ang: r3(b.model.ang, 2) } : null,
        guard: b.guard ? { ground_mm: r3(b.guard.ground_mm, 2), capsule_mm: r3(b.guard.capsule_mm, 2),
          settled: b.guard.settled, freeFlight: b.guard.freeFlight, pivot: b.guard.pivot } : null,
      })),
      pairs: f.pairs.map((p) => ({ ...p, gap_mm: r3(p.gap_mm, 1) })),
      totals: { KE_uJ: r3(f.totals.KE * 1e6, 2), maxPenetration_mm: r3(f.totals.maxPenetration_mm, 1),
        maxGroundPen_mm: r3(f.totals.maxGroundPen_mm, 1), maxSpeed: r3(f.totals.maxSpeed),
        resting: f.totals.resting, guardPairs: f.totals.guardPairs,
        energyScale: r3(f.totals.energyScale, 2), energyGain_uJ: r3(f.totals.energyGain_uJ, 2) },
      timing_ms: f.timing ? { features: r3(f.timing.features_ms, 1), graph: r3(f.timing.graph_ms, 1),
        net: r3(f.timing.net_ms, 1), N: f.timing.N, E: f.timing.E, backend: f.timing.backend } : null,
      anomalies: this.check(),
    };
  }

  /** Last n frames, newest last, compact. `fields` limits body fields. */
  history(n = 60, fields = null) {
    const out = [];
    for (let k = Math.min(n, this.count) - 1; k >= 0; k--) {
      const f = this.frame(k);
      out.push({
        step: f.step, t: r3(f.t, 3), action: f.action ? f.action.body : -1,
        bodies: f.bodies.map((b) => {
          const full = { pos: r3(b.pos), quat: r3(b.quat, 5), linvel: r3(b.linvel), angvel: r3(b.angvel, 3),
            speed: r3(b.speed), lowest_mm: r3(b.lowest * 1e3, 1), elevation_deg: r3(b.elevation_deg, 1),
            penetration_mm: r3(b.penetration_mm, 1), contacts: b.contacts.map((c) => c.other),
            resting: b.resting, guard_mm: b.guard ? r3(b.guard.ground_mm + b.guard.capsule_mm, 2) : 0 };
          if (!fields) return full;
          const sel = {};
          for (const key of fields) if (key in full) sel[key] = full[key];
          return sel;
        }),
        maxPenetration_mm: r3(f.totals.maxPenetration_mm, 1),
        anomalies: f.anomalies.map((a) => a.key),
      });
    }
    return out;
  }

  /** Time series for one body over the last n steps (arrays, newest last). */
  track(body, n = 120) {
    const s = { step: [], t: [], x: [], y: [], z: [], lowest_mm: [], speed: [], vz: [],
      angSpeed: [], elevation_deg: [], penetration_mm: [], contacts: [], resting: [],
      model_lin_z: [], guard_mm: [], free: [], settled: [] };
    for (let k = Math.min(n, this.count) - 1; k >= 0; k--) {
      const f = this.frame(k), b = f.bodies[body];
      if (!b) continue;
      s.step.push(f.step); s.t.push(r3(f.t, 3));
      s.x.push(r3(b.pos[0])); s.y.push(r3(b.pos[1])); s.z.push(r3(b.pos[2]));
      s.lowest_mm.push(r3(b.lowest * 1e3, 1)); s.speed.push(r3(b.speed)); s.vz.push(r3(b.linvel[2]));
      s.angSpeed.push(r3(b.angSpeed, 2)); s.elevation_deg.push(r3(b.elevation_deg, 1));
      s.penetration_mm.push(r3(b.penetration_mm, 1)); s.contacts.push(b.contacts.length);
      s.resting.push(b.resting ? 1 : 0);
      s.model_lin_z.push(b.model ? r3(b.model.lin[2], 3) : 0);
      s.guard_mm.push(b.guard ? r3(b.guard.ground_mm + b.guard.capsule_mm, 2) : 0);
      s.free.push(b.guard?.freeFlight ? 1 : 0);
      s.settled.push(b.guard?.settled ? 1 : 0);
      (s.pivot ??= []).push(b.guard?.pivot ? 1 : 0);
    }
    return s;
  }

  /** Anomalies active right now, with how long they have lasted. */
  check() {
    const step = this.sim.stepCount;
    return [...this.episodes.values()].map((e) => ({
      key: e.key, type: e.type, severity: e.severity, body: e.body ?? e.bodies ?? null,
      steps: step - e.startStep, peak: r3(e.peak, 3), message: e.message }));
  }

  events(n = 50, type = null) {
    const list = type ? this.eventsLog.filter((e) => e.type === type) : this.eventsLog;
    return list.slice(-n);
  }

  /** Aggregates over the last n steps. */
  summary(n = 120) {
    n = Math.min(n, this.count);
    if (!n) return null;
    const first = this.frame(n - 1), lastF = this.frame(0);
    const bodies = first.bodies.map((_, i) => {
      let maxSpeed = 0, maxPen = 0, minLowest = Infinity, ground = 0, rest = 0, guard = 0, maxAng = 0;
      for (let k = 0; k < n; k++) {
        const b = this.frame(k).bodies[i];
        maxSpeed = Math.max(maxSpeed, b.speed); maxAng = Math.max(maxAng, b.angSpeed);
        maxPen = Math.max(maxPen, b.penetration_mm); minLowest = Math.min(minLowest, b.lowest);
        if (b.groundContact) ground++; if (b.resting) rest++;
        if (b.guard) guard += b.guard.ground_mm + b.guard.capsule_mm;
      }
      const a = first.bodies[i], z = lastF.bodies[i];
      return { id: i, displacement_mm: r3(hyp(sub(z.pos, a.pos)) * 1e3, 1),
        rotation_deg: r3(axisAngleDeg(z.axis, a.axis), 1),
        dz_mm: r3((z.pos[2] - a.pos[2]) * 1e3, 1),
        maxSpeed: r3(maxSpeed), maxAngSpeed: r3(maxAng, 2), maxPenetration_mm: r3(maxPen, 1),
        minLowest_mm: r3(minLowest * 1e3, 1), groundContactFrac: r3(ground / n, 2),
        restingFrac: r3(rest / n, 2), guardTotal_mm: r3(guard, 2),
        final: { lowest_mm: r3(z.lowest * 1e3, 1), elevation_deg: r3(z.elevation_deg, 1),
          minGap_mm: r3(z.minGap_mm, 1),
          contacts: z.contacts.map((c) => c.other), unsupported: z.unsupported, resting: z.resting } };
    });
    const counts = {}, episodes = [];
    for (const e of this.eventsLog) if (e.type === "anomaly_start" && e.step >= first.step) {
      counts[e.key.split(":")[0]] = (counts[e.key.split(":")[0]] ?? 0) + 1;
      const end = this.eventsLog.find((x) => x.type === "anomaly_end" && x.key === e.key && x.step > e.step);
      const act = this.episodes.get(e.key);
      episodes.push({ key: e.key, start_step: e.step, severity: e.severity,
        steps: end ? end.steps : (act ? this.sim.stepCount - act.startStep : null),
        peak: r3(end ? end.peak : (act ? act.peak : 0), 2), ongoing: !end });
    }
    let actionSteps = 0, tSum = 0, tN = 0;
    for (let k = 0; k < n; k++) {
      const f = this.frame(k);
      if (f.action) actionSteps++;
      if (f.timing) { tSum += f.timing.features_ms + f.timing.graph_ms + f.timing.net_ms; tN++; }
    }
    return { steps: n, from_step: first.step, to_step: lastF.step, actionSteps,
      meanStep_ms: tN ? r3(tSum / tN, 1) : null, bodies, anomalyStarts: counts,
      episodes, active: this.check() };
  }

  export({ last = this.count } = {}) {
    const bs = this.sim.packet.bodies;
    return JSON.stringify({
      scene: this.sim.packet.name, exported: new Date().toISOString(),
      runtime: { dt: this.sim.rt.dt, gravity: this.sim.rt.gravity, step: this.sim.rt.step },
      bodies: bs.map((b, i) => ({ id: i, mass: b.mass, inertia: b.inertia, capsule: b.capsule,
        init_pos: b.pos, init_quat: b.quat })),
      load_correction: this.sim.loadCorrection ?? null,
      events: this.eventsLog, summary: this.summary(last), frames: this.history(last),
    });
  }
}

// ================================================================== probes
/**
 * Scripted experiments. Each one drives the same grab/poke code paths the
 * mouse uses, samples the diagnostics every step, and returns a report with
 * numbers instead of impressions. All are async and take real time (the
 * physics runs at its own rate); `reset` (default true) restarts the scene
 * first so results are reproducible.
 */
export class Probes {
  constructor(dbg, sim, diag) {
    this.dbg = dbg; this.sim = sim; this.diag = diag;
  }

  #cap(body) { return this.sim.packet.bodies[body].capsule; }

  /** Body-frame grab point: "center", "end" (+axis), "tip" (-axis),
   * a number in [-1, 1] (fraction along the axis), or [x, y, z]. */
  #local(body, where) {
    const cap = this.#cap(body);
    const along = (f) => cap.axis.map((a) => a * f * cap.half * 0.9);
    if (Array.isArray(where)) return where;
    if (where === "center") return [0, 0, 0];
    if (where === "end") return along(1);
    if (where === "tip") return along(-1);
    if (typeof where === "number") return along(where);
    return [0, 0, 0];
  }

  #world(body, local) {
    const R = quatToMatrix(this.sim.state.quat[body]), p = this.sim.state.pos[body];
    return [R[0] * local[0] + R[1] * local[1] + R[2] * local[2] + p[0],
      R[3] * local[0] + R[4] * local[1] + R[5] * local[2] + p[1],
      R[6] * local[0] + R[7] * local[1] + R[8] * local[2] + p[2]];
  }

  async #prepare(reset) {
    if (reset) this.dbg.resetScene();
    this.dbg.paused = false;
    await this.diag.waitSteps(2);
    // the page runs a few hidden settling-in steps after a reset; measure
    // from the state the viewer actually sees
    while ((this.dbg.preroll ?? 0) > 0) await this.diag.waitSteps(1);
    await this.diag.waitSteps(2);
  }

  /** Watch the scene untouched. */
  async rest({ steps = 180, reset = true } = {}) {
    await this.#prepare(reset);
    const start = this.diag.frame(0);
    await this.diag.waitSteps(steps);
    const s = this.diag.summary(steps);
    return { probe: "rest", steps, bodies: s.bodies, anomalyStarts: s.anomalyStarts, episodes: s.episodes,
      active: s.active, verdict: this.#verdict(s, { maxDrift_mm: 1, maxRot_deg: 1 }) };
  }

  /**
   * Grab `body` at `at` and move the grab point by `delta` (m) over `steps`
   * steps, hold, release, then watch `settle` steps.
   */
  async grab(body, { at = "center", delta = [0.05, 0, 0], steps = 60, hold = 20,
    settle = 120, reset = true } = {}) {
    await this.#prepare(reset);
    const local = this.#local(body, at);
    const p0 = this.#world(body, local);
    const before = this.diag.frame(0);
    const others = before.bodies.filter((b) => b.id !== body).map((b) => b.id);
    const errs = [], samples = [];
    this.diag.event("probe_grab_start", { body, at, delta, steps, hold });
    for (let k = 1; k <= steps + hold; k++) {
      const f = Math.min(1, k / steps);
      const target = [p0[0] + delta[0] * f, p0[1] + delta[1] * f, p0[2] + delta[2] * f];
      this.dbg.scriptDrag = { body, local, target };
      const fr = await this.diag.waitSteps(1);
      const wp = this.#world(body, local);
      const err = hyp(sub(target, wp));
      errs.push(err);
      samples.push({ step: fr.step, err_mm: r3(err * 1e3, 1), height_mm: r3(fr.bodies[body].height * 1e3, 1),
        speed: r3(fr.bodies[body].speed), pen_mm: r3(fr.totals.maxPenetration_mm, 1) });
    }
    const atRelease = this.diag.frame(0);
    this.dbg.scriptDrag = null;
    this.diag.event("probe_grab_release", { body });
    const after = await this.#watchRelease(body, settle);
    const bodyDisp = hyp(sub(atRelease.bodies[body].pos, before.bodies[body].pos));
    const otherMoves = others.map((i) => ({ id: i,
      atRelease_mm: r3(hyp(sub(atRelease.bodies[i].pos, before.bodies[i].pos)) * 1e3, 1),
      final_mm: r3(hyp(sub(this.diag.frame(0).bodies[i].pos, before.bodies[i].pos)) * 1e3, 1) }));
    const during = this.diag.summary(steps + hold + settle);
    return {
      probe: "grab", body, at, delta, steps, hold, settle,
      tracking: { mean_err_mm: r3(errs.reduce((a, b) => a + b, 0) / errs.length * 1e3, 1),
        max_err_mm: r3(Math.max(...errs) * 1e3, 1), final_err_mm: r3(errs[errs.length - 1] * 1e3, 1),
        requested_mm: r3(hyp(delta) * 1e3, 1), achieved_mm: r3(bodyDisp * 1e3, 1) },
      body_at_release: { height_mm: r3(atRelease.bodies[body].height * 1e3, 1),
        lowest_mm: r3(atRelease.bodies[body].lowest * 1e3, 1),
        elevation_deg: r3(atRelease.bodies[body].elevation_deg, 1),
        contacts: atRelease.bodies[body].contacts.map((c) => c.other) },
      others: otherMoves, after_release: after,
      maxPenetration_mm: r3(Math.max(...during.bodies.map((b) => b.maxPenetration_mm)), 1),
      anomalyStarts: during.anomalyStarts, episodes: during.episodes, active: during.active,
      samples: samples.filter((_, i) => i % 5 === 0),
    };
  }

  /** Lift a body straight up by `height` and drop it. */
  async lift(body, { height = 0.05, at = "center", steps = 60, hold = 30, settle = 150, reset = true } = {}) {
    const r = await this.grab(body, { at, delta: [0, 0, height], steps, hold, settle, reset });
    r.probe = "lift";
    r.lifted = r.body_at_release.lowest_mm > 5 && r.body_at_release.contacts.length === 0;
    return r;
  }

  /** Watch a released body: fall time, impact, bounces, final rest. */
  async #watchRelease(body, settle) {
    const dt = this.sim.rt.dt;
    const start = this.diag.frame(0);
    let firstContact = null, minLowest = Infinity, maxHeight = -Infinity, bounces = 0;
    let prevVz = 0, peakFallSpeed = 0, cameToRest = null, fallDrop = 0;
    const z0 = start.bodies[body].lowest;
    const series = [], freeVz = [];
    let contactOnset = null;     // first step the model pushed back (or a guard did)
    for (let k = 1; k <= settle; k++) {
      const f = await this.diag.waitSteps(1);
      const b = f.bodies[body];
      const free = !!b.guard?.freeFlight;
      if (k <= 40) series.push({ k, lowest_mm: r3(b.lowest * 1e3, 1), vz: r3(b.linvel[2], 3),
        elev: r3(b.elevation_deg, 1),
        model_z: b.model ? r3(b.model.lin[2], 2) : null, free: free ? 1 : 0,
        pivot: b.guard?.pivot ? 1 : 0,
        settled: b.guard?.settled ? 1 : 0, contacts: b.contacts.map((c) => c.other),
        ground: b.groundContact ? 1 : 0, guard_mm: b.guard ? r3(b.guard.ground_mm + b.guard.capsule_mm, 2) : 0 });
      maxHeight = Math.max(maxHeight, b.height); minLowest = Math.min(minLowest, b.lowest);
      const touching = b.groundContact || b.contacts.length > 0;
      if (contactOnset === null) {
        // a step is free fall only if nothing touched: the guard cancels the
        // approach velocity in the step the body arrives, which would drag
        // the fitted acceleration to zero
        const guarded = b.guard && (b.guard.ground_mm > 0 || b.guard.capsule_mm > 0);
        if (free && !touching && !guarded) freeVz.push(b.linvel[2]);
        else if (k > 1) contactOnset = k;
      }
      if (firstContact === null) {
        // guards cancel the approach velocity in the contact step itself, so
        // the impact speed is the fastest descent seen before contact
        peakFallSpeed = Math.max(peakFallSpeed, -b.linvel[2]);
        if (touching && k > 1) { firstContact = k; fallDrop = z0 - b.lowest; }
      }
      if (prevVz < -0.05 && b.linvel[2] > 0.02) bounces++;
      prevVz = b.linvel[2];
      if (cameToRest === null && firstContact !== null && b.resting && k > firstContact + 5) cameToRest = k;
    }
    // acceleration actually seen during free flight (should be -g)
    const freeAccel = freeVz.length >= 2
      ? (freeVz[freeVz.length - 1] - freeVz[0]) / ((freeVz.length - 1) * dt) : null;
    // analytic free fall over the height actually dropped before contact
    const analytic = fallDrop > 0 ? Math.sqrt(2 * fallDrop / G) : (z0 > 0 ? Math.sqrt(2 * z0 / G) : 0);
    const fin = this.diag.frame(0).bodies[body];
    return {
      start_lowest_mm: r3(z0 * 1e3, 1), fall_drop_mm: r3(fallDrop * 1e3, 1),
      free_steps: freeVz.length,
      free_fall_accel: freeAccel === null ? null : r3(freeAccel, 2),
      free_fall_ratio: freeAccel === null ? null : r3(-freeAccel / G, 2),   // 1.0 = exact gravity
      contact_onset_s: contactOnset ? r3(contactOnset * dt, 3) : null,
      capsule_contact_s: firstContact ? r3(firstContact * dt, 3) : null,
      fall_time_s: firstContact ? r3(firstContact * dt, 3) : null,
      analytic_fall_s: r3(analytic, 3), impact_speed: r3(peakFallSpeed, 3),
      analytic_impact_speed: r3(Math.sqrt(2 * G * Math.max(fallDrop, 0)), 3), bounces,
      min_lowest_mm: r3(minLowest * 1e3, 1), max_height_mm: r3(maxHeight * 1e3, 1),
      rest_after_s: cameToRest ? r3(cameToRest * dt, 2) : null,
      final: { lowest_mm: r3(fin.lowest * 1e3, 1), elevation_deg: r3(fin.elevation_deg, 1),
        contacts: fin.contacts.map((c) => c.other), unsupported: fin.unsupported,
        resting: fin.resting, speed: r3(fin.speed) },
      series,
    };
  }

  /** Flick: an impulse of `dv` m/s along `dir` at the grab point. */
  async poke(body, { dir = [1, 0, 0], dv = 0.3, at = "center", settle = 120, reset = true, maxDv = null } = {}) {
    await this.#prepare(reset);
    const rt = this.sim.rt, m = this.sim.mass[body];
    const n = hyp(dir); const u = dir.map((x) => x / n);
    const cap = maxDv ?? rt.poke.delta_v[1];     // maxDv: probe beyond the trained range
    const v = Math.min(dv, cap);
    const point = this.#world(body, this.#local(body, at));
    const force = u.map((x) => x * v * m / (rt.poke.steps * rt.dt));
    const before = this.diag.frame(0);
    this.dbg.scriptPoke = { body, point, force, left: rt.poke.steps };
    this.diag.event("probe_poke", { body, dir: u, dv: v, at });
    let peak = 0, stopAt = null, peakAng = 0;
    for (let k = 1; k <= settle; k++) {
      const f = await this.diag.waitSteps(1);
      const b = f.bodies[body];
      peak = Math.max(peak, b.speed); peakAng = Math.max(peakAng, b.angSpeed);
      if (stopAt === null && k > rt.poke.steps + 3 && b.resting) stopAt = k;
    }
    const fin = this.diag.frame(0), s = this.diag.summary(settle);
    return {
      probe: "poke", body, dir: u, dv_requested: dv, dv_applied: v,
      peak_speed: r3(peak), peak_angSpeed: r3(peakAng, 2),
      stop_after_s: stopAt ? r3(stopAt * rt.dt, 2) : null,
      displacement_mm: r3(hyp(sub(fin.bodies[body].pos, before.bodies[body].pos)) * 1e3, 1),
      rotation_deg: r3(axisAngleDeg(fin.bodies[body].axis, before.bodies[body].axis), 1),
      others: s.bodies.filter((b) => b.id !== body).map((b) => ({ id: b.id, displacement_mm: b.displacement_mm })),
      maxPenetration_mm: r3(Math.max(...s.bodies.map((b) => b.maxPenetration_mm)), 1),
      anomalyStarts: s.anomalyStarts, episodes: s.episodes, active: s.active,
    };
  }

  /** Flick: the grab point is pushed along `dir` at `speed` for `distance`
   * and let go, the same path the mouse flick uses. */
  async flick(body, { dir = [1, 0, 0], speed = 0.4, distance = 0.05, at = "center", settle = 120, reset = true } = {}) {
    await this.#prepare(reset);
    const before = this.diag.frame(0);
    this.dbg.flick(body, this.#local(body, at), dir, speed, distance);
    let peak = 0, peakAng = 0, stopAt = null, released = null;
    for (let k = 1; k <= settle; k++) {
      const f = await this.diag.waitSteps(1);
      const b = f.bodies[body];
      peak = Math.max(peak, b.speed); peakAng = Math.max(peakAng, b.angSpeed);
      if (released === null && !f.action) released = k;
      if (stopAt === null && released !== null && k > released + 3 && b.resting) stopAt = k;
    }
    const fin = this.diag.frame(0), s = this.diag.summary(settle);
    return {
      probe: "flick", body, dir, speed, distance, released_after_s: released ? r3(released * this.sim.rt.dt, 2) : null,
      peak_speed: r3(peak), peak_angSpeed: r3(peakAng, 2),
      stop_after_s: stopAt ? r3(stopAt * this.sim.rt.dt, 2) : null,
      displacement_mm: r3(hyp(sub(fin.bodies[body].pos, before.bodies[body].pos)) * 1e3, 1),
      rotation_deg: r3(axisAngleDeg(fin.bodies[body].axis, before.bodies[body].axis), 1),
      others: s.bodies.filter((b) => b.id !== body).map((b) => ({ id: b.id, displacement_mm: b.displacement_mm })),
      maxPenetration_mm: r3(Math.max(...s.bodies.map((b) => b.maxPenetration_mm)), 1),
      anomalyStarts: s.anomalyStarts, episodes: s.episodes, active: s.active,
    };
  }

  /** Pull the most load-bearing pencil out along its axis. */
  async pullBottom({ dist = 0.08, steps = 60, hold = 10, settle = 150, reset = true } = {}) {
    await this.#prepare(reset);
    const f0 = this.diag.frame(0);
    let best = null;
    for (const b of f0.bodies) {
      const score = b.supports.length * 10 - b.height;
      if (!best || score > best.score) best = { id: b.id, score, supports: b.supports };
    }
    const body = best.id;
    const axis = f0.bodies[body].axis;
    const delta = [axis[0] * dist, axis[1] * dist, 0];
    const supported = best.supports.map((i) => ({ id: i, lowest_before_mm: r3(f0.bodies[i].lowest * 1e3, 1) }));
    const r = await this.grab(body, { at: "center", delta, steps, hold, settle, reset: false });
    const fin = this.diag.frame(0);
    r.probe = "pullBottom";
    r.supported = supported.map((s) => ({ ...s, lowest_after_mm: r3(fin.bodies[s.id].lowest * 1e3, 1),
      dropped_mm: r3((f0.bodies[s.id].lowest - fin.bodies[s.id].lowest) * 1e3, 1),
      final_contacts: fin.bodies[s.id].contacts.map((c) => c.other), unsupported: fin.bodies[s.id].unsupported }));
    return r;
  }

  /** Teleport a body up by `height` with zero velocity and watch it fall. */
  async drop(body, { height = 0.05, settle = 150, reset = true } = {}) {
    await this.#prepare(reset);
    this.dbg.paused = true;
    await new Promise((r) => setTimeout(r, 80));
    this.sim.state.pos[body][2] += height;
    this.sim.state.linvel[body] = [0, 0, 0]; this.sim.state.angvel[body] = [0, 0, 0];
    this.diag.event("probe_drop", { body, height });
    this.dbg.paused = false;
    this.dbg.stepOnce = false;
    await this.diag.waitSteps(1);
    const after = await this.#watchRelease(body, settle);
    return { probe: "drop", body, height, ...after,
      fall_time_ratio: after.fall_time_s && after.analytic_fall_s ? r3(after.fall_time_s / after.analytic_fall_s, 2) : null,
      anomalyStarts: this.diag.summary(settle).anomalyStarts,
      episodes: this.diag.summary(settle).episodes };
  }

  /** Run the standard battery and return every report plus a tally. */
  async all({ log = console.log } = {}) {
    const B = this.sim.B, out = {};
    // the pencil to pick up: highest of those carrying nothing (a pencil
    // pinned under two others cannot be lifted at the trained force cap,
    // and that is a fact about the pile, not the physics)
    const f0 = this.diag.frame(0);
    const free = f0 ? f0.bodies.filter((b) => !b.supports.length) : [];
    const pool = free.length ? free : (f0?.bodies ?? []);
    const top = pool.reduce((a, b) => (b.height > a.height ? b : a), pool[0])?.id ?? 0;
    const steps = [
      ["rest", () => this.rest()],
      ["lift", () => this.lift(top)],
      ["grab_end", () => this.grab(top, { at: "end", delta: [0.06, 0, 0] })],
      ["poke", () => this.poke(top)],
      ["flick", () => this.flick(top)],
      ["pullBottom", () => this.pullBottom()],
      ["drop", () => this.drop(top)],
    ];
    for (const [name, fn] of steps) {
      log?.(`[probe] ${name}...`);
      try { out[name] = await fn(); } catch (e) { out[name] = { error: String(e) }; }
    }
    const tally = {};
    for (const r of Object.values(out)) for (const [k, v] of Object.entries(r.anomalyStarts ?? {}))
      tally[k] = (tally[k] ?? 0) + v;
    this.dbg.resetScene();
    return { scene: this.sim.packet.name, bodies: B, results: out, anomalyTally: tally };
  }

  #verdict(summary, { maxDrift_mm, maxRot_deg }) {
    const bad = summary.bodies.filter((b) => b.displacement_mm > maxDrift_mm || b.rotation_deg > maxRot_deg);
    return bad.length ? `moved: ${bad.map((b) => `body ${b.id} ${b.displacement_mm} mm / ${b.rotation_deg} deg`).join(", ")}` : "still";
  }
}

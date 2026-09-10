/**
 * God mode: every fact about every pencil, every step, in numbers.
 *
 * The diagnostics layer (diag.js) answers "did something go wrong" with
 * detectors and episodes. This answers a different question, the one you
 * actually ask when a pencil does something strange: WHAT IS TOUCHING WHAT,
 * WHERE, RIGHT NOW. It exists so a reader never has to infer physics from
 * pixels, and never has to guess which part of a pencil a contact is on.
 *
 * That last part is only possible because every body is now the same
 * canonical BIC Matic Grip (common/pencil.py, js/pencil.js). A contact has
 * a position along the pencil, that position is a fraction t of its length,
 * and the profile says what is at t. So a contact is not "bodies 2 and 3
 * overlap 1.4 mm", it is "body 2's grip is resting on body 3's cone, 41 mm
 * from body 3's point, carrying it".
 *
 * Everything here is derived, never stored twice: the sensors read the same
 * state the simulator integrates and the same profile the guard separates
 * bodies with, so a sensor reading can never disagree with the physics.
 *
 * Entry points, all on `physsplat.sensors` in the browser:
 *
 *   now()            everything, this step, as one object
 *   report()         the same thing as text, for reading
 *   touch()          just the contact map, including the table
 *   body(b)          one pencil in full
 *   surface(b)       where that pencil is being touched, along its length
 *   history(n)       the last n steps, compact
 *   track(path, n)   one number over time, e.g. "bodies.2.speed_mms"
 *   events()         phasing, hovering, self-motion, with step ranges
 *   watch(b)         start recording every contact change for one body
 *   diff(a, b)       what changed between two recorded steps
 */
import { capsuleClosest, capsuleWorld, lowestSurface, quatToMatrix } from "./physics.js";
import { LENGTH, PROFILE, radiusAtOffset } from "./pencil.js";

/**
 * Named regions of the pencil, as fractions of its length from the eraser.
 * These are the same boundaries the profile turns at, so a region name and
 * a radius never disagree.
 */
export const REGIONS = [
  [0.000, 0.020, "eraser"],
  [0.020, 0.065, "cap"],
  [0.065, 0.740, "barrel"],
  [0.740, 0.873, "grip"],
  [0.873, 0.929, "lower barrel"],
  [0.929, 0.984, "cone"],
  [0.984, 1.001, "point"],
];

/** Region name at fractional position t along the pencil. */
export function regionAt(t) {
  for (const [a, b, name] of REGIONS) if (t >= a && t < b) return name;
  return t < 0 ? "off the eraser end" : "off the point";
}

/** A contact position, described the way a person would describe it. */
export function describeAt(t) {
  const mmFromPoint = (1 - t) * LENGTH * 1000;
  return {
    t: +t.toFixed(4),
    region: regionAt(t),
    mm_from_point: +mmFromPoint.toFixed(1),
    mm_from_eraser: +(t * LENGTH * 1000).toFixed(1),
    radius_mm: +(radiusAtOffset((t - 0.5) * LENGTH) * 1000).toFixed(2),
  };
}

const hyp = (v) => Math.hypot(v[0], v[1], v[2]);
const r2 = (x) => Math.round(x * 100) / 100;
const r3 = (x) => Math.round(x * 1000) / 1000;

export class Sensors {
  /**
   * sim: a PhysSim. opts.capacity: how many steps of history to keep.
   * Contact gap threshold: surfaces closer than this count as touching. A
   * real contact sits at 0.0; the tolerance is for the guard's own slack.
   */
  constructor(sim, { capacity = 3600, touchTol = 5e-4 } = {}) {
    this.sim = sim;
    this.capacity = capacity;
    this.touchTol = touchTol;
    this.frames = [];
    this.head = 0;
    this.count = 0;
    this.watching = new Set();
    this.log = [];
    this.lastAction = -1e9;
  }

  get B() { return this.sim.B; }

  // ------------------------------------------------------------ geometry

  #segs() {
    const st = this.sim.state;
    return this.sim.packet.bodies.map((b, i) =>
      capsuleWorld(st.pos[i], st.quat[i], b.capsule));
  }

  /** World position of a point at fraction t along body b's length. */
  pointAt(b, t) {
    const st = this.sim.state;
    const R = quatToMatrix(st.quat[b]);
    const s = (t - 0.5) * LENGTH;
    return [
      st.pos[b][0] + R[0] * s, st.pos[b][1] + R[3] * s, st.pos[b][2] + R[6] * s,
    ];
  }

  /**
   * Where body b touches the table, and by how much.
   *
   * Uses the analytic surface, which is what the ground rule holds, so
   * "gap 0.0" here means the pencil is on the table and nothing else.
   */
  ground(b) {
    const cap = this.sim.packet.bodies[b].capsule;
    const st = this.sim.state;
    const z = lowestSurface(st.pos[b], st.quat[b], cap);
    // which point along the pencil is the lowest one
    const C = capsuleWorld(st.pos[b], st.quat[b], cap);
    const drop = Math.sqrt(Math.max(0, 1 - C.a[2] * C.a[2]));
    let bestT = 0.5, bestZ = Infinity;
    for (const [t, r] of PROFILE) {
      const zz = st.pos[b][2] + (t - 0.5) * 2 * C.h * C.a[2] - r * drop;
      if (zz < bestZ) { bestZ = zz; bestT = t; }
    }
    return {
      gap_mm: r3(z * 1000),
      touching: z <= this.touchTol,
      below_table_mm: z < 0 ? r3(-z * 1000) : 0,
      at: describeAt(bestT),
      world: this.pointAt(b, bestT),
    };
  }

  /**
   * Every pencil-to-pencil contact, with the part of each pencil involved.
   * `pen_mm` positive means the surfaces overlap, which for a settled scene
   * should be ~0 and for a phasing one is the number that matters.
   */
  touch() {
    const segs = this.#segs();
    const st = this.sim.state;
    const out = [];
    for (let i = 0; i < this.B; i++) {
      for (let j = i + 1; j < this.B; j++) {
        const c = capsuleClosest(segs[i], segs[j]);
        const gap = -c.pen;
        if (gap > 0.006) continue;                 // not near each other
        const ti = c.s / (2 * segs[i].h) + 0.5;
        const tj = c.t / (2 * segs[j].h) + 0.5;
        // who is on top: n points from j to i
        const above = c.n[2] > 0.2 ? i : c.n[2] < -0.2 ? j : null;
        // sliding: relative velocity of the two contact points, tangentially
        const vi = this.#pointVel(i, c.ca), vj = this.#pointVel(j, c.cb);
        const vrel = [vi[0] - vj[0], vi[1] - vj[1], vi[2] - vj[2]];
        const vn = vrel[0] * c.n[0] + vrel[1] * c.n[1] + vrel[2] * c.n[2];
        const vt = Math.hypot(vrel[0] - vn * c.n[0], vrel[1] - vn * c.n[1], vrel[2] - vn * c.n[2]);
        out.push({
          bodies: [i, j],
          touching: gap <= this.touchTol,
          gap_mm: r3(gap * 1000),
          pen_mm: c.pen > 0 ? r3(c.pen * 1000) : 0,
          on_i: describeAt(ti),
          on_j: describeAt(tj),
          normal: c.n.map(r3),
          upper: above,
          lower: above === null ? null : (above === i ? j : i),
          world: c.ca.map((x) => r2(x * 1000)),
          closing_mms: r2(-vn * 1000),
          sliding_mms: r2(vt * 1000),
        });
      }
    }
    // and the table
    for (let b = 0; b < this.B; b++) {
      const g = this.ground(b);
      if (g.gap_mm > 6) continue;
      out.push({
        bodies: [b, "table"],
        touching: g.touching,
        gap_mm: g.gap_mm,
        pen_mm: g.below_table_mm,
        on_i: g.at,
        on_j: null,
        normal: [0, 0, 1],
        upper: b,
        lower: "table",
        world: g.world.map((x) => r2(x * 1000)),
        closing_mms: r2(-st.linvel[b][2] * 1000),
        sliding_mms: r2(Math.hypot(st.linvel[b][0], st.linvel[b][1]) * 1000),
      });
    }
    return out;
  }

  /** Velocity of the material point of body b currently at world position p. */
  #pointVel(b, p) {
    const st = this.sim.state;
    const r = [p[0] - st.pos[b][0], p[1] - st.pos[b][1], p[2] - st.pos[b][2]];
    const w = st.angvel[b], v = st.linvel[b];
    return [
      v[0] + w[1] * r[2] - w[2] * r[1],
      v[1] + w[2] * r[0] - w[0] * r[2],
      v[2] + w[0] * r[1] - w[1] * r[0],
    ];
  }

  /**
   * One pencil, in full. Everything a reader could want without having to
   * ask a second question.
   */
  body(b) {
    const st = this.sim.state;
    const cap = this.sim.packet.bodies[b].capsule;
    const C = capsuleWorld(st.pos[b], st.quat[b], cap);
    const elevation = Math.asin(Math.min(1, Math.abs(C.a[2]))) * 180 / Math.PI;
    const g = this.ground(b);
    const contacts = this.touch().filter((c) => c.bodies.includes(b));
    const supports = contacts.filter((c) => c.touching && c.lower !== b && c.upper === b);
    const carrying = contacts.filter((c) => c.touching && c.upper !== b && c.lower === b);
    const speed = hyp(st.linvel[b]) * 1000;
    const spin = hyp(st.angvel[b]);
    return {
      id: b,
      pos_mm: st.pos[b].map((x) => r2(x * 1000)),
      quat: st.quat[b].map(r3),
      point_world_mm: this.pointAt(b, 1).map((x) => r2(x * 1000)),
      eraser_world_mm: this.pointAt(b, 0).map((x) => r2(x * 1000)),
      elevation_deg: r2(elevation),
      speed_mms: r2(speed),
      spin_rads: r3(spin),
      moving: speed > 0.5 || spin > 0.02,
      resting: (this.sim.restCount?.[b] ?? 0) > 0,
      rest_steps: this.sim.restCount?.[b] ?? 0,
      held_still: (this.sim.restCount?.[b] ?? 0) >= 15,
      floor: g,
      touching_bodies: contacts.filter((c) => c.touching)
        .map((c) => (c.bodies[0] === b ? c.bodies[1] : c.bodies[0])),
      resting_on: supports.map((c) => (c.bodies[0] === b ? c.bodies[1] : c.bodies[0])),
      carrying: carrying.map((c) => (c.bodies[0] === b ? c.bodies[1] : c.bodies[0])),
      deepest_pen_mm: contacts.reduce((m, c) => Math.max(m, c.pen_mm), 0),
      unsupported: !g.touching && !supports.length,
      contacts,
    };
  }

  /**
   * Where along body b's length it is being touched, as a strip. Answers
   * "is anything on its tip" directly, which is the question a pencil
   * standing up on a neighbour's point raises.
   */
  surface(b) {
    const strip = REGIONS.map(([a, z, name]) => ({ region: name, from: a, to: z, contacts: [] }));
    for (const c of this.touch()) {
      if (!c.bodies.includes(b)) continue;
      const mine = c.bodies[0] === b ? c.on_i : c.on_j;
      if (!mine) continue;
      const cell = strip.find((s) => mine.t >= s.from && mine.t < s.to);
      if (cell) {
        cell.contacts.push({
          with: c.bodies[0] === b ? c.bodies[1] : c.bodies[0],
          gap_mm: c.gap_mm, pen_mm: c.pen_mm,
          mm_from_point: mine.mm_from_point,
          bearing: c.lower === b ? "carrying" : c.upper === b ? "resting on it" : "side by side",
        });
      }
    }
    return strip;
  }

  // -------------------------------------------------------------- capture

  /** Record one step. The physics loop calls this after every sim.step(). */
  sample(actInfo = null) {
    if (actInfo) this.lastAction = this.sim.stepCount;
    const st = this.sim.state;
    const frame = {
      step: this.sim.stepCount,
      action: actInfo ? { kind: actInfo.kind, body: actInfo.body } : null,
      steps_since_action: this.sim.stepCount - this.lastAction,
      bodies: [],
      contacts: this.touch(),
      swept: this.sim.sweptInfo ? { ...this.sim.sweptInfo } : null,
      guard: this.sim.last ? {
        ground: Array.from(this.sim.last.guard.ground ?? []).map((x) => r3(x * 1000)),
        capsule: Array.from(this.sim.last.guard.capsule ?? []).map((x) => r3(x * 1000)),
      } : null,
    };
    for (let b = 0; b < this.B; b++) {
      const g = this.ground(b);
      frame.bodies.push({
        pos: st.pos[b].map((x) => r3(x * 1000)),
        speed_mms: r2(hyp(st.linvel[b]) * 1000),
        spin_rads: r3(hyp(st.angvel[b])),
        floor_mm: g.gap_mm,
        below_table_mm: g.below_table_mm,
        rest_steps: this.sim.restCount?.[b] ?? 0,
      });
    }
    frame.worst_pen_mm = frame.contacts.reduce((m, c) => Math.max(m, c.pen_mm), 0);
    if (this.frames.length < this.capacity) this.frames.push(frame);
    else this.frames[this.head] = frame;
    this.head = (this.head + 1) % this.capacity;
    this.count++;
    if (this.watching.size) this.#logChanges(frame);
    return frame;
  }

  #logChanges(frame) {
    for (const b of this.watching) {
      const now = frame.contacts
        .filter((c) => c.bodies.includes(b) && c.touching)
        .map((c) => `${c.bodies[0] === b ? c.on_i.region : c.on_j.region}` +
          ` <-> ${c.bodies[0] === b ? c.bodies[1] : c.bodies[0]}` +
          `${c.bodies[1] === "table" ? "" : ":" + (c.bodies[0] === b ? c.on_j.region : c.on_i.region)}`)
        .sort().join(" | ");
      const prev = this[`__last${b}`];
      if (now !== prev) {
        this.log.push({ step: frame.step, body: b, was: prev ?? "(start)", now });
        this[`__last${b}`] = now;
      }
    }
  }

  watch(b) { this.watching.add(b); return `watching body ${b}; read sensors.log`; }
  unwatch(b) { this.watching.delete(b); }

  /** The last n recorded steps, oldest first. */
  history(n = 60) {
    const out = [];
    const total = Math.min(this.count, this.frames.length);
    for (let k = Math.max(0, total - n); k < total; k++) {
      const idx = this.count <= this.frames.length
        ? k : (this.head + k) % this.frames.length;
      out.push(this.frames[idx]);
    }
    return out;
  }

  /** One number over time. path like "bodies.2.speed_mms" or "worst_pen_mm". */
  track(path, n = 300) {
    const keys = path.split(".");
    return this.history(n).map((f) => {
      let v = f;
      for (const k of keys) v = v?.[Array.isArray(v) ? Number(k) : k];
      return { step: f.step, value: v };
    });
  }

  // -------------------------------------------------------------- verdicts

  /**
   * Things that should not be happening, found by looking at the recorded
   * steps rather than by watching. Each says which body, which steps, and
   * how big, so it can be reproduced.
   */
  events({ n = 1800 } = {}) {
    const h = this.history(n);
    if (!h.length) return [];
    const out = [];
    const open = {};
    const start = (key, seed) => { open[key] ??= seed; };
    const end = (key, kind) => {
      if (!open[key]) return;
      out.push({ kind, ...open[key] });
      delete open[key];
    };
    for (const f of h) {
      for (let b = 0; b < (f.bodies?.length ?? 0); b++) {
        const bb = f.bodies[b];
        // through the table
        if (bb.below_table_mm > 0.05) {
          start(`sink${b}`, { body: b, from: f.step, peak_mm: 0 });
          open[`sink${b}`].peak_mm = Math.max(open[`sink${b}`].peak_mm, bb.below_table_mm);
          open[`sink${b}`].to = f.step;
        } else end(`sink${b}`, "through the table");
        // in the air with nothing under it
        const held = f.contacts.some((c) => c.touching && c.bodies.includes(b));
        if (bb.floor_mm > 1 && !held && bb.speed_mms < 1 && !f.action) {
          start(`hover${b}`, { body: b, from: f.step, height_mm: bb.floor_mm });
          open[`hover${b}`].to = f.step;
        } else end(`hover${b}`, "hovering, nothing under it");
        // moving with nothing driving it
        if (f.steps_since_action > 30 && bb.speed_mms > 2) {
          start(`self${b}`, { body: b, from: f.step, peak_mms: 0 });
          open[`self${b}`].peak_mms = Math.max(open[`self${b}`].peak_mms, bb.speed_mms);
          open[`self${b}`].to = f.step;
        } else end(`self${b}`, "moving with nothing touching or driving it");
      }
      // surfaces through each other
      for (const c of f.contacts) {
        if (c.pen_mm > 1) {
          const key = `pen${c.bodies.join("-")}`;
          start(key, { bodies: c.bodies, from: f.step, peak_mm: 0, where: [c.on_i, c.on_j] });
          open[key].peak_mm = Math.max(open[key].peak_mm, c.pen_mm);
          open[key].to = f.step;
        }
      }
      for (const key of Object.keys(open)) {
        if (!key.startsWith("pen")) continue;
        const pair = key.slice(3);
        if (!f.contacts.some((c) => c.bodies.join("-") === pair && c.pen_mm > 1)) {
          end(key, "surfaces overlapping");
        }
      }
    }
    for (const key of Object.keys(open)) {
      const kind = key.startsWith("sink") ? "through the table"
        : key.startsWith("hover") ? "hovering, nothing under it"
          : key.startsWith("self") ? "moving with nothing touching or driving it"
            : "surfaces overlapping";
      out.push({ kind, ...open[key], ongoing: true });
    }
    return out;
  }

  /** Everything, this step. */
  now() {
    return {
      step: this.sim.stepCount,
      steps_since_action: this.sim.stepCount - this.lastAction,
      bodies: Array.from({ length: this.B }, (_, b) => this.body(b)),
      contacts: this.touch(),
      swept: this.sim.sweptInfo ? { ...this.sim.sweptInfo } : null,
      scene_at_rest: Array.from({ length: this.B }, (_, b) => b)
        .every((b) => (this.sim.restCount?.[b] ?? 0) >= 15),
    };
  }

  /** The same thing, to read rather than to parse. */
  report() {
    const s = this.now();
    const L = [`step ${s.step}${s.scene_at_rest ? "  (whole scene at rest)" : ""}`];
    for (const b of s.bodies) {
      L.push(`body ${b.id}: ${b.moving ? `MOVING ${b.speed_mms} mm/s, ${b.spin_rads} rad/s`
        : b.held_still ? "still (held)" : "still"}` +
        `, ${b.elevation_deg} deg from flat` +
        `, floor gap ${b.floor.gap_mm} mm on its ${b.floor.at.region}` +
        (b.floor.below_table_mm ? `  *** ${b.floor.below_table_mm} mm THROUGH THE TABLE ***` : "") +
        (b.unsupported ? "  *** NOTHING UNDER IT ***" : ""));
      for (const c of b.contacts) {
        const other = c.bodies[0] === b.id ? c.bodies[1] : c.bodies[0];
        const mine = c.bodies[0] === b.id ? c.on_i : c.on_j;
        const theirs = c.bodies[0] === b.id ? c.on_j : c.on_i;
        L.push(`   ${c.touching ? "touches" : `${c.gap_mm} mm from`} ${other}` +
          `: its ${mine.region} (${mine.mm_from_point} mm from its point)` +
          (theirs ? ` against their ${theirs.region}` : " against the table") +
          (c.pen_mm ? `  *** OVERLAPPING ${c.pen_mm} mm ***` : "") +
          (c.sliding_mms > 1 ? `  sliding ${c.sliding_mms} mm/s` : ""));
      }
    }
    const ev = this.events({ n: 600 });
    if (ev.length) {
      L.push("", "recent problems:");
      for (const e of ev.slice(0, 20)) {
        L.push(`  ${e.kind}: ${e.bodies ? `bodies ${e.bodies.join(" and ")}` : `body ${e.body}`}` +
          `, steps ${e.from}-${e.to ?? e.from}` +
          `${e.peak_mm !== undefined ? `, peak ${e.peak_mm} mm` : ""}` +
          `${e.peak_mms !== undefined ? `, peak ${e.peak_mms} mm/s` : ""}` +
          `${e.height_mm !== undefined ? `, ${e.height_mm} mm up` : ""}` +
          `${e.ongoing ? " (still happening)" : ""}`);
      }
    }
    return L.join("\n");
  }

  /** What changed between two recorded steps. */
  diff(stepA, stepB) {
    const h = this.history(this.frames.length);
    const a = h.find((f) => f.step === stepA), b = h.find((f) => f.step === stepB);
    if (!a || !b) return "one of those steps is not in the buffer";
    const out = [];
    for (let i = 0; i < a.bodies.length; i++) {
      const d = Math.hypot(...a.bodies[i].pos.map((x, k) => b.bodies[i].pos[k] - x));
      if (d > 0.05) out.push(`body ${i} moved ${r2(d)} mm`);
    }
    const key = (c) => c.bodies.join("-");
    const was = new Set(a.contacts.filter((c) => c.touching).map(key));
    const now = new Set(b.contacts.filter((c) => c.touching).map(key));
    for (const k of now) if (!was.has(k)) out.push(`${k} started touching`);
    for (const k of was) if (!now.has(k)) out.push(`${k} stopped touching`);
    return out.join("\n") || "nothing changed";
  }
}

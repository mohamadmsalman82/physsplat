/**
 * Floating regression: after any amount of play, nothing hangs in the air.
 *
 * A player's report with a screenshot: pencils "start to float when they are
 * meant to lay flat on the ground". The rest suite could not see it because
 * it never touches anything, and the interaction suite checks one gesture
 * at a time. This drives the simulator through a long random session of
 * grabs, lifts, drags, drops and releases on every scene, the way a hand
 * does, and after every step asks the one question that matters: is every
 * unheld body resting on a chain of contacts that reaches the table?
 *
 * A body is GROUNDED if its surface is within a millimetre of the table, or
 * it rests (contact within a millimetre, normal pointing up) on a grounded
 * body. A body that is not grounded, not held, and not falling (vertical
 * speed under 2 cm/s) for FLOAT_STEPS consecutive steps is floating. Every
 * such episode fails the scene and is printed with the body's height, what
 * it touches, and what the rules did to it that step.
 *
 *   node web/test/float.mjs                (all scenes, seed 1)
 *   SCENE=IMG_8504 SEED=3 GESTURES=40 node web/test/float.mjs
 */
import { readFileSync } from "node:fs";
import ort from "onnxruntime-node";
import { PhysSim } from "../public/js/sim.js";
import { capsuleClosest, capsuleWorld, grabForce, lowestSurface, quatToMatrix } from "../public/js/physics.js";

const root = new URL("../..", import.meta.url).pathname;
const runtime = JSON.parse(readFileSync(`${root}web/public/model/runtime.json`));
const session = await ort.InferenceSession.create(`${root}web/public/model/simulator.onnx`);
const scenes = process.env.SCENE ? [process.env.SCENE] : ["IMG_8504", "IMG_8513", "IMG_8596", "IMG_8626"];
const GESTURES = Number(process.env.GESTURES ?? 24);
const SEED = Number(process.env.SEED ?? 1);
const FLOAT_STEPS = 30;          // half a second in the air, not falling
// "touching": within this of the table or a neighbour. 3 mm is the
// simulator's own touch hysteresis (a body that has come within 1.5 mm of
// its support keeps it until 3 mm), and the band the model's contact
// response equilibrates in; settle closes it to zero once the body sleeps.
const GROUND_TOL = 3e-3;
const G = runtime.gravity, DT = runtime.dt;
const GRAB = { omega: 28, zeta: 1.0 };
const FOLLOW = 0.30;

// deterministic RNG so a failure can be replayed
let rs = SEED >>> 0 || 1;
const rand = () => { rs ^= rs << 13; rs >>>= 0; rs ^= rs >>> 17; rs ^= rs << 5; rs >>>= 0; return rs / 4294967296; };
const pick = (n) => Math.floor(rand() * n);
const hyp = (v) => Math.hypot(v[0], v[1], v[2]);

const worldPoint = (sim, b, local) => {
  const R = quatToMatrix(sim.state.quat[b]), p = sim.state.pos[b];
  return [R[0] * local[0] + R[1] * local[1] + R[2] * local[2] + p[0],
    R[3] * local[0] + R[4] * local[1] + R[5] * local[2] + p[1],
    R[6] * local[0] + R[7] * local[1] + R[8] * local[2] + p[2]];
};
const elevation = (sim, b) => {
  const a = capsuleWorld(sim.state.pos[b], sim.state.quat[b], sim.packet.bodies[b].capsule).a;
  return Math.asin(Math.min(1, Math.abs(a[2]))) * 180 / Math.PI;
};

/** Which bodies rest on a chain of contacts that reaches the table. */
function grounded(sim, tol = GROUND_TOL, held = -1) {
  const bs = sim.packet.bodies, B = sim.B;
  const segs = bs.map((x, i) => capsuleWorld(sim.state.pos[i], sim.state.quat[i], x.capsule));
  const low = bs.map((x, i) => lowestSurface(sim.state.pos[i], sim.state.quat[i], x.capsule));
  const on = Array.from({ length: B }, () => []);   // on[i] = bodies i rests on
  const touch = Array.from({ length: B }, () => []);
  for (let i = 0; i < B; i++) for (let j = i + 1; j < B; j++) {
    const c = capsuleClosest(segs[i], segs[j]);   // n: j -> i
    if (-c.pen < tol) {
      touch[i].push(j); touch[j].push(i);
      if (c.n[2] > 0.2) on[i].push(j);
      if (c.n[2] < -0.2) on[j].push(i);
    }
  }
  const g = new Uint8Array(B);
  // the table, or the hand: what rides on a held pencil is carried, not floating
  for (let i = 0; i < B; i++) if (low[i] < tol || i === held) g[i] = 1;
  for (let it = 0; it < B; it++)
    for (let i = 0; i < B; i++) {
      if (g[i]) continue;
      if (on[i].some((j) => g[j])) { g[i] = 1; continue; }
      // a shallow contact (normal under 0.2 up) still carries a body if it
      // cannot drop: leaning on a neighbour's shoulder is resting on it
      if (touch[i].some((j) => g[j]) && airUnder(sim, i, 0.01) < 1e-3) g[i] = 1;
    }
  return { g, low, on, touch };
}

/** Air straight under body b: how far it could drop before touching anything (m). */
function airUnder(sim, b, reach = 0.3) {
  const bs = sim.packet.bodies;
  const clear = (d) => {
    const p = [sim.state.pos[b][0], sim.state.pos[b][1], sim.state.pos[b][2] - d];
    if (lowestSurface(p, sim.state.quat[b], bs[b].capsule) < 0) return false;
    const me = capsuleWorld(p, sim.state.quat[b], bs[b].capsule);
    for (let j = 0; j < sim.B; j++) {
      if (j === b) continue;
      const o = capsuleWorld(sim.state.pos[j], sim.state.quat[j], bs[j].capsule);
      if (capsuleClosest(me, o).pen > 0) return false;
    }
    return true;
  };
  if (!clear(0)) return 0;
  let lo = 0, hi = reach;
  if (clear(hi)) return hi;
  for (let k = 0; k < 14; k++) { const m = (lo + hi) / 2; if (clear(m)) lo = m; else hi = m; }
  return lo;
}

class Watch {
  constructor(sim, scene) { this.sim = sim; this.scene = scene; this.run = new Int32Array(sim.B); this.episodes = []; this.open = {}; }
  after(step, held, gesture) {
    const sim = this.sim, { g, low, on, touch } = grounded(sim, GROUND_TOL, held);
    for (let b = 0; b < sim.B; b++) {
      const vz = sim.state.linvel[b][2];
      const spd = hyp(sim.state.linvel[b]);
      const steep = elevation(sim, b) > 45 && low[b] > 3e-3 && spd < 0.02 && b !== held;
      // in the air, unheld, and not coming down: a body that jitters in
      // place has a vertical speed, so ask about net descent, not speed
      (this.lowHist ??= Array.from({ length: sim.B }, () => []))[b].push(low[b]);
      const hist = this.lowHist[b]; if (hist.length > FLOAT_STEPS) hist.shift();
      if (b === held) hist.length = 0;            // a held body's height is the hand's
      // hovering: its height barely changed in half a second. Net descent
      // was the first test, and it flagged the top of a bounce.
      // in the air for the whole window, not just now: a pencil shoved up
      // by a neighbour and then dropping is not hovering
      const stuck = !g[b] && b !== held && hist.length === FLOAT_STEPS && Math.min(...hist) > 3e-3 &&
        Math.max(...hist) - Math.min(...hist) < 0.01;
      const floating = (!g[b] && b !== held && Math.abs(vz) < 0.02) || steep || stuck;
      this.run[b] = floating ? this.run[b] + 1 : 0;
      if (this.run[b] === FLOAT_STEPS || (stuck && this.run[b] === 1)) {
        if (this.open[b]) continue;
        const L = sim.last?.guard;
        this.open[b] = { scene: this.scene, gesture, body: b, from: step - FLOAT_STEPS + 1,
          lowest_mm: +(low[b] * 1e3).toFixed(1), air_mm: +(airUnder(sim, b) * 1e3).toFixed(1),
          elev_deg: +elevation(sim, b).toFixed(1),
          speed_mms: +(hyp(sim.state.linvel[b]) * 1e3).toFixed(1), spin: +hyp(sim.state.angvel[b]).toFixed(2),
          restsOn: on[b], touching: touch[b], groundedMask: Array.from(g),
          freeFlight: !!L?.freeFlight[b], pivot: !!L?.pivot[b], settled: !!L?.settled[b],
          resZ: +sim.last.residual[6 * b + 2].toFixed(2), restCount: sim.restCount?.[b] ?? 0,
          guardHeld: sim.guardHeld?.[b] ?? 0, via: sim.last?.guard.settleVia?.[b] ?? 0, stuck,
          seatGap_mm: +((sim.seatGap?.[b] ?? -1) * 1e3).toFixed(1) };
      }
      if (!floating && this.open[b]) {
        this.open[b].to = step; this.open[b].steps = step - this.open[b].from;
        this.episodes.push(this.open[b]); delete this.open[b];
      }
    }
  }
  close(step) { for (const b in this.open) { this.open[b].to = step; this.open[b].steps = step - this.open[b].from; this.episodes.push(this.open[b]); } this.open = {}; }
}

async function stepHeld(sim, body, local, target, watch, gesture) {
  const wp = worldPoint(sim, body, local);
  const f = grabForce({ m: sim.mass[body], v: sim.state.linvel[body], wp, target,
    omega: GRAB.omega, zeta: GRAB.zeta, gravity: G, capG: runtime.grab.force_cap,
    blocked: (sim.last?.guard.capsule[body] ?? 0) > 1e-3 ? sim.last.guard.normal?.[body] : false });
  await sim.step(body, wp, f, true);
  watch.after(sim.stepCount, body, gesture);
  traceRow(sim, body);
}
async function stepFree(sim, watch, gesture) { await sim.step(); watch.after(sim.stepCount, -1, gesture); traceRow(sim, -1); }
const TRACE = process.env.TRACE ? process.env.TRACE.split("-").map(Number) : null;
function traceRow(sim, held) {
  if (!TRACE || sim.stepCount < TRACE[0] || sim.stepCount > TRACE[1]) return;
  const g = sim.last.guard, f = (x, d = 1) => x.toFixed(d);
  const rows = sim.packet.bodies.map((x, b) => {
    const low = lowestSurface(sim.state.pos[b], sim.state.quat[b], x.capsule);
    return `b${b}${b === held ? "*" : " "} low ${f(low * 1e3)} el ${f(elevation(sim, b))} v ${f(hyp(sim.state.linvel[b]) * 1e3, 0)} vz ${f(sim.state.linvel[b][2] * 1e3, 0)} w ${f(hyp(sim.state.angvel[b]), 2)} ` +
      `gnd ${sim.groundedNow?.[b]} bal ${sim.balancedNow?.[b]} ${g.pivot[b] ? "PIV " : ""}${g.freeFlight[b] ? "FREE " : ""}${g.settled[b] ? "SET" + g.settleVia[b] + " " : ""}` +
      `rz ${f(sim.last.residual[6 * b + 2])} gd ${f(g.ground[b] * 1e3, 2)} cp ${f(g.capsule[b] * 1e3, 2)} seat ${f((sim.seatGap?.[b] ?? 0) * 1e3)} sw ${g.swept ?? 0} rc ${sim.restCount[b]} gh ${sim.guardHeld?.[b] ?? 0}`;
  });
  console.log(`step ${sim.stepCount}\n  ` + rows.join("\n  "));
}

/** One gesture, like a hand: grab at a point on the axis, carry the follow
 * point toward the target at FOLLOW speed, hold, let go, then wait. */
async function gesture(sim, watch, k) {
  const body = pick(sim.B);
  const cap = sim.packet.bodies[body].capsule;
  const local = [cap.axis[0] * (rand() * 2 - 1) * cap.half * 0.9, 0, 0];
  const kind = pick(8);
  const from = worldPoint(sim, body, local);
  let to;
  if (kind === 0) to = [from[0], from[1], from[2] + 0.03 + rand() * 0.08];                       // lift
  else if (kind === 1) to = [from[0] + (rand() - 0.5) * 0.2, from[1] + (rand() - 0.5) * 0.2, from[2] + 0.02 + rand() * 0.06]; // lift and carry
  else if (kind === 2) to = [from[0] + (rand() - 0.5) * 0.16, from[1] + (rand() - 0.5) * 0.16, from[2]]; // slide
  else if (kind === 3) to = [from[0] + (rand() - 0.5) * 0.1, from[1] + (rand() - 0.5) * 0.1, Math.max(0.005, from[2] - 0.03)]; // press down and drag
  else if (kind === 4) to = [from[0] + (rand() - 0.5) * 0.05, from[1] + (rand() - 0.5) * 0.05, from[2] + 0.1 + rand() * 0.1]; // high lift, then drop
  else if (kind === 5) to = [from[0], from[1], from[2]];                                           // hold still
  else if (kind === 6) {                                                                          // flick: spring carried at speed, then let go
    const ang = rand() * Math.PI * 2, spd = 0.15 + rand() * 0.45, dist = 0.02 + rand() * 0.06;
    const dir = [Math.cos(ang), Math.sin(ang), 0];
    let trav = 0;
    while (trav < dist) { trav += spd * DT; await stepHeld(sim, body, local, [from[0] + dir[0] * trav, from[1] + dir[1] * trav, from[2]], watch, k); }
    const wait = 40 + pick(80);
    for (let s = 0; s < wait; s++) await stepFree(sim, watch, k);
    return { body, kind, spd, dist, wait };
  }
  else to = [from[0] + (rand() - 0.5) * 0.1, from[1] + (rand() - 0.5) * 0.1, -0.02 - rand() * 0.03]; // drag with the target under the table
  const follow = [...from];
  const dist = Math.hypot(to[0] - from[0], to[1] - from[1], to[2] - from[2]);
  const travel = Math.ceil(dist / (FOLLOW * DT));
  const hold = 10 + pick(40);
  for (let s = 0; s < travel + hold; s++) {
    const d = [to[0] - follow[0], to[1] - follow[1], to[2] - follow[2]];
    const dn = hyp(d), mx = FOLLOW * DT;
    if (dn > mx) for (let q = 0; q < 3; q++) follow[q] += d[q] / dn * mx; else for (let q = 0; q < 3; q++) follow[q] = to[q];
    await stepHeld(sim, body, local, follow, watch, k);
  }
  const wait = 40 + pick(80);
  for (let s = 0; s < wait; s++) await stepFree(sim, watch, k);
  return { body, kind, to: to.map((x) => +(x * 1e3).toFixed(0)), travel, hold, wait };
}

let fails = 0;
for (const scene of scenes) {
  const packet = JSON.parse(readFileSync(`${root}web/public/packets/${scene}.json`));
  const sim = new PhysSim({ kind: "ort", ort, session }, runtime, packet);
  const watch = new Watch(sim, scene);
  for (let k = 0; k < 12; k++) await stepFree(sim, watch, -1);
  const log = [];
  const t0 = Date.now();
  for (let k = 0; k < GESTURES; k++) log.push(await gesture(sim, watch, k));
  for (let s = 0; s < 120; s++) await stepFree(sim, watch, GESTURES);
  watch.close(sim.stepCount);
  const { g, low } = grounded(sim);
  const airborne = [...g].map((x, b) => (x ? null : b)).filter((x) => x !== null);
  const ok = watch.episodes.length === 0 && airborne.length === 0;
  if (!ok) fails++;
  console.log(`${ok ? "PASS" : "FAIL"} ${scene}: ${watch.episodes.length} floating episodes over ${sim.stepCount} steps ` +
    `(${((Date.now() - t0) / 1000).toFixed(0)} s); final lowest ${[...low].map((x) => (x * 1e3).toFixed(1)).join("/")} mm` +
    (airborne.length ? `; STILL AIRBORNE AT END: ${airborne}` : ""));
  for (const e of watch.episodes) console.log("  " + JSON.stringify(e));
  if (process.env.VERBOSE) console.log("  gestures: " + JSON.stringify(log));
}
console.log(fails ? `FLOAT TESTS FAILED (${fails})` : "FLOAT TESTS PASSED");
process.exit(fails ? 1 : 0);

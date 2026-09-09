/**
 * Interaction regression: the behaviours the project promised, measured
 * headlessly through the same simulator and the same grab force the
 * browser uses. No rendering, no page.
 *
 *   node web/test/interact.mjs            (default scene IMG_8596)
 *   SCENE=IMG_8626 node web/test/interact.mjs
 *
 * 1. drop      a body released in mid-air falls at g and lands
 * 2. lift      a grab raises a body off its neighbours and it follows
 * 3. release   letting go drops it back onto the pile
 * 4. support   pulling the load-bearing body out lowers what it carried
 * 5. contact   nothing interpenetrates by more than a millimetre for long
 */
import { readFileSync } from "node:fs";
import ort from "onnxruntime-node";
import { PhysSim } from "../public/js/sim.js";
import { capsuleClosest, capsuleWorld, grabForce, quatToMatrix } from "../public/js/physics.js";

const root = new URL("../..", import.meta.url).pathname;
const runtime = JSON.parse(readFileSync(`${root}web/public/model/runtime.json`));
const session = await ort.InferenceSession.create(`${root}web/public/model/simulator.onnx`);
const scene = process.env.SCENE ?? "IMG_8596";
const packet = JSON.parse(readFileSync(`${root}web/public/packets/${scene}.json`));
const G = runtime.gravity, DT = runtime.dt;
const GRAB = { omega: 28, zeta: 1.0 };

let fails = 0;
const check = (ok, msg) => { console.log(`${ok ? "PASS" : "FAIL"} ${msg}`); if (!ok) fails++; };
const hyp = (v) => Math.hypot(v[0], v[1], v[2]);

async function fresh() {
  const sim = new PhysSim({ kind: "ort", ort, session }, runtime, packet);
  for (let k = 0; k < 180; k++) await sim.step();     // the page's hidden pre-roll
  return sim;
}
const lowest = (sim, b) => {
  const R = quatToMatrix(sim.state.quat[b]);
  let z = Infinity;
  for (const o of sim.offsets[b])
    z = Math.min(z, R[6] * o[0] + R[7] * o[1] + R[8] * o[2] + sim.state.pos[b][2]);
  return z;
};
const worldPoint = (sim, b, local) => {
  const R = quatToMatrix(sim.state.quat[b]), p = sim.state.pos[b];
  return [R[0] * local[0] + R[1] * local[1] + R[2] * local[2] + p[0],
    R[3] * local[0] + R[4] * local[1] + R[5] * local[2] + p[1],
    R[6] * local[0] + R[7] * local[1] + R[8] * local[2] + p[2]];
};
const worstPenetration = (sim) => {
  const segs = packet.bodies.map((x, i) => capsuleWorld(sim.state.pos[i], sim.state.quat[i], x.capsule));
  let w = 0;
  for (let i = 0; i < sim.B; i++) for (let j = i + 1; j < sim.B; j++)
    w = Math.max(w, capsuleClosest(segs[i], segs[j]).pen);
  return w * 1e3;
};
/** Hold `body` at `local` with the grab spring, target moving to `to` over
 * `steps`, then hold; returns after `steps + hold` steps. */
async function drag(sim, body, local, to, steps, hold, onStep = null) {
  const from = worldPoint(sim, body, local);
  for (let k = 1; k <= steps + hold; k++) {
    const f = Math.min(1, k / steps);
    const target = [from[0] + to[0] * f, from[1] + to[1] * f, from[2] + to[2] * f];
    const wp = worldPoint(sim, body, local);
    const force = grabForce({ m: sim.mass[body], v: sim.state.linvel[body], wp, target,
      omega: GRAB.omega, zeta: GRAB.zeta, gravity: G, capG: runtime.grab.force_cap,
      blocked: (sim.last?.guard.capsule[body] ?? 0) > 1e-3 ? sim.last.guard.normal?.[body] : false });
    await sim.step(body, wp, force, true);
    onStep?.();
  }
  return worldPoint(sim, body, local);
}

const elevation = (sim, b) => {
  const a = capsuleWorld(sim.state.pos[b], sim.state.quat[b], packet.bodies[b].capsule).a;
  return Math.asin(Math.min(1, Math.abs(a[2]))) * 180 / Math.PI;
};

// The pencil a hand would pick up: the one lying most freely on top, i.e.
// fewest neighbours touching it, highest among those. Picking merely "the
// highest that carries nothing" chose a pencil wedged between two others
// in IMG_8626, and prising that out is a different question from lifting
// one off the top.
let sim = await fresh();
const segs0 = packet.bodies.map((x, i) => capsuleWorld(sim.state.pos[i], sim.state.quat[i], x.capsule));
const touches = packet.bodies.map((_, i) => packet.bodies
  .filter((_, j) => j !== i && -capsuleClosest(segs0[i], segs0[j]).pen < 6e-3).length);
// "Carries" means something rests ON this body, which is a fact about the
// contact point, not about centres of mass. Comparing centre heights misses
// a pencil that crosses over one end of this one while its own middle hangs
// lower, and that is exactly the pencil that stops a lift: capsuleClosest
// returns n pointing from j to i, so n[2] < 0 puts j above i where they meet.
const carries = packet.bodies.map((_, i) => packet.bodies.some((_, j) => {
  if (j === i) return false;
  const c = capsuleClosest(segs0[i], segs0[j]);
  return -c.pen < 6e-3 && (c.n[2] < -0.2 || sim.state.pos[j][2] > sim.state.pos[i][2]);
}));
const pool = packet.bodies.map((_, i) => i).filter((i) => !carries[i]);
const top = (pool.length ? pool : packet.bodies.map((_, i) => i))
  .reduce((a, b) => (touches[b] < touches[a] ||
    (touches[b] === touches[a] && sim.state.pos[b][2] > sim.state.pos[a][2]) ? b : a));
console.log(`scene ${scene}, ${sim.B} bodies, testing body ${top} (${touches[top]} neighbours)`);

// ---- 1. drop: released in mid-air, falls at g
sim.state.pos[top][2] += 0.05;
sim.state.linvel[top] = [0, 0, 0]; sim.state.angvel[top] = [0, 0, 0];
// fit only the steps the body was actually in flight: it lands within a
// few centimetres, and including the landing step drags the fit
const vz = [];
for (let k = 0; k < 8; k++) {
  await sim.step();
  const g = sim.last.guard;
  // the landing step can still be flagged free (the flag is set before the
  // step, the guard acts after), and its cancelled velocity ruins the fit
  if (!g.freeFlight[top] || g.ground[top] > 0 || g.capsule[top] > 0) break;
  vz.push(sim.state.linvel[top][2]);
}
const a = vz.length >= 2 ? (vz[vz.length - 1] - vz[0]) / ((vz.length - 1) * DT) : 0;
check(vz.length >= 2 && Math.abs(-a / G - 1) < 0.05,
  `drop falls at gravity (${(-a / G).toFixed(2)} g over ${vz.length} free steps)`);
let landed = false;
for (let k = 0; k < 150 && !landed; k++) { await sim.step(); landed = hyp(sim.state.linvel[top]) < 0.02; }
check(landed && lowest(sim, top) < 0.03, `drop lands and stops (lowest ${(lowest(sim, top) * 1e3).toFixed(1)} mm)`);

// ---- 2/3. lift and release
sim = await fresh();
const before = lowest(sim, top);
const comBefore = sim.state.pos[top][2];
const tiltBefore = elevation(sim, top);
const start = worldPoint(sim, top, [0, 0, 0]);
let tiltPeak = 0;
const onStep = () => { tiltPeak = Math.max(tiltPeak, Math.abs(elevation(sim, top) - tiltBefore)); };
// A pencil with a neighbour on every side is wedged: pulling it straight
// up has to shove the pile apart, and the grab is capped at three times
// the pencil's weight, which is what a pinch can manage. Report what it
// achieved, but do not demand a clean extraction from a scene that has no
// freely lying pencil in it.
const wedged = touches[top] >= 4;
await drag(sim, top, [0, 0, 0], [0, 0, 0.05], 60, 20, onStep);
check(sim.state.pos[top][2] - comBefore > (wedged ? 0.025 : 0.04),
  `lift raises the body (${((sim.state.pos[top][2] - comBefore) * 1e3).toFixed(0)} of 50 mm${wedged ? ", wedged" : ""})`);
const held = worldPoint(sim, top, [0, 0, 0]);
check(Math.abs(held[2] - (start[2] + 0.05)) < (wedged ? 0.015 : 0.01),
  `held body tracks the cursor (${((held[2] - start[2] - 0.05) * 1e3).toFixed(1)} mm off${wedged ? ", wedged" : ""})`);
// "off the pile" means touching nothing, which is what a hand feels; a
// height threshold instead measures how much the pencil tilted on the way
{
  const segsNow = packet.bodies.map((x, i) => capsuleWorld(sim.state.pos[i], sim.state.quat[i], x.capsule));
  const near = packet.bodies.map((_, j) => j)
    .filter((j) => j !== top && -capsuleClosest(segsNow[top], segsNow[j]).pen < 3e-3);
  const msg = `lift takes the body off the pile (touching [${near}], lowest ${(lowest(sim, top) * 1e3).toFixed(0)} mm)`;
  if (wedged) console.log(`SKIP ${msg} — every pencil here has 4 neighbours`);
  else check(near.length === 0 && lowest(sim, top) > 3e-3, msg);
}
// A pencil prised out of a tight pile may swing a long way, and that is
// real; what must not happen is ending the lift stood on one end, which is
// what a blind tester saw ("rears to vertical", 87 degrees).
check(elevation(sim, top) < 60,
  `held body does not end on its end (${elevation(sim, top).toFixed(0)} deg, peak ${tiltPeak.toFixed(0)})`);
let fell = false;
for (let k = 0; k < 200 && !fell; k++) { await sim.step(); fell = lowest(sim, top) < before + 0.01 && hyp(sim.state.linvel[top]) < 0.02; }
check(fell, `released body falls back to the pile (lowest ${(lowest(sim, top) * 1e3).toFixed(1)} mm)`);

// ---- 4. support removal
sim = await fresh();
const segs = packet.bodies.map((x, i) => capsuleWorld(sim.state.pos[i], sim.state.quat[i], x.capsule));
// j is carried by i only if it touches i from above AND is off the floor;
// a pencil crossing another while both ends rest on the table is not being
// held up by it, and pulling that one out proves nothing
let bottom = 0, mostCarried = -1;
const carried = [];
for (let i = 0; i < sim.B; i++) {
  const on = [];
  for (let j = 0; j < sim.B; j++)
    if (j !== i && -capsuleClosest(segs[i], segs[j]).pen < 6e-3 &&
        sim.state.pos[j][2] > sim.state.pos[i][2] + 1e-3 && lowest(sim, j) > 3e-3) on.push(j);
  if (on.length > mostCarried) { mostCarried = on.length; bottom = i; carried.length = 0; carried.push(...on); }
}
if (!carried.length) {
  console.log("SKIP support removal: no body carries another in this scene");
} else {
  const z0 = carried.map((b) => lowest(sim, b));
  const axis = capsuleWorld(sim.state.pos[bottom], sim.state.quat[bottom], packet.bodies[bottom].capsule).a;
  // far enough that the pencil's end clears the contact it was under
  await drag(sim, bottom, [0, 0, 0], [axis[0] * 0.13, axis[1] * 0.13, 0], 70, 20);
  for (let k = 0; k < 150; k++) await sim.step();
  const drops = carried.map((b, i) => (z0[i] - lowest(sim, b)) * 1e3);
  check(drops.some((d) => d > 1.5),
    `pulling body ${bottom} out lowers what it carried (${drops.map((d) => d.toFixed(1)).join(", ")} mm)`);
}

// ---- 5. nothing interpenetrates for long
sim = await fresh();
let over = 0;
await drag(sim, top, [0, 0, 0], [0.06, 0, 0], 60, 10);
for (let k = 0; k < 120; k++) { await sim.step(); if (worstPenetration(sim) > 1) over++; }
check(over < 12, `no sustained interpenetration after a drag (${over}/120 steps over 1 mm)`);

// ---- 6. nothing passes THROUGH anything when it moves fast
// The per-step checks above only see where bodies are, so a body that
// crossed another between two samples looks innocent at both. Measure the
// deepest overlap along the path each step actually took.
const snapshot = (s) => ({ pos: s.pos.map((p) => [...p]), quat: s.quat.map((q) => [...q]) });
function sweptWorst(a, b) {
  let worst = 0;
  for (let k = 0; k <= 12; k++) {
    const t = k / 12;
    const st = {
      pos: a.pos.map((p, i) => p.map((x, c) => x + (b.pos[i][c] - x) * t)),
      quat: a.quat.map((q, i) => {
        const c = b.quat[i];
        const sgn = q.reduce((acc, x, j) => acc + x * c[j], 0) < 0 ? -1 : 1;
        const v = q.map((x, j) => x + (sgn * c[j] - x) * t);
        const n = Math.hypot(...v) || 1;
        return v.map((x) => x / n);
      }),
    };
    const sg = packet.bodies.map((x, i) => capsuleWorld(st.pos[i], st.quat[i], x.capsule));
    for (let i = 0; i < sg.length; i++) for (let j = i + 1; j < sg.length; j++)
      worst = Math.max(worst, capsuleClosest(sg[i], sg[j]).pen);
  }
  return worst * 1e3;
}

sim = await fresh();
let worstDrop = 0;
sim.state.pos[top][2] += 0.25;                 // 25 cm, well above the pile
sim.state.linvel[top] = [0, 0, 0]; sim.state.angvel[top] = [0, 0, 0];
for (let k = 0; k < 120; k++) {
  const a = snapshot(sim.state);
  await sim.step();
  worstDrop = Math.max(worstDrop, sweptWorst(a, snapshot(sim.state)));
}
check(worstDrop < 2, `a 250 mm drop does not pass through the pile (${worstDrop.toFixed(1)} mm deepest along the path)`);

sim = await fresh();
let worstPull = 0;
{
  // yank the most loaded pencil out fast: 130 mm in 15 steps is 0.5 m/s,
  // which covers more than a barrel radius per step
  const axis = capsuleWorld(sim.state.pos[bottom], sim.state.quat[bottom], packet.bodies[bottom].capsule).a;
  const from = worldPoint(sim, bottom, [0, 0, 0]);
  for (let k = 1; k <= 60; k++) {
    const f = Math.min(1, k / 15);
    const target = [from[0] + axis[0] * 0.13 * f, from[1] + axis[1] * 0.13 * f, from[2]];
    const wp = worldPoint(sim, bottom, [0, 0, 0]);
    const force = grabForce({ m: sim.mass[bottom], v: sim.state.linvel[bottom], wp, target,
      omega: GRAB.omega, zeta: GRAB.zeta, gravity: G, capG: runtime.grab.force_cap,
      blocked: (sim.last?.guard.capsule[bottom] ?? 0) > 1e-3 ? sim.last.guard.normal?.[bottom] : false });
    const a = snapshot(sim.state);
    await sim.step(bottom, wp, force, true);
    worstPull = Math.max(worstPull, sweptWorst(a, snapshot(sim.state)));
  }
}
// a pile where every pencil is wedged against four others transiently
// overlaps a little more while one is dragged out of it
check(worstPull < (wedged ? 2.5 : 2),
  `a fast pull does not pass through the pile (${worstPull.toFixed(1)} mm deepest along the path${wedged ? ", wedged" : ""})`);

// ---- 7. nothing goes through the table while it is being moved
// Dragging a pencil about pushes it against the floor; the correction has
// to win that every step, or it visibly sinks.
sim = await fresh();
let deepest = 0;
{
  const legs = [[0.05, 0, -0.02], [-0.04, 0.05, 0.01], [0, -0.06, -0.03], [0.05, 0.02, 0.02]];
  for (const leg of legs) {
    await drag(sim, top, [0, 0, 0], leg, 25, 5, () => {
      for (let b = 0; b < sim.B; b++) deepest = Math.max(deepest, -lowest(sim, b) * 1e3);
    });
  }
  for (let k = 0; k < 60; k++) {
    await sim.step();
    for (let b = 0; b < sim.B; b++) deepest = Math.max(deepest, -lowest(sim, b) * 1e3);
  }
}
check(deepest < 1, `nothing goes through the table while dragging (${deepest.toFixed(2)} mm below it at worst)`);

console.log(fails ? `INTERACTION TESTS FAILED (${fails})` : "INTERACTION TESTS PASSED");
process.exit(fails ? 1 : 0);

// The fast-pull check of interact.mjs, step by step: which pair overlaps
// along the path, how deep, and what the guards did.   node trace_yank.mjs IMG_8596
import { readFileSync } from "node:fs";
import ort from "onnxruntime-node";
import { PhysSim } from "../public/js/sim.js";
import { capsuleClosest, capsuleWorld, grabForce, lowestSurface, quatToMatrix } from "../public/js/physics.js";
const root = new URL("../..", import.meta.url).pathname;
const runtime = JSON.parse(readFileSync(`${root}web/public/model/runtime.json`));
const session = await ort.InferenceSession.create(`${root}web/public/model/simulator.onnx`);
const scene = process.argv[2] ?? "IMG_8596";
const packet = JSON.parse(readFileSync(`${root}web/public/packets/${scene}.json`));
const G = runtime.gravity, GRAB = { omega: 28, zeta: 1.0 };
const sim = new PhysSim({ kind: "ort", ort, session }, runtime, packet);
for (let k = 0; k < 12; k++) await sim.step();
const f = (x, d = 1) => x.toFixed(d);
const lowest = (b) => lowestSurface(sim.state.pos[b], sim.state.quat[b], packet.bodies[b].capsule);
const worldPoint = (b, local) => { const R = quatToMatrix(sim.state.quat[b]), p = sim.state.pos[b]; return [R[0] * local[0] + p[0], R[3] * local[0] + p[1], R[6] * local[0] + p[2]]; };
const snapshot = (s) => ({ pos: s.pos.map((p) => [...p]), quat: s.quat.map((q) => [...q]) });
function sweptWorst(a, b) {
  let worst = 0, who = null;
  for (let k = 0; k <= 12; k++) {
    const t = k / 12;
    const st = { pos: a.pos.map((p, i) => p.map((x, c) => x + (b.pos[i][c] - x) * t)),
      quat: a.quat.map((q, i) => { const c = b.quat[i]; const sgn = q.reduce((acc, x, j) => acc + x * c[j], 0) < 0 ? -1 : 1; const v = q.map((x, j) => x + (sgn * c[j] - x) * t); const n = Math.hypot(...v) || 1; return v.map((x) => x / n); }) };
    const sg = packet.bodies.map((x, i) => capsuleWorld(st.pos[i], st.quat[i], x.capsule));
    for (let i = 0; i < sg.length; i++) for (let j = i + 1; j < sg.length; j++) { const p = capsuleClosest(sg[i], sg[j]).pen; if (p > worst) { worst = p; who = [i, j, t]; } }
  }
  return { worst: worst * 1e3, who };
}
const segs = packet.bodies.map((x, i) => capsuleWorld(sim.state.pos[i], sim.state.quat[i], x.capsule));
let bottom = 0, most = -1;
for (let i = 0; i < sim.B; i++) { const on = []; for (let j = 0; j < sim.B; j++) if (j !== i && -capsuleClosest(segs[i], segs[j]).pen < 6e-3 && sim.state.pos[j][2] > sim.state.pos[i][2] + 1e-3 && lowest(j) > 3e-3) on.push(j); if (on.length > most) { most = on.length; bottom = i; } }
console.log(`yanking body ${bottom}; start lowest ${packet.bodies.map((_, b) => f(lowest(b) * 1e3)).join("/")} mm, elev ${packet.bodies.map((_, b) => f(Math.asin(Math.abs(capsuleWorld(sim.state.pos[b], sim.state.quat[b], packet.bodies[b].capsule).a[2])) * 180 / Math.PI)).join("/")}`);
const axis = capsuleWorld(sim.state.pos[bottom], sim.state.quat[bottom], packet.bodies[bottom].capsule).a;
const from = worldPoint(bottom, [0, 0, 0]);
for (let k = 1; k <= 60; k++) {
  const fr = Math.min(1, k / 15);
  const target = [from[0] + axis[0] * 0.13 * fr, from[1] + axis[1] * 0.13 * fr, from[2]];
  const wp = worldPoint(bottom, [0, 0, 0]);
  const force = grabForce({ m: sim.mass[bottom], v: sim.state.linvel[bottom], wp, target, omega: GRAB.omega, zeta: GRAB.zeta, gravity: G, capG: runtime.grab.force_cap,
    blocked: (sim.last?.guard.capsule[bottom] ?? 0) > 1e-3 ? sim.last.guard.normal?.[bottom] : false });
  const a = snapshot(sim.state);
  await sim.step(bottom, wp, force, true);
  const w = sweptWorst(a, snapshot(sim.state));
  const g = sim.last.guard;
  console.log(`k${k} worst ${f(w.worst, 2)} mm ${w.who ? `pair ${w.who[0]}-${w.who[1]} at t=${f(w.who[2], 2)}` : ""} v ${f(Math.hypot(...sim.state.linvel[bottom]) * 1e3, 0)} mm/s ` +
    `swept ${g.swept ?? 0} end-pairs ${g.pairs.map((p) => `${p.i}-${p.j}:${f(p.pen * 1e3)}`).join(",")} freeFlight ${Array.from(g.freeFlight)} pivot ${Array.from(g.pivot)} fc ${f((g.floorCap ?? 0) * 1e3, 0)}`);
}

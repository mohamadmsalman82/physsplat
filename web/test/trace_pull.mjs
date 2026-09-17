// Trace the support-removal and hold-still checks of interact.mjs step by
// step, with every rule's verdict on every body.
//   node trace_pull.mjs IMG_8504 pull     node trace_pull.mjs IMG_8626 hold
import { readFileSync } from "node:fs";
import ort from "onnxruntime-node";
import { PhysSim } from "../public/js/sim.js";
import { capsuleClosest, capsuleWorld, grabForce, lowestSurface, quatToMatrix, supportPoints, supportAnalysis } from "../public/js/physics.js";
const root = new URL("../..", import.meta.url).pathname;
const runtime = JSON.parse(readFileSync(`${root}web/public/model/runtime.json`));
const session = await ort.InferenceSession.create(`${root}web/public/model/simulator.onnx`);
const scene = process.argv[2] ?? "IMG_8504", mode = process.argv[3] ?? "pull";
const packet = JSON.parse(readFileSync(`${root}web/public/packets/${scene}.json`));
const G = runtime.gravity, GRAB = { omega: 28, zeta: 1.0 };
const sim = new PhysSim({ kind: "ort", ort, session }, runtime, packet);
for (let k = 0; k < 12; k++) await sim.step();
const f = (x, d = 1) => x.toFixed(d);
const hyp = (v) => Math.hypot(v[0], v[1], v[2]);
const lowest = (b) => lowestSurface(sim.state.pos[b], sim.state.quat[b], packet.bodies[b].capsule);
const worldPoint = (b, local) => { const R = quatToMatrix(sim.state.quat[b]), p = sim.state.pos[b]; return [R[0] * local[0] + p[0], R[3] * local[0] + p[1], R[6] * local[0] + p[2]]; };
const elevation = (b) => Math.asin(Math.min(1, Math.abs(capsuleWorld(sim.state.pos[b], sim.state.quat[b], packet.bodies[b].capsule).a[2]))) * 180 / Math.PI;
function row(b, held) {
  const g = sim.last.guard, parts = sim.particlesWorld();
  const segs = packet.bodies.map((x, i) => capsuleWorld(sim.state.pos[i], sim.state.quat[i], x.capsule));
  let s = 0, start = 0; for (let i = 0; i < b; i++) s += sim.counts[i]; start = s;
  const sp = supportPoints(parts, start, sim.counts[b], segs, b, 3e-3, 6e-3);
  const an = supportAnalysis(sim.state.pos[b], sp.points, 2e-3, sp.gaps);
  return `b${b}${b === held ? "*" : " "} low ${f(lowest(b) * 1e3)} el ${f(elevation(b))} v ${f(hyp(sim.state.linvel[b]) * 1e3, 0)} w ${f(hyp(sim.state.angvel[b]), 2)} ` +
    `sup ${an.n}(f${sp.floor} c${sp.capsule}) d ${an.dist === Infinity ? "inf" : f(an.dist * 1e3)} hg ${f(an.hingeGap * 1e3)} ` +
    `gnd ${sim.groundedNow?.[b]} ${g.pivot[b] ? "PIV " : ""}${g.freeFlight[b] ? "FREE " : ""}${g.settled[b] ? "SET " : ""}` +
    `rz ${f(sim.last.residual[6 * b + 2])} ra ${f(Math.hypot(sim.last.residual[6 * b + 3], sim.last.residual[6 * b + 4], sim.last.residual[6 * b + 5]), 0)} gd ${f(g.ground[b] * 1e3, 2)} cp ${f(g.capsule[b] * 1e3, 2)} seat ${f((sim.seatGap?.[b] ?? 0) * 1e3)} ft ${f(g.floorTorque ?? 0, 0)} fc ${f((g.floorCap ?? 0) * 1e3, 0)}`;
}
async function drag(body, local, to, steps, hold, every = 1) {
  const from = worldPoint(body, local);
  for (let k = 1; k <= steps + hold; k++) {
    const fr = Math.min(1, k / steps);
    const target = [from[0] + to[0] * fr, from[1] + to[1] * fr, from[2] + to[2] * fr];
    const wp = worldPoint(body, local);
    const force = grabForce({ m: sim.mass[body], v: sim.state.linvel[body], wp, target, omega: GRAB.omega, zeta: GRAB.zeta, gravity: G, capG: runtime.grab.force_cap,
      blocked: (sim.last?.guard.capsule[body] ?? 0) > 1e-3 ? sim.last.guard.normal?.[body] : false });
    await sim.step(body, wp, force, true);
    if (k % every === 0) console.log(`drag ${k}\n  ` + packet.bodies.map((_, b) => row(b, body)).join("\n  "));
  }
}
if (mode === "pull") {
  const segs = packet.bodies.map((x, i) => capsuleWorld(sim.state.pos[i], sim.state.quat[i], x.capsule));
  let bottom = 0, most = -1; const carried = [];
  for (let i = 0; i < sim.B; i++) {
    const on = [];
    for (let j = 0; j < sim.B; j++) if (j !== i && -capsuleClosest(segs[i], segs[j]).pen < 6e-3 && sim.state.pos[j][2] > sim.state.pos[i][2] + 1e-3 && lowest(j) > 3e-3) on.push(j);
    if (on.length > most) { most = on.length; bottom = i; carried.length = 0; carried.push(...on); }
  }
  console.log(`bottom ${bottom} carries ${carried}; start lowest ${carried.map((b) => f(lowest(b) * 1e3))}`);
  const axis = capsuleWorld(sim.state.pos[bottom], sim.state.quat[bottom], packet.bodies[bottom].capsule).a;
  await drag(bottom, [0, 0, 0], [axis[0] * 0.13, axis[1] * 0.13, 0], 70, 20, 10);
  for (let k = 0; k < 150; k++) { await sim.step(); if (k % 5 === 0) console.log(`after ${k}\n  ` + packet.bodies.map((_, b) => row(b, -1)).join("\n  ")); }
  console.log(`end lowest ${carried.map((b) => f(lowest(b) * 1e3))}`);
} else {
  const touches = packet.bodies.map((_, i) => { const segs = packet.bodies.map((x, j) => capsuleWorld(sim.state.pos[j], sim.state.quat[j], x.capsule)); return packet.bodies.filter((_, j) => j !== i && -capsuleClosest(segs[i], segs[j]).pen < 6e-3).length; });
  const segs0 = packet.bodies.map((x, i) => capsuleWorld(sim.state.pos[i], sim.state.quat[i], x.capsule));
  const carries = packet.bodies.map((_, i) => packet.bodies.some((_, j) => { if (j === i) return false; const c = capsuleClosest(segs0[i], segs0[j]); return -c.pen < 6e-3 && (c.n[2] < -0.2 || sim.state.pos[j][2] > sim.state.pos[i][2]); }));
  const pool = packet.bodies.map((_, i) => i).filter((i) => !carries[i]);
  const top = process.argv[4] !== undefined ? Number(process.argv[4]) : (pool.length ? pool : packet.bodies.map((_, i) => i)).reduce((a, b) => (touches[b] < touches[a] || (touches[b] === touches[a] && sim.state.pos[b][2] > sim.state.pos[a][2]) ? b : a));
  console.log(`holding body ${top} (touches ${touches[top]})`);
  await drag(top, [0, 0, 0], [0, 0, 0], 1, 120, 4);
}

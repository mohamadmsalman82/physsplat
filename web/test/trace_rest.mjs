// Why does a settled packet move at rest? Per-step trace of every body's
// rule verdicts for the first seconds.  node trace_rest.mjs IMG_8504 120
import { readFileSync } from "node:fs";
import ort from "onnxruntime-node";
import { PhysSim } from "../public/js/sim.js";
import { capsuleWorld, lowestSurface, supportPoints, supportAnalysis } from "../public/js/physics.js";
const root = new URL("../..", import.meta.url).pathname;
const runtime = JSON.parse(readFileSync(`${root}web/public/model/runtime.json`));
const session = await ort.InferenceSession.create(`${root}web/public/model/simulator.onnx`);
const scene = process.argv[2] ?? "IMG_8504", STEPS = Number(process.argv[3] ?? 120);
const packet = JSON.parse(readFileSync(`${root}web/public/packets/${scene}.json`));
const sim = new PhysSim({ kind: "ort", ort, session }, runtime, packet);
const p0 = sim.state.pos.map((p) => [...p]);
const f = (x, d = 1) => x.toFixed(d);
const pre = () => { const parts = sim.particlesWorld(); const segs = packet.bodies.map((b, i) => capsuleWorld(sim.state.pos[i], sim.state.quat[i], b.capsule)); let s = 0; return packet.bodies.map((_, b) => { const n = sim.counts[b], start = s; s += n; const sp = supportPoints(parts, start, n, segs, b, 1.5e-3, 6e-3); const an = supportAnalysis(sim.state.pos[b], sp.points); return `pre[n${an.n} d${an.dist === Infinity ? "inf" : f(an.dist * 1e3)} w${f(Math.hypot(...sim.state.angvel[b]), 2)}]`; }); };
for (let k = 0; k < STEPS; k++) {
  const pr = pre();
  await sim.step();
  const g = sim.last.guard, parts = sim.particlesWorld();
  const segs = packet.bodies.map((b, i) => capsuleWorld(sim.state.pos[i], sim.state.quat[i], b.capsule));
  let s = 0;
  const rows = [];
  for (let b = 0; b < sim.B; b++) {
    const n = sim.counts[b], start = s; s += n;
    const sp = supportPoints(parts, start, n, segs, b, 1.5e-3, 6e-3);
    const an = supportAnalysis(sim.state.pos[b], sp.points);
    const low = lowestSurface(sim.state.pos[b], sim.state.quat[b], packet.bodies[b].capsule);
    const drift = Math.hypot(sim.state.pos[b][0] - p0[b][0], sim.state.pos[b][1] - p0[b][1]) * 1e3;
    rows.push(`b${b} ${pr[b]} low ${f(low * 1e3)} drift ${f(drift)} v ${f(Math.hypot(...sim.state.linvel[b]) * 1e3, 0)} ` +
      `sup ${an.n}(f${sp.floor} c${sp.capsule}) dist ${an.dist === Infinity ? "inf" : f(an.dist * 1e3)} ` +
      `${g.pivot[b] ? "PIV " : ""}${g.freeFlight[b] ? "FREE " : ""}${g.settled[b] ? "SET " : ""}` +
      `rz ${f(sim.last.residual[6 * b + 2])} gd ${f(g.ground[b] * 1e3, 2)} cp ${f(g.capsule[b] * 1e3, 2)} seat ${f((sim.seatGap?.[b] ?? 0) * 1e3)}`);
  }
  if (k < 200) console.log(`step ${k}\n  ` + rows.join("\n  "));
}

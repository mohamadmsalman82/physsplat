/**
 * A tangle in the air must fall.
 *
 * Every rule in sim.js asks a local question about a body: is there
 * something under it within the contact radius? The learned model answers
 * "+g, hold still" for any body with a contact beneath it. Neither asks
 * whether the thing beneath is itself standing on anything. Three pencils
 * woven so each rests on the next, with none touching the table, satisfy
 * every local test and hang in the air. This constructs that tangle at a
 * height and checks that it comes down.
 *
 *   node web/test/ring.mjs
 */
import { readFileSync } from "node:fs";
import ort from "onnxruntime-node";
import { PhysSim } from "../public/js/sim.js";
import { lowestSurface, capsuleWorld, capsuleClosest } from "../public/js/physics.js";

const root = new URL("../..", import.meta.url).pathname;
const runtime = JSON.parse(readFileSync(`${root}web/public/model/runtime.json`));
const session = await ort.InferenceSession.create(`${root}web/public/model/simulator.onnx`);
const packet = JSON.parse(readFileSync(`${root}web/public/packets/IMG_8626.json`));   // five bodies to borrow

const quatFromXTo = (d) => {           // rotate body x-axis onto unit d
  const x = [1, 0, 0];
  const c = x[0] * d[0] + x[1] * d[1] + x[2] * d[2];
  const ax = [x[1] * d[2] - x[2] * d[1], x[2] * d[0] - x[0] * d[2], x[0] * d[1] - x[1] * d[0]];
  const s = Math.hypot(...ax);
  if (s < 1e-9) return [0, 0, 0, 1];
  const ang = Math.atan2(s, c);
  return [ax[0] / s * Math.sin(ang / 2), ax[1] / s * Math.sin(ang / 2), ax[2] / s * Math.sin(ang / 2), Math.cos(ang / 2)];
};

/** n pencils woven into a regular polygon of side L, each resting on the
 * next, hanging with the lowest surface `z0` above the table. */
function weave(n, L, z0) {
  const bodies = [];
  const Rc = L / (2 * Math.sin(Math.PI / n));   // circumradius
  const rise = 0.0095;                           // one pencil diameter of climb per edge
  for (let i = 0; i < n; i++) {
    const a0 = (2 * Math.PI * i) / n, a1 = (2 * Math.PI * (i + 1)) / n;
    const v0 = [Rc * Math.cos(a0), Rc * Math.sin(a0)], v1 = [Rc * Math.cos(a1), Rc * Math.sin(a1)];
    const d = [v1[0] - v0[0], v1[1] - v0[1], rise];
    const dn = Math.hypot(...d);
    const u = d.map((x) => x / dn);
    const mid = [(v0[0] + v1[0]) / 2, (v0[1] + v1[1]) / 2, z0 + 0.0045 + rise / 2];
    bodies.push({ pos: mid, quat: quatFromXTo(u) });
  }
  return bodies;
}

let fails = 0;
const check = (ok, msg) => { console.log(`${ok ? "PASS" : "FAIL"} ${msg}`); if (!ok) fails++; };

for (const [n, z0] of [[3, 0.05], [4, 0.04], [3, 0.012]]) {
  const p = { name: `ring${n}`, bodies: packet.bodies.slice(0, n).map((b) => ({ ...b })), settled: { steps: 0, at_rest: false } };
  const w = weave(n, 0.09, z0);
  p.bodies.forEach((b, i) => { b.pos = w[i].pos; b.quat = w[i].quat; });
  const sim = new PhysSim({ kind: "ort", ort, session }, runtime, p);
  const low = () => p.bodies.map((b, i) => lowestSurface(sim.state.pos[i], sim.state.quat[i], b.capsule) * 1e3);
  const segs = () => p.bodies.map((b, i) => capsuleWorld(sim.state.pos[i], sim.state.quat[i], b.capsule));
  const s0 = segs();
  const gaps = [];
  for (let i = 0; i < n; i++) gaps.push(-capsuleClosest(s0[i], s0[(i + 1) % n]).pen * 1e3);
  console.log(`ring of ${n} at ${(z0 * 1e3).toFixed(0)} mm: start lowest ${low().map((x) => x.toFixed(1)).join("/")} mm, ` +
    `neighbour gaps ${gaps.map((x) => x.toFixed(1)).join("/")} mm`);
  const track = [];
  for (let k = 0; k < 240; k++) {
    await sim.step();
    if (k % 30 === 29) track.push(low().map((x) => x.toFixed(1)).join("/"));
  }
  const L = low();
  const guard = sim.last.guard;
  console.log(`  lowest every 0.5 s: ${track.join("  ")}`);
  console.log(`  after 4 s: settled ${Array.from(guard.settled)}, freeFlight ${Array.from(guard.freeFlight)}, ` +
    `pivot ${Array.from(guard.pivot)}, resZ ${Array.from({ length: n }, (_, b) => sim.last.residual[6 * b + 2].toFixed(1))}`);
  check(Math.min(...L) < 1.5, `ring of ${n} reaches the table (lowest ${Math.min(...L).toFixed(1)} mm)`);
  check(Math.max(...L) < 15, `ring of ${n}: every pencil comes down (highest lowest-point ${Math.max(...L).toFixed(1)} mm)`);
}
console.log(fails ? `RING TESTS FAILED (${fails})` : "RING TESTS PASSED");
process.exit(fails ? 1 : 0);

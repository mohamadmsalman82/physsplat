/**
 * A pencil placed a few millimetres above its support must come down.
 *
 * The model senses contact through a 6 mm radius and its response to a
 * body in that band can hold it there (the "hover" of several player
 * reports). These place a pencil at rest inside the band, above the desk
 * and above another pencil, and require it to reach contact within half a
 * second and stay there.
 *
 *   node web/test/hover.mjs
 */
import { readFileSync } from "node:fs";
import ort from "onnxruntime-node";
import { PhysSim } from "../public/js/sim.js";
import { capsuleClosest, capsuleWorld, lowestSurface, quatFromRotvec } from "../public/js/physics.js";

const root = new URL("../..", import.meta.url).pathname;
const runtime = JSON.parse(readFileSync(`${root}web/public/model/runtime.json`));
const session = await ort.InferenceSession.create(`${root}web/public/model/simulator.onnx`);
const packet = JSON.parse(readFileSync(`${root}web/public/packets/IMG_8626.json`));

let fails = 0;
const check = (ok, msg) => { console.log(`${ok ? "PASS" : "FAIL"} ${msg}`); if (!ok) fails++; };
const yaw = (a) => quatFromRotvec([0, 0, a]);

async function run(name, bodies, watch) {
  const p = { name, bodies: bodies.map((b, i) => ({ ...packet.bodies[i], pos: b.pos, quat: b.quat })), settled: { steps: 0, at_rest: false } };
  const sim = new PhysSim({ kind: "ort", ort, session }, runtime, p);
  const track = [];
  let reached = -1;
  for (let k = 0; k < 90; k++) {
    await sim.step();
    const gap = watch(sim, p);
    track.push(gap);
    if (reached < 0 && gap < 0.5e-3) reached = k;
  }
  const late = track.slice(45);
  const settledLow = Math.max(...late);
  console.log(`${name}: gap every 10 steps ${track.filter((_, i) => i % 10 === 0).map((x) => (x * 1e3).toFixed(1)).join("/")} mm; ` +
    `contact at step ${reached}, worst gap after 0.75 s ${(settledLow * 1e3).toFixed(2)} mm`);
  check(reached >= 0 && reached <= 30, `${name}: comes down within half a second (step ${reached})`);
  check(settledLow < 1e-3, `${name}: stays down (worst gap after 0.75 s ${(settledLow * 1e3).toFixed(2)} mm)`);
}

const r = packet.bodies[0].capsule.radius;
for (const h of [3.4e-3, 5.5e-3]) {
  await run(`lone pencil ${(h * 1e3).toFixed(1)} mm above the desk`,
    [{ pos: [0, 0, r + h], quat: [0, 0, 0, 1] }],
    (sim, p) => lowestSurface(sim.state.pos[0], sim.state.quat[0], p.bodies[0].capsule));
}
for (const h of [3e-3, 5e-3]) {
  // one pencil flat on the desk along x, another across it along y, centred
  // on the first, lying h above it
  await run(`pencil ${(h * 1e3).toFixed(1)} mm above a crossing pencil`,
    [{ pos: [0, 0, r], quat: [0, 0, 0, 1] }, { pos: [0, 0, 2 * r + h + 0.001], quat: yaw(Math.PI / 2) }],
    (sim, p) => {
      const a = capsuleWorld(sim.state.pos[0], sim.state.quat[0], p.bodies[0].capsule);
      const b = capsuleWorld(sim.state.pos[1], sim.state.quat[1], p.bodies[1].capsule);
      return Math.max(0, -capsuleClosest(b, a).pen);
    });
}
console.log(fails ? `HOVER TESTS FAILED (${fails})` : "HOVER TESTS PASSED");
process.exit(fails ? 1 : 0);

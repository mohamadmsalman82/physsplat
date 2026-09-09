/**
 * Rest regression: an untouched photo scene must stay put.
 *
 * The full browser sim (all analytic rules on, as the demo runs it) is
 * stepped for 10 s on every packet with no action at all. Nothing may rise,
 * rotate up, drift, or sink. This is the browser-free guard against the two
 * failures the diagnostics caught by hand: a resting pencil rearing up to
 * 89 degrees on its own, and a settled pile creeping sideways.
 *
 *   node web/test/rest.mjs
 */
import { readdirSync, readFileSync } from "node:fs";
import ort from "onnxruntime-node";
import { PhysSim } from "../public/js/sim.js";
import { capsuleWorld, quatToMatrix } from "../public/js/physics.js";

const root = new URL("../..", import.meta.url).pathname;
const runtime = JSON.parse(readFileSync(`${root}web/public/model/runtime.json`));
const session = await ort.InferenceSession.create(`${root}web/public/model/simulator.onnx`);

const STEPS = Number(process.env.REST_STEPS ?? 600);      // 10 s at 60 Hz
const LIMITS = { rise_mm: 2, tilt_deg: 3, drift_mm: 3, sink_mm: 1 };

const elevation = (sim, b) => {
  const a = capsuleWorld(sim.state.pos[b], sim.state.quat[b], sim.packet.bodies[b].capsule).a;
  return Math.asin(Math.min(1, Math.abs(a[2]))) * 180 / Math.PI;
};
const lowest = (sim, b) => {
  const R = quatToMatrix(sim.state.quat[b]);
  let z = Infinity;
  for (const o of sim.offsets[b])
    z = Math.min(z, R[6] * o[0] + R[7] * o[1] + R[8] * o[2] + sim.state.pos[b][2]);
  return z;
};

let fails = 0;
const names = readdirSync(`${root}web/public/packets`)
  .filter((f) => f.endsWith(".json") && f !== "index.json").sort();

for (const file of names) {
  const packet = JSON.parse(readFileSync(`${root}web/public/packets/${file}`));
  // guards on as the demo runs them; the two rejected filters stay off
  // unless REST_ENERGY / REST_SMOOTH ask for them
  const sim = new PhysSim({ kind: "ort", ort, session }, runtime, packet,
    { energyRule: process.env.REST_ENERGY === "1", smooth: process.env.REST_SMOOTH === "1" });
  // the page hides the first steps while the reconstruction settles
  for (let k = 0; k < 45; k++) await sim.step();
  const p0 = sim.state.pos.map((p) => [...p]);
  const e0 = packet.bodies.map((_, b) => elevation(sim, b));
  const worst = { rise: 0, tilt: 0, drift: 0, sink: 0, body: -1 };
  for (let k = 0; k < STEPS; k++) {
    await sim.step();
    for (let b = 0; b < sim.B; b++) {
      const rise = (sim.state.pos[b][2] - p0[b][2]) * 1e3;
      const tilt = Math.abs(elevation(sim, b) - e0[b]);
      const drift = Math.hypot(sim.state.pos[b][0] - p0[b][0],
        sim.state.pos[b][1] - p0[b][1]) * 1e3;
      const sink = -lowest(sim, b) * 1e3;
      if (rise > worst.rise || tilt > worst.tilt || drift > worst.drift) worst.body = b;
      worst.rise = Math.max(worst.rise, rise);
      worst.tilt = Math.max(worst.tilt, tilt);
      worst.drift = Math.max(worst.drift, drift);
      worst.sink = Math.max(worst.sink, sink);
    }
  }
  const ok = worst.rise <= LIMITS.rise_mm && worst.tilt <= LIMITS.tilt_deg &&
    worst.drift <= LIMITS.drift_mm && worst.sink <= LIMITS.sink_mm;
  if (!ok) fails++;
  console.log(`${ok ? "PASS" : "FAIL"} ${packet.name}: rise ${worst.rise.toFixed(1)} mm, ` +
    `tilt ${worst.tilt.toFixed(1)} deg, drift ${worst.drift.toFixed(1)} mm, ` +
    `sink ${worst.sink.toFixed(1)} mm (worst body ${worst.body})`);
}
console.log(fails ? `REST TESTS FAILED (${fails})` : "REST TESTS PASSED");
process.exit(fails ? 1 : 0);

/**
 * Settle each scene packet with the real model, once, and write the result
 * back into the packet.
 *
 * Two problems this solves at the same time.
 *
 * The packet's poses come from a photograph, and the bodies are now the
 * canonical 150 mm pencil rather than whatever length the reconstruction
 * measured, so a pile that was consistent as reconstructed is not quite
 * consistent as built. The pipeline's geometric settle pushes overlaps
 * apart, but only the learned model knows where the pile actually comes to
 * rest.
 *
 * And the demo used to do this at load: 180 steps of hidden pre-roll before
 * the first frame anyone sees. At 35 to 60 ms a step that is 6 to 12 seconds
 * of a frozen, visibly wrong pile every time the page opens or reset is
 * pressed, ending in every pencil snapping up to 26 mm into place. Doing it
 * here means the packet ships already at rest and the page can draw the
 * first frame immediately.
 *
 *   node web/test/settle_packets.mjs [scene ...]
 */
import { readFileSync, writeFileSync } from "node:fs";
import ort from "onnxruntime-node";
import { PhysSim } from "../public/js/sim.js";
import { lowestSurface } from "../public/js/physics.js";

const root = new URL("../..", import.meta.url).pathname;
const runtime = JSON.parse(readFileSync(`${root}web/public/model/runtime.json`));
const session = await ort.InferenceSession.create(`${root}web/public/model/simulator.onnx`);

const STEPS = Number(process.env.SETTLE_STEPS ?? 900);   // 15 s
const scenes = process.argv.slice(2).length
  ? process.argv.slice(2)
  : ["IMG_8504", "IMG_8513", "IMG_8596", "IMG_8626"];

const r5 = (a) => a.map((x) => Math.round(x * 1e5) / 1e5);
const r6 = (a) => a.map((x) => Math.round(x * 1e6) / 1e6);

for (const scene of scenes) {
  const file = `${root}web/public/packets/${scene}.json`;
  const packet = JSON.parse(readFileSync(file));
  const sim = new PhysSim({ kind: "ort", ort, session }, runtime, packet);

  const before = sim.state.pos.map((p) => [...p]);
  for (let k = 0; k < STEPS; k++) await sim.step();

  const moved = sim.state.pos.map((p, i) =>
    Math.hypot(p[0] - before[i][0], p[1] - before[i][1], p[2] - before[i][2]) * 1e3);
  const speed = sim.state.linvel.map((v) => Math.hypot(...v) * 1e3);
  const spin = sim.state.angvel.map((v) => Math.hypot(...v));
  const floor = packet.bodies.map((b, i) =>
    lowestSurface(sim.state.pos[i], sim.state.quat[i], b.capsule) * 1e3);
  const resting = Array.from(sim.restCount).every((c) => c > 30);

  packet.bodies.forEach((b, i) => {
    b.pos = r5(sim.state.pos[i]);
    b.quat = r6(sim.state.quat[i]);
  });
  packet.settled = { steps: STEPS, at_rest: resting };
  writeFileSync(file, JSON.stringify(packet));

  console.log(
    `${scene}: settled ${STEPS} steps, moved ${moved.map((x) => x.toFixed(1)).join("/")} mm, ` +
    `speed ${speed.map((x) => x.toFixed(2)).join("/")} mm/s, ` +
    `spin ${spin.map((x) => x.toFixed(3)).join("/")} rad/s, ` +
    `floor ${floor.map((x) => x.toFixed(2)).join("/")} mm, ` +
    `${resting ? "AT REST" : "NOT AT REST"}`);
}

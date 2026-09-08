/**
 * Headless end-to-end: the FULL JS sim (PhysSim + onnxruntime-node) must
 * reproduce Python LiveSim (running the same ONNX) over a 60-step passive
 * rollout of a real photo packet.
 *
 *   uv run python scripts/export_js_rollout_fixture.py
 *   node web/test/e2e.mjs
 */
import { readFileSync } from "node:fs";
import ort from "onnxruntime-node";
import { PhysSim } from "../public/js/sim.js";

const root = new URL("../..", import.meta.url).pathname;
const runtime = JSON.parse(readFileSync(`${root}web/public/model/runtime.json`));
const fixture = JSON.parse(readFileSync(new URL("./fixtures/rollout.json", import.meta.url)));
const packet = JSON.parse(
  readFileSync(`${root}web/public/packets/${fixture.packet}.json`));

const session = await ort.InferenceSession.create(
  `${root}web/public/model/simulator.onnx`);
const sim = new PhysSim(ort, session, runtime, packet);

let worst = 0;
const t0 = Date.now();
for (let t = 1; t <= fixture.steps; t++) {
  await sim.step();
  const ref = fixture.pos[String(t)];
  if (!ref) continue;
  ref.forEach((p, b) => {
    for (let k = 0; k < 3; k++)
      worst = Math.max(worst, Math.abs(p[k] - sim.state.pos[b][k]));
  });
}
const ms = (Date.now() - t0) / fixture.steps;
console.log(`60-step rollout: max |pos_js - pos_py| = ${worst.toExponential(2)} m`);
console.log(`JS physics: ${ms.toFixed(0)} ms/step (${(1000 / ms).toFixed(0)} Hz capable, node/CPU)`);
if (worst > 5e-3 || !isFinite(worst)) {
  console.log("E2E PARITY FAILED");
  process.exit(1);
}
console.log("E2E PARITY PASSED");

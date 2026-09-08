/**
 * Cross-language parity: the JS runtime must reproduce the Python
 * reference on recorded fixtures.
 *
 *   uv run python scripts/export_web_fixtures.py   (regenerate fixtures)
 *   node web/test/parity.mjs
 */
import { readFileSync } from "node:fs";
import {
  buildEdges, edgeFeatures, externalAccels, stepBodies,
} from "../js/physics.js";

const load = (p) => JSON.parse(readFileSync(new URL(p, import.meta.url)));
let failures = 0;
const check = (name, ok, detail = "") => {
  console.log(`${ok ? "PASS" : "FAIL"} ${name} ${detail}`);
  if (!ok) failures++;
};

// ------------------------------------------------------------- graph
{
  const fx = load("./fixtures/graph.json");
  const parts = Float64Array.from(fx.particles.flat());
  const vels = Float64Array.from(fx.velocities.flat());
  const { senders, receivers } = buildEdges(parts, vels, fx.contact_radius, fx.dt);
  const key = (s, r) => s * 100000 + r;
  const got = new Set([...senders].map((s, e) => key(s, receivers[e])));
  const want = new Set(fx.senders.map((s, e) => key(s, fx.receivers[e])));
  const missing = [...want].filter((k) => !got.has(k)).length;
  const extra = [...got].filter((k) => !want.has(k)).length;
  check("graph edges", missing === 0 && extra === 0,
    `E=${got.size} missing=${missing} extra=${extra}`);

  // edge features on python's edge ordering
  const ef = edgeFeatures(parts, Int32Array.from(fx.senders),
    Int32Array.from(fx.receivers), fx.body_ids, fx.contact_radius);
  let worst = 0;
  fx.edge_feats.flat().forEach((v, i) => {
    worst = Math.max(worst, Math.abs(v - ef[i]));
  });
  check("edge features", worst < 1e-4, `max err ${worst.toExponential(1)}`);
}

// ---------------------------------------------------------- integrator
{
  const fx = load("./fixtures/integrator.json");
  const s = {
    pos: fx.in.pos.map((r) => [...r]),
    quat: fx.in.quat.map((r) => [...r]),
    linvel: fx.in.linvel.map((r) => [...r]),
    angvel: fx.in.angvel.map((r) => [...r]),
  };
  const B = s.pos.length;
  const ext = externalAccels(s.pos, s.quat, fx.in.mass, fx.in.inertia,
    fx.in.act_body, fx.in.act_point, fx.in.act_force, B);
  const residual = Float64Array.from(fx.in.residual.flat());
  stepBodies(s, residual, ext, fx.dt, fx.gravity);
  let worst = 0;
  for (const k of ["pos", "quat", "linvel", "angvel"])
    fx.out[k].flat().forEach((v, i) => {
      const flat = s[k].flat();
      worst = Math.max(worst, Math.abs(v - flat[i]));
    });
  check("integrator step", worst < 1e-6, `max err ${worst.toExponential(1)}`);
}

process.exit(failures ? 1 : 0);

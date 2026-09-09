/**
 * Diagnostics unit test: derived quantities and anomaly detectors on a
 * mock simulator (no network). Run: node web/test/diag.mjs
 */
import { Diagnostics } from "../public/js/diag.js";

const cap = { axis: [1, 0, 0], half: 0.07, radius: 0.006 };
function offsets() {   // a few surface samples of the capsule
  const o = [];
  for (let s = -0.07; s <= 0.0701; s += 0.014)
    for (const [y, z] of [[0, 0.006], [0, -0.006], [0.006, 0], [-0.006, 0]]) o.push([s, y, z]);
  return o;
}
function mockSim(pos) {
  const B = pos.length;
  return {
    B, rt: { dt: 1 / 60, gravity: 9.81, step: 0 }, stepCount: 0, timing: null, loadCorrection: null,
    packet: { name: "mock", bodies: pos.map((p) => ({ capsule: cap, mass: 0.01, inertia: [1e-7, 1e-5, 1e-5], pos: p, quat: [0, 0, 0, 1] })) },
    state: { pos: pos.map((p) => [...p]), quat: pos.map(() => [0, 0, 0, 1]),
      linvel: pos.map(() => [0, 0, 0]), angvel: pos.map(() => [0, 0, 0]) },
    mass: pos.map(() => 0.01), inertia: pos.map(() => [1e-7, 1e-5, 1e-5]), offsets: pos.map(() => offsets()),
    last: null,
  };
}

let fails = 0;
const check = (cond, msg) => { console.log(`${cond ? "PASS" : "FAIL"} ${msg}`); if (!cond) fails++; };

// body 0 lying on the floor, body 1 resting on top of it (crossed would be
// nicer, but parallel capsules 12 mm apart touch exactly), body 2 in the air
const sim = mockSim([[0, 0, 0.006], [0, 0, 0.018], [0.2, 0, 0.05]]);
const d = new Diagnostics(sim, { capacity: 100 });
const f = d.record(null);
check(Math.abs(f.bodies[0].lowest) < 1e-9, "body 0 lowest point on the floor");
check(f.bodies[0].groundContact && !f.bodies[1].groundContact, "ground contact classification");
check(f.bodies[1].contacts.length === 1 && f.bodies[1].contacts[0].other === 0, "body 1 touches body 0");
check(f.bodies[1].supportedBy.includes(0) && f.bodies[0].supports.includes(1), "support relation 0 -> 1");
check(f.bodies[2].unsupported, "body 2 unsupported in the air");
check(Math.abs(f.pairs[0].gap_mm) < 1e-6, "pair 0-1 gap is zero");

// floating detector needs 20 resting steps with no action
for (let k = 0; k < 25; k++) { sim.stepCount++; d.record(null); }
const active = d.check().map((a) => a.key);
check(active.includes("floating:2"), `floating detected for body 2 (${active})`);
check(!active.some((k) => k.startsWith("floating:0") || k.startsWith("floating:1")), "supported bodies not flagged");

// penetration: push body 1 down into body 0 by 4 mm
sim.state.pos[1][2] = 0.014; sim.stepCount++; d.record(null);
check(d.check().some((a) => a.key === "penetration:0-1"), "penetration 0-1 flagged");
check(Math.abs(d.frame(0).bodies[1].penetration_mm - 4) < 1e-6, "penetration depth 4 mm");

// explosion
sim.state.linvel[2] = [5, 0, 0]; sim.stepCount++; d.record(null);
check(d.check().some((a) => a.key === "explosion:2"), "explosion flagged");
sim.state.linvel[2] = [0, 0, 0]; sim.stepCount++; d.record(null);
check(!d.check().some((a) => a.key === "explosion:2"), "explosion episode ends");
check(d.events(100, "anomaly_end").some((e) => e.key === "explosion:2"), "episode end logged");

// creep: body 0 drifts 3 mm while at rest
for (let k = 0; k < 70; k++) { sim.state.pos[0][0] += 0.05e-3; sim.stepCount++; d.record(null); }
check(d.check().some((a) => a.key === "creep:0"), "creep flagged after 3.5 mm drift at rest");

// summary / history / track / export shapes
const s = d.summary(50);
check(s.bodies.length === 3 && typeof s.bodies[0].displacement_mm === "number", "summary per body");
check(d.history(5).length === 5 && d.history(5, ["pos"])[0].bodies[0].pos.length === 3, "history with field filter");
check(d.track(0, 10).z.length === 10, "track series length");
const ex = JSON.parse(d.export({ last: 20 }));
check(ex.frames.length === 20 && ex.bodies.length === 3 && Array.isArray(ex.events), "export parses");

// action bookkeeping: floating must not fire on a held body
const sim2 = mockSim([[0, 0, 0.05]]);
const d2 = new Diagnostics(sim2, { capacity: 100 });
for (let k = 0; k < 30; k++) { sim2.stepCount++; d2.record({ kind: "grab", body: 0, force: [0, 0, 0.1] }); }
check(!d2.check().some((a) => a.type === "floating"), "held body in the air is not 'floating'");
for (let k = 0; k < 30; k++) { sim2.stepCount++; d2.record(null); }
check(d2.check().some((a) => a.type === "floating"), "released body hanging in the air is 'floating'");

console.log(fails ? `DIAG TESTS FAILED (${fails})` : "DIAG TESTS PASSED");
process.exit(fails ? 1 : 0);

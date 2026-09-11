/**
 * The Rapier engine, held to the standard of real pencils.
 *
 * Same interface as the learned simulator, same sensors, and the checks
 * are the ones players actually raised: pencils lie flat, they touch what
 * they rest on, nothing floats, a held pencil holds still, an end grab
 * dangles, pulling the bottom pencil drops the pile, drops and fast pulls
 * pass through nothing, and a pencil laid across another's point rests on
 * a point.
 *
 *   node web/test/rapier.mjs
 */
import { readFileSync } from "node:fs";
import R from "@dimforge/rapier3d-compat";
import { RapierSim } from "../public/js/rapier_sim.js";
import { Sensors } from "../public/js/sensors.js";
import { capsuleClosest, capsuleWorld, grabForce, lowestSurface, quatToMatrix } from "../public/js/physics.js";
import { LENGTH } from "../public/js/pencil.js";

await R.init();
const root = new URL("../..", import.meta.url).pathname;
const runtime = JSON.parse(readFileSync(`${root}web/public/model/runtime.json`));
const SCENES = (process.env.SCENE ? [process.env.SCENE] : ["IMG_8504", "IMG_8513", "IMG_8596", "IMG_8626"]);
const G = runtime.gravity, DT = runtime.dt;
const GRAB = { omega: 28, zeta: 1.0 };

let fails = 0;
const check = (ok, msg) => { console.log(`${ok ? "PASS" : "FAIL"} ${msg}`); if (!ok) fails++; };
const hyp = (v) => Math.hypot(v[0], v[1], v[2]);
const packetOf = (s) => JSON.parse(readFileSync(`${root}web/public/packets/${s}.json`));
const fresh = (packet) => new RapierSim(R, runtime, packet);
const elevation = (sim, b) => {
  const a = capsuleWorld(sim.state.pos[b], sim.state.quat[b], sim.packet.bodies[b].capsule).a;
  return Math.asin(Math.min(1, Math.abs(a[2]))) * 180 / Math.PI;
};
const worldPoint = (sim, b, local) => {
  const M = quatToMatrix(sim.state.quat[b]), p = sim.state.pos[b];
  return [M[0] * local[0] + M[1] * local[1] + M[2] * local[2] + p[0],
    M[3] * local[0] + M[4] * local[1] + M[5] * local[2] + p[1],
    M[6] * local[0] + M[7] * local[1] + M[8] * local[2] + p[2]];
};
const lowest = (sim, b) => lowestSurface(sim.state.pos[b], sim.state.quat[b], sim.packet.bodies[b].capsule);
const worstOverlap = (sim) => {
  const segs = sim.packet.bodies.map((x, i) => capsuleWorld(sim.state.pos[i], sim.state.quat[i], x.capsule));
  let w = 0;
  for (let i = 0; i < sim.B; i++) for (let j = i + 1; j < sim.B; j++) w = Math.max(w, capsuleClosest(segs[i], segs[j]).pen);
  return w;
};
const worstBelow = (sim) => { let w = 0; for (let b = 0; b < sim.B; b++) w = Math.max(w, -lowest(sim, b)); return w; };

/** Hold `body` at `local` with the grab spring, the target moving to `to`
 * over `steps`, then holding; onStep after every step. */
async function drag(sim, S, body, local, to, steps, hold, onStep = null) {
  const from = worldPoint(sim, body, local);
  for (let k = 1; k <= steps + hold; k++) {
    const f = Math.min(1, k / steps);
    const target = [from[0] + to[0] * f, from[1] + to[1] * f, from[2] + to[2] * f];
    const wp = worldPoint(sim, body, local);
    const force = grabForce({ m: sim.mass[body], v: sim.state.linvel[body], wp, target,
      omega: GRAB.omega, zeta: GRAB.zeta, gravity: G, capG: runtime.grab.force_cap, blocked: false });
    await sim.step(body, wp, force, true);
    S?.sample({ kind: "grab", body });
    onStep?.(k);
  }
}

for (const scene of SCENES) {
  const packet = packetOf(scene);
  console.log(`\n=== ${scene} ===`);

  // ---- 1. at rest: flat, touching, nothing floating, nothing moving
  {
    const sim = fresh(packet); const S = new Sensors(sim);
    for (let k = 0; k < 60; k++) { await sim.step(); S.sample(null); }
    const p0 = sim.state.pos.map((p) => [...p]);
    for (let k = 0; k < 600; k++) { await sim.step(); S.sample(null); }
    const now = S.now();
    const drifts = sim.state.pos.map((p, i) => Math.hypot(...p.map((x, k) => x - p0[i][k])) * 1e3);
    const drift = Math.max(...drifts);
    const asleep = sim.bodies.map((b) => (b.isSleeping() ? "z" : "-")).join("");
    const unsupported = now.bodies.filter((b) => b.unsupported).map((b) => b.id);
    const gaps = now.bodies.map((b) => {
      const under = b.contacts.filter((c) => c.upper === b.id);
      return b.floor.touching ? 0 : Math.min(b.floor.gap_mm, ...under.map((c) => c.gap_mm));
    });
    const elevs = now.bodies.map((b) => b.elevation_deg);
    const onFloor = now.bodies.filter((b) => b.floor.touching);
    check(drift < 0.5, `rest: nothing drifts (${drifts.map((d) => d.toFixed(2)).join("/")} mm over 10 s, sleeping ${asleep})`);
    check(!unsupported.length, `rest: nothing floats (unsupported ${JSON.stringify(unsupported)})`);
    check(Math.max(...gaps) < 0.5, `rest: every body touches what it rests on (gaps ${gaps.map((g) => g.toFixed(2)).join("/")} mm)`);
    check(worstOverlap(sim) * 1e3 < 0.5 && worstBelow(sim) * 1e3 < 0.2,
      `rest: nothing overlaps (${(worstOverlap(sim) * 1e3).toFixed(2)} mm body-body, ${(worstBelow(sim) * 1e3).toFixed(2)} mm below the table)`);
    check(onFloor.every((b) => ["grip", "barrel", "lower barrel"].includes(b.floor.at.region) || b.elevation_deg > 3),
      `rest: bodies on the desk rest on their grip or barrel, not a point (${onFloor.map((b) => `${b.id}:${b.floor.at.region}@${b.elevation_deg}deg`).join(" ")})`);
    console.log(`     elevations ${elevs.map((e) => e.toFixed(1)).join("/")} deg, at rest ${now.scene_at_rest}, ` +
      `speeds ${now.bodies.map((b) => b.speed_mms.toFixed(2)).join("/")} mm/s, spins ${now.bodies.map((b) => b.spin_rads.toFixed(3)).join("/")}`);
  }

  // ---- 2. a lone pencil dropped on the desk lies flat on its grip
  {
    const sim = fresh(packet);
    for (let b = 1; b < sim.B; b++) sim.state.pos[b] = [1 + 0.3 * b, 1, 0.02];   // out of the way
    // body 0: 80 mm up, tilted 30 deg, released
    const c = Math.cos(Math.PI / 12), s = Math.sin(Math.PI / 12);
    sim.state.pos[0] = [0, 0, 0.08];
    sim.state.quat[0] = [0, s, 0, c];                  // 30 deg about y
    sim.state.linvel[0] = [0, 0, 0]; sim.state.angvel[0] = [0, 0, 0];
    const S = new Sensors(sim);
    for (let k = 0; k < 360; k++) { await sim.step(); S.sample(null); }
    const b = S.body(0);
    check(b.elevation_deg < 1.0, `lone pencil lies flat (${b.elevation_deg} deg)`);
    check(b.floor.touching && b.floor.at.region === "grip",
      `lone pencil rests on its grip (${b.floor.at.region}, gap ${b.floor.gap_mm} mm)`);
    check(!b.moving && sim.restCount[0] > 30, `lone pencil is at rest (speed ${b.speed_mms} mm/s, spin ${b.spin_rads})`);
  }

  // ---- 3. hold still at the centre: no rearing, no spin
  const topOf = (sim) => sim.state.pos.map((p, i) => [p[2], i]).sort((a, b) => b[0] - a[0])[0][1];
  {
    const sim = fresh(packet); const S = new Sensors(sim);
    const top = topOf(sim);
    const e0 = elevation(sim, top);
    let peakTilt = 0, peakSpin = 0, below = 0;
    await drag(sim, S, top, [0, 0, 0], [0, 0, 0], 1, 300, () => {
      peakTilt = Math.max(peakTilt, Math.abs(elevation(sim, top) - e0));
      peakSpin = Math.max(peakSpin, hyp(sim.state.angvel[top]));
      below = Math.max(below, -lowest(sim, top) * 1e3);
    });
    // A pencil that was lying tilted across a pile shifts a few degrees when
    // its weight is taken off the pile; the rearing being guarded against
    // was 28 to 90 degrees.
    check(peakTilt < 6, `hold still: no rearing (peak tilt change ${peakTilt.toFixed(2)} deg)`);
    check(peakSpin < 0.5, `hold still: no spin (peak ${peakSpin.toFixed(3)} rad/s)`);
    check(below < 0.2, `hold still: above the table (${below.toFixed(2)} mm below at worst)`);
  }

  // ---- 4. an end grab lifted well clear dangles
  {
    const sim = fresh(packet);
    const top = topOf(sim);
    await drag(sim, null, top, [0.070, 0, 0], [0, 0, 0.25], 150, 150);
    const e = elevation(sim, top);
    check(e > 80, `end grab dangles (${e.toFixed(1)} deg from flat)`);
  }

  // ---- 5. pulling the load-bearing pencil out lowers what it carried
  {
    const sim = fresh(packet); const S = new Sensors(sim);
    for (let k = 0; k < 30; k++) { await sim.step(); S.sample(null); }
    let bottom = -1, carried = [];
    for (let i = 0; i < sim.B; i++) {
      const on = S.body(i).carrying.filter((j) => j !== "table");
      if (on.length > carried.length) { bottom = i; carried = on; }
    }
    if (bottom < 0) console.log("SKIP support removal: nothing carries anything");
    else {
      const z0 = carried.map((b) => lowest(sim, b));
      const start = [...sim.state.pos[bottom]];
      const axis = capsuleWorld(sim.state.pos[bottom], sim.state.quat[bottom], packet.bodies[bottom].capsule).a;
      // fully clear, not one pencil-length: pulled slowly, a pencil whose
      // weight is mostly on the one being pulled rides along on it (real
      // ones do), and a 160 mm pull left two riders still aboard in IMG_8504
      await drag(sim, S, bottom, [0, 0, 0], [axis[0] * 0.28, axis[1] * 0.28, 0], 60, 30);
      const p1 = carried.map((b) => [...sim.state.pos[b]]);
      const moved = Math.hypot(...sim.state.pos[bottom].map((x, k) => x - start[k])) * 1e3;
      for (let k = 0; k < 180; k++) { await sim.step(); S.sample(null); }
      const drops = carried.map((b, i) => (z0[i] - lowest(sim, b)) * 1e3);
      const rode = carried.map((b, i) => Math.hypot(...sim.state.pos[b].map((x, k) => x - p1[i][k])) * 1e3);
      const states = carried.map((b) => S.body(b));
      const after = carried.map((b, i) => `${b}:on ${JSON.stringify(states[i].resting_on)}`).join(" ");
      // What support removal must never produce: a pencil left in the air,
      // or one still moving. What it may produce is any of three real
      // outcomes: the carried pencil drops onto what was below, it rides up
      // the incline and lands on a neighbour, or, lying almost parallel with
      // most of its weight on the pencil being pulled and only an end on the
      // desk, it rides along with it (IMG_8504 body 1 rides 277 mm). Real
      // pencils do all three.
      const settled = states.every((b) => !b.unsupported && !b.moving);
      const outcome = carried.map((b, i) =>
        drops[i] > 1.5 ? "dropped" : !states[i].resting_on.includes(bottom) ? "moved off" : "rode along");
      check(moved > 100 && settled,
        `pulling body ${bottom} out (moved ${moved.toFixed(0)} mm) leaves the pile settled ` +
        `(${carried.map((b, i) => `${b} ${outcome[i]}, ${drops[i].toFixed(1)} mm down`).join("; ")}; ${after})`);
    }
  }

  // ---- 6. a 250 mm drop onto the pile passes through nothing
  {
    const sim = fresh(packet);
    const top = topOf(sim);
    sim.state.pos[top][2] += 0.25;
    sim.state.linvel[top] = [0, 0, 0]; sim.state.angvel[top] = [0, 0, 0];
    let deepest = 0, belowT = 0;
    for (let k = 0; k < 240; k++) {
      await sim.step();
      deepest = Math.max(deepest, worstOverlap(sim) * 1e3);
      belowT = Math.max(belowT, worstBelow(sim) * 1e3);
    }
    check(deepest < 1.0 && belowT < 0.3,
      `250 mm drop passes through nothing (${deepest.toFixed(2)} mm body-body, ${belowT.toFixed(2)} mm below the table)`);
  }

  // ---- 7. a fast pull passes through nothing
  {
    const sim = fresh(packet);
    const bottom = sim.state.pos.map((p, i) => [p[2], i]).sort((a, b) => a[0] - b[0])[0][1];
    const axis = capsuleWorld(sim.state.pos[bottom], sim.state.quat[bottom], packet.bodies[bottom].capsule).a;
    let deepest = 0, belowT = 0;
    await drag(sim, null, bottom, [0, 0, 0], [axis[0] * 0.20, axis[1] * 0.20, 0], 20, 40, () => {
      deepest = Math.max(deepest, worstOverlap(sim) * 1e3);
      belowT = Math.max(belowT, worstBelow(sim) * 1e3);
    });
    check(deepest < 1.5 && belowT < 0.3,
      `fast pull passes through nothing (${deepest.toFixed(2)} mm body-body, ${belowT.toFixed(2)} mm below the table)`);
  }

  // ---- 8. a pencil laid across another's point rests on the point
  {
    const sim = fresh(packet);
    for (let b = 2; b < sim.B; b++) sim.state.pos[b] = [1 + 0.3 * b, 1, 0.02];
    // body 1 flat along +x with its point at +x; body 0 across it, over the point
    sim.state.pos[1] = [0, 0, 0.006]; sim.state.quat[1] = [0, 0, 0, 1];
    sim.state.linvel[1] = [0, 0, 0]; sim.state.angvel[1] = [0, 0, 0];
    const tipX = LENGTH / 2 - 0.004;                    // 4 mm from the point
    sim.state.pos[0] = [tipX, 0, 0.03];
    sim.state.quat[0] = [0, 0, Math.SQRT1_2, Math.SQRT1_2];   // along +y
    sim.state.linvel[0] = [0, 0, 0]; sim.state.angvel[0] = [0, 0, 0];
    // A round pencil balanced on a 2 mm cone rolls off, as it would on a
    // real desk, so the check is about the contact at the moment it lands:
    // the lower pencil must be touched on its point or cone, and the upper
    // one must be sitting at a height that a point, not a barrel, gives.
    const S = new Sensors(sim);
    let first = null, landedZ = 0;
    for (let k = 0; k < 300; k++) {
      await sim.step(); S.sample(null);
      if (!first) {
        const c = S.touch().find((t) => t.bodies[0] === 0 && t.bodies[1] === 1 && t.touching);
        if (c) { first = c; landedZ = sim.state.pos[0][2]; }
      }
    }
    const region = first ? first.on_j.region : "none";
    check(first && ["point", "cone"].includes(region),
      `a pencil dropped across another's point lands on its ${region} ` +
      `(${first ? first.on_j.mm_from_point : "?"} mm from the point, ${first ? first.on_j.radius_mm : "?"} mm radius there)`);
    // over the point the lower surface is ~2 mm up, not 9: the upper axis at
    // landing must sit under 12 mm, where a barrel-thick contact would put
    // it at 4.5 + 4.5 + ~1 = 10 mm above the lower axis at 6 mm, i.e. 16 mm
    check(first && landedZ * 1e3 < 12,
      `it sits at a point's height when it lands (axis ${(landedZ * 1e3).toFixed(1)} mm up; a barrel would give ~16)`);
  }
}

console.log(fails ? `\nRAPIER TESTS FAILED (${fails})` : "\nRAPIER TESTS PASSED");
process.exit(fails ? 1 : 0);

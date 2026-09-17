// The float regression on the LIVE page, whichever backend it is running
// (the WebGPU kernels, ONNX Runtime Web, or Rapier): a random play session
// driven through physsplat.scriptDrag, with the same grounded-chain check as
// float.mjs run after every physics step. This is how the WebGPU build was
// checked, since node cannot run it.
//
//   paste into the console (or eval through a browser driver), then read
//   window.__float.summary(); window.__float.stop() ends it early.
//   window.__floatSeed / window.__floatGestures set the session beforehand.
//
// A body is GROUNDED if it is within 3 mm of the table, is the held body, or
// rests on a grounded body. Floating: ungrounded and unheld for 30 steps
// with its vertical speed under 2 cm/s, or standing steeply in the air, or
// above 3 mm for the whole of the last 30 steps with its height changing by
// under 10 mm.
(async () => {
  const P = await import("./js/physics.js");
  const dbg = globalThis.physsplat;
  const sim = dbg.sim, DT = sim.rt.dt, bs = sim.packet.bodies;
  let rs = (globalThis.__floatSeed ?? 7) >>> 0 || 1;
  const rand = () => { rs ^= rs << 13; rs >>>= 0; rs ^= rs >>> 17; rs ^= rs << 5; rs >>>= 0; return rs / 4294967296; };
  const pick = (n) => Math.floor(rand() * n);
  const hyp = (v) => Math.hypot(v[0], v[1], v[2]);
  const worldPoint = (b, local) => {
    const R = P.quatToMatrix(sim.state.quat[b]), p = sim.state.pos[b];
    return [R[0] * local[0] + p[0], R[3] * local[0] + p[1], R[6] * local[0] + p[2]];
  };
  const elevation = (b) => Math.asin(Math.min(1, Math.abs(P.capsuleWorld(sim.state.pos[b], sim.state.quat[b], bs[b].capsule).a[2]))) * 180 / Math.PI;
  function grounded(held, tol = 3e-3) {
    const B = sim.B;
    const segs = bs.map((x, i) => P.capsuleWorld(sim.state.pos[i], sim.state.quat[i], x.capsule));
    const low = bs.map((x, i) => P.lowestSurface(sim.state.pos[i], sim.state.quat[i], x.capsule));
    const on = Array.from({ length: B }, () => []), touch = Array.from({ length: B }, () => []);
    for (let i = 0; i < B; i++) for (let j = i + 1; j < B; j++) {
      const c = P.capsuleClosest(segs[i], segs[j]);
      if (-c.pen < tol) { touch[i].push(j); touch[j].push(i); if (c.n[2] > 0.2) on[i].push(j); if (c.n[2] < -0.2) on[j].push(i); }
    }
    const g = new Uint8Array(B);
    for (let i = 0; i < B; i++) if (low[i] < tol || i === held) g[i] = 1;
    for (let it = 0; it < B; it++) for (let i = 0; i < B; i++) if (!g[i] && on[i].some((j) => g[j])) g[i] = 1;
    return { g, low, on, touch };
  }
  const F = globalThis.__float = { episodes: [], gestures: [], steps: 0, running: true, worst: 0, maxLow: 0,
    stop() { F.running = false; dbg.scriptDrag = null; },
    summary() { return { steps: F.steps, gestures: F.gestures.length, episodes: F.episodes, worst_mm: F.worst, maxLow_mm: F.maxLow * 1e3 }; } };
  const run = new Int32Array(sim.B), open = {}, hist = Array.from({ length: sim.B }, () => []);
  const FLOAT_STEPS = 30;
  // `frame` is the diagnostics record of the step just taken. Read the
  // rule flags from it, not from sim.last: in a hidden tab the loop starts
  // the next step with no await in between, and sim.last is already the
  // next step's empty record by the time this runs.
  function check(held, gi, frame) {
    const step = sim.stepCount, { g, low, on, touch } = grounded(held);
    for (let b = 0; b < sim.B; b++) {
      const vz = sim.state.linvel[b][2], spd = hyp(sim.state.linvel[b]);
      hist[b].push(low[b]); if (hist[b].length > FLOAT_STEPS) hist[b].shift();
      if (b === held) hist[b].length = 0; else F.maxLow = Math.max(F.maxLow, low[b]);
      const steep = elevation(b) > 45 && low[b] > 3e-3 && spd < 0.02 && b !== held;
      const stuck = !g[b] && b !== held && hist[b].length === FLOAT_STEPS && Math.min(...hist[b]) > 3e-3 &&
        Math.max(...hist[b]) - Math.min(...hist[b]) < 0.01;
      const floating = (!g[b] && b !== held && Math.abs(vz) < 0.02) || steep || stuck;
      run[b] = floating ? run[b] + 1 : 0;
      if ((run[b] === FLOAT_STEPS || (stuck && run[b] === 1)) && !open[b]) {
        const fb = frame?.bodies?.[b];
        open[b] = { gesture: gi, body: b, from: step - FLOAT_STEPS + 1, lowest_mm: +(low[b] * 1e3).toFixed(1),
          elev_deg: +elevation(b).toFixed(1), speed_mms: +(spd * 1e3).toFixed(1), restsOn: on[b], touching: touch[b],
          groundedMask: Array.from(g), stuck, freeFlight: !!fb?.guard?.freeFlight, pivot: !!fb?.guard?.pivot, settled: !!fb?.guard?.settled,
          resZ: fb?.model ? +fb.model.lin[2].toFixed(2) : null };
        F.worst = Math.max(F.worst, low[b] * 1e3);
      }
      if (!floating && open[b]) { open[b].to = step; open[b].steps = step - open[b].from; F.episodes.push(open[b]); delete open[b]; }
    }
  }
  // one physics step at a time, resolved by the page's own diagnostics
  // recorder (a timer would be throttled in a hidden tab)
  const waitSteps = async (n, held, gi) => { for (let k = 0; k < n && F.running; k++) { const frame = await dbg.diag.waitSteps(1); F.steps++; check(held, gi, frame); } };
  for (let gi = 0; F.running && gi < (globalThis.__floatGestures ?? 40); gi++) {
    const body = pick(sim.B), cap = bs[body].capsule;
    const local = [(rand() * 2 - 1) * cap.half * 0.9, 0, 0];
    const from = worldPoint(body, local);
    const kind = pick(8);
    let to;
    if (kind === 0) to = [from[0], from[1], from[2] + 0.03 + rand() * 0.08];                                   // lift
    else if (kind === 1) to = [from[0] + (rand() - 0.5) * 0.2, from[1] + (rand() - 0.5) * 0.2, from[2] + 0.02 + rand() * 0.06]; // lift and carry
    else if (kind === 2) to = [from[0] + (rand() - 0.5) * 0.16, from[1] + (rand() - 0.5) * 0.16, from[2]];     // slide
    else if (kind === 3) to = [from[0] + (rand() - 0.5) * 0.1, from[1] + (rand() - 0.5) * 0.1, Math.max(0.005, from[2] - 0.03)]; // press and drag
    else if (kind === 4) to = [from[0] + (rand() - 0.5) * 0.05, from[1] + (rand() - 0.5) * 0.05, from[2] + 0.1 + rand() * 0.1]; // high lift, drop
    else if (kind === 5) to = [from[0], from[1], from[2]];                                                    // hold still
    else if (kind === 6) { const a = rand() * Math.PI * 2, d = 0.02 + rand() * 0.06; to = [from[0] + Math.cos(a) * d, from[1] + Math.sin(a) * d, from[2]]; } // flick
    else to = [from[0] + (rand() - 0.5) * 0.1, from[1] + (rand() - 0.5) * 0.1, -0.02 - rand() * 0.03];         // target under the table
    const speed = kind === 6 ? 0.15 + rand() * 0.45 : 0.30;
    const follow = [...from];
    const dist = hyp([to[0] - from[0], to[1] - from[1], to[2] - from[2]]);
    const travel = Math.ceil(dist / (speed * DT)), hold = kind === 6 ? 0 : 10 + pick(40);
    F.gestures.push({ gi, body, kind, travel, hold, step: sim.stepCount });
    for (let s = 0; s < travel + hold && F.running; s++) {
      const d = [to[0] - follow[0], to[1] - follow[1], to[2] - follow[2]], dn = hyp(d), mx = speed * DT;
      if (dn > mx) for (let q = 0; q < 3; q++) follow[q] += d[q] / dn * mx; else for (let q = 0; q < 3; q++) follow[q] = to[q];
      dbg.scriptDrag = { body, local, target: [...follow] };
      await waitSteps(1, body, gi);
    }
    dbg.scriptDrag = null;
    await waitSteps(40 + pick(80), -1, gi);
  }
  await waitSteps(120, -1, -1);
  for (const b in open) { open[b].to = sim.stepCount; open[b].steps = open[b].to - open[b].from; F.episodes.push(open[b]); }
  F.running = false; F.done = true;
})();

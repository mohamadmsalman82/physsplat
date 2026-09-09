/**
 * PhysSplat browser app: rendering, interaction, and the physics loop.
 * Physics runs client-side: PhysSim (js/sim.js) + ONNX Runtime Web.
 */
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { PhysSim } from "./sim.js";
import { Diagnostics, Probes } from "./diag.js";
import { buildPencil } from "./pencil_mesh.js";
import { grabForce } from "./physics.js";

const $ = (id) => document.getElementById(id);
const err = (m) => { $("err").textContent = String(m); console.error(m); };

// ------------------------------------------------------------- three scene
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x101216);
const cam = new THREE.PerspectiveCamera(42, innerWidth / innerHeight, 0.01, 10);
cam.position.set(0.16, -0.26, 0.2);
cam.up.set(0, 0, 1);
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(innerWidth, innerHeight);
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
document.body.appendChild(renderer.domElement);
const CAM_HOME = { pos: [0.16, -0.26, 0.2], target: [0, 0, 0.015] };
const controls = new OrbitControls(cam, renderer.domElement);
controls.target.set(...CAM_HOME.target);
controls.rotateSpeed = 0.55;
controls.zoomSpeed = 0.9;
controls.enableDamping = true;
controls.dampingFactor = 0.12;
controls.minDistance = 0.1;
controls.maxDistance = 0.9;
controls.minPolarAngle = 0.35;          // keep a usable elevated view
controls.maxPolarAngle = 1.25;          // never near the floor plane
const Q = new URLSearchParams(location.search);
renderer.shadowMap.enabled = Q.get("shadows") !== "0";
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
scene.add(new THREE.AmbientLight(0xffffff, 0.55));
scene.add(new THREE.HemisphereLight(0xbfd4ff, 0x2a2622, 0.5));
const sun = new THREE.DirectionalLight(0xffffff, 1.6);
sun.position.set(0.35, -0.45, 0.9);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
sun.shadow.camera.near = 0.1; sun.shadow.camera.far = 3;
sun.shadow.camera.left = sun.shadow.camera.bottom = -0.4;
sun.shadow.camera.right = sun.shadow.camera.top = 0.4;
// normalBias offsets the shadow lookup along the surface normal, so it has
// to be small next to the object: 20 mm on a 4.5 mm pencil detached every
// shadow from its caster (a blind tester saw "wrongly shaped streaks that
// match no pencil"). 1.5 mm is a third of the radius, enough for the
// curvature and invisible as an offset.
sun.shadow.bias = -0.0001;
sun.shadow.normalBias = 0.0015;
scene.add(sun);
// a lit wooden table, not a dark grid in a void
const ground = new THREE.Mesh(
  new THREE.CircleGeometry(0.6, 64),
  new THREE.MeshStandardMaterial({ color: 0x8a6f4e, roughness: 0.75, metalness: 0 }));
ground.receiveShadow = true;
scene.add(ground);

// ---------------------------------------------------------------- runtime
let sim = null, groups = [], proxies = [], loading = false;
// Debug / diagnostics handle (docs/diagnostics.md). ?pause=1 starts frozen;
// ?diag=1 opens the live panel. A test harness reads physsplat.diag.* and
// runs physsplat.probe.* instead of inferring physics from pixels.
let diag = null;
const dbg = {
  paused: Q.get("pause") === "1", stepOnce: false,
  scriptDrag: null,    // {body, local, target}: a grab driven by a probe
  scriptPoke: null,    // {body, point, force, left}: a poke driven by a probe
  get sim() { return sim; }, get proxies() { return proxies; },
  get diag() { return diag; }, probe: null,
  get preroll() { return preroll; },
  resetScene() { resetScene(); },
  state() { return diag?.snapshot(); },
  // Non-blocking probe runner for harnesses whose eval calls time out:
  //   id = physsplat.run("lift", 2, {height: 0.05}); ... physsplat.report(id)
  reports: {}, _runId: 0,
  run(name, ...args) {
    const id = ++dbg._runId;
    const rep = dbg.reports[id] = { id, name, args, status: "running", started: new Date().toISOString() };
    if (!dbg.probe?.[name]) { rep.status = "error"; rep.error = `no probe named ${name}`; return id; }
    dbg.probe[name](...args)
      .then((r) => { rep.status = "done"; rep.result = r; rep.finished = new Date().toISOString(); })
      .catch((e) => { rep.status = "error"; rep.error = String(e && e.stack || e); });
    return id;
  },
  report(id) { return dbg.reports[id ?? dbg._runId] ?? null; },
};
globalThis.physsplat = dbg;

// Yield to the event loop between physics steps. setTimeout is throttled to
// 1 Hz in a background tab, which froze scripted probes run from a hidden
// tab; a MessageChannel round-trip is not throttled, so use it whenever the
// page is hidden (the renderer is idle then anyway).
const yieldChannel = new MessageChannel();
let yieldResolve = null;
yieldChannel.port1.onmessage = () => { const r = yieldResolve; yieldResolve = null; r?.(); };
function yieldSlice(ms) {
  if (document.hidden) return new Promise((res) => { yieldResolve = res; yieldChannel.port2.postMessage(0); });
  return new Promise((res) => setTimeout(res, ms));
}

async function loadScene(name) {
  loading = true;                      // physics loop idles until rebuilt
  const packet = await (await fetch(`./packets/${name}.json`, { cache: "no-cache" })).json();
  groups.forEach((g) => scene.remove(g));
  proxies.forEach((p) => scene.remove(p));
  groups = []; proxies = [];
  prevState = currState = null;
  sim.packet = packet;
  sim.reset();
  dbg.scriptDrag = dbg.scriptPoke = null;
  if (!diag) { diag = new Diagnostics(sim); dbg.probe = new Probes(dbg, sim, diag); }
  else diag.reset(`scene:${name}`);
  packet.bodies.forEach((b, i) => {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position",
      new THREE.Float32BufferAttribute(b.render_verts.flat(), 3));
    geo.setAttribute("color", new THREE.Float32BufferAttribute(
      b.render_colors.flat().map((c) => c / 255), 3));
    let obj;
    if (b.capsule && Q.get("points") !== "1" && Q.get("mesh") !== "capsule") {
      // procedural pencil inside the physical capsule, coloured from the
      // reconstruction (js/pencil_mesh.js)
      obj = buildPencil(b);
    } else if (b.render_faces && b.render_faces.length && Q.get("points") !== "1") {
      // lit surface from the reconstruction's own triangles
      geo.setIndex(b.render_faces.flat());
      geo.computeVertexNormals();
      obj = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({
        vertexColors: true, roughness: 0.6, metalness: 0.05 }));
      obj.castShadow = true;
    } else {
      obj = new THREE.Points(geo, new THREE.PointsMaterial({
        size: 0.0035, vertexColors: true }));
    }
    scene.add(obj);
    groups.push(obj);
    // generous invisible pick volume: a real pencil is ~10 px wide on screen
    // and grabbing it should not require pixel precision
    const cap = new THREE.Mesh(
      new THREE.CapsuleGeometry(b.capsule.radius * 2.2, 2 * b.capsule.half, 4, 10),
      new THREE.MeshBasicMaterial({ visible: false }));
    // capsule geometry is y-aligned; rotate y onto the body-frame axis
    cap.quaternion.setFromUnitVectors(
      new THREE.Vector3(0, 1, 0), new THREE.Vector3(...b.capsule.axis).normalize());
    const wrap = new THREE.Group();
    wrap.add(cap);
    wrap.userData.body = i;
    scene.add(wrap);
    proxies.push(wrap);
  });
  syncTransforms();
  resetCamera();
  // A reconstructed pile is not exactly in equilibrium (single-view depth
  // error), and the physics rules move it into one during the first few
  // steps. Run those steps before showing motion, so the scene appears
  // already settled instead of twitching on load.
  preroll = PREROLL_STEPS;
  loading = false;
}
const PREROLL_STEPS = 45;   // 0.75 s: a lifted-at-load pencil drops and settles within this
let preroll = 0;

/**
 * A pencil pointing straight at the camera foreshortens into what looks like
 * a pencil standing on end (a blind test read exactly that as a physics
 * bug). Pick the viewing azimuth that stays furthest from every pencil's
 * axis, so each pencil reads as a horizontal stick from the home view.
 */
function bestAzimuth(packet) {
  const axes = packet.bodies.map((b) => {
    const q = new THREE.Quaternion(...b.quat);
    const a = new THREE.Vector3(...b.capsule.axis).applyQuaternion(q);
    return Math.atan2(a.y, a.x);
  });
  let best = -Math.PI / 2 - 0.55, bestScore = -1;
  for (let k = 0; k < 72; k++) {
    const az = (k / 72) * 2 * Math.PI;
    // score = smallest sin(angle to any pencil axis); axis sign is irrelevant
    const s = Math.min(...axes.map((a) => Math.abs(Math.sin(az - a))));
    if (s > bestScore + 1e-6) { bestScore = s; best = az; }
  }
  return best;
}

function resetCamera() {
  const dist = 0.36, elev = 0.62;            // radians above the table
  const az = sim?.packet?.bodies?.length ? bestAzimuth(sim.packet) : -1.0;
  cam.position.set(dist * Math.cos(elev) * Math.cos(az),
    dist * Math.cos(elev) * Math.sin(az), dist * Math.sin(elev));
  controls.target.set(...CAM_HOME.target);
  controls.update();
}

// Two physics states are kept so rendering can interpolate between them at
// display rate: physics runs at ~20-30 Hz, the screen at 60, and without
// this every pencil visibly hops between steps.
let prevState = null, currState = null, stateTime = 0, stateDt = 1 / 20;

function snapshot(state) {
  return { pos: state.pos.map((p) => [...p]), quat: state.quat.map((q) => [...q]) };
}

function syncTransforms() {
  if (!sim || proxies.length !== sim.B) return;   // mid scene switch
  prevState = currState ?? snapshot(sim.state);
  currState = snapshot(sim.state);
  stateTime = performance.now();
  sim.state.pos.forEach((p, i) => {
    proxies[i].position.set(...p);
    proxies[i].quaternion.set(...sim.state.quat[i]);
  });
}

const _qa = new THREE.Quaternion(), _qb = new THREE.Quaternion();
function renderInterpolated() {
  if (!currState || groups.length !== currState.pos.length) return;
  const alpha = Math.min(1, (performance.now() - stateTime) / (stateDt * 1000));
  currState.pos.forEach((p, i) => {
    const q0 = prevState.pos[i];
    groups[i].position.set(
      q0[0] + (p[0] - q0[0]) * alpha, q0[1] + (p[1] - q0[1]) * alpha,
      q0[2] + (p[2] - q0[2]) * alpha);
    _qa.set(...prevState.quat[i]); _qb.set(...currState.quat[i]);
    groups[i].quaternion.copy(_qa.slerp(_qb, alpha));
  });
}

// ------------------------------------------------------------ interaction
const ray = new THREE.Raycaster();
let drag = null;
const ndc = (e) => new THREE.Vector2(
  (e.clientX / innerWidth) * 2 - 1, -(e.clientY / innerHeight) * 2 + 1);

renderer.domElement.addEventListener("pointerdown", (e) => {
  if (!sim) return;
  ray.setFromCamera(ndc(e), cam);
  const hits = ray.intersectObjects(proxies, true);
  if (!hits.length) return;
  controls.enabled = false;
  const body = hits[0].object.parent.userData.body;
  // grab point in body frame, so it rides the body. Attach ON the pencil's
  // axis: the pick volume is wider than the pencil, and a spring pulling
  // 12 mm off the axis of a body with 2e-7 kg m^2 of axial inertia spun it
  // to 100-280 rad/s (blind test round 3). A fingertip pinch does not
  // torque a pencil about its own axis either.
  const p = hits[0].point;
  const inv = proxies[body].quaternion.clone().invert();
  const local = p.clone().sub(proxies[body].position).applyQuaternion(inv);
  const cap = sim.packet.bodies[body].capsule;
  const axis = new THREE.Vector3(...cap.axis).normalize();
  const along = THREE.MathUtils.clamp(local.dot(axis), -cap.half, cap.half);
  local.copy(axis).multiplyScalar(along);
  const attach = local.clone().applyQuaternion(proxies[body].quaternion).add(proxies[body].position);
  drag = { body, local, target: attach.clone(), p0: attach.clone(),
    t0: performance.now(), moved: false, sx: e.clientX, sy: e.clientY, px: 0 };
  diag?.event("pointer_down", { body, local: local.toArray().map((x) => +x.toFixed(4)) });
});
renderer.domElement.addEventListener("pointermove", (e) => {
  if (!drag) return;
  drag.moved = true;
  // travel in screen pixels, not world space: the drag plane is defined by
  // the camera, so orbit damping still coasting from an earlier gesture
  // moves the projected target even when the mouse has not moved at all,
  // which turned a stationary click into a 0.19 m/s flick
  drag.px = Math.max(drag.px, Math.hypot(e.clientX - drag.sx, e.clientY - drag.sy));
  ray.setFromCamera(ndc(e), cam);
  const n = new THREE.Vector3();
  cam.getWorldDirection(n);
  const plane = new THREE.Plane().setFromNormalAndCoplanarPoint(n, drag.p0);
  const t = new THREE.Vector3();
  if (ray.ray.intersectPlane(plane, t)) drag.target.copy(t);
});
// Demo grab gains: stiffer than the data-generation spring (omega 12.6) so
// a pencil follows the cursor within ~1-2 cm instead of ~6 cm of stretch.
// Forces stay under the same 3 m g cap the model was trained with.
const GRAB = { omega: 28, zeta: 1.0 };
const POKE_MS = 350;    // a press shorter than this that moved is a flick

/**
 * A flick is a short push, the way a finger does it: the grab point is
 * carried along the gesture at the cursor's speed for `distance`, then let
 * go. The trained impulse poke (3 steps, up to 0.3 m/s) barely moves a
 * pencil on a pile because the learned friction eats it, and the model's
 * response to larger impulses is erratic (a 1 m/s poke spun a pencil to
 * the 60 rad/s cap); the spring drag path is well-behaved at any speed.
 */
function startFlick(body, local, dir, speed, distance) {
  const d = new THREE.Vector3(...dir);
  if (d.length() < 1e-6) return;
  d.normalize();
  const spd = Math.min(Math.max(speed, 0.15), 0.6);
  const dist = Math.min(Math.max(distance, 0.02), 0.08);
  const from = new THREE.Vector3(...local).applyQuaternion(proxies[body].quaternion)
    .add(proxies[body].position);
  flick = { body, local: [...local], from, dir: d, speed: spd, dist, travelled: 0 };
  diag?.event("flick", { body, speed: +spd.toFixed(2), distance_mm: +(dist * 1e3).toFixed(0) });
}
let flick = null;
dbg.flick = (body, local, dir, speed, distance) => startFlick(body, local, dir, speed, distance);

const FLICK_MIN_TRAVEL = 0.006;   // 6 mm in the world
const FLICK_MIN_PIXELS = 10;      // and 10 px on screen: a click is a click

renderer.domElement.addEventListener("pointerup", () => {
  const delta = drag ? drag.target.clone().sub(drag.p0) : null;
  if (drag && drag.moved && drag.px > FLICK_MIN_PIXELS &&
      delta.length() > FLICK_MIN_TRAVEL &&
      performance.now() - drag.t0 < POKE_MS) {
    const held = Math.max(0.05, (performance.now() - drag.t0) / 1000);
    startFlick(drag.body, drag.local.toArray(), delta.toArray(), delta.length() / held * 1.5,
      Math.max(0.03, delta.length() * 2));
  } else if (drag) {
    diag?.event("grab_end", { body: drag.body, held_ms: Math.round(performance.now() - drag.t0) });
  }
  drag = null;
  controls.enabled = true;
});
renderer.domElement.addEventListener("pointercancel", () => { drag = null; controls.enabled = true; });
renderer.domElement.addEventListener("pointerleave", () => { if (drag) { drag = null; controls.enabled = true; } });

function resetScene() {
  if (!sim) return;
  sim.reset();
  dbg.scriptDrag = dbg.scriptPoke = null;
  pendingPoke = null; flick = null;
  prevState = currState = null;
  diag?.reset("reset");
  syncTransforms();
  resetCamera();
  preroll = PREROLL_STEPS;
}
$("reset").onclick = resetScene;

function worldGrabPoint(d) {
  return d.local.clone()
    .applyQuaternion(proxies[d.body].quaternion)
    .add(proxies[d.body].position);
}

/** The demo's grab: js/physics.js grabForce with this scene's constants. */
function springForce(body, wp, target) {
  const rt = sim.rt;
  return grabForce({
    m: sim.mass[body], v: sim.state.linvel[body], wp, target,
    omega: GRAB.omega, zeta: GRAB.zeta, gravity: rt.gravity,
    capG: rt.grab.force_cap,
    blocked: (sim.last?.guard.capsule[body] ?? 0) > 1e-3
      ? sim.last.guard.normal?.[body] : false,
  });
}

// ------------------------------------------------------------ physics loop
let pendingPoke = null, stepMs = 0, running = false, lastStepAt = performance.now();
let simClock = 0, clockStart = performance.now();

async function physicsLoop() {
  if (running) return;
  running = true;
  const rt = sim.rt;
  while (true) {
    if (loading || (dbg.paused && !dbg.stepOnce)) {
      await yieldSlice(30); continue;
    }
    dbg.stepOnce = false;
    const t0 = performance.now();
    let act = [-1, null, null], actInfo = null;
    const poke = (pendingPoke && pendingPoke.left > 0) ? pendingPoke
      : (dbg.scriptPoke && dbg.scriptPoke.left > 0) ? dbg.scriptPoke : null;
    if (poke) {
      act = [poke.body, poke.point, poke.force];
      poke.left--;
      actInfo = { kind: "poke", body: poke.body, force: poke.force };
    } else if (drag && performance.now() - drag.t0 >= POKE_MS) {
      const wp = worldGrabPoint(drag).toArray();
      const f = springForce(drag.body, wp, drag.target.toArray());
      act = [drag.body, wp, f];
      actInfo = { kind: "grab", body: drag.body, force: f, point: wp, target: drag.target.toArray() };
    } else if (flick) {
      // carry the grab point along the flick at its speed, then release
      flick.travelled += flick.speed * rt.dt;
      const target = flick.from.clone().add(flick.dir.clone().multiplyScalar(flick.travelled));
      const wp = new THREE.Vector3(...flick.local)
        .applyQuaternion(new THREE.Quaternion(...sim.state.quat[flick.body]))
        .add(new THREE.Vector3(...sim.state.pos[flick.body])).toArray();
      const f = springForce(flick.body, wp, target.toArray());
      act = [flick.body, wp, f];
      actInfo = { kind: "flick", body: flick.body, force: f, point: wp, target: target.toArray() };
      if (flick.travelled >= flick.dist) flick = null;
    } else if (dbg.scriptDrag) {
      const d = dbg.scriptDrag;
      const R = sim.state.quat[d.body];
      const wp = new THREE.Vector3(...d.local).applyQuaternion(new THREE.Quaternion(...R))
        .add(new THREE.Vector3(...sim.state.pos[d.body])).toArray();
      const f = springForce(d.body, wp, d.target);
      act = [d.body, wp, f];
      actInfo = { kind: "grab", body: d.body, force: f, point: wp, target: [...d.target], scripted: true };
    }
    try {
      await sim.step(...act);
      diag?.record(actInfo);
      if (preroll > 0) {
        if (--preroll === 0) { prevState = currState = null; syncTransforms(); }
        continue;                       // no render of the settling-in steps
      }
      syncTransforms();
    } catch (e) {
      err(e);
      await new Promise((r) => setTimeout(r, 500));   // never kill the loop
      continue;
    }
    stepMs = performance.now() - t0;
    lastStepAt = performance.now();
    $("err").textContent = "";                    // a completed step clears a stale error
    stateDt = Math.max(rt.dt, stepMs / 1000);   // interpolation window
    // Never run faster than real time. The timestep is a fixed 1/60 s, so
    // on a fast machine the loop was replaying the scene at up to 1.22x and
    // on a loaded one at 0.32x: the same drop played at different speeds
    // (a blind tester measured 18.9 to 73.4 steps/s). Sim time may still
    // fall behind when a step costs more than 1/60 s, which is honest, but
    // it may never get ahead.
    simClock += rt.dt * 1000;
    const ahead = simClock - (performance.now() - clockStart);
    if (ahead < -500) { simClock = performance.now() - clockStart; }   // resync after a stall
    // always yield a real slice to the renderer: the physics kernels share
    // the GPU with WebGL, and back-to-back steps starve the frame output
    const wait = Math.max(12, ahead);
    await yieldSlice(wait);
  }
  running = false;
}

// ---------------------------------------------------------------- startup
(async () => {
  try {
    const runtime = await (await fetch("./model/runtime.json")).json();
    const q = new URLSearchParams(location.search);
    // Backend: custom WebGPU kernels (js/gpu_net.js, ~30 ms/step) when
    // WebGPU exists; otherwise ONNX Runtime Web on wasm (correct, slow).
    // ?backend=ort forces the fallback for comparison.
    let backend = null;
    if (navigator.gpu && q.get("backend") !== "ort") {
      try {
        const { GpuNet } = await import("./gpu_net.js");
        const net = await GpuNet.create("./model/weights.json", "./model/weights.bin");
        backend = { kind: "gpu", net };
      } catch (e) { console.warn("[physsplat] WebGPU backend unavailable:", e.message); }
    }
    if (!backend) {
      const ortlib = globalThis.ort;
      ortlib.env.wasm.numThreads = Math.min(4, navigator.hardwareConcurrency || 2);
      ortlib.env.wasm.simd = true;
      const session = await ortlib.InferenceSession.create(
        "./model/simulator.onnx", { executionProviders: ["wasm"] });
      backend = { kind: "ort", ort: ortlib, session };
    }
    console.log(`[physsplat] backend: ${backend.kind}`);
    sim = new PhysSim(backend, runtime, { bodies: [] });
    const names = await (await fetch("./packets/index.json")).json();
    const sel = $("scene");
    names.forEach((n) => sel.add(new Option(n, n)));
    sel.onchange = () => loadScene(sel.value);
    const first = names.includes(q.get("scene")) ? q.get("scene") : names[0];
    sel.value = first;
    await loadScene(first);
    physicsLoop();
    initDiagPanel();
    setInterval(() => {
      const t = sim?.timing;
      const detail = t ? ` [${t.backend}: features ${t.features_ms.toFixed(0)} | graph ` +
        `${t.graph_ms.toFixed(0)} | net ${t.net_ms.toFixed(0)} ms; ` +
        `N=${t.N} E=${t.E}]` : "";
      // a tester once paused the physics from the console and read the
      // resulting silence as a hang; say so, and flag a real stall
      const since = performance.now() - lastStepAt;
      const state = dbg.paused ? " | PAUSED (physsplat.paused)"
        : (!loading && since > 3000 ? ` | PHYSICS STALLED ${(since / 1000).toFixed(0)} s` : "");
      // playback rate, stated rather than hidden: a step costs more than
      // 1/60 s on most machines, so the scene runs slower than life and a
      // viewer should know that rather than read it as low gravity
      const rate = Math.min(1, runtime.dt * 1000 / Math.max(stepMs, 1));
      $("stats").textContent =
        `model step ${runtime.step} | physics ${stepMs.toFixed(0)} ms/step | ` +
        `${rate.toFixed(2)}x real time${detail}${state}`;
    }, 500);
  } catch (e) { err(e); }
})();

// ------------------------------------------------------- diagnostics panel
// Live numbers for every body (press D or ?diag=1). The same data is
// available programmatically via physsplat.diag; this is the human view.
function initDiagPanel() {
  const panel = $("diag");
  if (!panel) return;
  let open = Q.get("diag") === "1";
  panel.hidden = !open;
  addEventListener("keydown", (e) => {
    if (e.key === "d" || e.key === "D") { open = !open; panel.hidden = !open; }
  });
  $("diag-copy").onclick = async () => {
    try { await navigator.clipboard.writeText(diag.export({ last: 600 })); $("diag-copy").textContent = "copied"; }
    catch (e) { $("diag-copy").textContent = "copy failed"; }
    setTimeout(() => { $("diag-copy").textContent = "copy JSON (10 s)"; }, 1500);
  };
  $("diag-probe").onclick = async () => {
    $("diag-probe").disabled = true; $("diag-probe").textContent = "running probes...";
    try {
      const r = await dbg.probe.all();
      console.log("[physsplat] probe report", r);
      $("diag-log").textContent = JSON.stringify(r, null, 1).slice(0, 4000);
    } finally { $("diag-probe").disabled = false; $("diag-probe").textContent = "run probes"; }
  };
  const fmt = (x, d = 1) => (x == null ? "-" : (+x).toFixed(d));
  setInterval(() => {
    if (!open || !diag) return;
    const s = diag.snapshot();
    if (!s) return;
    const rows = s.bodies.map((b) =>
      `<tr><td>${b.id}</td><td>${fmt(b.mass_g)}</td><td>${fmt(b.height_mm)}</td>` +
      `<td>${fmt(b.lowest_mm)}</td><td>${fmt(b.elevation_deg, 0)}</td>` +
      `<td>${fmt(b.speed, 3)}</td><td>${fmt(b.angSpeed, 2)}</td>` +
      `<td>${b.contacts.map((c) => `${c.other}(${fmt(c.gap_mm)})`).join(" ") || (b.groundContact ? "floor" : "none")}</td>` +
      `<td>${fmt(b.penetration_mm)}</td><td>${b.resting ? b.restingSteps : "-"}</td>` +
      `<td>${fmt(b.driftSinceRest_mm)}</td>` +
      `<td>${b.guard ? fmt(b.guard.ground_mm + b.guard.capsule_mm, 2) : "-"}</td></tr>`).join("");
    $("diag-table").innerHTML =
      `<tr><th>id</th><th>g</th><th>z mm</th><th>low mm</th><th>elev</th><th>v m/s</th>` +
      `<th>w r/s</th><th>contacts(gap mm)</th><th>pen mm</th><th>rest</th><th>drift</th><th>guard</th></tr>${rows}`;
    const an = s.anomalies.length
      ? s.anomalies.map((a) => `<span class="${a.severity}">${a.key} (${a.steps} steps): ${a.message}</span>`).join("<br>")
      : `<span class="ok">no active anomalies</span>`;
    const ev = diag.events(6).map((e) => `${e.t}s ${e.type}${e.key ? " " + e.key : ""}${e.body != null ? " b" + e.body : ""}`).join("<br>");
    $("diag-anom").innerHTML = an;
    $("diag-events").innerHTML = ev;
    $("diag-totals").textContent =
      `step ${s.step} t=${s.t}s | KE ${fmt(s.totals.KE_uJ, 2)} uJ | max pen ${fmt(s.totals.maxPenetration_mm)} mm | ` +
      `floor pen ${fmt(s.totals.maxGroundPen_mm)} mm | resting ${s.totals.resting}/${s.bodies.length}` +
      (s.action ? ` | action on ${s.action.body} (${fmt(s.action.force_N, 3)} N)` : "");
  }, 250);
}

addEventListener("resize", () => {
  cam.aspect = innerWidth / innerHeight;
  cam.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
});
(function render() {
  requestAnimationFrame(render);
  controls.update();
  renderInterpolated();
  renderer.render(scene, cam);
})();

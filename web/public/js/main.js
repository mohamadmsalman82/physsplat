/**
 * PhysSplat browser app: rendering, interaction, and the physics loop.
 * Physics runs client-side: PhysSim (js/sim.js) + ONNX Runtime Web.
 */
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { PhysSim } from "./sim.js";

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
sun.shadow.mapSize.set(1024, 1024);
sun.shadow.camera.near = 0.1; sun.shadow.camera.far = 3;
sun.shadow.camera.left = sun.shadow.camera.bottom = -0.4;
sun.shadow.camera.right = sun.shadow.camera.top = 0.4;
sun.shadow.bias = -0.0005;
scene.add(sun);
const ground = new THREE.Mesh(
  new THREE.CircleGeometry(0.6, 64),
  new THREE.MeshStandardMaterial({ color: 0x2c3038, roughness: 0.92 }));
ground.receiveShadow = true;
scene.add(ground);
const grid = new THREE.GridHelper(1, 40, 0x2a2f38, 0x20242c);
grid.rotation.x = Math.PI / 2;
grid.position.z = 0.0005;
scene.add(grid);

// ---------------------------------------------------------------- runtime
let sim = null, groups = [], proxies = [], loading = false;
// Debug handle (?pause=1 starts frozen): lets a test harness inspect body
// state and single-step the physics from the console.
const dbg = { paused: Q.get("pause") === "1", stepOnce: false,
  get sim() { return sim; }, get proxies() { return proxies; } };
globalThis.physsplat = dbg;

async function loadScene(name) {
  loading = true;                      // physics loop idles until rebuilt
  const packet = await (await fetch(`./packets/${name}.json`, { cache: "no-cache" })).json();
  groups.forEach((g) => scene.remove(g));
  proxies.forEach((p) => scene.remove(p));
  groups = []; proxies = [];
  prevState = currState = null;
  sim.packet = packet;
  sim.reset();
  packet.bodies.forEach((b, i) => {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position",
      new THREE.Float32BufferAttribute(b.render_verts.flat(), 3));
    geo.setAttribute("color", new THREE.Float32BufferAttribute(
      b.render_colors.flat().map((c) => c / 255), 3));
    let obj;
    if (b.render_faces && b.render_faces.length && Q.get("points") !== "1") {
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
  loading = false;
}

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
  // grab point in body frame, so it rides the body
  const p = hits[0].point;
  const inv = proxies[body].quaternion.clone().invert();
  const local = p.clone().sub(proxies[body].position).applyQuaternion(inv);
  drag = { body, local, target: p.clone(), p0: p.clone(),
    t0: performance.now(), moved: false };
});
renderer.domElement.addEventListener("pointermove", (e) => {
  if (!drag) return;
  drag.moved = true;
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
const POKE_MS = 260;    // a press shorter than this that moved is a flick

renderer.domElement.addEventListener("pointerup", () => {
  if (drag && drag.moved && performance.now() - drag.t0 < POKE_MS) {
    const rt = sim.rt;
    const dv = drag.target.clone().sub(drag.p0).multiplyScalar(3.0);
    const cap = rt.poke.delta_v[1];
    if (dv.length() > cap) dv.setLength(cap);
    const m = sim.mass[drag.body];
    pendingPoke = { body: drag.body,
      point: worldGrabPoint(drag).toArray(),
      force: dv.multiplyScalar(m / (rt.poke.steps * rt.dt)).toArray(),
      left: rt.poke.steps };
  }
  drag = null;
  controls.enabled = true;
});
renderer.domElement.addEventListener("pointercancel", () => { drag = null; controls.enabled = true; });
renderer.domElement.addEventListener("pointerleave", () => { if (drag) { drag = null; controls.enabled = true; } });
$("reset").onclick = () => { sim?.reset(); prevState = currState = null; syncTransforms(); resetCamera(); };

function worldGrabPoint(d) {
  return d.local.clone()
    .applyQuaternion(proxies[d.body].quaternion)
    .add(proxies[d.body].position);
}

// ------------------------------------------------------------ physics loop
let pendingPoke = null, stepMs = 0, running = false;

async function physicsLoop() {
  if (running) return;
  running = true;
  const rt = sim.rt;
  while (true) {
    if (loading || (dbg.paused && !dbg.stepOnce)) {
      await new Promise((r) => setTimeout(r, 30)); continue;
    }
    dbg.stepOnce = false;
    const t0 = performance.now();
    let act = [-1, null, null];
    if (pendingPoke && pendingPoke.left > 0) {
      act = [pendingPoke.body, pendingPoke.point, pendingPoke.force];
      pendingPoke.left--;
    } else if (drag && performance.now() - drag.t0 >= POKE_MS) {
      const b = drag.body;
      const wp = worldGrabPoint(drag);
      const m = sim.mass[b];
      const kp = m * GRAB.omega ** 2;
      const kd = 2 * GRAB.zeta * m * GRAB.omega;
      const v = sim.state.linvel[b];
      const f = [
        kp * (drag.target.x - wp.x) - kd * v[0],
        kp * (drag.target.y - wp.y) - kd * v[1],
        kp * (drag.target.z - wp.z) - kd * v[2]];
      const cap = rt.grab.force_cap * m * rt.gravity;
      const fn = Math.hypot(...f);
      if (fn > cap) for (let k = 0; k < 3; k++) f[k] *= cap / fn;
      act = [b, wp.toArray(), f];
    }
    try {
      await sim.step(...act);
      syncTransforms();
    } catch (e) {
      err(e);
      await new Promise((r) => setTimeout(r, 500));   // never kill the loop
      continue;
    }
    stepMs = performance.now() - t0;
    stateDt = Math.max(rt.dt, stepMs / 1000);   // interpolation window
    // always yield a real slice to the renderer: the physics kernels share
    // the GPU with WebGL, and back-to-back steps starve the frame output
    const wait = Math.max(12, rt.dt * 1000 - stepMs);
    await new Promise((res) => setTimeout(res, wait));
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
    setInterval(() => {
      const t = sim?.timing;
      const detail = t ? ` [${t.backend}: features ${t.features_ms.toFixed(0)} | graph ` +
        `${t.graph_ms.toFixed(0)} | net ${t.net_ms.toFixed(0)} ms; ` +
        `N=${t.N} E=${t.E}]` : "";
      $("stats").textContent =
        `model step ${runtime.step} | physics ${stepMs.toFixed(0)} ms/step ` +
        `(${(1000 / Math.max(stepMs, 1)).toFixed(0)} Hz capable)${detail}`;
    }, 500);
  } catch (e) { err(e); }
})();

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

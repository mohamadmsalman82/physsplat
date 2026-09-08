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
const controls = new OrbitControls(cam, renderer.domElement);
controls.target.set(0, 0, 0.015);
controls.maxDistance = 1.2;
scene.add(new THREE.AmbientLight(0xffffff, 0.75));
const sun = new THREE.DirectionalLight(0xffffff, 1.4);
sun.position.set(0.6, -0.8, 1.4);
scene.add(sun);
const ground = new THREE.Mesh(
  new THREE.CircleGeometry(0.5, 48),
  new THREE.MeshStandardMaterial({ color: 0x1b1f26, roughness: 0.95 }));
scene.add(ground);
const grid = new THREE.GridHelper(1, 40, 0x2a2f38, 0x20242c);
grid.rotation.x = Math.PI / 2;
grid.position.z = 0.0005;
scene.add(grid);

// ---------------------------------------------------------------- runtime
let sim = null, groups = [], proxies = [];

async function loadScene(name) {
  groups.forEach((g) => scene.remove(g));
  proxies.forEach((p) => scene.remove(p));
  groups = []; proxies = [];
  const packet = await (await fetch(`./packets/${name}.json`)).json();
  sim.packet = packet;
  sim.reset();
  packet.bodies.forEach((b, i) => {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position",
      new THREE.Float32BufferAttribute(b.render_verts.flat(), 3));
    geo.setAttribute("color", new THREE.Float32BufferAttribute(
      b.render_colors.flat().map((c) => c / 255), 3));
    const pts = new THREE.Points(geo, new THREE.PointsMaterial({
      size: 0.0035, vertexColors: true }));
    scene.add(pts);
    groups.push(pts);
    const cap = new THREE.Mesh(
      new THREE.CapsuleGeometry(b.capsule.radius * 1.15, 2 * b.capsule.half, 4, 10),
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
}

function syncTransforms() {
  if (!sim) return;
  sim.state.pos.forEach((p, i) => {
    groups[i].position.set(...p);
    groups[i].quaternion.set(...sim.state.quat[i]);
    proxies[i].position.set(...p);
    proxies[i].quaternion.set(...sim.state.quat[i]);
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
renderer.domElement.addEventListener("pointerup", () => {
  if (drag && drag.moved && performance.now() - drag.t0 < 160) {
    const rt = sim.rt;
    const dv = drag.target.clone().sub(drag.p0).multiplyScalar(2.0);
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
$("reset").onclick = () => { sim?.reset(); syncTransforms(); };

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
    const t0 = performance.now();
    let act = [-1, null, null];
    if (pendingPoke && pendingPoke.left > 0) {
      act = [pendingPoke.body, pendingPoke.point, pendingPoke.force];
      pendingPoke.left--;
    } else if (drag && performance.now() - drag.t0 >= 160) {
      const b = drag.body;
      const wp = worldGrabPoint(drag);
      const m = sim.mass[b];
      const kp = m * rt.grab.omega ** 2;
      const kd = 2 * rt.grab.zeta * m * rt.grab.omega;
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
    } catch (e) { err(e); break; }
    syncTransforms();
    stepMs = performance.now() - t0;
    const wait = Math.max(0, rt.dt * 1000 - stepMs);
    await new Promise((res) => setTimeout(res, wait));
  }
  running = false;
}

// ---------------------------------------------------------------- startup
(async () => {
  try {
    const ortlib = globalThis.ort;
    ortlib.env.wasm.numThreads = Math.min(4, navigator.hardwareConcurrency || 2);
    const runtime = await (await fetch("./model/runtime.json")).json();
    const session = await ortlib.InferenceSession.create(
      "./model/simulator.onnx",
      { executionProviders: ["webgpu", "wasm"] });
    sim = new PhysSim(ortlib, session, runtime, { bodies: [] });
    const names = await (await fetch("./packets/index.json")).json();
    const sel = $("scene");
    names.forEach((n) => sel.add(new Option(n, n)));
    sel.onchange = () => loadScene(sel.value);
    await loadScene(names[0]);
    physicsLoop();
    setInterval(() => {
      $("stats").textContent =
        `model step ${runtime.step} | physics ${stepMs.toFixed(0)} ms/step ` +
        `(${(1000 / Math.max(stepMs, 1)).toFixed(0)} Hz capable)`;
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
  renderer.render(scene, cam);
})();

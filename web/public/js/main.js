/**
 * PhysSplat browser app: rendering, interaction, and the physics loop.
 * Physics runs client-side: PhysSim (js/sim.js) + ONNX Runtime Web.
 */
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { PhysSim } from "./sim.js";
import { RapierSim } from "./rapier_sim.js";
import { Diagnostics, Probes } from "./diag.js";
import { Sensors } from "./sensors.js";
import { buildPencil } from "./pencil_mesh.js";
import { grabForce } from "./physics.js";
import { setupCamera } from "./camera.js";
import { initUI } from "./ui.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";

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
// Filmic tone mapping keeps the white desk from clipping and gives the
// plastic its highlights; an environment map gives it something to reflect.
// Filmic tone mapping is a Display toggle, off by default: the lights are
// tuned so a white desk renders white without it, and ACES pulled the
// pencils' colours toward pastel.
renderer.toneMapping = THREE.NoToneMapping;
renderer.toneMappingExposure = 1.0;
{
  const pmrem = new THREE.PMREMGenerator(renderer);
  scene.environment = pmrem.fromScene(new RoomEnvironment(renderer), 0.04).texture;
}
const CAM_HOME = { pos: [0.16, -0.26, 0.2], target: [0, 0, 0.015] };
const controls = new OrbitControls(cam, renderer.domElement);
controls.target.set(...CAM_HOME.target);
// limits, buttons, keys, double-click framing and eased moves all live in
// js/camera.js; `proxies` and `sim` are read lazily, after they exist
let followSpeed = 0.30;
const camera = setupCamera({
  cam, controls, dom: renderer.domElement,
  getProxies: () => proxies,
  homeAzimuth: () => (sim?.packet?.bodies?.length ? bestAzimuth(sim.packet) : -1.0),
  target: CAM_HOME.target,
  // where the photograph was taken from, when the scene's desk is registered
  photoView: () => deskInfo?.camera ?? null,
});
// A pencil claims the LEFT button before OrbitControls sees the press.
// Capture-phase listeners on the target run first, so by the time
// OrbitControls reads mouseButtons.LEFT it is no longer a rotate, and the
// right button, the middle button and the wheel keep working mid-drag.
renderer.domElement.addEventListener("pointerdown", (e) => {
  if (e.button !== 0 || !sim) return;
  ray.setFromCamera(ndc(e), cam);
  if (ray.intersectObjects(proxies, true).length) camera.holdingPencil(true);
}, true);
const Q = new URLSearchParams(location.search);
renderer.shadowMap.enabled = Q.get("shadows") !== "0";
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
// Lighting to match the photographs: a white desk under soft daylight from
// the upper left, with the pencils' shadows falling to the lower right and
// a lot of bounce off the desk. The previous warm key light and dark
// background were the look of a wooden table in a void, not of this desk.
// Intensities are in three's physical units: ambient and hemisphere light
// are divided by pi in the shader, so for a white desk (albedo 0.56 linear)
// to render at the photograph's brightness the three together have to sum
// to about pi on a horizontal surface. At 0.42 + 0.38 + 1.35 the desk came
// out a dim grey and the pencils dull with it.
const ambient = new THREE.AmbientLight(0xffffff, 0.95);
const hemi = new THREE.HemisphereLight(0xffffff, 0xc8c8cc, 0.80);
scene.add(ambient, hemi);
const sun = new THREE.DirectionalLight(0xffffff, 2.0);
sun.position.set(-0.35, 0.30, 0.85);         // upper left, slightly behind
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
sun.shadow.camera.near = 0.1; sun.shadow.camera.far = 3;
sun.shadow.camera.left = sun.shadow.camera.bottom = -0.4;
sun.shadow.camera.right = sun.shadow.camera.top = 0.4;
sun.shadow.radius = 4;                        // the photo's shadows are soft
// normalBias offsets the shadow lookup along the surface normal, so it has
// to be small next to the object: 20 mm on a 4.5 mm pencil detached every
// shadow from its caster (a blind tester saw "wrongly shaped streaks that
// match no pencil"). 1.5 mm is a third of the radius, enough for the
// curvature and invisible as an offset.
sun.shadow.bias = -0.0001;
sun.shadow.normalBias = 0.0015;
scene.add(sun);

// The desk is the desk in the photograph. scripts/desk_texture.py lifts it
// out of each scene's photo with the pencils and their shadows removed and
// writes where it sits in the world; loadDesk() lays that image on the
// table at that scale, extends it with the desk's own colour beyond the
// photo's edge, stands a wall where the photo has one, and paints the
// background to match, so the scene is the photo with the pencils alive.
let deskGroup = null;
let deskInfo = null;          // the scene's desk JSON, camera included
const deskBase = new THREE.Mesh(
  new THREE.PlaneGeometry(4, 4),
  new THREE.MeshStandardMaterial({ color: 0xbcbfc2, roughness: 0.9, metalness: 0 }));
deskBase.receiveShadow = true;
scene.add(deskBase);
scene.background = new THREE.Color(0x9a9ea3);

/** A soft-edged alpha so the photo fades into the plain desk around it. */
function featherAlpha(w, h, feather = 0.12) {
  const c = document.createElement("canvas");
  c.width = w; c.height = h;
  const g = c.getContext("2d");
  const img = g.createImageData(w, h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const fx = Math.min(x, w - 1 - x) / (w * feather);
    const fy = Math.min(y, h - 1 - y) / (h * feather);
    const a = Math.min(1, fx, fy);
    const s = a * a * (3 - 2 * a);               // smoothstep
    const i = 4 * (y * w + x);
    img.data[i] = img.data[i + 1] = img.data[i + 2] = 255 * s;
    img.data[i + 3] = 255;
  }
  g.putImageData(img, 0, 0);
  const t = new THREE.CanvasTexture(c);
  return t;
}

async function loadDesk(name) {
  if (deskGroup) { scene.remove(deskGroup); deskGroup = null; }
  let d = null;
  try {
    const r = await fetch(`./desk/${name}.json`, { cache: "no-cache" });
    if (r.ok) d = await r.json();
  } catch (e) { /* no desk for this scene: the plain base stays */ }
  // the plain desk beyond the photo takes the photo's BORDER colour, so
  // the seam between them is a colour match rather than a visible edge
  const deskRgb = d?.edge_rgb ?? d?.desk_rgb ?? [188, 191, 194];
  deskBase.material.color.setRGB(deskRgb[0] / 255, deskRgb[1] / 255, deskRgb[2] / 255);
  hemi.groundColor.copy(deskBase.material.color);
  const wallRgb = d?.wall_rgb ?? [150, 154, 160];
  scene.background.setRGB(
    (wallRgb[0] + 2 * deskRgb[0]) / 765, (wallRgb[1] + 2 * deskRgb[1]) / 765, (wallRgb[2] + 2 * deskRgb[2]) / 765);
  deskInfo = d;
  if (!d || !(d.rect_m || d.m_per_px)) return;
  deskGroup = new THREE.Group();
  const tex = await new THREE.TextureLoader().loadAsync(`./desk/${d.image}`);
  tex.colorSpace = THREE.SRGBColorSpace;
  deskTextures = { clean: tex, photo: null };
  displayState.comparePhoto = false;
  tex.anisotropy = renderer.capabilities.getMaxAnisotropy();
  let w, h, cx, cy, alphaMap;
  if (d.rect_m) {
    // registered: the texture IS the desk plane in metres, row 0 at +y
    // (scripts/register_photo.py found the camera; desk_texture.py warped
    // the photograph through it); its alpha is where the photo reached
    const [x0, y0, x1, y1] = d.rect_m;
    w = x1 - x0; h = y1 - y0; cx = (x0 + x1) / 2; cy = (y0 + y1) / 2;
    try { alphaMap = await new THREE.TextureLoader().loadAsync(`./desk/${d.alpha}`); }
    catch (e) { alphaMap = featherAlpha(256, Math.round(256 * h / w), 0.32); }
  } else {
    // unregistered: image right is +x, image up is +y, uniform scale
    const [W, H] = d.image_px, top = d.crop_top_px ?? 0;
    w = W * d.m_per_px; h = (H - top) * d.m_per_px;
    cx = (W / 2 - d.origin_px[0]) * d.m_per_px; cy = -((top + (H - top) / 2) - d.origin_px[1]) * d.m_per_px;
    alphaMap = featherAlpha(256, Math.round(256 * h / w), 0.32);
  }
  const photo = new THREE.Mesh(
    new THREE.PlaneGeometry(w, h),
    new THREE.MeshStandardMaterial({ map: tex, alphaMap, transparent: true, roughness: 0.9, metalness: 0 }));
  photo.position.set(cx, cy, 0.00005);
  photo.receiveShadow = true;
  deskGroup.add(photo);
  // No wall. The photographs' desks end 11 to 16 cm behind the pile, and a
  // vertical plane stood there was a dark slab across half of every
  // elevated view. The desk continues instead, and the background takes a
  // light neutral between the desk and what lay beyond it in the photo.
  deskGroup.visible = displayState.desk;
  scene.add(deskGroup);
}

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

// ------------------------------------------------------------- app API
// What the interface (js/ui.js) is built against. Everything the panels
// can do goes through here, so the physics and the page never know about
// each other's markup.
const listeners = { scene: new Set(), select: new Set(), pause: new Set(), speed: new Set() };
const frameListeners = new Set();
let selected = null;
const displayState = { desk: true, comparePhoto: false, shadows: true, filmic: false,
  labels: false, contacts: false, velocities: false, colliders: false };
const app = {
  scenes: [], scene: null,
  loadScene: (n) => loadScene(n),
  get engine() { return Q.get("engine") === "gnn" ? "gnn" : "rapier"; },
  switchEngine(name) {
    const u = new URL(location.href); u.searchParams.set("engine", name);
    if (app.scene) u.searchParams.set("scene", app.scene);
    location.href = u.toString();
  },
  get sim() { return sim; },
  get sensors() { return dbg.sensors; },
  get diag() { return diag; },
  get paused() { return dbg.paused; },
  pause() { dbg.paused = true; app.emit("pause", true); },
  play() { dbg.paused = false; app.emit("pause", false); },
  step() { dbg.paused = true; dbg.stepOnce = true; app.emit("pause", true); },
  reset() { resetScene(); },
  speed: 1,
  setSpeed(v) { app.speed = Math.max(0.05, Math.min(1, v)); app.emit("speed", app.speed); },
  camera,
  get selected() { return selected; },
  select: (i) => select(i),
  get bodyCount() { return sim?.B ?? 0; },
  bodyInfo(i) {
    if (!dbg.sensors || i === null || i >= sim.B) return null;
    const b = dbg.sensors.body(i);
    b.colorCss = bodyColorCss(i);
    return b;
  },
  project(i) {
    const p = proxies[i];
    if (!p) return { visible: false };
    const v = p.position.clone().project(cam);
    return { x: (v.x + 1) / 2 * innerWidth, y: (1 - v.y) / 2 * innerHeight, visible: v.z < 1 && Math.abs(v.x) < 1.05 && Math.abs(v.y) < 1.05 };
  },
  probe(name, ...args) { return dbg.run(name, ...args); },
  nudge(i) {
    // a short push along the desk, the way a fingertip flicks a pencil
    const dir = new THREE.Vector3(Math.random() - 0.5, Math.random() - 0.5, 0).normalize();
    startFlick(i, [0, 0, 0], dir.toArray(), 0.25, 0.04);
  },
  spawn(colorHex = null) { return spawnPencil(colorHex); },
  display: {
    get: (k) => displayState[k],
    set: (k, v) => { displayState[k] = v; applyDisplay(k); },
  },
  physics: {
    get available() { return !!sim?.setGravityScale; },
    get(k) {
      if (k === "grabStrength") return GRAB.omega;
      if (k === "followSpeed") return followSpeed;
      const p = sim?.params ?? { gravity: 1, frictionPencil: 0.28, frictionDesk: 0.45, restitution: 0.04 };
      return p[k];
    },
    set(k, v) {
      if (k === "grabStrength") { GRAB.omega = v; return; }
      if (k === "followSpeed") { followSpeed = v; return; }
      if (!sim?.setGravityScale) return;
      if (k === "gravity") sim.setGravityScale(v);
      else if (k === "frictionPencil") sim.setFriction(v, sim.params.frictionDesk);
      else if (k === "frictionDesk") sim.setFriction(sim.params.frictionPencil, v);
      else if (k === "restitution") sim.setRestitution(v);
    },
  },
  stats: null,
  onScene: (cb) => listeners.scene.add(cb),
  onSelect: (cb) => listeners.select.add(cb),
  onPause: (cb) => listeners.pause.add(cb),
  onSpeed: (cb) => listeners.speed.add(cb),
  onFrame: (cb) => frameListeners.add(cb),
  emit(kind, v) { for (const cb of listeners[kind] ?? []) { try { cb(v); } catch (e) { console.warn(e); } } },
};
dbg.app = app;

function bodyColorCss(i) {
  const g = groups[i];
  let rgb = null;
  g?.traverse((o) => { if (rgb || !o.isMesh || !o.geometry.getAttribute("color")) return;
    const c = o.geometry.getAttribute("color"); const n = c.count;
    // the barrel is most of the vertices: take the median colour
    const idx = Math.floor(n * 0.5); rgb = [c.getX(idx), c.getY(idx), c.getZ(idx)]; });
  if (!rgb) return "#888";
  return `rgb(${rgb.map((x) => Math.round(Math.sqrt(x) * 255)).join(",")})`;
}

// selection highlight: a faint emissive tint on the pencil itself
let selectedTint = [];
function select(i) {
  if (i !== null && (i < 0 || i >= (sim?.B ?? 0))) i = null;
  for (const [m, e] of selectedTint) m.emissive.setHex(e);
  selectedTint = [];
  selected = i;
  if (i !== null) groups[i]?.traverse((o) => {
    if (o.isMesh && o.material.emissive) { selectedTint.push([o.material, o.material.emissive.getHex()]); o.material.emissive.setHex(0x1b3c5a); }
  });
  app.emit("select", i);
}

/** Drop a new pencil above the pile. Rapier only; the learned engine's graph is fixed at load. */
function spawnPencil(colorHex) {
  if (!sim?.addBody) { app.toast?.("only the Rapier engine can add pencils"); return null; }
  const template = sim.packet.bodies[0];
  const c = new THREE.Color(colorHex ?? `hsl(${Math.floor(Math.random() * 360)}, 55%, 55%)`);
  const colors = template.render_colors.map(() => [Math.round(c.r * 255), Math.round(c.g * 255), Math.round(c.b * 255)]);
  const az = Math.random() * Math.PI, tilt = (Math.random() - 0.5) * 0.3;
  const q = new THREE.Quaternion().setFromEuler(new THREE.Euler(0, tilt, az));
  const pos = [(Math.random() - 0.5) * 0.06, (Math.random() - 0.5) * 0.06, 0.12];
  const i = sim.addBody(template, pos, [q.x, q.y, q.z, q.w], colors);
  buildBodyVisual(sim.packet.bodies[i], i);
  // the diagnostics keep per-body arrays sized at reset; tell them
  diag?.reset("spawn");
  dbg.sensors?.sample(null);
  prevState = currState = null;
  return i;
}

// overlays: contact points, velocity arrows, collider outlines
const overlay = { contacts: new THREE.Group(), velocities: new THREE.Group(), colliders: new THREE.Group() };
Object.values(overlay).forEach((g) => { g.visible = false; scene.add(g); });
// drawn on top of everything: a seam between two pencils is hidden by both
const contactGeo = new THREE.SphereGeometry(0.0018, 12, 10);
const contactMat = new THREE.MeshBasicMaterial({ color: 0xffcf6b, depthTest: false, depthWrite: false });
const contactPenMat = new THREE.MeshBasicMaterial({ color: 0xff5f5f, depthTest: false, depthWrite: false });
overlay.contacts.renderOrder = 10;
let overlayTick = 0;
function updateOverlays() {
  if (!sim || !dbg.sensors) return;
  if (displayState.contacts && (overlayTick++ % 3 === 0)) {
    const g = overlay.contacts; g.clear();
    for (const c of dbg.sensors.touch()) {
      if (!c.touching && c.pen_mm <= 0) continue;
      const m = new THREE.Mesh(contactGeo, c.pen_mm > 0.3 ? contactPenMat : contactMat);
      const w = c.surface ?? c.world;
      m.position.set(w[0] / 1000, w[1] / 1000, w[2] / 1000);
      m.renderOrder = 10;
      g.add(m);
    }
  }
  if (displayState.velocities) {
    const g = overlay.velocities; g.clear();
    for (let b = 0; b < sim.B; b++) {
      const v = sim.state.linvel[b]; const sp = Math.hypot(v[0], v[1], v[2]);
      if (sp < 0.005) continue;
      const dir = new THREE.Vector3(v[0], v[1], v[2]).normalize();
      const len = Math.min(0.12, sp * 0.25);
      g.add(new THREE.ArrowHelper(dir, new THREE.Vector3(...sim.state.pos[b]), len, 0x7cc4ff, len * 0.3, len * 0.18));
    }
  }
  if (displayState.colliders) {
    const g = overlay.colliders;
    if (g.children.length !== sim.B) {
      g.clear();
      for (let b = 0; b < sim.B; b++) g.add(colliderOutline(sim.packet.bodies[b]));
    }
    for (let b = 0; b < sim.B; b++) {
      const o = g.children[b]; if (!o) continue;
      o.position.set(...sim.state.pos[b]); o.quaternion.set(...sim.state.quat[b]);
    }
  }
}
function colliderOutline(b) {
  // the canonical profile as a wire silhouette, plus the clip box
  const pts = [];
  const prof = sim.packet.bodies[0].capsule?.taper ? PENCIL_PROFILE : null;
  const L = 0.150;
  const knots = prof ?? [[0, b.capsule.radius], [1, b.capsule.radius]];
  for (let k = 0; k < 4; k++) {
    const a = (k / 4) * Math.PI;
    for (let i = 1; i < knots.length; i++) {
      const [t0, r0] = knots[i - 1], [t1, r1] = knots[i];
      pts.push((t0 - 0.5) * L, r0 * Math.cos(a), r0 * Math.sin(a), (t1 - 0.5) * L, r1 * Math.cos(a), r1 * Math.sin(a));
      pts.push((t0 - 0.5) * L, -r0 * Math.cos(a), -r0 * Math.sin(a), (t1 - 0.5) * L, -r1 * Math.cos(a), -r1 * Math.sin(a));
    }
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.Float32BufferAttribute(pts, 3));
  const wire = new THREE.LineSegments(geo, new THREE.LineBasicMaterial({ color: 0x6ee7a8, transparent: true, opacity: 0.7 }));
  const clip = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(0.02775, 0.00396, 0.002)),
    new THREE.LineBasicMaterial({ color: 0x6ee7a8, transparent: true, opacity: 0.7 }));
  clip.position.set((-0.5 + 0.125) * L + 0.02775 / 2, 0, 0.0045 + 0.0012 - 0.001);
  const grp = new THREE.Group(); grp.add(wire, clip);
  return grp;
}
let PENCIL_PROFILE = null;
import("./pencil.js").then((m) => { PENCIL_PROFILE = m.PROFILE; });

function applyDisplay(k) {
  const v = displayState[k];
  if (k === "shadows") renderer.shadowMap.enabled = v, scene.traverse((o) => { if (o.material) o.material.needsUpdate = true; });
  else if (k === "desk") { if (deskGroup) deskGroup.visible = v; }
  else if (k === "comparePhoto") swapDeskPhoto(v);
  else if (k === "filmic") { renderer.toneMapping = v ? THREE.ACESFilmicToneMapping : THREE.NoToneMapping; scene.traverse((o) => { if (o.material) o.material.needsUpdate = true; }); }
  else if (k in overlay) { overlay[k].visible = v; if (!v) overlay[k].clear(); }
}
let deskTextures = { clean: null, photo: null };
async function swapDeskPhoto(on) {
  const photo = deskGroup?.children?.[0];
  if (!photo || !app.scene) return;
  if (on && !deskTextures.photo) {
    try {
      const t = await new THREE.TextureLoader().loadAsync(`./desk/${app.scene}_photo.jpg`);
      t.colorSpace = THREE.SRGBColorSpace; t.anisotropy = renderer.capabilities.getMaxAnisotropy();
      deskTextures.photo = t;
    } catch (e) { app.toast?.("no photo for this scene"); return; }
  }
  photo.material.map = on ? deskTextures.photo : deskTextures.clean;
  photo.material.needsUpdate = true;
}

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
  app.scene = name;
  selected = null; app.emit("select", null);
  app.emit("scene", name);
  const packet = await (await fetch(`./packets/${name}.json`, { cache: "no-cache" })).json();
  groups.forEach((g) => scene.remove(g));
  proxies.forEach((p) => scene.remove(p));
  groups = []; proxies = [];
  prevState = currState = null;
  sim.packet = packet;
  sim.reset();
  dbg.scriptDrag = dbg.scriptPoke = null;
  if (!diag) {
    diag = new Diagnostics(sim);
    dbg.probe = new Probes(dbg, sim, diag);
    // full-transparency sensor layer: what is touching what, where on
    // each pencil, and whether anything is moving that should not be
    dbg.sensors = new Sensors(sim);
  }
  else diag.reset(`scene:${name}`);
  packet.bodies.forEach((b, i) => buildBodyVisual(b, i));
  syncTransforms();
  resetCamera();
  loadDesk(name);                       // the photo's own desk, asynchronously
  finishLoad();
}

function buildBodyVisual(b, i) {
  {
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
      obj.traverse((o) => { if (o.isMesh) o.material.envMapIntensity = 0.55; });
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
  }
}

function finishLoad() {
  // A reconstructed pile is not exactly in equilibrium (single-view depth
  // error), and the physics rules move it into one during the first few
  // steps. Run those steps before showing motion, so the scene appears
  // already settled instead of twitching on load.
  preroll = PREROLL_STEPS;
  loading = false;
}
// 3 s. Sleep now waits until a body's pose is resolved against the floor
// or a neighbour, so the pile has to be given time to get there before it
// is shown; at 0.75 s a blind tester caught scenes frozen mid-settle with
// pencils millimetres in the air.
// The packets ship already settled (web/test/settle_packets.mjs runs the
// same model over each scene until every body is at rest and writes the
// pose back), so there is nothing to hide any more. This was 180, which is
// 3 s of simulated time but 6 to 12 s of wall time at 35 to 60 ms a step:
// every page load and every press of reset showed a frozen, visibly wrong
// pile for ten seconds and then snapped every pencil up to 26 mm into
// place. A dozen steps is insurance against a packet that was edited by
// hand and never re-settled, and it is imperceptible.
const PREROLL_STEPS = 12;
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

/**
 * Physics steps land irregularly (24-73 ms), so lerping between the last
 * two states over a fixed window reaches the newer one and then freezes
 * until the next arrives: the stutter a player reads as jitter. Chase the
 * latest state exponentially instead, at a rate set by the frame time, so
 * the drawn pose is continuous whatever the step timing does.
 */
const _qb = new THREE.Quaternion();
const RENDER_TAU = 0.045;         // seconds to close ~63% of the gap
let lastFrame = performance.now();
function renderInterpolated() {
  if (!currState || groups.length !== currState.pos.length) return;
  const now = performance.now();
  const dt = Math.min(0.1, (now - lastFrame) / 1000);
  lastFrame = now;
  const k = 1 - Math.exp(-dt / RENDER_TAU);
  currState.pos.forEach((p, i) => {
    const g = groups[i];
    if (!g.userData.shown) {            // first frame of a scene: snap
      g.position.set(p[0], p[1], p[2]);
      g.quaternion.set(...currState.quat[i]);
      g.userData.shown = true;
      return;
    }
    g.position.x += (p[0] - g.position.x) * k;
    g.position.y += (p[1] - g.position.y) * k;
    g.position.z += (p[2] - g.position.z) * k;
    _qb.set(...currState.quat[i]);
    g.quaternion.slerp(_qb, k);
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
  if (e.button !== 0) return;                 // right and middle stay the camera's
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
// the grab target's own speed limit, the speed grabs were generated at
const FOLLOW_SPEED = 0.30;
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
    // a press that never travelled is a click: select
    if (drag.px <= FLICK_MIN_PIXELS) select(drag.body);
  }
  drag = null;
  camera.holdingPencil(false);
});
renderer.domElement.addEventListener("pointercancel", () => { drag = null; camera.holdingPencil(false); });
renderer.domElement.addEventListener("pointerleave", () => { if (drag) { drag = null; camera.holdingPencil(false); } });

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
  groups.forEach((g) => { g.userData.shown = false; });   // snap, don't slide back
}
($("reset") ?? {}).onclick = resetScene;

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
      // Follow the cursor through a rate-limited point rather than
      // snapping to it. The physics runs at ~0.5x real time, so between
      // steps the cursor can jump centimetres; feeding that straight to
      // the spring gave a hard yank, an overshoot and a visible shake.
      // The follow point moves at most FOLLOW_SPEED, which is the speed a
      // grab was trained to move at, so what the model sees stays in
      // distribution too.
      drag.follow ??= drag.p0.clone();
      const step = drag.target.clone().sub(drag.follow);
      const maxStep = followSpeed * rt.dt;
      if (step.length() > maxStep) step.setLength(maxStep);
      drag.follow.add(step);
      const f = springForce(drag.body, wp, drag.follow.toArray());
      act = [drag.body, wp, f, true];      // a held grab is a pinch
      actInfo = { kind: "grab", body: drag.body, force: f, point: wp, target: drag.follow.toArray() };
    } else if (flick) {
      // carry the grab point along the flick at its speed, then release
      flick.travelled += flick.speed * rt.dt;
      const target = flick.from.clone().add(flick.dir.clone().multiplyScalar(flick.travelled));
      const wp = new THREE.Vector3(...flick.local)
        .applyQuaternion(new THREE.Quaternion(...sim.state.quat[flick.body]))
        .add(new THREE.Vector3(...sim.state.pos[flick.body])).toArray();
      const f = springForce(flick.body, wp, target.toArray());
      act = [flick.body, wp, f, true];     // carried on the spring, same
      actInfo = { kind: "flick", body: flick.body, force: f, point: wp, target: target.toArray() };
      if (flick.travelled >= flick.dist) flick = null;
    } else if (dbg.scriptDrag) {
      const d = dbg.scriptDrag;
      const R = sim.state.quat[d.body];
      const wp = new THREE.Vector3(...d.local).applyQuaternion(new THREE.Quaternion(...R))
        .add(new THREE.Vector3(...sim.state.pos[d.body])).toArray();
      const f = springForce(d.body, wp, d.target);
      act = [d.body, wp, f, true];
      actInfo = { kind: "grab", body: d.body, force: f, point: wp, target: [...d.target], scripted: true };
    }
    try {
      // The fourth element matters. sim.step's actPinch gates the pinch
      // damper and both held-speed clamps, and for a long time this call
      // passed three elements, so none of them ever ran in the browser while
      // the regression suite, which passes true, kept passing. Measured by
      // the sensor agent on the live build: a grab with the cursor held
      // still reared a pencil to 89.9 deg at 3.2 m/s and 60 rad/s, drove it
      // 31 mm through the table, and left it standing on its end. That was
      // the rearing, the runaway, and most of the phasing people reported.
      await sim.step(...act);
      diag?.record(actInfo);
      dbg.sensors?.sample(actInfo);
      if (preroll > 0 && --preroll === 0) { prevState = currState = null; }
      // draw every step, pre-roll included. Skipping the draw meant the last
      // thing on screen during pre-roll was the raw loaded pose, so the user
      // watched a stale pile sit still and then jump.
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
    // slow motion is a smaller step rate against the same wall clock: the
    // physics step itself is unchanged, so it is real physics, only slower
    simClock += rt.dt * 1000 / app.speed;
    const elapsed = () => performance.now() - clockStart;
    if (simClock - elapsed() < -500) simClock = elapsed();      // fell behind: resync
    // And never AHEAD. yieldSlice skips timers in a hidden tab so scripted
    // probes keep running there, which was harmless while a step cost 33 ms
    // and the loop could never outrun the wall clock. A rigid-body step
    // costs 1 to 2 ms, so a hidden tab ran at ten times real time, sim time
    // raced minutes ahead, and when the tab was shown again the loop waited
    // for the wall clock to catch up: zero steps, and pencils that could not
    // be moved. Sim time may lead by at most a frame or two; a hidden tab
    // paces itself on message ping-pong instead of timers.
    if (simClock - elapsed() > 40) simClock = elapsed() + 40;
    if (document.hidden) {
      while (simClock > elapsed()) await yieldSlice(0);
    } else {
      // and a real slice for the renderer: the learned engine's kernels share
      // the GPU with WebGL and back-to-back steps starve the frame output
      const floor = sim.backend.kind === "rapier" ? 4 : 12;
      await yieldSlice(Math.max(floor, simClock - elapsed()));
    }
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
    // The engine. Rapier, a real rigid-body solver, by default; the learned
    // graph network behind ?engine=gnn. Both present the same interface, so
    // everything after this line is the same code. See js/rapier_sim.js for
    // why the solver is the default.
    const engine = q.get("engine") === "gnn" ? "gnn" : "rapier";
    let backend = null;
    if (engine === "rapier") {
      const mod = await import("../vendor/rapier.mjs");
      const R = mod.default ?? mod;
      await R.init();
      sim = new RapierSim(R, runtime, { bodies: [] });
      console.log("[physsplat] engine: rapier " + (R.version ? R.version() : ""));
    }
    if (engine === "gnn" && navigator.gpu && q.get("backend") !== "ort") {
      try {
        const { GpuNet } = await import("./gpu_net.js");
        const net = await GpuNet.create("./model/weights.json", "./model/weights.bin");
        backend = { kind: "gpu", net };
      } catch (e) { console.warn("[physsplat] WebGPU backend unavailable:", e.message); }
    }
    if (engine === "gnn" && !backend) {
      const ortlib = globalThis.ort;
      ortlib.env.wasm.numThreads = Math.min(4, navigator.hardwareConcurrency || 2);
      ortlib.env.wasm.simd = true;
      const session = await ortlib.InferenceSession.create(
        "./model/simulator.onnx", { executionProviders: ["wasm"] });
      backend = { kind: "ort", ort: ortlib, session };
    }
    if (engine === "gnn") {
      console.log(`[physsplat] backend: ${backend.kind}`);
      sim = new PhysSim(backend, runtime, { bodies: [] });
    }
    const names = await (await fetch("./packets/index.json")).json();
    app.scenes = names;
    const first = names.includes(q.get("scene")) ? q.get("scene") : names[0];
    app.scene = first;
    await loadScene(first);
    physicsLoop();
    initDiagPanel();
    initUI(app);
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
      app.stats = { engine: t?.backend === "rapier" ? "Rapier" : (t?.backend ?? "…"), stepMs, rate, steps: sim?.stepCount ?? 0,
        N: t?.N, E: t?.E, stalled: !loading && since > 3000, detail: `${detail}${state}` };
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
  camera.update();                 // keys and eased moves, before damping
  controls.update();
  updateOverlays();
  for (const cb of frameListeners) { try { cb(); } catch (e) { /* a UI listener must never stop the frame */ } }
  renderInterpolated();
  renderer.render(scene, cam);
})();

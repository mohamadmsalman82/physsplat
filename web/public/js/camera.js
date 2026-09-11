/**
 * Camera mechanics on top of OrbitControls.
 *
 * OrbitControls does the dragging and the wheel; this adds what it lacks
 * for a sandbox you play with rather than merely look at:
 *
 *   orbit and zoom while HOLDING a pencil. The left button belongs to the
 *   pencil once it has hit one, but the right button, the middle button and
 *   the wheel stay the camera's, so you can turn the scene around a pencil
 *   you are carrying;
 *
 *   presets: home (a three-quarter view chosen so no pencil is seen end-on),
 *   top, side, low (table level), and photo (the viewpoint the photograph
 *   was taken from, since the desk under the pencils IS that photograph);
 *
 *   follow: the orbit centre tracks a chosen pencil, so the next orbit and
 *   zoom are about it and it stays in frame while you carry it;
 *
 *   auto-orbit: a slow turn about the centre for looking, not doing;
 *
 *   keys: arrows or WASD orbit, Q and E zoom, T looks straight down, F
 *   frames the pencil under the cursor (or the pile), R goes home. All of
 *   them are rate-based and eased, so holding a key glides rather than
 *   stepping;
 *
 *   double-click a pencil to bring the orbit centre to it.
 *
 * Every move that is not driven by the hand is tweened over a third of a
 * second instead of cutting. The orbit centre is kept just above the
 * table. Panning is in the table plane, not the screen plane, so the desk
 * never tilts away underfoot.
 */
import * as THREE from "three";

const KEY_ORBIT = 1.6;      // rad/s at full press
const KEY_ZOOM = 0.9;       // fraction of distance per second
const TWEEN_S = 0.35;
const FOLLOW_EASE = 8;      // 1/s, how fast the centre catches a followed pencil

// presets as [elevation rad, distance m, azimuth: number (rad) or "keep"/"home"]
const PRESETS = {
  home: { elev: 0.62, dist: 0.36, az: "home" },
  top: { elev: 1.45, dist: 0.34, az: "keep" },
  side: { elev: 0.16, dist: 0.34, az: "keep" },
  low: { elev: 0.07, dist: 0.26, az: "keep" },
  // the photographs were taken from the -y side of the desk, phone tilted
  // about 55 degrees down, close enough that the pencils filled the frame
  photo: { elev: 0.96, dist: 0.40, az: -Math.PI / 2 },
};

export function setupCamera({ cam, controls, dom, getProxies, homeAzimuth, target }) {
  controls.zoomToCursor = true;
  controls.screenSpacePanning = false;        // pan along the table
  controls.minPolarAngle = 0.08;              // almost straight down
  controls.maxPolarAngle = 1.48;              // low, but never under the desk
  controls.minDistance = 0.06;
  controls.maxDistance = 1.4;
  controls.rotateSpeed = 0.6;
  controls.zoomSpeed = 1.0;
  controls.panSpeed = 0.7;
  controls.enableDamping = true;
  controls.dampingFactor = 0.1;
  controls.mouseButtons = {
    LEFT: THREE.MOUSE.ROTATE, MIDDLE: THREE.MOUSE.PAN, RIGHT: THREE.MOUSE.ROTATE,
  };
  controls.touches = { ONE: THREE.TOUCH.ROTATE, TWO: THREE.TOUCH.DOLLY_PAN };
  dom.addEventListener("contextmenu", (e) => e.preventDefault());

  const keys = new Set();
  let tween = null;
  let following = null;        // body index or null
  let orbiting = false;
  let orbitSpeed = 0.25;       // rad/s
  const sph = new THREE.Spherical();
  const off = new THREE.Vector3();
  const listeners = new Set();
  const emit = () => { for (const l of listeners) l(state()); };

  /** The left button is the pencil's for this drag; keep the rest. */
  function holdingPencil(on) {
    controls.mouseButtons.LEFT = on ? -1 : THREE.MOUSE.ROTATE;
  }

  function spherical() {
    off.copy(cam.position).sub(controls.target);
    sph.setFromVector3(off);
    return sph;
  }
  const azimuthNow = () => Math.atan2(cam.position.y - controls.target.y, cam.position.x - controls.target.x);
  const elevationNow = () => {
    const d = cam.position.clone().sub(controls.target);
    return Math.asin(Math.max(-1, Math.min(1, d.z / Math.max(d.length(), 1e-6))));
  };

  function goTo(pos, tgt) {
    tween = { t: 0, p0: cam.position.clone(), p1: pos.clone(),
      c0: controls.target.clone(), c1: tgt.clone() };
  }

  function placeAt(elev, dist, az, tgt) {
    return new THREE.Vector3(
      tgt.x + dist * Math.cos(elev) * Math.cos(az),
      tgt.y + dist * Math.cos(elev) * Math.sin(az),
      tgt.z + dist * Math.sin(elev));
  }

  function centreOfPile() {
    const proxies = getProxies();
    const c = new THREE.Vector3();
    if (!proxies.length) return c.set(...target);
    for (const p of proxies) c.add(p.position);
    c.divideScalar(proxies.length);
    c.z = Math.max(c.z, 0.01);
    return c;
  }

  /** Move to a named preset, eased. Keeps the current centre. */
  function preset(name) {
    const p = PRESETS[name];
    if (!p) return;
    const az = p.az === "home" ? homeAzimuth() : p.az === "keep" ? azimuthNow() : p.az;
    const tgt = name === "home" || name === "photo" ? centreOfPile() : controls.target.clone();
    goTo(placeAt(p.elev, p.dist, az, tgt), tgt);
    emit();
  }
  const home = () => preset("home");
  const top = () => preset("top");

  /** Frame a body, or the whole pile if none is given. */
  function frame(body = null) {
    const proxies = getProxies();
    const centre = body !== null && proxies[body] ? proxies[body].position.clone() : centreOfPile();
    centre.z = Math.max(centre.z, 0.01);
    const delta = cam.position.clone().sub(controls.target);
    const d = body !== null ? Math.min(delta.length(), 0.22) : delta.length();
    delta.setLength(d);
    goTo(centre.clone().add(delta), centre);
  }

  /** Follow a body (null to stop): the orbit centre tracks it. */
  function follow(body) {
    following = body;
    if (body !== null) frame(body);
    emit();
  }

  function autoOrbit(on, speed = orbitSpeed) {
    orbiting = !!on; orbitSpeed = speed;
    emit();
  }

  function setFov(deg) {
    cam.fov = Math.max(20, Math.min(90, deg));
    cam.updateProjectionMatrix();
    emit();
  }

  function state() {
    return {
      following, orbiting, orbitSpeed, fov: cam.fov,
      elevation_deg: elevationNow() * 180 / Math.PI,
      azimuth_deg: azimuthNow() * 180 / Math.PI,
      distance_m: cam.position.distanceTo(controls.target),
    };
  }

  addEventListener("keydown", (e) => {
    if (e.target && /INPUT|SELECT|TEXTAREA/.test(e.target.tagName)) return;
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    const k = e.key.toLowerCase();
    if (k === "r") { follow(null); home(); return; }
    if (k === "t") { top(); return; }
    if (k === "f") { frame(hoverBody); return; }
    if (k === "1") { preset("home"); return; }
    if (k === "2") { preset("top"); return; }
    if (k === "3") { preset("side"); return; }
    if (k === "4") { preset("low"); return; }
    if (k === "5") { preset("photo"); return; }
    if (k === "o") { autoOrbit(!orbiting); return; }
    if (["arrowleft", "arrowright", "arrowup", "arrowdown", "w", "a", "s", "d", "q", "e"].includes(k)) {
      keys.add(k); e.preventDefault();
    }
  });
  addEventListener("keyup", (e) => keys.delete(e.key.toLowerCase()));
  addEventListener("blur", () => keys.clear());
  // any hand-driven move ends auto-orbit
  controls.addEventListener("start", () => { if (orbiting) { orbiting = false; emit(); } });

  // which pencil the cursor is over, for F and for double-click
  let hoverBody = null;
  const ray = new THREE.Raycaster();
  const ndc = (e) => new THREE.Vector2((e.clientX / innerWidth) * 2 - 1, -(e.clientY / innerHeight) * 2 + 1);
  dom.addEventListener("pointermove", (e) => {
    ray.setFromCamera(ndc(e), cam);
    const hits = ray.intersectObjects(getProxies(), true);
    hoverBody = hits.length ? hits[0].object.parent.userData.body : null;
  });
  dom.addEventListener("dblclick", (e) => {
    ray.setFromCamera(ndc(e), cam);
    const hits = ray.intersectObjects(getProxies(), true);
    frame(hits.length ? hits[0].object.parent.userData.body : null);
  });

  let lastT = performance.now();
  /** Call once per rendered frame, before controls.update(). */
  function update() {
    const now = performance.now();
    const dt = Math.min(0.05, (now - lastT) / 1000);
    lastT = now;

    if (tween) {
      tween.t += dt / TWEEN_S;
      const u = tween.t >= 1 ? 1 : 1 - Math.pow(1 - tween.t, 3);   // ease out
      cam.position.lerpVectors(tween.p0, tween.p1, u);
      controls.target.lerpVectors(tween.c0, tween.c1, u);
      if (tween.t >= 1) tween = null;
      return;
    }
    // follow: ease the centre onto the body, carrying the camera with it
    if (following !== null) {
      const p = getProxies()[following];
      if (p) {
        const want = p.position.clone(); want.z = Math.max(want.z, 0.01);
        const shift = want.sub(controls.target).multiplyScalar(Math.min(1, FOLLOW_EASE * dt));
        controls.target.add(shift); cam.position.add(shift);
      } else following = null;
    }
    if (orbiting) {
      const s = spherical();
      const az = azimuthNow() + orbitSpeed * dt, el = elevationNow(), r = s.radius;
      cam.position.copy(placeAt(el, r, az, controls.target));
    }
    if (!keys.size) return;
    const s = spherical();
    let dAz = 0, dEl = 0, zoom = 0;
    if (keys.has("arrowleft") || keys.has("a")) dAz += 1;
    if (keys.has("arrowright") || keys.has("d")) dAz -= 1;
    if (keys.has("arrowup") || keys.has("w")) dEl += 1;
    if (keys.has("arrowdown") || keys.has("s")) dEl -= 1;
    if (keys.has("q")) zoom += 1;
    if (keys.has("e")) zoom -= 1;
    const r = s.radius * (1 - zoom * KEY_ZOOM * dt);
    let az = azimuthNow() + dAz * KEY_ORBIT * dt;
    let el = elevationNow() + dEl * KEY_ORBIT * dt;
    el = Math.max(Math.PI / 2 - controls.maxPolarAngle, Math.min(Math.PI / 2 - controls.minPolarAngle, el));
    const rr = Math.max(controls.minDistance, Math.min(controls.maxDistance, r));
    cam.position.copy(placeAt(el, rr, az, controls.target));
  }

  return {
    update, home, top, frame, preset, follow, autoOrbit, setFov, holdingPencil, state,
    presets: Object.keys(PRESETS),
    onChange(l) { listeners.add(l); return () => listeners.delete(l); },
    get hoverBody() { return hoverBody; },
    get following() { return following; },
  };
}

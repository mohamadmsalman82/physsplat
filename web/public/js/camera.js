/**
 * Camera mechanics on top of OrbitControls.
 *
 * OrbitControls does the dragging and the wheel; this adds what it lacks
 * for a sandbox you play with rather than merely look at:
 *
 *   orbit and zoom while HOLDING a pencil. The left button belongs to the
 *   pencil once it has hit one, but the right button, the middle button and
 *   the wheel stay the camera's, so you can turn the scene around a pencil
 *   you are carrying. Before this the camera was switched off for the whole
 *   drag, which a blind tester called out and which a person feels as the
 *   view being stuck exactly when they want to look closer;
 *
 *   keys: arrows or WASD orbit, Q and E zoom, T looks straight down, F
 *   frames the pencil under the cursor (or the pile), R goes home. All of
 *   them are rate-based and eased, so holding a key glides rather than
 *   stepping;
 *
 *   double-click a pencil to bring the orbit centre to it, so the next
 *   orbit and zoom are about that pencil;
 *
 *   moves that are not driven by the hand, home and frame and top, are
 *   tweened over a third of a second instead of cutting.
 *
 * The orbit centre is kept just above the table. Panning is in the table
 * plane, not the screen plane, so the desk never tilts away underfoot.
 */
import * as THREE from "three";

const KEY_ORBIT = 1.6;      // rad/s at full press
const KEY_ZOOM = 0.9;       // fraction of distance per second
const TWEEN_S = 0.35;
const HOME_ELEV = 0.62, HOME_DIST = 0.36;
const TOP_ELEV = 1.45;      // not quite vertical: keeps a sense of which way is up

export function setupCamera({ cam, controls, dom, getProxies, homeAzimuth, target }) {
  controls.zoomToCursor = true;
  controls.screenSpacePanning = false;        // pan along the table
  controls.minPolarAngle = 0.08;              // almost straight down
  controls.maxPolarAngle = 1.38;              // low, but never under the desk
  controls.minDistance = 0.08;
  controls.maxDistance = 1.2;
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
  const sph = new THREE.Spherical();
  const off = new THREE.Vector3();

  /** The left button is the pencil's for this drag; keep the rest. */
  function holdingPencil(on) {
    controls.mouseButtons.LEFT = on ? -1 : THREE.MOUSE.ROTATE;
  }

  function spherical() {
    off.copy(cam.position).sub(controls.target);
    sph.setFromVector3(off);
    return sph;
  }

  function goTo(pos, tgt) {
    tween = { t: 0, p0: cam.position.clone(), p1: pos.clone(),
      c0: controls.target.clone(), c1: tgt.clone() };
  }

  function home() {
    const az = homeAzimuth();
    const p = new THREE.Vector3(
      HOME_DIST * Math.cos(HOME_ELEV) * Math.cos(az),
      HOME_DIST * Math.cos(HOME_ELEV) * Math.sin(az),
      HOME_DIST * Math.sin(HOME_ELEV));
    goTo(p, new THREE.Vector3(...target));
  }

  function top() {
    const s = spherical();
    const az = Math.atan2(cam.position.y - controls.target.y, cam.position.x - controls.target.x);
    const d = Math.max(0.2, s.radius);
    const p = new THREE.Vector3(
      controls.target.x + d * Math.cos(TOP_ELEV) * Math.cos(az),
      controls.target.y + d * Math.cos(TOP_ELEV) * Math.sin(az),
      controls.target.z + d * Math.sin(TOP_ELEV));
    goTo(p, controls.target);
  }

  /** Frame a body, or the whole pile if none is given. */
  function frame(body = null) {
    const proxies = getProxies();
    const centre = new THREE.Vector3();
    if (body !== null && proxies[body]) {
      centre.copy(proxies[body].position);
    } else if (proxies.length) {
      for (const p of proxies) centre.add(p.position);
      centre.divideScalar(proxies.length);
    } else {
      centre.set(...target);
    }
    centre.z = Math.max(centre.z, 0.01);
    const delta = cam.position.clone().sub(controls.target);
    const d = body !== null ? Math.min(delta.length(), 0.22) : delta.length();
    delta.setLength(d);
    goTo(centre.clone().add(delta), centre);
  }

  addEventListener("keydown", (e) => {
    if (e.target && /INPUT|SELECT|TEXTAREA/.test(e.target.tagName)) return;
    const k = e.key.toLowerCase();
    if (k === "r") { home(); return; }
    if (k === "t") { top(); return; }
    if (k === "f") { frame(hoverBody); return; }
    if (["arrowleft", "arrowright", "arrowup", "arrowdown", "w", "a", "s", "d", "q", "e"].includes(k)) {
      keys.add(k); e.preventDefault();
    }
  });
  addEventListener("keyup", (e) => keys.delete(e.key.toLowerCase()));
  addEventListener("blur", () => keys.clear());

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
    if (!keys.size) return;
    const s = spherical();
    let dAz = 0, dEl = 0, zoom = 0;
    if (keys.has("arrowleft") || keys.has("a")) dAz += 1;
    if (keys.has("arrowright") || keys.has("d")) dAz -= 1;
    if (keys.has("arrowup") || keys.has("w")) dEl += 1;
    if (keys.has("arrowdown") || keys.has("s")) dEl -= 1;
    if (keys.has("q")) zoom += 1;
    if (keys.has("e")) zoom -= 1;
    // three's Spherical is about +y; the scene is z-up, so work in our own
    // azimuth/elevation and write the position back directly
    const r = s.radius * (1 - zoom * KEY_ZOOM * dt);
    let az = Math.atan2(off.y, off.x) + dAz * KEY_ORBIT * dt;
    let el = Math.asin(Math.max(-1, Math.min(1, off.z / Math.max(s.radius, 1e-6)))) + dEl * KEY_ORBIT * dt;
    el = Math.max(Math.PI / 2 - controls.maxPolarAngle, Math.min(Math.PI / 2 - controls.minPolarAngle, el));
    const rr = Math.max(controls.minDistance, Math.min(controls.maxDistance, r));
    cam.position.set(
      controls.target.x + rr * Math.cos(el) * Math.cos(az),
      controls.target.y + rr * Math.cos(el) * Math.sin(az),
      controls.target.z + rr * Math.sin(el));
  }

  return { update, home, top, frame, holdingPencil, get hoverBody() { return hoverBody; } };
}

/**
 * A pencil that looks like a pencil.
 *
 * The physics body is a capsule (axis, half length, radius) fitted to the
 * reconstruction, and the packet carries a capsule mesh coloured from the
 * photo. A blind tester rated the capsules "smooth shaded pills; nothing
 * identifies them as pencils". This builds, inside the same physical
 * envelope, the parts a BIC Matic Grip has: a barrel that keeps the
 * photo's colour bands, a rubber grip band, a metal cone tip, and a clip
 * at the eraser end. Nothing here is seen by the physics.
 */
import * as THREE from "three";

const TIP_LEN = 0.009;        // metal cone, m
const GRIP_LEN = 0.028, GRIP_FROM_TIP = 0.010, GRIP_BULGE = 0.0003;
const CLIP_LEN = 0.024, CLIP_W = 0.0026, CLIP_T = 0.0009, CLIP_FROM_END = 0.004;

/** Nearest packet vertex colour for a body-frame point (the bands vary
 * along the axis, so nearest-by-position keeps grip / barrel / cap tones). */
function makeSampler(verts, colors) {
  const n = verts.length;
  return (p) => {
    let best = 0, bd = Infinity;
    for (let i = 0; i < n; i++) {
      const v = verts[i];
      const d = (v[0] - p[0]) ** 2 + (v[1] - p[1]) ** 2 + (v[2] - p[2]) ** 2;
      if (d < bd) { bd = d; best = i; }
    }
    const c = colors[best];
    return [c[0] / 255, c[1] / 255, c[2] / 255];
  };
}

function colorize(geo, sampler, transform, tint = 1) {
  const pos = geo.getAttribute("position");
  const out = new Float32Array(pos.count * 3);
  const p = new THREE.Vector3();
  for (let i = 0; i < pos.count; i++) {
    p.fromBufferAttribute(pos, i).applyMatrix4(transform);
    const c = sampler([p.x, p.y, p.z]);
    out[3 * i] = c[0] * tint; out[3 * i + 1] = c[1] * tint; out[3 * i + 2] = c[2] * tint;
  }
  geo.setAttribute("color", new THREE.BufferAttribute(out, 3));
}

/**
 * body: a packet body ({capsule:{axis,half,radius}, render_verts,
 * render_colors}). Returns a THREE.Group in the body frame.
 */
export function buildPencil(body) {
  const cap = body.capsule;
  const r = cap.radius, half = cap.half;
  const L = 2 * half + 2 * r;                       // full physical length
  const axis = new THREE.Vector3(...cap.axis).normalize();
  const sampler = makeSampler(body.render_verts, body.render_colors);

  // which end is the tip: the darker end (metal cone + dark grip vs the
  // coloured eraser cap). A wrong guess still looks like a pencil.
  let dark = 0, bright = 0, nd = 0, nb = 0;
  body.render_verts.forEach((v, i) => {
    const c = body.render_colors[i], lum = c[0] + c[1] + c[2];
    const along = v[0] * axis.x + v[1] * axis.y + v[2] * axis.z;
    if (along > 0) { bright += lum; nb++; } else { dark += lum; nd++; }
  });
  const tipSign = (nb && nd && bright / nb < dark / nd) ? 1 : -1;

  const g = new THREE.Group();
  // local frame: pencil along +y (three.js cylinders are y-aligned), tip at +y
  const frame = new THREE.Quaternion().setFromUnitVectors(
    new THREE.Vector3(0, 1, 0), axis.clone().multiplyScalar(tipSign));
  const M = new THREE.Matrix4().makeRotationFromQuaternion(frame);
  const mat = (extra = {}) => new THREE.MeshStandardMaterial({
    vertexColors: true, roughness: 0.55, metalness: 0.05, ...extra });

  // barrel: cylinder from the eraser hemisphere to the cone base
  const eraserEnd = -L / 2 + r, coneBase = L / 2 - TIP_LEN;
  const barrelLen = coneBase - eraserEnd;
  // closed, not open-ended: with front-face culling an open tube lets you
  // see straight through the barrel to whatever is behind it, which a
  // blind tester read as "the pencils are semi-transparent"
  const barrel = new THREE.CylinderGeometry(r, r, barrelLen, 28, 1, false);
  barrel.translate(0, eraserEnd + barrelLen / 2, 0);
  colorize(barrel, sampler, M);
  g.add(new THREE.Mesh(barrel, mat()));

  // eraser-end hemisphere (photo colour)
  const dome = new THREE.SphereGeometry(r, 24, 12, 0, Math.PI * 2, Math.PI / 2, Math.PI / 2);
  dome.translate(0, eraserEnd, 0);
  colorize(dome, sampler, M);
  g.add(new THREE.Mesh(dome, mat()));

  // rubber grip: a slightly proud band, the photo colour darkened
  const grip = new THREE.CylinderGeometry(r + GRIP_BULGE, r + GRIP_BULGE, GRIP_LEN, 28, 1, false);
  grip.translate(0, coneBase - GRIP_FROM_TIP - GRIP_LEN / 2, 0);
  colorize(grip, sampler, M, 0.6);
  g.add(new THREE.Mesh(grip, mat({ roughness: 0.9 })));

  // metal cone tip with a dark lead point
  const cone = new THREE.CylinderGeometry(0.0007, r * 0.95, TIP_LEN, 24, 1, false);
  cone.translate(0, coneBase + TIP_LEN / 2, 0);
  const cc = new Float32Array(cone.getAttribute("position").count * 3).fill(0.8);
  cone.setAttribute("color", new THREE.BufferAttribute(cc, 3));
  g.add(new THREE.Mesh(cone, mat({ roughness: 0.35, metalness: 0.7 })));
  const lead = new THREE.CylinderGeometry(0.0004, 0.0004, 0.0015, 8);
  lead.translate(0, L / 2 + 0.0005, 0);
  const lc = new Float32Array(lead.getAttribute("position").count * 3).fill(0.12);
  lead.setAttribute("color", new THREE.BufferAttribute(lc, 3));
  g.add(new THREE.Mesh(lead, mat({ roughness: 0.6 })));

  // clip at the eraser end, lying along the barrel
  const clip = new THREE.BoxGeometry(CLIP_W, CLIP_LEN, CLIP_T);
  clip.translate(0, eraserEnd + CLIP_FROM_END + CLIP_LEN / 2, r + CLIP_T / 2);
  const kc = new Float32Array(clip.getAttribute("position").count * 3);
  const base = sampler([0, 0, 0]);
  for (let i = 0; i < kc.length; i += 3) { kc[i] = base[0] * 0.9 + 0.1; kc[i + 1] = base[1] * 0.9 + 0.1; kc[i + 2] = base[2] * 0.9 + 0.1; }
  clip.setAttribute("color", new THREE.BufferAttribute(kc, 3));
  g.add(new THREE.Mesh(clip, mat({ roughness: 0.4, metalness: 0.3 })));

  g.quaternion.copy(frame);
  g.traverse((o) => { if (o.isMesh) { o.castShadow = true; o.receiveShadow = true; } });
  // the renderer drives the outer group's pose every frame; the inner group
  // keeps the fixed y-to-axis rotation of the parts
  const outer = new THREE.Group();
  outer.add(g);
  return outer;
}

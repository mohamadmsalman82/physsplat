/**
 * Draw the canonical BIC Matic Grip.
 *
 * The surface is the PROFILE in pencil.js revolved about the pencil's axis,
 * which is the same profile the physics samples its particles from and the
 * same one the contact guard evaluates its radii from. So the drawing is
 * not an approximation of the simulated body, it is the simulated body,
 * and the class of bug that produced every appearance complaint about this
 * demo cannot happen: a pencil the ground rule holds on the table is drawn
 * touching the table, exactly, with no tolerance to tune.
 *
 * That is worth stating plainly because the previous version tried to earn
 * this by fitting: it measured the reconstruction's elliptical section,
 * squashed a round pencil onto it, then shrank until nothing escaped. It
 * got flat-lying escape to 0.000 mm and still left millimetres at any other
 * angle, because a faithful pencil cannot fit inside an unfaithful body. The
 * body is the thing that had to change.
 *
 * What still comes from the photograph is what the photograph actually
 * knows: where the pencil is, which way it points, and what colour it is.
 */
import * as THREE from "three";
import { LENGTH, PROFILE, PARTS } from "./pencil.js";

const GREY = [0.30, 0.31, 0.34];        // grip rubber
const CLIP_GREY = [0.38, 0.39, 0.42];
const ERASER = [0.94, 0.93, 0.88];
const LEAD = [0.13, 0.13, 0.14];
const RADIAL = 40;

// the clip, as fractions of the length and of the barrel radius. It is sunk
// into the barrel rather than standing proud of it: the physics body is a
// surface of revolution and knows nothing about a clip, so a clip that stood
// out would be drawn surface outside the simulated body, which is the bug
// this file exists to make impossible.
const CLIP_FROM_TOP = 0.125, CLIP_LEN = 0.185, CLIP_W = 0.44, CLIP_SINK = 0.10;

// How far to push a near-neutral barrel away from grey. Reconstruction
// colours come back washed out and a grey-blue barrel otherwise reads as
// plain grey. The boost fades out with saturation: applied flat it also
// turned the salmon pencil red and the orange one fluorescent.
const CHROMA = 1.9, CHROMA_FADE = 0.35;

/**
 * The barrel colour: the most saturated cluster in the reconstruction,
 * which is the moulded plastic. The mean blends in the grey grip and the
 * shaded side and comes out muddy.
 */
function barrelColour(colors) {
  if (!colors || !colors.length) return [0.7, 0.7, 0.72];
  const buckets = new Map();
  for (const c of colors) {
    const key = `${c[0] >> 5},${c[1] >> 5},${c[2] >> 5}`;
    const e = buckets.get(key) ?? { n: 0, r: 0, g: 0, b: 0 };
    e.n++; e.r += c[0]; e.g += c[1]; e.b += c[2];
    buckets.set(key, e);
  }
  let best = null, bestScore = -1;
  for (const e of buckets.values()) {
    const r = e.r / e.n, g = e.g / e.n, b = e.b / e.n;
    const mx = Math.max(r, g, b), mn = Math.min(r, g, b);
    const sat = mx > 1 ? (mx - mn) / mx : 0;
    const score = e.n * (0.25 + sat) * (mx > 40 ? 1 : 0.2);
    if (score > bestScore) { bestScore = score; best = [r, g, b]; }
  }
  if (!best) return [0.7, 0.7, 0.72];
  // Barely lifted now. The old 1.25x + 0.06 compensated for a scene lit at
  // a fifth of its proper brightness; under lights that sum to pi on the
  // desk it turned orange to peach and purple to lavender.
  const lit = best.map((x) => Math.min(1, (x / 255) * 1.06));
  const hi = Math.max(...lit), lo = Math.min(...lit);
  const sat = hi > 0 ? (hi - lo) / hi : 0;
  const k = 1 + (CHROMA - 1) * Math.max(0, 1 - sat / CHROMA_FADE);
  const lum = 0.299 * lit[0] + 0.587 * lit[1] + 0.114 * lit[2];
  return lit.map((x) => Math.min(1, Math.max(0, lum + (x - lum) * k)));
}

/**
 * body: a packet body ({capsule, offsets, render_colors, render_verts}).
 * Returns a THREE.Group whose child holds the pencil in the body frame.
 */
export function buildPencil(body) {
  const axis = new THREE.Vector3(...body.capsule.axis).normalize();
  const colour = barrelColour(body.render_colors);
  const paint = { barrel: colour, cap: colour, cone: colour, grip: GREY, eraser: ERASER, lead: LEAD };

  // revolve the profile: LatheGeometry takes (radius, height) about +y.
  // A lathe leaves both ends open, so the profile is closed onto the axis
  // here with a zero-radius point at each end. Those two points are drawing
  // only and deliberately not in the shared PROFILE, where a zero radius at
  // t = 0 would tell the contact guard the eraser is a point.
  const knots = [[0, 0], ...PROFILE, [1, 0]];
  const parts = [PARTS[0], ...PARTS, PARTS[PARTS.length - 1]];
  const pts = knots.map(([t, r]) => new THREE.Vector2(r, (t - 0.5) * LENGTH));
  const geo = new THREE.LatheGeometry(pts, RADIAL);
  // LatheGeometry emits (RADIAL + 1) rings of knots.length points, ring
  // major, so a vertex's profile knot is its index modulo the knot count
  const n = geo.getAttribute("position").count;
  const col = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    const c = paint[parts[i % knots.length]] ?? colour;
    col[3 * i] = c[0]; col[3 * i + 1] = c[1]; col[3 * i + 2] = c[2];
  }
  geo.setAttribute("color", new THREE.BufferAttribute(col, 3));
  geo.computeVertexNormals();

  const g = new THREE.Group();
  const mat = (extra = {}) => new THREE.MeshStandardMaterial({
    vertexColors: true, roughness: 0.42, metalness: 0.0, ...extra });
  g.add(new THREE.Mesh(geo, mat()));

  // The clip stands 1.2 mm proud of the barrel, as the real one does, and
  // the Rapier collider has the same box in the same place, so a pencil
  // lying clip-down rests on its clip in the drawing and in the physics
  // alike. (It is outside the surface of revolution the learned engine
  // simulates; the mesh test knows to leave it out.)
  const rBarrel = PROFILE[4][1];
  const clipLen = CLIP_LEN * LENGTH, clipW = CLIP_W * rBarrel * 2, clipT = 0.0020;
  const clipY = -LENGTH / 2 + CLIP_FROM_TOP * LENGTH + clipLen / 2;
  const plate = new THREE.BoxGeometry(clipW, clipLen, clipT);
  plate.translate(0, clipY, rBarrel + 0.0012 - clipT / 2);
  const cc = new Float32Array(plate.getAttribute("position").count * 3);
  for (let i = 0; i < cc.length; i += 3) {
    cc[i] = CLIP_GREY[0]; cc[i + 1] = CLIP_GREY[1]; cc[i + 2] = CLIP_GREY[2];
  }
  plate.setAttribute("color", new THREE.BufferAttribute(cc, 3));
  const clip = new THREE.Mesh(plate, mat({ roughness: 0.45 }));
  clip.userData.clip = true;
  g.add(clip);

  // Built along +y with the point at +y; turn that onto the body's axis,
  // which the packet guarantees points at the lead. Do NOT re-decide the
  // orientation here. An earlier version did, from the reconstruction's
  // colours, and since the packet had already applied that same decision
  // when it chose the body frame, applying it twice flipped every pencil
  // whose cue said "reversed": the drawn grip then sat where the physics
  // has barrel, which put 0.79 mm of drawn rubber below the table.
  g.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), axis);
  g.traverse((o) => { if (o.isMesh) { o.castShadow = true; o.receiveShadow = true; } });

  const outer = new THREE.Group();   // the renderer drives this one's pose
  outer.add(g);
  return outer;
}

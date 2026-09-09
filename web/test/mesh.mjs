/**
 * The drawn pencil must agree with the simulated one about where its
 * surface is.
 *
 * This is not a cosmetic test. Two reported bugs came from the mesh and the
 * physics disagreeing, and neither is visible in any physics test, because
 * the physics was right:
 *
 *   the barrel was drawn at the capsule's MEDIAN particle radius while the
 *   physics body reached the 95th percentile, so a pencil the ground rule
 *   held exactly on the table was drawn 1.7 mm above it and looked like it
 *   was hovering;
 *
 *   the clip stood 0.9 mm proud of the drawn barrel, outside the body the
 *   ground rule holds, so whichever face pointed down passed through the
 *   table and looked like it was sinking.
 *
 * The comparison is between SUPPORT FUNCTIONS, because that is exactly what
 * the ground rule computes: it projects every particle on -z and lifts the
 * body until the lowest sits at the table. So for a direction n the physics
 * surface is max over particles of n.p and the drawn surface is max over
 * mesh vertices of n.v, and the difference is, in millimetres, how far the
 * drawing hangs below the table (positive) or floats above it (negative).
 *
 * Directions are split, because the two families mean different things:
 *
 *   ACROSS the pencil is how a pencil lies on a table. The drawing must
 *   never escape here, and it does not: 0.000 mm on all four scenes, which
 *   the mesh earns by fitting the particle cloud's elliptical section and
 *   then shrinking onto its true silhouette.
 *
 *   TILTED off that is a pencil resting on a pile. Several millimetres of
 *   escape remain here and the cause is not the drawing: the
 *   reconstructions taper over their last 10 to 15 mm, where a real Matic
 *   Grip is straight to within a few millimetres of the cap. A faithful
 *   pencil cannot fit inside a tapered cloud near its ends. The fix belongs
 *   in src/physsplat/recon/pipeline.py, which already knows the pencil's
 *   real length and radius and could regularise the particle cloud to the
 *   cylinder it knows the object is. The bound below records where that
 *   stands so it cannot quietly get worse first.
 *
 * Run: node test/mesh.mjs
 */
import { readFileSync } from "fs";
import { register } from "module";
// the app resolves "three" through the page's import map; give node the
// same specifier before loading any app code (test/three-resolve.mjs)
register("./three-resolve.mjs", import.meta.url);
const THREE = await import("three");
const { buildPencil } = await import("../public/js/pencil_mesh.js");

const SCENES = ["IMG_8504", "IMG_8513", "IMG_8596", "IMG_8626"];

// Across the pencil, which is how it rests on a table. Nothing drawn may
// escape; the slack is for floating-point, not for geometry.
const FLAT_OUT_MM = 0.05;
// The same directions, the other way: the drawing may not sit so far inside
// the simulated body that the pencil appears to float. What is left here is
// reconstruction lumps, places where the cloud bulges past the pencil.
const FLAT_GAP_MM = 2.5;
// Tilted up to 45 degrees off that, where the reconstructions' tapered ends
// dominate. See the note above.
const TILT_OUT_MM = 7.0;
const TILTS = [10, 20, 30, 45];
const AZIMUTHS = 360;

let failed = 0;
const check = (ok, msg) => { console.log(`${ok ? "PASS" : "FAIL"} ${msg}`); if (!ok) failed++; };

/** World-space vertices of every mesh in a built pencil, in the body frame. */
function meshVertices(group) {
  const out = [];
  group.updateMatrixWorld(true);
  const v = new THREE.Vector3();
  group.traverse((o) => {
    if (!o.isMesh) return;
    const p = o.geometry.getAttribute("position");
    for (let i = 0; i < p.count; i++) {
      v.fromBufferAttribute(p, i).applyMatrix4(o.matrixWorld);
      out.push([v.x, v.y, v.z]);
    }
  });
  return out;
}

/** max over points of n . p */
function support(points, n) {
  let m = -Infinity;
  for (const p of points) {
    const d = n[0] * p[0] + n[1] * p[1] + n[2] * p[2];
    if (d > m) m = d;
  }
  return m;
}

for (const scene of SCENES) {
  const packet = JSON.parse(
    readFileSync(new URL(`../public/packets/${scene}.json`, import.meta.url), "utf8"));
  let flatOut = 0, flatOutBody = -1, flatGap = 0, flatGapBody = -1;
  let tiltOut = 0, tiltOutBody = -1, tiltAt = 0;

  packet.bodies.forEach((b, bi) => {
    if (!b.capsule || !b.offsets || !b.offsets.length) return;
    const verts = meshVertices(buildPencil(b));
    const axis = new THREE.Vector3(...b.capsule.axis).normalize();
    // a basis across the pencil
    const u = new THREE.Vector3(1, 0, 0);
    if (Math.abs(u.dot(axis)) > 0.9) u.set(0, 1, 0);
    const e1 = u.sub(axis.clone().multiplyScalar(u.dot(axis))).normalize();
    const e2 = new THREE.Vector3().crossVectors(axis, e1);

    for (let k = 0; k < AZIMUTHS; k++) {
      const t = (k / AZIMUTHS) * 2 * Math.PI;
      const perp = e1.clone().multiplyScalar(Math.cos(t))
        .add(e2.clone().multiplyScalar(Math.sin(t)));

      const flat = [perp.x, perp.y, perp.z];
      const d = support(verts, flat) - support(b.offsets, flat);
      if (d > flatOut) { flatOut = d; flatOutBody = bi; }
      if (-d > flatGap) { flatGap = -d; flatGapBody = bi; }

      for (const deg of TILTS) {
        const c = Math.cos((deg * Math.PI) / 180), s = Math.sin((deg * Math.PI) / 180);
        for (const sg of [1, -1]) {
          const n = perp.clone().multiplyScalar(c)
            .add(axis.clone().multiplyScalar(sg * s)).normalize();
          const e = support(verts, [n.x, n.y, n.z]) - support(b.offsets, [n.x, n.y, n.z]);
          if (e > tiltOut) { tiltOut = e; tiltOutBody = bi; tiltAt = deg; }
        }
      }
    }
  });

  check(flatOut * 1000 <= FLAT_OUT_MM,
    `${scene}: lying flat, nothing drawn goes below the lowest particle ` +
    `(${(flatOut * 1000).toFixed(3)} mm, body ${flatOutBody})`);
  check(flatGap * 1000 <= FLAT_GAP_MM,
    `${scene}: lying flat, the drawing does not float inside the body ` +
    `(${(flatGap * 1000).toFixed(2)} mm, body ${flatGapBody})`);
  check(tiltOut * 1000 <= TILT_OUT_MM,
    `${scene}: tilted up to 45 deg, escape stays bounded ` +
    `(${(tiltOut * 1000).toFixed(2)} mm at ${tiltAt} deg, body ${tiltOutBody})`);
}

// And the point goes on the end that is actually the point. Getting this
// backwards draws the blunt eraser over the reconstruction's tapered tip,
// which is the worst case in the tilted family and looks wrong besides.
//
// Only bodies where the reconstruction actually distinguishes its two ends
// are asserted on. It often does not: across these seventeen the mean
// particle radius over the outer sixth differs by as little as 0.02 mm, so
// for those the question has no answer to get right and a test that claimed
// otherwise would be measuring rounding. The ones with a clear thin end are
// asserted; the rest are counted and reported.
{
  let wrong = 0, total = 0, ambiguous = 0;
  const CLEAR_MM = 0.3;
  for (const scene of SCENES) {
    const packet = JSON.parse(
      readFileSync(new URL(`../public/packets/${scene}.json`, import.meta.url), "utf8"));
    for (const b of packet.bodies) {
      if (!b.capsule || !b.offsets || !b.offsets.length) continue;
      total++;
      const axis = new THREE.Vector3(...b.capsule.axis).normalize();
      const verts = meshVertices(buildPencil(b));
      // where the drawing puts its thin end
      let mLo = Infinity, mHi = -Infinity;
      for (const v of verts) {
        const a = v[0] * axis.x + v[1] * axis.y + v[2] * axis.z;
        if (a < mLo) mLo = a; if (a > mHi) mHi = a;
      }
      const radialAt = (pts, near, span) => {
        let n = 0, s = 0;
        for (const p of pts) {
          const a = p[0] * axis.x + p[1] * axis.y + p[2] * axis.z;
          if (Math.abs(a - near) < span) {
            n++;
            s += Math.hypot(p[0] - a * axis.x, p[1] - a * axis.y, p[2] - a * axis.z);
          }
        }
        return n ? s / n : Infinity;
      };
      // For the drawing, look at the last few millimetres only: the lead is
      // there on the point end and the eraser on the other, and they are far
      // apart. Over a whole sixth the eraser (0.62 of the barrel radius)
      // reads as thin too and the comparison stops meaning anything.
      const tip = (mHi - mLo) / 40;
      const drawnThinAtLo = radialAt(verts, mLo, tip) < radialAt(verts, mHi, tip);
      // For the cloud, a sixth: it is sparse near the ends.
      const sixth = (mHi - mLo) / 6;
      const cLo = radialAt(b.offsets, mLo, sixth), cHi = radialAt(b.offsets, mHi, sixth);
      if (Math.abs(cLo - cHi) * 1000 < CLEAR_MM) { ambiguous++; continue; }
      if (drawnThinAtLo !== cLo < cHi) wrong++;
    }
  }
  const decided = total - ambiguous;
  check(wrong === 0,
    `the drawn point is on the reconstruction's thin end ` +
    `(${decided - wrong}/${decided} bodies; ${ambiguous} of ${total} have no ` +
    `distinguishable thin end)`);
}

console.log(failed ? `\nMESH TESTS FAILED (${failed})` : "\nMESH TESTS PASSED");
process.exit(failed ? 1 : 0);

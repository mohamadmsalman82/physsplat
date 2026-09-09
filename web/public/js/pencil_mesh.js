/**
 * A BIC Matic Grip, built to the real pencil rather than to a capsule.
 *
 * Working from a photograph of the pencil itself, the parts and their
 * proportions over the 150 mm length are:
 *
 *   white eraser standing ~3 mm out of the top
 *   a cap of barrel colour, slightly narrower than the barrel
 *   the barrel: one solid colour, constant diameter
 *   a grey clip, a wide flat plate lying along the upper barrel
 *   a grey rubber grip near the lower third, a little fatter than the barrel
 *   a short run of barrel below the grip, then a cone OF BARREL COLOUR
 *   a dark lead sleeve and the lead itself at the very point
 *
 * The colour comes from the reconstruction of that pencil; everything else
 * is the same on every Matic Grip. Two earlier versions got this wrong in
 * ways worth recording: a metal ferrule (this pencil has none, the cone is
 * plastic and body-coloured) and a barrel that carried the photo's colour
 * bands, which made it look mottled rather than moulded.
 *
 * Every drawn surface stays inside the particle hull the physics moves, so
 * a pencil the ground rule holds on the table is drawn touching it.
 */
import * as THREE from "three";

// fractions of the full length, measured off the reference photograph
const ERASER_OUT = 0.020;      // eraser protruding past the cap
const CAP_LEN = 0.045;
const CLIP_FROM_TOP = 0.125, CLIP_LEN = 0.185, CLIP_W = 0.44, CLIP_T = 0.22;
const GRIP_FROM_TIP = 0.127, GRIP_LEN = 0.133;
const CONE_LEN = 0.055;
const LEAD_LEN = 0.016;
// the real pencil's grip is about 11 mm across and its barrel about 9:
// the physics hull's fattest band is the grip, so the barrel follows from
// it. Taking the barrel from the hull's median instead drew a grip far too
// fat, because the reconstruction is lumpy.
const GRIP_OVER_BARREL = 1.22;

// The grip rubber and clip are a dark charcoal on the real pencil. An
// earlier mid-grey vanished against the grey-blue barrels, which is two of
// the six pencils in this set.
const GREY = [0.30, 0.31, 0.34];        // grip rubber
const CLIP_GREY = [0.38, 0.39, 0.42];
const ERASER = [0.94, 0.93, 0.88];
const LEAD = [0.13, 0.13, 0.14];
// How far to push a near-neutral barrel away from grey. Reconstruction
// colours come back washed out and a grey-blue barrel otherwise reads as
// plain grey. The boost has to fade out with saturation: applied flat it
// also turned the salmon pencil red and the orange one fluorescent, which
// is further from the real object than the problem it was fixing.
const CHROMA = 1.9, CHROMA_FADE = 0.35;

const solid = (geo, rgb) => {
  const n = geo.getAttribute("position").count;
  const c = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) { c[3 * i] = rgb[0]; c[3 * i + 1] = rgb[1]; c[3 * i + 2] = rgb[2]; }
  geo.setAttribute("color", new THREE.BufferAttribute(c, 3));
  return geo;
};

/**
 * The barrel colour: the most saturated colour in the reconstruction,
 * which is the moulded plastic. Averaging everything would blend in the
 * grey grip and the shaded side and come out muddy.
 */
function barrelColour(colors) {
  let best = null, bestScore = -1;
  const buckets = new Map();
  for (const c of colors) {
    const key = `${c[0] >> 5},${c[1] >> 5},${c[2] >> 5}`;
    const e = buckets.get(key) ?? { n: 0, r: 0, g: 0, b: 0 };
    e.n++; e.r += c[0]; e.g += c[1]; e.b += c[2];
    buckets.set(key, e);
  }
  for (const e of buckets.values()) {
    const r = e.r / e.n, g = e.g / e.n, b = e.b / e.n;
    const mx = Math.max(r, g, b), mn = Math.min(r, g, b);
    const sat = mx > 1 ? (mx - mn) / mx : 0;
    // popularity weighted by how coloured it is, so the barrel wins over
    // the grey grip and over dark shadow
    const score = e.n * (0.25 + sat) * (mx > 40 ? 1 : 0.2);
    if (score > bestScore) { bestScore = score; best = [r, g, b]; }
  }
  if (!best) return [0.7, 0.7, 0.72];
  // lift it a little: reconstruction colours come back darker than the object
  const lit = best.map((x) => Math.min(1, (x / 255) * 1.25 + 0.06));
  // and push it away from grey about its own luminance, so a barrel that is
  // genuinely close to neutral still shows which way it leans. A barrel that
  // already has colour is left alone.
  const hi = Math.max(...lit), lo = Math.min(...lit);
  const sat = hi > 0 ? (hi - lo) / hi : 0;
  const k = 1 + (CHROMA - 1) * Math.max(0, 1 - sat / CHROMA_FADE);
  const lum = 0.299 * lit[0] + 0.587 * lit[1] + 0.114 * lit[2];
  return lit.map((x) => Math.min(1, Math.max(0, lum + (x - lum) * k)));
}

/**
 * body: a packet body ({capsule, offsets, render_colors}). Returns a
 * THREE.Group whose child holds the parts in the body frame.
 */
export function buildPencil(body) {
  const cap = body.capsule;
  const axis = new THREE.Vector3(...cap.axis).normalize();
  const cx = cap.axis;

  // Size from the physics particles, not from the capsule: `capsule.radius`
  // is their MEDIAN distance from the axis, so a barrel drawn at it floats
  // above the particle the ground rule is holding at the table.
  // a basis across the pencil, for measuring its cross-section
  const e1 = new THREE.Vector3(1, 0, 0);
  if (Math.abs(e1.dot(axis)) > 0.9) e1.set(0, 1, 0);
  e1.sub(axis.clone().multiplyScalar(e1.dot(axis))).normalize();
  const e2 = new THREE.Vector3().crossVectors(axis, e1);

  let r = cap.radius, half = cap.half, gripR = r, centre = 0, tipAtLo = false;
  let squash = 1, squashDir = 0, offP = 0, offQ = 0;
  if (body.offsets && body.offsets.length) {
    const BINS = 24;
    const along = [], radial = [], pp = [], qq = [];
    let lo = Infinity, hi = -Infinity;
    for (const o of body.offsets) {
      const a = o[0] * cx[0] + o[1] * cx[1] + o[2] * cx[2];
      along.push(a);
      radial.push(Math.hypot(o[0] - a * cx[0], o[1] - a * cx[1], o[2] - a * cx[2]));
      pp.push(o[0] * e1.x + o[1] * e1.y + o[2] * e1.z);
      qq.push(o[0] * e2.x + o[1] * e2.y + o[2] * e2.z);
      if (a < lo) lo = a;
      if (a > hi) hi = a;
    }
    // The particles are offsets from the centre of MASS, and a pencil whose
    // far end was occluded in the photo reconstructs asymmetrically about
    // it: up to 9.3 mm off centre across these four scenes. Sizing the
    // drawing by the longer half and centring it on the origin therefore
    // overhung the short end by as much as 19 mm, which is drawn pencil
    // sticking out of the simulated body and lands on the table when
    // nothing is holding it. Span what the particles actually span.
    centre = (lo + hi) / 2;
    const span = Math.max(hi - lo, 0.02);
    const bin = new Float64Array(BINS);
    along.forEach((a, i) => {
      const k = Math.min(BINS - 1, Math.max(0, Math.floor((a - lo) / span * BINS)));
      bin[k] = Math.max(bin[k], radial[i]);
    });
    gripR = Math.max(...bin);
    r = gripR / GRIP_OVER_BARREL;
    half = Math.max(span / 2 - r, 0.01);

    // Which end is the point. A pencil's point end is the thin one, so
    // compare the mean particle radius over the outer sixth at each end.
    // The obvious alternative, putting the point on whichever side the
    // fattest band (the grip) leans towards, reads a single lumpy particle
    // and gets it backwards on a lumpy reconstruction: IMG_8504 body 2 came
    // out with the blunt eraser drawn over the real pencil's tapered point,
    // which is 7 mm of drawn pencil hanging outside the simulated body at
    // the exact end a tilted pencil rests on.
    let nLo = 0, sLo = 0, nHi = 0, sHi = 0;
    const edge = span / 6;
    along.forEach((a, i) => {
      if (a < lo + edge) { nLo++; sLo += radial[i]; }
      else if (a > hi - edge) { nHi++; sHi += radial[i]; }
    });
    tipAtLo = nLo && nHi ? (sLo / nLo) < (sHi / nHi) : false;

    // A round pencil does not fit inside these reconstructions. Their
    // cross-sections come back elliptical, roughly 1.4 to 1, so a circle
    // drawn at the fattest particle reaches 1 to 2 mm past the cloud in the
    // narrow direction, and since the ground rule holds the lowest PARTICLE,
    // that is drawn pencil below the table. Measure the ellipse and squash
    // the drawing onto it: at 1.4 to 1 across 9 mm the flattening is not
    // visible, and it is the difference between a pencil that rests on the
    // table and one that sinks into it.
    let sp = 0, sq = 0, spq = 0;
    for (let i = 0; i < pp.length; i++) { sp += pp[i] * pp[i]; sq += qq[i] * qq[i]; spq += pp[i] * qq[i]; }
    squashDir = 0.5 * Math.atan2(2 * spq / pp.length, (sp - sq) / pp.length);
    const c1 = Math.cos(squashDir), s1 = Math.sin(squashDir);
    let loS = Infinity, hiS = -Infinity, loT = Infinity, hiT = -Infinity;
    for (let i = 0; i < pp.length; i++) {
      const s = pp[i] * c1 + qq[i] * s1, t = -pp[i] * s1 + qq[i] * c1;
      if (s < loS) loS = s; if (s > hiS) hiS = s;
      if (t < loT) loT = t; if (t > hiT) hiT = t;
    }
    const halfS = (hiS - loS) / 2, halfT = (hiT - loT) / 2;
    const major = Math.max(halfS, halfT);
    if (major > 1e-5) {
      // draw at the major half-width and flatten the minor one onto the hull
      gripR = major;
      r = gripR / GRIP_OVER_BARREL;
      const minor = Math.min(halfS, halfT);
      squash = minor / major;
      // squashDir must point along the NARROW axis, the one being compressed
      if (halfT < halfS) squashDir += Math.PI / 2;
      // the cross-section is not centred on the axis either
      const cs = (loS + hiS) / 2, ct = (loT + hiT) / 2;
      offP = cs * c1 - ct * s1;
      offQ = cs * s1 + ct * c1;

      // The silhouette is not actually an ellipse, so the ellipse through
      // its extremes still pokes out of it in between, by about a
      // millimetre. Shrink the drawn section about its own centre until it
      // is inside the silhouette in every direction across the pencil.
      // This is the last millimetre of the sinking report, and it is worth
      // the 360 dot products once per body at load.
      const cd = Math.cos(squashDir), sd = Math.sin(squashDir);
      let f = 1;
      for (let k = 0; k < 360; k++) {
        const th = k * Math.PI / 180, np = Math.cos(th), nq = Math.sin(th);
        let cloud = -Infinity;
        for (let i = 0; i < pp.length; i++) {
          const v = np * pp[i] + nq * qq[i];
          if (v > cloud) cloud = v;
        }
        // support of the drawn ellipse about its centre, per unit scale
        const rad = Math.hypot(minor * (np * cd + nq * sd), major * (-np * sd + nq * cd));
        if (rad > 1e-9) f = Math.min(f, (cloud - (np * offP + nq * offQ)) / rad);
      }
      gripR = major * Math.min(1, Math.max(0.3, f));
      r = gripR / GRIP_OVER_BARREL;
    }
  }
  const L = 2 * half + 2 * r;
  const colour = barrelColour(body.render_colors);

  // parts are built with the point at +y, so turn +y onto whichever end of
  // the axis the particles say is the thin one
  const tipSign = tipAtLo ? -1 : 1;

  const g = new THREE.Group();
  const frame = new THREE.Quaternion().setFromUnitVectors(
    new THREE.Vector3(0, 1, 0), axis.clone().multiplyScalar(tipSign));
  const mat = (extra = {}) => new THREE.MeshStandardMaterial({
    vertexColors: true, roughness: 0.42, metalness: 0.0, ...extra });

  // One transform takes a part from the frame it is built in (along +y,
  // round, centred on the origin) to where the physics body actually is:
  //   translate onto the particle cloud's centre, which is off the body
  //   origin both along the pencil and across it;
  //   squash across the narrow axis of the cloud's elliptical section;
  //   turn +y onto the pencil's axis.
  // Baked into the geometry rather than hung on the scene graph, so the
  // squash composes with the axis rotation in the right order and the
  // renderer still drives one clean group transform.
  const nd = e1.clone().multiplyScalar(Math.cos(squashDir))
    .add(e2.clone().multiplyScalar(Math.sin(squashDir)));
  const Rd = new THREE.Matrix4().makeBasis(nd, axis, new THREE.Vector3().crossVectors(nd, axis));
  const M = new THREE.Matrix4()
    .makeTranslation(
      centre * axis.x + offP * e1.x + offQ * e2.x,
      centre * axis.y + offP * e1.y + offQ * e2.y,
      centre * axis.z + offP * e1.z + offQ * e2.z)
    .multiply(Rd)
    .multiply(new THREE.Matrix4().makeScale(squash, 1, 1))
    .multiply(Rd.clone().transpose())
    .multiply(new THREE.Matrix4().makeRotationFromQuaternion(frame));
  const add = (geo, m) => g.add(new THREE.Mesh(geo.applyMatrix4(M), m));

  // y runs from the eraser end (-L/2) to the point (+L/2)
  const top = -L / 2, tip = L / 2;
  const capLen = CAP_LEN * L, coneLen = CONE_LEN * L, leadLen = LEAD_LEN * L;
  const gripLen = GRIP_LEN * L;
  const gripEnd = tip - GRIP_FROM_TIP * L;          // grip's lower edge
  const gripStart = gripEnd - gripLen;
  const coneBase = tip - coneLen - leadLen;

  // barrel: cap edge to the base of the cone, one solid colour
  const barrelTop = top + capLen;
  const barrel = new THREE.CylinderGeometry(r, r, coneBase - barrelTop, 32, 1, false);
  barrel.translate(0, (barrelTop + coneBase) / 2, 0);
  add(solid(barrel, colour), mat());

  // cap: a slightly narrower collar the eraser sits in
  const capGeo = new THREE.CylinderGeometry(r * 0.86, r * 0.94, capLen, 28, 1, false);
  capGeo.translate(0, top + capLen / 2, 0);
  add(solid(capGeo, colour), mat());

  // eraser: white, standing out of the cap
  const eraLen = ERASER_OUT * L + capLen * 0.5;
  const era = new THREE.CylinderGeometry(r * 0.62, r * 0.62, eraLen, 20, 1, false);
  era.translate(0, top - ERASER_OUT * L + eraLen / 2, 0);
  add(solid(era, ERASER), mat({ roughness: 0.85 }));

  // grip: grey rubber, the fattest part of the pencil and what it rests on.
  // Moulded, so it swells out of the barrel over a couple of millimetres at
  // each end rather than sitting on it as a sleeve.
  const flare = Math.min(gripLen * 0.18, 0.003);
  const gripMid = new THREE.CylinderGeometry(gripR, gripR, gripLen - 2 * flare, 32, 1, false);
  gripMid.translate(0, (gripStart + gripEnd) / 2, 0);
  add(solid(gripMid, GREY), mat({ roughness: 0.95 }));
  const flareTop = new THREE.CylinderGeometry(gripR, r * 1.02, flare, 32, 1, false);
  flareTop.translate(0, gripStart + flare / 2, 0);
  add(solid(flareTop, GREY), mat({ roughness: 0.95 }));
  const flareBot = new THREE.CylinderGeometry(r * 1.02, gripR, flare, 32, 1, false);
  flareBot.translate(0, gripEnd - flare / 2, 0);
  add(solid(flareBot, GREY), mat({ roughness: 0.95 }));

  // cone and lead: the cone is plastic in the body colour on this pencil,
  // not a metal ferrule
  const cone = new THREE.CylinderGeometry(r * 0.30, r * 0.97, coneLen, 28, 1, false);
  cone.translate(0, coneBase + coneLen / 2, 0);
  add(solid(cone, colour), mat({ roughness: 0.35 }));
  const lead = new THREE.CylinderGeometry(r * 0.07, r * 0.24, leadLen, 12, 1, false);
  lead.translate(0, tip - leadLen / 2, 0);
  add(solid(lead, LEAD), mat({ roughness: 0.6 }));

  // clip: a wide grey plate lying along the upper barrel, its outer face
  // flush with the barrel so it is never what touches the table
  const clipLen = CLIP_LEN * L, clipW = CLIP_W * r * 2, clipT = CLIP_T * r;
  const clipY = top + CLIP_FROM_TOP * L + clipLen / 2;
  // it stands proud, but never past the grip's radius, so the clip is
  // never the part of the pencil that meets the table
  const clipOut = Math.min(r + clipT * 0.5, gripR - 1e-4);
  const plate = new THREE.BoxGeometry(clipW, clipLen, clipT);
  plate.translate(0, clipY, clipOut - clipT / 2);
  add(solid(plate, CLIP_GREY), mat({ roughness: 0.45 }));
  // the rounded lip at its free end, and the bridge back to the cap
  const lip = new THREE.CylinderGeometry(clipT * 0.6, clipT * 0.6, clipW, 10);
  lip.rotateZ(Math.PI / 2);
  lip.translate(0, clipY + clipLen / 2, clipOut - clipT / 2);
  add(solid(lip, CLIP_GREY), mat({ roughness: 0.45 }));
  const bridge = new THREE.BoxGeometry(clipW * 0.6, clipLen * 0.18, clipT * 2.2);
  bridge.translate(0, clipY - clipLen / 2, r * 0.9);
  add(solid(bridge, CLIP_GREY), mat({ roughness: 0.45 }));

  g.traverse((o) => { if (o.isMesh) { o.castShadow = true; o.receiveShadow = true; } });
  const outer = new THREE.Group();   // the renderer drives this one's pose
  outer.add(g);
  return outer;
}

/**
 * PhysSplat browser physics runtime.
 *
 * Line-by-line port of the Python reference (fixture-tested in
 * web/test/parity.mjs against values emitted by the Python code):
 *   buildEdges / edgeFeatures  <- src/physsplat/model/graph.py
 *   quat*, externalAccels,step <- src/physsplat/model/integrator.py
 *   nodeFeatures               <- src/physsplat/model/normalize.py
 *   actionFeature              <- src/physsplat/common/actions.py
 *
 * All constants come from runtime.json (written by the ONNX exporter) so
 * the browser can never drift from the training-time values.
 */

// ---------------------------------------------------------------- graph

export function buildEdges(parts, vels, contactRadius, dt) {
  // uniform grid hash over union of current and velocity-extrapolated
  // positions; returns both directions of every edge, i < j pairs found
  const N = parts.length / 3;
  const r = contactRadius;
  const inv = 1 / r;                      // cell size = contact radius
  const pairs = new Set();
  // integer cell key packed into one double (exact below 2^53): cells are
  // 6 mm, so 16 bits per axis covers +/-196 m. String keys cost ~10 ms/step.
  const OFF = 1 << 15, SH = 1 << 16;
  const key = (x, y, z) => ((x + OFF) * SH + (y + OFF)) * SH + (z + OFF);

  // the predictive pass only matters when something moves; at rest the
  // extrapolated positions are the current ones and the pass is wasted
  let moving = false;
  if (vels) for (let i = 0; i < N * 3; i++) if (Math.abs(vels[i]) * dt > 1e-7) { moving = true; break; }
  const passes = moving ? 2 : 1;
  const px = new Float64Array(N * 3);
  const cx = new Int32Array(N), cy = new Int32Array(N), cz = new Int32Array(N);
  for (let pass = 0; pass < passes; pass++) {
    const grid = new Map();
    for (let i = 0; i < N * 3; i++)
      px[i] = pass === 0 ? parts[i] : parts[i] + vels[i] * dt;
    for (let i = 0; i < N; i++) {
      cx[i] = Math.floor(px[3 * i] * inv);
      cy[i] = Math.floor(px[3 * i + 1] * inv);
      cz[i] = Math.floor(px[3 * i + 2] * inv);
      const k = key(cx[i], cy[i], cz[i]);
      const b = grid.get(k);
      if (b) b.push(i); else grid.set(k, [i]);
    }
    for (let i = 0; i < N; i++) {
      const xi = px[3 * i], yi = px[3 * i + 1], zi = px[3 * i + 2];
      for (let dx = -1; dx <= 1; dx++)
        for (let dy = -1; dy <= 1; dy++)
          for (let dz = -1; dz <= 1; dz++) {
            const bucket = grid.get(key(cx[i] + dx, cy[i] + dy, cz[i] + dz));
            if (!bucket) continue;
            for (let n = 0; n < bucket.length; n++) {
              const j = bucket[n];
              if (j <= i) continue;
              const ddx = xi - px[3 * j], ddy = yi - px[3 * j + 1],
                ddz = zi - px[3 * j + 2];
              if (ddx * ddx + ddy * ddy + ddz * ddz < r * r)
                pairs.add(i * N + j);
            }
          }
    }
  }
  const E2 = pairs.size;
  const senders = new Int32Array(2 * E2);
  const receivers = new Int32Array(2 * E2);
  const sorted = [...pairs].sort((a, b) => a - b);
  sorted.forEach((p, e) => {
    const i = Math.floor(p / N), j = p % N;
    senders[e] = i; receivers[e] = j;
    senders[E2 + e] = j; receivers[E2 + e] = i;
  });
  return { senders, receivers };
}

export function edgeFeatures(parts, senders, receivers, bodyIds, contactRadius) {
  const E = senders.length;
  const f = new Float32Array(E * 5);
  for (let e = 0; e < E; e++) {
    const s = senders[e], r = receivers[e];
    const dx = (parts[3 * r] - parts[3 * s]) / contactRadius;
    const dy = (parts[3 * r + 1] - parts[3 * s + 1]) / contactRadius;
    const dz = (parts[3 * r + 2] - parts[3 * s + 2]) / contactRadius;
    f[5 * e] = dx; f[5 * e + 1] = dy; f[5 * e + 2] = dz;
    f[5 * e + 3] = Math.sqrt(dx * dx + dy * dy + dz * dz);
    f[5 * e + 4] = bodyIds[s] === bodyIds[r] ? 1 : 0;
  }
  return f;
}

// ------------------------------------------------------------ quaternions
// xyzw convention throughout, matching scipy/PyBullet/three.js

export function quatMul(a, b) {
  const [ax, ay, az, aw] = a, [bx, by, bz, bw] = b;
  return [
    aw * bx + ax * bw + ay * bz - az * by,
    aw * by - ax * bz + ay * bw + az * bx,
    aw * bz + ax * by - ay * bx + az * bw,
    aw * bw - ax * bx - ay * by - az * bz,
  ];
}

export function quatFromRotvec(v) {
  const angle = Math.hypot(v[0], v[1], v[2]);
  const half = 0.5 * angle;
  const k = angle < 1e-8 ? 0.5 - (angle * angle) / 48
    : Math.sin(half) / Math.max(angle, 1e-12);
  return [v[0] * k, v[1] * k, v[2] * k, Math.cos(half)];
}

export function quatToMatrix(q) {
  const [x, y, z, w] = q;
  return [
    1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
    2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
    2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
  ]; // row-major 3x3
}

export function matVec(m, v) {
  return [
    m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
    m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
    m[6] * v[0] + m[7] * v[1] + m[8] * v[2],
  ];
}

// ------------------------------------------------------------- integrator

export function externalAccels(pos, quat, mass, inertia, actBody, actPoint, actForce, B) {
  const lin = new Float64Array(B * 3), ang = new Float64Array(B * 3);
  if (actBody >= 0) {
    const m = mass[actBody];
    for (let k = 0; k < 3; k++) lin[3 * actBody + k] = actForce[k] / m;
    const R = quatToMatrix(quat[actBody]);
    // I_world = R diag(I) R^T
    const I = inertia[actBody];
    const IW = new Array(9);
    for (let i = 0; i < 3; i++)
      for (let j = 0; j < 3; j++) {
        let s = 0;
        for (let k = 0; k < 3; k++) s += R[3 * i + k] * I[k] * R[3 * j + k];
        IW[3 * i + j] = s;
      }
    const r = [actPoint[0] - pos[actBody][0], actPoint[1] - pos[actBody][1],
      actPoint[2] - pos[actBody][2]];
    const tq = [r[1] * actForce[2] - r[2] * actForce[1],
      r[2] * actForce[0] - r[0] * actForce[2],
      r[0] * actForce[1] - r[1] * actForce[0]];
    const w = solve3(IW, tq);
    for (let k = 0; k < 3; k++) ang[3 * actBody + k] = w[k];
  }
  return { lin, ang };
}

export function solve3(A, b) {
  // Cramer's rule for 3x3 (row-major A)
  const d = A[0] * (A[4] * A[8] - A[5] * A[7]) - A[1] * (A[3] * A[8] - A[5] * A[6])
    + A[2] * (A[3] * A[7] - A[4] * A[6]);
  const inv = 1 / d;
  const x = (b[0] * (A[4] * A[8] - A[5] * A[7]) - A[1] * (b[1] * A[8] - A[5] * b[2])
    + A[2] * (b[1] * A[7] - A[4] * b[2])) * inv;
  const y = (A[0] * (b[1] * A[8] - A[5] * b[2]) - b[0] * (A[3] * A[8] - A[5] * A[6])
    + A[2] * (A[3] * b[2] - b[1] * A[6])) * inv;
  const z = (A[0] * (A[4] * b[2] - b[1] * A[7]) - A[1] * (A[3] * b[2] - b[1] * A[6])
    + b[0] * (A[3] * A[7] - A[4] * A[6])) * inv;
  return [x, y, z];
}

export function stepBodies(state, residual, ext, dt, gravity) {
  // semi-implicit Euler on SE(3); state mutated in place
  const B = state.pos.length;
  for (let b = 0; b < B; b++) {
    state.linvel[b][0] += (ext.lin[3 * b] + residual[6 * b]) * dt;
    state.linvel[b][1] += (ext.lin[3 * b + 1] + residual[6 * b + 1]) * dt;
    state.linvel[b][2] += (-gravity + ext.lin[3 * b + 2] + residual[6 * b + 2]) * dt;
    state.angvel[b][0] += (ext.ang[3 * b] + residual[6 * b + 3]) * dt;
    state.angvel[b][1] += (ext.ang[3 * b + 1] + residual[6 * b + 4]) * dt;
    state.angvel[b][2] += (ext.ang[3 * b + 2] + residual[6 * b + 5]) * dt;
    for (let k = 0; k < 3; k++) state.pos[b][k] += state.linvel[b][k] * dt;
    const dq = quatFromRotvec([state.angvel[b][0] * dt,
      state.angvel[b][1] * dt, state.angvel[b][2] * dt]);
    let q = quatMul(dq, state.quat[b]);
    const n = Math.hypot(q[0], q[1], q[2], q[3]);
    state.quat[b] = q.map((x) => x / n);
  }
}

// ---------------------------------------------------------------- capsules
// Every demo body carries a capsule proxy {axis (body frame), half, radius}.
// Shared by the guards (sim.js) and the diagnostics (diag.js) so both
// report the same contact geometry.

const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));

/** Capsule in world space: center p, unit axis a, half length h, radius r. */
export function capsuleWorld(pos, quat, cap) {
  const R = quatToMatrix(quat);
  return { p: pos, a: matVec(R, cap.axis), h: cap.half, r: cap.radius };
}

/**
 * Closest points between the axis segments of two capsules (p +/- h*a).
 * Returns the axis distance, the overlap `pen` (positive when the capsules
 * interpenetrate), the unit normal from B toward A, and both points.
 */
export function capsuleClosest(A, B) {
  const r = [A.p[0] - B.p[0], A.p[1] - B.p[1], A.p[2] - B.p[2]];
  const dot = (u, v) => u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
  const aa = dot(A.a, A.a), ee = dot(B.a, B.a), bb = dot(A.a, B.a);
  const cc = dot(A.a, r), ff = dot(B.a, r);
  const den = aa * ee - bb * bb;
  let s = den > 1e-12 ? (bb * ff - cc * ee) / den : 0;
  s = clamp(s, -A.h, A.h);
  let t = clamp((bb * s + ff) / ee, -B.h, B.h);
  s = clamp((bb * t - cc) / aa, -A.h, A.h);
  const ca = [A.p[0] + s * A.a[0], A.p[1] + s * A.a[1], A.p[2] + s * A.a[2]];
  const cb = [B.p[0] + t * B.a[0], B.p[1] + t * B.a[1], B.p[2] + t * B.a[2]];
  const d = [ca[0] - cb[0], ca[1] - cb[1], ca[2] - cb[2]];
  const dist = Math.hypot(d[0], d[1], d[2]);
  const n = dist > 1e-9 ? d.map((x) => x / dist) : [0, 0, 1];
  return { dist, pen: A.r + B.r - dist, n, ca, cb, s, t };
}

// ---------------------------------------------------------------- support
// Static-equilibrium bookkeeping shared by the pivot rule (sim.js) and the
// diagnostics (diag.js): where a body is held up, and whether its centre of
// mass sits over that support.

/**
 * Support points of body `b`: its particles within `floorTol` of the floor
 * and the closest points to every other capsule within `gapTol`.
 * parts: world particles (flat), [start, start+count) belong to b;
 * segs: world capsules for all bodies.
 */
// Tolerances default to the model's contact radius (6 mm): the network
// acts on anything that close, so anything that close is "support" as far
// as the model is concerned. Tighter tests left a band where the model
// held a body (edges exist) but no analytic rule applied (no support), and
// a pencil hovered 5.5 mm above its neighbour for half a second.
export function supportPoints(parts, start, count, segs, b, floorTol = 6e-3, gapTol = 6e-3) {
  const points = [], capsule = [];
  let floor = 0;
  for (let i = start; i < start + count; i++)
    if (parts[3 * i + 2] < floorTol) {
      points.push([parts[3 * i], parts[3 * i + 1], parts[3 * i + 2]]); floor++;
    }
  for (let j = 0; j < segs.length; j++) {
    if (j === b) continue;
    const c = capsuleClosest(segs[b], segs[j]);
    // a neighbour supports b only from below: its contact normal (j -> b)
    // must point up. A pencil touched only by pencils lying on top of it
    // was counted as supported and hovered with two others on its back.
    if (-c.pen < gapTol && c.n[2] > 0.2) { points.push(c.ca); capsule.push(j); }
  }
  return { points, floor, capsule };
}

/**
 * Horizontal distance from p to the convex hull of pts, and the 3D point
 * on the hull boundary nearest to p (the hinge a tipping body rotates
 * about: an edge of the support polygon, or a lone contact).
 */
export function distToHull2D(p, pts) {
  if (!pts.length) return { dist: Infinity, nearest: null };
  const P = pts.slice();
  if (P.length === 1) return { dist: Math.hypot(p[0] - P[0][0], p[1] - P[0][1]), nearest: [...P[0]] };
  // monotone chain hull in xy, keeping the 3D points
  P.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lower = [], upper = [];
  for (const q of P) { while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], q) <= 0) lower.pop(); lower.push(q); }
  for (let i = P.length - 1; i >= 0; i--) { const q = P[i]; while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], q) <= 0) upper.pop(); upper.push(q); }
  const hull = lower.slice(0, -1).concat(upper.slice(0, -1));
  const segNearest = (a, b) => {
    const dx = b[0] - a[0], dy = b[1] - a[1], l2 = dx * dx + dy * dy;
    const t = l2 > 0 ? Math.max(0, Math.min(1, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / l2)) : 0;
    const q = [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]), a[2] + t * (b[2] - a[2])];
    return { d: Math.hypot(p[0] - q[0], p[1] - q[1]), q };
  };
  if (hull.length < 3) {
    const s = segNearest(P[0], P[P.length - 1]);
    return { dist: s.d, nearest: s.q };
  }
  let inside = true, best = { d: Infinity, q: null };
  for (let i = 0; i < hull.length; i++) {
    const a = hull[i], b = hull[(i + 1) % hull.length];
    if (cross(a, b, p) < 0) inside = false;
    const s = segNearest(a, b);
    if (s.d < best.d) best = s;
  }
  return { dist: inside ? 0 : best.d, nearest: best.q };
}

/**
 * Is the centre of mass over the support? `tol` is the half-width a round
 * contact effectively spans (a 6 mm pencil on a point contact is stable
 * within about its radius). Also reports the support's spread and the
 * hinge point (nearest point of the support polygon's boundary) for the
 * pivot rule.
 */
export function supportAnalysis(com, points, tol = 6e-3) {
  if (!points.length) return { n: 0, balanced: false, dist: Infinity, spread: 0, hinge: null };
  let spread = 0;
  for (let i = 0; i < points.length; i++) for (let j = i + 1; j < points.length; j++)
    spread = Math.max(spread, Math.hypot(points[i][0] - points[j][0], points[i][1] - points[j][1]));
  const { dist, nearest } = distToHull2D(com, points);
  return { n: points.length, balanced: dist <= tol, dist, spread, hinge: nearest };
}

// ------------------------------------------------------------------ grabs

/**
 * Spring-damper grab force pulling the grab point `wp` toward `target`,
 * with gravity feed-forward so a held body sits at the cursor instead of
 * sagging g/omega^2 below it. Capped at the force the model was trained
 * with, and cut while a guard is pushing the held body out of another (at
 * the full cap the spring drove it ~9 mm into the neighbour every step).
 * Shared by the demo's pointer path and the headless interaction tests.
 */
export function grabForce({ m, v, wp, target, omega, zeta, gravity, capG, blocked = false }) {
  const kp = m * omega * omega, kd = 2 * zeta * m * omega;
  const f = [
    kp * (target[0] - wp[0]) - kd * v[0],
    kp * (target[1] - wp[1]) - kd * v[1],
    kp * (target[2] - wp[2]) - kd * v[2] + m * gravity];
  // `blocked` used to cut the whole force to a third while a guard was
  // separating the held body, which also meant a wedged pencil could not
  // be lifted out of a pile at all (it never rose, and rotated to 77
  // degrees instead). Only the component pushing further into the
  // obstruction is cut; the rest of the grab still works.
  const cap = capG * m * gravity;
  if (Array.isArray(blocked)) {
    const n = blocked;                       // unit normal out of the obstruction
    const into = f[0] * n[0] + f[1] * n[1] + f[2] * n[2];
    if (into < 0) for (let k = 0; k < 3; k++) f[k] -= 0.66 * into * n[k];
  }
  const fn = Math.hypot(f[0], f[1], f[2]);
  if (fn > cap) for (let k = 0; k < 3; k++) f[k] *= cap / fn;
  return f;
}

// ---------------------------------------------------------------- features

export function actionFeature(parts, sel, point, force, mass, sigma) {
  // Gaussian falloff, weights sum to 1, scaled by particle count
  const idx = [];
  for (let i = 0; i < sel.length; i++) if (sel[i]) idx.push(i);
  const w = new Float64Array(idx.length);
  let wsum = 0;
  idx.forEach((i, k) => {
    const dx = parts[3 * i] - point[0], dy = parts[3 * i + 1] - point[1],
      dz = parts[3 * i + 2] - point[2];
    w[k] = Math.exp(-(dx * dx + dy * dy + dz * dz) / (2 * sigma * sigma));
    wsum += w[k];
  });
  const out = new Map();
  idx.forEach((i, k) => {
    const s = (w[k] / Math.max(wsum, 1e-12)) * idx.length;
    out.set(i, [force[0] / mass * s, force[1] / mass * s, force[2] / mass * s]);
  });
  return out;
}

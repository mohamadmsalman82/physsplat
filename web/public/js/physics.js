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
  const cell = r;
  const pairs = new Set();
  const key = (x, y, z) => `${x},${y},${z}`;

  const passes = vels ? 2 : 1;
  for (let pass = 0; pass < passes; pass++) {
    const grid = new Map();
    const px = new Float64Array(N * 3);
    for (let i = 0; i < N * 3; i++)
      px[i] = pass === 0 ? parts[i] : parts[i] + vels[i] * dt;
    for (let i = 0; i < N; i++) {
      const cx = Math.floor(px[3 * i] / cell),
        cy = Math.floor(px[3 * i + 1] / cell),
        cz = Math.floor(px[3 * i + 2] / cell);
      const k = key(cx, cy, cz);
      if (!grid.has(k)) grid.set(k, []);
      grid.get(k).push(i);
    }
    for (let i = 0; i < N; i++) {
      const cx = Math.floor(px[3 * i] / cell),
        cy = Math.floor(px[3 * i + 1] / cell),
        cz = Math.floor(px[3 * i + 2] / cell);
      for (let dx = -1; dx <= 1; dx++)
        for (let dy = -1; dy <= 1; dy++)
          for (let dz = -1; dz <= 1; dz++) {
            const bucket = grid.get(key(cx + dx, cy + dy, cz + dz));
            if (!bucket) continue;
            for (const j of bucket) {
              if (j <= i) continue;
              const ddx = px[3 * i] - px[3 * j],
                ddy = px[3 * i + 1] - px[3 * j + 1],
                ddz = px[3 * i + 2] - px[3 * j + 2];
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

/**
 * RapierSim: the pencils simulated by a real rigid-body solver.
 *
 * Rapier (github.com/dimforge/rapier) is a Rust physics engine compiled to
 * WebAssembly: continuous collision detection, Coulomb friction, restitution,
 * sleeping, and a solver that stacks long thin bodies without drifting.
 * This is the engine the demo runs by default. The learned graph network
 * (sim.js) is still here behind ?engine=gnn, and the two present the same
 * interface, so the diagnostics, the sensors, the probes and the page
 * itself do not know which one is underneath.
 *
 * Why this exists, plainly. The learned model is an approximation trained
 * on synthetic capsules, and eight review rounds and three player reports
 * were spent adding analytic rules around its failure modes: free flight,
 * pivoting, settling, seating, stiction, swept collision, held-body caps.
 * Each rule was measured and each earned its place, and the pencils still
 * did not lie flat, still hovered, still twitched. A rigid-body solver has
 * none of those failure modes to patch. What made a solver usable here is
 * the one change that came out of that work: every pencil is now one
 * canonical shape (js/pencil.js), so there is a clean collider to give it.
 *
 * Units. Rapier's tolerances are tuned for metre-scale objects and a pencil
 * is 9 mm across, so the world is simulated in CENTIMETRES: positions x100,
 * gravity 981, density in kg/cm^3 so masses stay in kg, forces x100. The
 * interface converts, and everything a caller sees is in metres.
 *
 * The pencil's collider is a compound of five convex hulls revolved from
 * the canonical profile: eraser and cap, barrel, grip, lower barrel, cone
 * and lead. A single hull would fill the step between barrel and grip; five
 * keep the grip 1 mm fatter than the barrel, which is what a Matic Grip
 * actually rests on.
 */
import { LENGTH, PROFILE, MASS } from "./pencil.js";
import { quatToMatrix } from "./physics.js";

const CM = 100;                      // metres to simulation units
const DENSITY = MASS / 9.38e-6;      // kg/m^3 for the canonical solid, 661
// Per 1/60 s step. Measured on a 250 mm drop onto the IMG_8596 pile: at 4
// substeps the impact sank 7.1 mm into the pile for a frame, at 8 it was
// 1.9 mm, at 16 it is 0.45 mm and no longer at the impact at all. A step
// costs 0.46 ms at 16, against 33 ms for the learned model, so there is no
// reason to be coarser.
const SUBSTEPS = 16;
// Plastic on plastic is low: at 0.42 a pile rode along on a pencil pulled
// slowly out from under it instead of staying and dropping. Rapier
// averages the two coefficients at a contact, so pencil on desk is 0.36.
const FRICTION_PENCIL = 0.28, FRICTION_DESK = 0.45;
// Nearly inelastic. At 0.12 a pencil balanced on its grip and clip rocked
// at 0.58 rad/s indefinitely in IMG_8596 without ever sleeping; a real
// plastic-on-plastic contact eats that in a second.
const RESTITUTION_PENCIL = 0.04, RESTITUTION_DESK = 0.03;
const AIR_DAMPING_LIN = 0.05, AIR_DAMPING_ANG = 0.08;
// Angular damping while held, 1/s. Fingers hold a pencil's orientation;
// at 30 the pile unloading under a held pencil still kicked it to 2 rad/s.
// 200 is a 5 ms time constant, and an end grab still dangles: gravity's
// torque on it is sustained, 438 rad/s^2 over this damping is 2.2 rad/s.
const PINCH_DAMPING = 200;
const SETTLE_STEPS = 240;            // on reset: 4 s at 60 Hz, a few ms of wall time
const REST_V = 2e-3, REST_W = 0.05;  // m/s, rad/s: below this a body counts as at rest

// how the profile splits into convex parts: [t0, t1, name]
const PARTS = [
  [0.000, 0.065, "cap"],
  [0.065, 0.740, "barrel"],
  [0.740, 0.873, "grip"],
  [0.873, 0.929, "lower barrel"],
  [0.929, 1.000, "cone"],
];
// A hull is an inscribed polygon, so it sits (1 - cos(pi/n)) of the radius
// inside the true circle: 0.1 mm per body at 16 sides, which read as 0.2 mm
// of "penetration" against the analytic profile at rest. 32 sides is 0.03.
const AZIMUTHS = 32;
// The clip, standing proud of the barrel on the body's +z side, the way the
// real one does and the way the drawing puts it. It is not decoration: a
// round pencil rolls off a pile at the slightest nudge, and the clip is
// what stops it. With 16-sided hulls the flats were doing that job by
// accident; with round ones IMG_8504 rolled 33 mm in its first ten seconds.
const CLIP_FROM_TOP = 0.125, CLIP_LEN = 0.185, CLIP_W = 0.44, CLIP_PROUD = 0.0012, CLIP_T = 0.0020;

/** Points (cm, body frame, +x along the pencil) of the profile revolved
 * between t0 and t1, including both end rings. */
function hullPoints(t0, t1) {
  const knots = PROFILE.filter(([t]) => t >= t0 - 1e-9 && t <= t1 + 1e-9);
  const pts = [];
  for (const [t, r] of knots) {
    const x = (t - 0.5) * LENGTH * CM;
    const rr = Math.max(r, 1e-4) * CM;
    for (let k = 0; k < AZIMUTHS; k++) {
      const a = (k / AZIMUTHS) * 2 * Math.PI;
      pts.push(x, rr * Math.cos(a), rr * Math.sin(a));
    }
  }
  return new Float32Array(pts);
}

export class RapierSim {
  /**
   * R: the initialised rapier module. runtime: the same runtime.json the
   * learned simulator uses, for dt and gravity. packet: a scene packet, or
   * {bodies: []} until loadScene assigns one and calls reset().
   */
  constructor(R, runtime, packet, opts = {}) {
    this.R = R;
    this.rt = runtime;
    this.packet = packet;
    this.substeps = opts.substeps ?? SUBSTEPS;
    this.ccdSubsteps = opts.ccdSubsteps ?? 4;
    this.solverIterations = opts.solverIterations ?? 8;
    this.backend = { kind: "rapier" };
    this.groundGuard = true;             // the solver is the guard
    this.swept = true;
    this.energyRule = false; this.smooth = false;
    this.world = null;
    this.gen = 0;
    this.reset();
  }

  reset() {
    const R = this.R;
    if (this.world) { this.world.free(); this.world = null; }
    const bs = this.packet.bodies;
    this.gen++;
    this.B = bs.length;
    this.mass = bs.map(() => MASS);
    this.inertia = bs.map((b) => b.inertia);
    this.offsets = bs.map((b) => b.offsets);
    this.counts = bs.map((b) => b.offsets.length);
    this.N = this.counts.reduce((a, c) => a + c, 0);
    this.state = {
      pos: bs.map((b) => [...b.pos]),
      quat: bs.map((b) => [...b.quat]),
      linvel: bs.map(() => [0, 0, 0]),
      angvel: bs.map(() => [0, 0, 0]),
    };
    this.restCount = new Int32Array(this.B);
    this.stepCount = 0;
    this.last = null;
    this.sweptInfo = null;
    this.loadCorrection = { lift_mm: bs.map(() => 0) };
    this.timing = { features_ms: 0, graph_ms: 0, net_ms: 0, N: this.N, E: 0, backend: "rapier" };
    this.held = -1;
    this.seatGap = new Float64Array(this.B);
    this.balancedNow = new Uint8Array(this.B).fill(1);
    this.pivoting = new Uint8Array(this.B);

    this.world = new R.World({ x: 0, y: 0, z: -9.81 * CM });
    this.world.timestep = this.rt.dt / this.substeps;
    this.world.maxCcdSubsteps = this.ccdSubsteps;
    const ip = this.world.integrationParameters;
    ip.numSolverIterations = this.solverIterations;

    // the desk: a slab whose top face is z = 0
    const desk = this.world.createRigidBody(R.RigidBodyDesc.fixed().setTranslation(0, 0, -5));
    this.world.createCollider(
      R.ColliderDesc.cuboid(500, 500, 5).setFriction(FRICTION_DESK).setRestitution(RESTITUTION_DESK), desk);

    this.bodies = [];
    this.colliders = [];
    bs.forEach((b, i) => {
      const q = b.quat;
      const desc = R.RigidBodyDesc.dynamic()
        .setTranslation(b.pos[0] * CM, b.pos[1] * CM, b.pos[2] * CM)
        .setRotation({ x: q[0], y: q[1], z: q[2], w: q[3] })
        .setLinearDamping(AIR_DAMPING_LIN)
        .setAngularDamping(AIR_DAMPING_ANG)
        .setCcdEnabled(true)
        .setCanSleep(true);
      const body = this.world.createRigidBody(desc);
      const cols = [];
      for (const [t0, t1] of PARTS) {
        const cd = R.ColliderDesc.convexHull(hullPoints(t0, t1));
        if (!cd) continue;
        cd.setDensity(DENSITY / (CM * CM * CM))       // kg/m^3 -> kg/cm^3
          .setFriction(FRICTION_PENCIL)
          .setRestitution(RESTITUTION_PENCIL);
        cols.push(this.world.createCollider(cd, body));
      }
      {
        const rBarrel = PROFILE[4][1];
        const clipLen = CLIP_LEN * LENGTH, clipW = CLIP_W * rBarrel * 2;
        const cx = (-0.5 + CLIP_FROM_TOP) * LENGTH + clipLen / 2;
        const cz = rBarrel + CLIP_PROUD - CLIP_T / 2;
        const cd = R.ColliderDesc.cuboid(clipLen / 2 * CM, clipW / 2 * CM, CLIP_T / 2 * CM)
          .setTranslation(cx * CM, 0, cz * CM)
          .setDensity(DENSITY / (CM * CM * CM))
          .setFriction(FRICTION_PENCIL)
          .setRestitution(RESTITUTION_PENCIL);
        cols.push(this.world.createCollider(cd, body));
      }
      this.bodies.push(body);
      this.colliders.push(cols);
    });
    // masses as the solver sees them, for the interface and for the grab
    this.mass = this.bodies.map((b) => b.mass());
    this.#readBack();

    // Let the pile come to rest under this solver before anyone sees it.
    // The packet's poses were settled by the learned model, which is not
    // this model, and a few millimetres of disagreement is a pencil that
    // drops or rolls in the first frames otherwise.
    // until everything sleeps, within reason: a pile that is still moving
    // when the first frame is drawn is a pile that appears to move by itself
    for (let k = 0; k < SETTLE_STEPS * 5; k++) {
      this.#advance();
      if (k >= SETTLE_STEPS && this.bodies.every((b) => b.isSleeping())) break;
    }
    this.#readBack();
    this.stepCount = 0;
  }

  #newLast() {
    const B = this.B;
    return {
      residual: new Float64Array(B * 6),
      ext: { lin: new Float64Array(B * 3), ang: new Float64Array(B * 3) },
      action: null,
      guard: {
        ground: new Float64Array(B), capsule: new Float64Array(B), pairs: [],
        settled: new Uint8Array(B), freeFlight: new Uint8Array(B), pivot: new Uint8Array(B),
        energyScale: 1, normal: {}, seat: 0, swept: 0,
      },
    };
  }

  #advance() {
    for (let s = 0; s < this.substeps; s++) this.world.step();
  }

  #readBack() {
    for (let i = 0; i < this.B; i++) {
      const b = this.bodies[i];
      const t = b.translation(), q = b.rotation(), v = b.linvel(), w = b.angvel();
      this.state.pos[i] = [t.x / CM, t.y / CM, t.z / CM];
      this.state.quat[i] = [q.x, q.y, q.z, q.w];
      this.state.linvel[i] = [v.x / CM, v.y / CM, v.z / CM];
      this.state.angvel[i] = [w.x, w.y, w.z];
    }
  }

  /** Write the interface state back into the solver (a harness that edits
   * sim.state.pos or linvel directly, as the suites do, expects it to take). */
  #writeState() {
    for (let i = 0; i < this.B; i++) {
      const b = this.bodies[i];
      const p = this.state.pos[i], q = this.state.quat[i], v = this.state.linvel[i], w = this.state.angvel[i];
      const t = b.translation(), r = b.rotation(), lv = b.linvel(), av = b.angvel();
      const near = (a, c, tol) => Math.abs(a - c) < tol;
      if (!near(t.x, p[0] * CM, 1e-6) || !near(t.y, p[1] * CM, 1e-6) || !near(t.z, p[2] * CM, 1e-6))
        b.setTranslation({ x: p[0] * CM, y: p[1] * CM, z: p[2] * CM }, true);
      if (!near(r.x, q[0], 1e-9) || !near(r.y, q[1], 1e-9) || !near(r.z, q[2], 1e-9) || !near(r.w, q[3], 1e-9))
        b.setRotation({ x: q[0], y: q[1], z: q[2], w: q[3] }, true);
      if (!near(lv.x, v[0] * CM, 1e-6) || !near(lv.y, v[1] * CM, 1e-6) || !near(lv.z, v[2] * CM, 1e-6))
        b.setLinvel({ x: v[0] * CM, y: v[1] * CM, z: v[2] * CM }, true);
      if (!near(av.x, w[0], 1e-9) || !near(av.y, w[1], 1e-9) || !near(av.z, w[2], 1e-9))
        b.setAngvel({ x: w[0], y: w[1], z: w[2] }, true);
    }
  }

  /**
   * One 1/60 s step. actBody/actPoint/actForce: a force (N) applied at a
   * world point (m) on a body; actPinch: the action is a held grab, which
   * also holds orientation the way fingers do.
   */
  async step(actBody = -1, actPoint = null, actForce = null, actPinch = false) {
    const gen = this.gen;
    const t0 = performance.now();
    this.#writeState();
    this.last = this.#newLast();

    // pinch: fingers resist rotation; restore air damping on release
    if (this.held >= 0 && (this.held !== actBody || !actPinch)) {
      this.bodies[this.held]?.setAngularDamping(AIR_DAMPING_ANG);
      this.held = -1;
    }
    if (actPinch && actBody >= 0 && this.held !== actBody) {
      this.bodies[actBody].setAngularDamping(PINCH_DAMPING);
      this.held = actBody;
    }

    for (const b of this.bodies) { b.resetForces(false); b.resetTorques(false); }
    if (actBody >= 0 && actPoint && actForce) {
      const b = this.bodies[actBody];
      b.wakeUp();
      // N -> kg cm/s^2
      b.addForceAtPoint(
        { x: actForce[0] * CM, y: actForce[1] * CM, z: actForce[2] * CM },
        { x: actPoint[0] * CM, y: actPoint[1] * CM, z: actPoint[2] * CM }, true);
      const m = this.mass[actBody];
      for (let k = 0; k < 3; k++) this.last.ext.lin[3 * actBody + k] = actForce[k] / m;
      this.last.action = { body: actBody, point: actPoint, force: actForce, pinch: actPinch };
    }

    this.#advance();
    if (gen !== this.gen) return this.state;
    this.#readBack();

    let E = 0;
    for (let i = 0; i < this.B; i++) {
      const b = this.bodies[i];
      const v = this.state.linvel[i], w = this.state.angvel[i];
      const slow = Math.hypot(v[0], v[1], v[2]) < REST_V && Math.hypot(w[0], w[1], w[2]) < REST_W;
      this.restCount[i] = (b.isSleeping() || slow) ? this.restCount[i] + 1 : 0;
      this.last.guard.settled[i] = b.isSleeping() ? 1 : 0;
      for (const c of this.colliders[i]) this.world.contactPairsWith(c, () => { E++; });
    }
    this.timing = { features_ms: 0, graph_ms: 0, net_ms: performance.now() - t0,
      N: this.N, E, backend: "rapier" };
    this.stepCount++;
    return this.state;
  }

  /** Flat Float64Array of world particle positions (m), as PhysSim gives. */
  particlesWorld() {
    const out = new Float64Array(this.N * 3);
    let k = 0;
    for (let b = 0; b < this.B; b++) {
      const Rm = quatToMatrix(this.state.quat[b]), p = this.state.pos[b];
      for (const o of this.offsets[b]) {
        out[k++] = Rm[0] * o[0] + Rm[1] * o[1] + Rm[2] * o[2] + p[0];
        out[k++] = Rm[3] * o[0] + Rm[4] * o[1] + Rm[5] * o[2] + p[1];
        out[k++] = Rm[6] * o[0] + Rm[7] * o[1] + Rm[8] * o[2] + p[2];
      }
    }
    return out;
  }
}

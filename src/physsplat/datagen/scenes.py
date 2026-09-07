"""PyBullet scene generation and trajectory recording.

One call to `simulate(seed)` produces one trajectory (or None if rejected by
the physics filters): a randomized scene from one of seven regimes, simulated
at INTERNAL_HZ and recorded at 1/DT, with a poke-and-grab action channel.

Regimes (design doc, data section):
  stack       generic shapes stacked with offsets, settled
  scattered   objects lying flat, settled                (pencils common case)
  drop        objects falling at recording start
  bundle      capsules parallel and touching, settled            (pencil)
  crosshatch  2-4 alternating layers, settled                    (pencil)
  pyramid     capsules resting in grooves of a parallel pair     (pencil)
  pile        sequential criss-cross drops -> tangle             (pencil, demo)

Actions: pokes (short force bursts) and grabs (mass-scaled spring-damper to
a moving target, often on a load-bearing body). Both recorded per frame as
(body, world point, force) so training targets can subtract the analytic
Newton-Euler contribution.
"""

from dataclasses import dataclass, field

import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation

from ..common import constants as C
from ..common.geometry import particles_world, random_yaw_flat_quat
from ..common.particles import sample_mesh_surface
from ..verify import invariants as inv
from .shapes import Shape, pencil, random_generic

SUBSTEPS = C.INTERNAL_HZ // int(round(1 / C.DT))  # 4

PENCIL_REGIMES = ["bundle", "crosshatch", "pyramid", "pile", "scattered"]
GENERIC_REGIMES = ["stack", "scattered", "drop"]


@dataclass
class Body:
    shape: Shape
    pid: int
    mass: float
    inertia_diag: np.ndarray
    offsets: np.ndarray          # (P, 3) body-frame particle offsets


@dataclass
class Trajectory:
    seed: int
    regime: str
    bodies: list[Body]
    pos: np.ndarray              # (T, B, 3)
    quat: np.ndarray             # (T, B, 4) xyzw
    linvel: np.ndarray           # (T, B, 3)
    angvel: np.ndarray           # (T, B, 3)
    act_body: np.ndarray         # (T,) int, -1 = no action
    act_point: np.ndarray        # (T, 3) world application point
    act_force: np.ndarray        # (T, 3) applied force (N)
    stats: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# Scene construction
# --------------------------------------------------------------------------

def _spawn(shape: Shape, pos, quat) -> int:
    kw = {}
    if shape.kind == "capsule":
        col = p.createCollisionShape(
            p.GEOM_CAPSULE, radius=shape.dims["radius"], height=shape.dims["height"]
        )
    elif shape.kind == "box":
        col = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=np.asarray(shape.dims["extents"]) / 2
        )
    else:
        col = p.createCollisionShape(p.GEOM_MESH, vertices=shape.vertices, **kw)
    pid = p.createMultiBody(
        baseMass=shape.mass, baseCollisionShapeIndex=col,
        basePosition=pos, baseOrientation=quat,
    )
    p.changeDynamics(
        pid, -1,
        lateralFriction=shape.friction,
        restitution=shape.restitution,
        rollingFriction=shape.rolling_friction,
        spinningFriction=0.002,
    )
    if shape.kind == "capsule":
        # continuous collision detection: thin fast bodies must not tunnel
        p.changeDynamics(pid, -1, ccdSweptSphereRadius=shape.dims["radius"] * 0.4)
    return pid


def _lying(rng, shape: Shape, x, y, extra_z=0.0, yaw=None):
    """Pose for a capsule lying flat (or a generic shape resting)."""
    if shape.kind == "capsule":
        r = shape.dims["radius"]
        if yaw is None:
            q = random_yaw_flat_quat(rng)
        else:
            q = (Rotation.from_euler("z", yaw) * Rotation.from_euler("y", 90, degrees=True)).as_quat()
        return [x, y, r + extra_z], q
    h = 0.03 if shape.kind == "poly" else max(shape.dims["extents"]) / 2
    return [x, y, h + 0.005 + extra_z], Rotation.from_euler("z", rng.uniform(0, 6.28)).as_quat()


def build_scene(rng: np.random.Generator) -> tuple[str, list[tuple[Shape, list, np.ndarray]]]:
    """Choose a regime and return (regime, [(shape, pos, quat), ...])."""
    pencil_scene = rng.random() < 0.65
    regime = str(rng.choice(PENCIL_REGIMES if pencil_scene else GENERIC_REGIMES))
    make = pencil if pencil_scene else random_generic
    placed = []

    if regime == "bundle":
        n = int(rng.integers(2, 5))
        shapes = [make(rng) for _ in range(n)]
        y = 0.0
        for s in shapes:
            r = s.dims["radius"]
            y += r
            pos, q = _lying(rng, s, rng.uniform(-0.005, 0.005), y, yaw=rng.normal(0, 0.03))
            placed.append((s, pos, q))
            y += r + 0.0002

    elif regime == "crosshatch":
        layers = int(rng.integers(2, 5))
        gap = rng.uniform(0.03, 0.06)
        z = 0.0
        for k in range(layers):
            s1, s2 = make(rng), make(rng)
            r = max(s1.dims["radius"], s2.dims["radius"])
            along_x = k % 2 == 0
            for s, off in [(s1, -gap / 2), (s2, gap / 2)]:
                yaw = (0 if along_x else np.pi / 2) + rng.normal(0, 0.05)
                x, y = (rng.normal(0, 0.004), off) if along_x else (off, rng.normal(0, 0.004))
                pos, q = _lying(rng, s, x, y, extra_z=z, yaw=yaw)
                placed.append((s, pos, q))
            z += 2 * r + 0.0005

    elif regime == "pyramid":
        base_n = int(rng.integers(2, 4))
        shapes = [make(rng) for _ in range(base_n)]
        # High-friction base (the BIC's rubber grip regime): with low friction
        # the top pencil wedges the base pair apart and every pyramid
        # collapses in settle; some must survive to appear in the data.
        for s in shapes:
            s.friction = float(rng.uniform(0.55, 0.9))
        r0 = shapes[0].dims["radius"]
        y = 0.0
        ys = []
        for s in shapes:
            r = s.dims["radius"]
            y += r
            ys.append(y)
            pos, q = _lying(rng, s, 0.0, y, yaw=rng.normal(0, 0.02))
            placed.append((s, pos, q))
            y += r
        for i in range(base_n - 1):
            s = make(rng)
            rt = s.dims["radius"]
            vy = (ys[i] + ys[i + 1]) / 2
            # resting height of a cylinder in the groove of a touching pair:
            # centers are (r0 + rt) apart, base pair spaced 2*r0 -> the top
            # sits at z = sqrt((r0+rt)^2 - r0^2) above the base centers
            groove_h = float(np.sqrt((r0 + rt) ** 2 - r0**2))
            # _lying puts the center at rt + extra_z; base centers are at r0
            pos, q = _lying(rng, s, 0.0, vy, extra_z=r0 + groove_h - rt + 0.0005,
                            yaw=rng.normal(0, 0.02))
            placed.append((s, pos, q))

    elif regime == "pile":
        n = int(rng.integers(3, 7))
        for i in range(n):
            s = make(rng)
            q = (
                Rotation.from_euler("z", rng.uniform(0, 6.28))
                * Rotation.from_euler("y", 90 + rng.normal(0, 8), degrees=True)
            ).as_quat()
            # 4 cm vertical spacing: a near-flat 15 cm pencil spans ~2 cm
            # vertically, so consecutive spawns can never interpenetrate
            posn = [rng.uniform(-0.03, 0.03), rng.uniform(-0.03, 0.03), 0.04 + i * 0.04]
            placed.append((s, posn, q))

    elif regime == "scattered":
        n = int(rng.integers(2, 6))
        for _ in range(n):
            s = make(rng)
            pos, q = _lying(rng, s, rng.uniform(-0.10, 0.10), rng.uniform(-0.10, 0.10),
                            extra_z=0.005)
            placed.append((s, pos, q))

    elif regime == "stack":
        n = int(rng.integers(2, 5))
        z, x0, y0 = 0.0, rng.uniform(-0.02, 0.02), rng.uniform(-0.02, 0.02)
        for _ in range(n):
            s = random_generic(rng)
            while s.kind == "capsule":       # capsules don't stack face-on
                s = random_generic(rng)
            h = 0.04 if s.kind == "poly" else s.dims["extents"][2]
            pos = [x0 + rng.normal(0, 0.006), y0 + rng.normal(0, 0.006), z + h / 2 + 0.003]
            placed.append((s, pos, Rotation.from_euler("z", rng.uniform(0, 6.28)).as_quat()))
            z += h + 0.003

    else:  # drop
        n = int(rng.integers(2, 6))
        for i in range(n):
            s = make(rng)
            # 7 cm vertical spacing: even a 7 cm box at any orientation
            # cannot overlap its neighbor at spawn
            posn = [rng.uniform(-0.04, 0.04), rng.uniform(-0.04, 0.04), 0.05 + i * 0.07]
            placed.append((s, posn, Rotation.random(rng=rng).as_quat()))

    return regime, placed


# --------------------------------------------------------------------------
# Actions
# --------------------------------------------------------------------------

@dataclass
class Poke:
    start: int
    body: int
    local_point: np.ndarray
    force: np.ndarray
    frames: int = C.IMPULSE_STEPS


@dataclass
class Grab:
    start: int
    frames: int
    body: int
    local_point: np.ndarray
    waypoints: np.ndarray        # (K, 3) world targets


def _schedule_actions(rng, bodies: list[Body], regime: str, settled: bool):
    actions = []
    first_ok = 30 if settled else 80    # let drops land first
    n_pokes = int(rng.integers(0, 3))
    for _ in range(n_pokes):
        b = int(rng.integers(len(bodies)))
        body = bodies[b]
        dv = rng.uniform(*C.POKE_DELTA_V)
        direction = rng.normal(size=3)
        direction[2] = abs(direction[2]) * rng.uniform(0, 0.8)   # mostly lateral
        direction /= np.linalg.norm(direction)
        force = direction * body.mass * dv / (C.IMPULSE_STEPS * C.DT)
        pt = body.offsets[rng.integers(len(body.offsets))]
        actions.append(Poke(int(rng.integers(first_ok, C.RECORD_STEPS - 10)), b, pt, force))

    if rng.random() < 0.5:
        b = int(rng.integers(len(bodies)))
        body = bodies[b]
        start = int(rng.integers(first_ok, C.RECORD_STEPS - 40))
        frames = int(rng.integers(30, min(120, C.RECORD_STEPS - start)))
        pt = body.offsets[rng.integers(len(body.offsets))]
        k = int(rng.integers(2, 4))
        way = np.cumsum(
            np.concatenate([np.zeros((1, 3)),
                            rng.uniform([-0.06, -0.06, 0.0], [0.06, 0.06, 0.10], (k, 3))]),
            axis=0,
        )
        actions.append(Grab(start, frames, b, pt, way))
    return actions


# --------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------

def simulate(seed: int) -> tuple[Trajectory | None, str]:
    """Returns (trajectory, reason). reason is "ok" for kept trajectories,
    else which physics filter rejected it (the datagen health metric)."""
    rng = np.random.default_rng(seed)
    cid = p.connect(p.DIRECT)
    try:
        p.setGravity(0, 0, -C.GRAVITY)
        # Stiff, clean contacts: the learned model imitates this ground truth,
        # penetration included, so overlap here becomes overlap in the demo.
        p.setPhysicsEngineParameter(
            fixedTimeStep=1.0 / C.INTERNAL_HZ, numSolverIterations=100,
            restitutionVelocityThreshold=0.03,
            contactERP=0.9, useSplitImpulse=1,
            splitImpulsePenetrationThreshold=-0.0015,
        )
        plane = p.createCollisionShape(p.GEOM_PLANE)
        ground = p.createMultiBody(0, plane)
        p.changeDynamics(ground, -1, lateralFriction=1.0, restitution=1.0)

        regime, placed = build_scene(rng)
        bodies: list[Body] = []
        for shape, pos, quat in placed:
            pid = _spawn(shape, pos, quat)
            info = p.getDynamicsInfo(pid, -1)
            offsets = sample_mesh_surface(shape.to_trimesh(), seed=seed + len(bodies))
            bodies.append(Body(shape, pid, shape.mass, np.array(info[2]), offsets))

        settled = regime in ("stack", "scattered", "bundle", "crosshatch", "pyramid")
        if settled:
            # Long settle: metastable arrangements (solver-creep topples) must
            # fail HERE, not during recording -- a "settled" scene that falls
            # spontaneously teaches the model to knock over stable stacks.
            for _ in range(3 * C.SETTLE_STEPS * SUBSTEPS):
                p.stepSimulation()
            speeds = [np.linalg.norm(p.getBaseVelocity(b.pid)[0]) for b in bodies]
            if max(speeds) > 0.02:
                for _ in range(2 * C.SETTLE_STEPS * SUBSTEPS):
                    p.stepSimulation()
            # kill residual solver jitter so recording starts from true rest
            for b in bodies:
                p.resetBaseVelocity(b.pid, [0, 0, 0], [0, 0, 0])

        actions = _schedule_actions(rng, bodies, regime, settled)

        T, B = C.RECORD_STEPS, len(bodies)
        pos = np.zeros((T, B, 3), np.float32)
        quat = np.zeros((T, B, 4), np.float32)
        linvel = np.zeros((T, B, 3), np.float32)
        angvel = np.zeros((T, B, 3), np.float32)
        act_body = np.full(T, -1, np.int16)
        act_point = np.zeros((T, 3), np.float32)
        act_force = np.zeros((T, 3), np.float32)
        pen_per_frame = np.zeros(T, np.float32)

        for t in range(T):
            active = None
            for a in actions:
                if a.start <= t < a.start + a.frames:
                    active = a
                    break
            for _ in range(SUBSTEPS):
                if active is not None:
                    body = bodies[active.body]
                    bp, bq = p.getBasePositionAndOrientation(body.pid)
                    world_pt = particles_world(
                        active.local_point[None], np.array(bp), np.array(bq)
                    )[0]
                    if isinstance(active, Poke):
                        force = active.force
                    else:
                        frac = np.clip((t - active.start) / active.frames, 0, 1)
                        seg = frac * (len(active.waypoints) - 1)
                        i0 = min(int(seg), len(active.waypoints) - 2)
                        tgt_off = active.waypoints[i0] + (seg - i0) * (
                            active.waypoints[i0 + 1] - active.waypoints[i0]
                        )
                        if not hasattr(active, "_anchor"):
                            active._anchor = world_pt.copy()
                        target = active._anchor + tgt_off
                        vel = np.array(p.getBaseVelocity(body.pid)[0])
                        m = body.mass
                        kp, kd = m * C.GRAB_OMEGA**2, 2 * C.GRAB_ZETA * m * C.GRAB_OMEGA
                        force = kp * (target - world_pt) - kd * vel
                        cap = C.GRAB_FORCE_CAP * m * C.GRAVITY
                        n = np.linalg.norm(force)
                        if n > cap:
                            force = force * (cap / n)
                    p.applyExternalForce(body.pid, -1, force.tolist(),
                                         world_pt.tolist(), p.WORLD_FRAME)
                p.stepSimulation()

            for b, body in enumerate(bodies):
                bp, bq = p.getBasePositionAndOrientation(body.pid)
                lv, av = p.getBaseVelocity(body.pid)
                pos[t, b], quat[t, b] = bp, bq
                linvel[t, b], angvel[t, b] = lv, av
            if active is not None:
                act_body[t] = active.body
                act_point[t] = world_pt
                act_force[t] = force
            for c in p.getContactPoints():
                pen_per_frame[t] = max(pen_per_frame[t], -min(c[8], 0.0))

        # ---------------- rejection filters (verify/) ----------------
        pen_p95 = float(np.percentile(pen_per_frame, 95))
        pen_max = float(pen_per_frame.max())
        # Settled scenes must STAY settled until acted on. Metastable
        # arrangements that creep-topple on their own would teach the model
        # to knock over stable structures spontaneously.
        drifted = False
        if settled:
            first_act = min((a.start for a in actions), default=T)
            if first_act > 0:
                disp = np.linalg.norm(pos[:first_act] - pos[0], axis=-1)
                drifted = bool(disp.max() > 0.005)
        reason = "ok"
        if not np.isfinite(pos).all():
            reason = "nan"
        elif drifted:
            reason = "unsettled"
        elif inv.max_speed(linvel) > C.SPEED_REJECT:
            reason = "speed"
        elif inv.min_height(pos) < -0.005:
            reason = "below_ground"
        elif not inv.within_bounds(pos):
            reason = "bounds"
        elif pen_p95 > C.PENETRATION_SUSTAINED or pen_max > C.PENETRATION_SPIKE:
            reason = "penetration"
        if reason != "ok":
            return None, reason

        return Trajectory(
            seed, regime, bodies, pos, quat, linvel, angvel,
            act_body, act_point, act_force,
            stats={"max_penetration": pen_max, "penetration_p95": pen_p95,
                   "max_speed": inv.max_speed(linvel)},
        ), reason
    finally:
        p.disconnect(cid)

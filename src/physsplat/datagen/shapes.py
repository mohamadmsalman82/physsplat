"""Random object family for synthetic scenes.

Three shape types, all convex (the project's scope boundary):
- capsule: pencils, markers (elongated cylinders use capsule collision
  primitives because cylinder-cylinder point contacts are numerically
  fragile in PyBullet)
- box: erasers, blocks, dice
- poly: random convex polyhedra, irregular odds and ends

Every shape is centered on its center of mass, so PyBullet's base frame,
the trimesh frame, and the particle body frame all coincide. Mass follows
volume at a randomized density; inertia is whatever PyBullet computes for
the collision shape (recorded via getDynamicsInfo), because the SIM's
inertia is the ground truth the model must match.
"""

from dataclasses import dataclass, field

import numpy as np
import trimesh

from ..common.constants import DENSITY_RANGE, FRICTION_RANGE, RESTITUTION_RANGE

# BIC Matic Grip envelope (docs/objects.md): 150mm x ~9-11mm, ~6 g.
PENCIL_RADIUS_RANGE = (0.0035, 0.0055)   # m
PENCIL_LENGTH_RANGE = (0.120, 0.160)     # m


@dataclass
class Shape:
    kind: str                 # "capsule" | "box" | "poly"
    dims: dict                # kind-specific dimensions
    density: float
    friction: float
    restitution: float
    rolling_friction: float
    vertices: np.ndarray | None = field(default=None, repr=False)  # poly only

    def to_trimesh(self) -> trimesh.Trimesh:
        if self.kind == "capsule":
            m = trimesh.creation.capsule(
                radius=self.dims["radius"], height=self.dims["height"]
            )
        elif self.kind == "box":
            m = trimesh.creation.box(extents=self.dims["extents"])
        else:
            m = trimesh.Trimesh(vertices=self.vertices).convex_hull
        # Guarantee COM at origin regardless of trimesh's creation convention,
        # so the mesh frame == the PyBullet base frame == the particle frame.
        m.apply_translation(-m.center_mass)
        return m

    @property
    def volume(self) -> float:
        return float(self.to_trimesh().volume)

    @property
    def mass(self) -> float:
        return self.density * self.volume


def _materials(rng: np.random.Generator) -> dict:
    return dict(
        density=float(rng.uniform(*DENSITY_RANGE)),
        friction=float(rng.uniform(*FRICTION_RANGE)),
        restitution=float(rng.uniform(*RESTITUTION_RANGE)),
        # Rolling resistance is what lets pencils stop rolling; randomize it.
        # Floor at 1e-3: real desks stop a BIC pencil in a second or two, and
        # a 1e-4 floor made poked pencils roll clean out of the scene bounds.
        rolling_friction=float(10 ** rng.uniform(-3, -2)),
    )


def pencil(rng: np.random.Generator) -> Shape:
    r = float(rng.uniform(*PENCIL_RADIUS_RANGE))
    length = float(rng.uniform(*PENCIL_LENGTH_RANGE))
    return Shape("capsule", {"radius": r, "height": length - 2 * r}, **_materials(rng))


def random_generic(rng: np.random.Generator) -> Shape:
    kind = rng.choice(["box", "capsule", "poly"], p=[0.45, 0.30, 0.25])
    mats = _materials(rng)
    if kind == "box":
        e = rng.uniform(0.02, 0.07, 3)
        e[rng.integers(3)] *= rng.uniform(0.5, 1.0)  # some flat/long boxes
        return Shape("box", {"extents": e.astype(float).tolist()}, **mats)
    if kind == "capsule":
        r = float(rng.uniform(0.004, 0.012))
        length = float(rng.uniform(0.03, 0.16))
        return Shape("capsule", {"radius": r, "height": max(length - 2 * r, 0.01)}, **mats)
    pts = rng.normal(size=(16, 3)) * rng.uniform(0.012, 0.03, 3)
    hull = trimesh.Trimesh(vertices=pts).convex_hull
    verts = np.asarray(hull.vertices - hull.center_mass, dtype=np.float64)
    return Shape("poly", {}, vertices=verts, **mats)

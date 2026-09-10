"""The canonical BIC Matic Grip.

One shape, defined once, used by everything: the particles the physics
moves, the radii the contact guard separates bodies by, and the mesh the
browser draws. They agreed only approximately before, and every visible
complaint about the demo traced back to that.

Why a canonical shape at all. The photo determines where each pencil is,
which way it points, and what colour it is. It does not determine the
pencil's shape, because the shape is not in question: they are all the same
mass-produced object, 150 mm long, 9 mm across the barrel and 11 mm across
the rubber grip. Taking the shape from the reconstruction instead gave
bodies that measured 99 to 150 mm long with elliptical cross-sections and
ends that tapered where a real pencil is straight, so no two pencils in a
scene were the same object.

Three reported failures follow from that and from the uniform-radius
capsule the contact guard used:

  a pencil resting on another pencil's POINT was held up as though the
  point were 9 mm thick, because the capsule carries the barrel's radius
  all the way to the tip;

  pencils hovered, because the drawn surface sat inside the body the guard
  was separating;

  pencils twitched and slid on their own, because the guard pushed with a
  radius the object does not have and gravity pulled back, every step.

The profile below is a radius as a function of position along the axis,
measured off a real pencil, running from the eraser end (t = 0) to the lead
point (t = 1). Revolving it gives the solid; sampling that solid with the
shared particle sampler gives the physics body; evaluating it at the
contact parameter gives the guard its radius; and web/public/js/pencil.js
holds the same numbers for the drawing.

Keep the two files in step. `test_pencil_profile_matches_js` in
tests/test_pencil.py fails if they drift.
"""

from __future__ import annotations

import numpy as np
import trimesh

LENGTH = 0.150          # m, tip of the lead to the top of the eraser
BARREL_R = 0.0045       # m, the 9 mm barrel
GRIP_R = 0.0055         # m, the 11 mm rubber grip
MASS = 0.0062           # kg, weighed: a real Matic Grip is 6.2 g

# (t along the pencil, radius in m). t = 0 is the eraser end, t = 1 the
# point. Duplicated t values are deliberate: they are the square shoulders
# where one part meets the next.
PROFILE: tuple[tuple[float, float], ...] = (
    (0.000, 0.00279),   # eraser, standing out of the cap
    (0.020, 0.00279),
    (0.020, 0.00387),   # cap
    (0.065, 0.00423),
    (0.065, BARREL_R),  # barrel
    (0.740, BARREL_R),
    (0.740, BARREL_R),  # grip, moulded so it swells out of the barrel. The
    (0.760, GRIP_R),    # repeated knots here and at 0.873 are zero-length
    (0.853, GRIP_R),    # and change no geometry; they exist so the drawing
    (0.873, BARREL_R),  # can put a crisp colour edge on the rubber.
    (0.873, BARREL_R),
    (0.929, BARREL_R),  # cone: barrel-coloured plastic on this pencil
    (0.929, 0.00437),
    (0.984, 0.00135),
    (0.984, 0.00108),   # lead
    (1.000, 0.00032),
)


def radius_at(t: np.ndarray | float) -> np.ndarray:
    """Radius (m) at fractional position t along the pencil, t = 0 at the
    eraser and t = 1 at the point. Outside [0, 1] the radius is 0, so a
    caller that runs off the end gets no contact rather than a phantom one.
    """
    t = np.asarray(t, dtype=np.float64)
    ts = np.array([p[0] for p in PROFILE])
    rs = np.array([p[1] for p in PROFILE])
    # np.interp takes the LAST of duplicated x values, which is what the
    # shoulders want: at t = 0.065 the barrel radius wins over the cap's.
    out = np.interp(np.clip(t, 0.0, 1.0), ts, rs)
    return np.where((t < 0.0) | (t > 1.0), 0.0, out)


def radius_at_offset(s: np.ndarray | float, half: float = LENGTH / 2) -> np.ndarray:
    """Radius (m) at signed distance `s` from the pencil's centre, with the
    point at +s. This is the form the contact guard wants, since it works in
    body-frame offsets along the axis rather than in fractions.
    """
    return radius_at((np.asarray(s, dtype=np.float64) + half) / (2 * half))


def solid(sections: int = 48) -> trimesh.Trimesh:
    """The pencil as a watertight surface of revolution about +x, centred on
    its own length, point at +x.
    """
    ts = np.array([p[0] for p in PROFILE])
    rs = np.array([p[1] for p in PROFILE])
    x = (ts - 0.5) * LENGTH
    # trimesh.creation.revolve takes the linestring as (radius, height) and
    # spins it about z. Closing the profile onto the axis at both ends is
    # what makes the result a solid rather than an open tube.
    poly = np.column_stack([
        np.concatenate([[0.0], rs, [0.0]]),
        np.concatenate([[x[0]], x, [x[-1]]]),
    ])
    m = trimesh.creation.revolve(poly, sections=sections)
    # built along z; turn z into the body's long axis, point at +x
    m.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
    m.merge_vertices()
    if not m.is_volume:
        m.fill_holes()
    return m


def particles(spacing: float | None = None) -> np.ndarray:
    """Body-frame particle offsets (P, 3) for the physics, sampled off the
    canonical solid with the SAME sampler that dots the synthetic training
    objects. That shared sampler is the whole sim-to-real mechanism, so this
    goes through it rather than placing points analytically.
    """
    from .particles import farthest_point_sample
    from .constants import PARTICLE_SPACING

    surf = solid()
    pts = trimesh.sample.sample_surface(surf, 40000, seed=1)[0]
    return farthest_point_sample(pts, spacing=spacing or PARTICLE_SPACING)


def inertia(mass: float = MASS) -> np.ndarray:
    """Principal moments (3,) about the centre of mass, body frame, x along
    the pencil. Taken from the real solid at uniform density rather than
    from a cylinder approximation, so the grip's extra material and the
    cone's missing material both count.
    """
    m = solid()
    return np.abs(np.diag(m.moment_inertia * (mass / m.volume)))


def density(mass: float = MASS) -> float:
    """Uniform density (kg/m^3) that makes the canonical pencil weigh
    `mass`. Reported so it can be checked against the training range: the
    network sees mass and inertia as inputs, and a pencil outside
    DENSITY_RANGE would be out of distribution.
    """
    return mass / solid().volume

"""Action-feature encoding, shared between training and inference.

A recorded or live action is (body, world application point, force vector).
The per-particle feature spreads the force over the acted body's particles
with Gaussian falloff from the application point. The SAME function must run
at training time (here) and in the demo (TypeScript port, parity-tested):
any mismatch teaches the model one poke and shows it another.
"""

import numpy as np

from .constants import PARTICLE_SPACING

FALLOFF_SIGMA = 2.0 * PARTICLE_SPACING


def action_feature(
    world_particles: np.ndarray,   # (P, 3) particles of the acted body
    point: np.ndarray,             # (3,) world application point
    force: np.ndarray,             # (3,) force in N
    mass: float,
) -> np.ndarray:
    """Per-particle acceleration feature (P, 3). Weights sum to 1 so the
    encoded total equals F/m regardless of particle count."""
    d2 = ((world_particles - point) ** 2).sum(-1)
    w = np.exp(-d2 / (2 * FALLOFF_SIGMA**2))
    w = w / max(w.sum(), 1e-12)
    return (force / mass)[None, :] * w[:, None] * len(world_particles)

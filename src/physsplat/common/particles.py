"""The shared surface particle sampler.

This exact function runs on synthetic training objects (Phase 1) AND on
photo-reconstructed objects (Phase 6). Sim-to-real transfer depends on both
sides producing statistically identical particle sets, so there is one
implementation, here, and nowhere else.

Fixed target SPACING (not fixed count): a pencil and a large box then have
the same local particle density, so one global contact radius works for all
shapes. Farthest-point sampling gives near-uniform coverage regardless of
how the underlying surface points are distributed (dense mesh samples here,
lumpy Gaussians in Phase 6).
"""

import numpy as np
import trimesh

from .constants import MAX_PARTICLES_PER_BODY, PARTICLE_SPACING


def farthest_point_sample(
    points: np.ndarray,
    spacing: float = PARTICLE_SPACING,
    cap: int = MAX_PARTICLES_PER_BODY,
    seed: int = 0,
) -> np.ndarray:
    """Subsample (N, 3) points to ~uniform `spacing` via farthest-point
    sampling. Deterministic for a given input and seed. Returns (P, 3).

    Stops when the farthest remaining point is closer than `spacing` to the
    picked set (coverage achieved) or when `cap` is hit (huge bodies get
    coarser sampling, which the cap documents deliberately).
    """
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) == 0:
        raise ValueError("no points to sample")
    rng = np.random.default_rng(seed)
    first = int(rng.integers(len(pts)))
    picked = [first]
    d = np.linalg.norm(pts - pts[first], axis=1)
    while len(picked) < cap:
        i = int(d.argmax())
        if d[i] < spacing:
            break
        picked.append(i)
        d = np.minimum(d, np.linalg.norm(pts - pts[i], axis=1))
    return pts[picked].astype(np.float32)


def sample_mesh_surface(
    mesh: trimesh.Trimesh,
    spacing: float = PARTICLE_SPACING,
    cap: int = MAX_PARTICLES_PER_BODY,
    seed: int = 0,
    oversample: int = 20000,
) -> np.ndarray:
    """Sample a mesh's surface at ~uniform spacing. Returns (P, 3) points in
    the MESH's own frame (the body frame, since shapes are COM-centered)."""
    dense, _ = trimesh.sample.sample_surface(mesh, oversample, seed=seed)
    return farthest_point_sample(dense, spacing=spacing, cap=cap, seed=seed)

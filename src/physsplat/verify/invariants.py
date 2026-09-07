"""Numeric physics invariants. All pure functions of arrays."""

import numpy as np

from ..common.constants import GRAVITY, SCENE_XY_MAX, SCENE_Z_MAX
from ..common.geometry import quat_to_mat


def rigidity_error(offsets: np.ndarray, pos: np.ndarray, quat: np.ndarray) -> float:
    """Max deviation of particle pairwise distances from the body-frame
    reference, for one body at one pose. Rigid motion -> ~machine epsilon.
    offsets (P,3); pos (3,); quat (4,) xyzw."""
    world = offsets @ quat_to_mat(quat).T + pos
    ref = np.linalg.norm(offsets[:, None] - offsets[None, :], axis=-1)
    cur = np.linalg.norm(world[:, None] - world[None, :], axis=-1)
    return float(np.abs(cur - ref).max())


def rotation_validity_error(quat: np.ndarray) -> float:
    """How far R is from a proper rotation: max of |R R^T - I| and |det-1|.
    quat (..., 4) xyzw."""
    r = quat_to_mat(quat)
    eye = np.eye(3)
    ortho = np.abs(r @ np.swapaxes(r, -1, -2) - eye).max()
    det = np.abs(np.linalg.det(r) - 1.0).max()
    return float(max(ortho, det))


def energy(
    pos: np.ndarray, linvel: np.ndarray, angvel: np.ndarray,
    mass: np.ndarray, inertia_diag: np.ndarray, quat: np.ndarray,
) -> float:
    """Total mechanical energy of a frame: KE(lin) + KE(rot) + PE.
    pos/linvel/angvel (B,3); mass (B,); inertia_diag (B,3) body-frame; quat (B,4).
    Rotational KE uses omega expressed in the body frame."""
    ke_lin = 0.5 * (mass * (linvel**2).sum(-1)).sum()
    r = quat_to_mat(quat)                       # (B,3,3)
    w_body = np.einsum("bij,bj->bi", r.swapaxes(-1, -2), angvel)
    ke_rot = 0.5 * (inertia_diag * w_body**2).sum()
    pe = GRAVITY * (mass * pos[:, 2]).sum()
    return float(ke_lin + ke_rot + pe)


def max_speed(linvel: np.ndarray) -> float:
    return float(np.linalg.norm(linvel, axis=-1).max())


def min_height(pos: np.ndarray) -> float:
    return float(pos[..., 2].min())


def within_bounds(pos: np.ndarray) -> bool:
    return bool(
        (np.abs(pos[..., :2]) < SCENE_XY_MAX).all() and (pos[..., 2] < SCENE_Z_MAX).all()
    )

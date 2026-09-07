"""Rotation and frame helpers shared by datagen, model, and recon.

Conventions (fixed project-wide):
- Quaternions are (x, y, z, w), matching both PyBullet and scipy.
- z is up; gravity acts along -z.
- A body pose is (position of center of mass, rotation). A particle with
  body-frame offset r sits at x_world = R @ r + t.
"""

import numpy as np
from scipy.spatial.transform import Rotation


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    """(..., 4) xyzw quaternion -> (..., 3, 3) rotation matrix."""
    return Rotation.from_quat(np.asarray(q)).as_matrix()


def particles_world(offsets: np.ndarray, pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
    """Body-frame offsets (P, 3) + pose -> world positions (P, 3)."""
    return offsets @ quat_to_mat(quat).T + np.asarray(pos)


def random_yaw_flat_quat(rng: np.random.Generator) -> np.ndarray:
    """Orientation for a capsule lying flat on the ground: its long (z) axis
    rotated into the horizontal plane, then a random yaw. Returns xyzw."""
    lie = Rotation.from_euler("y", 90, degrees=True)
    yaw = Rotation.from_euler("z", rng.uniform(0, 360), degrees=True)
    return (yaw * lie).as_quat()


def random_quat(rng: np.random.Generator) -> np.ndarray:
    return Rotation.random(rng=rng).as_quat()

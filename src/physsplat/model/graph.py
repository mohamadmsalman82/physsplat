"""Contact graph construction.

Edges connect particle pairs within CONTACT_RADIUS, from the union of
current positions and velocity-extrapolated positions (x + v*dt): a fast
approach is sensed one step early instead of crossing the sensing shell
between steps (the tunneling budget, design doc).

Runs on CPU numpy (scipy cKDTree); called by the dataloader collate at
training time and by the rollout loop at inference. The TypeScript port
reimplements this with a grid hash; the cross-language parity test pins
equivalence.
"""

import numpy as np
from scipy.spatial import cKDTree

from ..common.constants import CONTACT_RADIUS, DT


def build_edges(
    particles: np.ndarray,          # (N, 3)
    velocities: np.ndarray | None,  # (N, 3) latest per-particle velocity
    body_ids: np.ndarray,           # (N,)
    radius: float = CONTACT_RADIUS,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (senders, receivers), each (E,), with BOTH directions of every
    edge present and no self edges."""
    pairs = set(map(tuple, cKDTree(particles).query_pairs(radius, output_type="ndarray")))
    if velocities is not None:
        pred = particles + velocities * DT
        pairs |= set(map(tuple, cKDTree(pred).query_pairs(radius, output_type="ndarray")))
    if not pairs:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    e = np.array(sorted(pairs), np.int64)               # (E/2, 2), i<j
    senders = np.concatenate([e[:, 0], e[:, 1]])
    receivers = np.concatenate([e[:, 1], e[:, 0]])
    return senders, receivers


def edge_features(
    particles: np.ndarray, senders: np.ndarray, receivers: np.ndarray,
    body_ids: np.ndarray, radius: float = CONTACT_RADIUS,
) -> np.ndarray:
    """(E, 5): displacement/R (3), distance/R (1), same-body flag (1).
    Normalized by the connectivity radius (GNS convention), so edge features
    need no dataset statistics."""
    disp = (particles[receivers] - particles[senders]) / radius
    dist = np.linalg.norm(disp, axis=-1, keepdims=True)
    same = (body_ids[senders] == body_ids[receivers]).astype(np.float32)[:, None]
    return np.concatenate([disp, dist, same], -1).astype(np.float32)

"""Autoregressive rollout: a loop over LiveSim.

Action replay is open-loop: recorded world-space application points and
forces are applied verbatim even as the rollout diverges from the recorded
trajectory. Good enough for evaluation; the live demo computes spring
forces closed-loop against the simulated state (see LiveSim users).
"""

import numpy as np
import torch

from .live import LiveSim
from .normalize import Normalizer


@torch.no_grad()
def rollout(
    model,
    normalizer: Normalizer,
    scene: dict,          # offsets_list, mass, inertia (numpy)
    init: dict,           # pos/quat/linvel/angvel: (HISTORY, B, ...) numpy
    T: int,
    actions: dict | None = None,   # act_body (T,), act_point (T,3), act_force
    device: str = "cpu",
) -> dict:
    """Returns pos (T, B, 3), quat (T, B, 4) numpy arrays."""
    sim = LiveSim(model, normalizer, scene, init, device)
    out_pos, out_quat = [], []
    for t in range(T):
        if actions is not None and actions["act_body"][t] >= 0:
            p, q = sim.step(int(actions["act_body"][t]),
                            actions["act_point"][t], actions["act_force"][t])
        else:
            p, q = sim.step()
        out_pos.append(p)
        out_quat.append(q)
    return {"pos": np.stack(out_pos), "quat": np.stack(out_quat)}

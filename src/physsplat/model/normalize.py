"""Feature and target normalization.

Velocity and target statistics come from data/stats.json (computed over the
train split; ships with every checkpoint). Mass, inertia, ground distance,
and the action feature use fixed analytic normalizers, chosen so typical
values land in ~[-2, 2]; fixed constants ship trivially to the browser.
"""

import json
from pathlib import Path

import numpy as np
import torch

from ..common.constants import CONTACT_RADIUS, GRAVITY

# masses grams..half-kilo -> log10 in [-2.7, -0.3]
MASS_LOG_SHIFT, MASS_LOG_SCALE = 1.5, 1.0
# inertia diagonals ~1e-8..1e-3 kg m^2
INERTIA_LOG_SHIFT, INERTIA_LOG_SCALE = 6.0, 1.5
A_EXT_SCALE = 3.0 * GRAVITY      # grab force cap is 3 m g


class Normalizer:
    def __init__(self, stats_path: str | Path):
        s = json.loads(Path(stats_path).read_text())
        self.vel_mean = torch.tensor(s["vel_mean"], dtype=torch.float32)
        self.vel_std = torch.tensor(s["vel_std"], dtype=torch.float32)
        self.tgt_mean = torch.tensor(s["target_mean"], dtype=torch.float32)
        self.tgt_std = torch.tensor(s["target_std"], dtype=torch.float32)

    def to(self, device):
        for k in ("vel_mean", "vel_std", "tgt_mean", "tgt_std"):
            setattr(self, k, getattr(self, k).to(device))
        return self

    def node_features(self, batch: dict) -> torch.Tensor:
        """(N, HISTORY*3 + 1 + 1 + 3 + 3) normalized node input."""
        n = batch["particles"].shape[0]
        vel = (batch["vel_hist"] - self.vel_mean) / self.vel_std      # (N,H,3)
        dist = (batch["dist_ground"] / CONTACT_RADIUS)[:, None]
        m = batch["mass"][batch["body_ids"]]
        mass = ((torch.log10(m) + MASS_LOG_SHIFT) / MASS_LOG_SCALE)[:, None]
        iner = batch["inertia"][batch["body_ids"]]
        inertia = (torch.log10(iner.clamp_min(1e-12)) + INERTIA_LOG_SHIFT) / INERTIA_LOG_SCALE
        a_ext = batch["a_ext"] / A_EXT_SCALE
        return torch.cat([vel.reshape(n, -1), dist, mass, inertia, a_ext], -1)

    def body_scalars(self, mass: torch.Tensor, inertia: torch.Tensor) -> torch.Tensor:
        """(B, 4) normalized log mass + log inertia diagonal (decoder input)."""
        m = ((torch.log10(mass) + MASS_LOG_SHIFT) / MASS_LOG_SCALE)[:, None]
        i = (torch.log10(inertia.clamp_min(1e-12)) + INERTIA_LOG_SHIFT) / INERTIA_LOG_SCALE
        return torch.cat([m, i], -1)

    def norm_target(self, t: torch.Tensor) -> torch.Tensor:
        return (t - self.tgt_mean) / self.tgt_std

    def denorm_target(self, t: torch.Tensor) -> torch.Tensor:
        return t * self.tgt_std + self.tgt_mean


NODE_DIM = 5 * 3 + 1 + 1 + 3 + 3
EDGE_DIM = 5

"""PyTorch dataset over recorded trajectories.

One sample = one scene at one instant t (t >= HISTORY-1), containing
everything the model needs to predict the step t -> t+1:

  particles      (N, 3)  world positions of all bodies' particles at t
  vel_hist       (N, HISTORY, 3)  per-particle velocities (rigid-derived)
  body_ids       (N,)    which body each particle belongs to
  dist_ground    (N,)    clip(z, CONTACT_RADIUS)
  a_ext          (N, 3)  action feature (zero unless a poke/grab is active)
  mass           (B,)    per body
  inertia        (B, 3)  per body, principal frame
  quat           (B, 4)  body orientations at t (for world-inertia rotation)
  target         (B, 6)  RESIDUAL accelerations: what actually happened
                         minus gravity minus the applied force's analytic
                         Newton-Euler contribution. Pure contact/friction.

Training-time corruptions (both OFF for eval):
  noise injection    random-walk noise on the velocity history, teaching the
                     model to correct its own rollout errors
  domain randomization  surface noise + particle dropout on offsets, so
                     lumpy photo-reconstructed bodies look in-distribution
"""

from bisect import bisect_right
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from ..common import constants as C
from ..common.actions import action_feature
from ..common.geometry import quat_to_mat


def _split_files(data_dir: str | Path) -> list[Path]:
    files = sorted(Path(data_dir).glob("*.h5"))
    if not files:
        raise FileNotFoundError(f"no .h5 chunks in {data_dir}")
    return files


class TrajectoryDataset(Dataset):
    """Flat index over (trajectory, t) pairs across HDF5 chunk files.

    Split is BY TRAJECTORY (never by frame: adjacent frames are nearly
    identical, and splitting by frame leaks eval into train).
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",           # train | val | test -> 90/5/5
        noise_std: float = 0.0,         # accumulated random-walk noise (m)
        domain_rand: bool = False,
        seed: int = 0,
    ):
        self.files = _split_files(data_dir)
        self.noise_std = noise_std
        self.domain_rand = domain_rand
        self.rng = np.random.default_rng(seed)

        keys = []   # (file_idx, traj_key)
        for fi, path in enumerate(self.files):
            with h5py.File(path) as f:
                keys += [(fi, k) for k in sorted(f.keys())]
        order = np.random.default_rng(12345).permutation(len(keys))  # fixed
        n = len(keys)
        lo, hi = {"train": (0, 0.9), "val": (0.9, 0.95), "test": (0.95, 1.0)}[split]
        self.keys = [keys[i] for i in order[int(lo * n): int(hi * n)]]

        self.frames_per_traj = C.RECORD_STEPS - C.HISTORY  # need t-H+1..t and t+1
        self._handles: dict[int, h5py.File] = {}
        self._cum = np.arange(1, len(self.keys) + 1) * self.frames_per_traj

    def __len__(self):
        return len(self.keys) * self.frames_per_traj

    def _file(self, fi: int) -> h5py.File:
        if fi not in self._handles:                    # one handle per worker
            self._handles[fi] = h5py.File(self.files[fi], "r")
        return self._handles[fi]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ti = bisect_right(self._cum, idx)
        t = idx - (self._cum[ti - 1] if ti else 0) + C.HISTORY - 1
        fi, key = self.keys[ti]
        g = self._file(fi)[key]

        pos = g["pos"][t - C.HISTORY + 1: t + 2]       # (H+1, B, 3)
        quat = g["quat"][t - C.HISTORY + 1: t + 2]
        linvel = g["linvel"][t - C.HISTORY + 1: t + 2]
        angvel = g["angvel"][t - C.HISTORY + 1: t + 2]
        mass = g["mass"][:]
        inertia = g["inertia_diag"][:]
        counts = g["particle_counts"][:]
        offsets = np.split(g["particle_offsets"][:], np.cumsum(counts)[:-1])
        act_body = int(g["act_body"][t])
        act_point = g["act_point"][t]
        act_force = g["act_force"][t]

        B = len(mass)
        H = C.HISTORY

        if self.domain_rand:
            offsets = [
                (o + self.rng.normal(0, self.rng.uniform(0.0005, 0.002), o.shape))[
                    self.rng.random(len(o)) > self.rng.uniform(0.05, 0.15)
                ].astype(np.float32)
                for o in offsets
            ]

        # world particles at t, and per-particle velocity history (rigid):
        # v_i = v_b + omega_b x (R r_i)
        parts, vels, body_ids = [], [], []
        R_hist = quat_to_mat(quat)                     # (H+1, B, 3, 3)
        for b in range(B):
            r_w = offsets[b] @ R_hist[-2, b].T         # frame t is index -2
            parts.append(r_w + pos[-2, b])
            v = (
                linvel[:-1, None, b]
                + np.cross(angvel[:-1, None, b], offsets[b][None] @ np.swapaxes(R_hist[:-1, b], -1, -2))
            )                                          # (H, P, 3)
            vels.append(np.swapaxes(v, 0, 1))          # (P, H, 3)
            body_ids.append(np.full(len(offsets[b]), b))
        particles = np.concatenate(parts).astype(np.float32)
        vel_hist = np.concatenate(vels).astype(np.float32)
        body_ids = np.concatenate(body_ids).astype(np.int64)

        if self.noise_std > 0:
            walk = self.rng.normal(0, self.noise_std / np.sqrt(H), (len(particles), H, 3))
            vel_hist = vel_hist + np.cumsum(walk, axis=1).astype(np.float32) / C.DT * 0.05

        # residual targets: subtract gravity and the analytic Newton-Euler
        # contribution of the recorded external force
        dv = (linvel[-1] - linvel[-2]) / C.DT          # (B, 3) actual accel
        dw = (angvel[-1] - angvel[-2]) / C.DT
        target = np.concatenate([dv, dw], -1).astype(np.float32)
        target[:, 2] += C.GRAVITY
        a_ext = np.zeros((len(particles), 3), np.float32)
        if act_body >= 0:
            m = float(mass[act_body])
            target[act_body, :3] -= act_force / m
            R = R_hist[-2, act_body]
            I_w = R @ np.diag(inertia[act_body]) @ R.T
            torque = np.cross(act_point - pos[-2, act_body], act_force)
            target[act_body, 3:] -= np.linalg.solve(I_w, torque)
            sel = body_ids == act_body
            a_ext[sel] = action_feature(particles[sel], act_point, act_force, m)

        return {
            "particles": torch.from_numpy(particles),
            "vel_hist": torch.from_numpy(vel_hist),
            "body_ids": torch.from_numpy(body_ids),
            "dist_ground": torch.from_numpy(
                np.clip(particles[:, 2], 0, C.CONTACT_RADIUS).astype(np.float32)
            ),
            "a_ext": torch.from_numpy(a_ext),
            "mass": torch.from_numpy(mass),
            "inertia": torch.from_numpy(inertia),
            "quat": torch.from_numpy(quat[-2]),
            "target": torch.from_numpy(target),
        }

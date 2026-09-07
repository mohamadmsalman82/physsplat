"""PyTorch dataset over recorded trajectories.

One sample = one scene at one instant t (t >= HISTORY-1), containing
everything the model needs to predict the step t -> t+1:

  particles      (N, 3)  world positions of all bodies' particles at t
  vel_hist       (N, HISTORY, 3)  per-particle velocities (rigid-derived)
  body_ids       (N,)    which body each particle belongs to
  dist_ground    (N,)    clip(z, CONTACT_RADIUS)
  a_ext          (N, 3)  action feature (zero unless a poke/grab is active)
  mass, inertia  (B,), (B, 3)
  quat           (B, 4)  body orientations at t
  target         (B, 6)  RESIDUAL accelerations: gravity and the recorded
                         external force's Newton-Euler terms subtracted.

Noise injection (GNS-style, matching the reference implementation): a
random-walk perturbation is applied to the BODY state sequence (positions,
orientations, and the velocities implied by the walk), and the target is
recomputed against the noisy state, so that integrating the target from the
perturbed state lands on the TRUE next state. That is what teaches the
model to pull its own rollout errors back toward the data manifold.

Domain randomization corrupts particle offsets only (surface noise +
dropout): lumpy photo-reconstructed bodies must look in-distribution.
"""

from bisect import bisect_right
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset

from ..common import constants as C
from ..common.actions import action_feature
from ..common.geometry import quat_to_mat

# rotational noise per meter of positional noise: 1 mm of walk ~ 1.1 deg
ROT_NOISE_FACTOR = 20.0


def _split_files(data_dir: str | Path) -> list[Path]:
    files = sorted(Path(data_dir).glob("*.h5"))
    if not files:
        raise FileNotFoundError(f"no .h5 chunks in {data_dir}")
    return files


class TrajectoryDataset(Dataset):
    """Flat index over (trajectory, t) pairs across HDF5 chunk files.
    Split is BY TRAJECTORY (frames of one trajectory are nearly identical;
    splitting by frame would leak eval into train)."""

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",           # train | val | test -> 90/5/5
        noise_std: float = 0.0,         # accumulated walk std at frame t (m)
        domain_rand: bool = False,
        seed: int = 0,
        limit_trajs: int | None = None, # overfit mode: use only the first k
    ):
        self.files = _split_files(data_dir)
        self.noise_std = noise_std
        self.domain_rand = domain_rand
        self.rng = np.random.default_rng(seed)

        keys = []
        for fi, path in enumerate(self.files):
            with h5py.File(path) as f:
                keys += [(fi, k) for k in sorted(f.keys())]
        order = np.random.default_rng(12345).permutation(len(keys))  # fixed
        n = len(keys)
        lo, hi = {"train": (0, 0.9), "val": (0.9, 0.95), "test": (0.95, 1.0)}[split]
        self.keys = [keys[i] for i in order[int(lo * n): int(hi * n)]]
        if limit_trajs is not None:
            self.keys = self.keys[:limit_trajs]

        self.frames_per_traj = C.RECORD_STEPS - C.HISTORY
        self._handles: dict[int, h5py.File] = {}
        self._cum = np.arange(1, len(self.keys) + 1) * self.frames_per_traj

    def __getstate__(self):
        s = self.__dict__.copy()
        s["_handles"] = {}          # h5py handles don't survive worker spawn
        return s

    def __len__(self):
        return len(self.keys) * self.frames_per_traj

    def _file(self, fi: int) -> h5py.File:
        if fi not in self._handles:
            self._handles[fi] = h5py.File(self.files[fi], "r")
        return self._handles[fi]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ti = bisect_right(self._cum, idx)
        t = idx - (self._cum[ti - 1] if ti else 0) + C.HISTORY - 1
        fi, key = self.keys[ti]
        g = self._file(fi)[key]

        H = C.HISTORY
        pos = g["pos"][t - H + 1: t + 2].astype(np.float64)     # (H+1, B, 3)
        quat = g["quat"][t - H + 1: t + 2].astype(np.float64)
        linvel = g["linvel"][t - H + 1: t + 2].astype(np.float64)
        angvel = g["angvel"][t - H + 1: t + 2].astype(np.float64)
        mass = g["mass"][:]
        inertia = g["inertia_diag"][:]
        counts = g["particle_counts"][:]
        offsets = np.split(g["particle_offsets"][:].astype(np.float64),
                           np.cumsum(counts)[:-1])
        act_body = int(g["act_body"][t])
        act_point = g["act_point"][t].astype(np.float64)
        act_force = g["act_force"][t].astype(np.float64)
        B = len(mass)

        # ------- noise injection on the body state sequence -------
        # clean t+1 state is kept aside as the truth the target points to
        true_lin_next, true_ang_next = linvel[-1].copy(), angvel[-1].copy()
        if self.noise_std > 0:
            # random walk over the H history/current frames (indices 0..H-1);
            # index H (the true next state) stays clean, so the recomputed
            # target teaches active correction back to the data manifold
            stepn = self.rng.normal(0, self.noise_std / np.sqrt(H), (H, B, 3))
            walk = np.concatenate([np.zeros((1, B, 3)), np.cumsum(stepn, 0)])
            rstep = self.rng.normal(
                0, self.noise_std * ROT_NOISE_FACTOR / np.sqrt(H), (H, B, 3))
            rwalk = np.concatenate([np.zeros((1, B, 3)), np.cumsum(rstep, 0)])
            pos[:-1] += walk[:-1]
            for i in range(H):
                quat[i] = (Rotation.from_rotvec(rwalk[i])
                           * Rotation.from_quat(quat[i])).as_quat()
            # velocities implied by the walk (finite differences); index 0
            # keeps its clean velocity (walk starts at zero there)
            dwalk = np.diff(walk, axis=0) / C.DT                # entry i -> index i+1
            drwalk = np.diff(rwalk, axis=0) / C.DT
            linvel[1:-1] += dwalk[:-1]
            angvel[1:-1] += drwalk[:-1]

        if self.domain_rand:
            offsets = [
                (o + self.rng.normal(0, self.rng.uniform(0.0005, 0.002), o.shape))[
                    self.rng.random(len(o)) > self.rng.uniform(0.05, 0.15)
                ]
                for o in offsets
            ]

        # ------- particles at t, per-particle velocity history -------
        R_hist = quat_to_mat(quat)                              # (H+1, B, 3, 3)
        parts, vels, body_ids = [], [], []
        for b in range(B):
            parts.append(offsets[b] @ R_hist[-2, b].T + pos[-2, b])
            v = (
                linvel[:-1, None, b]
                + np.cross(angvel[:-1, None, b],
                           offsets[b][None] @ np.swapaxes(R_hist[:-1, b], -1, -2))
            )
            vels.append(np.swapaxes(v, 0, 1))
            body_ids.append(np.full(len(offsets[b]), b))
        particles = np.concatenate(parts)
        vel_hist = np.concatenate(vels)
        body_ids = np.concatenate(body_ids).astype(np.int64)

        # ------- residual target, computed against the (noisy) state -------
        dv = (true_lin_next - linvel[-2]) / C.DT
        dw = (true_ang_next - angvel[-2]) / C.DT
        target = np.concatenate([dv, dw], -1)
        target[:, 2] += C.GRAVITY
        a_ext = np.zeros((len(particles), 3))
        if act_body >= 0:
            m = float(mass[act_body])
            target[act_body, :3] -= act_force / m
            R = R_hist[-2, act_body]
            I_w = R @ np.diag(inertia[act_body]) @ R.T
            torque = np.cross(act_point - pos[-2, act_body], act_force)
            target[act_body, 3:] -= np.linalg.solve(I_w, torque)
            sel = body_ids == act_body
            a_ext[sel] = action_feature(particles[sel], act_point, act_force, m)

        f32 = lambda x: torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
        return {
            "particles": f32(particles),
            "vel_hist": f32(vel_hist),
            "body_ids": torch.from_numpy(body_ids),
            "dist_ground": f32(np.clip(particles[:, 2], 0, C.CONTACT_RADIUS)),
            "a_ext": f32(a_ext),
            "mass": f32(mass),
            "inertia": f32(inertia),
            "quat": f32(quat[-2]),
            "target": f32(target),
        }

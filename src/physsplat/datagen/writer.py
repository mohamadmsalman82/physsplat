"""HDF5 trajectory storage.

Stores per-body POSES plus fixed body-frame particle offsets, never raw
particle clouds: ~100x smaller, and ground truth stays exactly rigid by
construction (world particles are always derived as R @ r + t).
"""

from pathlib import Path

import h5py
import numpy as np

from .scenes import Trajectory

SHAPE_KIND = {"capsule": 0, "box": 1, "poly": 2}


def write_trajectories(path: str | Path, trajs: list[Trajectory]) -> None:
    with h5py.File(path, "w") as f:
        for i, tr in enumerate(trajs):
            g = f.create_group(f"traj_{i:05d}")
            g.attrs["seed"] = tr.seed
            g.attrs["regime"] = tr.regime
            for k, v in tr.stats.items():
                g.attrs[k] = v
            for name in ("pos", "quat", "linvel", "angvel", "act_body",
                         "act_point", "act_force"):
                g.create_dataset(name, data=getattr(tr, name), compression="gzip")
            counts = np.array([len(b.offsets) for b in tr.bodies], np.int32)
            g.create_dataset("particle_counts", data=counts)
            g.create_dataset(
                "particle_offsets",
                data=np.concatenate([b.offsets for b in tr.bodies]).astype(np.float32),
                compression="gzip",
            )
            g.create_dataset("mass", data=np.array([b.mass for b in tr.bodies], np.float32))
            g.create_dataset(
                "inertia_diag",
                data=np.stack([b.inertia_diag for b in tr.bodies]).astype(np.float32),
            )
            g.create_dataset(
                "shape_kind",
                data=np.array([SHAPE_KIND[b.shape.kind] for b in tr.bodies], np.int8),
            )
            g.create_dataset(
                "friction",
                data=np.array([b.shape.friction for b in tr.bodies], np.float32),
            )
            g.create_dataset(
                "restitution",
                data=np.array([b.shape.restitution for b in tr.bodies], np.float32),
            )


def load_trajectory(f: h5py.File, key: str) -> dict:
    g = f[key]
    d = {name: g[name][:] for name in g}
    d.update({k: g.attrs[k] for k in g.attrs})
    counts = d["particle_counts"]
    splits = np.cumsum(counts)[:-1]
    d["offsets_list"] = np.split(d["particle_offsets"], splits)
    return d

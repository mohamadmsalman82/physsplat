"""Compute normalization statistics over the TRAIN split only.

Writes data/stats.json: mean/std for velocity features and 6D residual
acceleration targets. These exact numbers ship to the browser in Phase 7;
recomputing them after training starts invalidates the checkpoint.

Edge displacement features are normalized by CONTACT_RADIUS (a constant, per
GNS convention), so they need no dataset statistics.

Usage: uv run python scripts/compute_stats.py data/raw.nosync/train
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from physsplat.train.dataset import TrajectoryDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir", type=str)
    ap.add_argument("--samples", type=int, default=2000)
    ap.add_argument("--out", type=str, default="data/stats.json")
    args = ap.parse_args()

    ds = TrajectoryDataset(args.data_dir, split="train")
    idx = np.random.default_rng(0).choice(len(ds), min(args.samples, len(ds)), replace=False)

    vels, targets = [], []
    for i in idx:
        s = ds[int(i)]
        vels.append(s["vel_hist"].reshape(-1, 3))
        targets.append(s["target"])
    v = torch.cat(vels).numpy()
    t = torch.cat(targets).numpy()

    stats = {
        "vel_mean": v.mean(0).tolist(),
        "vel_std": (v.std(0) + 1e-8).tolist(),
        "target_mean": t.mean(0).tolist(),
        "target_std": (t.std(0) + 1e-8).tolist(),
        "n_samples": int(len(idx)),
    }
    Path(args.out).write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()

"""Evaluate a checkpoint on held-out test trajectories.

    uv run python scripts/evaluate.py checkpoints/run01/latest.pt --n 20

Prints the scorecard, writes eval/history.csv (append-only: the trend
across checkpoints IS the self-debugging signal: improving / worsening /
not working).
"""

import argparse
import csv
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from physsplat.datagen.writer import load_trajectory
from physsplat.eval.metrics import evaluate_trajectory
from physsplat.model.gnn import Simulator
from physsplat.model.normalize import Normalizer
from physsplat.train.dataset import TrajectoryDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--data", default="data/raw.nosync/train")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    ck = torch.load(args.checkpoint, map_location=device)
    model = Simulator(head=ck.get("head", "body")).to(device).eval()
    model.load_state_dict(ck["model"])
    normalizer = Normalizer(ck.get("stats_path", "data/stats.json")).to(device)

    ds = TrajectoryDataset(args.data, split=args.split)
    rows = []
    for fi, key in ds.keys[: args.n]:
        with h5py.File(ds.files[fi]) as f:
            d = load_trajectory(f, key)
        m = evaluate_trajectory(model, normalizer, d, device)
        m["regime"] = d["regime"]
        rows.append(m)
        print(f"{key} [{d['regime']}]: " + " ".join(
            f"{k}={v:.4f}" for k, v in m.items()
            if isinstance(v, float)), flush=True)

    agg = {}
    for k in ("trans_50", "trans_150", "trans_294", "rot_50", "rot_150",
              "rot_294", "axis_50", "axis_150", "axis_294", "penetration"):
        vals = [r[k] for r in rows if k in r]
        if vals:
            agg[k] = float(np.mean(vals))
    stab = [r["stable"] for r in rows if "stable" in r]
    agg["stability"] = float(np.mean(stab)) if stab else float("nan")

    print("\n===== SCORECARD =====")
    for k, v in agg.items():
        print(f"  {k:14s} {v:.4f}")

    hist = Path("eval/history.csv")
    hist.parent.mkdir(exist_ok=True)
    new = not hist.exists()
    with open(hist, "a") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "checkpoint", "step", "n"] + list(agg))
        w.writerow([time.strftime("%Y-%m-%d %H:%M"), args.checkpoint,
                    ck.get("step", -1), len(rows)] + [f"{v:.5f}" for v in agg.values()])
    print(f"appended to {hist}")


if __name__ == "__main__":
    main()

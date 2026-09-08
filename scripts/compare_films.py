"""Side-by-side filmstrips: ground truth (top row) vs learned model (bottom
row), same initial state and same recorded actions.

    uv run python scripts/compare_films.py checkpoints/run01/latest.pt --n 6

The single most informative artifact for judging whether the learned
physics looks real (Gate 2 material).
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image, ImageDraw

from filmstrip import FRAMES, render_traj
from physsplat.common import constants as C
from physsplat.datagen.writer import load_trajectory
from physsplat.model.gnn import Simulator
from physsplat.model.normalize import Normalizer
from physsplat.model.rollout import rollout
from physsplat.train.dataset import TrajectoryDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--data", default="data/raw.nosync/train")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default="data/audit/compare")
    ap.add_argument("--regimes", default=None,
                    help="comma-separated filter, e.g. pile,crosshatch,pyramid")
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    ck = torch.load(args.checkpoint, map_location=device)
    model = Simulator(head=ck.get("head", "body")).to(device).eval()
    model.load_state_dict(ck["model"])
    normalizer = Normalizer(ck.get("stats_path", "data/stats.json")).to(device)

    ds = TrajectoryDataset(args.data, split=args.split)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    H = C.HISTORY

    want = set(args.regimes.split(",")) if args.regimes else None
    done = 0
    for fi, key in ds.keys:
        if done >= args.n:
            break
        with h5py.File(ds.files[fi]) as f:
            d = load_trajectory(f, key)
        if want and d["regime"] not in want:
            continue
        done += 1
        scene = {"offsets_list": d["offsets_list"], "mass": d["mass"],
                 "inertia": d["inertia_diag"]}
        init = {k: d[k][:H] for k in ("pos", "quat", "linvel", "angvel")}
        T = C.RECORD_STEPS - H
        actions = {k: d[k][H:] for k in ("act_body", "act_point", "act_force")}
        out = rollout(model, normalizer, scene, init, T, actions, device)

        gt = render_traj(d)
        dm = dict(d)
        # pad model rollout so FRAMES indices line up with ground truth
        dm["pos"] = np.concatenate([d["pos"][:H], out["pos"]])
        dm["quat"] = np.concatenate([d["quat"][:H], out["quat"]])
        pred = render_traj(dm)

        combo = Image.new("RGB", (gt.width, gt.height * 2 + 24), "white")
        combo.paste(gt, (0, 0))
        combo.paste(pred, (0, gt.height + 24))
        dr = ImageDraw.Draw(combo)
        dr.text((6, gt.height + 5),
                f"^ GROUND TRUTH (PyBullet)   v LEARNED MODEL (step {ck.get('step')})",
                fill=(180, 0, 0))
        path = Path(args.out) / f"{key}_{d['regime']}_step{ck.get('step')}.png"
        combo.save(path)
        print(path, flush=True)


if __name__ == "__main__":
    main()

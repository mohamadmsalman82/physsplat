"""Scrub recorded trajectories in the Rerun viewer (Gate 1 review tool).

Usage:
    uv run python scripts/view_traj.py data/raw/dev.h5 --n 5
Timeline scrubber at the bottom; one entity per body, colored; red arrow =
active poke/grab force.
"""

import argparse

import h5py
import numpy as np
import rerun as rr

from physsplat.common.geometry import particles_world
from physsplat.datagen.writer import load_trajectory

COLORS = np.array([
    [140, 70, 175], [242, 115, 33], [26, 140, 140], [115, 128, 140],
    [204, 51, 90], [51, 90, 204], [153, 153, 51],
], np.uint8)


def log_traj(d: dict, name: str):
    T, B = d["pos"].shape[:2]
    for t in range(T):
        rr.set_time("frame", sequence=t)
        for b in range(B):
            pts = particles_world(d["offsets_list"][b], d["pos"][t, b], d["quat"][t, b])
            rr.log(f"{name}/body{b}", rr.Points3D(pts, radii=0.0012,
                                                  colors=COLORS[b % len(COLORS)]))
        if d["act_body"][t] >= 0:
            f = d["act_force"][t]
            scale = 0.03 / max(np.linalg.norm(f), 1e-9)
            rr.log(f"{name}/action", rr.Arrows3D(
                origins=[d["act_point"][t]], vectors=[f * scale],
                colors=[[220, 30, 30]]))
        else:
            rr.log(f"{name}/action", rr.Clear(recursive=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("h5", type=str)
    ap.add_argument("--n", type=int, default=3)
    args = ap.parse_args()
    rr.init("physsplat-trajectories", spawn=True)
    rr.log("ground", rr.Boxes3D(centers=[[0, 0, -0.001]], half_sizes=[[0.25, 0.25, 0.001]],
                                colors=[[220, 220, 215]]), static=True)
    with h5py.File(args.h5) as f:
        for k in sorted(f.keys())[: args.n]:
            d = load_trajectory(f, k)
            log_traj(d, f"{k}_{d['regime']}")
    print("logged; scrub the timeline in the Rerun viewer")


if __name__ == "__main__":
    main()

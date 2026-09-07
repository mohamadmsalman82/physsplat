"""Render trajectory filmstrips to PNG for visual audit.

Usage:
    uv run python scripts/filmstrip.py data/raw/dev.h5 --n 6 --out data/audit
Each trajectory becomes one row of frames (t=0 .. t=end) so floating,
sinking, exploding, or never-settling scenes are visible at a glance.
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
import pybullet as p
from PIL import Image, ImageDraw

from physsplat.datagen.writer import load_trajectory

COLORS = [
    (0.55, 0.27, 0.68, 1), (0.95, 0.45, 0.13, 1), (0.10, 0.55, 0.55, 1),
    (0.45, 0.50, 0.55, 1), (0.80, 0.20, 0.35, 1), (0.20, 0.35, 0.80, 1),
    (0.60, 0.60, 0.20, 1),
]
FRAMES = [0, 40, 80, 120, 160, 200, 250, 299]
W = H = 300


def render_traj(d: dict) -> Image.Image:
    cid = p.connect(p.DIRECT)
    try:
        pids = []
        n_bodies = d["pos"].shape[1]
        for b in range(n_bodies):
            kind = int(d["shape_kind"][b])
            offs = d["offsets_list"][b]
            if kind == 0:  # capsule: infer dims from offsets extent
                half_len = offs[:, 2].max()
                radius = np.abs(offs[:, :2]).max()
                vis = p.createVisualShape(
                    p.GEOM_CAPSULE, radius=radius,
                    length=max(2 * (half_len - radius), 0.01),
                    rgbaColor=COLORS[b % len(COLORS)],
                )
            else:
                ext = offs.max(0) - offs.min(0)
                vis = p.createVisualShape(
                    p.GEOM_BOX, halfExtents=ext / 2, rgbaColor=COLORS[b % len(COLORS)]
                )
            pids.append(p.createMultiBody(0, baseVisualShapeIndex=vis))
        plane_vis = p.createVisualShape(
            p.GEOM_BOX, halfExtents=[0.25, 0.25, 0.001], rgbaColor=(0.92, 0.92, 0.9, 1)
        )
        p.createMultiBody(0, baseVisualShapeIndex=plane_vis, basePosition=[0, 0, -0.001])

        view = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[0, 0, 0.02], distance=0.42,
            yaw=45, pitch=-35, roll=0, upAxisIndex=2,
        )
        proj = p.computeProjectionMatrixFOV(fov=45, aspect=1.0, nearVal=0.01, farVal=2.0)

        tiles = []
        for t in FRAMES:
            for b, pid in enumerate(pids):
                p.resetBasePositionAndOrientation(pid, d["pos"][t, b], d["quat"][t, b])
            _, _, rgb, _, _ = p.getCameraImage(
                W, H, view, proj, renderer=p.ER_TINY_RENDERER
            )
            img = Image.fromarray(
                np.reshape(np.asarray(rgb, dtype=np.uint8), (H, W, 4))[:, :, :3]
            )
            dr = ImageDraw.Draw(img)
            dr.text((6, 6), f"t={t}", fill=(0, 0, 0))
            if d["act_body"][t] >= 0:
                dr.text((6, 20), f"ACT b{d['act_body'][t]}", fill=(200, 0, 0))
            tiles.append(img)
        strip = Image.new("RGB", (W * len(FRAMES), H + 22), "white")
        for i, im in enumerate(tiles):
            strip.paste(im, (i * W, 22))
        # list every action window (a 3-frame poke is invisible in 8 sampled
        # frames; without this the audit misreads pokes as spontaneous motion)
        ab = d["act_body"]
        windows = []
        t0 = None
        for t in range(len(ab)):
            if ab[t] >= 0 and t0 is None:
                t0 = t
            if (ab[t] < 0 or t == len(ab) - 1) and t0 is not None:
                windows.append(f"b{ab[t0]}@{t0}-{t}")
                t0 = None
        dr = ImageDraw.Draw(strip)
        dr.text((6, 4), f"seed={d['seed']} regime={d['regime']} "
                        f"maxpen={d.get('max_penetration', 0)*1000:.1f}mm "
                        f"actions[{', '.join(windows) or 'none'}]", fill=(0, 0, 0))
        return strip
    finally:
        p.disconnect(cid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("h5", type=str)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--out", type=str, default="data/audit")
    args = ap.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    with h5py.File(args.h5) as f:
        keys = sorted(f.keys())[: args.n]
        for k in keys:
            d = load_trajectory(f, k)
            img = render_traj(d)
            out = Path(args.out) / f"{k}_{d['regime']}.png"
            img.save(out)
            print(out)


if __name__ == "__main__":
    main()

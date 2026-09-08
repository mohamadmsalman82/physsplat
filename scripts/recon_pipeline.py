"""Reconstruction PLY -> scene packet, with a clustering audit image.

    uv run python scripts/recon_pipeline.py data/recon.nosync/IMG_8596.ply

Writes data/packets.nosync/<name>.npz and data/audit/recon/<name>_clusters.png
(left: canonicalized cloud in true colors; right: colored by body id,
gray = noise). The audit image is the pass/fail artifact for Step 53.
"""

import argparse
import pickle
from pathlib import Path

import matplotlib
import numpy as np
import trimesh

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from physsplat.recon.pipeline import mesh_to_packet

ID_COLORS = np.array([
    [140, 70, 175], [242, 115, 33], [26, 140, 140], [115, 128, 140],
    [204, 51, 90], [51, 90, 204], [153, 153, 51], [90, 200, 90],
])


def audit_image(packet, out_png):
    v = packet["verts_scaled"]
    lab = packet["labels"]
    fig = plt.figure(figsize=(12, 5))
    for i, (title, col) in enumerate([
        ("canonicalized (true color)", packet["colors"] / 255.0),
        (f"bodies: {len(packet['offsets_list'])} (gray=noise)",
         np.where(lab[:, None] >= 0,
                  ID_COLORS[lab % len(ID_COLORS)], [200, 200, 200]) / 255.0),
    ]):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        ax.scatter(v[:, 0], v[:, 1], v[:, 2], c=col, s=0.5)
        ax.set_title(title)
        ax.view_init(elev=35, azim=45)
        ax.set_box_aspect((np.ptp(v[:, 0]), np.ptp(v[:, 1]),
                           max(np.ptp(v[:, 2]), 0.01)))
    fig.tight_layout()
    fig.savefig(out_png, dpi=85)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("plys", nargs="+")
    ap.add_argument("--out", default="data/packets.nosync")
    ap.add_argument("--audit", default="data/audit/recon")
    ap.add_argument("--color-weight", type=float, default=0.35)
    ap.add_argument("--min-cluster", type=int, default=120)
    args = ap.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    Path(args.audit).mkdir(parents=True, exist_ok=True)

    for ply in args.plys:
        name = Path(ply).stem
        mesh = trimesh.load(ply, process=False)
        packet = mesh_to_packet(mesh)
        out = Path(args.out) / f"{name}.pkl"
        with open(out, "wb") as f:
            pickle.dump(packet, f)
        audit_image(packet, Path(args.audit) / f"{name}_clusters.png")
        masses = [f"{m*1000:.1f}g" for m in packet["mass"]]
        print(f"{name}: {len(packet['offsets_list'])} bodies, masses {masses} -> {out}")


if __name__ == "__main__":
    main()

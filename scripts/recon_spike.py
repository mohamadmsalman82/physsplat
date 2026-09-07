"""Reconstruction spike: do the pencil photos survive single-image
reconstruction? (The load-bearing unknown of Phase 6.)

    uv run python scripts/recon_spike.py data/photos/A_calibration/IMG_8463.jpg ...

For each photo: rembg -> TripoSR -> mesh with vertex colors -> save PLY to
data/recon.nosync/ + a 4-view turntable PNG to data/audit/recon/ for visual
audit. CPU/MPS, ~1-2 min per image.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "third_party" / "shims"))     # torchmcubes shim
sys.path.insert(0, str(ROOT / "third_party" / "TripoSR"))

import rembg
from tsr.system import TSR
from tsr.utils import remove_background, resize_foreground


def _remap_vit_keys(ckpt: dict) -> dict:
    """The TripoSR checkpoint stores its DINO ViT under the pre-4.5x
    transformers naming; newer transformers renamed the module tree. Pure
    mechanical rename, no weight changes."""
    out = {}
    for k, v in ckpt.items():
        if not k.startswith("image_tokenizer."):
            out[k] = v
            continue
        nk = (k
              .replace(".model.encoder.layer.", ".model.layers.")
              .replace(".attention.attention.query", ".attention.q_proj")
              .replace(".attention.attention.key", ".attention.k_proj")
              .replace(".attention.attention.value", ".attention.v_proj")
              .replace(".attention.output.dense", ".attention.o_proj")
              .replace(".intermediate.dense", ".mlp.fc1"))
        if ".mlp.fc1" not in nk:
            nk = nk.replace(".output.dense", ".mlp.fc2")
        out[nk] = v
    return out


def load_model(device: str):
    from huggingface_hub import hf_hub_download
    from omegaconf import OmegaConf

    config_path = hf_hub_download("stabilityai/TripoSR", "config.yaml")
    weight_path = hf_hub_download("stabilityai/TripoSR", "model.ckpt")
    cfg = OmegaConf.load(config_path)
    OmegaConf.resolve(cfg)
    model = TSR(cfg)
    ckpt = torch.load(weight_path, map_location="cpu")
    try:
        model.load_state_dict(ckpt)
    except RuntimeError:
        model.load_state_dict(_remap_vit_keys(ckpt))
        print("loaded with ViT key remap (new transformers naming)", flush=True)
    model.renderer.set_chunk_size(8192)
    return model.to(device)


def turntable(mesh, out_png: Path, views=4, size=420):
    """Render N azimuths of the vertex-colored mesh as point clouds via
    matplotlib (dependency-free, good enough for auditing shape+color)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pts = mesh.vertices
    col = (mesh.visual.vertex_colors[:, :3] / 255.0
           if mesh.visual.kind == "vertex" else "gray")
    fig, axes = plt.subplots(1, views, figsize=(4 * views, 4),
                             subplot_kw={"projection": "3d"})
    for i, ax in enumerate(np.atleast_1d(axes)):
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=col, s=0.4)
        ax.view_init(elev=18, azim=i * 360 / views)
        ax.set_axis_off()
        ax.set_box_aspect((np.ptp(pts[:, 0]), np.ptp(pts[:, 1]), np.ptp(pts[:, 2])))
    fig.tight_layout()
    fig.savefig(out_png, dpi=80)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--out", default="data/recon.nosync")
    ap.add_argument("--audit", default="data/audit/recon")
    ap.add_argument("--resolution", type=int, default=256)
    args = ap.parse_args()

    device = "cpu"  # TSR triplane query is memory-heavy; CPU is reliable
    Path(args.out).mkdir(parents=True, exist_ok=True)
    Path(args.audit).mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    model = load_model(device)
    print(f"model loaded in {time.time()-t0:.0f}s", flush=True)
    session = rembg.new_session()

    for img_path in args.images:
        name = Path(img_path).stem
        t0 = time.time()
        image = remove_background(Image.open(img_path), session)
        image = resize_foreground(image, 0.85)
        # composite onto gray (TSR convention)
        arr = np.array(image).astype(np.float32) / 255.0
        arr = arr[:, :, :3] * arr[:, :, 3:4] + (1 - arr[:, :, 3:4]) * 0.5
        image = Image.fromarray((arr * 255).astype(np.uint8))

        with torch.no_grad():
            scene_codes = model([image], device=device)
        mesh = model.extract_mesh(scene_codes, True, resolution=args.resolution)[0]
        ply = Path(args.out) / f"{name}.ply"
        mesh.export(ply)
        turntable(mesh, Path(args.audit) / f"{name}.png")
        print(f"{name}: {len(mesh.vertices)} verts, {time.time()-t0:.0f}s -> {ply}",
              flush=True)


if __name__ == "__main__":
    main()

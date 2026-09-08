"""Export a checkpoint to ONNX and run the parity gate.

    uv run python scripts/export_onnx.py checkpoints/run01/latest.pt

Parity: PyTorch and onnxruntime must agree over a 100-step rollout driven
by identical inputs (graph building and integration shared in Python).
No frontend model code gets written until this passes.
"""

import argparse

import h5py
import numpy as np
import onnxruntime as ort
import torch

from physsplat.common import constants as C
from physsplat.datagen.writer import load_trajectory
from physsplat.export.to_onnx import export
from physsplat.model.gnn import Simulator
from physsplat.model.live import LiveSim
from physsplat.model.normalize import Normalizer
from physsplat.train.dataset import TrajectoryDataset


class OnnxModel:
    """Drop-in replacement for the torch Simulator inside LiveSim."""

    def __init__(self, path: str):
        self.sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])

    def __call__(self, node_feats, edge_feats, senders, receivers,
                 body_ids, n_bodies, body_scalars):
        out = self.sess.run(None, {
            "node_feats": node_feats.cpu().numpy(),
            "edge_feats": edge_feats.cpu().numpy(),
            "senders": senders.cpu().numpy(),
            "receivers": receivers.cpu().numpy(),
            "body_ids": body_ids.cpu().numpy(),
            "body_scalars": body_scalars.cpu().numpy(),
        })[0]
        return torch.from_numpy(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--out", default="web/public/model")
    ap.add_argument("--data", default="data/raw.nosync/train")
    ap.add_argument("--steps", type=int, default=100)
    args = ap.parse_args()

    onnx_path = export(args.checkpoint, args.out)
    print(f"exported {onnx_path}")

    ck = torch.load(args.checkpoint, map_location="cpu")
    torch_model = Simulator(head=ck.get("head", "body")).eval()
    torch_model.load_state_dict(ck["model"])
    onnx_model = OnnxModel(str(onnx_path))
    norm = Normalizer(ck.get("stats_path", "data/stats.json"))

    ds = TrajectoryDataset(args.data, split="test")
    fi, key = ds.keys[0]
    with h5py.File(ds.files[fi]) as f:
        d = load_trajectory(f, key)
    H = C.HISTORY
    scene = {"offsets_list": d["offsets_list"], "mass": d["mass"],
             "inertia": d["inertia_diag"]}
    init = {k: d[k][:H] for k in ("pos", "quat", "linvel", "angvel")}

    sims = [LiveSim(m, norm, scene, init, "cpu")
            for m in (torch_model, onnx_model)]
    worst = 0.0
    for t in range(args.steps):
        outs = [s.step() for s in sims]
        worst = max(worst, float(np.abs(outs[0][0] - outs[1][0]).max()))
    print(f"parity over {args.steps} steps: max |pos_torch - pos_onnx| "
          f"= {worst:.2e} m")
    assert worst < 1e-4, "PARITY FAILED"
    print("PARITY PASSED")


if __name__ == "__main__":
    main()

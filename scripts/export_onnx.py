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
    """Drop-in replacement for the torch Simulator inside LiveSim. Applies
    the export ordering contract (edges sorted by receiver, segment
    pointers) on the host, exactly as the browser runtime does."""

    def __init__(self, path: str):
        self.sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])

    def __call__(self, node_feats, edge_feats, senders, receivers,
                 body_ids, n_bodies, body_scalars):
        from physsplat.export.to_onnx import sort_for_export
        bids = body_ids.cpu().numpy()
        snd, rcv, ef, seg_ptr, body_ptr = sort_for_export(
            senders.cpu().numpy(), receivers.cpu().numpy(),
            edge_feats.cpu().numpy(), len(bids), bids, n_bodies)
        out = self.sess.run(None, {
            "node_feats": node_feats.cpu().numpy(),
            "edge_feats": ef, "senders": snd, "receivers": rcv,
            "seg_ptr": seg_ptr, "body_ids": bids.astype(np.int64),
            "body_ptr": body_ptr,
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

    # Gate 1 (exact): the exported network must reproduce the torch network
    # on identical inputs. This is the export-correctness test.
    sim = LiveSim(torch_model, norm, scene, init, "cpu")
    captured = {}
    orig_forward = torch_model.forward

    def spy(*a, **k):
        out = orig_forward(*a, **k)
        captured["args"], captured["out"] = a, out
        return out
    torch_model.forward = spy
    with torch.no_grad():
        sim.step()
    torch_model.forward = orig_forward
    onnx_out = onnx_model(*captured["args"])
    net_err = float((onnx_out - captured["out"]).abs().max())
    print(f"single-step network parity: max |onnx - torch| = {net_err:.2e}")
    assert net_err < 1e-4, "NETWORK PARITY FAILED"

    # Gate 2 (sanity): rollout divergence. Contact dynamics are chaotic, so
    # float32 op-order differences grow over steps; this bounds the growth
    # rather than demanding exactness.
    sims = [LiveSim(m, norm, scene, init, "cpu")
            for m in (torch_model, onnx_model)]
    worst = 0.0
    with torch.no_grad():
        for t in range(args.steps):
            outs = [s.step() for s in sims]
            worst = max(worst, float(np.abs(outs[0][0] - outs[1][0]).max()))
    print(f"rollout divergence over {args.steps} steps: {worst:.2e} m")
    assert worst < 5e-3, "ROLLOUT DIVERGENCE TOO LARGE"
    print("PARITY PASSED")


if __name__ == "__main__":
    main()

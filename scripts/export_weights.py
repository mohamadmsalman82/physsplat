"""Dump Simulator weights for the custom WebGPU backend.

    uv run python scripts/export_weights.py checkpoints/final_v1.pt

Writes web/public/model/weights.bin (all float32 tensors concatenated) and
weights.json (name -> {offset (floats), shape}). Linear weights are stored
as [out, in] exactly as torch keeps them; the WGSL GEMM reads W^T.
Also writes a parity fixture: one real forward pass (inputs captured from a
LiveSim step on a photo packet, outputs from the torch model) so the GPU
implementation can be checked in the browser to ~1e-4.
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch

from physsplat.export.to_onnx import sort_for_export
from physsplat.model.gnn import Simulator
from physsplat.model.live import LiveSim
from physsplat.model.normalize import Normalizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--out", default="web/public/model")
    ap.add_argument("--packet", default="data/packets.nosync/IMG_8596.pkl")
    args = ap.parse_args()
    out = Path(args.out)
    ck = torch.load(args.checkpoint, map_location="cpu")
    model = Simulator(head=ck.get("head", "body")).eval()
    model.load_state_dict(ck["model"])

    # ---- weights ----
    manifest, chunks, offset = {}, [], 0
    for name, t in model.state_dict().items():
        a = t.detach().cpu().numpy().astype(np.float32).ravel()
        manifest[name] = {"offset": offset, "shape": list(t.shape)}
        chunks.append(a)
        offset += a.size
    (out / "weights.bin").write_bytes(np.concatenate(chunks).tobytes())
    (out / "weights.json").write_text(json.dumps({
        "tensors": manifest, "latent": model.blocks[0].edge_mlp[0].out_features,
        "layers": len(model.blocks), "ln_eps": 1e-5, "step": ck.get("step")}))
    print(f"weights: {offset*4/1e6:.1f} MB, {len(manifest)} tensors")

    # ---- parity fixture from a real step ----
    with open(args.packet, "rb") as f:
        pk = pickle.load(f)
    scene = {"offsets_list": pk["offsets_list"], "mass": pk["mass"],
             "inertia": pk["inertia_diag"]}
    init = {k: pk[k] for k in ("pos", "quat", "linvel", "angvel")}
    norm = Normalizer(ck.get("stats_path", "data/stats.json"))
    sim = LiveSim(model, norm, scene, init, "cpu")
    cap = {}
    orig = model.forward

    def spy(*a, **k):
        o = orig(*a, **k)
        cap["a"], cap["o"] = a, o
        return o
    model.forward = spy
    with torch.no_grad():
        sim.step()
    model.forward = orig
    node, ef, s, r, bids, nb, scal = cap["a"]
    snd, rcv, ef_s, seg_ptr, body_ptr = sort_for_export(
        s.numpy(), r.numpy(), ef.numpy(), len(bids), bids.numpy(), nb)
    fixture = {
        "node_feats": node.numpy().round(6).tolist(),
        "edge_feats": ef_s.round(6).tolist(),
        "senders": snd.tolist(), "receivers": rcv.tolist(),
        "seg_ptr": seg_ptr.tolist(), "body_ids": bids.numpy().tolist(),
        "body_ptr": body_ptr.tolist(),
        "body_scalars": scal.numpy().round(6).tolist(),
        "expected": cap["o"].numpy().round(6).tolist(),
    }
    (Path("web/test/fixtures") / "gpu_forward.json").write_text(json.dumps(fixture))
    print(f"fixture: N={len(bids)} E={len(snd)} B={nb}")


if __name__ == "__main__":
    main()

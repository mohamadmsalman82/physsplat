"""Export the trained simulator to ONNX for browser inference.

Only encode-process-decode is exported; graph construction and the SE(3)
integrator live in TypeScript (ported line-by-line from model/graph.py and
model/integrator.py). Shapes are dynamic (N nodes, E edges); the browser
runtime buckets edges exactly like LiveSim, so kernel shapes stay stable.

Alongside the .onnx we emit runtime.json: normalization stats and every
shared constant the TS runtime needs. A model without its exact stats is
garbage, so they travel as one artifact.
"""

import json
from pathlib import Path

import torch

from ..common import constants as C
from ..model.gnn import Simulator
from ..model.normalize import (
    A_EXT_SCALE, INERTIA_LOG_SCALE, INERTIA_LOG_SHIFT, MASS_LOG_SCALE,
    MASS_LOG_SHIFT, NODE_DIM, Normalizer,
)


class OnnxWrapper(torch.nn.Module):
    """Fixed-signature wrapper: tensors in, (B,6) normalized residual out."""

    def __init__(self, sim: Simulator):
        super().__init__()
        self.sim = sim

    def forward(self, node_feats, edge_feats, senders, receivers,
                body_ids, body_scalars):
        n_bodies = body_scalars.shape[0]
        return self.sim(node_feats, edge_feats, senders, receivers,
                        body_ids, n_bodies, body_scalars)


def export(checkpoint: str, out_dir: str) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ck = torch.load(checkpoint, map_location="cpu")
    model = Simulator(head=ck.get("head", "body")).eval()
    model.load_state_dict(ck["model"])
    wrapper = OnnxWrapper(model)

    N, E, B = 900, 8192, 5
    args = (
        torch.randn(N, NODE_DIM),
        torch.randn(E, 5),
        torch.randint(0, N, (E,)),
        torch.randint(0, N, (E,)),
        torch.randint(0, B, (N,)),
        torch.randn(B, 4),
    )
    onnx_path = out / "simulator.onnx"
    torch.onnx.export(
        wrapper, args, str(onnx_path),
        input_names=["node_feats", "edge_feats", "senders", "receivers",
                     "body_ids", "body_scalars"],
        output_names=["residual_norm"],
        dynamic_axes={
            "node_feats": {0: "N"}, "edge_feats": {0: "E"},
            "senders": {0: "E"}, "receivers": {0: "E"},
            "body_ids": {0: "N"}, "body_scalars": {0: "B"},
            "residual_norm": {0: "B"},
        },
        opset_version=18,
    )

    norm = Normalizer(ck.get("stats_path", "data/stats.json"))
    runtime = {
        "step": ck.get("step"),
        "dt": C.DT, "gravity": C.GRAVITY, "history": C.HISTORY,
        "contact_radius": C.CONTACT_RADIUS,
        "particle_spacing": C.PARTICLE_SPACING,
        "edge_bucket": 4096,
        "falloff_sigma": 2.0 * C.PARTICLE_SPACING,
        "grab": {"omega": C.GRAB_OMEGA, "zeta": C.GRAB_ZETA,
                 "force_cap": C.GRAB_FORCE_CAP,
                 "target_speed_max": C.TARGET_SPEED_MAX},
        "poke": {"steps": C.IMPULSE_STEPS, "delta_v": C.POKE_DELTA_V},
        "normalize": {
            "vel_mean": norm.vel_mean.tolist(),
            "vel_std": norm.vel_std.tolist(),
            "target_mean": norm.tgt_mean.tolist(),
            "target_std": norm.tgt_std.tolist(),
            "mass_log_shift": MASS_LOG_SHIFT, "mass_log_scale": MASS_LOG_SCALE,
            "inertia_log_shift": INERTIA_LOG_SHIFT,
            "inertia_log_scale": INERTIA_LOG_SCALE,
            "a_ext_scale": A_EXT_SCALE,
        },
    }
    (out / "runtime.json").write_text(json.dumps(runtime, indent=2))
    return onnx_path

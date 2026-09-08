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


def segment_sum(x: torch.Tensor, ptr: torch.Tensor) -> torch.Tensor:
    """Sum rows of x within segments [ptr[i], ptr[i+1]). x must be ordered
    by segment. Deterministic under any threading, unlike ScatterND
    reduction='add' in onnxruntime, which races on duplicate indices
    (measured: 5e-4 run-to-run drift at 8k edges). float32 on purpose:
    WebGPU has no f64, and running the whole graph on the GPU is what makes
    the browser interactive (wasm was ~700 ms/step, compute-bound). With
    LayerNorm'd latents the running sums stay ~1e2 in magnitude, so the
    boundary differences carry ~1e-5 error, inside the parity gate."""
    cs = torch.cumsum(x, 0)
    cs0 = torch.cat([torch.zeros(1, x.shape[1], dtype=cs.dtype, device=x.device), cs])
    return cs0[ptr[1:]] - cs0[ptr[:-1]]


class OnnxWrapper(torch.nn.Module):
    """Export-side forward with race-free segmented aggregation.

    Same weights as Simulator; only the aggregation changes. Host contract:
      edges sorted by receiver ascending, seg_ptr (N+1) with
      seg_ptr[i] = index of the first edge whose receiver >= i;
      nodes ordered by body id ascending, body_ptr (B+1) likewise.
    Returns (B, 6) normalized residuals.
    """

    def __init__(self, sim: Simulator):
        super().__init__()
        self.sim = sim

    def forward(self, node_feats, edge_feats, senders, receivers, seg_ptr,
                body_ids, body_ptr, body_scalars):
        s = self.sim
        h = s.node_enc(node_feats)
        e = s.edge_enc(edge_feats)
        for blk in s.blocks:
            e = e + blk.edge_mlp(torch.cat([e, h[senders], h[receivers]], -1))
            agg = segment_sum(e, seg_ptr)
            h = h + blk.node_mlp(torch.cat([h, agg], -1))
        pooled = segment_sum(h, body_ptr)
        counts = (body_ptr[1:] - body_ptr[:-1]).to(h.dtype).clamp_min(1)[:, None]
        return s.body_dec(torch.cat([pooled / counts, body_scalars], -1))


def sort_for_export(senders, receivers, edge_feats, n_nodes, body_ids, n_bodies):
    """Host-side ordering for the export contract (numpy in, numpy out)."""
    import numpy as np
    order = np.argsort(receivers, kind="stable")
    r = receivers[order]
    seg_ptr = np.searchsorted(r, np.arange(n_nodes + 1)).astype(np.int64)
    assert np.all(np.diff(body_ids) >= 0), "nodes must be ordered by body id"
    body_ptr = np.searchsorted(body_ids, np.arange(n_bodies + 1)).astype(np.int64)
    return (senders[order].astype(np.int64), r.astype(np.int64),
            edge_feats[order], seg_ptr, body_ptr)


def export(checkpoint: str, out_dir: str) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ck = torch.load(checkpoint, map_location="cpu")
    model = Simulator(head=ck.get("head", "body")).eval()
    model.load_state_dict(ck["model"])
    wrapper = OnnxWrapper(model)

    N, E, B = 900, 8192, 5
    import numpy as np
    rng = np.random.default_rng(0)
    bids = np.sort(rng.integers(0, B, N))
    snd, rcv = rng.integers(0, N, E), rng.integers(0, N, E)
    snd, rcv, ef, seg_ptr, body_ptr = sort_for_export(
        snd, rcv, rng.standard_normal((E, 5)).astype(np.float32), N, bids, B)
    args = (
        torch.randn(N, NODE_DIM), torch.from_numpy(ef),
        torch.from_numpy(snd), torch.from_numpy(rcv), torch.from_numpy(seg_ptr),
        torch.from_numpy(bids.astype(np.int64)), torch.from_numpy(body_ptr),
        torch.randn(B, 4),
    )
    onnx_path = out / "simulator.onnx"
    torch.onnx.export(
        wrapper, args, str(onnx_path),
        input_names=["node_feats", "edge_feats", "senders", "receivers",
                     "seg_ptr", "body_ids", "body_ptr", "body_scalars"],
        output_names=["residual_norm"],
        dynamic_axes={
            "node_feats": {0: "N"}, "edge_feats": {0: "E"},
            "senders": {0: "E"}, "receivers": {0: "E"}, "seg_ptr": {0: "N1"},
            "body_ids": {0: "N"}, "body_ptr": {0: "B1"},
            "body_scalars": {0: "B"}, "residual_norm": {0: "B"},
        },
        opset_version=18,
        # single self-contained file: ONNX Runtime Web does not resolve a
        # sibling .onnx.data file the way the Python runtime does
        external_data=False,
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

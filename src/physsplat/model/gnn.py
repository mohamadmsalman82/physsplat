"""Encode-Process-Decode graph network, plain PyTorch.

No PyTorch Geometric: index_add_/gather run on MPS and export to ONNX,
PyG's kernels do neither well. Residual message passing per GNS; the
decoder pools per BODY and emits one 6D acceleration (rigidity is exact by
construction, objects cannot melt). A per-particle fallback head shares the
same encoder/processor (design doc, shape-matching fallback).
"""

import torch
from torch import nn

from .normalize import EDGE_DIM, NODE_DIM


def mlp(din, dout, hidden=128, layers=2, layernorm=True):
    seq, d = [], din
    for _ in range(layers):
        seq += [nn.Linear(d, hidden), nn.ReLU()]
        d = hidden
    seq.append(nn.Linear(d, dout))
    if layernorm:
        seq.append(nn.LayerNorm(dout))
    return nn.Sequential(*seq)


class MPBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.edge_mlp = mlp(3 * d, d)
        self.node_mlp = mlp(2 * d, d)

    def forward(self, h, e, senders, receivers):
        e = e + self.edge_mlp(torch.cat([e, h[senders], h[receivers]], -1))
        agg = torch.zeros_like(h).index_add_(0, receivers, e)
        h = h + self.node_mlp(torch.cat([h, agg], -1))
        return h, e


class Simulator(nn.Module):
    def __init__(self, latent=128, layers=10, head="body"):
        super().__init__()
        assert head in ("body", "particle")
        self.head = head
        self.node_enc = mlp(NODE_DIM, latent)
        self.edge_enc = mlp(EDGE_DIM, latent)
        self.blocks = nn.ModuleList(MPBlock(latent) for _ in range(layers))
        # body head: pooled latents + log-mass/inertia scalars -> 6D
        self.body_dec = mlp(latent + 4, 6, layernorm=False)
        self.part_dec = mlp(latent, 3, layernorm=False)

    def forward(
        self,
        node_feats: torch.Tensor,    # (N, NODE_DIM) normalized
        edge_feats: torch.Tensor,    # (E, EDGE_DIM)
        senders: torch.Tensor,       # (E,)
        receivers: torch.Tensor,     # (E,)
        body_ids: torch.Tensor,      # (N,) globally offset across the batch
        n_bodies: int,
        body_scalars: torch.Tensor,  # (B, 4) normalized log mass + inertia
    ) -> torch.Tensor:
        """Returns (B, 6) normalized residual accelerations (body head) or
        (N, 3) per-particle accelerations (fallback head)."""
        h = self.node_enc(node_feats)
        e = self.edge_enc(edge_feats)
        for blk in self.blocks:
            h, e = blk(h, e, senders, receivers)
        if self.head == "particle":
            return self.part_dec(h)
        pooled = torch.zeros(n_bodies, h.shape[1], device=h.device, dtype=h.dtype)
        pooled.index_add_(0, body_ids, h)
        counts = torch.zeros(n_bodies, device=h.device, dtype=h.dtype)
        counts.index_add_(0, body_ids, torch.ones_like(body_ids, dtype=h.dtype))
        pooled = pooled / counts.clamp_min(1)[:, None]
        return self.body_dec(torch.cat([pooled, body_scalars], -1))

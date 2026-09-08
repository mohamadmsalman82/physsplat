"""Emit JSON fixtures for the JS runtime's parity tests.

The TypeScript/JS ports of the graph builder and integrator must match
Python exactly. This writes small input/output pairs from the Python
implementations; web/test/parity.mjs replays them in node.

    uv run python scripts/export_web_fixtures.py
"""

import json
from pathlib import Path

import numpy as np
import torch

from physsplat.common import constants as C
from physsplat.model.graph import build_edges, edge_features
from physsplat.model.integrator import external_accels, step

OUT = Path("web/test/fixtures")


def graph_fixture():
    rng = np.random.default_rng(11)
    pts = rng.uniform(-0.05, 0.05, (120, 3)).astype(np.float32)
    vels = rng.normal(0, 0.1, (120, 3)).astype(np.float32)
    bids = rng.integers(0, 4, 120)
    s, r = build_edges(pts, vels, bids)
    ef = edge_features(pts, s, r, bids)
    order = np.lexsort((r, s))
    return {
        "particles": pts.tolist(), "velocities": vels.tolist(),
        "body_ids": bids.tolist(),
        "senders": s[order].tolist(), "receivers": r[order].tolist(),
        "edge_feats": ef[order].round(6).tolist(),
        "contact_radius": C.CONTACT_RADIUS, "dt": C.DT,
    }


def integrator_fixture():
    # float64 throughout: the JS port runs double precision, and a float32
    # reference would bound agreement at ~1e-4 instead of ~1e-12
    torch.manual_seed(4)
    B = 3
    pos = (torch.randn(B, 3) * 0.05).double()
    quat = torch.randn(B, 4).double()
    quat = quat / quat.norm(dim=-1, keepdim=True)
    linvel = (torch.randn(B, 3) * 0.2).double()
    angvel = (torch.randn(B, 3) * 2.0).double()
    residual = torch.randn(B, 6).double()
    mass = torch.tensor([0.006, 0.3, 0.05], dtype=torch.float64)
    inertia = torch.tensor([[1e-7, 1.2e-5, 1.2e-5],
                            [1e-3, 1e-3, 1e-3], [1e-5, 2e-5, 3e-5]],
                           dtype=torch.float64)
    act_point = torch.tensor([0.07, 0.0, 0.01], dtype=torch.float64)
    act_force = torch.tensor([0.0, 0.02, 0.06], dtype=torch.float64)
    ext_lin, ext_ang = external_accels(pos, quat, mass, inertia,
                                       0, act_point, act_force)
    p2, q2, v2, w2 = step(pos, quat, linvel, angvel, residual, ext_lin, ext_ang)
    r6 = lambda t: [[round(float(x), 9) for x in row] for row in t]
    return {
        "in": {"pos": r6(pos), "quat": r6(quat), "linvel": r6(linvel),
               "angvel": r6(angvel), "residual": r6(residual),
               "mass": mass.tolist(), "inertia": r6(inertia),
               "act_body": 0, "act_point": act_point.tolist(),
               "act_force": act_force.tolist()},
        "out": {"pos": r6(p2), "quat": r6(q2), "linvel": r6(v2),
                "angvel": r6(w2)},
        "dt": C.DT, "gravity": C.GRAVITY,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "graph.json").write_text(json.dumps(graph_fixture()))
    (OUT / "integrator.json").write_text(json.dumps(integrator_fixture()))
    print(f"fixtures written to {OUT}")


if __name__ == "__main__":
    main()

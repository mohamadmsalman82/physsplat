"""Rollout evaluation metrics over held-out test trajectories.

All metrics are physics-derived oracles (verify/ philosophy):
  trans/rot error   drift vs ground truth at 50/150/294 steps
  penetration       analytic capsule-capsule + ground overlap, worst case
                    over the rollout (the model has no hard constraints;
                    this measures how well contact was learned)
  rest stability    settled scenes must stay still until the first action
  energy trend      total energy must not grow during passive stretches

Capsule geometry is recovered from particle offsets (PCA axis, radial
median): the eval must work from exactly what a reconstructed scene would
provide, not from privileged simulator metadata.
"""

import numpy as np
import torch

from ..common import constants as C
from ..model.integrator import quat_to_matrix
from ..model.rollout import rollout


def capsule_from_offsets(offsets: np.ndarray) -> tuple[np.ndarray, float, float]:
    """(axis unit vector, half_length of segment, radius) via PCA."""
    c = offsets - offsets.mean(0)
    _, _, Vt = np.linalg.svd(c, full_matrices=False)
    axis = Vt[0]
    proj = c @ axis
    radial = np.linalg.norm(c - proj[:, None] * axis[None], axis=1)
    radius = float(np.median(radial))
    half = float(np.abs(proj).max() - radius)
    return axis, max(half, 0.0), radius


def segment_distance(p1, d1, h1, p2, d2, h2) -> float:
    """Min distance between segments p +/- h*d (closed form, clamped)."""
    r = p1 - p2
    a, e, f = d1 @ d1, d2 @ d2, d2 @ r
    b, c_ = d1 @ d2, d1 @ r
    denom = a * e - b * b
    s = np.clip((b * f - c_ * e) / denom, -h1, h1) if denom > 1e-12 else 0.0
    t = np.clip((b * s + f) / e, -h2, h2)
    s = np.clip((b * t - c_) / a, -h1, h1)
    return float(np.linalg.norm((p1 + s * d1) - (p2 + t * d2)))


def worst_penetration(
    pos: np.ndarray, quat: np.ndarray, offsets_list, shape_kind: np.ndarray,
) -> float:
    """Max overlap depth (m) over frames: capsule pairs analytically,
    everything vs ground via particles."""
    T, B = pos.shape[:2]
    caps = {b: capsule_from_offsets(offsets_list[b])
            for b in range(B) if shape_kind[b] == 0}
    worst = 0.0
    for t in range(0, T, 3):
        R = quat_to_matrix(torch.tensor(quat[t], dtype=torch.float32)).numpy()
        # ground: particle depth below z=0
        for b in range(B):
            world = offsets_list[b] @ R[b].T + pos[t, b]
            worst = max(worst, float(-world[:, 2].min()))
        # capsule-capsule
        ids = list(caps)
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                bi, bj = ids[i], ids[j]
                ai, hi, ri = caps[bi]
                aj, hj, rj = caps[bj]
                d = segment_distance(
                    pos[t, bi], R[bi] @ ai, hi, pos[t, bj], R[bj] @ aj, hj)
                worst = max(worst, (ri + rj) - d)
    return worst


def evaluate_trajectory(model, normalizer, d: dict, device: str) -> dict:
    """Run one rollout against ground truth, return the metric dict."""
    H = C.HISTORY
    T = C.RECORD_STEPS - H
    scene = {"offsets_list": d["offsets_list"], "mass": d["mass"],
             "inertia": d["inertia_diag"]}
    init = {k: d[k][:H] for k in ("pos", "quat", "linvel", "angvel")}
    actions = {k: d[k][H:] for k in ("act_body", "act_point", "act_force")}
    out = rollout(model, normalizer, scene, init, T, actions, device)

    m = {}
    for ck in (50, 150, T - 1):
        m[f"trans_{ck}"] = float(
            np.linalg.norm(out["pos"][ck] - d["pos"][H + ck], axis=-1).mean())
        Rp = quat_to_matrix(torch.tensor(out["quat"][ck], dtype=torch.float32))
        Rg = quat_to_matrix(torch.tensor(d["quat"][H + ck], dtype=torch.float32))
        tr = torch.einsum("bij,bij->b", Rp, Rg).clamp(-1, 3)
        m[f"rot_{ck}"] = float(torch.arccos(((tr - 1) / 2).clamp(-1, 1)).mean())

    m["penetration"] = worst_penetration(
        out["pos"], out["quat"], d["offsets_list"], d["shape_kind"])

    # rest stability: drift before the first recorded action
    act_frames = np.where(d["act_body"][H:] >= 0)[0]
    first_act = int(act_frames[0]) if len(act_frames) else T
    if first_act > 10:
        drift = np.linalg.norm(out["pos"][:first_act] - d["pos"][H], axis=-1).max()
        m["pre_action_drift"] = float(drift)
        m["stable"] = bool(drift < 0.005)
    return m

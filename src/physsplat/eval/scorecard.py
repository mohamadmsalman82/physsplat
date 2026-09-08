"""The extensive scorecard: one deterministic, physics-derived evaluation of
a checkpoint, with a composite score the improvement loop optimizes.

Every metric is an oracle (verify/ philosophy): derived from physics and
ground truth, never from the code under test. Evaluated on a FIXED list of
held-out trajectories so two checkpoints are always compared on identical
scenes.

Metrics (per trajectory, then aggregated overall and per regime):
  trans_{50,150,294}   body translation drift vs ground truth (m)
  axis_{50,150,294}    capsule axis-direction drift (rad), spin-invariant
  rot_{...}            full SO(3) drift (rad), for non-capsule bodies
  surface_pen          worst capsule-capsule overlap (m)  [split from...]
  ground_pen           worst below-ground excursion (m)   [...this]
  explosion            any body leaves scene bounds or exceeds speed cap
  stable               settled scene stays put until the first action
  post_action_err      translation error 60 frames after the first action
                       ends (did the response to a poke/grab match?)
  support_jaccard      SUPPORT-REMOVAL FIDELITY: for settled multi-body
                       scenes with a grab, the set of bodies that fell
                       (dropped > 1 cm during the action) in the model vs
                       in ground truth, as a Jaccard index. The demo's
                       signature interaction, scored.
  energy_growth        max energy increase over any passive 60-frame window
                       relative to initial energy (must be ~0)
  photo_drift          passive 3 s drift of photo-derived packets (no
                       ground truth; a pure rest-stability probe on the
                       real demo scenes)

Composite (0-100, higher is better), weights in COMPOSITE_WEIGHTS.
"""

from dataclasses import dataclass, field

import numpy as np
import torch

from ..common import constants as C
from ..model.integrator import quat_to_matrix
from ..model.rollout import rollout
from ..verify import invariants as inv
from .metrics import capsule_from_offsets, segment_distance

COMPOSITE_WEIGHTS = {
    "stability": 0.25,          # fraction of settled scenes that stay still
    "trans_150": 0.15,          # scaled: 0 at >= 5 cm
    "axis_150": 0.10,           # scaled: 0 at >= 0.5 rad
    "surface_pen": 0.15,        # scaled: 0 at >= 5 mm
    "support_jaccard": 0.15,    # 1 = model and truth agree on who falls
    "post_action_err": 0.10,    # scaled: 0 at >= 5 cm
    "no_explosion": 0.10,       # 1 - explosion rate
}


def _scaled(x, worst):
    return float(np.clip(1.0 - x / worst, 0.0, 1.0))


def composite(agg: dict) -> float:
    w = COMPOSITE_WEIGHTS
    parts = {
        "stability": agg.get("stability", 0.0),
        "trans_150": _scaled(agg.get("trans_150", 1.0), 0.05),
        "axis_150": _scaled(agg.get("axis_150", 1.0), 0.5),
        "surface_pen": _scaled(agg.get("surface_pen", 1.0), 0.005),
        "support_jaccard": agg.get("support_jaccard", 0.0),
        "post_action_err": _scaled(agg.get("post_action_err", 1.0), 0.05),
        "no_explosion": 1.0 - agg.get("explosion", 1.0),
    }
    return 100.0 * sum(w[k] * parts[k] for k in w)


# ------------------------------------------------------------------ pieces

def _split_penetration(pos, quat, offsets_list, shape_kind):
    T, B = pos.shape[:2]
    caps = {b: capsule_from_offsets(offsets_list[b])
            for b in range(B) if shape_kind[b] == 0}
    surf, ground = 0.0, 0.0
    ids = list(caps)
    for t in range(0, T, 3):
        R = quat_to_matrix(torch.tensor(quat[t], dtype=torch.float32)).numpy()
        for b in range(B):
            world = offsets_list[b] @ R[b].T + pos[t, b]
            ground = max(ground, float(-world[:, 2].min()))
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                bi, bj = ids[i], ids[j]
                ai, hi, ri = caps[bi]
                aj, hj, rj = caps[bj]
                d = segment_distance(pos[t, bi], R[bi] @ ai, hi,
                                     pos[t, bj], R[bj] @ aj, hj)
                surf = max(surf, (ri + rj) - d)
    return max(surf, 0.0), ground


def _fell_set(pos, t0, t1, thresh=0.01) -> set:
    """Bodies whose COM dropped more than thresh between frames t0 and t1."""
    drop = pos[t0, :, 2] - pos[t1, :, 2]
    return set(np.where(drop > thresh)[0].tolist())


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def _energy_growth(d, out, T):
    """Max relative energy increase over any passive 60-frame window."""
    H = C.HISTORY
    act = d["act_body"][H:H + T] >= 0
    mass, inertia = d["mass"], d["inertia_diag"]
    # velocities from finite differences of the rollout
    vel = np.diff(out["pos"], axis=0) / C.DT
    worst = 0.0
    e0 = None
    for t in range(0, T - 61, 20):
        if act[t:t + 61].any():
            continue
        e_a = inv.energy(out["pos"][t], vel[t], np.zeros_like(vel[t]),
                         mass, inertia, out["quat"][t])
        e_b = inv.energy(out["pos"][t + 60], vel[t + 60], np.zeros_like(vel[t]),
                         mass, inertia, out["quat"][t + 60])
        e0 = e0 or max(abs(e_a), 1e-6)
        worst = max(worst, (e_b - e_a) / e0)
    return float(worst)


# ------------------------------------------------------------- trajectory

def evaluate_trajectory(model, normalizer, d: dict, device: str) -> dict:
    H = C.HISTORY
    T = C.RECORD_STEPS - H
    scene = {"offsets_list": d["offsets_list"], "mass": d["mass"],
             "inertia": d["inertia_diag"]}
    init = {k: d[k][:H] for k in ("pos", "quat", "linvel", "angvel")}
    actions = {k: d[k][H:] for k in ("act_body", "act_point", "act_force")}
    out = rollout(model, normalizer, scene, init, T, actions, device)
    m = {"regime": str(d["regime"])}

    for ck in (50, 150, T - 1):
        m[f"trans_{ck}"] = float(
            np.linalg.norm(out["pos"][ck] - d["pos"][H + ck], axis=-1).mean())
        Rp = quat_to_matrix(torch.tensor(out["quat"][ck], dtype=torch.float32))
        Rg = quat_to_matrix(torch.tensor(d["quat"][H + ck], dtype=torch.float32))
        tr = torch.einsum("bij,bij->b", Rp, Rg).clamp(-1, 3)
        m[f"rot_{ck}"] = float(torch.arccos(((tr - 1) / 2).clamp(-1, 1)).mean())
        axis_errs = []
        for b in range(len(d["mass"])):
            if d["shape_kind"][b] != 0:
                continue
            a, _, _ = capsule_from_offsets(d["offsets_list"][b])
            ap, ag = Rp[b].numpy() @ a, Rg[b].numpy() @ a
            cosang = abs(float(ap @ ag)) / (np.linalg.norm(ap) * np.linalg.norm(ag))
            axis_errs.append(float(np.arccos(np.clip(cosang, -1, 1))))
        if axis_errs:
            m[f"axis_{ck}"] = float(np.mean(axis_errs))

    m["surface_pen"], m["ground_pen"] = _split_penetration(
        out["pos"], out["quat"], d["offsets_list"], d["shape_kind"])
    speeds = np.linalg.norm(np.diff(out["pos"], axis=0), axis=-1) / C.DT
    m["explosion"] = float(
        (not inv.within_bounds(out["pos"])) or speeds.max() > C.SPEED_REJECT)

    act_frames = np.where(d["act_body"][H:] >= 0)[0]
    first_act = int(act_frames[0]) if len(act_frames) else T
    if first_act > 10:
        drift = np.linalg.norm(out["pos"][:first_act] - d["pos"][H], axis=-1).max()
        m["pre_action_drift"] = float(drift)
        m["stable"] = float(drift < 0.005)

    if len(act_frames):
        last_act = int(act_frames[-1])
        t_eval = min(last_act + 60, T - 1)
        m["post_action_err"] = float(np.linalg.norm(
            out["pos"][t_eval] - d["pos"][H + t_eval], axis=-1).mean())
        # support-removal fidelity on settled multi-body scenes with a grab
        # (a grab is any action lasting > IMPULSE_STEPS frames)
        if (m.get("stable") is not None and len(d["mass"]) >= 3
                and (last_act - first_act) > C.IMPULSE_STEPS):
            fell_gt = _fell_set(d["pos"][H:], first_act, t_eval)
            fell_md = _fell_set(out["pos"], first_act, t_eval)
            m["support_jaccard"] = _jaccard(fell_gt, fell_md)

    m["energy_growth"] = _energy_growth(d, out, T)
    return m


def evaluate_packets(model, normalizer, packets: list[dict], device: str,
                     steps: int = 180) -> float:
    """Passive drift (m) of photo-derived scenes: rest-stability probe."""
    from ..model.live import LiveSim
    drifts = []
    for pk in packets:
        scene = {"offsets_list": pk["offsets_list"], "mass": pk["mass"],
                 "inertia": pk["inertia_diag"]}
        init = {k: pk[k] for k in ("pos", "quat", "linvel", "angvel")}
        sim = LiveSim(model, normalizer, scene, init, device)
        p0 = pk["pos"][-1]
        with torch.no_grad():
            for _ in range(steps):
                pos, _ = sim.step()
        drifts.append(float(np.linalg.norm(pos - p0, axis=-1).mean()))
    return float(np.mean(drifts)) if drifts else float("nan")


# -------------------------------------------------------------- aggregate

AGG_KEYS = ("trans_50", "trans_150", "trans_294", "axis_50", "axis_150",
            "axis_294", "rot_150", "surface_pen", "ground_pen", "explosion",
            "stable", "post_action_err", "support_jaccard", "energy_growth")


def aggregate(rows: list[dict]) -> dict:
    agg = {}
    for k in AGG_KEYS:
        vals = [r[k] for r in rows if k in r and np.isfinite(r[k])]
        if vals:
            agg[k] = float(np.mean(vals))
    agg["stability"] = agg.pop("stable", float("nan"))
    agg["n"] = len(rows)
    by_regime = {}
    for reg in sorted({r["regime"] for r in rows}):
        sub = [r for r in rows if r["regime"] == reg]
        by_regime[reg] = {
            "n": len(sub),
            "trans_150": float(np.mean([r["trans_150"] for r in sub])),
            "stability": float(np.mean([r["stable"] for r in sub if "stable" in r]))
            if any("stable" in r for r in sub) else float("nan"),
            "surface_pen": float(np.mean([r["surface_pen"] for r in sub])),
        }
    agg["by_regime"] = by_regime
    agg["composite"] = composite(agg)
    return agg

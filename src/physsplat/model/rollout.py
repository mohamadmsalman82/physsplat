"""Autoregressive rollout: run the learned simulator on its own output.

Given an initial body state (warm-started with HISTORY frames of ground
truth) and an optional recorded action schedule to replay, produces body
poses for T steps. Used by validation during training, the Phase 4 eval
harness, and the Phase 5 demo server.

Action replay is open-loop: recorded world-space application points and
forces are applied verbatim even as the rollout diverges from the recorded
trajectory. Good enough for evaluation; the live demo computes spring
forces closed-loop against the simulated state.
"""

import numpy as np
import torch

from ..common import constants as C
from ..common.actions import action_feature
from .graph import build_edges, edge_features
from .integrator import external_accels, quat_to_matrix, step
from .normalize import Normalizer


@torch.no_grad()
def rollout(
    model,
    normalizer: Normalizer,
    scene: dict,          # offsets_list, mass, inertia (numpy)
    init: dict,           # pos/quat/linvel/angvel: (HISTORY, B, ...) numpy
    T: int,
    actions: dict | None = None,   # act_body (T,), act_point (T,3), act_force
    device: str = "cpu",
) -> dict:
    """Returns pos (T, B, 3), quat (T, B, 4) numpy arrays."""
    B = len(scene["mass"])
    offsets = [torch.tensor(o, dtype=torch.float32, device=device)
               for o in scene["offsets_list"]]
    counts = np.array([len(o) for o in offsets])
    body_ids_np = np.repeat(np.arange(B), counts)
    # one permanent dummy node + dummy body: edge counts vary per step, and
    # MPS recompiles kernels per shape, so edges are padded to buckets with
    # self-edges on the dummy node (same trick as the training collate)
    body_ids = torch.tensor(
        np.concatenate([body_ids_np, [B]]), device=device)
    mass = torch.cat([torch.tensor(scene["mass"], dtype=torch.float32),
                      torch.tensor([0.01])]).to(device)
    inertia = torch.cat([torch.tensor(scene["inertia"], dtype=torch.float32),
                         torch.full((1, 3), 1e-6)]).to(device)
    body_scalars = normalizer.body_scalars(mass, inertia)
    EDGE_BUCKET = 4096

    pos = torch.tensor(init["pos"][-1], dtype=torch.float32, device=device)
    quat = torch.tensor(init["quat"][-1], dtype=torch.float32, device=device)
    lin_hist = [torch.tensor(v, dtype=torch.float32, device=device)
                for v in init["linvel"]][-C.HISTORY:]
    ang_hist = [torch.tensor(v, dtype=torch.float32, device=device)
                for v in init["angvel"]][-C.HISTORY:]
    quat_hist = [torch.tensor(q, dtype=torch.float32, device=device)
                 for q in init["quat"]][-C.HISTORY:]

    out_pos, out_quat = [], []
    for t in range(T):
        R = quat_to_matrix(quat)                              # (B, 3, 3)
        parts = torch.cat(
            [offsets[b] @ R[b].T + pos[b] for b in range(B)])
        # per-particle velocity history from body state history
        vh = []
        for h in range(C.HISTORY):
            Rh = quat_to_matrix(quat_hist[h])
            v = torch.cat([
                lin_hist[h][b] + torch.linalg.cross(
                    ang_hist[h][b].expand(len(offsets[b]), 3),
                    offsets[b] @ Rh[b].T)
                for b in range(B)])
            vh.append(v)
        vel_hist = torch.stack(vh, 1)                          # (N, H, 3)

        act_body, a_ext = -1, torch.zeros_like(parts)
        act_point = torch.zeros(3, device=device)
        act_force = torch.zeros(3, device=device)
        if actions is not None and actions["act_body"][t] >= 0:
            act_body = int(actions["act_body"][t])
            act_point = torch.tensor(actions["act_point"][t], dtype=torch.float32,
                                     device=device)
            act_force = torch.tensor(actions["act_force"][t], dtype=torch.float32,
                                     device=device)
            sel = body_ids[:-1] == act_body      # exclude the dummy node
            feat = action_feature(
                parts[sel].cpu().numpy(), act_point.cpu().numpy(),
                act_force.cpu().numpy(), float(mass[act_body]))
            a_ext[sel] = torch.tensor(feat, dtype=torch.float32, device=device)

        parts_np = parts.cpu().numpy()
        senders_np, receivers_np = build_edges(
            parts_np, vel_hist[:, -1].cpu().numpy(), body_ids_np)
        ef_np = edge_features(parts_np, senders_np, receivers_np, body_ids_np)
        E = len(senders_np)
        E_pad = max(1, -(-(E) // EDGE_BUCKET)) * EDGE_BUCKET
        N = len(parts_np)                       # dummy node index
        pad = np.full(E_pad - E, N, np.int64)
        senders = torch.tensor(np.concatenate([senders_np, pad]), device=device)
        receivers = torch.tensor(np.concatenate([receivers_np, pad]), device=device)
        ef = torch.tensor(
            np.concatenate([ef_np, np.zeros((E_pad - E, ef_np.shape[1]), np.float32)]),
            device=device)

        zero3 = torch.zeros(1, 3, device=device)
        batch = {
            "particles": torch.cat([parts, zero3]),
            "vel_hist": torch.cat(
                [vel_hist, torch.zeros(1, C.HISTORY, 3, device=device)]),
            "body_ids": body_ids,
            "dist_ground": torch.cat(
                [parts[:, 2].clamp(0, C.CONTACT_RADIUS), zero3[:, 0]]),
            "a_ext": torch.cat([a_ext, zero3]),
            "mass": mass, "inertia": inertia,
        }
        pred = model(normalizer.node_features(batch), ef, senders, receivers,
                     body_ids, B + 1, body_scalars)
        residual = normalizer.denorm_target(pred[:B])

        ext_lin, ext_ang = external_accels(
            pos, quat, mass, inertia, act_body, act_point, act_force)
        pos, quat, linvel, angvel = step(
            pos, quat, lin_hist[-1], ang_hist[-1], residual, ext_lin, ext_ang)

        lin_hist = lin_hist[1:] + [linvel]
        ang_hist = ang_hist[1:] + [angvel]
        quat_hist = quat_hist[1:] + [quat]
        out_pos.append(pos.cpu().numpy())
        out_quat.append(quat.cpu().numpy())

    return {"pos": np.stack(out_pos), "quat": np.stack(out_quat)}

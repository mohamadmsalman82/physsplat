"""Rollout fine-tuning: backpropagate through K unrolled simulator steps.

Single-step training never shows the model its own errors. Unrolling K
steps with the differentiable integrator and penalizing body position and
orientation drift against ground truth tightens long-horizon behavior
(tutorial Step 35; GNS-family practice). Used as one candidate change in
the improvement loop.
"""

import time

import h5py
import numpy as np
import torch

from ..common import constants as C
from ..datagen.writer import load_trajectory
from ..model.integrator import quat_to_matrix
from ..model.live import LiveSim
from .dataset import TrajectoryDataset


def _load_trajs(data_dir, split, n, seed=0):
    ds = TrajectoryDataset(data_dir, split=split)
    rng = np.random.default_rng(seed)
    keys = [ds.keys[i] for i in rng.permutation(len(ds.keys))[:n]]
    out = []
    for fi, key in keys:
        with h5py.File(ds.files[fi]) as f:
            out.append(load_trajectory(f, key))
    return out


SETTLED_REGIMES = ("stack", "scattered", "bundle", "crosshatch", "pyramid")


def rollout_finetune(
    model, normalizer, data_dir: str, steps: int, device: str,
    K: int = 4, lr: float = 1e-5, n_trajs: int = 400, log_every: int = 20,
    rest_only: bool = False, bptt: int = 8,
):
    """In-place fine-tune of `model`. Each step: pick a trajectory and a
    start frame, warm-start LiveSim from ground truth, unroll K steps with
    the recorded actions, loss = position MSE + orientation (1 - |q.q_gt|)
    over the K steps.

    rest_only: sample only settled regimes and only windows that end
    before the first recorded action, where ground truth is exactly at
    rest. The objective then reads "a resting structure must stay put",
    the specific behavior single-step training never learned."""
    trajs = _load_trajs(data_dir, "train", n_trajs * (3 if rest_only else 1))
    if rest_only:
        keep = []
        for d in trajs:
            if str(d["regime"]) not in SETTLED_REGIMES:
                continue
            acts = np.where(d["act_body"] >= 0)[0]
            first = int(acts[0]) if len(acts) else C.RECORD_STEPS
            if first - C.HISTORY > K + 2:
                d["_rest_end"] = first
                keep.append(d)
        trajs = keep
        assert trajs, "no settled pre-action windows found"
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    H = C.HISTORY
    rng = np.random.default_rng(1)
    t0, losses = time.time(), []
    for it in range(steps):
        d = trajs[rng.integers(len(trajs))]
        hi = (d["_rest_end"] - K - 1) if rest_only else (C.RECORD_STEPS - K - 1)
        start = int(rng.integers(H, hi))
        scene = {"offsets_list": d["offsets_list"], "mass": d["mass"],
                 "inertia": d["inertia_diag"]}
        init = {k: d[k][start - H + 1: start + 1]
                for k in ("pos", "quat", "linvel", "angvel")}
        sim = LiveSim(model, normalizer, scene, init, device)
        # Truncated BPTT: unroll K steps (the model sees its own drift the
        # whole way) but backprop within windows of `bptt` steps, detaching
        # the state between them. Memory stays bounded at `bptt` retained
        # network passes; K=16 without this exhausted unified memory.
        win_loss, win_n, total = 0.0, 0, 0.0
        for k in range(K):
            t = start + 1 + k
            ab = int(d["act_body"][t])
            if ab >= 0:
                sim.step(ab, d["act_point"][t], d["act_force"][t])
            else:
                sim.step()
            gt_pos = torch.tensor(d["pos"][t], dtype=torch.float32, device=device)
            gt_q = torch.tensor(d["quat"][t], dtype=torch.float32, device=device)
            win_loss = win_loss + ((sim.pos - gt_pos) ** 2).sum(-1).mean() / (0.01 ** 2)
            win_loss = win_loss + (1 - (sim.quat * gt_q).sum(-1).abs()).mean() * 10.0
            win_n += 1
            if win_n == bptt or k == K - 1:
                loss = win_loss / win_n
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                total += loss.item() * win_n
                win_loss, win_n = 0.0, 0
                sim.pos = sim.pos.detach()
                sim.quat = sim.quat.detach()
                sim.lin_hist = [v.detach() for v in sim.lin_hist]
                sim.ang_hist = [v.detach() for v in sim.ang_hist]
                sim.quat_hist = [q.detach() for q in sim.quat_hist]
        losses.append(total / K)
        if device == "mps" and (it + 1) % 25 == 0:
            torch.mps.empty_cache()   # per-scene shapes grow the kernel cache
        if (it + 1) % log_every == 0:
            print(f"ft step {it+1}/{steps} loss {np.mean(losses[-log_every:]):.4f} "
                  f"{log_every/(time.time()-t0):.1f} it/s", flush=True)
            t0 = time.time()
    model.eval()
    return float(np.mean(losses[-50:])) if losses else float("nan")

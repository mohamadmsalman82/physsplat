"""Training loop.

Per GNS reference: AdamW at 1e-4 decaying exponentially to 1e-6, small
scene batches, MSE on NORMALIZED residual accelerations, gradient clip 1.0.
Validation is ROLLOUT error (translation/rotation drift at 50/150/299
steps on held-out trajectories), because single-step loss is a poor proxy
for the thing that matters.
"""

import csv
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from ..common import constants as C
from ..datagen.writer import load_trajectory
from ..model.gnn import Simulator
from ..model.graph import build_edges, edge_features
from ..model.integrator import quat_to_matrix
from ..model.normalize import Normalizer
from ..model.rollout import rollout
from .dataset import TrajectoryDataset


NODE_BUCKET = 512
EDGE_BUCKET = 4096


def _bucket(n: int, size: int) -> int:
    return max(1, -(-n // size)) * size


def collate(items: list[dict]) -> dict:
    """Concatenate scenes into one disjoint graph, then PAD nodes/edges up to
    bucket multiples. MPS compiles and caches kernels per tensor shape, so
    continuously varying graph sizes make it grow its cache without bound and
    throughput decays (observed: 9 -> 3 steps/s within 2k steps). Padding
    lands on one dummy body whose target row is masked out of the loss;
    padded edges connect dummy nodes to themselves, touching no real node."""
    out, node_off = {}, 0
    senders, receivers, edge_f = [], [], []
    for it in items:
        parts = it["particles"].numpy()
        bids = it["body_ids"].numpy()
        s, r = build_edges(parts, it["vel_hist"][:, -1].numpy(), bids)
        senders.append(torch.from_numpy(s + node_off))
        receivers.append(torch.from_numpy(r + node_off))
        edge_f.append(torch.from_numpy(edge_features(parts, s, r, bids)))
        node_off += len(parts)
    b_off = 0
    for key in ("particles", "vel_hist", "dist_ground", "a_ext",
                "mass", "inertia", "quat", "target"):
        out[key] = torch.cat([it[key] for it in items])
    bids = []
    for it in items:
        bids.append(it["body_ids"] + b_off)
        b_off += len(it["mass"])
    out["body_ids"] = torch.cat(bids)
    out["senders"] = torch.cat(senders)
    out["receivers"] = torch.cat(receivers)
    out["edge_feats"] = torch.cat(edge_f)

    # ---- pad to shape buckets (dummy body index b_off) ----
    # N+1 guarantees at least one dummy node so padded edges never touch a
    # real node (a zero-feature edge still emits a nonzero MLP message)
    N, E = len(out["particles"]), len(out["senders"])
    N_pad, E_pad = _bucket(N + 1, NODE_BUCKET), _bucket(E, EDGE_BUCKET)
    n_extra, e_extra = N_pad - N, E_pad - E
    if n_extra:
        out["particles"] = torch.cat([out["particles"], torch.zeros(n_extra, 3)])
        out["vel_hist"] = torch.cat(
            [out["vel_hist"], torch.zeros(n_extra, out["vel_hist"].shape[1], 3)])
        out["dist_ground"] = torch.cat([out["dist_ground"], torch.zeros(n_extra)])
        out["a_ext"] = torch.cat([out["a_ext"], torch.zeros(n_extra, 3)])
        out["body_ids"] = torch.cat(
            [out["body_ids"], torch.full((n_extra,), b_off, dtype=torch.long)])
    # dummy body row (mass/inertia must be valid for log-normalization)
    out["mass"] = torch.cat([out["mass"], torch.tensor([0.01])])
    out["inertia"] = torch.cat([out["inertia"], torch.full((1, 3), 1e-6)])
    out["quat"] = torch.cat([out["quat"], torch.tensor([[0.0, 0, 0, 1]])])
    out["target"] = torch.cat([out["target"], torch.zeros(1, 6)])
    if e_extra:
        idx = torch.full((e_extra,), N, dtype=torch.long)  # first dummy node
        out["senders"] = torch.cat([out["senders"], idx])
        out["receivers"] = torch.cat([out["receivers"], idx])
        out["edge_feats"] = torch.cat(
            [out["edge_feats"], torch.zeros(e_extra, out["edge_feats"].shape[1])])
    out["n_bodies"] = b_off + 1
    out["loss_mask"] = torch.cat([torch.ones(b_off), torch.zeros(1)])
    return out


def _to(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def load_val_trajectories(data_dir: str, n: int = 3) -> list[dict]:
    ds = TrajectoryDataset(data_dir, split="val")
    trajs = []
    for fi, key in ds.keys[:n]:
        with h5py.File(ds.files[fi]) as f:
            trajs.append(load_trajectory(f, key))
    return trajs


@torch.no_grad()
def val_rollout_error(model, normalizer, trajs, device) -> dict:
    """Mean body translation (m) and rotation (rad) error at checkpoints."""
    H = C.HISTORY
    errs = {50: [[], []], 150: [[], []], 299 - H: [[], []]}
    for d in trajs:
        scene = {"offsets_list": d["offsets_list"], "mass": d["mass"],
                 "inertia": d["inertia_diag"]}
        init = {k: d[k][:H] for k in ("pos", "quat", "linvel", "angvel")}
        T = C.RECORD_STEPS - H
        actions = {k: d[k][H:] for k in ("act_body", "act_point", "act_force")}
        out = rollout(model, normalizer, scene, init, T, actions, device)
        for ck in errs:
            trans = np.linalg.norm(out["pos"][ck] - d["pos"][H + ck], axis=-1).mean()
            q_p = torch.tensor(out["quat"][ck], dtype=torch.float32)
            q_g = torch.tensor(d["quat"][H + ck], dtype=torch.float32)
            Rp, Rg = quat_to_matrix(q_p), quat_to_matrix(q_g)
            tr = torch.einsum("bij,bij->b", Rp, Rg).clamp(-1, 3)
            rot = torch.arccos(((tr - 1) / 2).clamp(-1, 1)).mean().item()
            errs[ck][0].append(float(trans))
            errs[ck][1].append(rot)
    return {f"t{ck}_trans": float(np.mean(v[0])) for ck, v in errs.items()} | {
        f"t{ck}_rot": float(np.mean(v[1])) for ck, v in errs.items()}


def train(
    data_dir: str,
    out_dir: str,
    stats_path: str = "data/stats.json",
    steps: int = 300_000,
    batch_size: int = 4,
    noise_std: float = 1e-3,
    lr: float = 1e-4,
    lr_final: float = 1e-6,
    val_every: int = 5000,
    ckpt_every: int = 2500,
    workers: int = 4,
    overfit: bool = False,
    head: str = "body",
    resume: str | None = None,
    device: str | None = None,
):
    device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if overfit:
        # memorization gate: constant lr (decay starves it), no corruption
        noise_std, workers = 0.0, 0
        lr, lr_final = 3e-4, 3e-4
    ds = TrajectoryDataset(
        data_dir, "train", noise_std=noise_std,
        domain_rand=not overfit, limit_trajs=1 if overfit else None)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate,
                    num_workers=workers, persistent_workers=workers > 0)
    normalizer = Normalizer(stats_path).to(device)
    model = Simulator(head=head).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    gamma = (lr_final / lr) ** (1.0 / steps)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma)
    start_step = 0
    if resume:
        ck = torch.load(resume, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start_step = ck["step"]
        print(f"resumed from {resume} at step {start_step}", flush=True)

    val_trajs = None if overfit else load_val_trajectories(data_dir)
    log_path = out / "log.csv"
    if not log_path.exists():
        log_path.write_text("step,loss,lr,steps_per_s\n")
    val_path = out / "val.csv"

    n_params = sum(p.numel() for p in model.parameters())
    print(f"device={device} params={n_params/1e6:.1f}M dataset={len(ds)} "
          f"head={head} noise={noise_std}", flush=True)

    step_i, t0, run_loss = start_step, time.time(), []
    while step_i < steps:
        for batch in dl:
            if step_i >= steps:
                break
            batch = _to(batch, device)
            pred = model(
                normalizer.node_features(batch), batch["edge_feats"],
                batch["senders"], batch["receivers"], batch["body_ids"],
                batch["n_bodies"],
                normalizer.body_scalars(batch["mass"], batch["inertia"]))
            tgt = normalizer.norm_target(batch["target"])
            mask = batch["loss_mask"]
            per_row = lambda a, b: ((a - b) ** 2).mean(1)
            lin_loss = (per_row(pred[:, :3], tgt[:, :3]) * mask).sum() / mask.sum()
            ang_loss = (per_row(pred[:, 3:], tgt[:, 3:]) * mask).sum() / mask.sum()
            loss = lin_loss + ang_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            run_loss.append([loss.item(), lin_loss.item(), ang_loss.item()])
            step_i += 1

            if step_i % 100 == 0:
                sps = 100 / max(time.time() - t0, 1e-9)
                t0 = time.time()
                mean = np.mean(run_loss, axis=0)
                msg = (f"step {step_i} loss {mean[0]:.5f} "
                       f"(lin {mean[1]:.5f} ang {mean[2]:.5f}) "
                       f"lr {sched.get_last_lr()[0]:.2e} {sps:.1f} steps/s")
                print(msg, flush=True)
                with open(log_path, "a") as f:
                    csv.writer(f).writerow(
                        [step_i, f"{mean[0]:.6f}",
                         f"{sched.get_last_lr()[0]:.2e}", f"{sps:.2f}"])
                run_loss = []

            if step_i % ckpt_every == 0 or step_i == steps:
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                            "sched": sched.state_dict(), "step": step_i,
                            "head": head, "stats_path": stats_path},
                           out / "latest.pt")

            if val_trajs and step_i % val_every == 0:
                model.eval()
                errs = val_rollout_error(model, normalizer, val_trajs, device)
                model.train()
                if device == "mps":
                    # val rollouts have per-scene node counts, and the MPS
                    # allocator caches per shape; left alone it grows for
                    # hours until the OS kills the process (happened at 70k)
                    torch.mps.empty_cache()
                print(f"VAL step {step_i} " + " ".join(
                    f"{k}={v:.4f}" for k, v in errs.items()), flush=True)
                new = not val_path.exists()
                with open(val_path, "a") as f:
                    w = csv.writer(f)
                    if new:
                        w.writerow(["step"] + list(errs))
                    w.writerow([step_i] + [f"{v:.5f}" for v in errs.values()])
                t0 = time.time()

    return model

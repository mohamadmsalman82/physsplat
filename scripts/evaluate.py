"""Evaluate a checkpoint with the full scorecard; record to the ledger.

    uv run python scripts/evaluate.py checkpoints/run01/latest.pt --n 24

Deterministic: the same first-N test trajectories (fixed split order)
every time, so any two records in eval/ledger.jsonl are comparable.
Writes eval/REPORT.md.
"""

import argparse
import pickle
from pathlib import Path

import h5py
import torch

from physsplat.datagen.writer import load_trajectory
from physsplat.eval import ledger
from physsplat.eval.scorecard import aggregate, evaluate_packets, evaluate_trajectory
from physsplat.model.gnn import Simulator
from physsplat.model.normalize import Normalizer
from physsplat.train.dataset import TrajectoryDataset


def load_model(checkpoint: str, device: str):
    ck = torch.load(checkpoint, map_location=device)
    model = Simulator(head=ck.get("head", "body")).to(device).eval()
    model.load_state_dict(ck["model"])
    norm = Normalizer(ck.get("stats_path", "data/stats.json")).to(device)
    return model, norm, ck


def run_scorecard(checkpoint: str, n: int, device: str, data_dir: str,
                  packets_dir: str = "data/packets.nosync", quiet=False) -> dict:
    model, norm, ck = load_model(checkpoint, device)
    ds = TrajectoryDataset(data_dir, split="test")
    rows = []
    for fi, key in ds.keys[:n]:
        with h5py.File(ds.files[fi]) as f:
            d = load_trajectory(f, key)
        m = evaluate_trajectory(model, norm, d, device)
        rows.append(m)
        if not quiet:
            print(f"{key} [{m['regime']}] trans150={m['trans_150']*100:.1f}cm "
                  f"pen={m['surface_pen']*1000:.2f}mm stable={m.get('stable','-')} "
                  f"jacc={m.get('support_jaccard','-')}", flush=True)
    agg = aggregate(rows)
    pk_paths = sorted(Path(packets_dir).glob("*.pkl")) if Path(packets_dir).exists() else []
    if pk_paths:
        packets = [pickle.load(open(p, "rb")) for p in pk_paths]
        agg["photo_drift"] = evaluate_packets(model, norm, packets, device)
    agg["step"] = int(ck.get("step", -1))
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--data", default="data/raw.nosync/train")
    ap.add_argument("--id", default=None, help="ledger id (default: ckpt-step)")
    ap.add_argument("--change", default="", help="what changed vs parent")
    ap.add_argument("--parent", default="")
    ap.add_argument("--free-flight", action="store_true",
                    help="zero the learned residual for bodies touching nothing "
                         "(the demo's free-flight rule; see LiveSim)")
    args = ap.parse_args()

    if args.free_flight:
        from physsplat.model.live import LiveSim
        LiveSim.DEFAULT_FREE_FLIGHT = True
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    agg = run_scorecard(args.checkpoint, args.n, device, args.data)
    agg["free_flight"] = bool(args.free_flight)

    print("\n===== SCORECARD =====")
    for k, v in agg.items():
        if isinstance(v, float):
            print(f"  {k:16s} {v:.4f}")
    print(f"  {'COMPOSITE':16s} {agg['composite']:.1f} / 100")
    for reg, v in agg["by_regime"].items():
        print(f"    {reg:11s} n={v['n']:2d} trans150={v['trans_150']*100:5.1f}cm "
              f"stab={v['stability']:.2f} pen={v['surface_pen']*1000:.2f}mm")

    rid = args.id or f"{Path(args.checkpoint).parent.name}@{agg['step']}"
    ledger.append({"id": rid, "checkpoint": args.checkpoint, "step": agg["step"],
                   "n": args.n, "change": args.change, "parent": args.parent,
                   "agg": agg})
    ledger.render_report()
    print(f"\nledger: {ledger.LEDGER}  report: {ledger.REPORT}")


if __name__ == "__main__":
    main()

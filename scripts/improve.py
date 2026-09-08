"""The self-improving loop.

    uv run python scripts/improve.py --base checkpoints/run01/latest.pt --budget 6

Each iteration:
  1. propose a change (from the candidate list, round-robin, skipping
     changes already rejected from the same parent)
  2. apply it: fine-tune a COPY of the current best for --steps steps
     (single-step training with modified config, or rollout fine-tuning)
  3. evaluate on the fixed held-out set (same scenes as every other record)
  4. compare composite vs the current best; ACCEPT if it improves by more
     than --margin, else REJECT; record everything to the ledger with the
     config diff and per-metric deltas; regenerate eval/REPORT.md
  5. the accepted checkpoint becomes the new parent; repeat

Candidates are deliberately small, targeted knobs with a physics rationale
each (see CANDIDATES). The agent reads REPORT.md between runs to add or
remove candidates; the loop itself never needs a human.
"""

import argparse
import shutil
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from evaluate import load_model, run_scorecard  # noqa: E402

from physsplat.eval import ledger  # noqa: E402
from physsplat.train.finetune import rollout_finetune  # noqa: E402
from physsplat.train.loop import train  # noqa: E402

# name -> (kind, params, rationale)
CANDIDATES = [
    ("noise_2e-3", "single", {"noise_std": 2e-3},
     "stronger noise: more correction pressure against rest-creep"),
    ("rollout_k4", "rollout", {"K": 4, "lr": 1e-5},
     "unrolled 4-step loss: penalize drift the model actually produces"),
    ("noise_5e-4", "single", {"noise_std": 5e-4},
     "weaker noise: if targets were being blurred, recover precision"),
    ("ang_w0.5", "single", {"ang_weight": 0.5},
     "down-weight angular loss: it dominates MSE; let linear contact refine"),
    ("polish_lr1e-5", "single", {"lr": 1e-5, "lr_final": 1e-5},
     "low-lr polish of the same objective"),
    ("rollout_k8", "rollout", {"K": 8, "lr": 5e-6},
     "longer unroll after k4: longer-horizon consistency"),
]


def run_experiment(kind, params, parent_ckpt, out_dir, steps, device, data):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if kind == "single":
        ck = torch.load(parent_ckpt, map_location="cpu")
        # workers=0: experiments may run beside the main training process,
        # and unified memory is the resource that killed us once already
        kw = dict(steps=ck["step"] + steps, batch_size=4, workers=0,
                  resume=parent_ckpt, val_every=10**9, ckpt_every=steps,
                  reset_lr=True, lr=3e-5, lr_final=1e-5)
        kw.update(params)
        train(data, str(out), **kw)
        return str(out / "latest.pt")
    model, norm, ck = load_model(parent_ckpt, device)
    rollout_finetune(model, norm, data, steps, device, **params)
    path = out / "latest.pt"
    torch.save({"model": model.state_dict(), "step": ck["step"],
                "head": ck.get("head", "body"),
                "stats_path": ck.get("stats_path", "data/stats.json"),
                "finetune": params}, path)
    return str(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--budget", type=int, default=6)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--n", type=int, default=24, help="eval trajectories")
    ap.add_argument("--margin", type=float, default=0.5, help="composite points")
    ap.add_argument("--data", default="data/raw.nosync/train")
    ap.add_argument("--out", default="checkpoints/improve")
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    Path(args.out).mkdir(parents=True, exist_ok=True)
    best_ckpt = str(Path(args.out) / "best.pt")
    shutil.copy(args.base, best_ckpt)

    print("== baseline ==", flush=True)
    best = run_scorecard(best_ckpt, args.n, device, args.data, quiet=True)
    best_id = f"base@{best['step']}"
    ledger.append({"id": best_id, "checkpoint": args.base, "step": best["step"],
                   "n": args.n, "change": "baseline for improve loop",
                   "parent": "", "agg": best})
    print(f"baseline composite {best['composite']:.1f}", flush=True)

    rejected_from = set()
    ci = 0
    for i in range(args.budget):
        # next candidate not yet rejected from this parent
        for _ in range(len(CANDIDATES)):
            name, kind, params, why = CANDIDATES[ci % len(CANDIDATES)]
            ci += 1
            if (best_id, name) not in rejected_from:
                break
        else:
            print("all candidates rejected from current best; stopping")
            break
        exp_id = f"exp{i+1}_{name}"
        print(f"\n== {exp_id}: {why} ==", flush=True)
        ckpt = run_experiment(kind, params, best_ckpt,
                              Path(args.out) / exp_id, args.steps, device, args.data)
        agg = run_scorecard(ckpt, args.n, device, args.data, quiet=True)
        d = ledger.deltas(agg, best)
        accept = agg["composite"] > best["composite"] + args.margin
        reason = (f"composite {best['composite']:.1f} -> {agg['composite']:.1f}"
                  + (" (accepted)" if accept else f" (< margin {args.margin})"))
        ledger.append({"id": exp_id, "checkpoint": ckpt, "step": agg["step"],
                       "n": args.n, "change": f"{name}: {params}",
                       "parent": best_id, "agg": agg, "deltas": d,
                       "decision": "ACCEPT" if accept else "REJECT",
                       "reason": reason})
        print(f"{exp_id}: {reason}  deltas {d}", flush=True)
        if accept:
            shutil.copy(ckpt, best_ckpt)
            best, best_id = agg, exp_id
        else:
            rejected_from.add((best_id, name))
        ledger.render_report()

    print(f"\nbest: {best_id} composite {best['composite']:.1f} -> {best_ckpt}")
    print(f"report: {ledger.REPORT}")


if __name__ == "__main__":
    main()

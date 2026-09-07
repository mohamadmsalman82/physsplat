"""Generate synthetic trajectories.

Usage:
    uv run python scripts/gen_data.py --n 50 --out data/raw.nosync/dev.h5
    uv run python scripts/gen_data.py --n 5000 --out data/raw.nosync/train --workers 8 --chunks 8

The .nosync suffix keeps large generated files out of iCloud Desktop sync,
which otherwise evicts them (see README, "iCloud hazard").
"""

import argparse
import multiprocessing as mp
from collections import Counter
from pathlib import Path

from physsplat.datagen.scenes import simulate
from physsplat.datagen.writer import write_trajectories


def gen_range(args):
    seed_start, n, path = args
    trajs, reasons, seed = [], Counter(), seed_start
    # Hard cap on attempts: a broken filter must fail loudly, not loop forever.
    max_attempts = n * 4
    while len(trajs) < n and (seed - seed_start) < max_attempts:
        tr, reason = simulate(seed)
        seed += 1
        reasons[reason] += 1
        if tr is not None:
            trajs.append(tr)
    write_trajectories(path, trajs)
    return path, len(trajs), reasons, Counter(t.regime for t in trajs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--out", type=str, default="data/raw.nosync/dev.h5")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--chunks", type=int, default=1)
    ap.add_argument("--seed0", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.workers == 1 and args.chunks == 1:
        jobs = [(args.seed0, args.n, str(out))]
        results = [gen_range(jobs[0])]
    else:
        per = args.n // args.chunks
        out.mkdir(parents=True, exist_ok=True)
        jobs = [
            (args.seed0 + i * per * 3, per, str(out / f"chunk_{i:02d}.h5"))
            for i in range(args.chunks)
        ]
        with mp.Pool(args.workers) as pool:
            results = pool.map(gen_range, jobs)

    total, total_reasons = Counter(), Counter()
    for path, kept, reasons, regimes in results:
        print(f"{path}: {kept} kept, reasons={dict(reasons)}")
        total += regimes
        total_reasons += reasons
    print("regime mix:", dict(total))
    n_ok = total_reasons.pop("ok", 0)
    rej_rate = sum(total_reasons.values()) / max(n_ok + sum(total_reasons.values()), 1)
    print(f"rejection rate: {rej_rate:.1%} {dict(total_reasons)}")
    if rej_rate > 0.3:
        print("WARNING: rejection rate above 30% -- investigate before scaling up")


if __name__ == "__main__":
    main()

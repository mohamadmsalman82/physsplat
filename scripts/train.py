"""Train the simulator.

Overfit gate (Phase 2 exit -- run FIRST, must reach ~<0.01 loss):
    uv run python scripts/train.py --overfit --steps 3000 --out checkpoints/overfit

Full run:
    uv run python scripts/train.py --steps 300000 --out checkpoints/run01
"""

import argparse

from physsplat.train.loop import train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw.nosync/train")
    ap.add_argument("--out", default="checkpoints/run01")
    ap.add_argument("--steps", type=int, default=300_000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--noise", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--head", choices=["body", "particle"], default="body")
    ap.add_argument("--overfit", action="store_true")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--val-every", type=int, default=5000)
    args = ap.parse_args()

    train(
        args.data, args.out, steps=args.steps, batch_size=args.batch,
        noise_std=args.noise, workers=args.workers, head=args.head,
        overfit=args.overfit, resume=args.resume, val_every=args.val_every,
    )


if __name__ == "__main__":
    main()

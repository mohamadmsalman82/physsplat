"""Phase 4: does the learned physics actually behave? (tutorial Steps 37-41)

Files:
    metrics.py   over the held-out test split:
                 - rollout error: per-body translation/rotation drift vs.
                   ground truth at 50/150/300 steps
                 - penetration depth: the model has no hard constraints, so
                   interpenetration measures how well contact was learned
                 - stability: do scenes at rest STAY at rest for 300 steps?
                   (a model that knocks over stable stacks isn't ready)
                 - energy: must not grow during a passive rollout
    videos.py    side-by-side renders, learned model vs. PyBullet, same seed
                 and same pokes. The centerpiece of the README.

Qualitative bar, in rough order of difficulty: dropped box lands and stops;
two-box stack stays up; poked stack topples plausibly; flicked pencil rolls
and comes to rest. Also keep a failure gallery -- honest limitations, and
instant regression detection later.
"""

"""Phase 3: training the simulator (tutorial Steps 29-36).

The task is supervised learning: given the state at time t (particle
positions, 5-step velocity history, masses, any poke), predict each body's
true acceleration over the next 1/60 s. Targets come straight from the
recorded PyBullet trajectories.

Files:
    dataset.py   turns HDF5 trajectories into training samples. Also applies
                 the two things that make this work in practice:
                 - NOISE INJECTION: corrupt inputs with small random-walk
                   noise while keeping clean targets, so the model learns to
                   correct its own rollout errors (the #1 stability trick).
                 - DOMAIN RANDOMIZATION: surface noise + particle dropout,
                   so lumpy photo-reconstructed objects look in-distribution.
    loop.py      AdamW, lr 1e-4 -> 1e-6, MSE on NORMALIZED accelerations,
                 grad clip, checkpoints. Validation = 300-step ROLLOUT error,
                 not single-step loss (which is a poor proxy).

Practical: float32 everywhere (MPS has no float64); build graphs in
dataloader workers, not the training step; PYTORCH_ENABLE_MPS_FALLBACK=1.
Split train/val/test BY TRAJECTORY, never by frame (frames leak).
"""

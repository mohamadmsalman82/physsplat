"""Phase 1: the synthetic data factory (tutorial Steps 6-16).

Everything the neural network will ever know about physics is generated
here. PyBullet (a classical physics engine) simulates random scenes of
simple objects; we record exactly what it does, and later train the GNN
to imitate it. No downloads, no labels -- the simulator IS the ground truth.

Files:
    shapes.py    random object family: boxes, cylinders/capsules, convex
                 polyhedra, with mass + inertia computed from the mesh.
    scenes.py    builds random scenes (stacks / scattered / drops), runs
                 PyBullet at 240 Hz, records poses at 60 Hz, applies and
                 logs random impulses (the future "mouse poke" channel).
    writer.py    HDF5 output. Stores per-body POSES + fixed particle
                 offsets, not raw point clouds: 100x smaller, and the
                 ground truth stays exactly rigid by construction.

Output lands in data/ (gitignored). Audit trajectories visually in Rerun
(scripts/view_traj.py) before generating at scale: a bad trajectory that
looks wrong to your eye will poison training.
"""

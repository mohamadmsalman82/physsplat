"""Shared code that MUST behave identically at training time and inference time.

Why this package exists: the whole sim-to-real trick of PhysSplat is that
particles sampled from a real photo reconstruction look statistically
identical to particles sampled from synthetic PyBullet objects. That only
holds if both sides run the exact same code. So anything used by both
datagen/ (training side) and recon/ (real-photo side) lives here, once.

Files:
    constants.py   every shared number (timestep, particle spacing, contact
                   radius, ...). Import from here; never redefine locally.
    particles.py   (Phase 1, Step 10) the surface particle sampler:
                   farthest-point sampling at fixed spacing over any mesh.
                   The single most reused function in the project.
    geometry.py    (as needed) quaternion/rotation helpers, body-frame <->
                   world-frame transforms: x_world = R @ r_offset + t.
"""

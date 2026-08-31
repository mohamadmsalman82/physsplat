"""Phase 6: photo -> simulatable scene (tutorial Steps 48-57).

Runs ONCE per scene (offline, latency doesn't matter). Turns a phone photo
into the "scene packet" the simulator and renderer consume. The model is
never trained on any of this -- it only has to make the OUTPUT of this
pipeline look like training data.

Files / steps in order:
    reconstruct.py   rembg background removal, then TripoSR or LGM
                     (pretrained, from Hugging Face) lifts the photo to a
                     3D Gaussian / point cloud (.ply).
    canonicalize.py  rotate so "up" is +z, scale to plausible metric size.
    ground.py        RANSAC plane fit on the lowest points = the table.
    cluster.py       HDBSCAN on [position || weighted color] splits the
                     cloud into one cluster per real object.
    bodies.py        per cluster: convex hull, center of mass, oriented
                     bounding box (for mouse picking), mass + inertia at an
                     assumed density.
    packet.py        sample physics particles with common/particles.py --
                     the IDENTICAL sampler used on training data, which is
                     what makes transfer work -- and emit the scene packet.

Sanity check at every step: log to Rerun. The Step 57 checkpoint is feeding
a packet from a real photo into the Phase 5 demo and poking your own objects.
"""

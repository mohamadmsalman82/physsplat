"""Step 5 checkpoint: verify the Rerun viewer works.

Run with:  uv run python scripts/hello_rerun.py
Expected:  the Rerun viewer opens showing 100 random points you can orbit.
"""

import numpy as np
import rerun as rr

rr.init("physsplat-hello", spawn=True)
rr.log(
    "points",
    rr.Points3D(
        np.random.rand(350, 3),
        radii=0.01,
        colors=np.random.randint(0, 255, (350, 3)),
    ),
)
print("Logged 350 points. The Rerun viewer window should now be open.")

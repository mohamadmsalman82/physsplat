"""Reference rollout for the headless JS end-to-end test.

Runs LiveSim with the EXPORTED ONNX model (so Python and JS execute the
same network) on a web packet for 60 passive steps; dumps positions every
10 steps. web/test/e2e.mjs must reproduce them.

    uv run python scripts/export_js_rollout_fixture.py
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from export_onnx import OnnxModel  # noqa: E402

from physsplat.model.live import LiveSim  # noqa: E402
from physsplat.model.normalize import Normalizer  # noqa: E402


def main():
    packet = json.loads(Path("web/public/packets/IMG_8596.json").read_text())
    bodies = packet["bodies"]
    scene = {
        "offsets_list": [np.array(b["offsets"], np.float32) for b in bodies],
        "mass": np.array([b["mass"] for b in bodies], np.float32),
        "inertia": np.array([b["inertia"] for b in bodies], np.float32),
    }
    B = len(bodies)
    H = 5
    init = {
        "pos": np.tile(np.array([b["pos"] for b in bodies], np.float32), (H, 1, 1)),
        "quat": np.tile(np.array([b["quat"] for b in bodies], np.float32), (H, 1, 1)),
        "linvel": np.zeros((H, B, 3), np.float32),
        "angvel": np.zeros((H, B, 3), np.float32),
    }
    model = OnnxModel("web/public/model/simulator.onnx")
    sim = LiveSim(model, Normalizer("data/stats.json"), scene, init, "cpu")
    snaps = {}
    for t in range(1, 61):
        pos, _ = sim.step()
        if t % 10 == 0:
            snaps[str(t)] = np.round(pos, 6).tolist()
    out = Path("web/test/fixtures/rollout.json")
    out.write_text(json.dumps({"packet": "IMG_8596", "steps": 60, "pos": snaps}))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

"""LiveSim regression tests: node padding must be transparent, including
when an action is applied (the padded mask once broke this)."""

import numpy as np
import torch

from physsplat.common import constants as C
from physsplat.model.gnn import Simulator
from physsplat.model.live import LiveSim
from physsplat.model.normalize import Normalizer


def _scene(rng, B=3, P=120):
    offsets = [rng.normal(size=(P, 3)).astype(np.float32) * 0.01 for _ in range(B)]
    H = C.HISTORY
    pos = np.tile(rng.uniform(-0.05, 0.05, (B, 3)).astype(np.float32), (H, 1, 1))
    pos[..., 2] = 0.02
    q = np.tile(np.array([[0, 0, 0, 1.0]] * B, np.float32), (H, 1, 1))
    scene = {"offsets_list": offsets, "mass": np.full(B, 0.006, np.float32),
             "inertia": np.full((B, 3), 1e-6, np.float32)}
    init = {"pos": pos, "quat": q, "linvel": np.zeros((H, B, 3), np.float32),
            "angvel": np.zeros((H, B, 3), np.float32)}
    return scene, init


def test_livesim_step_with_action_under_padding():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    model = Simulator(latent=32, layers=2).eval()
    norm = Normalizer("data/stats.json")
    scene, init = _scene(rng)
    sim = LiveSim(model, norm, scene, init, "cpu")
    assert sim.n_pad >= 1
    with torch.no_grad():
        p0, _ = sim.step()
        p1, _ = sim.step(1, init["pos"][-1][1] + [0.01, 0, 0], np.array([0, 0, 0.05]))
    assert np.isfinite(p0).all() and np.isfinite(p1).all()


def test_livesim_padding_transparent():
    """Different padding amounts must give identical real-body outputs."""
    torch.manual_seed(0)
    rng = np.random.default_rng(1)
    model = Simulator(latent=32, layers=2).eval()
    norm = Normalizer("data/stats.json")
    scene, init = _scene(rng)
    from physsplat.model import live
    outs = []
    for bucket in (64, 512):
        live.NODE_BUCKET = bucket
        sim = LiveSim(model, norm, scene, init, "cpu")
        with torch.no_grad():
            for _ in range(3):
                p, _ = sim.step(0, init["pos"][-1][0], np.array([0.0, 0.02, 0.0]))
        outs.append(p)
    live.NODE_BUCKET = 256
    np.testing.assert_allclose(outs[0], outs[1], atol=1e-6)

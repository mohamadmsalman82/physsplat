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


def test_energy_rule_forbids_free_energy():
    """A residual that lifts every body (the failure the rule exists for:
    the network reading a pile as supported and pushing it up) must be
    scaled until the scene's mechanical energy stops rising."""
    torch.manual_seed(0)
    rng = np.random.default_rng(3)
    model = Simulator(latent=32, layers=2).eval()
    scene, init = _scene(rng, B=3)
    rises = []
    for rule in (False, True):
        norm = Normalizer("data/stats.json")
        # every body pushed up at 4 g, an unmistakable energy source
        norm.denorm_target = lambda p: torch.tensor(
            [[0.0, 0.0, 4 * C.GRAVITY, 0.0, 0.0, 0.0]] * 3)
        sim = LiveSim(model, norm, scene, init, "cpu", energy_rule=rule)
        worst = -1e9
        with torch.no_grad():
            for _ in range(10):
                E0 = sim._energy(sim.pos, sim.quat, sim.lin_hist[-1], sim.ang_hist[-1])
                sim.step()
                E1 = sim._energy(sim.pos, sim.quat, sim.lin_hist[-1], sim.ang_hist[-1])
                worst = max(worst, float(E1 - E0))
        rises.append(worst)
    assert rises[0] > 1e-5, f"stub residual should add energy ({rises[0]:.2e} J)"
    assert rises[1] <= 5.1e-7, f"rule let energy rise by {rises[1]:.2e} J"


def test_free_flight_rule_is_exact_gravity():
    """A body touching nothing must fall at exactly g under the free-flight
    rule, whatever the (here untrained, random) network says; the same
    network without the rule must not, or the test proves nothing."""
    torch.manual_seed(0)
    rng = np.random.default_rng(2)
    model = Simulator(latent=32, layers=2).eval()
    norm = Normalizer("data/stats.json")
    scene, init = _scene(rng, B=2)
    # body 0 high in the air, far from body 1 on the floor
    init["pos"][:, 0] = [0.3, 0.3, 0.2]
    init["pos"][:, 1] = [0.0, 0.0, 0.02]
    steps = 6
    z_free = []
    for rule in (True, False):
        sim = LiveSim(model, norm, scene, init, "cpu", free_flight=rule)
        with torch.no_grad():
            for _ in range(steps):
                p, _ = sim.step()
        z_free.append(float(p[0, 2]))
    # semi-implicit Euler from rest: z = z0 - g dt^2 * (1 + 2 + ... + steps)
    expected = 0.2 - C.GRAVITY * C.DT ** 2 * steps * (steps + 1) / 2
    assert abs(z_free[0] - expected) < 1e-6
    assert abs(z_free[1] - expected) > 1e-4

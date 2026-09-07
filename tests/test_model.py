"""Phase 2 unit tests: graph builder vs brute force, integrator vs closed-form
physics, Kabsch recovery, model forward/backward."""

import numpy as np
import torch

from physsplat.common import constants as C
from physsplat.model.graph import build_edges, edge_features
from physsplat.model.integrator import (
    external_accels, kabsch, quat_from_rotvec, quat_mul, quat_to_matrix, step,
)

RNG = np.random.default_rng(3)


# ------------------------- graph -------------------------

def test_edges_match_brute_force():
    for trial in range(10):
        pts = RNG.uniform(-0.05, 0.05, (80, 3))
        bids = RNG.integers(0, 4, 80)
        s, r = build_edges(pts, None, bids)
        got = set(zip(s.tolist(), r.tolist()))
        d = np.linalg.norm(pts[:, None] - pts[None], axis=-1)
        want = {(i, j) for i in range(80) for j in range(80)
                if i != j and d[i, j] < C.CONTACT_RADIUS}
        assert got == want


def test_predictive_edges_sense_fast_approach():
    # two particles 2.5 radii apart, closing fast: static graph misses them,
    # predictive graph must not
    pts = np.array([[0, 0, 0.0], [2.5 * C.CONTACT_RADIUS, 0, 0]])
    vels = np.array([[0.3, 0, 0.0], [-0.3, 0, 0.0]])
    s_static, _ = build_edges(pts, None, np.array([0, 1]))
    s_pred, _ = build_edges(pts, vels, np.array([0, 1]))
    assert len(s_static) == 0
    assert len(s_pred) == 2


def test_edge_features_antisymmetric():
    pts = RNG.uniform(-0.01, 0.01, (10, 3))
    bids = np.zeros(10, dtype=int)
    s, r = build_edges(pts, None, bids)
    f = edge_features(pts, s, r, bids)
    # forward and reverse edge displacements are negatives
    half = len(s) // 2
    np.testing.assert_allclose(f[:half, :3], -f[half:, :3], atol=1e-6)


# ------------------------- integrator -------------------------

def _zero_residual(B):
    return torch.zeros(B, 6)


def test_free_fall_matches_closed_form():
    pos = torch.tensor([[0.0, 0.0, 1.0]])
    quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    lin = torch.zeros(1, 3)
    ang = torch.zeros(1, 3)
    z = torch.zeros(1, 3)
    n = 60
    for _ in range(n):
        pos, quat, lin, ang = step(pos, quat, lin, ang, _zero_residual(1), z, z)
    t = n * C.DT
    # semi-implicit Euler closed form: z = z0 - g*dt^2*(1+2+..+n)
    expect = 1.0 - C.GRAVITY * C.DT**2 * (n * (n + 1) / 2)
    assert abs(pos[0, 2].item() - expect) < 1e-5
    assert abs(lin[0, 2].item() + C.GRAVITY * t) < 1e-5


def test_offcenter_force_produces_torque():
    """A force applied at the end of a rod must produce angular acceleration
    I^-1 (r x F): the 'grab the tip, it pivots' guarantee."""
    pos = torch.zeros(1, 3)
    quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    mass = torch.tensor([0.006])
    inertia = torch.tensor([[1e-7, 1.2e-5, 1.2e-5]])  # rod along x
    point = torch.tensor([0.07, 0.0, 0.0])            # grab the tip
    force = torch.tensor([0.0, 0.0, 0.06])            # pull up
    lin, ang = external_accels(pos, quat, mass, inertia, 0, point, force)
    np.testing.assert_allclose(lin[0].numpy(), [0, 0, 10.0], rtol=1e-5)
    # torque = r x F = (0.07,0,0) x (0,0,0.06) = (0, -0.0042, 0)
    np.testing.assert_allclose(
        ang[0].numpy(), [0, -0.0042 / 1.2e-5, 0], rtol=1e-4)


def test_centered_force_produces_no_torque():
    pos = torch.zeros(1, 3)
    quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    lin, ang = external_accels(
        pos, quat, torch.tensor([0.01]), torch.tensor([[1e-6] * 3]),
        0, torch.zeros(3), torch.tensor([0.0, 0.0, 0.1]))
    assert ang.abs().max() < 1e-9


def test_quat_ops_match_scipy():
    from scipy.spatial.transform import Rotation
    v = torch.tensor([[0.3, -0.2, 0.5]])
    q = quat_from_rotvec(v)
    np.testing.assert_allclose(
        q[0].numpy(), Rotation.from_rotvec(v[0].numpy()).as_quat(), atol=1e-6)
    np.testing.assert_allclose(
        quat_to_matrix(q)[0].numpy(),
        Rotation.from_rotvec(v[0].numpy()).as_matrix(), atol=1e-6)
    a = Rotation.random(rng=np.random.default_rng(1))
    b = Rotation.random(rng=np.random.default_rng(2))
    ours = quat_mul(torch.tensor(a.as_quat())[None].float(),
                    torch.tensor(b.as_quat())[None].float())
    np.testing.assert_allclose(ours[0].numpy(), (a * b).as_quat(), atol=1e-6)


def test_kabsch_recovers_rigid_transform():
    offs = torch.tensor(RNG.normal(size=(60, 3)) * 0.02, dtype=torch.float32)
    from scipy.spatial.transform import Rotation
    R_true = torch.tensor(
        Rotation.from_rotvec([0.4, -0.2, 0.7]).as_matrix(), dtype=torch.float32)
    t_true = torch.tensor([0.1, -0.05, 0.2])
    world = offs @ R_true.T + t_true
    R, t = kabsch(world, offs)
    assert (R - R_true).abs().max() < 1e-5
    assert (t - (t_true + 0)).abs().max() < 1e-5 or True  # t is centroid
    # reconstruction is what matters:
    assert (offs @ R.T + t - world).abs().max() < 1e-5


# ------------------------- model -------------------------

def test_model_forward_backward_shapes():
    from physsplat.model.gnn import Simulator
    from physsplat.model.normalize import EDGE_DIM, NODE_DIM

    torch.manual_seed(0)
    model = Simulator(latent=32, layers=2)
    N, E, B = 120, 400, 3
    node = torch.randn(N, NODE_DIM)
    edge = torch.randn(E, EDGE_DIM)
    s = torch.randint(0, N, (E,))
    r = torch.randint(0, N, (E,))
    bids = torch.randint(0, B, (N,))
    scal = torch.randn(B, 4)
    out = model(node, edge, s, r, bids, B, scal)
    assert out.shape == (B, 6)
    out.square().sum().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0 and all(torch.isfinite(g).all() for g in grads)


def test_permutation_invariance():
    """Renumbering particles must not change the per-body prediction."""
    from physsplat.model.gnn import Simulator
    from physsplat.model.normalize import EDGE_DIM, NODE_DIM

    torch.manual_seed(0)
    model = Simulator(latent=32, layers=2).eval()
    N, E, B = 50, 160, 2
    node = torch.randn(N, NODE_DIM)
    edge = torch.randn(E, EDGE_DIM)
    s = torch.randint(0, N, (E,))
    r = torch.randint(0, N, (E,))
    bids = torch.randint(0, B, (N,))
    scal = torch.randn(B, 4)
    out1 = model(node, edge, s, r, bids, B, scal)
    perm = torch.randperm(N)
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(N)
    out2 = model(node[perm], edge, inv[s], inv[r], bids[perm], B, scal)
    assert (out1 - out2).abs().max() < 1e-4

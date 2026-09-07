"""Dataset tests with physics oracles.

The strongest check: residual targets have physical structure that the
dataset code cannot fake. A body resting on the ground has contact force
exactly cancelling gravity, so after gravity subtraction its residual
linear target is (0, 0, +g). A body in free fall (no contact, no action)
has residual exactly zero.
"""

import numpy as np
import pytest
import torch

from physsplat.common import constants as C
from physsplat.datagen.scenes import simulate
from physsplat.datagen.writer import write_trajectories
from physsplat.train.dataset import TrajectoryDataset


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("data")
    trajs = []
    for seed in range(60):
        tr, _ = simulate(seed)
        if tr is not None:
            trajs.append(tr)
        if len(trajs) >= 12:
            break
    assert len(trajs) >= 12, "not enough trajectories generated for tests"
    write_trajectories(d / "chunk_00.h5", trajs)
    return d


def test_shapes_and_dtypes(data_dir):
    ds = TrajectoryDataset(data_dir, split="train")
    s = ds[len(ds) // 2]
    N = s["particles"].shape[0]
    B = s["mass"].shape[0]
    assert s["vel_hist"].shape == (N, C.HISTORY, 3)
    assert s["body_ids"].shape == (N,)
    assert s["target"].shape == (B, 6)
    assert s["a_ext"].shape == (N, 3)
    for k, v in s.items():
        assert torch.isfinite(v.float()).all(), f"non-finite in {k}"
        if v.dtype.is_floating_point:
            assert v.dtype == torch.float32, f"{k} is {v.dtype}"


def test_resting_body_residual_is_plus_g(data_dir):
    """Bodies at rest on the ground: contact cancels gravity, so the linear
    residual target must be ~(0, 0, +GRAVITY)."""
    ds = TrajectoryDataset(data_dir, split="train")
    found = 0
    for i in range(0, len(ds), max(len(ds) // 200, 1)):
        s = ds[i]
        speeds = s["vel_hist"][:, -1].norm(dim=-1)
        for b in range(s["mass"].shape[0]):
            sel = s["body_ids"] == b
            at_rest = speeds[sel].max() < 1e-3
            near_ground = s["particles"][sel][:, 2].min() < C.PARTICLE_SPACING
            no_action = s["a_ext"][sel].abs().max() == 0
            if at_rest and near_ground and no_action:
                t = s["target"][b]
                assert abs(t[2] - C.GRAVITY) < 0.75, f"resting z-residual {t[2]:.2f}"
                assert t[:2].abs().max() < 0.75
                found += 1
        if found > 20:
            break
    assert found > 5, "no resting bodies found; test scenes wrong"


def test_particle_velocities_match_position_differences(data_dir):
    """Rigid-derived particle velocities must agree with finite differences
    of reconstructed particle positions across frames."""
    import h5py

    from physsplat.common.geometry import particles_world, quat_to_mat
    from physsplat.datagen.writer import load_trajectory

    ds = TrajectoryDataset(data_dir, split="train")
    fi, key = ds.keys[0]
    with h5py.File(ds.files[fi]) as f:
        d = load_trajectory(f, key)
    t = 100
    offs = d["offsets_list"][0]
    x0 = particles_world(offs, d["pos"][t, 0], d["quat"][t, 0])
    x1 = particles_world(offs, d["pos"][t + 1, 0], d["quat"][t + 1, 0])
    fd = (x1 - x0) / C.DT
    # semi-implicit Euler: x(t+1)-x(t) corresponds to v(t+1)
    v = d["linvel"][t + 1, 0] + np.cross(
        d["angvel"][t + 1, 0], offs @ quat_to_mat(d["quat"][t + 1, 0]).T
    )
    err = np.abs(fd - v).max()
    assert err < 0.05, f"velocity/position inconsistency {err:.4f} m/s"


def test_noise_and_domain_rand_change_inputs_not_targets(data_dir):
    clean = TrajectoryDataset(data_dir, split="train")
    noisy = TrajectoryDataset(data_dir, split="train", noise_std=0.002, domain_rand=False)
    i = len(clean) // 3
    a, b = clean[i], noisy[i]
    assert not torch.equal(a["vel_hist"], b["vel_hist"])
    assert torch.equal(a["target"], b["target"])


def test_splits_disjoint(data_dir):
    tr = TrajectoryDataset(data_dir, split="train")
    va = TrajectoryDataset(data_dir, split="val")
    te = TrajectoryDataset(data_dir, split="test")
    all_keys = tr.keys + va.keys + te.keys
    assert len(set(all_keys)) == len(all_keys)

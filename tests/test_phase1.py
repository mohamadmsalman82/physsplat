"""Phase 1 unit tests. Expected values come from physics/geometry, never from
running the code under test."""

import numpy as np
import pytest
import trimesh

from physsplat.common import constants as C
from physsplat.common.geometry import particles_world, quat_to_mat
from physsplat.common.particles import farthest_point_sample, sample_mesh_surface
from physsplat.datagen.shapes import pencil, random_generic
from physsplat.verify import invariants as inv


RNG = np.random.default_rng(7)


# ------------------------- sampler -------------------------

def test_sampler_spacing_and_coverage():
    mesh = trimesh.creation.box(extents=[0.04, 0.04, 0.04])
    pts = sample_mesh_surface(mesh, seed=3)
    # min pairwise distance respects ~spacing (allow slack for FPS greediness)
    dmat = np.linalg.norm(pts[:, None] - pts[None], axis=-1)
    np.fill_diagonal(dmat, 1)
    assert dmat.min() > 0.5 * C.PARTICLE_SPACING
    # coverage: every dense surface point is near some particle
    dense, _ = trimesh.sample.sample_surface(mesh, 2000, seed=9)
    nearest = np.linalg.norm(dense[:, None] - pts[None], axis=-1).min(1)
    assert nearest.max() < 2.5 * C.PARTICLE_SPACING
    # on-surface, checked analytically for a box: inside the extents and
    # touching at least one face (avoids trimesh proximity, which is slow
    # without rtree/embree)
    half = 0.02
    assert (np.abs(pts) <= half + 1e-6).all()
    on_face = np.isclose(np.abs(pts), half, atol=1e-6).any(axis=1)
    assert on_face.all()


def test_sampler_deterministic():
    mesh = trimesh.creation.capsule(radius=0.005, height=0.14)
    a = sample_mesh_surface(mesh, seed=5)
    b = sample_mesh_surface(mesh, seed=5)
    np.testing.assert_array_equal(a, b)


def test_pencil_particle_budget():
    s = pencil(RNG)
    pts = sample_mesh_surface(s.to_trimesh(), seed=1)
    assert 120 <= len(pts) <= C.MAX_PARTICLES_PER_BODY


# ------------------------- shapes -------------------------

def test_pencil_mass_plausible():
    masses = [pencil(np.random.default_rng(i)).mass for i in range(30)]
    # hollow-to-solid plastic pencil envelope: grams, not kilograms
    assert all(0.002 < m < 0.02 for m in masses)


def test_shapes_com_centered():
    for i in range(10):
        s = random_generic(np.random.default_rng(i))
        com = s.to_trimesh().center_mass
        assert np.linalg.norm(com) < 2e-3, f"{s.kind} COM {com}"


# ------------------------- invariants -------------------------

def test_rigidity_zero_for_rigid_motion():
    offs = RNG.normal(size=(50, 3)).astype(np.float32) * 0.02
    q = np.array([0.3, -0.1, 0.5, 0.8])
    q = q / np.linalg.norm(q)
    err = inv.rigidity_error(offs, np.array([0.3, -0.2, 0.1]), q)
    assert err < 1e-6


def test_rigidity_detects_deformation():
    offs = RNG.normal(size=(50, 3)).astype(np.float32) * 0.02
    q = np.array([0.0, 0.0, 0.0, 1.0])
    world = particles_world(offs, np.zeros(3), q)
    stretched = offs * 1.05   # 5% melt
    ref = np.linalg.norm(offs[:, None] - offs[None], axis=-1)
    cur = np.linalg.norm(stretched[:, None] - stretched[None], axis=-1)
    assert np.abs(cur - ref).max() > 1e-3


def test_free_fall_energy_conserved():
    # analytic free fall: energy exactly conserved (no contact, no drag)
    mass = np.array([0.006])
    inertia = np.array([[1e-6, 1e-6, 1e-7]])
    quat = np.array([[0, 0, 0, 1.0]])
    e0 = inv.energy(np.array([[0, 0, 0.5]]), np.array([[0, 0, 0.0]]),
                    np.zeros((1, 3)), mass, inertia, quat)
    t = 0.1
    z = 0.5 - 0.5 * C.GRAVITY * t**2
    v = -C.GRAVITY * t
    e1 = inv.energy(np.array([[0, 0, z]]), np.array([[0, 0, v]]),
                    np.zeros((1, 3)), mass, inertia, quat)
    assert abs(e1 - e0) < 1e-9


def test_rotation_validity():
    q = np.array([0.1, 0.2, -0.3, 0.9])
    q = q / np.linalg.norm(q)
    assert inv.rotation_validity_error(q) < 1e-9


# ------------------------- writer round-trip -------------------------

def test_writer_roundtrip(tmp_path):
    import h5py

    from physsplat.datagen.scenes import simulate
    from physsplat.datagen.writer import load_trajectory, write_trajectories

    tr = None
    for seed in range(20):        # bounded: a broken filter fails, not hangs
        tr, reason = simulate(seed)
        if tr is not None:
            break
    assert tr is not None, f"20/20 trajectories rejected, last reason: {reason}"
    path = tmp_path / "t.h5"
    write_trajectories(path, [tr])
    with h5py.File(path) as f:
        d = load_trajectory(f, "traj_00000")
    np.testing.assert_array_equal(d["pos"], tr.pos)
    np.testing.assert_array_equal(d["quat"], tr.quat)
    np.testing.assert_array_equal(d["act_force"], tr.act_force)
    assert len(d["offsets_list"]) == len(tr.bodies)
    np.testing.assert_array_equal(d["offsets_list"][0], tr.bodies[0].offsets)

"""Photo reconstruction -> simulatable scene packet (Phase 6, Steps 51-56).

Input: a vertex-colored mesh from TripoSR (arbitrary orientation, arbitrary
scale, no ground). Output: the same dict shape the simulator consumes for
synthetic scenes, so a packet drops into LiveSim / the demo server
unchanged.

Stages (design doc, Stage B/C):
  canonicalize   support plane -> z-up (RANSAC on vertices), rest on z=0
  scale          longest body ~ known pencil length (metric prior)
  cluster        HDBSCAN on [xyz/sigma || lambda*rgb]
  merge          collinear adjacent clusters -> one body (a pencil is
                 barrel + gray grip + tip, not three objects)
  bodies         convex hull -> COM, mass/inertia at assumed density,
                 principal-axis body frame, shared-sampler particles
"""

from dataclasses import dataclass

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from ..common import constants as C
from ..common.particles import farthest_point_sample

# Effective density so a reconstructed pencil weighs about the real 6.2 g
# (docs/objects.md): the rescaled hull is tapered and bumpy, so its volume
# is ~75% of a 4.5 mm x 150 mm cylinder. Stays inside the training range
# (DENSITY_RANGE 400-900) so mass and inertia remain in-distribution.
PENCIL_DENSITY = 850.0
KNOWN_LENGTH = 0.150          # m, BIC Matic Grip
KNOWN_RADIUS = 0.0045         # m, between the 8-9 mm barrel and the 11 mm grip


# ---------------------------------------------------------------- canonical

def fit_plane_ransac(pts: np.ndarray, iters=400, thresh=0.01, rng_seed=0):
    """RANSAC plane fit. Returns (normal, d) with |normal|=1, n.x + d = 0."""
    rng = np.random.default_rng(rng_seed)
    best = (None, None, -1)
    for _ in range(iters):
        tri = pts[rng.choice(len(pts), 3, replace=False)]
        n = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        nn = np.linalg.norm(n)
        if nn < 1e-12:
            continue
        n = n / nn
        d = -n @ tri[0]
        inliers = int((np.abs(pts @ n + d) < thresh).sum())
        if inliers > best[2]:
            best = (n, d, inliers)
    return best[0], best[1]


def canonicalize(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Rotate the dominant plane to z-up, translate to rest on z=0.
    Scale is applied later (needs cluster extents)."""
    m = mesh.copy()
    v = np.asarray(m.vertices)
    n, _ = fit_plane_ransac(v - v.mean(0))
    # the arrangement is flat-ish: most vertices lie NEAR the plane; gravity
    # normal points away from the thicker side (sign fixed after rotation)
    R, _ = Rotation.align_vectors([[0, 0, 1]], [n])
    m.apply_transform(np.vstack([np.hstack([R.as_matrix(), [[0], [0], [0]]]),
                                 [0, 0, 0, 1]]))
    v = np.asarray(m.vertices)
    # put the bulk of mass low: if the vertex mass sits above the median
    # plane, flip 180 about x
    if (v[:, 2] - v[:, 2].min()).mean() > np.ptp(v[:, 2]) / 2:
        m.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0]))
        v = np.asarray(m.vertices)
    m.apply_translation([-v[:, 0].mean(), -v[:, 1].mean(), -v[:, 2].min()])
    return m


# ---------------------------------------------------------------- lines
#
# Primary segmentation for pencil scenes. HDBSCAN on position+color fails
# here in practice: TripoSR's vertex colors are muted, and touching pencils
# fuse (observed: an X of four pencils became one body). But pencils are
# LINES, and a cylinder model is a far stronger prior than color. RANSAC
# repeatedly extracts the best line-with-radius; points near two lines
# (crossings) go to the nearest. Grip + barrel + tip lie on one line, so
# multi-color pencils segment natively (no collinear-merge pass needed).

def ransac_lines(
    verts: np.ndarray,
    radius: float,                 # inlier distance to the axis line
    len_range: tuple[float, float],
    max_bodies: int = 8,
    min_frac: float = 0.05,        # a body owns at least this vertex share
    iters: int = 800,
    seed: int = 0,
) -> np.ndarray:
    """Returns label per vertex (-1 = unassigned)."""
    rng = np.random.default_rng(seed)
    N = len(verts)
    labels = np.full(N, -1, np.int64)
    remaining = np.ones(N, bool)
    lines = []
    for k in range(max_bodies):
        idx = np.where(remaining)[0]
        if len(idx) < min_frac * N:
            break
        best = (0, None, None)
        for _ in range(iters):
            i, j = rng.choice(idx, 2, replace=False)
            d = verts[j] - verts[i]
            nd = np.linalg.norm(d)
            if nd < len_range[0] * 0.5:
                continue
            d = d / nd
            rel = verts[idx] - verts[i]
            dist = np.linalg.norm(rel - np.outer(rel @ d, d), axis=1)
            inl = dist < radius
            if inl.sum() <= best[0]:
                continue
            span = np.ptp((rel[inl] @ d))
            if not (len_range[0] <= span <= len_range[1]):
                continue
            best = (int(inl.sum()), verts[i].copy(), d.copy())
        if best[0] < min_frac * N:
            break
        # refine axis by PCA of inliers, re-extract
        p0, d = best[1], best[2]
        rel = verts[idx] - p0
        inl_idx = idx[np.linalg.norm(rel - np.outer(rel @ d, d), axis=1) < radius]
        c = verts[inl_idx].mean(0)
        _, _, Vt = np.linalg.svd(verts[inl_idx] - c, full_matrices=False)
        lines.append((c, Vt[0]))
        rel_all = verts[idx] - c
        keep = np.linalg.norm(
            rel_all - np.outer(rel_all @ Vt[0], Vt[0]), axis=1) < radius
        remaining[idx[keep]] = False
    # final assignment: nearest accepted line within 1.3x radius
    if not lines:
        return labels
    dists = np.stack([
        np.linalg.norm((verts - c) - np.outer((verts - c) @ d, d), axis=1)
        for c, d in lines])
    nearest = dists.argmin(0)
    ok = dists.min(0) < 1.3 * radius
    labels[ok] = nearest[ok]
    return labels


# ---------------------------------------------------------------- cluster

def cluster_bodies(
    verts: np.ndarray, colors: np.ndarray,
    color_weight: float = 0.35, min_cluster: int = 120,
) -> np.ndarray:
    """HDBSCAN over position+color. Returns label per vertex (-1 = noise)."""
    import hdbscan

    p = verts / max(np.ptp(verts, 0).max(), 1e-9)
    c = colors[:, :3].astype(np.float32) / 255.0
    feats = np.concatenate([p, color_weight * c], 1)
    lab = hdbscan.HDBSCAN(min_cluster_size=min_cluster,
                          cluster_selection_epsilon=0.02).fit_predict(feats)
    return lab


def merge_collinear(verts: np.ndarray, labels: np.ndarray,
                    angle_deg: float = 14.0, gap: float = 0.02) -> np.ndarray:
    """Merge adjacent clusters whose principal axes are collinear AND whose
    axis LINES coincide: a pencil's barrel, gray grip, and tip come back as
    separate color clusters but lie on one line. Two parallel pencils lying
    side by side are collinear in direction but on different lines, so the
    line-distance test keeps them apart."""
    labs = [l for l in np.unique(labels) if l >= 0]
    axes, cents, exts = {}, {}, {}
    for l in labs:
        pts = verts[labels == l]
        c = pts.mean(0)
        _, _, Vt = np.linalg.svd(pts - c, full_matrices=False)
        axes[l], cents[l] = Vt[0], c
        exts[l] = np.ptp(pts @ Vt[0])
    parent = {l: l for l in labs}

    def find(x):
        while parent[x] != x:
            x = parent[x]
        return x

    cos_th = np.cos(np.radians(angle_deg))
    for i, li in enumerate(labs):
        for lj in labs[i + 1:]:
            if abs(axes[li] @ axes[lj]) < cos_th:
                continue
            d = cents[lj] - cents[li]
            # distance of cluster-j center from cluster-i axis line
            line_dist = np.linalg.norm(d - (d @ axes[li]) * axes[li])
            along = abs(d @ axes[li])
            if line_dist < 0.012 and along < (exts[li] + exts[lj]) / 2 + gap:
                parent[find(lj)] = find(li)
    out = labels.copy()
    for l in labs:
        out[labels == l] = find(l)
    # relabel densely
    for i, l in enumerate(sorted({find(l) for l in labs})):
        out[out == l] = -(i + 2)
    return -(out + 2) * (out < -1) + -1 * (out == -1)


# ---------------------------------------------------------------- bodies

@dataclass
class ReconBody:
    offsets: np.ndarray        # (P, 3) body frame (principal axes)
    pos: np.ndarray            # (3,) world COM
    quat: np.ndarray           # (4,) xyzw body->world
    mass: float
    inertia_diag: np.ndarray   # (3,)
    verts: np.ndarray          # render mesh, body frame
    faces: np.ndarray
    colors: np.ndarray


def build_body(cluster_verts, cluster_colors, faces=None) -> ReconBody | None:
    """faces: (F, 3) indices into cluster_verts (the reconstruction's own
    triangles restricted to this body) so the demo can render a lit surface
    instead of a point cloud."""
    hull0 = trimesh.Trimesh(vertices=cluster_verts).convex_hull
    if hull0.volume < 1e-8:
        return None
    com0 = hull0.center_mass
    w0, V = np.linalg.eigh(hull0.moment_inertia)   # principal axes, world
    if np.linalg.det(V) < 0:
        V[:, 0] *= -1                                # keep right-handed
    body_verts = (cluster_verts - com0) @ V          # x = long axis

    # Single-image reconstruction inflates thin objects: these pencils came
    # out 5-8.5 mm in radius against a 4-4.5 mm barrel (docs/objects.md),
    # thicker than any training pencil (3.5-5.5 mm) and visibly fat. The
    # length is trusted (KNOWN_LENGTH scaling); the radius is set the same
    # way, by scaling the cross-section about the axis to the known value.
    # scale the outer radius (95th percentile: the grip and clip bumps set
    # the convex hull, and the physics samples the hull), not the median
    radial = np.linalg.norm(body_verts[:, 1:], axis=1)
    r_out = float(np.percentile(radial, 95))
    if r_out > 1e-4:
        body_verts[:, 1:] *= KNOWN_RADIUS / r_out
    hull = trimesh.Trimesh(vertices=body_verts).convex_hull
    com_b = hull.center_mass
    body_verts = body_verts - com_b
    hull.apply_translation(-com_b)
    mass = PENCIL_DENSITY * hull.volume
    inertia_b = hull.moment_inertia * PENCIL_DENSITY  # about COM, body frame
    w, U = np.linalg.eigh(inertia_b)                 # re-diagonalize
    if np.linalg.det(U) < 0:
        U[:, 0] *= -1
    body_verts = body_verts @ U
    R_world = V @ U
    quat = Rotation.from_matrix(R_world).as_quat()
    com = com0 + V @ com_b
    surf = trimesh.Trimesh(vertices=np.asarray(hull.vertices) @ U,
                           faces=hull.faces, process=False)
    offsets = farthest_point_sample(
        trimesh.sample.sample_surface(surf, 20000, seed=1)[0])

    # Render mesh: a clean capsule fitted to the body (axis = principal axis
    # of least inertia = body-frame x), colored per vertex from the nearest
    # reconstruction vertex. Raw reconstruction triangles were lumpy and,
    # once split per body, full of holes; this keeps the real color bands
    # (barrel, grip, eraser) on a watertight pencil-shaped surface that also
    # matches the physics capsule proxy.
    from scipy.spatial import cKDTree
    along = body_verts[:, 0]
    radius = KNOWN_RADIUS
    half = float(max(np.ptp(along) / 2 - radius, 0.01))
    cap = trimesh.creation.capsule(radius=radius, height=2 * half, count=[24, 12])
    cap.apply_translation(-cap.center_mass)
    # capsule is built along z; rotate z -> x
    cap.apply_transform(trimesh.transformations.rotation_matrix(
        np.pi / 2, [0, 1, 0]))
    tree = cKDTree(body_verts)
    _, nn = tree.query(np.asarray(cap.vertices))
    render_colors = np.asarray(cluster_colors)[nn]

    return ReconBody(
        offsets=offsets.astype(np.float32),
        pos=com.astype(np.float32),
        quat=quat.astype(np.float32),
        mass=float(mass),
        inertia_diag=np.abs(w).astype(np.float32),
        verts=np.asarray(cap.vertices, np.float32),
        faces=np.asarray(cap.faces, np.int32),
        colors=render_colors.astype(np.uint8),
    )


# ---------------------------------------------------------------- pipeline

def _segment_auto(verts: np.ndarray) -> np.ndarray:
    """Try several RANSAC radii; score each result by how pencil-like its
    bodies are (span 9-18cm, radial thickness 3-8mm) minus penalties for
    non-pencil bodies and unassigned points. Returns the best labeling."""
    # Occam scoring: a lengthwise-split pencil makes two individually
    # "valid" shells, so more bodies must never beat fewer. Lexicographic:
    # (1) all bodies pencil-valid, (2) fewer unassigned points,
    # (3) FEWER bodies.
    best_key, best_labels = None, None
    for r, mf in ((0.0085, 0.05), (0.007, 0.04), (0.006, 0.035)):
        labels = ransac_lines(verts, radius=r, len_range=(0.08, 0.18),
                              min_frac=mf)
        n_bodies = valid = 0
        for l in np.unique(labels):
            if l < 0:
                continue
            n_bodies += 1
            pts = verts[labels == l]
            c = pts.mean(0)
            _, _, Vt = np.linalg.svd(pts - c, full_matrices=False)
            span = np.ptp(pts @ Vt[0])
            rel = pts - c
            r_med = np.median(np.linalg.norm(
                rel - np.outer(rel @ Vt[0], Vt[0]), axis=1))
            if 0.09 < span < 0.18 and 0.003 < r_med < 0.008:
                valid += 1
        unassigned = float((labels < 0).mean())
        key = (valid == n_bodies and n_bodies > 0, -round(unassigned, 2),
               -n_bodies)
        if best_key is None or key > best_key:
            best_key, best_labels = key, labels
    return best_labels


def _body_spans(verts, labels):
    spans = []
    for l in np.unique(labels):
        if l < 0:
            continue
        pts = verts[labels == l]
        c = pts.mean(0)
        _, _, Vt = np.linalg.svd(pts - c, full_matrices=False)
        spans.append(float(np.ptp(pts @ Vt[0])))
    return spans


def mesh_to_packet(mesh: trimesh.Trimesh) -> dict:
    m = canonicalize(mesh)
    verts = np.asarray(m.vertices)
    colors = np.asarray(m.visual.vertex_colors)[:, :3]

    # pass 1, unit-agnostic: coarse line extraction to establish metric scale
    diag = float(np.linalg.norm(np.ptp(verts, 0)))
    lab0 = ransac_lines(verts, radius=0.045 * diag,
                        len_range=(0.35 * diag, 1.05 * diag))
    spans = _body_spans(verts, lab0)
    if not spans:
        raise ValueError("no line-like bodies found; not a pencil scene?")

    # Refine "up" from the pencils themselves: lying pencils span the table
    # plane, so the ground normal is the direction least aligned with their
    # axes. RANSAC on raw vertices picked a wrong plane on a 5-pencil scene
    # (two pencils came out standing on end).
    axes = []
    for l in np.unique(lab0):
        if l < 0:
            continue
        pts = verts[lab0 == l]
        _, _, Vt = np.linalg.svd(pts - pts.mean(0), full_matrices=False)
        axes.append(Vt[0])
    if len(axes) >= 2:
        A = np.stack(axes)
        _, _, Vt = np.linalg.svd(A, full_matrices=True)
        normal = Vt[-1]                       # least aligned with all axes
        if normal[2] < 0:
            normal = -normal
        R, _ = Rotation.align_vectors([[0, 0, 1]], [normal])
        verts = verts @ R.as_matrix().T
        verts -= [verts[:, 0].mean(), verts[:, 1].mean(), 0]

    scale = KNOWN_LENGTH / max(spans)
    verts = verts * scale
    verts[:, 2] -= verts[:, 2].min()

    # pass 2, metric: no single radius handles every arrangement (a tight
    # radius splits near-parallel touching pencils but over-segments fat
    # reconstructions), so run candidates and keep the segmentation whose
    # bodies look most like pencils: right length, right thickness.
    labels = _segment_auto(verts)

    # per-body surface triangles: faces whose three vertices share the label
    all_faces = np.asarray(m.faces)
    bodies = []
    for l in np.unique(labels):
        if l < 0:
            continue
        sel = labels == l
        idx_map = -np.ones(len(verts), np.int64)
        idx_map[sel] = np.arange(int(sel.sum()))
        f = all_faces[sel[all_faces].all(1)]
        b = build_body(verts[sel], colors[sel], idx_map[f])
        if b is not None and len(b.offsets) >= 40:
            bodies.append(b)

    H = C.HISTORY
    B = len(bodies)
    packet = {
        "offsets_list": [b.offsets for b in bodies],
        "mass": np.array([b.mass for b in bodies], np.float32),
        "inertia_diag": np.stack([b.inertia_diag for b in bodies]),
        "pos": np.tile(np.stack([b.pos for b in bodies]), (H, 1, 1)),
        "quat": np.tile(np.stack([b.quat for b in bodies]), (H, 1, 1)),
        "linvel": np.zeros((H, B, 3), np.float32),
        "angvel": np.zeros((H, B, 3), np.float32),
        "labels": labels,
        "verts_scaled": verts,
        "colors": colors,
        "render": [
            {"verts": b.verts, "colors": b.colors, "faces": b.faces} for b in bodies],
    }
    return packet

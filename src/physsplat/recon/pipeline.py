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
from ..common import pencil as PENCIL
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
    tip_conf: float = 0.0      # 0 = which end is the point is a coin flip


def _tip_sign(pts, colors, com, axis) -> tuple[int, float]:
    """Which way along `axis` the pencil's POINT faces, and how much to
    believe it.

    The canonical body is the same object every time, so nothing in the
    geometry says which end is which any more; it has to come from colour.
    Three cues, all weak on their own, all measured on the reconstruction:
    the lead is nearly black, the eraser is white, and the rubber grip is
    the least saturated thing on the pencil and sits nearer the point.

    They are weak because they should be. The eraser is a 3 mm cap on a
    150 mm pencil, the lead is smaller still, and in a pile the ends are
    exactly what is occluded. Measured over these four scenes the three
    cues disagree with each other on roughly half the bodies, so this
    returns a margin as well as a sign and the packet records it. A body
    with a low margin is a coin flip and should be read as one rather than
    trusted because it came out of code.
    """
    c = np.asarray(colors, float)
    a = (np.asarray(pts, float) - com) @ axis
    span = a.max() - a.min()
    if span < 1e-6 or len(c) != len(a):
        return 1, 0.0
    u = (a - a.min()) / span                       # 0 at one end, 1 at the other
    lum = 0.299 * c[:, 0] + 0.587 * c[:, 1] + 0.114 * c[:, 2]
    mx, mn = c.max(1), c.min(1)
    sat = np.where(mx > 1, (mx - mn) / np.maximum(mx, 1), 0.0)

    votes = []
    dark = u[lum <= np.percentile(lum, 5)]
    if len(dark):
        votes.append(dark.mean() - 0.5)            # lead -> the point
    light = u[lum >= np.percentile(lum, 95)]
    if len(light):
        votes.append(0.5 - light.mean())           # eraser -> away from it
    grey = u[sat <= np.percentile(sat, 10)]
    if len(grey):
        votes.append(grey.mean() - 0.5)            # grip -> nearer the point
    if not votes:
        return 1, 0.0
    score = float(np.mean(votes))
    agree = float(np.mean([np.sign(v) == np.sign(score) for v in votes]))
    # margin: how far from a coin flip, 0 when the cues cancel or disagree
    return (1 if score >= 0 else -1), abs(score) * 2 * agree


def _pair_overlap(pa, Ra, pb, Rb):
    """Deepest overlap between two canonical pencils, and the direction to
    push A away from B. Mirrors capsuleClosest in web/public/js/physics.js:
    the radius varies along the pencil, so the deepest overlap is generally
    not at the closest approach of the two axes.
    """
    half = PENCIL.LENGTH / 2
    da, db = Ra[:, 0], Rb[:, 0]           # body +x is the pencil's axis
    best = (-1e9, None, None)
    for s in np.linspace(-half, half, 33):
        ca = pa + s * da
        t = float(np.clip((ca - pb) @ db, -half, half))
        cb = pb + t * db
        d = ca - cb
        dist = float(np.linalg.norm(d))
        pen = float(PENCIL.radius_at_offset(s) + PENCIL.radius_at_offset(t)) - dist
        if pen > best[0]:
            best = (pen, d / dist if dist > 1e-9 else np.array([0.0, 0.0, 1.0]), s)
    return best[0], best[1]


def _settle(bodies, iters: int = 300, tol: float = 1e-5):
    """Push the scene apart until nothing overlaps, then set it on the table.

    Needed because the canonical pencil is 150 mm and the reconstructions it
    replaces measured 99 to 150, so growing each one back to its real length
    drives ends into neighbours the photograph shows merely touching. This
    only resolves overlap and re-grounds; the demo's own pre-roll then
    settles the scene with the learned model.
    """
    if not bodies:
        return
    R = [Rotation.from_quat(b.quat).as_matrix() for b in bodies]
    for _ in range(iters):
        worst = 0.0
        for i in range(len(bodies)):
            for j in range(i + 1, len(bodies)):
                pen, n = _pair_overlap(bodies[i].pos, R[i], bodies[j].pos, R[j])
                if pen <= tol:
                    continue
                worst = max(worst, pen)
                shift = (0.5 * pen) * n
                bodies[i].pos = (bodies[i].pos + shift).astype(np.float32)
                bodies[j].pos = (bodies[j].pos - shift).astype(np.float32)
        if worst <= tol:
            break
    # and set the pile on the table, measured on the particles the ground
    # rule actually holds rather than on the reconstruction's vertices
    drop = min(float((R[i] @ b.offsets.T).T[:, 2].min() + b.pos[2])
               for i, b in enumerate(bodies))
    for b in bodies:
        b.pos = (b.pos - np.array([0.0, 0.0, drop], np.float32)).astype(np.float32)


def build_body(cluster_verts, cluster_colors, faces=None) -> ReconBody | None:
    """One pencil.

    The shape is not measured, it is known: see common/pencil.py. What the
    photograph contributes is where this pencil is, which way it lies, which
    end is the point, and what colour it is. Everything else comes from the
    canonical object, so every pencil in every scene is the same 150 mm,
    9 mm, 6.2 g BIC Matic Grip, which is what they are.

    Taking the shape from the reconstruction instead is what produced the
    bodies this replaces: 99 to 150 mm long, elliptical in section, tapered
    at the ends where a real pencil is straight, and no two alike.
    """
    pts = np.asarray(cluster_verts, float)
    if len(pts) < 20:
        return None
    com0 = pts.mean(0)
    _, _, Vt = np.linalg.svd(pts - com0, full_matrices=False)
    axis = Vt[0]
    a = (pts - com0) @ axis
    # centre on the midpoint of the extremes, not the centroid: sampling is
    # denser in the middle of a reconstruction, which drags the mean
    centre = com0 + axis * ((a.min() + a.max()) / 2)

    sign, tip_conf = _tip_sign(pts, cluster_colors, com0, axis)
    x_axis = axis * sign                          # body +x is the point
    tmp = np.array([0.0, 0.0, 1.0])
    if abs(float(tmp @ x_axis)) > 0.9:
        tmp = np.array([0.0, 1.0, 0.0])
    y_axis = np.cross(tmp, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    z_axis = np.cross(x_axis, y_axis)
    R_world = np.column_stack([x_axis, y_axis, z_axis])

    surf = PENCIL.solid()
    body_verts = np.asarray(surf.vertices, float)   # body frame, +x = point
    from scipy.spatial import cKDTree
    tree = cKDTree(pts)
    _, nn = tree.query(centre + body_verts @ R_world.T)

    return ReconBody(
        offsets=PENCIL.particles().astype(np.float32),
        pos=centre.astype(np.float32),
        quat=Rotation.from_matrix(R_world).as_quat().astype(np.float32),
        mass=float(PENCIL.MASS),
        inertia_diag=PENCIL.inertia().astype(np.float32),
        verts=body_verts.astype(np.float32),
        faces=np.asarray(surf.faces, np.int32),
        colors=np.asarray(cluster_colors)[nn].astype(np.uint8),
        tip_conf=float(tip_conf),
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

    _settle(bodies)

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
        "canonical_pencil": True,
        "tip_conf": [b.tip_conf for b in bodies],
        "render": [
            {"verts": b.verts, "colors": b.colors, "faces": b.faces} for b in bodies],
    }
    return packet

"""Where the camera was: register each scene's reconstruction to its photograph.

    uv run python scripts/register_photo.py [IMG_8596 ...]

The pencils are the calibration target. The packet's pencils are rendered
as silhouettes through a candidate pinhole camera (the focal length is the
EXIF one, 26 mm equivalent) and scored against the photograph's pencil
mask: Dice overlap plus, for the coloured pencils, hue agreement. The
camera pose is searched on a grid over azimuth, elevation, distance and
roll, then refined; both handednesses are tried, because a single-image
reconstruction can come out as the photograph's mirror image.

Two earlier attempts are worth recording. Assuming the reconstruction's
frame was the image's frame (image right = +x, image up = +y) was wrong by
about 60 degrees of in-plane rotation on IMG_8626. Fitting lines to the
pencil mask instead failed on the crowded, foreshortened piles: collinear
pencils merged into one 200 mm line and a pencil's grip beside another's
barrel became a fragment. Rendering the pencils and comparing silhouettes
has neither problem.

What the fit says about the reconstructions themselves: the handedness is
right in all four scenes, and the layouts are only approximate. IMG_8626
overlaps its photograph at Dice 0.76; the other three at about 0.6, which
is a pile of the right pencils in roughly the right places, not the
photographed pile. IMG_8626's packet also has five pencils where the
photograph has four. So the "compare with the photo" view shows the
reconstruction's error as much as the simulator's, which is the point.

Output: data/registration/<scene>.json with the camera and the homography
from desk-plane metres to full-resolution pixels, plus an overlay PNG.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import desk_texture as dt  # noqa: E402

OUT = ROOT / "data" / "registration"
F35 = 26.0            # 35 mm-equivalent focal length, EXIF, diagonal convention
SCORE_W = 400         # working width for the silhouette match
PENCIL_R = 0.00475    # a little over the barrel: the mask has soft edges
T_ALONG = np.linspace(-0.075, 0.075, 13)


def hue_of(c):
    c = np.asarray(c, np.float32)
    r, g, b = c[..., 0], c[..., 1], c[..., 2]
    mx = c.max(-1); mn = c.min(-1); d = mx - mn
    h = np.where(d < 1e-6, 0, np.where(mx == r, ((g - b) / np.maximum(d, 1e-6)) % 6,
                 np.where(mx == g, (b - r) / np.maximum(d, 1e-6) + 2, (r - g) / np.maximum(d, 1e-6) + 4)))
    return h * 60


def photo_masks(scene: str):
    rgb = dt.load(scene); H, W = rgb.shape[:2]
    k = SCORE_W / W
    small = np.asarray(Image.fromarray(rgb).resize((SCORE_W, int(H * k)), Image.LANCZOS))
    work = np.asarray(Image.fromarray(rgb).resize((dt.WORK_W, int(H * dt.WORK_W / W)), Image.LANCZOS))
    top = int(dt.desk_top_row(work) * k / (dt.WORK_W / W))
    pm = np.zeros(small.shape[:2], bool)
    pm[top:] = dt.pencil_mask(small[top:])
    lab, n = ndimage.label(pm)
    sizes = ndimage.sum(pm, lab, range(1, n + 1)) if n else []
    pm = np.isin(lab, [i + 1 for i, sz in enumerate(sizes) if sz >= 60])
    lab, n = ndimage.label(pm); keep = np.zeros_like(pm); h, w = pm.shape
    for kk in range(1, n + 1):                      # a desk edge touches the border; a pencil never does
        comp = lab == kk; ys, xs = np.nonzero(comp)
        if xs.min() == 0 or ys.min() == 0 or xs.max() == w - 1 or ys.max() == h - 1:
            continue
        keep |= comp
    f = small.astype(np.float32) / 255
    mx, mn = f.max(-1), f.min(-1)
    sat = np.where(mx > 0.02, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    return {"small": small, "pm": keep, "hue": hue_of(f), "sat": sat > 0.22, "k": k, "W": W, "H": H}


def packet_bodies(scene: str):
    packet = json.loads((ROOT / "web/public/packets" / f"{scene}.json").read_text())
    out = []
    for b in packet["bodies"]:
        x, y, z, w = b["quat"]
        ax = np.array([1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w)])
        col = np.array(b["render_colors"]).mean(0) / 255
        out.append({"c": np.array(b["pos"]), "d": ax / np.linalg.norm(ax),
                    "hue": float(hue_of(col)), "sat": float((col.max() - col.min()) / max(col.max(), 1e-6))})
    return out


def cam_from(az, el, dist, roll, tx, ty, pile):
    """Camera looking at pile + (tx, ty) from spherical (az, el, dist), rolled about its axis."""
    tgt = pile + [tx, ty, 0]
    C = tgt + dist * np.array([np.cos(az) * np.cos(el), np.sin(az) * np.cos(el), np.sin(el)])
    z = tgt - C; z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1]); x /= np.linalg.norm(x); y = np.cross(z, x)
    R = (Rotation.from_rotvec(roll * z) * Rotation.from_matrix(np.stack([x, y, z]))).as_matrix()
    return R, C


class Fit:
    def __init__(self, scene):
        self.scene = scene
        self.ph = photo_masks(scene)
        self.bodies = packet_bodies(scene)
        sh = self.ph["small"].shape
        self.f = F35 / 43.27 * np.hypot(*sh[:2])
        self.cx, self.cy = sh[1] / 2, sh[0] / 2
        self.pile = np.mean([b["c"] for b in self.bodies], 0); self.pile[2] = 0

    def render(self, R, C, mirror):
        S = np.array([-1, 1, 1]) if mirror else np.array([1, 1, 1])
        sh = self.ph["small"].shape
        im = Image.new("L", (sh[1], sh[0]), 0); dr = ImageDraw.Draw(im)
        order = []
        for i, b in enumerate(self.bodies):
            P = (b["c"] * S)[None, :] + T_ALONG[:, None] * (b["d"] * S)[None, :]
            Pc = (P - C) @ R.T
            if (Pc[:, 2] < 0.03).any():
                return None
            u = self.f * Pc[:, 0] / Pc[:, 2] + self.cx; v = self.f * Pc[:, 1] / Pc[:, 2] + self.cy
            wpx = 2 * self.f * PENCIL_R / Pc[:, 2]
            order.append((Pc[:, 2].mean(), i, u, v, wpx))
        for _, i, u, v, wpx in sorted(order, reverse=True):          # far first, near overwrites
            for j in range(len(T_ALONG) - 1):
                dr.line([(u[j], v[j]), (u[j + 1], v[j + 1])], fill=i + 1,
                        width=max(1, int(round((wpx[j] + wpx[j + 1]) / 2))))
        return np.asarray(im)

    def score(self, lbl):
        if lbl is None:
            return (-1.0, 0.0, 0.0)
        pm = self.ph["pm"]; rm = lbl > 0
        dice = 2 * (rm & pm).sum() / max(rm.sum() + pm.sum(), 1)
        agree, wts = [], []
        for i, b in enumerate(self.bodies):
            if b["sat"] < 0.25:
                continue
            sel = (lbl == i + 1) & pm & self.ph["sat"]
            if sel.sum() < 5:
                continue
            dh = np.abs(self.ph["hue"][sel] - b["hue"]); dh = np.minimum(dh, 360 - dh)
            agree.append(np.cos(np.radians(dh)).mean()); wts.append(sel.sum())
        hue_term = float(np.average(agree, weights=wts)) if wts else 0.0
        return (dice + 0.25 * hue_term, float(dice), hue_term)

    def fit(self, mirror):
        cands = []
        for az in np.radians(np.arange(0, 360, 15)):
            for el in np.radians((15, 25, 35, 45, 55, 65, 75)):
                for dist in (0.22, 0.32, 0.45, 0.65):
                    for roll in np.radians((-20, -10, 0, 10, 20)):
                        R, C = cam_from(az, el, dist, roll, 0, 0, self.pile)
                        cands.append((self.score(self.render(R, C, mirror))[0], (az, el, dist, roll, 0.0, 0.0)))
        cands.sort(key=lambda c: -c[0])
        best = None
        for _, p0 in cands[:4]:
            def neg(p):
                R, C = cam_from(*p, self.pile)
                return -self.score(self.render(R, C, mirror))[0]
            r = minimize(neg, np.array(p0), method="Nelder-Mead",
                         options={"xatol": 1e-4, "fatol": 1e-5, "maxiter": 1500})
            if best is None or r.fun < best.fun:
                best = r
        R, C = cam_from(*best.x, self.pile)
        return best.x, self.score(self.render(R, C, mirror))

    def run(self):
        res = {m: self.fit(m) for m in (False, True)}
        mirror = res[True][1][0] > res[False][1][0]
        p, s = res[mirror]
        R, C = cam_from(*p, self.pile)
        k = self.ph["k"]; W, H = self.ph["W"], self.ph["H"]
        ff = self.f / k
        K = np.array([[ff, 0, W / 2], [0, ff, H / 2], [0, 0, 1]])
        Rt = np.hstack([R, (-R @ C)[:, None]])
        Hm = K @ Rt[:, [0, 1, 3]]; Hm /= Hm[2, 2]
        out = {
            "scene": self.scene, "mirror": bool(mirror), "dice": s[1], "hue_agreement": s[2],
            "score_as_is": res[False][1][0], "score_mirrored": res[True][1][0],
            "camera_pos_m": C.tolist(), "look_at_m": (self.pile + [p[4], p[5], 0]).tolist(),
            "R_world_to_cam": R.tolist(), "elevation_deg": float(np.degrees(p[1])),
            "azimuth_deg": float(np.degrees(p[0]) % 360), "distance_m": float(p[2]), "roll_deg": float(np.degrees(p[3])),
            "focal_px_fullres": float(ff), "fov_v_deg": float(np.degrees(2 * np.arctan(H / 2 / ff))),
            "H_px_from_m": Hm.tolist(),
        }
        lbl = self.render(R, C, mirror)
        ov = self.ph["small"].astype(np.float32).copy(); pm = self.ph["pm"]
        edge = ndimage.binary_dilation(lbl > 0, iterations=1) & ~(lbl > 0)
        ov[lbl > 0] = ov[lbl > 0] * 0.55 + np.array([255, 0, 0]) * 0.45
        ov[edge] = [255, 255, 0]
        ov[pm & ~(lbl > 0)] = ov[pm & ~(lbl > 0)] * 0.5 + np.array([0, 255, 0]) * 0.5
        return out, Image.fromarray(ov.astype(np.uint8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes", nargs="*", default=["IMG_8504", "IMG_8513", "IMG_8596", "IMG_8626"])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for scene in args.scenes:
        out, ov = Fit(scene).run()
        (OUT / f"{scene}.json").write_text(json.dumps(out, indent=1))
        ov.save(OUT / f"{scene}.png")
        print(f"{scene}: {'MIRROR' if out['mirror'] else 'as is'}, dice {out['dice']:.2f}, hue {out['hue_agreement']:.2f}, "
              f"camera {out['distance_m']*100:.0f} cm away, {out['elevation_deg']:.0f} deg up, azimuth {out['azimuth_deg']:.0f} deg, "
              f"roll {out['roll_deg']:.0f} deg  (as is {out['score_as_is']:.3f} vs mirrored {out['score_mirrored']:.3f})")


if __name__ == "__main__":
    main()

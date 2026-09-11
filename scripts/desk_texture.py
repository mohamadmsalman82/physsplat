"""The desk itself: lift it out of each photograph so the demo plays on it.

    uv run python scripts/desk_texture.py [IMG_8596 ...]

For each scene this writes web/public/desk/<scene>.jpg, the desk from the
photograph with the pencils and their shadows removed, and <scene>.json,
which says how that image lies in the simulator's world (metres per pixel,
which pixel is the world origin) and where the desk ends and the wall
begins. With those the browser puts the real desk under the simulated
pencils at the right scale, with the photograph's own lighting gradient,
grain and colour, and stands a wall where the photograph has one.

Two things this deliberately does NOT do, and why.

It does not inpaint. Biharmonic inpainting of a 27% hole left cloudy grey
ghosts of the pencils with their colours bled into the fill. The desk is
nearly uniform, so the right fill is the desk's own smooth gradient,
estimated from every pixel that is not pencil by normalized convolution,
plus the desk's own grain, re-sampled from elsewhere on the desk so the
filled region has real texture rather than a smooth patch.

It does not fit the registration from pencil blobs. Crossing pencils are
one connected component, so per-pencil matching fails on every scene. The
reconstruction's frame is camera-centric and canonicalize() keeps it so:
image right is world +x, image up is world +y (checked on IMG_8596, where
the orange pencil is left and the teal one top in both). Scale and origin
come from the pile's bounding box in both frames.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "web" / "public" / "desk"
WORK_W = 1600
OUT_W = 2048
PENCIL_LEN_M = 0.150


def load(scene: str) -> np.ndarray:
    im = ImageOps.exif_transpose(Image.open(ROOT / "data" / "photos" / "B_core" / f"{scene}.jpg"))
    return np.asarray(im.convert("RGB"))


def luminance(rgb: np.ndarray) -> np.ndarray:
    f = rgb.astype(np.float32) / 255.0
    return 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]


def desk_top_row(rgb: np.ndarray) -> int:
    """First row, from the top, where the desk begins: rows above it are the
    wall or the desk's far edge, darker than the desk. Row medians are used
    so a pencil crossing a row does not count."""
    lum = luminance(rgb)
    rows = np.median(lum, axis=1)
    desk = np.median(rows[len(rows) // 3:])          # the lower two thirds are desk
    bright = rows > desk - 0.20     # the far desk is in the gradient's shade, keep it
    # the first row from which the next 40 rows are all desk
    for r in range(len(rows) - 40):
        if bright[r:r + 40].all():
            return r
    return 0


def pencil_mask(rgb: np.ndarray) -> np.ndarray:
    """Pencils on a bright desk: clearly darker than it, or clearly coloured."""
    lum = luminance(rgb)
    f = rgb.astype(np.float32) / 255.0
    mx, mn = f.max(-1), f.min(-1)
    sat = np.where(mx > 0.02, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    desk = np.median(lum)
    return (lum < desk - 0.25) | (sat > 0.26)    # pencils are far darker than the far desk


def shadow_mask(rgb: np.ndarray, near: np.ndarray) -> np.ndarray:
    """Pixels a little darker than the smooth desk around them, near a pencil."""
    lum = luminance(rgb)
    w = ndimage.gaussian_filter((~near).astype(np.float32), 70)
    base = ndimage.gaussian_filter(np.where(near, 0.0, lum), 70) / np.maximum(w, 1e-3)
    return (lum < base - 0.02) & ndimage.binary_dilation(near, iterations=130)


def fill(rgb: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Rebuild the whole desk as its smooth field plus its grain.

    Not only inside the mask. The pencils' shadows fade out over tens of
    millimetres and no mask catches all of that, so filling only the hole
    left a lighter blotch the exact shape of the pile, ringed by desk that
    still carried the shadows of pencils that were no longer there. Instead
    the smooth field (the desk's lighting gradient, estimated from every
    pixel that is not pencil, at a scale far wider than any shadow) is used
    everywhere, and the desk's fine grain is put back on top of it: the
    original grain where the desk was visible, grain re-sampled from
    elsewhere where it was not. A white desk is a gradient plus grain, and
    this is that, with no pencil-shaped memory in it."""
    f = rgb.astype(np.float32) / 255.0
    keep = (~mask).astype(np.float32)
    smooth = np.empty_like(f)
    for c in range(3):
        num = ndimage.gaussian_filter(f[..., c] * keep, 120)
        den = ndimage.gaussian_filter(keep, 120)
        smooth[..., c] = num / np.maximum(den, 1e-4)
    # Keep the desk's gradient, but not all of it. The phone was close and
    # its lighting fell off fast across the frame, so the photo's field is a
    # strong bright-to-grey ramp that, laid on the table, read as a soft
    # grey patch with a rectangular boundary. Pulled 45% toward its own mean
    # it is still the desk's light, and it no longer looks like a sheet.
    mean = smooth.reshape(-1, 3).mean(0)
    smooth = mean + 0.55 * (smooth - mean)
    fine = np.empty_like(f)
    for c in range(3):
        fine[..., c] = f[..., c] - ndimage.gaussian_filter(f[..., c], 6)
    src = np.nonzero(~mask)
    dst = np.nonzero(mask)
    pick = rng.integers(0, len(src[0]), size=len(dst[0]))
    grain = fine.copy()
    grain[dst[0], dst[1]] = fine[src[0][pick], src[1][pick]]
    return (np.clip(smooth + grain, 0, 1) * 255).astype(np.uint8)


def body_layout(scene: str):
    p = json.loads((ROOT / "web" / "public" / "packets" / f"{scene}.json").read_text())
    out = []
    for b in p["bodies"]:
        x, y, z, w = b["quat"]
        ax = np.array([1 - 2 * (y * y + z * z), 2 * (x * y + z * w)])
        out.append({"centre": np.array(b["pos"][:2]), "axis": ax / np.linalg.norm(ax)})
    return out


def register(scene: str, pm: np.ndarray, scale_back: float, row0: int):
    """image right = world +x, image up = world +y. Scale from AREA, not from
    a bounding box: the pile's mask area is N pencils of 150 x ~9.5 mm less
    a little at each crossing, which does not care what shape the pile is,
    while a bounding box was thrown by any dark thing near the pile. Only
    components that do not touch the image border count, since a desk edge
    or a dark corner always does and a pencil never does. Pixel coordinates
    are in the FULL-RES, uncropped image; row0 is where the crop starts."""
    bodies = body_layout(scene)
    if not bodies:
        return None
    lab, n = ndimage.label(pm)
    keep = np.zeros_like(pm)
    h, w = pm.shape
    for k in range(1, n + 1):
        comp = lab == k
        ys, xs = np.nonzero(comp)
        if len(xs) < 800:
            continue
        if xs.min() == 0 or ys.min() == 0 or xs.max() == w - 1 or ys.max() == h - 1:
            continue
        keep |= comp
    ys, xs = np.nonzero(keep)
    if len(xs) < 800:
        return None
    area_px = float(len(xs)) * scale_back * scale_back
    area_m2 = len(bodies) * PENCIL_LEN_M * 0.0095 * 0.96      # 4% lost to crossings
    s = float(np.sqrt(area_m2 / area_px))
    img_c = np.array([xs.mean(), ys.mean() + row0 / scale_back]) * scale_back
    wld_c = np.array([b["centre"] for b in bodies]).mean(0)
    origin_px = img_c - np.array([wld_c[0], -wld_c[1]]) / s
    return {"m_per_px": s, "origin_px": [float(origin_px[0]), float(origin_px[1])],
            "pile_area_px": int(len(xs) * scale_back * scale_back)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes", nargs="*", default=["IMG_8504", "IMG_8513", "IMG_8596", "IMG_8626"])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for scene in args.scenes:
        rgb = load(scene)
        H, W = rgb.shape[:2]
        k = WORK_W / W
        small = np.asarray(Image.fromarray(rgb).resize((WORK_W, int(H * k)), Image.LANCZOS))

        top = desk_top_row(small)
        wall = small[:max(top - 8, 1)] if top > 20 else None
        desk = small[top:]

        pm = pencil_mask(desk)
        lab, n = ndimage.label(pm)
        sizes = ndimage.sum(pm, lab, range(1, n + 1)) if n else []
        # pencils are big; specks are sensor noise and dust
        pm = np.isin(lab, [i + 1 for i, sz in enumerate(sizes) if sz >= 60])
        # for the bounding box, only components a pencil could be: anything
        # smaller is a desk mark or a dark corner, and one of those far from
        # the pile halved the scale on IMG_8596
        big = np.isin(lab, [i + 1 for i, sz in enumerate(sizes) if sz >= 1500])
        pm_d = ndimage.binary_dilation(pm, iterations=22)
        sm = shadow_mask(desk, pm_d)
        mask = ndimage.binary_dilation(pm_d | sm, iterations=8)
        clean = fill(desk, mask, rng)
        # second pass: anything still clearly darker than the desk after the
        # fill escaped the mask (a lead tip, a speck of shadow); mask and refill
        again = luminance(clean) < np.median(luminance(clean)) - 0.12
        if again.any():
            mask2 = ndimage.binary_dilation(again, iterations=10) | mask
            clean = fill(desk, mask2, rng)
            mask = mask2

        out_h = int(clean.shape[0] * OUT_W / clean.shape[1])
        Image.fromarray(clean).resize((OUT_W, out_h), Image.LANCZOS).save(
            OUT / f"{scene}.jpg", quality=88, optimize=True)
        avg = clean[~mask].reshape(-1, 3).mean(0)
        b = max(8, clean.shape[0] // 12)
        border = np.concatenate([clean[:b].reshape(-1, 3), clean[-b:].reshape(-1, 3),
                                 clean[:, :b].reshape(-1, 3), clean[:, -b:].reshape(-1, 3)])
        edge = border.mean(0)

        reg = register(scene, pm, 1.0 / k, top)
        desc = {
            "image": f"{scene}.jpg",
            "image_px": [W, H],
            "crop_top_px": int(top / k),
            "desk_rgb": [int(round(v)) for v in avg],
            "edge_rgb": [int(round(v)) for v in edge],
            "wall_rgb": [int(round(v)) for v in wall.reshape(-1, 3).mean(0)] if wall is not None else None,
            "masked_fraction": float(mask.mean()),
            "orientation": "image right = +x, image up = +y (camera-centric reconstruction frame)",
        }
        if reg:
            desc.update(reg)
            # the desk's far edge, in world metres along +y
            desc["wall_y_m"] = float(-(top / k - reg["origin_px"][1]) * reg["m_per_px"])
        (OUT / f"{scene}.json").write_text(json.dumps(desc, indent=1))
        print(f"{scene}: desk starts at row {int(top / k)} of {H}, masked {mask.mean()*100:.1f}% of it, "
              f"desk rgb {desc['desk_rgb']}, wall rgb {desc['wall_rgb']}, "
              + (f"{reg['m_per_px']*1e6:.0f} um/px -> photo spans {W*reg['m_per_px']:.2f} x "
                 f"{(H-top/k)*reg['m_per_px']:.2f} m, wall at y={desc['wall_y_m']*100:.0f} cm"
                 if reg else "NO REGISTRATION"))


if __name__ == "__main__":
    main()

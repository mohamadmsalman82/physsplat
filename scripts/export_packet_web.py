"""Scene packet (.pkl) -> web JSON under web/public/packets/.

    uv run python scripts/export_packet_web.py data/packets.nosync/*.pkl

Each packet carries physics data (offsets, mass, inertia, init pose),
render point clouds (true reconstruction colors), and a capsule proxy per
body for raycast picking. An index.json lists available scenes.
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from physsplat.common import pencil as PENCIL
from physsplat.eval.metrics import capsule_from_offsets


def vivid(colors: np.ndarray, sat: float = 1.6, val: float = 1.15) -> np.ndarray:
    """TripoSR vertex colors come back muted; restore saturation for display
    only (the physics never sees color)."""
    import colorsys
    out = np.empty_like(colors)
    for i, (r, g, b) in enumerate(np.asarray(colors, np.float64) / 255.0):
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        rr, gg, bb = colorsys.hsv_to_rgb(h, min(1.0, s * sat), min(1.0, v * val))
        out[i] = (int(rr * 255), int(gg * 255), int(bb * 255))
    return out


def convert(pkl_path: str, out_dir: Path) -> str:
    with open(pkl_path, "rb") as f:
        d = pickle.load(f)
    name = Path(pkl_path).stem
    r3 = lambda a, n=5: np.round(np.asarray(a, np.float64), n).tolist()
    bodies = []
    for b in range(len(d["offsets_list"])):
        # Every body is the canonical pencil, so its contact proxy is stated
        # rather than fitted: the axis is body +x by construction (the point
        # is at +x), the half length is the real 75 mm, and `taper` tells the
        # browser to read the radius off the profile instead of treating the
        # pencil as a uniform tube. Fitting a capsule to the particles gave
        # the MEDIAN radius and carried it to the tip, which is what held a
        # pencil up on a neighbour's point as though the point were 9 mm
        # thick. capsule_from_offsets stays for non-pencil bodies.
        if d.get("canonical_pencil"):
            axis, half, radius = np.array([1.0, 0.0, 0.0]), PENCIL.LENGTH / 2, PENCIL.GRIP_R
            taper = True
        else:
            axis, half, radius = capsule_from_offsets(d["offsets_list"][b])
            taper = False
        bodies.append({
            "offsets": r3(d["offsets_list"][b]),
            "mass": float(d["mass"][b]),
            "inertia": r3(d["inertia_diag"][b], 12),
            "pos": r3(d["pos"][-1][b]),
            "quat": r3(d["quat"][-1][b], 6),
            "tip_confidence": round(float(d.get("tip_conf", [0.0] * 99)[b]), 3),
            "capsule": {"axis": r3(axis, 4), "half": round(float(half), 4),
                        "radius": round(float(radius), 5), "taper": taper},
            "render_verts": r3(d["render"][b]["verts"], 4),
            "render_colors": vivid(np.asarray(d["render"][b]["colors"], np.uint8)).tolist(),
            "render_faces": np.asarray(d["render"][b].get("faces", []), np.int32).tolist(),
        })
    out = {"name": name, "bodies": bodies}
    path = out_dir / f"{name}.json"
    path.write_text(json.dumps(out))
    print(f"{path} ({path.stat().st_size/1e6:.1f} MB, {len(bodies)} bodies)")
    return name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pkls", nargs="+")
    ap.add_argument("--out", default="web/public/packets")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    names = [convert(p, out) for p in args.pkls]
    (out / "index.json").write_text(json.dumps(names))


if __name__ == "__main__":
    main()

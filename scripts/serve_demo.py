"""Phase 5 local demo server: WebSocket physics from a checkpoint.

    uv run python scripts/serve_demo.py checkpoints/run01/latest.pt
    open http://localhost:8765

Protocol (JSON):
  server -> client on connect:  {type: "scene", bodies: [{kind, dims,
      color, ...}], dt}
  server -> client at 60 Hz:    {type: "state", pos: [[x,y,z]...],
      quat: [[x,y,z,w]...]}
  client -> server:
      {type: "poke",  body, point [3], dv [3]}
      {type: "grab",  body, point [3], target [3]}   (repeat on mousemove)
      {type: "release"}
      {type: "reset"}

Grabs run the SAME mass-scaled spring-damper law as data generation
(closed-loop against the simulated state), pokes the same burst encoding.
"""

import argparse
import asyncio
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from physsplat.common import constants as C
from physsplat.datagen.writer import load_trajectory
from physsplat.eval.metrics import capsule_from_offsets
from physsplat.model.gnn import Simulator
from physsplat.model.live import LiveSim
from physsplat.model.normalize import Normalizer
from physsplat.train.dataset import TrajectoryDataset

app = FastAPI()
STATE: dict = {}


def load_scene(data_dir: str, index: int) -> dict:
    ds = TrajectoryDataset(data_dir, split="test")
    fi, key = ds.keys[index]
    with h5py.File(ds.files[fi]) as f:
        d = load_trajectory(f, key)
    return d


def load_packet(path: str) -> dict:
    """A photo-derived scene packet (scripts/recon_pipeline.py)."""
    import pickle

    with open(path, "rb") as f:
        d = pickle.load(f)
    d["regime"] = f"photo:{Path(path).stem}"
    return d


def scene_message(d) -> dict:
    bodies = []
    for b, offs in enumerate(d["offsets_list"]):
        kind = d["shape_kind"][b] if "shape_kind" in d else 0  # packets: capsules
        if kind == 0:
            _, half, radius = capsule_from_offsets(offs)
            bodies.append({"kind": "capsule", "half": half, "radius": radius})
        else:
            ext = (offs.max(0) - offs.min(0)).tolist()
            bodies.append({"kind": "box", "extents": ext})
    return {"type": "scene", "bodies": bodies, "dt": C.DT,
            "regime": str(d.get("regime", ""))}


def make_sim(d, model, normalizer, device) -> LiveSim:
    H = C.HISTORY
    scene = {"offsets_list": d["offsets_list"], "mass": d["mass"],
             "inertia": d["inertia_diag"]}
    init = {k: d[k][:H] for k in ("pos", "quat", "linvel", "angvel")}
    return LiveSim(model, normalizer, scene, init, device, ground_guard=True)


@app.get("/")
async def index():
    return HTMLResponse((Path(__file__).parent / "demo.html").read_text())


@app.websocket("/ws")
async def ws(sock: WebSocket):
    await sock.accept()
    d, model, normalizer, device = (STATE[k] for k in
                                    ("scene", "model", "norm", "device"))
    sim = make_sim(d, model, normalizer, device)
    await sock.send_text(json.dumps(scene_message(d)))
    grab = None            # {body, local_point(np), target(np)}
    poke_left, poke = 0, None

    async def recv_loop():
        nonlocal grab, poke, poke_left, sim
        while True:
            msg = json.loads(await sock.receive_text())
            if msg["type"] == "reset":
                sim = make_sim(d, model, normalizer, device)
                grab, poke_left = None, 0
            elif msg["type"] == "poke":
                m = float(d["mass"][msg["body"]])
                dv = np.clip(np.array(msg["dv"]), -C.POKE_DELTA_V[1],
                             C.POKE_DELTA_V[1])
                poke = (msg["body"], np.array(msg["point"]),
                        m * dv / (C.IMPULSE_STEPS * C.DT))
                poke_left = C.IMPULSE_STEPS
            elif msg["type"] == "grab":
                b = msg["body"]
                # store grab point in BODY frame so it rides the body
                from physsplat.model.integrator import quat_to_matrix
                R = quat_to_matrix(sim.quat[b]).cpu().numpy()
                local = R.T @ (np.array(msg["point"]) - sim.pos[b].cpu().numpy())
                if grab is None or grab["body"] != b:
                    grab = {"body": b, "local": local}
                grab["target"] = np.array(msg["target"])
            elif msg["type"] == "release":
                grab = None

    recv_task = asyncio.create_task(recv_loop())
    try:
        while True:
            t0 = time.perf_counter()
            act = (-1, None, None)
            if poke_left > 0:
                act = poke
                poke_left -= 1
            elif grab is not None:
                from physsplat.model.integrator import quat_to_matrix
                b = grab["body"]
                R = quat_to_matrix(sim.quat[b]).cpu().numpy()
                world_pt = R @ grab["local"] + sim.pos[b].cpu().numpy()
                vel = sim.lin_hist[-1][b].cpu().numpy()
                m = float(d["mass"][b])
                kp = m * C.GRAB_OMEGA**2
                kd = 2 * C.GRAB_ZETA * m * C.GRAB_OMEGA
                force = kp * (grab["target"] - world_pt) - kd * vel
                cap = C.GRAB_FORCE_CAP * m * C.GRAVITY
                n = np.linalg.norm(force)
                if n > cap:
                    force *= cap / n
                act = (b, world_pt, force)
            with torch.no_grad():
                pos, quat = sim.step(act[0], act[1], act[2])
            await sock.send_text(json.dumps(
                {"type": "state", "pos": np.round(pos, 5).tolist(),
                 "quat": np.round(quat, 5).tolist()}))
            await asyncio.sleep(max(0.0, C.DT - (time.perf_counter() - t0)))
    except WebSocketDisconnect:
        pass
    finally:
        recv_task.cancel()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--data", default="data/raw.nosync/train")
    ap.add_argument("--scene", type=int, default=0)
    ap.add_argument("--packet", default=None,
                    help="serve a photo-derived packet (.pkl) instead")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    ck = torch.load(args.checkpoint, map_location=device)
    model = Simulator(head=ck.get("head", "body")).to(device).eval()
    model.load_state_dict(ck["model"])
    STATE["model"] = model
    STATE["norm"] = Normalizer(ck.get("stats_path", "data/stats.json")).to(device)
    STATE["device"] = device
    STATE["scene"] = (load_packet(args.packet) if args.packet
                      else load_scene(args.data, args.scene))
    print(f"serving scene {args.scene} ({STATE['scene']['regime']}) "
          f"from step-{ck.get('step')} checkpoint on :{args.port}")
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()

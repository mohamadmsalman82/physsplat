# scripts/

Runnable entry points. The library code lives in `src/physsplat/`; these are
thin wrappers you actually execute, one per task, run as:

    uv run python scripts/<name>.py

Grows roughly one script per tutorial phase:

- `hello_rerun.py` — Phase 0 checkpoint: proves the Rerun viewer works.
- `gen_data.py` — (Phase 1) generate the full synthetic dataset into `data/`.
- `view_traj.py` — (Phase 1) scrub a recorded trajectory in Rerun for auditing.
- `train.py` — (Phase 3) launch training, writes to `checkpoints/`.
- `evaluate.py` — (Phase 4) metrics table + side-by-side videos.
- `serve_demo.py` — (Phase 5) WebSocket physics server for the local demo.
- `recon_photo.py` — (Phase 6) photo in, scene packet out.
- `export_onnx.py` — (Phase 7) export the model + parity test.

Rule of thumb: scripts parse arguments and call into `physsplat.*`; they
contain no logic worth unit-testing themselves.

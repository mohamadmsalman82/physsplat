"""Phase 7 (Python side): ship the trained model to the browser (Steps 58-59).

The demo runs physics CLIENT-SIDE (server round-trips would add 50-300 ms
per step and kill interactivity). Only the neural network is exported; the
graph builder and integrator are re-implemented in TypeScript in web/.

Files:
    to_onnx.py   export encode-process-decode to ONNX with dynamic node/edge
                 counts. Note: the fallback head's SVD does not export; if
                 that head ships, Kabsch is done in TypeScript.
    parity.py    THE gate for this phase: drive PyTorch and ONNX Runtime
                 with identical inputs over a 100-step rollout and assert
                 agreement to 1e-4. Write zero frontend model code until
                 this passes -- debugging numerics through a browser is
                 misery. A second parity test later compares Python vs. the
                 TypeScript port end to end.

Also exported alongside the model: data/stats.json (the normalization
constants). The browser must normalize inputs and denormalize outputs with
the exact training-time numbers or predictions are garbage.
"""

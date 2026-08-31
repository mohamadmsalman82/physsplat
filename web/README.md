# web/

Phase 7: the public-facing Next.js app (empty until then). Everything
physics-related runs in the visitor's browser; there is no physics server.

What goes here when the time comes:

- Next.js + React Three Fiber scene with a Gaussian-splat renderer
  (per-body splat batches, so moving an object updates 6 floats, not a
  100k-Gaussian buffer)
- ONNX Runtime Web running the trained GNN (WebGPU, WASM fallback)
- TypeScript ports of the graph builder and SE(3) integrator (line-by-line
  from `src/physsplat/model/`, verified by a cross-language parity test)
- Interaction: raycast picking against body OBBs, drag-to-impulse encoded
  exactly like training-data pokes, sliders, reset
- Loads precomputed scene packets served as static files (zero cost, no
  cold starts)

Deploys to Vercel.

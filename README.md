# PhysSplat

**A photo of everyday objects in. Interactive, learned 3D physics in your browser out.**

PhysSplat turns a single photograph of simple objects on a flat surface (blocks, pencils, erasers) into a fully interactive 3D simulation. The scene is reconstructed as 3D Gaussian Splats, decomposed into rigid bodies without supervision, and simulated by a graph neural network trained entirely on synthetic data — no hand-coded collision solver. You orbit the scene, poke objects with the mouse, and watch them topple, slide, and roll, with every physics step running client-side in the browser.

```
photo ──► 3D Gaussian Splats ──► unsupervised object          ──► learned GNN ──► interactive
          (TripoSR / LGM)         decomposition + particles        dynamics         WebGL demo
          once per scene          once per scene                   60 Hz, in-browser (ONNX)
```

## Why

Generative 3D reconstruction produces photorealistic scenes that are physically inert — no mass, no contact, no gravity. Classical physics engines simulate rigid bodies robustly but require clean meshes and hand-tuned parameters, so they cannot ingest a photograph. PhysSplat bridges the two with a learned simulator that operates on surface point clouds: exactly the representation a photo reconstruction can provide.

## How it works

Three ideas carry the project:

1. **Rigidity by construction.** The GNN detects contact through particle-level message passing, but decodes one linear + one angular acceleration *per body* and integrates on SE(3), so objects can never deform or "melt" over long rollouts.
2. **Noise injection.** Training inputs are corrupted with small random-walk noise against clean targets, so the model learns to correct its own errors — the difference between rollouts that drift in seconds and rollouts that stay plausible indefinitely.
3. **One shared sampler.** The exact same fixed-spacing surface sampler dots synthetic training objects and photo-reconstructed objects with physics particles, making real scenes statistically indistinguishable from training data. That is the whole sim-to-real transfer mechanism.

Training data is manufactured, not collected: PyBullet simulates thousands of randomized scenes (stacks, scatters, drops, random pokes) and the network learns to imitate it — from a photo-compatible representation.

## Status

Early development, built in phases (each ends with a runnable artifact):

- [x] **Phase 0** — environment, package skeleton, shared constants
- [ ] **Phase 1** — synthetic dataset generator (PyBullet → HDF5)
- [ ] **Phase 2** — the graph network simulator
- [ ] **Phase 3** — training (Apple Silicon / MPS)
- [ ] **Phase 4** — rollout evaluation + side-by-side videos
- [ ] **Phase 5** — local interactive demo
- [ ] **Phase 6** — photo → scene reconstruction pipeline
- [ ] **Phase 7** — in-browser inference (ONNX Runtime Web) + splat rendering
- [ ] **Phase 8** — public deployment

## Documentation

- [`docs/PhysSplat.pdf`](docs/PhysSplat.pdf) — the design document: full architecture, the mathematics of the learned simulator, data spec, evaluation plan.
- [`docs/PhysSplat-Tutorial.pdf`](docs/PhysSplat-Tutorial.pdf) — a 70-step build guide with background theory, per-step checkpoints, and common traps.

## Repository layout

```
src/physsplat/
  common/     shared constants + the particle sampler (used by datagen AND recon)
  datagen/    PyBullet scene generation, trajectory recording, HDF5 writer
  model/      graph builder, encode-process-decode GNN, SE(3) integrator
  train/      dataset, noise injection, training loop
  eval/       rollout metrics, comparison videos
  recon/      photo → splats → bodies → physics particles
  export/     ONNX export + Python/browser parity tests
scripts/      runnable entry points (one per task)
web/          Next.js app: splat renderer + client-side physics (Phase 7)
docs/         design document and build tutorial (LaTeX + PDF)
```

Each package's `__init__.py` documents what lives there and why.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and (for later phases) Node 20+.

```sh
uv sync
uv run python -c "import torch; print(torch.backends.mps.is_available())"  # True on Apple Silicon
uv run python scripts/hello_rerun.py                                       # Rerun viewer opens
```

<details>
<summary>Note: building PyBullet on macOS 26</summary>

PyBullet 3.2.7 ships no macOS arm64 wheel, and its vendored zlib collides with
the macOS 26 SDK (`fdopen` macro). If `uv sync` rebuilds it from source:

```sh
CFLAGS="-Dfdopen=fdopen -Wno-error=implicit-function-declaration -Wno-error=incompatible-function-pointer-types" uv sync
```

uv caches the built wheel, so this is a once-per-machine fix.
</details>

<details>
<summary>Note: developing inside an iCloud-synced folder</summary>

iCloud's "Optimize Mac Storage" evicts file contents under `~/Desktop`,
replacing them with dataless stubs, and flags files hidden. Two concrete
failures this caused: Python 3.12 silently skips hidden `.pth` files (the
project vanished from `sys.path`), and cold imports took 60+ seconds while
evicted libraries re-downloaded. Mitigations in place:

- the real venv lives in `.venv.nosync/` (iCloud never syncs `*.nosync`),
  with `.venv` a symlink to it; `uv sync` works through the symlink
- large generated data goes to `data/raw.nosync/` for the same reason

If imports get slow or modules go missing, check for eviction:
`find <dir> -type f -flags +dataless | wc -l`. The durable fix is moving the
repo off the synced Desktop or disabling Desktop sync in iCloud settings.
</details>

## License

[MIT](LICENSE)

# PhysSplat

**A photo of everyday objects in. Interactive, learned 3D physics in your browser out.**

**Live demo: [physsplat.vercel.app](https://physsplat.vercel.app)** (physics runs entirely in your browser; drag a pencil to grab it, flick to poke)

PhysSplat turns a single photograph of simple objects on a flat surface (pencils today; the pipeline is object-agnostic) into a fully interactive 3D simulation. The scene is reconstructed to 3D, decomposed into rigid bodies without supervision, and simulated by a graph neural network trained entirely on synthetic data, with no hand-coded collision solver. You orbit the scene, grab and poke objects with the mouse, and watch them slide, pivot, and topple, with every physics step running client-side in the browser.

```
photo ──► single-image 3D    ──► unsupervised object          ──► learned GNN ──► interactive
          reconstruction         decomposition + particles        dynamics         three.js demo
          (TripoSR)              once per scene                   in-browser, custom WebGPU kernels
```

## Why

Generative 3D reconstruction produces photorealistic scenes that are physically inert — no mass, no contact, no gravity. Classical physics engines simulate rigid bodies robustly but require clean meshes and hand-tuned parameters, so they cannot ingest a photograph. PhysSplat bridges the two with a learned simulator that operates on surface point clouds: exactly the representation a photo reconstruction can provide.

## How it works

Three ideas carry the project:

1. **Rigidity by construction.** The GNN detects contact through particle-level message passing, but decodes one linear + one angular acceleration *per body* and integrates on SE(3), so objects can never deform or "melt" over long rollouts.
2. **Noise injection.** Training inputs are corrupted with small random-walk noise against clean targets, so the model learns to correct its own errors — the difference between rollouts that drift in seconds and rollouts that stay plausible indefinitely.
3. **One shared sampler.** The exact same fixed-spacing surface sampler dots synthetic training objects and photo-reconstructed objects with physics particles, making real scenes statistically indistinguishable from training data. That is the whole sim-to-real transfer mechanism.

Training data is manufactured, not collected: PyBullet simulates thousands of randomized scenes (stacks, scatters, drops, random pokes) and the network learns to imitate it — from a photo-compatible representation.

## What the network is not allowed to claim

A learned simulator run far outside its training distribution (photo-reconstructed pencils, not PyBullet capsules) will assert things that are not physics. Rather than hide that, the demo states the invariants explicitly and lets the diagnostics show when they bind:

- **Free flight.** A body touching nothing feels only gravity and the applied force, so its learned residual is zero. Without this a released pencil hovered: the network had never seen a motionless unsupported body.
- **No free energy.** Contact with static things cannot add mechanical energy, and an applied force adds only the work it does. Each step is trial-integrated and the residual scaled down if it would break that. Without this a resting pencil reared up to 89° on its own.
- **Support and pivot.** Support is counted only from below; a body whose centre of mass is outside its support polygon swings about the nearest support edge until it lies flat, instead of balancing on its end.

Everything these rules do is recorded per step and visible in the diagnostics, so a reader can always tell the model's answer from the correction ([`docs/diagnostics.md`](docs/diagnostics.md)). `scripts/evaluate.py --free-flight --energy` scores the model with the rules on, so their effect is measured, not assumed.

## Status

Early development, built in phases (each ends with a runnable artifact):

- [x] **Phase 0** — environment, package skeleton, shared constants
- [x] **Phase 1** — synthetic dataset generator (PyBullet → HDF5): 5,000 trajectories across 7 scene regimes with a poke-and-grab action channel, physics-invariant rejection filters, and visual audit tooling
- [x] **Phase 2** — the graph network simulator: predictive-edge contact graphs, per-body rigid decoder, analytic Newton-Euler integration, overfit gate passed
- [x] **Phase 3** — training on Apple Silicon (150k single-step steps), then a self-improving loop of physics-scored fine-tuning experiments: composite score 31.5 → 61.0, and 63.6 with the two analytic rules the same scorecard kept (`eval/REPORT.md`, `docs/eval-loop.md`)
- [x] **Phase 4** — extensive rollout scorecard (drift, penetration, rest stability, support-removal fidelity, energy) with per-regime breakdown, provenance ledger, and side-by-side films
- [x] **Phase 5** — local interactive demo (WebSocket server + three.js client, closed-loop spring grabs)
- [x] **Phase 6** — photo → scene pipeline: TripoSR reconstruction, RANSAC line segmentation of pencils, four real-photo scenes packeted
- [x] **Phase 7** — in-browser physics: race-free ONNX export, JS runtime parity-verified to microns against Python, then a custom WebGPU backend (hand-written WGSL kernels, ~7x faster than ONNX Runtime Web) so a 4-5 pencil scene steps at 12-16 Hz on an M-series laptop
- [x] **Phase 8** — public deployment on Vercel, plus a blind-tester loop: an independent agent plays the live demo, scores it harshly, and its findings drive the next round of fixes (`docs/demo-critic.md`). Five rounds so far; every fix it prompted is now covered by a headless regression test that drives the browser simulator with no page (`web/test/rest.mjs`, `web/test/interact.mjs`), and every analytic rule it prompted was scored on the synthetic test set before being kept or dropped.

## Documentation

- [`docs/PhysSplat.pdf`](docs/PhysSplat.pdf) — the design document: full architecture, the mathematics of the learned simulator, data spec, evaluation plan.
- [`docs/PhysSplat-Tutorial.pdf`](docs/PhysSplat-Tutorial.pdf) — a 70-step build guide with background theory, per-step checkpoints, and common traps.
- [`docs/diagnostics.md`](docs/diagnostics.md) — the demo's diagnostic API: per-step motion and contact data, anomaly detectors, and scripted probes that measure grabs, drops and support removal.
- [`docs/demo-critic.md`](docs/demo-critic.md) — the blind-tester ledger: what each round scored, what it measured, and what changed.

## Repository layout

```
src/physsplat/
  common/     shared constants + the particle sampler (used by datagen AND recon)
  datagen/    PyBullet scene generation, trajectory recording, HDF5 writer
  model/      graph builder, encode-process-decode GNN, SE(3) integrator
  train/      dataset, noise injection, training loop
  eval/       rollout metrics, comparison videos
  recon/      photo → 3D reconstruction → bodies → physics particles
  export/     ONNX export + Python/browser parity tests
scripts/      runnable entry points (one per task)
web/          static three.js app + client-side physics (js/sim.js, js/gpu_net.js)
web/test/     parity, end-to-end, diagnostics, rest and interaction suites (npm test)
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

# PhysSplat

![Left: a photograph of four BIC Matic Grip mechanical pencils lying in a lattice on a table. Right: the same four pencils reconstructed to 3D and simulated in the browser, from the same viewpoint.](docs/img/real-vs-sim.png)

**A learned, action-conditioned world model of rigid-body physics, built from a single photograph and run in the browser.**

### [Open the live demo →](https://physsplat.vercel.app)

[![live demo](https://img.shields.io/badge/demo-physsplat.vercel.app-2ea44f)](https://physsplat.vercel.app)
[![Python 3.12](https://img.shields.io/badge/python-3.12-3776ab)](pyproject.toml)
[![PyTorch](https://img.shields.io/badge/model-PyTorch-ee4c2c)](src/physsplat/model)
[![WebGPU](https://img.shields.io/badge/runtime-WebGPU-005a9c)](web/public/js/gpu_net.js)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

PhysSplat is not a physics engine. There is no collision detection, constraint
solver, or contact model written by hand. The dynamics of the scene are
predicted by a graph neural network that learned contact, friction, sliding,
pivoting and toppling from data, and that predicts each next state from the
current state and the user's action.

The input is one photograph of objects on a flat surface. The output is an
interactive 3D scene in which every physics step is a forward pass of the
network, executed on the viewer's GPU through hand-written WebGPU compute
kernels. There is no server in the loop.

Drag a pencil to grab it, flick to poke, and scroll or drag empty space to orbit.

## The world model

The system learns a transition function

```
s_{t+1} = f_θ(s_t, a_t)
```

| term | definition |
|---|---|
| state `s_t` | every object as a cloud of surface particles, with body pose, a five-step velocity history, mass and inertia |
| action `a_t` | a force applied to a body at a 3D point (a grab or a poke) |
| model `f_θ` | an encode-process-decode graph neural network over a contact graph |

The model is run autoregressively: each prediction becomes the next input,
closed-loop against live user input, for as long as the scene is open.

## How it works

```
photo ──► single-image 3D ──► object          ──► particle  ──► learned world model ──► interactive
.jpg      reconstruction      decomposition       sampler       (GNN, WebGPU)            3D scene
          (TripoSR)           (unsupervised)
          └──────────────── once per scene ────────────────┘    └──── every frame, in the browser ────┘
```

**1. Synthetic data.** PyBullet generates 5,000 randomized trajectories across
seven scene regimes (stacks, scatters, drops, pokes, grabs), with randomized
density, friction and restitution. Trajectories that violate physical
invariants are rejected. The simulator is used only to produce training data;
it is not present at inference.

**2. Representation.** A single fixed-spacing surface sampler converts every
object, synthetic or reconstructed from a photo, into physics particles. Real
scenes are therefore expressed in exactly the representation the model was
trained on. This shared sampler is the sim-to-real transfer mechanism.

**3. Contact graph.** At each step, particles within a contact radius are
connected, including at their velocity-extrapolated positions so that fast
approaches are detected one step early. Edges between different bodies carry
contact. The model sees only relative displacements, never absolute positions.

**4. Graph network.** MLP encoders lift node and edge features to 128
dimensions; ten residual message-passing blocks propagate contact information
through the graph; the decoder pools latents per body and outputs one linear
and one angular acceleration per object. Rigidity is exact by construction:
objects cannot deform over a rollout.

**5. Integration.** Gravity is applied analytically and the network predicts
only the contact and friction residual. Bodies are integrated on SE(3) with a
differentiable semi-implicit Euler step.

**6. Training.** 150,000 single-step iterations on Apple Silicon (PyTorch,
MPS), with random-walk noise injected into inputs against clean targets so the
model learns to correct its own drift. This is followed by rollout fine-tuning
that backpropagates through up to 24 unrolled steps.

**7. Photo to scene.** TripoSR reconstructs the photograph in 3D; RANSAC line
segmentation decomposes it into individual bodies without supervision; each
body is canonicalized and resampled into particles. The photograph's camera is
recovered by matching rendered silhouettes to the image, so the simulated
scene can be overlaid on the original photo.

**8. Browser inference.** The model is exported to ONNX and reimplemented as
five hand-written WGSL compute kernels in a single WebGPU pass. The JavaScript
graph builder, integrator and network are verified against PyTorch to micron
precision. A four-to-five body scene steps in 15 to 33 ms. ONNX Runtime Web's
WebGPU provider measured 190 to 490 ms on the same model; it remains as the
fallback where WebGPU is unavailable.

## Evaluation

Single-step loss does not measure a world model. The model is scored on long
rollouts over a held-out synthetic test set:

| failure mode | metric | best run |
|---|---|---|
| drift | translation error after 150 steps | 1.9 cm |
| interpenetration | worst surface penetration | 1.50 mm |
| rest stability | fraction of settled scenes that remain settled | 0.83 |
| support structure | contact set vs ground truth (Jaccard) | 0.60 |

An automated improvement loop proposed and scored fine-tuning experiments,
raising the composite score from 31.5 to 61.0. Every analytic rule applied
around the network was then accepted or rejected on the same scorecard:

| configuration | composite | decision |
|---|---|---|
| learned model alone | 61.0 | baseline |
| + free flight (zero residual for a body touching nothing) | 62.3 | kept |
| + angular contact fade (torque vanishes as a body separates) | 63.6 | kept |
| + no free energy, every step | 42.4 | rejected |
| + two-tap residual mean | 34.2 | rejected |

Every correction is recorded per step, so the model's prediction is always
distinguishable from a rule's ([`docs/diagnostics.md`](docs/diagnostics.md)).
The live demo was additionally reviewed over multiple rounds by independent
blind testers, with every fix pinned by a headless regression test
([`docs/demo-critic.md`](docs/demo-critic.md)).

## Classical baseline

The demo includes [Rapier](https://github.com/dimforge/rapier), a conventional
rigid-body solver compiled to WebAssembly, behind `?engine=rapier`. Both engines
share one interface and are measured on the same sensor checks:

| behaviour | learned world model | Rapier |
|---|---|---|
| lone pencil dropped on the desk | settles within 5° of flat | 0.5° |
| pencil held still at its centre | 3 to 35° of tilt | 0.2 to 3.8° |
| end grab lifted clear | dangles | dangles |
| 250 mm drop onto the pile, worst overlap | 0.1 to 3.4 mm | 0.1 to 1.0 mm |
| cost per 1/60 s step | 15 to 33 ms (WebGPU) | 0.5 ms (WASM) |

The solver is programmed with the laws of contact; the world model acquired
them from 5,000 trajectories.

## What this project demonstrates

- **World modelling:** an action-conditioned dynamics model, stable under long autoregressive rollout and closed-loop interaction.
- **Geometric deep learning:** relational inductive biases, per-body rigid decoding, SE(3) integration, translation-invariant features.
- **Sim-to-real transfer:** a model trained only on synthetic data, applied to objects reconstructed from a real photograph.
- **3D computer vision:** single-image reconstruction, unsupervised object decomposition, camera pose recovery.
- **ML systems:** synthetic data generation at scale, training on Apple Silicon, ONNX export, custom GPU kernels for in-browser inference.
- **Evaluation:** rollout-level metrics, an automated experiment loop with a provenance ledger, and ablations for every rule.

## Interface

![The PhysSplat interface: a top bar with four scene thumbnails and the engine badge; five simulated pencils on the photographed desk with a label over each one and yellow dots where they touch; a card for the selected pencil reading at rest, 0.5° from flat, on the desk, resting on the desk, carrying pencil 3; the Display tab of the dock with its toggles; the transport bar reading Rapier, 1.2 ms per step, real time.](docs/img/ui.png)

The top bar selects a scene and switches the engine. The dock provides
playback controls (**Play**), camera views including the photograph's own
viewpoint (**Camera**), live physical parameters for the baseline solver
(**Physics**), overlays for contacts, velocities and an overlay of the real
photograph (**Display**), and a live contact report (**Sensors**). Selecting a
pencil shows its state, tilt, support and load. `?` lists the shortcuts.

## Documentation

| document | contents |
|---|---|
| [`docs/PhysSplat.pdf`](docs/PhysSplat.pdf) | design document: architecture, mathematics of the learned simulator, data specification, evaluation plan |
| [`docs/PhysSplat-Tutorial.pdf`](docs/PhysSplat-Tutorial.pdf) | 70-step build guide with background theory and per-step checkpoints |
| [`eval/REPORT.md`](eval/REPORT.md) | the scorecard, with the decision for every experiment |
| [`docs/diagnostics.md`](docs/diagnostics.md) | the demo's diagnostic API: per-step motion and contact data, anomaly detectors, scripted probes |
| [`docs/demo-critic.md`](docs/demo-critic.md) | the blind-tester ledger |

<details>
<summary><b>Repository layout</b></summary>

```
src/physsplat/
  common/     shared constants + the particle sampler (used by datagen AND recon)
  datagen/    PyBullet scene generation, trajectory recording, HDF5 writer
  model/      graph builder, encode-process-decode GNN, SE(3) integrator
  train/      dataset, noise injection, training and rollout fine-tuning
  eval/       rollout metrics, scorecard, provenance ledger
  recon/      photo → 3D reconstruction → bodies → physics particles
  export/     ONNX export
scripts/      runnable entry points (one per task)
web/          three.js app + in-browser inference (js/sim.js, js/gpu_net.js)
web/test/     parity, end-to-end, diagnostics, rest and interaction suites (npm test)
docs/         design document and build tutorial (LaTeX + PDF)
```

</details>

<details>
<summary><b>Build phases</b></summary>

- [x] **Phase 0:** environment, package skeleton, shared constants
- [x] **Phase 1:** synthetic dataset generator (PyBullet → HDF5), 5,000 trajectories, 7 regimes, action channel, invariant-based rejection filters
- [x] **Phase 2:** graph network simulator, predictive contact edges, per-body rigid decoder, SE(3) integration
- [x] **Phase 3:** training on Apple Silicon and an automated fine-tuning loop, composite 31.5 → 63.6
- [x] **Phase 4:** rollout scorecard with per-regime breakdown, provenance ledger, comparison films
- [x] **Phase 5:** local interactive demo with closed-loop spring grabs
- [x] **Phase 6:** photo → scene pipeline: TripoSR, unsupervised segmentation, four real-photo scenes
- [x] **Phase 7:** in-browser inference: ONNX export, micron-level parity, custom WebGPU kernels
- [x] **Phase 8:** public deployment, blind-tester review loop, classical baseline

</details>

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Node 20+.

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
replacing them with dataless stubs and flagging files hidden. Python 3.12
silently skips hidden `.pth` files, and cold imports can take 60+ seconds.
The real venv lives in `.venv.nosync/` with `.venv` a symlink to it, and large
generated data goes to `data/raw.nosync/`. To check for eviction:
`find <dir> -type f -flags +dataless | wc -l`.
</details>

## License

[MIT](LICENSE)

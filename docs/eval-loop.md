# Evaluation and the self-improving loop

How PhysSplat measures its learned physics, and how it improves itself
without a human in the loop.

## 1. The scorecard (`physsplat/eval/scorecard.py`)

One checkpoint in, one deterministic scorecard out. Always the same
held-out trajectories in the same order, so any two records are directly
comparable. Every metric is a physics oracle: derived from ground truth or
conservation laws, never from the code being tested.

| metric | what it measures | why it matters for the demo |
|---|---|---|
| `trans_{50,150,294}` | body drift vs. PyBullet at three horizons | does the motion match reality, short and long term |
| `axis_{...}` | capsule axis-direction drift, spin-invariant | orientation error that is actually visible on a pencil |
| `surface_pen` | worst capsule-capsule overlap (analytic) | pencils must not pass through each other |
| `ground_pen` | worst below-floor excursion | nothing sinks |
| `explosion` | left scene bounds or exceeded the speed cap | rollouts must stay bounded |
| `stability` | settled scenes stay still until the first action | piles must not creep or fall on their own |
| `post_action_err` | drift 60 frames after the last action | pokes and grabs produce the right response |
| `support_jaccard` | who fell when a support was pulled, model vs. truth | *the* demo moment, scored |
| `energy_growth` | energy gained over passive windows | no free energy |
| `photo_drift` | passive drift of the real photo scenes | rest stability where it matters most |

Aggregates are reported overall and **per regime** (pile, crosshatch,
pyramid, bundle, scattered, stack, drop), so a change that helps boxes and
hurts pencils is visible.

### Composite score
A single 0-100 number the loop optimizes, weights in `COMPOSITE_WEIGHTS`:
stability 0.25, trans_150 0.15, surface_pen 0.15, support_jaccard 0.15,
axis_150 0.10, post_action_err 0.10, no-explosion 0.10. Distances are
scaled linearly to zero at a "clearly broken" threshold (5 cm drift, 5 mm
overlap, 0.5 rad axis error). The weights encode the demo's priorities:
things staying put and pencils not interpenetrating outrank millimetre
tracking accuracy.

## 2. Provenance: the ledger (`physsplat/eval/ledger.py`)

`eval/ledger.jsonl` is append-only. Each record carries the checkpoint,
training step, git commit, the **change** relative to its parent, the full
scorecard, per-metric **deltas**, and for experiments the ACCEPT/REJECT
decision with a reason. `eval/REPORT.md` is regenerated from it after
every record: leaderboard, trend table in evaluation order, decision log,
and the leader's per-regime breakdown. This is what answers "what did I
change, and what did it do?" at a glance, and it is what the agent reads
between loop runs.

## 3. The loop (`scripts/improve.py`)

```
best <- base checkpoint; evaluate -> baseline record
repeat (budget):
    change <- next candidate not already rejected from `best`
    child  <- fine-tune a COPY of best with `change` for N steps
    score  <- scorecard(child)              # same scenes as every record
    if score.composite > best.composite + margin: ACCEPT, best <- child
    else: REJECT (remember: this change failed from this parent)
    ledger.append(change, deltas, decision, reason); regenerate REPORT.md
```

Candidates are small, physically motivated knobs, each with a rationale:
stronger or weaker noise injection (correction pressure vs. target
blur), unrolled 4- and 8-step rollout fine-tuning (penalize the drift the
model actually produces), angular-loss down-weighting (let linear contact
refine), and a low-lr polish. Rejections from a parent are never retried
from that same parent, so the loop cannot spin on a dead end; it stops when
every candidate has been rejected from the current best.

Safety rails: experiments train with `workers=0` so they can coexist with
the main run in unified memory; evaluation is deterministic; the margin
(0.5 composite points) stops noise-level "improvements" from being accepted.

## 4. How the agent uses it

- During long training, validation rollout errors stream to the monitor;
  the 50k / 150k / final reviews run the full scorecard plus side-by-side
  films, and the review reads the trend rather than a single number.
- After training, `improve.py` runs with a budget; the agent reads
  `REPORT.md`, adds or retires candidates based on what moved which metric,
  and runs again. The accepted best is what gets exported to ONNX.
- Regressions are unmissable: every record sits in the trend table next to
  its predecessors, per regime.

# Demo diagnostics

The live demo carries a transparent diagnostic layer so that anyone testing
it, a person or an agent, reads what the pencils are doing from numbers
instead of inferring it from pixels. Everything below is computed from the
physics state, the capsule proxies and the per-step record the simulator
keeps (model residual, external accelerations, guard corrections). Nothing
is read from the screen.

Open the demo, press **D** (or add `?diag=1` to the URL) for the live panel.
In the browser console (or through any automation that can evaluate
JavaScript on the page) the same data is under `physsplat`.

Units are SI unless the field name ends in `_mm`, `_deg`, `_ms`, `_uJ`.
Quaternions are `[x, y, z, w]`. Body ids index `packet.bodies`.

## Reading state

| Call | Returns |
|---|---|
| `physsplat.state()` / `physsplat.diag.snapshot()` | the current frame with rounded numbers: per body position, axis, speed, angular speed, height, lowest surface point, elevation of the axis from horizontal, contacts with gaps, support relations, penetration, resting status and drift since coming to rest, kinetic energy, the model's predicted acceleration, and what each guard did this step; plus scene totals, timing, and active anomalies |
| `physsplat.diag.history(n, fields?)` | the last `n` frames (newest last), compact; `fields` limits per-body keys, e.g. `["pos","speed","contacts"]` |
| `physsplat.diag.track(body, n)` | time series for one body: `z`, `lowest_mm`, `vz`, `speed`, `angSpeed`, `elevation_deg`, `penetration_mm`, `contacts`, `resting`, `model_lin_z`, `guard_mm`, `free`, `settled` |
| `physsplat.diag.check()` | anomalies active right now, each with type, severity, body, duration in steps, peak value, message |
| `physsplat.diag.events(n, type?)` | the event log: scene loads and resets, pointer grabs and pokes, probe actions, anomaly episodes starting and ending |
| `physsplat.diag.summary(n)` | aggregates over the last `n` steps: per body displacement, rotation, max speed, max penetration, lowest point, fraction of steps in ground contact or at rest, total guard correction, final contacts and resting gap; anomaly counts; mean step time |
| `physsplat.diag.export({last})` | one JSON string with the scene, body parameters, load corrections, events, summary and frames |

The simulator's own per-step record is at `physsplat.sim.last`
(`residual` in SI accelerations per body, `ext`, `action`, `guard`), the raw
state at `physsplat.sim.state`, and `physsplat.paused = true` /
`physsplat.stepOnce = true` freeze and single-step the physics.

## Anomaly detectors

Run on every frame. An episode starts when a detector first fires and ends
when it stops; both are logged as events with duration and peak.

| key | fires when |
|---|---|
| `explosion:<b>` | speed > 3 m/s or angular speed > 100 rad/s |
| `sinking:<b>` | a surface particle more than 1 mm below the floor (error above 3 mm) |
| `floating:<b>` | resting for 20 steps, no contact, lowest point above 6 mm, no action on it in the last 20 steps |
| `tip_balance:<b>` | axis more than 60 degrees from horizontal, touching the floor, still for 30 steps |
| `tilt_hold:<b>` | axis more than 4 degrees from horizontal, one end on the floor, nothing else touching it, still for 30 steps (a pencil cannot rest like that) |
| `creep:<b>` | classified at rest for 60 steps yet drifted more than 2 mm from where it stopped |
| `creep_rot:<b>` | same, rotated more than 3 degrees |
| `penetration:<i>-<j>` | capsule overlap deeper than 2 mm (error above 5 mm) |
| `jitter:<b>` | vertical velocity changed sign more than 8 times in 30 steps at amplitudes above 15 mm/s (a visible tremble) |
| `unbalanced_rest:<b>` | held still for 30 steps with its centre of mass outside its support polygon |
| `rearing:<b>` | rotating upward past 45 degrees with one end on the floor, faster than 0.5 rad/s, with no action in 30 steps |
| `spontaneous_motion:scene` | kinetic energy rising for 10 steps with no action for 90 steps |

Contacts use a 3 mm gap tolerance between capsule surfaces because the
model resolves contact at particle level (6 mm contact radius) and settles
stacked bodies a millimetre or three apart. The resting gap is reported so
it can be judged rather than hidden. `support` uses the model's full 6 mm
contact zone and counts only contacts from below (the neighbour's contact
normal points up) plus floor contacts; `support.balanced` says whether the
centre of mass projects inside that support's convex hull (6 mm
tolerance), `support.dist_mm` how far outside it is. `resting` is the
simulator's own settle verdict, so `creep` means a held body moved.

## Probes: scripted experiments

Probes drive the same grab and poke code paths as the mouse, sample the
diagnostics every step, and return a report with measurements. They are
`async`, take real time, and by default reset the scene first so results
are reproducible. Grab points are `"center"`, `"end"`, `"tip"`, a number in
[-1, 1] along the axis, or a body-frame `[x, y, z]`.

```js
await physsplat.probe.rest({steps: 180})                 // does anything move untouched?
await physsplat.probe.lift(2, {height: 0.05})            // grab, raise 5 cm, hold, release
await physsplat.probe.grab(2, {at: "end", delta: [0.06, 0, 0]})   // off-center drag
await physsplat.probe.poke(2, {dir: [1, 0, 0], dv: 0.3}) // trained impulse poke
await physsplat.probe.flick(2, {dir: [1, 0, 0], speed: 0.4, distance: 0.05}) // what a mouse flick does
await physsplat.probe.pullBottom()                       // pull the most load-bearing pencil out
await physsplat.probe.drop(2, {height: 0.05})            // teleport up, watch it fall
await physsplat.probe.all()                              // the whole battery, ~1 min
```

If your harness cannot wait on a promise for that long, use the
non-blocking runner:

```js
const id = physsplat.run("lift", 2, {height: 0.05});    // returns immediately
physsplat.report(id)        // {status: "running"|"done"|"error", result}
```

What the reports contain:

- **rest**: per-body displacement, rotation, guard totals, resting fraction, anomalies, and a verdict (`still` or which bodies moved).
- **grab / lift / pullBottom**: tracking error between the cursor target and the grab point (mean, max, final, requested vs achieved displacement), the body's pose at release, how far every other body moved, and an `after_release` block: fitted free-fall acceleration and its ratio to g, contact onset, impact speed against the analytic value, bounces, lowest point reached, time to rest, final contacts, plus a 40-step series of height, vertical velocity, model residual and guard flags. `pullBottom` adds, for each body the pulled pencil was supporting, how far it dropped and what it now rests on.
- **poke**: peak speed and angular speed against the requested velocity change, stop time, displacement, rotation, how far neighbours moved. The trained 3-step impulse barely moves a pencil on a pile (the learned friction eats it: 0.3 m/s gives 3 mm), so the mouse flick does not use it.
- **flick**: the grab point is pushed along the gesture at the cursor's speed for a short distance and let go, the same path the mouse uses; reports when it released, peak speed, travel, rotation, stop time. A 0.3 m/s flick travels 13 mm, 0.5 m/s travels 29 mm.
- **drop**: the `after_release` block for a body released from rest in mid-air.

## Guards and the model

The learned model does the contact physics. Three analytic rules clean up
after it in the demo (never during evaluation), and every intervention is
recorded per step in `guard`:

- **free flight**: a body with no edge to another body and no particle within the contact radius of the floor gets a zero residual; only gravity and the applied force act on it. Without this, a pencil released in mid-air hovered (the network never saw a motionless unsupported body in training). The same applies to a body that touches things but has nothing below it (only pencils on its back): it falls, and they come down with it.
- **pivot**: a body whose centre of mass is not over its support (more than 7 mm outside the support polygon, with hysteresis back to 4 mm) is integrated as a pendulum about the hinge, the nearest point of the support polygon's boundary (an edge between two contacts, or a lone contact): gravity's torque over the inertia about the hinge, model residual dropped, centre of mass moving with omega x r; velocities are cut to 0.2 when the swing lands on a new support. Without this a pencil that landed on its end was held 9-14 degrees up with the other end in the air, and pencils balanced on their tips (the round-1 tester's complaint).
- **ground / capsule guards**: residual overlap with the floor or another capsule is removed and the approaching velocity cancelled.
- **settle**: a supported body that has been slow (under 3 cm/s, 0.6 rad/s) for 15 steps is held exactly still (pose restored) until something acts on it; slow motion on a support is damped by half each step first, because the model's contact response rings for a second after a landing. Zeroing velocity alone left a slow sideways creep driven by the guards. At the moment a body settles, a gap of up to 6 mm to the floor or to the nearest capsule is closed once, because the model's contact response equilibrates 2-5 mm above whatever it landed on.
- **no free energy**: contact with static things cannot add mechanical energy to a pile; only an applied force can, and only as much work as it does. Each step is trial-integrated, and if the learned residual would raise the scene's kinetic plus gravitational energy by more than the action's work (plus 0.5 uJ, about 0.5 mm/s of lift, for pushing out of overlaps), the residual is scaled down until it does not (`guard.energyScale`, `totals.energyGain_uJ`). Without this the top pencil of IMG_8596 reared up to 89 degrees on its own, twice in ten seconds, and balanced on its end.
- **caps**: 3 m/s and 60 rad/s; nothing in a pencil pile moves faster, and a runaway must not leave the table.
- **at load**: overlaps left by the single-view reconstruction are resolved by lifting the upper body of each overlapping pair straight up (`load_correction.lift_mm` in the scene event), never sideways, and the first 24 physics steps run before anything is drawn, so the pile appears already settled. Grabs attach on the pencil's axis, whatever the cursor hit, so the spring cannot torque the pencil about its own axis.

Each rule sets a flag in the per-step `guard` record (`freeFlight`, `pivot`, `settled`) so a report can always say whether the model or a rule produced a motion.

## Why this exists

The first blind test of the demo produced impressions ("feels wrong",
"pencils phase through each other") that took hours to trace. The first
scripted probe found, in one run, that a released pencil never fell, that
resting pencils crept 9 mm in four seconds, and that a held pencil sagged
12.5 mm below the cursor, each with the mechanism visible in the numbers.
`docs/demo-critic.md` keeps the ledger of rounds.

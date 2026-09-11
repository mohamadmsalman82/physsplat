# Sensors: god mode for the demo

`physsplat.sensors` in the browser (or `new Sensors(sim)` from
`web/public/js/sensors.js` headless) answers the question a strange-looking
pencil actually raises: **what is touching what, where on each pencil, right
now, and is anything moving that should not be.**

It exists so nobody has to infer physics from pixels. Every reading is
derived from the same state the simulator integrates and the same pencil
profile the contact guard separates bodies with, so a sensor reading cannot
disagree with the physics.

## Why the readings can name parts of a pencil

Every body is the same canonical BIC Matic Grip (`src/physsplat/common/pencil.py`,
`web/public/js/pencil.js`): a radius profile from the eraser (t = 0) to the
lead (t = 1). A contact has a position along the pencil, that position is a
fraction of its length, and the profile says what is there. So a contact is
reported as *"body 2's grip is resting on body 3's cone, 41 mm from body 3's
point"*, not as *"bodies 2 and 3, overlap 1.4 mm"*.

| region | t from | t to | what it is |
|---|---|---|---|
| eraser | 0.000 | 0.020 | white eraser standing out of the cap |
| cap | 0.020 | 0.065 | the collar the eraser sits in |
| barrel | 0.065 | 0.740 | 9 mm barrel, carries the clip |
| grip | 0.740 | 0.873 | 11 mm rubber grip, the fattest part |
| lower barrel | 0.873 | 0.929 | barrel between grip and cone |
| cone | 0.929 | 0.984 | plastic cone, body-coloured |
| point | 0.984 | 1.000 | lead sleeve and lead |

## API

All methods are synchronous and return plain objects. Millimetres, degrees,
mm/s and rad/s throughout, rounded to two or three places.

### `now()`

Everything, this step, as one object:

```js
{
  step, steps_since_action, scene_at_rest,
  bodies: [ body(0), body(1), ... ],      // see body()
  contacts: touch(),                       // see touch()
  swept: { maxDisp, K, hits, t } | null,   // what the swept collision check did
}
```

### `report()`

The same thing as text, for reading. Flags `THROUGH THE TABLE`, `NOTHING
UNDER IT` and `OVERLAPPING n mm` in capitals and appends recent problems from
`events()`.

```
step 60  (whole scene at rest)
body 0: still (settled), 3.67 deg from flat, floor gap 0 mm on its grip
   touches table: its grip (22.1 mm from its point) against the table
   1.747 mm from 4: its grip (22.6 mm from its point) against their barrel
body 4: still (settled), 1.92 deg from flat, floor gap 9.915 mm on its barrel  *** NOTHING UNDER IT ***
```

That last line is a real reading from before the seating rule existed, and
it is the hovering a player had reported.

### `body(b)`

One pencil in full:

| field | meaning |
|---|---|
| `pos_mm`, `quat` | pose |
| `point_world_mm`, `eraser_world_mm` | where its two ends are |
| `elevation_deg` | angle of its axis above the table |
| `speed_mms`, `spin_rads`, `moving` | motion, with `moving` thresholded at 0.5 mm/s or 0.02 rad/s |
| `resting`, `rest_steps`, `held_still` | the settle rule's view of it; `held_still` means the rule is pinning its pose |
| `floor` | `{gap_mm, touching, below_table_mm, at, world}` from the analytic surface |
| `touching_bodies`, `resting_on`, `carrying` | contact graph from this body's point of view |
| `deepest_pen_mm` | worst overlap it is part of |
| `unsupported` | true if it touches neither the table nor anything below it |
| `contacts` | its rows of `touch()` |

### `touch()`

Every pencil-to-pencil and pencil-to-table contact within 6 mm:

| field | meaning |
|---|---|
| `bodies` | `[i, j]` or `[b, "table"]` |
| `touching` | gap within 0.5 mm |
| `gap_mm`, `pen_mm` | separation, or overlap if positive |
| `on_i`, `on_j` | `{t, region, mm_from_point, mm_from_eraser, radius_mm}` for each body at the contact |
| `normal` | unit vector from j toward i |
| `upper`, `lower` | which body is on top at this contact, from the normal; `null` if side by side |
| `world` | the contact point, mm |
| `closing_mms` | relative speed along the normal, positive when approaching |
| `sliding_mms` | relative speed across the contact |

### `surface(b)`

Body b's length as a strip of regions, each listing what touches it there
and whether that neighbour is being carried, resting on it, or beside it.
The direct answer to *"is anything on its tip"*.

### `sample(actInfo)`

Records one step. The physics loop calls it after every `sim.step()`; a
headless harness must call it itself. `actInfo` is `{kind, body}` for a
grab, poke or flick, or `null`.

### `history(n)`, `track(path, n)`

The last n recorded steps, compact, oldest first; or one number over time,
e.g. `track("bodies.2.speed_mms", 300)` or `track("worst_pen_mm")`.

### `events({n})`

Things that should not be happening, found by reading the recorded steps:

| kind | fires when |
|---|---|
| `through the table` | any body's analytic surface below z = 0 by more than 0.05 mm |
| `hovering, nothing under it` | floor gap over 1 mm, no touching contact, slow, no action |
| `moving with nothing touching or driving it` | over 2 mm/s more than 30 steps after the last action |
| `surfaces overlapping` | any contact deeper than 1 mm |

Each event names the body or pair, the step range, the peak, where on each
pencil it happened, and whether it is still going on.

### `watch(b)`, `log`, `diff(stepA, stepB)`

`watch(b)` records every change in body b's contact set into `sensors.log`
as `{step, body, was, now}`. `diff(a, b)` says what moved and which contacts
started or stopped between two recorded steps.

## Reading it from a harness

```js
const P = window.physsplat;
P.sensors.report();                              // text, right now
P.sensors.body(2).floor;                         // {gap_mm: 0, touching: true, at: {region: "grip", ...}}
P.sensors.surface(2).find(s => s.region === "point").contacts;   // what is on its tip
P.sensors.events({n: 600});                      // recent problems
P.sensors.track("bodies.2.floor_mm", 120);       // was it hovering, and when
```

`physsplat.paused = true` stops the page's own loop so a harness can drive
`sim.step()` by hand; `sample()` still has to be called per step in that
case. Do not drive the simulator from two places at once: two loops calling
`step()` interleave and every reading becomes meaningless.

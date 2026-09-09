# Blind-tester loop for the live demo

The scorecard in `eval/` measures the model against PyBullet. It cannot tell
whether the demo *feels* right to a person. So the last stage of the project
is a second loop: an independent agent that has never seen the code plays
the live site in Chrome, scores it harshly, and lists what is wrong with
repro steps. Every round gets a fresh tester so earlier fixes are not graded
by the critic who asked for them. Rounds with a genuinely low score include a
deep dive: the tester uses the page's debug handle (`physsplat.paused`,
`physsplat.stepOnce`, `physsplat.sim.state`) to measure the failure and
argue a cause, rather than only describe a symptom.

Scores are 1-10. This file is the ledger of what each round found, what was
changed in response, and what the next round said about it.

## Round 1 (2026-09-08, first public build)

| Visuals | Physics | Interaction | Overall |
|---|---|---|---|
| 3 | 1 | 2 | 2 |

Worst findings, in the tester's order:

1. Switching scenes crashed the physics loop (`syncTransforms` on the
   rebuilt groups) and the page needed a reload.
2. Pencils at rest crept and slowly sank; nothing was ever fully still.
3. A dragged pencil could end up balanced on its tip indefinitely.
4. Pencils passed through each other and through the floor.
5. A pencil could not be lifted; the grab spring stretched instead.
6. Flicking did nothing recognisable as a poke.
7. Orbiting the camera disturbed the pencils (the drag also grabbed them).
8. No zoom; the orbit was hypersensitive; reset did not restore the camera.
9. The HUD was translucent over the scene and hard to read.
10. The initial pile jolted on load.
11. Reconstruction surfaces were lumpy and full of holes.

What changed in response (commits `55731ff`, `4d26320`):

- Loading guard on scene switch; the loop never exits on an error.
- Settle damping for bodies lying on the table; ground guard and
  capsule-capsule guard for residual penetration, both restricted to lying
  bodies so a standing pencil is left to gravity and the model.
- Stiffer grab spring (omega 28 vs 12.6 in training), a 260 ms flick window
  for pokes, a 2.2x pick volume.
- Orbit disabled while grabbing; camera distance and elevation limits;
  gentler rotate speed; zoom; reset restores the home camera.
- Opaque HUD.
- Overlaps resolved at load (up to 12 mm from single-view depth error).
- Each body rendered as a capsule fitted to its reconstruction, colored
  from the nearest reconstruction vertex.
- Home camera azimuth chosen per scene so no pencil points at the camera:
  a pencil seen end-on foreshortens into what looks like a pencil standing
  on its tip, and that illusion read as a physics bug.

## Round 2 (2026-09-08): instrumented

The second blind tester was lost to an API session limit before it could
report, so this round is the diagnostic layer's first run instead
(`docs/diagnostics.md`): scripted probes on the live build, scene IMG_8596,
before and after the fixes they motivated. Numbers are what the probes
returned, not impressions.

| probe | before | after |
|---|---|---|
| rest, 180 steps | bodies 1-3 moved 6-9 mm and rotated up to 13 degrees; guard corrections 11-13 mm; one body ended 4.8 mm above the floor touching nothing | all bodies within 0.7 mm and 0.4 degrees; resting gaps 0-0.8 mm; no anomalies |
| lift 5 cm, hold, release | held pencil sagged 12 mm below the cursor; after release it hovered at 53 mm, "unsupported, resting", never fell; neighbours shifted 3-9 mm | tracks the cursor within 1.7 mm (max 2.7); free-fall acceleration 1.00 g over the free steps; contact response begins 0.10 s in; impact 0.82 m/s vs 0.96 analytic; one small bounce; at rest 0.37 s after release on the pencil beneath; neighbours moved under 0.6 mm |
| drop from 5 cm | (not run) | free-fall ratio 1.00, contact onset 0.083 s, no bounce, at rest 0.43 s later |
| pull the load-bearing pencil out 8 cm | (not run) | pencil follows within 10 mm mean during the pull; the two pencils it carried drop 10.5 mm and 20.1 mm and come to rest on the floor pencil |
| off-center drag 6 cm | (not run) | 12 mm mean lag during the move, 60.4 mm achieved; pencils underneath dragged 9-10 mm by friction |
| flick, dv 0.2 m/s | (not run) | peak 0.062 m/s, 0.6 mm travel, 2.6 degrees turn. A 0.2 m/s flick over 3 steps is a 4 m/s^2 push, below the friction threshold of a pencil on a pile, so PyBullet would barely move it either; the UI flick reads weak for that reason |

What the probes exposed, and the fixes (commit after `6668ee3`):

1. **A released pencil hovered.** The model never saw a motionless
   unsupported body in training (bodies at rest are always supported) and
   answered "+g, hold still" for one. Fix: free-flight rule. A body with no
   contact edge and no particle near the floor gets a zero learned
   residual; only gravity and the applied force act on it. Measured
   free-fall acceleration after the fix: 1.00 g.
2. **Resting piles crept.** The model under-supports a resting pencil by
   0.15-0.36 m/s^2 (its residual reads 9.45-9.66 against g = 9.81), so
   bodies sank into their supports; the capsule guard pushed them back out
   along the contact normal, and that position-only push crept the pile
   sideways at about 2 mm/s with zero velocity. Fix: settle now holds a
   supported slow body exactly still (pose restored), for bodies resting on
   other bodies as well as on the floor. Rest probe: 9 mm drift to 0.7 mm.
3. **Held pencils sagged 12.5 mm** (g / omega^2 for the grab spring). Fix:
   gravity feed-forward in the grab force, still under the trained cap.
4. Capsule guard ran after the ground guard and could push a body 1 mm into
   the floor; order swapped.
5. **A pencil that landed on its end stayed tilted**, 9-14 degrees up
   with the far end in the air and nothing under it: the model reads the
   one floor contact as support (residual +9.5) and settle then locked the
   pose. Fix: pivot rule. A body whose only support is one region of the
   floor while tilted more than 3 degrees is integrated as a pendulum about
   that contact until it lies flat (with 0.2 restitution at the far end's
   impact). Measured: 24.7 to 0.4 degrees in 5 steps, flat and at rest in
   0.30 s, no bounce-back. This is the analytic answer to the round-1
   "tip balancing" complaint as well. Detector `tilt_hold` now flags any
   case that slips through.
6. **Landed pencils hovered 2-5 mm** above whatever they landed on (the
   model's contact equilibrium), and settle locked the gap. Fix: the gap
   is closed once when the body settles. Measured: 1.9 mm to 0.0 mm.

Model-quality numbers worth keeping an eye on: the residual shortfall at
rest (0.15-0.36 m/s^2), a 2-3 mm gap before contact response engages, and
the 0.82 vs 0.96 m/s impact speed (contact begins a little early). All
three point at the contact residual on reconstructed pencils and are the
candidates for a fine-tuning round with photo-packet geometry.

The free-flight rule was also scored on the synthetic test set
(`scripts/evaluate.py --free-flight`, ledger id `final_v2_freeflight`):
composite 62.3 against 61.0 without it. Translation error at 150 steps
0.0205 m vs 0.0220, support-removal Jaccard 0.60 vs 0.50, but ground
penetration 1.04 mm vs 0.38 mm and post-action error 0.29 vs 0.20: bodies
now reach the floor at the true speed, and the model's landing response
is softer than PyBullet's. The rule is honest physics, so it stays; the
landing response is the model's problem to fix, not the rule's.

## Round 3 (2026-09-08): first blind tester with the diagnostics API

Tested the build deployed at `7751d95` (diagnostics, free-flight rule,
settle-hold, grab feed-forward) in a hidden tab, 194 tool calls, all four
scenes, mouse drags plus probes.

| Visuals | Physics | Interaction | Robustness | Overall |
|---|---|---|---|---|
| 4 | 2 | 3 | 3 | 3 |

Its ranked findings, with what each turned out to be:

1. **Mouse grabs spun pencils to 100-280 rad/s.** Real and new: the pick
   volume is 2.2x the pencil, so the spring attached up to 13 mm off the
   axis of a body with 2e-7 kg m^2 axial inertia. The probes attach on the
   axis and never saw it. Fix: attach on the axis; 60 rad/s and 3 m/s caps.
2. **Pencils climbed into standing poses after a nudge and stayed.** The
   model reads any nearby contact as support (residual +9.5 to +10.5) and
   the guard turned residual penetration into upward displacement. Fix:
   support is now "from below" only, a body with nothing below it falls
   (free-flight rule), and the pivot rule tips any body whose centre of
   mass is outside its support polygon. Detectors `tilt_hold` and
   `unbalanced_rest`.
3. **Piles hovered at load** (a pencil 10.8 mm up with nothing under it,
   carrying two others). Same cause as 2; the tester's own reading of
   `supportedBy: []` was exactly the missing rule.
4. **"Physics loop hangs silently after a drag."** Not reproduced. Its own
   log shows zero steps during a 5 s hold on a scene whose step counter had
   just been reset, and `physsplat.paused = true` was in its toolkit. The
   HUD now says PAUSED when the physics is paused and PHYSICS STALLED after
   3 s without a step, so that reading cannot happen again.
5. **Collisions interpenetrated up to 9.2 mm while a held pencil was pushed
   into another.** Real: at 3 m g the spring drove the pencil 8 mm into the
   neighbour per step and the guard shoved it back. Fix: the grab force
   drops to a third while the guard is pushing the held body out.
6. **The loader moved pencils 5-6 cm sideways** (IMG_8504). Real: overlaps
   were resolved along contact normals for 30 iterations and the cascade
   spread the pile. Fix: overlaps are resolved by lifting the upper body
   straight up (heights are what the reconstruction gets wrong), and the
   first 24 steps run before anything is drawn.
7. **Rest was a fight between model and guard** (guard totals 34-73 mm per
   300 steps with zero displacement). Cosmetic: settle restored the pose
   after the guard's push, and the record kept the push. Now zeroed.
8. **Grab tracking 15-46 mm when touching anything.** Mostly the off-axis
   attach (1) and the trained force cap. Re-measure.
9. **Pokes weak, flicks logged at 0.007-0.02 m/s.** Flick strength now comes
   from cursor speed and reaches the trained maximum easily.
10. **Reset threw `TypeError ... 'residual'`.** Fixed before the report
    landed (a reset during an in-flight GPU step); a completed step now
    clears the HUD error line.

Also from this round: the reconstructed pencils were 10-17 mm thick against
an 8-9 mm real barrel and 7-11 mm training pencils. The pipeline now scales
each pencil's cross-section to the known radius (4.5 mm outer) and its
density to the real 6.2 g; that puts the photo scenes back inside the
training distribution for radius, mass and inertia.

## Round 4

Pending.

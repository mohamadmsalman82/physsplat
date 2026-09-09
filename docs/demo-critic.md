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

## Round 3

Pending: a fresh blind tester with the diagnostics API in its brief.

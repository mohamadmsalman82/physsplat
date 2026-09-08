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

## Round 2

Pending.

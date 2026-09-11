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
training distribution for radius, mass and inertia. The tester's visual
complaint ("smooth pills, nothing identifies them as pencils") is
answered with a procedural pencil built inside each physical capsule:
barrel and eraser dome in the photo's colours, a rubber grip band, a
metal cone tip with a lead, and a clip.

Two things the diagnostics found on the fixed build before the next
tester could:

- **A pencil at rest reared up on its own.** The elevation track of the
  top pencil in IMG_8596 went 0, 6, 69, 45, 83, 79, 26, 18 degrees in the
  first seconds, held 17 degrees for five seconds, rose again to 89, stood
  on its end for two seconds, and fell. The packet starts flat; the
  network's angular residual on the reconstructed pile injected the
  energy, and the tip-balance detector missed it because a standing body
  never settles. Fix: the no-free-energy rule (`docs/diagnostics.md`):
  the residual may not raise the scene's mechanical energy by more than
  the action's work. Detector `rearing`; `tip_balance` now uses speed, not
  the settle verdict. Twenty seconds at rest afterwards: every elevation
  constant, zero interventions.
- **Reset during an in-flight GPU step stalled the loop** with a null
  record; a generation counter now aborts that step.

Probe batteries on the fixed build, IMG_8596 and IMG_8626: rest still;
lift tracks within 3-4 mm and the released pencil falls at 1.00 g and
lands; drop free-fall ratio 1.00; pulling the load-bearing pencil drops
what it carried; no unbalanced or tilted rest poses; the remaining
anomalies are single-step penetrations of 2-3 mm and sub-millimetre
jitter for a few steps after landings.

## Round 4 (2026-09-09)

The tester run for this round is still out. What the diagnostics and the
scorecard established in the meantime:

**The first no-free-energy rule was wrong, and the scorecard said so.**
It policed the whole scene's mechanical energy every step. An impact is a
legitimate energy spike for the bodies involved, so the rule cut contact
impulses: composite 42.4 against 62.3 for free-flight alone, stability
0.833 to 0.375, support-removal Jaccard 0.60 to 0.27, post-action error
0.29 to 0.57. It is now restricted to bodies that are sitting still
(under 5 cm/s and 1 rad/s, not held, not beside something moving) and
polices only their own energy, which is exactly the failure it was built
for: a resting pencil rearing up on its own. Rerunning the scorecard.

This is the value of scoring an analytic rule instead of assuming it. The
free-flight rule earned its place the same way (62.3 against 61.0, with
translation error 0.0205 m against 0.0220 and Jaccard 0.60 against 0.50,
paid for with ground penetration 1.04 mm against 0.38: bodies now reach
the floor at the true speed and the model's landing response is softer
than PyBullet's).

**Probe battery on IMG_8513**, a scene not covered in round 3: rest
still; lift tracks the cursor within 1.9 mm and the released pencil falls
at 1.00 g, impact 0.49 m/s against 0.504 analytic, at rest 0.45 s later;
pulling the load-bearing pencil out 78 mm drops the one above it 8.5 mm
onto the floor; the flick travels 23 mm and stops in 0.85 s.

**Flicks.** The trained 3-step impulse cannot move a pencil on a pile:
0.3 m/s of impulse gives 3 mm of travel, 0.6 m/s also 3 mm, and 1.0 m/s
spins the pencil to the 60 rad/s cap instead of sliding it. A mouse flick
is now a short push along the gesture through the spring path: 0.3 m/s
travels 13 mm, 0.5 m/s travels 29 mm, stopping within half a second.

**A detector was wrong too.** `tilt_hold` fired nine times per battery on
a tapered pencil lying flat, because it tested for missing capsule
contacts within 3 mm rather than for what it meant. It now asks whether
the body is tilted and held up over less than 30 mm.

### The tester's report

| Visuals | Physics | Interaction | Robustness | Overall |
|---|---|---|---|---|
| 6 | 3 | 3 | 6 | 3 |

Zero console errors across eight scene switches, three probe suites and
dozens of drags; reset bit-exact. The physics and interaction scores came
from measurements worth keeping:

1. **A pencil frozen in an impossible pose**: tilted 6.2 degrees, one end
   on the floor, the other unsupported, held 3867 steps. Settle latched a
   pose that is not an equilibrium.
2. **A pencil resting 11.6 mm up touching nothing**, with no anomaly
   raised, because the detector asked about 3 mm contacts while the body
   had 6 mm support.
3. **Contact resolved in position, not velocity**: 5.7 and 10.5 mm of
   movement in a step whose reported velocity was 11-45 mm/s, impact
   0.025 m/s against 0.356 analytic, and never a bounce.
4. **Impulses clamped and then destroyed**: 0.3 m/s applied, 0.13 m/s
   peak, with a spurious 24-27 rad/s spin out of a purely linear poke.
5. **A centre lift reared the pencil to 87 degrees** and it never left the
   floor.
6. **Grab steady-state error 8-13.6 mm** after the cursor stopped moving.
7. **A stationary click produced a 0.15 m/s flick**; four of them
   restacked the pile.
8. **Playback speed 0.32x to 1.22x real time** depending on GPU load.

One of its ten was a misreading worth recording: the "collision shell is
2.200x the drawn pencil" measurement is of `physsplat.proxies`, the
invisible ray-pick volumes, which are deliberately 2.2x so a pencil can be
grabbed without pixel precision. The collision capsules the guards use are
the fitted 4.5 mm radius, the same as the drawn barrel.

### What changed, and what the measurements said

Two rules written earlier in this round were **measured and rejected**:

| | composite | stability | photo drift |
|---|---|---|---|
| free flight only | 62.3 | 0.83 | 9 mm |
| + no-free-energy (whole scene) | 42.4 | 0.38 | 18 mm |
| + no-free-energy (quiescent bodies only) | 48.1 | 0.42 | 17 mm |
| + two-tap residual mean | 34.2 | 0.21 | 30 mm |

Both are off by default now, kept behind flags so the numbers can be
reproduced. What replaced the first of them, the angular contact fade, was
then scored the same way and **kept**, because it is the first analytic
rule since free flight that makes the model better rather than merely
safer:

| | composite | trans 150 | axis 150 | surface pen | stability | photo drift |
|---|---|---|---|---|---|---|
| model alone | 61.0 | 22.0 mm | 0.177 | 1.6 mm | 0.83 | 9 mm |
| + free flight | 62.3 | 21.2 mm | 0.186 | 1.6 mm | 0.83 | 9 mm |
| + angular fade | **63.6** | **18.6 mm** | **0.180** | **1.5 mm** | 0.83 | 17 mm |

Both kept rules ship in the demo and both are flags on
`scripts/evaluate.py`. The one regression is honest and worth stating:
passive drift on the photo scenes rises from 9 mm to 17 mm over three
seconds without any guards, which the demo's settle rule then hides.

What replaced them:

- **Angular contact fade.** Contact torque must vanish as a body
  separates; the model's does not, so lifting a pencil out of a pile it
  kept applying hundreds of rad/s^2 across millimetres of gap. The angular
  residual now fades linearly to zero at the contact radius, where free
  flight takes over. The linear part is untouched: it holds the pile up.
- **A pinch, not a point.** A single-point spring at the centre of mass
  resists no rotation at all, so any spin picked up while separating
  persisted and the pencil hung at 50-78 degrees. A held body now gets
  angular damping, in the interaction model rather than the physics.
- **A grab blocked by a neighbour** loses only the component pressing into
  it, not two thirds of the whole force, which had made a wedged pencil
  impossible to lift.
- **Flicks need 6 mm of travel**, so a click is a click.
- **The loop never runs faster than real time** and resynchronises after a
  stall, instead of replaying at up to 1.22x.
- Shadow map doubled with a normal bias (the dark bands across barrels
  were shadow acne), and the ground reads as a table rather than a void.

### Regression tests, so this is not re-argued by eye

`web/test/rest.mjs` and `web/test/interact.mjs` drive the browser
simulator headlessly. Rest: all four scenes still to within 0.4 degrees
and 0.2 mm over 20 seconds, against 29-32 degrees of tilt and 56-60 mm of
sinking for the model with the rules off. Interaction, per scene: drop at
1.00 g, grab tracking 0.0 mm, lift off the pile, release and land, support
removal, no interpenetration over 1 mm.

**Known limitation.** IMG_8626, a tight five-pencil pile where every body
touches four others, still fails three of those: the grabbed pencil cannot
be prised out, ends at 71 degrees, and pulling the bottom one out does not
drop what it carried. The other three scenes pass in full.

## Round 5 (2026-09-09)

| Visuals | Physics | Interaction | Robustness | Overall |
|---|---|---|---|---|
| 4 | 2 | 3 | 6 | 3 |

The sharpest report so far, and its root-cause section was right. Zero
console errors, eight scene switches, bit-exact reset, no blow-ups; and
underneath that, this:

1. **A pencil walked 102 mm across the table on its own** in 77 s on
   IMG_8626, at a steady 1.6 mm/s with no decay, `resting: false` from
   step 0, and no anomaly raised. Its diagnosis: a body that never quite
   reaches the settle threshold rides the guards' positional corrections,
   which are not velocity-consistent, so every push adds a little energy
   and the cycle never ends. It measured 68.65 mm of cumulative guard
   correction per 10 s on that body and 0 on the four that rest.
   The CPU regression test could not reproduce it over 40 s; the cycle
   needs the GPU backend's slightly different numbers to stay just above
   the threshold. Fixed by counting the steps on which the guards push a
   body that is going nowhere and letting settle claim it after half a
   second. Verified on the live page: 35 s, 0.0 mm drift on all five.
2. **Nothing was touching anything at rest**: gaps of 1.8-3.0 mm between
   bodies that the panel called contacts. The gap-closing snap only ran on
   the step a body first settled, so anything that settled during the load
   pre-roll kept its gap forever. It now runs every settled step. Live
   gaps afterwards: 0.1-0.5 mm. This also fixed the IMG_8626 lift, which
   had been failing because the pencil was not in contact with what it was
   supposedly resting on.
3. **The pencils looked semi-transparent.** They were: the barrel was an
   open-ended cylinder, so front-face culling let you see straight through
   it. Closed.
4. **Shadows were detached streaks matching no pencil.** `normalBias` was
   0.02, four times the pencil radius. 1.5 mm.
5. **A stationary click could still throw a pencil.** The drag plane is
   defined by the camera, so orbit damping coasting from an earlier
   gesture moved the projected target with the mouse still. Travel is now
   measured in screen pixels too.
6. **Everything ran at 0.45x real time.** True, and not fixable by a flag:
   a step costs 27-53 ms against the 16.7 ms budget. The HUD now states
   the playback rate rather than letting it read as low gravity.
7. The ground is a lit wooden table rather than a near-black grid in a
   void.

Its recommendation to put the energy limiter back is the one I did not
take, and the reason is in the table above: it was measured against the
model, not against a frozen scene, and it cost 14 points of composite and
half the stability. The walking pencil had a cheaper cause and a cheaper
fix.

Interaction battery afterwards: IMG_8596, IMG_8504 and IMG_8513 pass in
full; IMG_8626 fails only support removal, down from three failures.

## Deferred: fine-tuning under the kept rules

`scripts/improve.py --rules` trains the model against the residual the
runtime actually allows (free flight plus the angular fade) and scores it
the same way, so the loop stays comparable within itself. It is written,
tested and ready, and it has not been run to completion: three attempts
were killed by the OS for memory. This machine has 24 GB with about 4 GB
free, 8 GB compressed and swap 91% full, so a rollout fine-tune's few
gigabytes cannot be had while the desktop is loaded. It needs either a
quieter machine or some applications closed, not a code change.

## Round 6 (2026-09-09)

| Visuals | Physics | Interaction | Robustness | Overall |
|---|---|---|---|---|
| 6 | 3 | 4 | 7 | 4 |

Every category up, and four of the seven claimed fixes confirmed by
independent measurement: shadow normal bias read back at 1.5 mm with a
pencil's shadow meeting its tip, a stationary click moving all four bodies
0.000 mm, playback rate measured at 0.623 against the HUD's 0.65, and the
barrel materials verified opaque with blending disabled. It also settled
an old question: the "see-through pencils" of rounds 4 and 5 were never
transparency. Turning off shadow casting removed the effect entirely, so
what looked like x-ray was a contact shadow tracing the occluder's
silhouette onto the barrel underneath.

It agreed the energy limiter should stay off, having found `energyScale`
at 1 and `energyGain_uJ` at 0 across eight scene loads and three probe
suites, and added the better argument: a limiter would have clipped the
2.8x energy gain it measured on a poke and hidden the fact that the model
injects energy at contact.

Its two open findings were both right, and both had the same cause: sleep
was asking whether a body was supported and slow, never whether its pose
was resolved.

1. **Scenes shipped frozen mid-settle.** A pair of pencils asleep 2.3 and
   14.8 mm in the air, other barrels 0.5-3.8 mm inside the tabletop, and
   the panel reporting "resting 4/4, no anomalies" over it. Its proof that
   the solver could fix it was decisive: one 5 s drag brought every body
   within 0.9 mm of the table. Sleep now also requires contact within a
   third of a millimetre, and the hidden pre-roll runs 3 s rather than
   0.75 s so the pile gets there before anyone sees it.
2. **The walking pencil was exempt by construction.** The guard-held sleep
   rule of round 5 needs a body to be going nowhere; this one was going
   somewhere, 1.86 mm/s. The answer is friction, which the model does not
   supply: a body touching something, barely moving in every direction,
   untouched, and slow for ten consecutive steps has its horizontal
   velocity zeroed. The ten-step count keeps a body that has just lost its
   support out of it, and a first attempt without that count did break
   support removal, which is how I know the count matters. A first attempt
   also gated on floor contact only, and the live GPU build showed the
   pencil still drifting 50 mm in 25 s because it slides across its
   neighbours 6 mm up; the gate is now any contact.

Measured on the deployed build afterwards, GPU backend: IMG_8596 and
IMG_8626 both hold every body at 0.00 mm of drift over 30 s with contact
gaps of 0.0-0.3 mm and no anomalies. All four scenes pass both regression
suites in full, the first time that has been true; IMG_8626 is no longer
a known limitation.

Left open from its report, and worth stating: grab tracking lags 12-38 mm
on a moving cursor, the diagnostics panel's `lowest_mm` is 1.5-3.9 mm
optimistic against the drawn mesh because the drawn barrel takes the
capsule's median radius while the physics body is a particle hull scaled
at the 95th percentile, and contact response is applied in one 16.7 ms
step, which is where its 4.4 mm of transient overlap and one-frame angular
spikes come from.

## Player feedback (2026-09-09)

Not a scored round: three complaints from someone playing the deployed
demo. All three had causes worth writing down.

**"Very jittery if I try to move it with my cursor."** Two causes. The
physics runs at about half real time, so between steps the cursor jumps
centimetres and the spring received a hard yank, overshot, and shook; the
grab now follows a rate-limited point moving at 0.30 m/s, the speed grabs
were generated at, which also keeps the model's action input in
distribution. And the renderer lerped between the last two physics states
over a fixed window, so at step times of 24-73 ms it reached the newer
state and froze until the next arrived. It now chases the latest state
exponentially at a rate set by the frame time. Measured after: one jitter
episode per drag, against three before.

**"Sometimes the pencils just hover."** The drawn barrel used the capsule's
MEDIAN particle radius, 3.9 mm, while the physics body reaches 5.6 mm at
the grip. A pencil the ground rule held exactly on the table was therefore
drawn 1.7 mm above it, every time. The mesh now reads the radius profile
off the particles: barrel 4.7-5.1 mm and grip 5.2-6.0 mm placed where the
fattest particles are, which matches both a real Matic Grip and what
actually touches.

**"Sometimes they sink inside the table."** The same mismatch from the
other side: the clip stood 0.9 mm proud of the drawn barrel and the grip
0.3 mm, so whichever faced down passed through the surface. Both now sit
inside the barrel, and nothing drawn lies outside the body the ground rule
holds.

**"They don't fall fully flat."** A physics cause as well: the balance test
allowed the centre of mass 6 mm outside its support before tipping, enough
for a pencil to see-saw on a single crossing indefinitely. A crossing
contact patch is a couple of millimetres wide, so the tolerance is 2 mm and
a pencil now tips until an end reaches the table.

**"Pull a pencil out from under a pile quickly and it phases through. Drop
one from high up and it phases through."** Real, and the regression suite
had not been asking. Contact was resolved only at the end of each 16.7 ms
step, so a pencil moving at 2 m/s travelled 33 mm between tests and could
start a step above a neighbour and finish below it. Three things fixed it.
Swept collision detection now samples the segment each body travelled and
refuses any pair that deepens more than 0.3 mm beyond where it started the
step, which is what makes it work while a pulled pencil is legitimately
touching everything; an early version skipped pairs already in contact and
so switched itself off in exactly the case that needed it. The guard now
iterates to convergence rather than running three fixed passes, and a held
body's speed is clamped to 0.25 m/s after integration, not before, where a
first attempt clamped the previous step's velocity and did nothing.
Measured: drops went from 7.8 mm of pass-through to 0.0-0.3 mm, fast pulls
from 7.7 mm to 0.4-2.1 mm. Two new suite checks cover both.

**"Whenever I am moving the pencils around they sometimes phase through the
ground."** The convergence loop's stopping test measured pencil-pencil
overlap only. Ground penetration was capped at 1.5 mm of lift per pass, so
a body pushed 10 mm under the table by a drag needed seven passes, and the
loop was free to stop after one because no capsule pair overlapped. The
test now takes the worst of both. All four scenes report 0.00 mm below the
table across a four-leg drag, and there is a suite check for it.

**"The pencils look nothing like how they actually look. Make it a 1:1
replica."** They were capsules with colour bands sampled off the photograph,
which read as mottled rather than moulded. Rebuilt from a reference
photograph of the pencil as real parts: white eraser standing out of a
barrel-coloured cap, one solid-colour constant-diameter barrel, a grey clip
plate with a rounded lip, a moulded grey rubber grip that flares out of the
barrel, and a cone that is barrel-coloured plastic ending in dark lead. An
intermediate version drew a metal ferrule, which this pencil does not have.
Barrel colour is now the most saturated cluster in the reconstruction
rather than the mean, so the grey grip and the shaded side no longer muddy
it. Geometry is still tied to the physics: the grip is drawn at the
particle hull's fattest band and the barrel follows from the real pencil's
11 mm over 9 mm ratio, so nothing drawn lies outside the body the ground
rule holds.

## What the pencil rewrite found underneath (2026-09-09)

Building the 1:1 pencil needed a test that the drawing and the physics
agree about where the object's surface is, since the hovering and sinking
reports were exactly that disagreement and no physics test could see
either. `web/test/mesh.mjs` compares them as support functions, which is
the same projection the ground rule computes, so its output is literally
how far the drawing hangs below the lowest particle.

Writing it turned up three more defects in the drawing, all fixed: the
pencil was centred on the body origin while the particles are offsets from
the centre of mass and sit up to 9.3 mm off it, so it overhung the short
end by up to 19 mm; the reconstructions' cross-sections are elliptical at
about 1.4 to 1, so a round pencil drawn at the fattest particle reached 1
to 2 mm outside them; and which end got the point was taken from the
fattest particle band, which a lumpy reconstruction gets backwards.
Flat-lying escape went from 2.58 mm to 0.000 mm on all four scenes.

It also turned up something the demo cannot fix. The reconstructions are
poor pencils in ways that reach the screen because the physics samples
them: they taper over their last 10 to 15 mm where a real Matic Grip is
straight, nine of the seventeen bodies do not distinguish their two ends
by even 0.3 mm of mean radius, and reconstructed lengths run 99 to 150 mm
for an object that is always 150.

`cylinder-bodies` is a branch that fixes this at the source, sampling the
physics particles off a cylinder of the known radius rather than off the
convex hull of the reconstruction. It measures better on everything it was
written for: drawn-to-simulated standoff 2.28 mm to 0.33 mm, tilted escape
6.09 mm to 2.73 mm, masses 6.3 to 8.1 g where the hull gave 4.1 to 5.7 g
against a real pencil's 6.2, and every scene starting with its lowest
particle at 0.00 mm once the scene is grounded on the particles instead of
on the reconstruction's vertices.

It is not merged, for two measured reasons and one structural one.
IMG_8513's fast pull tunnels 2.1 mm against a 1.5 mm bar, up from 0.9 mm,
and it is not the guard running out of passes (24 passes gives the same
number) but larger transient mid-step overlap between fatter bodies.
IMG_8626's body 0 will not lift clear of the pile. And a cylinder is
symmetric, so the drawing loses every geometric cue for which end is the
point; that has to come from the render colours, the white eraser and the
dark lead, before it can ship. Recorded here so the next round starts from
the numbers rather than rediscovering them.

## Round 7 (2026-09-09): 49/100

Fifteen findings, most of them measured. Two of the three it ranked highest
were real and are fixed below; the second-ranked one did not reproduce and
is worth recording as a lesson about the harness.

**Verified, in the code, and fixed.** The reset button froze the page for
6 to 12 seconds and then teleported every pencil up to 26 mm: the 180-step
pre-roll ran at 35 to 60 ms a step with nothing drawn, so the last frame on
screen was the raw loaded pose until the settled one snapped in. The
packets now ship settled by the real model (`web/test/settle_packets.mjs`)
and the pre-roll is 12 steps, drawn. Ground was a 600 mm disc over an
infinite invisible floor. The grab spring did not engage for the first 350
ms of a press, to tell a flick from a drag, and the follow point was capped
at 0.30 m/s of simulated time; both are still open, and both are what a
player feels as lag.

**Did not reproduce: "IMG_8626 never comes to rest, body 4 creeps 40 mm."**
Headless with the ONNX backend: 0.00 mm drift on every body over 25 s,
all five at rest. Live with the WebGPU backend, driven by hand with the
page loop paused: 0.00 mm over 17 s. Live with the page loop running
normally for 5,522 steps: all five at rest, zero velocity, zero anomaly
episodes. The critic's session had, by its own account, wedged the physics
loop twice by writing to an undocumented field, and its measurement
combined the page's own loop with `waitSteps` calls from a second driver.
Two drivers interleave steps and produce exactly this kind of number. I
reproduced the same artefact myself before finding it, which is why
`docs/sensors.md` now says so in capitals.

**Conflated: contact bookkeeping "3.4 mm too generous per side".** The
8.36 mm figure is the pick volume, `capsule.radius * 2.2`; the contact
proxies were 3.7 to 4.6 mm. The underlying complaint, that a uniform
radius carried to the tip holds a pencil up on a neighbour's point, was
nonetheless right, and is what the next section fixes.

**Confirmed and useful.** Friction collapses above about 0.8 m/s, outside
the training distribution; the flick path clamps to 0.6 m/s so it is
reachable only indirectly. Contact integrity was the strongest result it
found: sampling every step through grabs, pulls and drops it could not
make anything pass through anything, and no particle ever went below the
table. Free flight, restitution and in-distribution sliding friction all
measured exact.

## Player report (2026-09-10): one shape for everything

Three complaints, with a photograph of a pencil visibly floating above the
two it rested on: the pencils were not all the same size; they hovered,
sank into the table, and were lifted off it by a neighbour's tip; and they
twitched and slid by themselves.

All three had one cause. Every pencil's shape came from its own
reconstruction (99 to 150 mm long, elliptical, tapered at the ends), the
contact proxy was a uniform-radius tube, and the drawing was a third
approximation of the same object. Now there is one canonical Matic Grip
profile (`common/pencil.py`, `js/pencil.js`) and the physics particles,
the contact radii, the ground rule and the drawn mesh all come from it.
Measured after: every body 150.0 mm and 6.2 g; drawing against physics
0.0000 mm; no body floats in any scene; every body rests at 0.00 mm from
what is under it; worst residual overlap 0.086 mm; all four scenes at
rest; parity 5.0e-7 m.

Also from this: `web/public/js/sensors.js`, a layer that names which part
of which pencil touches which part of which other pencil in millimetres
from the point. Its first reading on IMG_8626 found body 4 at 9.9 mm above
the table with its only contact a pencil ABOVE it, which is the hovering
in the photograph, and which led to the seating rule. Documented in
`docs/sensors.md`.

Open after this: a fast pull in IMG_8504 shows 5.7 mm of mid-step overlap
and a 250 mm drop in IMG_8513 3.4 mm, both under deliberately violent
motion between bodies that are now genuinely 11 mm across at the grip. A
whole-step correction budget was tried against them and measured worse
(5.7 to 6.8 mm), so it is documented in `sim.js` and not in. Which end of
each pencil is the point comes from colour cues that disagree on about
half the bodies; the packet records a confidence per body, 0.00 to 0.46,
so nobody trusts it more than it deserves.

## Round 8 (2026-09-10): 39/100, sensor-driven

The first round with `physsplat.sensors`, and the first whose every
finding came with the part of the pencil it happened on. Its top-ranked
finding was a one-line bug and it explains most of what players had been
reporting since round 5.

**`actPinch` was never passed.** `main.js` called `sim.step(body, point,
force)` with three arguments. The fourth gates the pinch damper and both
held-speed clamps, so none of them ever ran in the browser, while
`interact.mjs`, which passes `true`, kept passing. Live: a grab with the
cursor held still reared a pencil to 89.9 deg at 3.2 m/s and 60 rad/s,
drove it 31 mm through the table and left it standing on its end; a
40 mm/s drag reached 1.1 m/s; a held pencil in mid-air rotated at 0.337
rad/s, unchanged in the third decimal for 181 steps, because nothing
damped it. The critic proved the mechanism by grabbing at the exact
centre of mass with the cursor still, so the applied force was m*g with
zero torque, and reading the model's angular residual: 123, 303, 583,
725, 980, 1004 rad/s^2, alternating in sign, 1431 at worst.

Passing the flag was necessary and not sufficient: even damped, a held
pencil reared 28 to 35 deg. The learned angular residual on a held body
is not physics, the model having never seen a stationary held body, so
on the held body only it is now two-tap averaged (the ringing is exactly
step-alternating, so the average cancels it and nothing else) and capped
at 3 rad/s^2, with pinch damping at 60/s. Hold-still tilt after: 3 to 7
deg, spin 0.09 rad/s, all four scenes. Three new suite checks pin it.

**Three mechanisms under it, all named by the report with numbers.** The
swept collision check looped over bodies and never the table, which is
why every deep penetration it found (29 and 31 mm) was a point or eraser
through the tabletop; the table is in it now. The balance test counted
particles within the 6 mm contact radius as floor support, so a 4 deg
tilt read as 86 mm of floor (6 / sin 4 deg), the centre of mass fell
inside it, and settle froze pencils with 14 mm of air under one end; it
is 1.5 mm now. And settle's "touching" included a neighbour resting on
TOP, so a pencil with a load on it and nothing beneath it was frozen
15 mm in the air; a body with air under it, as the seating pass measures
it, is not at rest whatever its speed.

**Not a defect, but a harness mismatch worth recording.** The suite
pre-rolled 180 steps where the page pre-rolls 12, and those 168 steps
changed which body the support-removal check picked in IMG_8504, from
one whose load drops cleanly to one lying at 18 deg on its cone whose
load rides up the incline as it is pulled, slides off the end, and lands
flat on a neighbour higher than it started. The sensors showed every
step of that and it is what the physics should do with that geometry.
The harness pre-rolls 12 now.

**Right, per the report, and worth keeping.** Rest is genuinely
motionless (13,713 steps to the last digit). Gravity is 9.810 m/s^2
from the release trace. A pencil laid across another's POINT sits 6.2 mm
lower than across its barrel and the grip's extra millimetre is modelled
to 0.00 mm, which is the fix from the previous section working. Drops
onto a pile overlap 0.14 to 0.56 mm.

## The engine (2026-09-11): Rapier by default

Not a round. A player, after everything above: "the pencils never lay flat
... find a current open source repository that maps the physical behavior
of pencils being stacked on each other and fix it."

That repository is Rapier (dimforge/rapier), a rigid-body solver in Rust
compiled to WebAssembly. It is the demo's default engine now, with the
learned model behind `?engine=gnn`, both behind one interface so the
sensors, probes and page do not know which is underneath. Held to the
same sensor-driven checks on all four scenes: a lone pencil dropped on
the desk lies at 0.52 deg on its grip; nothing drifts (0.00 to 0.17 mm
over 10 s); nothing floats; every body rests at 0.00 mm from what is
under it; holding a pencil still tilts it 0.2 to 3.8 deg (was 3 to 35)
with 0.04 to 0.40 rad/s of spin (was 0.09 to 13.8); an end grab dangles
to exactly 90.0 deg; a 250 mm drop overlaps 0.1 to 1.0 mm for a frame; a
pencil dropped across another's point lands on its cone at a point's
height. A step costs 0.5 ms against 15 to 33.

What the solver taught, each measured: 16-sided collider hulls had been
stopping pencils rolling with their flats, and with round hulls IMG_8504
rolled 33 mm in its first ten seconds, which is why a real pencil has a
clip and why the clip is now a collider standing 1.2 mm proud (an 8 rad/s
spin rolls 0.1 mm before it catches); plastic on plastic at 0.42 made a
pile ride along on a slowly pulled pencil, at 0.28 it drops or slides
off, though a pencil lying almost parallel with most of its weight aboard
still rides 277 mm, as a real one would; an impact at 2.2 m/s sinks 7 mm
into the pile at 4 substeps a frame and 0.45 mm at 16; and settling the
loaded scene quasi-statically, under heavy damping, is what keeps the
photo's arrangement, since a pencil the solver finds a few millimetres
off equilibrium otherwise drops, kicks a neighbour, and sends it 15 cm.

The learned model stays, in the repo, the paper and the demo, because the
comparison is the finding.

## Round 9

Pending.

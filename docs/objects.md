# Demo object datasheet: BIC mechanical pencils

The demo objects are BIC mechanical pencils (Matic Grip line or similar) in
purple, gray, orange, and turquoise, stacked on each other. This sheet records
their measured/derived physical properties and every consequence for the
pipeline. Photos of the actual pencils pending (re-send as JPEG/PNG).

## Physical properties

| Property | Value | Source / derivation |
|---|---|---|
| Length | 150 mm | BIC Matic Grip spec |
| Diameter | ~11 mm at grip, ~8-9 mm barrel | BIC Matic Grip spec |
| Mass | 6.2 g | BIC Matic Grip spec |
| Effective density | ~530 kg/m^3 | 6.2 g over solid-cylinder volume (d=10, L=150) |
| Reconstruction density | 850 kg/m^3 | `recon/pipeline.py`: the rescaled hull (outer radius 4.5 mm, tapered) has ~75% of that cylinder's volume, so 850 lands near 6.2 g and stays inside the training range |
| Aspect ratio | ~14-15 : 1 | within the 15:1 datagen cap, barely |
| Barrel material | molded plastic, often translucent/frosted | recon risk, see below |
| Grip | rubber sleeve near tip | non-uniform friction along the body |
| Extras | pocket clip, eraser, tapered metal-ish tip | small convexity bumps; multi-colored parts |
| Friction (plastic on plastic) | mu ~ 0.2-0.35 | typical values; grip rubber much higher (~0.8) |
| Cross-section | round (rolls freely until clip touches) | rolling is a first-class behavior |

## Consequences for the pipeline

### Data generation (Phase 1)
1. **Density randomized, not fixed.** Real pencils are hollow (~530 kg/m^3);
   solid wood/plastic is 600-900. Constants change from `DENSITY = 800` to
   `DENSITY_RANGE = (400, 900)`. Mass and inertia are model inputs, so the
   network handles the range.
2. **Friction floor lowered to 0.2.** Smooth plastic barrels sit below the old
   0.3 floor. New range mu in [0.2, 0.9]; the high end covers the rubber grip.
   Friction is not an input feature (unknowable from a photo), so the wide
   range teaches an averaged contact response.
3. **Pencil-stack scene regimes become first-class.** Add to the initial
   condition mix: (d) parallel bundles lying flat and touching, (e) crosshatch
   / log-cabin lattices 2-4 layers, (f) groove pyramids (pencils resting in
   the valleys between a parallel pair). These are the demo, so they get
   heavy weight, along with rolling-to-rest trajectories.
4. **Capsules over cylinders in PyBullet.** Cylinder-cylinder point contacts
   (crossed pencils) are numerically finicky; capsule collision math is
   stable. Calibration item: capsule primitives, possibly with raised solver
   iterations, verified against the momentum/penetration invariants.

### Model / particles (Phase 2)
5. **Particle budget is fine.** At 4 mm spacing a pencil carries ~8 particles
   per cross-section ring, ~300 total, under the 400 cap. No constant change.
6. **Rolling fidelity is the thing to test early.** A coarse 8-particle ring
   must still let the GNN learn roll-and-settle. The Phase 2 overfit test
   should use a pencil-roll trajectory, not just a box drop.

### Reconstruction / clustering (Phase 6)
7. **Hardest object class for single-image recon.** Thin + possibly
   translucent barrels is the worst case for TripoSR/LGM. The week-1 recon
   spike is now load-bearing. Mitigations, in order: frame tight so pencils
   span many pixels; if barrels are translucent, wrap them in matte colored
   tape (opaque, distinct, slightly thicker, all wins); fall back to the
   multi-view path, which handles thin objects far better.
8. **Clustering needs an over-segmentation merge pass.** A pencil is not one
   color: colored barrel + white/gray grip + eraser + tip. Naive
   position+color clustering may split one pencil into 2-3 clusters. Add a
   merge step: combine adjacent clusters whose principal axes are collinear
   (they belong to the same pencil). This is a planned code change in
   `recon/cluster.py`, not a hope.
9. **Clip changes rolling at the margins.** The convex hull includes the
   pocket clip as a bump, which is roughly correct: real pencils roll until
   the clip stops them, and a hull with a bump does something similar.
   Accepted approximation.

## Confirmed from reference photos (2026-09-07)

- **Model: BIC Matic Grip 0.7mm HB #2** (printed on the barrel). Datasheet
  numbers above hold.
- **Barrels are opaque.** No translucency problem, no tape needed.
- **The set has duplicates:** at least 2 purple and 2 gray-blue pencils plus
  orange and teal. Two same-colored pencils must never touch in a demo scene
  or they merge into one body.
- **Every pencil wears the same gray rubber grip** covering roughly a quarter
  of its length, plus a colored tip and eraser. Two consequences:
  1. The over-segmentation merge pass is mandatory, not precautionary: every
     single pencil will initially cluster as barrel + grip + tip pieces.
  2. **The gray-on-gray trap.** The gray pencils' barrels are nearly the same
     color as the grip that every other pencil wears. Where a gray pencil
     touches another pencil's grip, neither position nor color marks the
     boundary, and clustering cannot split them. Collinearity in the merge
     pass helps only when the two pencils point in different directions.
     Scene rule: gray pencils go in spread-out arrangements or touch other
     pencils on their colored barrel sections; the core stacking trio is
     purple, orange, teal.
- **The demo arrangement is a loose criss-cross pile**, pencils dropped
  crossing each other at shallow random angles, not a neat lattice. Datagen
  gets a matching regime: sequential drops at random yaw over a small area,
  settling into a tangle. Multi-contact shallow-angle pencil-on-pencil
  resting is the dominant contact mode to learn.
- **Framing note:** the reference pile photo has the pencils in the bottom
  third of a tall frame. For reconstruction input the pencils must fill most
  of the frame (see capture-guide.md).

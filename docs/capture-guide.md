# Capture guide

How to photograph a scene that reconstructs well. Pull this up on your phone
while shooting.

## What to use

Three to five objects, each a **clearly different color** from its neighbors.
Good choices: wooden blocks, Duplo or Lego bricks, erasers, dice, small boxes,
chunky markers, highlighters, a thick pencil.

Color separation is a technical requirement, not a style choice. The pipeline
splits the scene into objects by clustering on position *and* color, so two
touching objects of the same color merge into one body and will move as a
single glued lump.

**Avoid:** anything transparent or glassy, glossy or metallic, pure black or
white, very thin or wiry, and anything concave like a mug or a cup. All four
break the reconstruction stage, and concave shapes are outside the physics
scope (everything is simulated as its convex hull).

## Setup

- **Background:** plain and matte. A sheet of white poster board, a clean desk,
  or a plain-colored towel. No patterns, no clutter, no visible edges of other
  objects.
- **Light:** soft and even. Near a window on an overcast day is close to ideal.
  Avoid direct sun and overhead spotlights. You want minimal hard shadow and no
  blown-out highlights.
- **Camera:** phone is fine. Hold it 30 to 45 degrees above the table, not
  straight down and not at table level. Get the whole arrangement in frame with
  some margin around it. Tap to focus. Keep it steady.

## What to shoot

Ten photos, varying the arrangement:

1. Three objects in a loose row, not touching
2. Same three, touching side by side
3. A two-object stack
4. A three-object stack
5. A stack with something leaning against it
6. Objects scattered flat, including a pencil lying down
7. A tall precarious stack
8. Four to five objects in a cluster
9. Any arrangement, shot from a lower angle
10. Any arrangement, shot from a higher angle

Different arrangements matter more than different objects. Some will
reconstruct better than others and we pick the winners.

## Optional: the high-quality path

For one hero scene, walk a slow circle around it taking 30 to 60 photos,
keeping the arrangement untouched and the whole scene in frame each time.
This produces a much better reconstruction than a single photo. Worth doing
once, later, for the demo video.

## Where to put them

Drop the files in `data/photos/` in this repo. They are gitignored, so nothing
large gets committed. Name them anything.

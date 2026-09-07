# Capture guide

How to photograph a scene that reconstructs well. Pull this up on your phone
while shooting.

## What to use

Three to five objects. Good choices: wooden blocks, Duplo or Lego bricks,
erasers, dice, small boxes, chunky markers, highlighters, a thick pencil.

### The color rule

**Only objects that touch each other need different colors.** Objects sitting
apart are separated by position alone, so two identical red blocks a few
centimetres apart are fine.

The reason: the pipeline splits the scene into objects by clustering on
position *and* color. Where two objects touch, position runs continuously
across the contact seam, and color is the only thing that marks the boundary.
Two same-colored blocks stacked directly on each other merge into one body and
will then move glued together when you poke either one.

In practice that means you need roughly three distinct colors, not five. In a
five-object scene you can reuse a color as long as the two never touch.

"Different" means easily distinguishable, in brightness as well as hue. Red
against blue is safe. Red against orange, or two shades of natural wood, is
risky. Also keep the objects contrasting against the background, since
background removal runs before anything else.

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

## Shooting the BIC pencils (the chosen demo objects)

The demo objects are BIC mechanical pencils in purple, gray, orange, and
turquoise (full datasheet in `objects.md`). Two pencil-specific rules:

- **Frame tight.** Thin objects are the hardest case for reconstruction, so
  the pencils should fill most of the frame, not sit small in a wide shot.
- **Check for translucency.** Hold a barrel up to the light. If you can see
  through it, reconstruction will struggle: either pick opaque pencils or
  wrap each barrel in a distinct color of matte tape (washi or masking tape).
  Slightly thicker and fully opaque is a double win.

## How many photos

**Set up these 8 arrangements, and shoot each one from 2 or 3 angles.**
That is roughly 20 to 25 photos total, and about 30 to 45 minutes of work.

The arrangements:

1. Three pencils parallel on the table, not touching
2. Three pencils parallel and touching, like a raft
3. A crosshatch stack: two parallel pencils, two more across them (# shape)
4. A three-layer crosshatch (log-cabin style)
5. A groove pyramid: two parallel touching pencils with one resting in the
   valley between them
6. Pencils scattered at random angles, flat on the table
7. One pencil leaning across another (X shape)
8. All four pencils in your intended demo stack (the arrangement from your
   reference photo)

Each photo is an independent candidate scene, because the reconstruction runs
from a single image. Multiple angles of the same arrangement cost nothing and
matter more than you'd expect: reconstruction quality varies a lot with
viewpoint, so shooting one arrangement three ways gives three chances at a
good result from one setup. Move around the table between shots rather than
rotating the objects.

Keep the originals at full resolution and do not crop them.

**Optional insurance:** repeat the same 8 arrangements under a second lighting
condition (window light, then lamp light). Lighting problems are systematic,
so a second condition protects the whole batch in a way that extra angles of
one setup cannot. Roughly 40 to 50 photos total if you do this.

These photos are demo scenes, not training data; the model trains purely on
synthetic simulation. A batch this size exists to yield 3 to 5 good scenes
for the final demo, with slack for reconstruction failures, and reshooting
later after we learn what works is cheap and expected.

## Optional: the high-quality path

For one hero scene, walk a slow circle around it taking **30 to 60 photos of
that single arrangement**, keeping the objects untouched and the whole scene in
frame each time. Step maybe 10 degrees between shots and vary your height a
little. This produces a much better reconstruction than any single photo.

Worth doing once, later, for the demo video. Not needed now.

## Where to put them

Drop the files in `data/photos/` in this repo. They are gitignored, so nothing
large gets committed. Name them anything.

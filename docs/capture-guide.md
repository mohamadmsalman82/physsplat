# Capture guide: the full photo campaign

The demo objects are BIC Matic Grip pencils (opaque, confirmed): purple x2,
gray-blue x2, orange, teal. Full datasheet in `objects.md`.

**Total: ~260 photos in 5 sets, about 3 hours.** Each set answers a specific
question. Shoot them in order; folder names below.

## Ground rules (apply to every shot)

- Soft even light, no direct sun, no harsh shadows. Don't shadow the scene
  with your own body.
- Tap to focus on the pencils. Hold steady. No zoom, move your feet instead.
- Pencils fill ~70% of the frame. Background is table, not wall. Keep panel
  seams and table edges out from behind the scene.
- Same-colored pencils never touch (two purples, two grays: keep pairs apart).
- Gray pencils stay out of tangles except where the stress set says otherwise.
- HEIC straight off the iPhone is fine for files on disk; AirDrop originals
  into the folders below and they get converted during processing.

## Set A: Calibration singles (~20 photos, 15 min)

Folder: `data/photos/A_calibration/`
Unit tests for reconstruction, scale, and color clustering.

1. Each color alone on the white desk, 2 angles each (8 photos)
2. One pencil beside a ruler or tape measure, 2 shots (metric scale check)
3. Minimal two-pencil contacts with purple + orange: X cross, parallel
   touching, one leaning on the other; 2 angles each (6 photos)
4. One gray pencil and one purple pencil on a dark matte background
   (dark poster board, dark cloth, or dark desk), 2 shots each (4 photos)

## Set B: Core demo arrangements (~60 photos, 45 min)

Folder: `data/photos/B_core/`
The demo scene candidates. **12 arrangements x 5 angles = 60 photos.**

The five angles per arrangement: three elevations (about 30, 45, 60 degrees
above the table) from one side, one shot from a different side at 45, and
one closer detail shot.

1. Three pencils parallel, not touching
2. Three parallel touching (raft)
3. Four-pencil raft
4. Crosshatch: 2 + 2 across them (# shape)
5. Log cabin: three layers of 2 (use 6 pencils; keep the two purples and two
   grays in non-touching positions)
6. Groove pyramid: 2 parallel touching + 1 resting in the valley
7. Bigger pyramid: 3 + 2
8. X lean: one pencil across another
9. Lean cluster: three pencils leaning against each other at angles
10. Loose criss-cross pile of 4 (purple, orange, teal + one more)
11. Loose criss-cross pile of 5-6 (the reference-photo style)
12. Your intended showcase arrangement, exactly as you want it in the demo

## Set C: Lighting and background variation (~24 photos, 20 min)

Folder: `data/photos/C_variation/`
Measures how much conditions matter, so the best setup is chosen on evidence.

Re-shoot arrangements 2, 4, 6, and 11 from Set B:
- under a second light source (if Set B was window light, use lamp light),
  3 angles each (12 photos)
- on the dark background, original lighting, 3 angles each (12 photos)

## Set D: Stress set (~15 photos, 15 min)

Folder: `data/photos/D_stress/`
Deliberately hard shots that probe predicted failure modes. These are
expected to fail; they map the boundary the demo must stay inside.

1. A gray pencil resting directly on another pencil's gray grip, 2 angles
2. The two purple pencils touching, 2 angles
3. Dense tangle of all 6-7 pencils, 3 angles
4. The tallest, most precarious stack you can balance, 2 angles
5. One extreme close-up of a pile, one wide shot with the pile small in
   frame (like the original reference photo, for a framing comparison)
6. Two or three chunky non-pencil objects (eraser, block, die) in one shot,
   2 angles: the insurance objects

## Set E: Multi-view orbits (3 scenes x ~45 photos, ~60 min)

Folders: `data/photos/E_orbit_pile/`, `E_orbit_pyramid/`, `E_orbit_raft/`
Hero-quality reconstruction, and ground truth for scoring the single-image
path (same scene, one photo vs. many).

Scenes: (1) the showcase criss-cross pile, (2) a groove pyramid, (3) a raft
with one pencil leaning on it.

For each scene, without touching the arrangement:
- Walk a full circle at roughly 10-degree steps, camera ~45 degrees above
  the table: ~36 photos
- A second partial pass higher up (~60 degrees), every ~30 degrees: ~9 photos
- 3 closer detail shots

Stills, not video. Keep the whole scene in every frame, keep exposure
consistent (lock it if you know how), and prioritize sharpness: pause,
steady, shoot.

## After shooting

Drop everything into the folders above (they are gitignored) and say the
photos are in. Processing order is A first (pipeline unit tests), then B.

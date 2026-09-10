/**
 * The canonical BIC Matic Grip, browser side.
 *
 * These numbers are the same numbers as src/physsplat/common/pencil.py, and
 * they must stay the same numbers. That file explains why the shape is
 * canonical rather than reconstructed; the short version is that the photo
 * says where each pencil is and what colour it is, and the object says what
 * shape it is, because they are all the same mass-produced pencil.
 *
 * Everything downstream reads the profile from here:
 *
 *   pencil_mesh.js revolves it into the drawn surface, so the drawing IS
 *   the physics surface rather than an approximation of it;
 *
 *   physics.js evaluates it at the contact point, so a pencil resting on
 *   another pencil's POINT is held up by a point and not by the barrel's
 *   radius carried to the tip, which is what made pencils hover, twitch,
 *   and get lifted off the table by a neighbour's tip.
 *
 * t runs from 0 at the eraser to 1 at the lead. Duplicated t values are the
 * square shoulders where one part meets the next.
 */
export const LENGTH = 0.150;
export const BARREL_R = 0.0045;
export const GRIP_R = 0.0055;
export const MASS = 0.0062;

export const PROFILE = [
  [0.000, 0.00279],   // eraser, standing out of the cap
  [0.020, 0.00279],
  [0.020, 0.00387],   // cap
  [0.065, 0.00423],
  [0.065, BARREL_R],  // barrel
  [0.740, BARREL_R],
  [0.740, BARREL_R],  // grip, moulded so it swells out of the barrel. The
  [0.760, GRIP_R],    // repeated knots here and at 0.873 are zero-length
  [0.853, GRIP_R],    // and change no geometry; they exist so the drawing
  [0.873, BARREL_R],  // can put a crisp colour edge on the rubber.
  [0.873, BARREL_R],
  [0.929, BARREL_R],  // cone: barrel-coloured plastic on this pencil
  [0.929, 0.00437],
  [0.984, 0.00135],
  [0.984, 0.00108],   // lead
  [1.000, 0.00032],
];

/** Radius (m) at fractional position t. 0 outside [0, 1], so a caller that
 * runs off the end of the pencil gets no contact rather than a phantom one. */
export function radiusAt(t) {
  if (t < 0 || t > 1) return 0;
  // walk to the last knot at or before t, so a duplicated t takes the new
  // part's radius: at t = 0.065 the barrel wins over the cap
  let i = 0;
  while (i < PROFILE.length - 1 && PROFILE[i + 1][0] <= t) i++;
  if (i >= PROFILE.length - 1) return PROFILE[PROFILE.length - 1][1];
  const [t0, r0] = PROFILE[i], [t1, r1] = PROFILE[i + 1];
  return t1 > t0 ? r0 + ((t - t0) / (t1 - t0)) * (r1 - r0) : r1;
}

/** Radius (m) at signed distance `s` from the pencil's centre along its
 * axis, with the point at +s. The form the contact guard works in. */
export function radiusAtOffset(s, half = LENGTH / 2) {
  return radiusAt((s + half) / (2 * half));
}

/**
 * Which part each PROFILE knot belongs to, for colouring the drawn surface.
 * Indexed, not computed from t, because the shoulders are duplicated knots
 * that share a t and must take different colours.
 */
export const PARTS = [
  "eraser", "eraser",
  "cap", "cap",
  "barrel", "barrel",
  "grip", "grip", "grip", "grip",
  "barrel", "barrel",
  "cone", "cone",
  "lead", "lead",
];

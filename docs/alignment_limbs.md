# Limb pieces, collision resolution and "properly aligned"

How `alignment_engine` places pieces on arm and leg points (sleeves, cuffs, trouser legs), which way the solver moves a
piece that touches something, and what every answer is checked for. Code: `arrange.py` (`Layout._anchor`,
`clear_radius`, `_start_radii`), `solver.py` (body rows), `arrange_report.py` (`_placement_check`), `validity.py`
(`pair_violations`). Tests: `tests/test_sleeves.py`, `tests/test_point_sweep.py`.

## Axial anchoring of limb pieces (`anchor`)

A limb point (`UpperArm_Outer_R`, `Thigh_Front_L`, ...) is the middle of a short volume (the synthetic upper arm is
26 cm). A 62 cm sleeve centred there reaches 18 cm into the torso. So a piece rolled (`roll` or `around`) on an arm or
leg ring is registered along the limb by the placement field `anchor`:

| `anchor` | where the piece sits along the limb |
|---|---|
| `auto` (default) | `root` when the piece is longer than the point's volume, or when centred on the point it would reach into the torso; otherwise `point` |
| `point` | the reference on the point (the old behaviour) |
| `root` | the top edge of the pattern (sleeve cap, trouser waist) at the limb root: the piece slides along the limb to the position closest to the torso where no torso vertex lies under its footprint (outline buffered by `clearance_body`) within `clearance_body` of its cylinder |
| `tip` | the bottom edge of the pattern (hem) on the end of the limb (wrist, ankle joint) |

* The orientation is unchanged: the point's up runs toward the torso, so the pattern's top (the cap) is at the
  shoulder and the hem toward the wrist.
* The limb root is found on the body itself (the torso region of `shells.py`, which includes the shoulder cap), at the
  radius the piece will have, so "where the limb leaves the torso" adapts to the body and to the sleeve width. A
  `shift_m` given by the user is added on top (move the cap further down the arm with a negative `t`).
* Cuffs, patches and other pieces that fit inside the point's volume keep `point`. A placed piece keeps its place; its
  anchor is recomputed so that its slide is measured from the same place.
* The answer says what was done: `arrange.anchor = {mode, shift_m, reason}`.

## Which way a piece moves (collision resolution)

Every avatar triangle belongs to a body region (`shells.py`: torso incl. shoulder caps and hips, neck, arm_L/R,
leg_L/R; head, hands and feet belong to none). A rolled piece's own region is its ring.

1. **Start radius** (`clear_radius`): the smallest radius that keeps the body under the piece's footprint
   `clearance_body` inside the cylinder. A limb piece counts only its own limb: a radius cannot clear the torso at the
   arm root (the torso is not around the arm's axis), it only lifts the sleeve off the arm. Torso and neck pieces
   count every vertex (a larger radius shrinks the angular footprint, which is how a panel clears the arm beside it).
2. **Anchoring** (above) clears the torso at the root of a limb piece by an axial slide.
3. **Solver** (`solver.py`): a body sample too close to a triangle of the piece's own region may push the piece out
   (radius) or slide it; a sample too close to another region (the torso under a sleeve cap, a T-pose arm under a side
   panel, the head above a collar) may only slide it (`s` around, `t` along the axis), away from that part. Piece
   against piece: as before (side by side: apart; facing: the outer one out, which for a sleeve beside a torso panel is
   a slide outward along the arm).
4. **Radius growth** is the last resort, in 2 cm steps up to `max_standoff`, and is reported (`limits`).

## "Properly aligned": what every answer is checked for

`validity.valid` is true only if all of these hold (each failure is a violation with `pieces`, `value_m`,
`required_m`, `limit` and, for the placement checks, a `hint`):

| check | violation |
|---|---|
| every sample at least `clearance_body` from the body | `CLEARANCE_BODY` |
| every pair of pieces at least `gap_piece` apart (`gap_layer` facing) | `GAP_PIECES` |
| sizes exact | `SIZE_CHANGED` |
| mirror pair: equal radius and `t`, mirrored `s` (1 mm) | `PAIR_ASYMMETRY` |
| mirror pair on **one ring** (centre-line pair, or side pair on the torso ring): side by side | `PAIR_STACKED` |
| no `max_standoff` reached, and the standoff at the point within `max_standoff_from_point` (default 12 cm) or within what the ring needs (piece width, ring fit of the pieces side by side, body clearance, layer) plus `clearance_body` | `STANDOFF_EXCEEDED` |
| no `max_shift` reached, and the solver's slide from the anchor at most 15 cm (`MAX_SLIDE`) | `NOT_AT_POINT` |

* A side pair on two rings (left and right sleeve) is naturally far apart: only its symmetry is checked.
* A pair with exactly one locked half: its asymmetry is a `PAIR_ASYMMETRY` warning naming the lock (the free half
  cannot follow a piece the user fixed), not a violation.
* `arrange.limits` lists `max_standoff` / `max_shift` when reached (also when the radius ended exactly at
  `max_standoff`); a solver that stopped short of its 0.3 mm margin while the measured layout holds reports nothing.
  So `limits` is never non-empty on a valid answer.
* Per piece the answer explains itself: `arrange.reasons.radius` (piece width, ring fit, body clearance, layer, or
  how far the solver pushed it out), `arrange.reasons.shift` (solver slide or the anchor), `slide_m`,
  `point_distance_m` (point to the nearest part of the piece).
* Locked pieces are the user's: they are not checked for standoff or slide.
* `ARM_ZONE` (warning): a torso piece reaches into the space a sleeve needs around the arm (the arm axis beyond the
  shoulder joint, arm radius + clearance + gap); a sleeve placed there slides outward along the arm.

The sweep test (`tests/test_point_sweep.py`) places every piece of the shirt, the trousers and the skirt on every point
of its class, `roll` and `flat`, and checks on the result: sizes exact; a valid answer has no limits, keeps the
clearance, is not farther than `max_standoff_from_point` from the body, keeps the pattern's up along the point's up,
wraps its limb (radius <= 15 cm); an invalid one names the failure and how to fix it; left and right are mirror
images; the same call gives the same answer.

## Points (`avatar_engine`)

A point is a ray from its volume's axis to the surface, then `offset_m` out along the surface normal. Every resolved
point is checked: 0.5 to 3 radii (+ offset) from the volume's axis (flat hands and feet: at least the offset), normal
pointing away from the axis. A point that fails is re-cast from the volume vertex nearest the ray, and if that fails
too the build stops with a `PointConfigError` naming the point. (The synthetic body's legs are single 40 cm triangle
bands: the offset step ignored the hit triangle, whose vertices are all far from the hit, and walked to 3-11 m; the
hit triangle now always takes part.)

## Limits

* The T-pose is assumed: the arm ring is the straight line shoulder -> wrist, the leg hip -> ankle. Bent limbs need a
  piecewise ring.
* Mirror symmetry is exact (1 mm) on a symmetric body when the solver has little to do; a piece the solver slides far
  (a sleeve on a thigh, beside the other leg) can differ by a few millimetres left to right, because its iterations
  break ties by vertex order, which a mirrored mesh does not keep. Mirror *pairs* are exact always (the clone follows
  its source). On a real scan left and right are each registered on their own side
  (the pair still moves as one, from the source half).
* Trouser panels on a leg point: a cylinder around one leg cannot hold the rise (the part above the crotch) clear of
  the hips; they are registered below the hips or reported. Use a hip or waist point (torso ring) for a whole panel.
* `MAX_SLIDE` (15 cm) and `max_standoff_from_point` (12 cm, an option) are heuristics of "at its point".

# Mirror pairs, the garment ring and the vertical datum

How `alignment_engine` places a piece and its mirrored clone (`FrontPanel` / `FrontPanel_L`), and how front, back and
side panels of one garment end up concentric and on one chest line. Code: `arrange.py` (`Layout.prepare`,
`_find_pairs`, `_groups`, `_datum`, `_start_radii`), checks: `validity.py` (`pair_violations`, `concentric_warnings`).

## Pair model

A *pair* is a piece and a derived mirror clone of it (`derived_pieces`, `op: mirror`) placed

* on the same centre point (`same_anchor`, e.g. both on `Chest_Front`), same rotation, or
* on left/right counterpart points (`mirror_anchor`, e.g. `Chest_BackSide_R` / `_L`), rotation negated,

with the same `wrap` and the same `reference`. A pair is one rigid body: both halves share the ring, the axis, the
radius `r` and the axial shift `t`; the slide is mirrored, `s_clone = -s_source`. In the solver the clone has no
parameters of its own (it follows its source, `Piece.sign = (1, -1, 1)`), so a pair can only move symmetrically: both
out together, or both along the centre line.

A plain copy (`op: copy`, the web app's **Clone**) is never a pair, wherever it is placed: `engine.py` passes only
mirror clones to the layout as clone -> source, and every pair, group and `PAIR_*` check is built from that map.

Same-anchor halves start with their **centre-line edge** half a gap (`gap_piece / 2`) from the centre line, so the two
edges are exactly `gap_piece` apart and can be sewn. Which edge that is comes from the mirror axis of the clone (the
half lies on one side of it); for a `bbox_center` axis, from where most of the piece lies about its reference.

### Symmetric pair solve, outward direction, no crossing

The halves are mirror images about the plane x = 0 (the ring axis of a symmetric ring lies in it). A connected half and
its mirror image are `gap_piece` apart **exactly** when every vertex of the half is at least `gap_piece / 2` from that
plane on its own side. So the solver has no collision rows between the two halves of a pair; it keeps that bound instead
(`solver.pair_slide`):

* **Side.** `Piece.pair_side` (lead half only) is the world side it must stay on: the sign of its point's x for a side
  pair, the side of the mirror axis (`_centre_shift`) mapped to world x for a centre-line pair. Never local x about the
  reference: with a bbox-centre reference the samples near the centre line have local x > 0 on a right half, which used
  to push both halves *toward* each other until they passed through one another (swapped, `PAIR_STACKED`).
* **Outward.** The direction of `s` that moves the half's innermost vertex away from the plane (from world geometry,
  `d x / d s`). The pair is one 1-DOF problem in that slide: the smallest outward slide that gives the half gap (bisection;
  world x depends only on `x + s`), within `max_shift` of the anchor. `r` and `t` are untouched by it: a pair never
  grows its radius to part itself (the radius grows only for the body or other pieces, both halves together).
* **Hard bound.** Applied at the start (`Layout._sync_clones`: overlapping halves start apart, and the body-clearance
  radius is measured there) and after every solver step (`Solver.apply`): no step can close the pair or move a half
  across the plane.
* **No room.** If `max_shift` does not allow the slide, the halves stop at the limit, still each on its own side (never
  the crossed state), and the answer has one `PAIR_NO_ROOM` violation with the needed and the available slide and what to
  change (points farther apart, a reference nearer the centre-line edge, a narrower piece, a larger `max_shift`). It
  replaces the `GAP_PIECES`, `PAIR_STACKED` and `NOT_AT_POINT` violations of that pair.

Limb pairs (`UpperArm_*`, `Forearm_*`, `Thigh_*`) use the same rule; their halves are far from x = 0, so the bound holds
from the start and never moves them.

### Incremental `align` (the web app, one piece at a time)

* A half placed alone on a centre point (its clone defined but not placed yet) already leaves the half gap, and the
  ring is sized as if its other half were there (`fit_radius` counts it twice on the torso/neck ring).
* When the other half arrives, the placed half is solved again *with* it: the new half starts as the exact mirror image
  of the placed one (from its `solved` block), and the pair is only corrected if something collides, symmetrically.
  Both halves come back with new `solved` blocks; the web app replaces every `solved` block of the answer.
* Clone placed before its source works the same way (the source is derived from the clone).
* A locked half is never moved; a pair with a locked half is not linked (and is checked by validity, below).

## Reference on the mirror axis

A same-anchor pair joins exactly along the centre line only if the reference lies on the mirror axis. 

* `/api/pair` (`AlignmentEngine.pair`): a reference off the axis is replaced by the named point *on* the axis nearest
  in height to it (e.g. `bbox_centre` -> `A32` on the CF edge), reported as `REFERENCE_ADJUSTED` in `warnings`. With no
  named point on the axis it is still `ReferenceNotOnAxisError`.
* `align` / `arrange`: a pair half whose reference is not on its centre-line edge is placed by the edge (the shift `s`
  takes the difference) and gets a `REFERENCE_ADJUSTED` piece warning. Its reference still sets its height.

## One garment ring, concentric

* Every torso panel rolls on the **torso** ring: `Hip_*`, `Waist_*`, `Chest_*`, and `Neck_*` points for pieces taller
  than the neck volume (`RING_TORSO` warning). Collars (shorter than the neck) stay on the neck ring. An explicit
  `ring` in the placement wins.
* The ring radius is the smallest that fits every layer-0 piece of the ring side by side with gaps (`fit_radius`,
  row sums of the pattern widths, placed pieces included) and clears the body under the new pieces.
* Incremental: if the placed layer-0 pieces of the ring are too small a ring for the new ones, they **move out
  together** (radius only; `s`, `t` and their layers kept; `RING_GROWN` note, new `solved` blocks). The body alone
  never moves placed pieces: a new piece that needs more clearance sits further out by just that much.
* **Layer** is depth order for pieces in the *same angular sector* only (a pocket over a panel, a yoke over a back):
  footprints are compared on the unrolled ring at the fit radius (wrap-around aware); a piece covered `>= 40 %` by a
  larger one starts one `gap_layer` out. Front vs back and left vs right are side by side: layer 0. A placement's
  `layer` wins. The answer's `arrange.layer_reason` tells why a piece is above layer 0 (`user`, `over <piece>`,
  `solver` = a collision resolved radially), `arrange.pair_with` names the other half.

## Vertical datum (`pattern_origin_m`)

Piece JSON (`schema_version` 3) carries `pattern_origin_m`: where the piece origin (bbox bottom-left) lies in the
pattern's own drafting frame (y up, metres). Geometry and OBJ files are unchanged; schema 1 and 2 files still load
(without it there is no datum). A mirror clone keeps the source's heights (`pattern_origin_m - origin_shift_m`, vertical
axis only).

Front and back of a pattern are usually drafted on shared base lines (the shirt's front and back hems are on the same
drafting line), so `pattern_origin_m` puts them on one chest line whatever references the user picked:

* the first placed torso panel is the datum (else the tallest of the call); its reference fixes the garment height;
* every new torso panel of similar height (`>= 50 %` of the datum's, `DATUM_HEIGHT_RATIO`) that spans the datum's
  drafting line starts with the axial shift `t` that puts its drafting lines on the datum's (`DATUM` note); a user
  `shift_m` is added on top;
* pockets, yokes, plackets (much shorter) keep their own reference; rotated pieces are left out.

The first panel placed decides the absolute height: placing the back first and the front second puts the garment at
the back's height.

## Validity

Measured on every answer (`validity`):

| check | kind | effect |
|---|---|---|
| pair halves: equal `r` and `t`, `s` mirrored, within 1 mm | `PAIR_ASYMMETRY` | violation, `valid: false` |
| pair halves side by side: their arcs about the ring axis do not overlap | `PAIR_STACKED` | violation, `valid: false` |
| pair halves' centre-line gap `>= gap_piece` | `GAP_PIECES` (exact distance check) | violation |
| pair halves cannot slide apart within `max_shift` | `PAIR_NO_ROOM` (needed vs available slide, hint) | violation, replaces the pair's `GAP_PIECES` / `PAIR_STACKED` / `NOT_AT_POINT` |
| side-by-side pieces of one ring (arcs apart, heights overlapping) differ in radius by more than `gap_layer` + 5 mm | `NOT_CONCENTRIC` | warning (locked and user-layered pieces are exempt) |

`validity.pairs_ok` is false when a pair check fails. Sizes are never touched: every placement is still an isometry
(`size_ok`), `allow_distortion` stays false by default.

## Limitations

* Pairs assume a mirror axis that is vertical in the pattern and rotation 0 for the datum; a rotated same-anchor pair is
  linked as before (equal rotations), which is a true mirror only for rotation 0.
* The datum needs pieces drafted on shared base lines; patterns whose front and back are drafted at unrelated heights
  should set a reference on the same line in both, or place the panels and adjust `shift_m`.
* An asymmetric body: the ring axis stays in the sagittal plane and the pair stays mirrored, so one half may sit
  further from the body than it needs to (never closer than `clearance_body`). The no-crossing bound is computed on the
  lead half only; on an asymmetric avatar the clone's points are not exact mirror images, so its gap is only checked
  afterwards (`GAP_PIECES`).
* The mirror plane is x = 0, vertical. A garment mirrored about another plane is not a pair in this sense.
* `PAIR_NO_ROOM` comes from the solve: a pair carried over with both `solved` blocks is not solved again, and an
  overlap left over from an earlier answer shows as `GAP_PIECES` / `PAIR_STACKED`.
* A non-symmetric garment (pieces without clones) is placed piece by piece as before; only the ring radius and the
  layer rule apply to it.

# alignment_engine

Places flat pattern pieces onto avatar arrangement points and reports fit numbers. Metres, y up; the pieces, the
avatar and the placed pieces share one coordinate system. Nothing is scaled, and nothing is stretched: every placed
piece is an exact copy of its flat pattern (checked to 0.05 % after every placement), no piece comes closer than
`clearance_body` (3 cm) to the avatar and no two pieces closer than `gap_piece` (3 cm) to each other.

```powershell
alignment-engine run --avatar outputs/avatar/male_173 --pieces outputs/pieces --spec examples/placement.json --out outputs/aligned
```

```python
from alignment_engine import AlignmentEngine
result = AlignmentEngine().align(avatar_dir, piece_dir, spec_json_text, out_dir=None, cache_dir=None)
result.doc          # the alignment_result document (schema_version 4): placements, validity, arrangement
result.placed       # PlacedPiece: vertices, faces, transform, size_check, flat (the same mesh flat)
```

Inputs: the folder written by `avatar-engine` (`avatar_bundle.json` and the files it names), the folder written by
`sm2d-engine` (`*.piece.json` and the OBJs), and a placement spec. Outputs: `alignment_result.json`,
`<piece>_placed_m.obj` for each placement, `scene_m.obj` (avatar + placed pieces, one `g` group each). All files are
validated against `contracts/*.schema.json`.

## Placement spec (`contracts/placement_spec.schema.json`)

```json
{"schema_version": 1,
 "placements": [{"piece": "FrontPanel", "point": "Chest_Front", "offset_m": null, "rotation_deg": 0,
                 "mirror": false, "wrap": "flat", "flip_facing": false, "reference": "bbox_centre"}],
 "stitches": [{"a": {"piece": "A", "from_point": "P1", "to_point": "P2"}, "b": {"piece": "B", "from_point": "Q1", "to_point": "Q2"}}],
 "rings": [{"name": "chest", "pieces": ["FrontPanel", "BackPanel"], "avatar_measurement": "chest_circ"}]}
```

Only `piece` and `point` are required. Defaults: `offset_m` null (the point's own offset), `rotation_deg` 0,
`mirror` false, `wrap` **flat** (`flat` | `roll` | `around`; `body_slice` only with `allow_distortion`), `flip_facing`
false, `reference` `bbox_centre` (or any named point of the piece). Also: `solved` (`{radius_m, shift_m}` from an earlier
answer: the piece stays exactly there), `locked`, `ring`, `layer`, `shift_m`, and `arrange_options`
(`clearance_body` 0.03, `gap_piece` 0.03, `gap_layer` = `gap_piece`, `max_standoff` 0.30, `max_shift` 0.25,
`refine_max_edge` 0.005, `incremental` true, `allow_distortion` false). `schema_version` 1 to 4 is accepted.

## Size-exact surfaces and the solver (`roll.py`, `solver.py`, `arrange.py`, `validity.py`)

Every placement is an isometry of the flat piece with three free numbers `theta = (r, s, t)`:

* `flat`: `origin + r n + (x + s) e_x + (y + t) e_y` (rigid; `r` the distance along the point normal);
* `roll` (default in the web app on torso, neck, arm and leg points): rolled onto a cylinder about the axis of the
  body region (ring) of the point: angle `(x + s) / r`, axial position `y + t`, radius `r`;
* `around`: the same on the axis of the point's own bounding volume.

A cylinder is developable, so no length, angle or area changes. Pieces are refined to edges <= 5 mm first. The solver
changes only `theta` until every sample of every piece is `clearance_body` from the avatar and every pair of pieces is
`gap_piece` apart (exact point-triangle distances, all pairs, all layers, fixed pieces too); pairs that face each other
are separated radially, and a piece being placed goes on top. `align` is incremental: placements with `solved` stay
where they are, the others are solved in spec order against them. `arrange` re-solves everything that is not locked.
After every solve `validity` measures sizes and distances again on the placed meshes; a size change of more than 0.05 %
raises `SizeChangedError`, a remaining collision makes `validity.valid` false with the exact violations and the limit
that stopped the solver (`max_standoff`, `max_shift`, `iterations`). Details and limits:
`docs/LLD_avatar_alignment_web.md` section 12.

## Rigid maths (`placement.py`, `wrap.py`; pure functions, unit tested)

* **Frame.** `n` = point normal, `u` = point `up` projected perpendicular to `n`, `t = u x n`. The rigid map sends
  piece x to `t`, y to `u`, z (out of the flat face) to `n`: a proper rotation. Then `rotation_deg` about `n`
  (counter-clockwise), then the reference moves to `position + (offset - point_offset) * n`. **`offset_m` replaces the
  point's default offset** (the point's `position_m` already contains it), so with no offset given the reference lands
  exactly on `position_m`.
* **`flip_facing`** turns the piece's face to `-n`; the offset still pushes the piece away from the body.
* **`mirror`** reflects the piece across `x = reference.x` before placing and flips the triangle winding, so every
  triangle normal keeps its direction. The result matrix includes the reflection (`mirrored: true`); renderers must
  flip winding or draw double-sided.
* **`wrap: "around"` / `"roll"`** bend the piece around an axis (`roll.Surface`): a vertex at offset `(s, h)` from the
  reference (after the user rotation) goes to angle `phi = s / R` about the axis and height `h` along it. `R` starts at
  the distance from the axis to the target and is never smaller than what keeps the piece from overlapping itself; the
  solver may make it larger. `wrapped_vertices` / `wrapped_faces` are the refined (5 mm) mesh; the matrix is the tangent
  frame at the reference (exact for flat pieces). `wrap.wrap_vertices` is the pure single-cylinder function the tests
  compare against.

## Half pieces: clone & mirror, plain clones (copies), derived pieces, pairs

The `.sm2d` samples carry no quantity or fold data, so the engine cannot know that `FrontPanel` is one half of a pair.
Like CLO3D, it keeps **two pieces** (never one merged polygon): a **derived piece** is a linked mirrored copy.

```json
"derived_pieces": [{"id": "FrontPanel_L", "source": "FrontPanel", "op": "mirror",
                    "axis": {"type": "edge", "from_point": "A33", "to_point": "A32"}}]
```

* `axis`: `bbox_center` (flip in place), `edge` (the line through two named points, any direction) or
  `x` (a vertical line at piece x). The derived piece is built first and then placed like any piece.
* `mirror_clone(piece_bundle, vertices, faces, axis, new_id)` (`mirror.py`) reflects the mesh, `outline_m`, named points,
  edges, bbox, default reference and grainline, fixes the outline order and triangle winding (normals stay +z), shifts the
  geometry so the bbox starts at (0, 0) (`origin_shift_m`), keeps the point names and records `mirror_of` and its own
  `axis` (mirroring the clone with it restores the source to 1e-12 m). Edge lengths, area and perimeter are preserved
  exactly. Derived-of-derived is an error (`DerivedOfDerivedError`); ids must be unused (`DuplicatePieceError`).
* Derived meshes are cached as `<id>_m.obj` + `<id>.piece.json` (pass `derived_dir` or `out_dir`).
* For an `edge` axis the result suggests the seam (`suggested_stitch`) between that edge on both copies.

**Copy (plain clone) vs mirror.** `{"id": "Pocket_copy", "source": "Pocket", "op": "copy"}` (no `axis`) is an exact
copy, same orientation, as CLO3D "Duplicate". Use a **copy** for identical pieces and pieces cut twice that are not
left/right halves (two pockets, belt loops, a second collar stand, a symmetric piece cut twice); use a **mirror** for
left/right halves (shirt fronts, sleeves of an asymmetric pattern, trouser legs).

* `copy_clone(piece_bundle, vertices, faces, new_id)` (`mirror.py`) keeps mesh, outline, named points, edges, fold edge,
  grainline, bbox, default reference and `pattern_origin_m`; it records `clone_of`, no `mirror_of` / `axis`,
  `origin_shift_m = [0, 0]`. A missing `op` means `mirror` (old specs load); `axis` on a copy or no `axis` on a mirror is
  an error. A copy may be made of a derived piece defined earlier in the list (a copy of a mirror clone copies the
  mirrored geometry, `clone_of` names the clone); a mirror of any derived piece is `DerivedOfDerivedError`.
* **A copy is never half of a pair**: only `op: mirror` enters the clone -> source map the layout builds pairs, groups,
  `PAIR_*` checks and the half-gap logic from, and `pair()` refuses a copy. A copy on the counterpart point with a
  negated rotation is two independent pieces (gap, layers and body clearance as for any two pieces); a copy on the same
  point as its source is layered out like any overlapping piece. In the result a copy lists `op`, `source`, `obj`,
  `bundle`, `bbox_cm` (no `axis`, no `suggested_stitch`).

`AlignmentEngine.pair(avatar, pieces, derived_pieces, base_placement, derived_id, mode)` returns the two placements:

| mode | what | when |
|---|---|---|
| `same_anchor` (centre-line pair) | same point, reference, offset, wrap, rotation | e.g. the two halves of a shirt front on `Chest_Front`. The reference must lie on the mirror axis (1 mm), else `ReferenceNotOnAxisError`: only then do the CF edges coincide exactly |
| `mirror_anchor` (side pair) | the clone goes on the counterpart point (`_L`/`_R`, `_Left`/`_Right`; a centre point maps to itself), rotation negated | sleeves, trouser legs. `NoMirroredPointError` if there is no counterpart |

A half piece is anchored by a seam point (for example a point on the centre-front edge), not by its bbox centre: choose
that point as `reference`. Nothing is simulated, so the placement itself has to be accurate.

## Side awareness (warnings, not errors)

Each placement reports `extends` (`left` / `right` / `centre`): where the area-weighted centroid of the placed piece lies
relative to its reference along the wearer's x axis (centre if within 1 cm or 5 % of the width). Warnings are
`CODE: text` strings: `SIDE_UNLIKELY` (a piece whose name, or whose source's name, contains `front` on a `Back` point
or the reverse), `SIDE_MISMATCH` (an `_R` point and a piece extending left, or `_L` and right), `WRAP_OVERRUN`,
`WRAP_FALLBACK`, `BODY_SLICE_RANGE`.

## Distorting torso wrap: `wrap: "body_slice"` (only with `allow_distortion: true`)

**This mapping changes the piece's measurements** (diagonal edges stretch where the torso is double curved). It is kept
only for comparison: without `arrange_options.allow_distortion` the size check raises `SizeChangedError`; with it the
placement says `changes_piece_size: true`, `validity.valid` is false, the piece is never moved (an obstacle), and the web
app does not offer it. On a torso point (Neck, Chest, Waist, Hip; a limb point falls back to `around`, which is
isometric, with `WRAP_FALLBACK`) `body_slice` uses the avatar's own horizontal cross-sections:

* The torso is found from the bounding volumes (axis polyline through Hip, Waist, Chest and Neck; triangles inside the
  arm volumes are left out, except the shoulder cap), sliced every 1 cm from the crotch to mid neck into radial profiles
  (`theta` 0 = front, 90 = +x), smoothed, and cached for the next call. Gaps where an arm was cut out are interpolated.
* A piece vertex at row `h`, offset `s` from the reference goes to world height `y_anchor + h` and to the point of that
  height's loop whose arc length from the anchor angle is `s`, on the skin loop pushed out by the offset. **Arc length
  along every row and vertical distance in world height are exact.**
* The result's `fit` reports, at 8 rows and at the chest, waist and hip levels, the garment width against the body loop:
  `ease_quarter_cm` (a half front or back covers about a quarter of the loop) and `ease_half_cm`.
* Limits: along steep surfaces (shoulder slope, bust) the vertical distance is preserved in world height, not along the
  surface; the offset is horizontal, so on a slope the perpendicular distance to the skin is a little smaller; the arm
  holes are closed by interpolation; a piece taller than the torso range (crotch to mid neck) uses the end sections and
  warns (`BODY_SLICE_RANGE`); offsets beyond half the loop are clamped and warn (`WRAP_OVERRUN`).

## Results (`alignment_result.json`)

Per placement: `extends`, the 4x4 row-major `matrix` (original piece vertices to world; for a rolled piece the tangent
frame at the reference), `mirrored`, `wrapped_vertices` and `wrapped_faces` (or null for flat pieces),
`reference_position_m`, `bbox_cm` (the **flat** width and height: unchanged by placement), `warnings`, `fit` (placed 3D
extent, standoff, and for rolled pieces the radius, circumference and angle covered), `size_check` (measured:
`max_edge_change_pct`, `perimeter_change_pct`, `area_change_pct`, `named_point_change_pct`, `max_mesh_edge_m`),
`changes_piece_size`, `solved` (`{radius_m, shift_m}`: send it back to keep the piece there), `arrange` (ring, mode,
layer, radius, standoff, shift, exact distances to the avatar and to the pieces within 6 cm, flags, limits hit) and the
world outline, edges and named points.

Top level: `validity` (`valid`, `size_ok`, `clearance_ok`, `gaps_ok`, `max_size_change_pct`,
`min_distance_to_avatar_m`, `min_distance_between_pieces_m`, `violations[]` with pieces, value, requirement, place and
limit) and `arrangement` (options, rings with the true ease, seam candidates, violations, timing).

**Stitch report** (no simulation): the outline between two named points (the shorter way round), the two lengths in cm,
`diff_cm = a - b`, `diff_pct` relative to `b`, and 21 samples per edge resampled by arc length and paired into drawable
segments (world coordinates, only if both pieces are placed).

**Ring check**: sum of the widths of the listed pieces against the avatar's measurement; ease in cm and %. A listed
piece also counts its derived clones (`include_clones`, default true), so `["FrontPanel", "BackPanel"]` with both mirrored
is `2 x front + 2 x back`. The width is the bbox width, or with `level_point` the width of the outline at the height of
that named point (for example the chest line; a missing point falls back to the bbox width with a warning).

## Errors

`UnknownPieceError`, `UnknownPointError`, `UnknownNamedPointError`, `UnknownMeasurementError`, `BundleError`,
`DerivedOfDerivedError`, `DuplicatePieceError`, `AxisError`, `ReferenceNotOnAxisError`, `NoMirroredPointError`,
`RingNotFoundError`, `SizeChangedError` (a placement would change the piece; never returned), and `ContractError` for
documents that do not match their schema. A layout that cannot be made collision-free is not an error: it is returned
with `validity.valid = false` (warning code `ArrangementInfeasible`).

## Auto-arrange

`AlignmentEngine.arrange(avatar_dir, piece_dir, spec_json_text, out_dir=None, derived_dir=None, cache_dir=None)` re-solves
every piece that is not locked, jointly, from the anchors: pieces of a ring start side by side on one cylinder whose
radius is the smallest that fits them with gaps and clears the body under them, pieces lying over others one gap further
out, and the solver separates the rest. Same document as `align`. The standoff (roll radius minus the skin radius) is
display space, never ease: ease is reported per ring as garment loop minus skin loop, independent of every option.
`cache_dir` keeps the 5 mm meshes between runs. Web: `POST /api/align` (every placement action) and `POST /api/arrange`
(the Auto-arrange button); both answer with `validity`.

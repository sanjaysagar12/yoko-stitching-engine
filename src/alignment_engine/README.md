# alignment_engine

Places flat pattern pieces onto avatar arrangement points and reports fit numbers. Metres, y up; the pieces, the
avatar and the placed pieces share one coordinate system. Nothing is scaled.

```powershell
alignment-engine run --avatar outputs/avatar/male_173 --pieces outputs/pieces --spec examples/placement.json --out outputs/aligned
```

```python
from alignment_engine import AlignmentEngine
result = AlignmentEngine().align(avatar_dir, piece_dir, spec_json_text, out_dir=None)  # out_dir=None writes nothing
result.doc          # the alignment_result document
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
`mirror` false, `wrap` **flat** (`flat` | `around` | `body_slice`), `flip_facing` false, `reference` `bbox_centre` (or
any named point of the piece). `schema_version` 1 or 2 is accepted; 2 adds `derived_pieces` and the result fields below.

## Maths (`placement.py`, `wrap.py`; pure functions, unit tested)

* **Frame.** `n` = point normal, `u` = point `up` projected perpendicular to `n`, `t = u x n`. The rigid map sends
  piece x to `t`, y to `u`, z (out of the flat face) to `n`: a proper rotation. Then `rotation_deg` about `n`
  (counter-clockwise), then the reference moves to `position + (offset - point_offset) * n`. **`offset_m` replaces the
  point's default offset** (the point's `position_m` already contains it), so with no offset given the reference lands
  exactly on `position_m`.
* **`flip_facing`** turns the piece's face to `-n`; the offset still pushes the piece away from the body.
* **`mirror`** reflects the piece across `x = reference.x` before placing and flips the triangle winding, so every
  triangle normal keeps its direction. The result matrix includes the reflection (`mirrored: true`); renderers must
  flip winding or draw double-sided.
* **`wrap: "around"`** bends the piece around the bounding-volume axis of the point. A vertex at offset `(s, h)` from the
  reference (after the user rotation) goes to angle `phi = s / R` about the axis and height `h` along it, where `R` is the
  distance from the axis to the target. This preserves arc length along x and distance along y. A piece wider than
  `2 pi R` triggers `WrapWidthExceededWarning` and is clamped, never scaled. The mesh is first subdivided to edges of at
  most 1 cm so triangles do not cut through the body, so `wrapped_vertices` / `wrapped_faces` are a refined mesh; the
  matrix is the underlying rigid step.

## Half pieces: clone & mirror, derived pieces, pairs

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

## Torso wrap: `wrap: "body_slice"`

`flat` leaves a large piece floating as a tangent plane and `around` bends it on one cylinder; a torso is neither. On a
torso point (Neck, Chest, Waist, Hip; a limb point falls back to `around` with `WRAP_FALLBACK`) `body_slice` uses the
avatar's own horizontal cross-sections:

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

Per placement: `extends`, the 4x4 row-major `matrix` (original piece vertices to world), `mirrored`, `wrapped_vertices` and
`wrapped_faces` (or null), `reference_position_m`, `bbox_cm` (the **flat** width and height: unchanged by placement),
`warnings`, and `fit`: placed 3D extent, nearest avatar vertex distance (an approximation: vertex spacing is about 1 cm),
and for wraps the radius, circumference, angle covered and width over circumference.

**Stitch report** (no simulation): the outline between two named points (the shorter way round), the two lengths in cm,
`diff_cm = a - b`, `diff_pct` relative to `b`, and 21 samples per edge resampled by arc length and paired into drawable
segments (world coordinates, only if both pieces are placed).

**Ring check**: sum of the widths of the listed pieces against the avatar's measurement; ease in cm and %. A listed
piece also counts its derived clones (`include_clones`, default true), so `["FrontPanel", "BackPanel"]` with both mirrored
is `2 x front + 2 x back`. The width is the bbox width, or with `level_point` the width of the outline at the height of
that named point (for example the chest line; a missing point falls back to the bbox width with a warning).

## Errors

`UnknownPieceError`, `UnknownPointError`, `UnknownNamedPointError`, `UnknownMeasurementError`, `BundleError`,
`DerivedOfDerivedError`, `DuplicatePieceError`, `AxisError`, `ReferenceNotOnAxisError`, `NoMirroredPointError`, and
`ContractError` for documents that do not match their schema.

## Auto-arrange

`AlignmentEngine.arrange(avatar_dir, piece_dir, spec_json_text, out_dir=None, derived_dir=None, cache_dir=None)` spreads the pieces
of a placement spec on offset body shells so nothing overlays and seams can be drawn (spec fields `ring`, `layer`, `locked`,
`shift_m`, `arrange_options`; result `schema_version` 3 with an `arrangement` block). The standoff is display space, never
ease: ease is reported as garment loop minus skin loop per ring. Algorithm and limits: `docs/LLD_avatar_alignment_web.md`, section 12.
Web: `POST /api/arrange`, the Auto-arrange panel.

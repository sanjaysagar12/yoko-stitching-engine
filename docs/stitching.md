# Stitching: stitching JSON -> sewing lines with point-to-point correspondences

The stitcher (`src/stitcher`) reads an assembly's alignment result and a hand-written stitching JSON and writes the
sewing lines of every seam on the already placed pieces, with point-to-point correspondences. It moves nothing: pieces
and avatar keep their real size and pose (checked by a hash before and after). It is not a simulation; its output,
`stitching.result.json`, is the input of the later simulation phase.

```
python -m stitcher suggest --assembly assert/assemblies/<g>.assembly.json [--synthetic]   # candidates + edge table
python -m stitcher run     --stitching assert/stitchings/<g>.stitching.json [--synthetic] [--out DIR] [--fast]
python -m stitcher check   --stitching ... [--result DIR/stitching.result.json]          # independent re-check
python -m stitcher render  --stitching ... [--result ...]                                 # images only
python -m assembler stitch assert/assemblies/<g>.assembly.json --stitching ... --out DIR [--synthetic]
```

`run` exits 1 when the verification fails, 2 on an invalid file. Default output: `build/stitching/<g>-syn|real/`:
`stitching.result.json`, `stitch_verify.json|md`, `images/`.

## Inputs, and the engine boundary

The stitcher imports no engine. It reads files only:

* the output folder of `python -m assembler run` (`result.json`: the alignment result with `world_outline_m`,
  `world_edges`, `world_named_points`, `wrapped_vertices/faces`; `verify.json`: the input folders and the size audit;
  `derived/`: the derived piece bundles), the piece bundles (`*.piece.json`) and the avatar OBJ it names;
* without `--run DIR`, the run is taken from `build/_stitch_runs/<garment>-syn|real-<key>` or made there by a
  subprocess `python -m assembler run --fast --no-render`. The key hashes the assembly, the job's sm2d/smis/body files,
  the SMPL-X version, `tests/synthetic_avatar.py` (synthetic) and the sources of the three engines plus
  `assembler/run.py` and `assembler/verify.py`.

Its only foreign imports are the CPU renderer `assembler.render` and the exact point-triangle code of
`assembler.verify` (`tests/stitcher/test_stitcher.py::test_stitcher_imports_no_engine`). The web backend and the
assembler may import the stitcher.

## Stitching JSON (`contracts/stitching.schema.json`)

```json
{
  "schema_version": "1.0", "garment": "male_shirt", "assembly": "assert/assemblies/male_shirt.assembly.json",
  "defaults": {"samples_per_cm": 2, "max_length_diff_pct": 3, "max_gap_m": 0.3},
  "seams": [{
    "id": "side_L", "label": "left side seam", "kind": "plain",
    "a": {"piece": "FrontPanel", "chain": ["A24", "A30", "A47"]},
    "b": {"piece": "BackPanel",  "chain": ["A24", "A31", "A48"], "reverse": false},
    "ease": {"mode": "none"}, "tolerance_pct": 2, "required": true
  }],
  "ignore_edges": [{"piece": "FrontPanel", "chain": ["A47", "A8", "A33"], "reason": "hem"}],
  "expect": {"required_seams": ["side_L"]}
}
```

* `assembly`: repository path or `sha256:<hex>` of a file in `assert/assemblies` (`--assembly` overrides).
* A **side** is one part `{piece, chain, reverse?, trim_start_m?, trim_end_m?}`, a part `{piece, line}`, or
  `{parts: [...], reverse?}` sewn one after the other (an armhole of front, yoke and back). Side b may instead be
  `{onto: <piece>}` (kind `attach`).
* **chain**: named points on the outline in sewing order. The outline is the cycle of the piece's on-outline named
  points ordered by `outline_s_m`; the chain walks it from the first to the last point the way that passes every listed
  point in order (wrap-around and zero-length edges included). A list that is exactly one way's points wins; points
  coinciding with a neighbour (D3/D13 of the collar stand) do not count. If both ways pass the points the chain is
  ambiguous (error naming the points to add); unknown or off-outline points are errors naming the valid ones.
* **trim_start_m / trim_end_m**: cut that much off the chain as listed, for seams that end between named points
  (the shirt's collar stand ends mid button stand; the sleeve slit runs on into 6.13 cm of hem).
* **line**: a straight polyline through named points inside the piece (a dart leg down to its apex): flat exact,
  world linear between the named points' world positions and put onto the placed surface (closest point).
* **reverse** flips a part; on a `{parts}` side it reverses the part order and every part. The engine never flips.
* **kind**: `plain`; `ease` (ease mode `distribute`); `gather`; `fold` (both pieces end on the same side of the seam:
  a facing turned in, a pocket bag sewn round its edge, a placket folded over a slit); `join_mirror` (a piece and its
  `mirror_of` / `clone_of` twin along the centre edge); `dart` (two legs of one piece); `attach` (side a sewn onto the
  surface of the host `{onto}`: patch pockets, flaps, a fly shield).
* **ease** `{mode, ratio}`: `none` lengths equal within `tolerance_pct` (default `defaults.max_length_diff_pct`);
  `distribute` up to 5 %; `gather` up to the declared `ratio` (long / short), recorded for the simulation.
* **allow_far**: the world gap may exceed `defaults.max_gap_m`. **allow_crossings** `"<reason>"`: the initial layout
  crosses although the seam is oriented right (pieces placed apart or rotated, a jog in the outline, a step between
  layers); crossings are then reported, not judged. The sewing rule is always judged.
* `ignore_edges`: `{piece, edge | chain, reason}`, the outline that is not sewn (hems, finished edges, closures).

## Engine (`stitcher.chains`, `stitcher.seam`, `stitcher.result`)

* Geometry per placed piece: knots at every outline vertex (`world_outline_m`, same index as the bundle's
  `outline_m`) and every `world_edges` sample (arc-length samples between named points), each with its flat arc
  position. A point at any arc length is interpolated between knots (sub-vertex). The placed mesh vertices are never
  indexed for outline points.
* Lengths are flat (true) arc lengths; the world polyline length is reported as a cross-check.
* Correspondences: `N = ceil(samples_per_cm * max(length_a, length_b) * 100) + 1` pairs at normalised arc length
  `s_i = i / (N - 1)` on each side (ends included), so any length difference is spread along the whole seam. Attach:
  each side-a sample is paired with the closest point of the host's placed surface.
* Orientation:
  * **sewing rule**: walking a chain, its piece lies left (+1) or right (-1) of it in the flat frame (chain direction x
    outline winding). A join leaves the two pieces on opposite sides; a `fold` on the same side.
  * **crossings**: pairs of sewing segments (`pa_i -> pb_i`, at most N/4 samples apart) that intersect in their own
    plane (side a's step and the mean across direction). Coincident pairs (< 1 mm) are skipped. The reversed pairing
    is counted too: a better reversed pairing is reported as "probably reverse: true".
* Coverage: every outline edge is sewn, ignored or unstitched (more than 5 mm neither sewn nor ignored); two claims
  sharing more than 1 mm of an edge are an overlap. Sewing segments more than 1 cm inside the body are reported.

## Verifier (`stitcher.verify`, `python -m stitcher check`)

Independent of the engine: chains re-walked on the alignment result's `world_edges` graph (not the bundle's arc order),
lengths summed from the alignment engine's edge lengths, positions between named points interpolated along the
`world_edges` samples, claims and coverage kept per edge, crossings recounted by separate code, attach samples
re-projected with `assembler.verify.Surf`, the sewing rule from the edge order and the bundle's shoelace winding.

Fails: invalid result schema; stitching / assembly / alignment result changed since the run; unresolved seam (unknown
piece or point, not contiguous, ambiguous); chain not through the declared points; flat length off the alignment
engine's (> 0.01 cm); world length off the flat length (> 2 %); ratio outside the ease limit; samples not monotonic or
wrong count; ends not at the chain ends (> 1 mm); NaN or repeated points; gap > `max_gap_m` without `allow_far`;
crossings without `allow_crossings`; sewing rule broken; attach sample not the host's closest point (> 1 mm); an edge
claimed twice (> 1 mm; also a piece sewn to itself over the same edge); a required seam missing (seams with `required`,
`expect.required_seams` of the stitching or the assembly file); geometry hash changed; any `size.*` row of the
assembler's size audit failing or a non-rigid matrix; the engine's unstitched edges differing from the independent count.
Warnings: orphan edges, gathers over 1.5, sewing segments through the body.

## Result (`contracts/stitching_result.schema.json`): what the simulation phase reads

Units: metres (world and uv), `*_cm` lengths in cm. World frame: the alignment result's (y up, +z front, +x the
wearer's left). uv: the piece bundle's `outline_m` frame (the `<piece>_m.obj` frame).

```
{schema_version "1.0", garment, units "m", convention, inputs{stitching, assembly, run, *_sha256, synthetic},
 defaults, seams[], errors[], overlaps[], unstitched_edges[], ignored_edges[], zero_length_edges[],
 pieces_untouched_hash{pieces, avatar}, untouched, required_seams[], ease_modes{}, summary{}}
seam: {id, label, kind, required, allow_far,
       a|b: {piece, edge_ids[], polyline_m[[x,y,z]], parts[{piece, chain[], direction outline|against_outline|line|onto,
              trim_m[2], edge_ids[], length_cm, world_length_cm, polyline_m}]},
       length_a_cm, length_b_cm (flat), world_length_*_cm, ratio (a/b),
       ease{mode, declared_ratio, tolerance_pct, applied_ratio (long/short), gathered_side a|b|none, excess_cm},
       samples{count, s[], a_xyz_m[], b_xyz_m[], a_uv_m[], b_uv_m[] | null, a_part[], b_part[], a_arc_m[], b_arc_m[],
               b_tri[], b_bary[] (attach only)},
       world_gap_m{min, mean, max}, orientation{ok, sewing_rule_ok, interior{a[], b[]}, crossings_judged,
       allow_crossings, crossing_count, reversed_crossing_count, window, hint}, through_body{segments, max_depth_m},
       warnings[]}
```

For the simulation:

* Pair `i` of a seam joins `a_xyz_m[i]` and `b_xyz_m[i]`; on the flat pieces they are `a_uv_m[i]` / `b_uv_m[i]` of
  the part `a_part[i]` / `b_part[i]`, at flat distance `a_arc_m[i]` / `b_arc_m[i]` along that part's chain.
* Attach seams have no b chain: side b is the host's placed mesh at triangle `b_tri[i]` of `wrapped_faces` with
  barycentric `b_bary[i]` (`b_uv_m` is null).
* Ease and gathers: the longer side (`gathered_side`) carries `applied_ratio` times the shorter one's length, spread
  uniformly by arc length. `gather` seams in the committed files: shirt cuffs 1.55 (two pleats + slit), shirt yoke
  seams 1.11 (box pleat), trouser front darts 1.09 (asymmetric legs as drafted).
* Gaps are large on purpose where pieces are placed apart (shirt shoulders and neck up to 49 cm, trouser inseams 44
  cm): the simulation closes them. `allow_crossings` seams start crossed (rotated or layered layout); their
  orientation is guaranteed by the sewing rule, not by the initial layout.
* Closures (shirt centre front, trouser fly and waistband ends) are sewn shut for the simulation.

## Images (`stitcher.render`, CPU, deterministic)

`images/stitch_{front,back,left,right,top,three_quarter}.png` (seams on both pieces in their colour, every few
correspondences, id labels; hidden parts thin and pale), `stitch_seam_<id>.png` (close-up from outside, the flat pieces
with the chains, a dot at s = 0 and a square at s = 1, the numbers and the seam's verification rows),
`stitch_unstitched.png` (every piece flat, edges in their seam's colour, unstitched red, ignored grey),
`stitch_sheet.png`.

## Web app

The stitching panel (bottom left of the viewer): Import stitching, Stitch (the server aligns the layout on screen
incrementally, so nothing moves, then runs the stitcher), the seam list with ratio, ease and verdict (click: highlight
and zoom), Export JSON, Export result (`stitching.result.json`), Export images (zip: result, verification, every stitch
image). Backend: `POST /api/stitching/import|run|result|export` (`src/web/backend/stitching.py`); the run folder comes
from `assembler.webio.stitch_run_folder`. The mirror-clone `suggested_stitch` of `/api/clone` is unchanged and feeds
`suggest` through the alignment result's `derived_pieces`.

## suggest

`python -m stitcher suggest` writes `build/stitching/<garment>.suggested.json` (never the hand-written file): the edge
table, mirror / clone joins from `suggested_stitch`, chain matches (contiguous runs of up to 6 edges whose flat lengths
agree within 3 %, scored by ratio, world distance of the ends and shared Seamly labels; a dart's legs share exactly one
end) and the alignment engine's `seam_candidates`, with confidence and the edges nothing covers. Composite sides, gathers,
trims and attach seams are not guessed.

## The loop and the committed files

`assert/stitchings/{male_shirt,aldrich_skirt,trousers}.stitching.json` were written with the loop (suggest -> write ->
run -> look at every image -> fix) on the synthetic and the SMPL-X avatar; the iterations are in
`docs/stitching_log_<garment>.md`, the final sheets in `docs/stitchings/`. Tests: `tests/stitcher/`.

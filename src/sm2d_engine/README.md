# sm2d-engine

Turns a Seamly2D pattern (`.sm2d`) plus an individual measurement file (`.smis`) into **real-scale meshes**, one
OBJ per pattern piece. **1 OBJ unit = 1 metre**: a 60 cm piece is 0.60 in the file. The purpose is a fit check of
pattern pieces against an SMPL avatar, so the scale is never normalised, fitted to a box or adjusted by eye.

Design notes: [docs/LLD.md](../../docs/LLD.md) (this engine) and [docs/LLD_avatar_alignment_web.md](../../docs/LLD_avatar_alignment_web.md) (how it connects to the avatar, alignment and web components).

## Install

```
pip install -e ".[dev]"      # numpy, mapbox_earcut, pydantic (+ pytest, mypy, ruff)
pytest                       # 48 tests, includes the 8 acceptance tests
```

Python >= 3.10. Input files live in `assert/sm2d` and `assert/smis` (the folder name is deliberate).

## CLI

```
sm2d-engine run job.json
sm2d-engine run job.json --out ./meshes        # overrides job output_dir
sm2d-engine inspect job.json                   # lists pieces + bbox, writes nothing
```

Options: `--sm2d-dir`, `--smis-dir` (or env `SM2D_ENGINE_SM2D_DIR`, `SM2D_ENGINE_SMIS_DIR`), `-v` for debug logging.
Exit code: `0` all pieces ok, `1` some pieces failed (the others are still written), `2` fatal (bad job, unreadable
file, invalid XML, unknown unit, unsupported format).

The CLI reads the files and passes their *text* to the engine (`EngineRequest(sm2d_xml, smis_xml, options)`).
The engine writes the meshes and `report.json` through a `MeshWriter`.

### job.json

Unknown keys are an error.

| key | default | meaning |
|---|---|---|
| `sm2d` | required | file name resolved against `assert/sm2d` (or `--sm2d-dir`); absolute paths accepted |
| `smis` | required | same, against `assert/smis` |
| `output_dir` | `"out"` | relative to the current directory |
| `pieces` | `null` | list of piece names, `null` = all |
| `format` | `"obj"` | `"obj"`; `"ply"` is a stub and is a fatal error |
| `combined_layout` | `true` | also write all pieces left to right, one `g <name>` group each |
| `layout_gap_cm` | `5` | gap between pieces in the combined layout |
| `mirror_fold_pieces` | `false` | only acts if fold information exists, otherwise a warning is recorded |

### Output

Besides the OBJ files and `report.json`, every piece gets a `<piece>.piece.json` (contract: `contracts/piece_bundle.schema.json`) with `outline_m`, the Seamly `named_points` that lie on or inside the outline (with their arc position on the outline), `edges` between consecutive named points on the outline (their lengths sum to the perimeter), `fold_edge` (`null` unless the pattern defines one; never guessed), `grainline` (or `null`) and `default_reference` (the bbox centre). The alignment engine reads these.


* `<piece_name_slug>_m.obj`: vertices in metres, z = 0, x right, y up, origin = piece bounding-box bottom-left.
  Header comments record units, origin, piece name, size, source `.sm2d` and `.smis` names.
* `<pattern>_all_pieces_layout_m.obj`: optional combined layout (group names are the slugs).
* `report.json`: per piece `width_cm, height_cm, area_cm2, perimeter_cm, mesh_area_cm2, area_diff_pct, vertices,
  triangles, self_intersections, gaps, status ("ok"/"failed"), reason, warnings, quantity, on_fold, file`; plus the
  pattern and measurement file names, units, flatten tolerance and tool versions.
* OBJ and report are deterministic (no timestamps, `%.6f`, LF newlines): two runs give byte-identical files.

## The scale guarantee

Units: `<unit>` is read from both files. Seamly evaluates formulas in the *pattern* unit, so the engine converts
the `.smis` values to cm on read (`units.py`), to the pattern unit when a formula looks them up, evaluates in the
pattern unit, converts the finished outline to cm once (assembler), and to metres once (mesh build). All
conversions live in `units.py`.

What is tested automatically (`pytest`):

* a single `endLine` of 10 cm gives two vertices exactly 0.100000 m apart;
* Aldrich skirt (size 12): A to A1 = `5*#CM`, hip width A1..A7 = `hip_circ/4 + 1.5*#CM`, A1 to A3, A1 to A9 match
  their formulas to 0.01 mm; Skirt Back 30.457 x 59.971 cm (1574.76 cm2), Skirt Front 28.957 x 59.95 cm (1484.99 cm2);
* male shirt: 8 piece sizes within 0.01 cm, all 17 pieces without gaps, mesh area = outline area to < 0.001 %,
  at most one self-intersection and only on `FullSleeve`;
* `hip_circ` +4 cm changes the skirt hip width by exactly +1 cm;
* `.smis` in mm and in cm give identical meshes; a mm pattern with a cm `.smis` gives the same mesh (to 2 micrometres)
  as the equivalent cm pattern;
* an unsupported tool fails only its piece, the others are written, exit code 1;
* CLI end to end, and determinism;
* trousers: per-piece sanity (see below) and a check that rotated geometry lands where its source says.

What is *not* guaranteed:

* There is no reference output from Seamly2D itself for the trousers: their sizes are regression numbers, not
  verified measurements.
* Curves are flattened with a maximum chord deviation of 0.1 mm (configurable, `flatten_tol_mm`). The polygon is
  therefore inscribed: areas come out up to about 0.03 cm2 smaller than the exact curve area (visible in the skirt
  area test, which allows 0.05 cm2). Bounding boxes are exact because curve extrema are always vertices.
* Arc lengths used inside formulas (`Spl_*`, `cutSpline`, ...) are computed from a 2000-samples-per-segment
  polyline, error < 0.001 mm.
* Formulas follow Seamly's own evaluation, including quirks such as `3.14` as pi in trousers angle formulas.

## Supported tools

| `type` | Kind | Status |
|---|---|---|
| `single`, `endLine`, `alongLine`, `normal`, `intersectXY` | point | supported |
| `height`, `lineIntersect`, `lineIntersectAxis`, `curveIntersectAxis`, `shoulder` | point | supported |
| `cutSpline`, `cutSplinePath` | point | supported (cut clamped to [1 mm, full-1 mm]) |
| `cubicBezier`, `cubicBezierPath`, `simpleInteractive`, `pathInteractive` | curve | supported |
| `rotation` | operation | supported (points and curves, chained operations) |
| `bisector` | point | **unsupported** (`UnsupportedToolError`) |
| `pointOfContact` | point | **unsupported** |
| `trueDarts` | point | **unsupported** |
| `flippingByLine` and other operations (`flippingByAxis`, `moving`) | operation | **unsupported** |
| `cutArc`, arcs, `NodeArc` piece nodes | curve/node | **unsupported** |
| `path` (and `simple`) objects used by the jackets | curve | **unsupported** |
| `anchor` and anything else not listed above | any | **unsupported** |

An unsupported tool or an unknown measurement fails only the piece that needs it (`status: "failed"`, `reason`
names the tool or variable, and the piece for tools).

Tools are strategy objects registered with `@register("type")` in `tools/`. Adding one never touches the assembler or
the engine. Evaluation is lazy (looked up by id on first use, with cycle detection), never in document order.

## Known limits

* **No seam allowance**: only the main (sewing) path is meshed.
* **No interior mesh**: boundary-only ear clipping (`mapbox_earcut`). `Triangulator` is a protocol, so an
  interior-refinement triangulator can be added later without touching anything else.
* **No mirroring**: half pieces (`onFold="true"`) are written as drawn. `mirror_fold_pieces` is only a request:
  if there is no fold information (`foldPosition` missing or `Undefined`, as in all sample files) or the position is
  not implemented, a warning is recorded and nothing is mirrored. `quantity` and `onFold` are `null` plus a warning
  when absent (the shirt has neither).
* **Jackets** (`jacket1..6`, mm, multisize) cannot be built: they need measurements that no `.smis` here provides
  (the first failing variable is named in the report) and tools listed as unsupported above.
* **Trousers sample quirks** (properties of the pattern, reproduced faithfully, not engine bugs):
  * `Front_no_pocket`, `Front_with_pocket` and `Front_Pleats` have one self-intersection: the waist curve at the
    centre front (control point `A28`) overshoots the centre-front line. Mesh area then differs from outline area by up
    to 0.3 % because ear clipping handles a crossing outline approximately.
  * `Front_no_pocket` references `A43` (a pocket opening point) where `Front_with_pocket` uses `A32`.
  * `PocketMouth` joins `A50` to the start of its curve (`Pocket2`, 2.116 cm away) with an implicit straight edge, as
    Seamly does. The engine reports it as a gap note instead of hiding it, and the outline has one self-intersection.
* The Aldrich file has 7 pieces: only the two skirt pieces are supported; the others fail with their reason.
* `aldrich_skirt_size12_h164.smis` has `height = 166`, so `#CM` is 1.0 (the file name says 164).

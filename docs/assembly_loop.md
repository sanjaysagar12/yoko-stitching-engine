# Assembly loop: assembly JSON -> engine -> independent verification -> images

The orchestration layer `src/assembler` (it may import all three engines, like `src/web/backend`) runs one garment
assembly end to end and checks it independently of the engine. The loop (write the JSON, run, read `verify.md`,
look at the images, change the JSON or the engine, repeat) produced the three committed assemblies in
`assert/assemblies/`; the iteration logs and final sheets are in `docs/assemblies/<garment>/`.

```
python -m assembler run assert/assemblies/<garment>.assembly.json --out build/<garment>/iter_NN [--synthetic] [--fast] [--no-render]
python -m assembler check assert/assemblies/<garment>.assembly.json          # schema only
python scripts/run_assembly_loop_real.py [--publish] [garment ...]           # all three on the real avatars
```

Hard rules enforced in code: no scaling anywhere (the verifier's size audit fails a run), `allow_distortion` and
`body_slice` are rejected by `load_assembly`, thresholds (`clearance_body`, `gap_piece`, `gap_layer`,
`max_standoff`, `max_shift`) are the placement spec's and are never relaxed by the assembler.

## Assembly file (`contracts/assembly.schema.json`)

A thin superset of the placement spec: `{schema_version: 1, garment, job, notes, mode, derived_pieces[],
placements[], arrange_options{}, stitches[], rings[], expect{}, views[]}`. `placements` and `derived_pieces` are
placement-spec entries (validated against `placement_spec` with `schema_version` 5). `mode`: `arrange` (default, the
joint solve) or `incremental`. `expect`:

| key | meaning |
|---|---|
| `required_pieces` | every one must be placed |
| `registration` | `{piece, level, piece_point \| edge, offset_m}`: the one line that must sit on its body level (1.5 cm) |
| `grain` | `{piece: "any"}` exempts a piece from the grain check |
| `far_ok` | pieces allowed far from their point |
| `rings` | `{name, pieces, levels, dy_m, centre_xz, concentric}`: concentricity, gaps and coverage on the ring, ease |
| `sections` | levels of the cross-section plots |

## What a run does (`src/assembler/run.py`)

1. Inputs, cached by content (`build/_cache`): pieces of the job (`sm2d_engine`), key = sm2d + smis + piece list + the
   source of `sm2d_engine`; the avatar (`avatar_engine` with `models/smplx`, or the synthetic test body), key = body
   JSON + the source of `avatar_engine` + the model version. The avatar's mesh hash is stored with it.
2. `primary`: `arrange` (or `align`) of the whole spec; engine files in `engine/`, the answer in `result.json`.
3. `replay` and `replay_reversed`: the web app's way, one placement at a time with every earlier `solved` block
   carried, in the listed and in the reversed order (earlier bugs only showed there).
4. `carry`: `align` with every `solved` block of the primary answer: nothing may move by more than 1 mm.
5. `repeat`: the primary run again in a fresh engine: same answer (determinism).
6. Verification of every run (`verify.json`, `verify.md`), `scene.glb`, images.

## Verifier (`src/assembler/verify.py`, independent of the engine's validity)

Computed from the world meshes with its own nearest-triangle search (vertex pre-pass, exact point-triangle distance
for samples within 6 cm). Rows `{id, status: pass|fail|check|info, value, limit, piece, hint}`.

* Size audit (hard fail): avatar mesh hash = the freshly built one, height = the body JSON's (0.5 cm), feet on y = 0;
  per piece the per-triangle Jacobian (world edge = J flat edge) has singular values 1 (0.1 %; slivers of a
  self-crossing outline are left out), area, perimeter and every outline edge (boundary edges located on the outline
  by arc position, so slits are fine) within 0.1 %, the rigid matrix without scale, `changes_piece_size` false.
* Clearance: piece-avatar >= `clearance_body` (signed: inside is negative), piece-piece >= `gap_piece`, no
  self-intersection, min y >= the clearance (floor), standoff <= `max_standoff`, region: within 12 cm (plus its
  distance from the body) of its arrangement point, or of its host piece for a piece laid `over` another one.
* Orientation: the pattern's +y maps along the body-volume axis of its point (the whole limb, root to end, for limb
  points; upright for torso points) within 5 degrees (+ any explicit tilt); the grainline (or the long axis of a tall
  piece) parallel to it; the pattern face points away from the body. An explicit `rotation_deg` / quarter-turn
  `orient` is reported, not judged.
* Registration: each `expect.registration` within 1.5 cm of its level (+ declared `offset_m`); every named level line
  (`*Waist*Line`, `HipLine`, `CrotchLine`, `KneeLine`, `HeelLine`, `BustLine`) is reported against the body level,
  over 8 cm as `check` (pattern and body differ on purpose: fit data, never a failure).
* Pairs: equal radius and axial shift, mirrored slide (1 mm); the boundaries are mirror images about x = 0 (1.5 mm);
  each half on its own side (a centre pair half a gap from the centre line); `expect.rings` concentric
  (radius spread <= `gap_layer` + 5 mm); no `PAIR_*` violation or `NOT_CONCENTRIC` warning from the engine.
* Semantics: required pieces placed; per ring and level the angular gaps between pieces (>= the gap) and the coverage;
  ease (garment width on the ring vs body girth at the level), informational.
* Engine honesty: the engine's `validity.valid` must equal the independent verdict; a mismatch is a failure (and
  each one found during the loop was an engine bug, fixed).

## Renderer (`src/assembler/render.py`)

CPU only (numpy z-buffer + Pillow, no GL, no GPU, no browser), orthographic (rulers in metres), flat Lambert shading,
outlines from the id buffer, labels, legend. Per iteration in `images/`: front, back, left, right, top, four 3/4
views, `points_front` (arrangement points used), `heat_front` / `heat_back` (red below the clearance, amber within
1 cm above it, green), `sections.png` (top-down slices of body and pieces at the expected levels), one close-up per
piece (seen from its outward normal with the nearby body) and `sheet.png` (every view and the sections). About 1-3 s
per view at 1100 px. Pixels are deterministic. `--renderer web` (three.js through Playwright) is not implemented: the
soft renderer is used and the CLI says so.

## Engine and avatar changes made by the loop

| finding (evidence) | change |
|---|---|
| female_165 no longer built: `Shin_Front_L` 3.8 cm from the shin axis rejected by a 0.5-radius floor (skirt baseline) | avatar point sanity floor: offset + 0.2 radii |
| skirt centred on its waist point, top 30 cm above the waist; trousers slid 48 cm down the leg (baselines) | `levels.py` (body levels measured on the avatar) and level registration `anchor: {level, piece_point \| edge, offset_m}`; a registered piece keeps its height (axial slide weight 1e4); schema 5 registers bottoms by default |
| trouser `Back` below the floor reported valid (baseline) | validity: `FLOOR`, `OFF_REGION`, `REGISTRATION`, `ORIENTATION` |
| 1.1 m trouser panel on `Hip_Left`: torso ring grown to R 0.48 m to dodge the T-pose arms (baseline) | registration keeps such panels at the waist; registered limb pieces wrap the torso too (clear radius counts it, the other limb is slid away from) |
| left/right trouser fronts 4 mm apart at the crotch (trousers iter 1) | `pair_slide` maps every sample when the axis leans (leg axes do) |
| false `NOT_AT_POINT` "limit max_shift" at 9 of 25 cm (trousers iter 2) | each solver stage clears the stage-1 slide cap |
| pockets need to sit on their fronts (trousers iter 4) | `over: {piece, point}`; a missing host is a warning; a placed piece over a new host follows it |
| band registration 2.5 cm off on a tilted ring (trousers iter 4) | one defined edge point (extreme height, nearest the middle) for engine and verifier |
| carried pieces turned invalid when others were added (trousers iter 6, 7) | `solved.needed_radius_m` carried; layered pieces explained by their ring's base; carried `over` anchor |
| mirrored sleeve plackets (orient 90 / 270) not a pair (shirt iter 1) | rotations compared modulo 360 |
| `tilt` raised SizeChangedError on `Hip_Front` (orientation tests): the frame was already upside down | avatar_engine: every `Hip_*` point had its `up` pointing to the floor (the Hip volume runs pelvis -> crotch); central points' up is now up (avatar_engine 0.2.1). Pieces on Hip points were upside down in the web app (the report's trouser panel on `Hip_Left`). The verifier's torso reference is world-up, not the point's up |

## Orientation tools (engine, API, web UI)

* `rotation_deg` (in plane), `orient: auto | grain | none | 0 | 90 | 180 | 270` (added; `auto` is the default from
  schema 5: grainline, else the long axis of a tall piece, along the point's up / ring axis, top up; a grain more than
  45 degrees off is a cross-grain cut and kept), `tilt: {pitch_deg, roll_deg}` (the frame and a cylinder's axis turn
  about the reference; clamped to 45 degrees). All isometries. The answer reports `arrange.applied_rotation_deg`,
  `arrange.orient`, `arrange.tilt_deg`, `arrange.registration`.
* Web UI (`ui/orientation.ts`, in the selected-piece panel): angle field, -90 -15 -5 -1 +1 +5 +15 +90, Align to grain,
  Flip face, Upside down, tilt sliders, the applied angles, undo / redo of orientation edits, and "Hang from level"
  (level, piece point or top / bottom edge, offset). The backend sends `schema_version` 5.

## Web import/export

Workflow: Claude Code's loop writes `assert/assemblies/<garment>.assembly.json` and verifies it (`python -m assembler
run`) -> **Import assembly** in the web viewer (`src/web`) builds the job, makes the derived pieces, places and
assembles -> adjust by hand (the pieces keep their real size; each edit is an incremental align) -> **Export assembly**
(`<garment>.assembly.json`, accepted by `python -m assembler check` and `run`) and **Export images** (zip with every
angle, `sheet.png`, `sections.png`, heat maps, close-ups; the layout on screen, not re-solved).

* `src/assembler/webio.py`: `parse` / `check_build` / `normalise` (import: every problem listed with a code
  `invalid_assembly`, `unknown_job`, `unknown_piece`, `unknown_point`, `unsupported_option`; `allow_distortion` and
  `body_slice` are refused as in `load_assembly`), `export_doc` (validated by `run.check_doc`), `write_zip`
  (deterministic).
* An exported file with `solved` blocks reopens with `align` (nothing moves, identical); without them, with the
  file's `mode` (`arrange` re-solves: the same garment within 1 mm in the tests).
* `python -m assembler render <assembly.json> --out DIR [--synthetic]` solves the assembly and writes only the images
  (all nine views, points, heat maps, sections, sheet, close-ups) with the same `run.images` function the loop and the
  web export use.
* Not in the file: avatar data, meshes, absolute paths. `expect`, `views` and `notes` of an imported file are kept
  and shown; an export of a hand-made layout gets `expect.required_pieces` = every placed piece.

## Tests

`tests/test_assembler.py` (schema good/bad, caching, the three assemblies end to end on the synthetic avatar,
golden numbers in `tests/golden/`, determinism and carry, the verifier catches an injected scale, a piece below the
floor, a rotated piece, a swapped and a stacked pair, a skirt centred on its point), `tests/test_render.py`,
`tests/test_orientation.py`.

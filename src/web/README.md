# web

A thin FastAPI backend (orchestration and HTTP only) and a Vite + TypeScript + three.js frontend. The backend imports
only the public facade of each engine; the frontend never computes placement, it asks `/api/align`.

## Run

```powershell
./scripts/start.ps1                      # from the repo root: install, build the frontend, serve on port 8000
python -m web.backend --port 8000        # backend only (serves src/web/frontend/dist if it exists)
cd src/web/frontend; npm run dev         # frontend dev server on :5173, proxies /api to :8000
```

Folders and overrides (`web/backend/config.py`): inputs `assert/sm2d`, `assert/smis`, `assert/body`, jobs
`assert/jobs`, models `models/smplx`, outputs `outputs/`. Environment variables `SM2D_WEB_SM2D_DIR`, `_SMIS_DIR`,
`_BODY_DIR`, `_JOBS_DIR`, `_MODELS_DIR`, `_OUTPUTS_DIR`, `_FRONTEND_DIST`.

## Jobs: `assert/jobs/<name>.json`

```json
{"name": "male_shirt", "sm2d": "male_shirt.sm2d", "smis": "male_shirt.smis", "body": "male_173.json", "pieces": null}
```

File names resolve against `assert/sm2d`, `assert/smis`, `assert/body` (absolute paths work too); unknown keys are an
error. Provided: `male_shirt`, `aldrich_skirt` (the two skirt pieces, female body), `trousers`.

## API

| Method and path | Purpose |
|---|---|
| `GET /api/jobs` | jobs from `assert/jobs/*.json` |
| `POST /api/jobs/{name}/build` | start (or reuse) the build: `{build_id, status, cached}` |
| `GET /api/builds/{id}` | status (`queued` / `running` / `done` / `failed`), stage, progress; the result when done |
| `GET /api/builds/{id}/files/{path}` | avatar and piece OBJ / JSON of that build |
| `POST /api/align` | `{build, placements, derived_pieces?, stitches?, rings?, arrange_options?, compact?, known_meshes?}` to the `alignment_result` document: every piece placed size-exact and collision-free, `validity` and per piece `size_check` and `solved` (placements sent with `solved` stay where they are; derived pieces are cached under `derived/`, 5 mm meshes under `refined/`). `compact: true` sends rolled meshes as base64 (`wrapped_mesh`), faces only for ids not in `known_meshes` |
| `POST /api/arrange` | same body: re-solve every piece that is not locked (Auto-arrange); same answer |
| `POST /api/clone` | `{build, source, axis, id}`: a linked mirrored copy of a piece (size, named points, files, suggested seam); `{build, source, id, op: "copy", derived_pieces?}`: an exact copy (no axis; the UI's **Clone**, for pieces cut twice / identical pieces; never a pair) |
| `POST /api/pair` | `{build, base, derived_id, mode, derived_pieces}`: the two placements of a pair (`same_anchor` or `mirror_anchor`) |
| `GET` / `PUT /api/jobs/{name}/placements` | load / save `outputs/<job>/placements.json` (placements and `derived_pieces`) |
| `POST /api/assembly/import?job=&build=&skip_missing=` | body = the assembly JSON text (max 2 MB, UTF-8, BOM accepted). Without `build` only the file and its `job` are checked (schema, placement spec, no `allow_distortion`, no `body_slice`, `job` a known job name). With the id of that job's build every piece (original or declared derived) and every `point` is checked too. Answer: `{garment, job, notes, mode, run, derived_pieces, placements, arrange_options, stitches, rings, expect, views, warnings[]}`; `run` is `arrange`, `incremental` or `align` (every placement carries `solved`: nothing moves). The older `{placements, derived_pieces}` shape is upgraded (name its job with `?job=`). `skip_missing` leaves out placements of pieces the job does not have (listed in `warnings`) |
| `POST /api/assembly/export` | `{job, placements, derived_pieces?, arrange_options?, stitches?, rings?, garment?, notes?, mode?, include_solved?, expect?, views?}` to the assembly file text (schema_version 1, 2-space indent, LF, stable key order), validated like `python -m assembler check` before it is returned. `expect` and `views` of an imported file are kept; otherwise `expect.required_pieces` = every placed piece |
| `POST /api/export/images` | `{build, placements (with solved), derived_pieces, arrange_options, stitches, rings, garment, notes?, mode?, expect?, views?, options: {closeups, heat, verify, assembly}, viewport?: {name: png data URL}}`: starts a background render of the layout on screen (aligned incrementally with every `solved` block: nothing moves, nothing is solved from scratch) and returns `{id, status, stage, progress, files, failures}`; the same layout and options give the same id (cached) |
| `GET /api/export/images/{id}` / `DELETE` | status (`queued` / `running` / `done` / `failed` / `cancelled`) / cancel |
| `GET /api/export/images/{id}/download` | `<garment>_images.zip` (deterministic bytes): `front back left right top front_left front_right back_left back_right points_front heat_front heat_back sections sheet` as PNG, `closeups/<piece>.png`, `assembly.json`, `verify.md` / `verify.json` (option), `viewport/*.png` (option), `export_report.txt` (a view that failed to render is listed there; the others are kept) |
| `GET /api/export/{id}?format=obj\|glb` | avatar plus the job's saved placements (avatar only if none are saved) |

Builds run in a background thread pool and are cached in `outputs/<job>/<hash>/`, keyed by a hash of the three input
files, the piece selection and the engine versions. The result lists the pieces (size in cm, area, named points), the
avatar bundle, arrangement points, bounding volumes and the measurement report.

Errors are always `{"error": {"code", "message", "piece"?, "tool"?}}`, never a stack trace (codes: `job_not_found`,
`job_file_missing`, `unsupported_tool`, `missing_measurement`, `model_files_not_found`, `fit_did_not_converge`,
`unknown_piece`, `unknown_point`, `unknown_named_point`, `reference_not_on_axis`, `no_mirrored_point`,
`derived_of_derived`, `duplicate_piece`, `ring_not_found`, `size_changed`, `invalid_document`, `invalid_request`,
`not_found`, `internal_error`, and for assembly files `invalid_assembly`, `unknown_job`, `unknown_piece`, `unknown_point`, `unsupported_option`: those errors also carry `problems: [{code, message}]`, every problem of the file, not only the first; `no_placements`, `export_not_found`, `export_not_ready`, ...). **A failed
piece does not fail the build**: it is listed with `status: "failed"`, its `reason`, and the `tool` or variable named.

## Frontend

Modules in `src/web/frontend/src`: `api.ts`, `state.ts`, `scene.ts`, `avatarView.ts`, `pointsView.ts`,
`piecesView.ts`, `lib.ts` (display helpers), `ui/*` (top bar, piece list, controls, measurements, errors).
TypeScript is `strict`; ESLint and Vitest are configured (`npm run lint`, `npm test`).

* Top bar: job drop-down (a build starts on selection, with progress and an error panel), label / volume toggles,
  avatar opacity, views, save / load, export.
* 3D view: semi-transparent avatar, 1 cm floor grid (10 cm lines), a vertical ruler, arrangement points coloured by body
  part (names on hover, or all with the Labels toggle), bounding volumes as wireframe capsules, pieces laid out left to
  right to the right of the avatar.
* Placement: click a piece (list or 3D) then a point, or the point first then the piece. The frontend sends
  `/api/align` with every placement (earlier pieces carry their `solved` place, so they do not move) and shows the
  answer. A **status strip** under the view gives the engine's verdict after every placement: green
  `Size unchanged (max x %) · Body clearance x cm · Smallest gap between pieces x cm` only when `valid`; red with the
  violating pairs otherwise (click one to frame it).
* Right panel (per piece): the **piece inspector** (outline with its named points as clickable dots; the dashed line is
  a mirror axis), **Clone & mirror**, offset in cm, rotation, mirror, flip facing, wrap (`flat`, `roll`, `around`; new torso, neck,
  arm and leg placements default to `roll`, head, hand and foot placements to `around`; all keep sizes exact), reference point, remove placement, pair buttons, the side
  indicator (`extends -> wearer's left / right`), non-blocking notices for the engine warnings, and the **scale check**
  (flat size, the size the backend reports after placement, the engine's measured size check of the placed piece, the
  3D extent, the pair total width against the avatar's half loop); the standoff is shown as display space, not ease.
* **Half pieces**: Clone & mirror opens a dialog (axis: flip in place, across an edge through two named points, or across
  a vertical line through a point; name; place the pair now; pair mode). The clone is a separate linked piece nested
  under its source (`mirror of ...`, delete with the cross); both are placed independently or as a pair
  (`Place pair (same anchor)` for a centre-line pair, `Place pair (mirrored anchor)` for a side pair; hovering the
  second highlights both points). Derived pieces are saved with the placements and included in the export; the seam
  along the mirror edge is drawn as a stitch.
* **Picking**: points behind the avatar from the current camera are dimmed and not pickable (a depth map of the avatar
  is compared with each point; **X-ray** makes all points pickable again). Front / Back / Left / Right / Top buttons set
  the camera, a text field filters point names, and hovering a point highlights its mirror counterpart.

The axis convention (+x = wearer's left, +y up, +z front, metres) is printed in the left panel.

## Tests

`tests/test_backend.py` (API, including clone and pair), `tests/e2e/test_web_e2e.py` (headless Chromium through
Playwright: drop-down lists the three jobs; select `FrontPanel`, click `Chest_Front`, check the reference and that the
bounding box in cm is unchanged; set the CF point as reference in the inspector, Clone & mirror across the CF edge with
`Place pair now`, click `Chest_Front` and check that both copies share the anchor and the CF edges coincide; a sleeve
side pair; a back point is not pickable in the front view until X-ray), and `frontend/tests` (Vitest).
Needs `npm run build` once and `python -m playwright install chromium`.

## Import and export of assemblies, image export

Topbar buttons (all with `data-testid`): **Import assembly** (`import-assembly`, file chooser `import-assembly-input`; a
`.json` can also be dropped on the viewer), **Export assembly** (`export-assembly`), **Export images** (`export-images`);
**Cancel** (`cancel-busy`) appears while an import or image export runs. Export buttons are disabled until a job is built.

* Import: the file is read on the server (`/api/assembly/import`), its job is built if it is not the one on show (without
  touching the current view), then derived pieces (mirrors and copies, no `/api/clone` calls: the first align/arrange
  makes them), placements (every field), arrange options, stitches and rings are applied and assembled (`arrange` joint
  solve, `incremental` one placement at a time). The panel over the viewer shows `Assembly: <garment> (<job>, N pieces,
  M derived) - valid / N problems`, the import warnings, the file's notes and `expect` (read-only), **Re-assemble from
  file** and **Revert** (the layout before the import). Any error keeps the previous layout; a missing piece offers
  "Import without the missing pieces". A second import while one runs is refused with a message.
* Export assembly: garment name, notes, "include solved poses" (on: the file reopens exactly as it is), mode.
* Export images: close-ups, heat maps, verification report (slow: minutes on a real body), the assembly.json, and
  "viewport screenshots" (the canvas from the five preset sides, under `viewport/`). The server renders with the
  assembler's renderer (`python -m assembler render <assembly.json> --out DIR` uses the same function).
* The conversion lives in `src/assembler/webio.py`; the frontend only calls the API and applies the answer.

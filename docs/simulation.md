# Simulation: a stitched garment sewn and draped on the avatar

The simulator (`src/simulator`) takes the stitcher's `stitching.result.json` and its assembly run and lets a cloth
solver pull every seam shut and drape the garment on the avatar under gravity. It changes nothing upstream: the pieces,
their placed meshes and the avatar are read from files; the draped garment is written next to them, vertex for vertex
in the placed meshes' order.

```
python -m simulator run --stitching assert/stitchings/<g>.stitching.json [--synthetic] [--result R] [--run DIR]
                        [--fabric cotton_shirting] [--quality draft|standard|fine] [--frames 240] [--sew-time 0.25]
                        [--out DIR] [--no-render]
python -m simulator finish --stitching ... --out DIR      # checks, outputs and images again from the solver frames
python -m simulator fabrics                                # the presets in measured and in solver units
python -m simulator solver                                 # where the solver is
```

`run` stitches first when there is no stitching result (`python -m stitcher run`), exits 1 when a check fails and 2 on
an invalid request. Default output: `build/simulation/<garment>-syn|real-<key>/`.

## The cloth solver

[ZOZO's Contact Solver](https://github.com/st-tech/ppf-contact-solver) (`ppf-contact-solver`, Apache 2.0): built for
garments; contact is penetration free for every step it takes, the fabric cannot stretch past a strain limit, and seams
are built in (a stitch joins a point of one triangle to a point of another). It ships its own Python with its `frontend`
package; `simulator/ppf_runner.py` runs in that Python in a subprocess and imports only numpy and the solver.

Install (Windows, NVIDIA GPU with a current driver; no CUDA toolkit, Python or Docker needed):

1. From <https://github.com/st-tech/ppf-contact-solver/releases> download `ppf-contact-solver-<date>-win64.zip` (the
   release named "ZOZO's Contact Solver <date>", not the Blender add-on). Tested: `2026-10-13-12-32`.
2. Unblock the zip (Properties -> Unblock) and unzip it next to this repository (`../ppf-contact-solver-<date>-win64`)
   or into `tools/`, or anywhere and set `PPF_DIR` to the folder.
3. `python -m simulator solver` must say `"available": true`. The solver's own `headless.bat` is its smoke test.

The solver keeps its sessions inside its own folder (`local/share/ppf-cts/`); the runner deletes a session once its
frames are copied out. One simulation runs at a time per GPU.

## How the garment is built for the solver (`scene.py`, `geometry.py`)

* **Pieces.** Every piece a seam names. The alignment result stores a placed piece as a 5 mm world mesh only; its flat
  (pattern) coordinates are recovered from the outline and a harmonic extension (`flat_coords`): the boundary from
  `world_outline_m` / the bundle's `outline_m` (shortest path along the boundary edges between two outline vertices,
  which also walks the shirt sleeve's self-touching placket slit), the interior from the cotangent Laplace equation,
  refined as-rigid-as-possible when the mesh has slivers (the alignment engine's fallback mesh). Measured against the
  alignment engine's own flat meshes: 0.001 mm.
* **Simulation mesh.** Each piece is meshed again in the flat for the solver (`sim_mesh`): a conforming Delaunay
  triangulation with edges of about `h` (draft 25 mm, standard 18 mm, fine 12 mm; narrow pieces finer), corners kept,
  smallest angle at least 12 degrees. Slits drafted narrower than 4 mm (the sleeve slit is 0.1 mm wide; the back
  fish-eye dart near its tips) are opened to 4 mm, because the solver refuses cloth that starts closer to itself than
  its contact gap; a needle tip under 15 degrees and 4 mm (the collar stand's button end) is cut, because it can only
  be meshed with a sliver the strain limiter cannot move. Every simulation vertex is put at the world point of its
  flat position on the placed mesh, then every triangle is made exactly its pattern shape (`isometric_fit`): the solver
  takes each triangle's rest shape from its start. Start shape vs pattern on the shirt: within 0.03 mm.
* **Display.** Every vertex of the placed mesh is tied to the simulation mesh (outline vertices by their outline arc,
  the others by their flat position): `draped.npz` and the viewer show the placed meshes moved by the simulation.
* **Stitches.** Every correspondence of the stitching result becomes one six-column stitch row: a point of a triangle
  of side a to a point of a triangle of side b, with barycentric weights. Outline points are found by their outline
  arc (`chain[0]`, `direction`, `trim_m`, `a_arc_m`), never by position (two sides of a slit lie 0.1 mm apart); dart
  legs (`line`) by their flat point; attach seams by the host triangle the stitcher found. Gathers and ease are the
  stitcher's: the longer side is spread evenly over the shorter, and the cloth buckles into the gathers by itself.
* **Interfacing.** A piece is interfaced when the pattern has a `<piece>Interface` piece (the shirt: collar, stand and
  cuffs); it is not simulated as a second layer, its bending stiffness is multiplied by `interfaced_bend_factor`.
* **Body.** The run's avatar OBJ, pinned (a fixed collider), friction 0.8 (the solver uses the smaller coefficient of
  the two, so the fabric's decides).

## Fabrics (`fabrics.py`)

Presets in measured units: areal density (g/m2), bending rigidity B (KES, N m), initial tensile modulus E (N/m), Poisson
ratio, strain limit, friction. The solver normalises its stiffnesses by density: `young-mod = E / rho`, and its shell
bending is `1.28e-5 * bend * rho` (the scale its authors calibrated on Cusick drape tests), so `bend = B / (1.28e-5 *
rho)`. The values are typical published figures for each class of cloth, not a lab measurement of a particular fabric.

| preset | g/m2 | B (gf cm2/cm) | E (N/m) | strain limit |
|---|---|---|---|---|
| cotton_shirting (default) | 115 | 0.06 | 400 | 4 % |
| cotton_oxford | 140 | 0.12 | 500 | 4 % |
| linen | 170 | 0.25 | 700 | 3 % |
| silk | 70 | 0.015 | 250 | 5 % |
| denim | 400 | 1.2 | 1500 | 3 % |
| jersey | 180 | 0.02 | 60 | 25 % |

## The run (`ppf_runner.py`)

The garment is one Baraff-Witkin shell with the flat pattern as its UV (the warp / weft frame). First a sewing phase
(`sew_time`, default 0.25 s) with inactive momentum and 1 ms steps: the cloth does not keep its velocity, so the
stitches pull the panels together calmly; then normal dynamics with 10 ms steps under gravity until `frames` (at 60 fps;
default 240 = 4 s). Every frame's largest and mean stitch gap is recorded; every `keep_every` (4) frame is kept for the
animation.

## Checks (`result.py`) and outputs

| check | fails when |
|---|---|
| `solver.finished` | the solver stopped before the last frame |
| `result.finite` | a position is NaN or infinite |
| `seam.closed` (per seam) | a stitch is still more than 1 cm open (warn above 5 mm) |
| `body.penetration` | a vertex is more than 2 mm inside the body (generalised winding number, so a body made of overlapping parts is judged right) |
| `fabric.stretch` | an edge is longer than the pattern by more than the strain limit + 0.5 % |
| `garment.above_floor` | the garment fell below the floor |
| `fabric.compression` | info only: gathers and folds compress the cloth |

`simulation.result.json` (contracts/simulation_result.schema.json): inputs with their hashes, solver, options, fabric
(measured and solver units), scene (per piece mesh size, interfacing, opened slit points, start error), run (frames,
wall time), closure per frame, checks, summary. Also `draped.npz` (per piece the draped placed mesh), `draped_m.obj`
(the garment and the avatar), `solver/frames.npz` (the kept frames of the simulation mesh) and `images/`
(`sim_<view>.png`, `sim_stretch_front|back.png`, `sim_placed_front.png`, `sim_sheet.png`).

## The shirt (synthetic body, cotton shirting, standard mesh, Tesla T4)

5930 simulation vertices, 10537 triangles, 1910 stitch rows. The largest gap closes from 48.9 cm (the shoulders) to
0.85 cm by frame 6 (0.1 s); after 240 frames: every seam closed (largest gap 0.55 cm, mean 0.12 cm: the contact gap of
two cloth layers), stretch at most 3.3 % against the 4 % limit, no vertex inside the body. Wall time 13 min: about 40 s
per frame while sewing (1 ms steps), under 1 s per frame afterwards. A draft mesh and fewer frames are much quicker.

## Web app

The simulation panel (bottom right of the viewer) follows the stitching panel: after **Stitch**, **Simulate** (fabric,
mesh quality, frames) runs on the server; the newest frame is shown live on the placed meshes while it runs. When it
is done: the verdict, **Play** and a frame slider, **Show placed / Show draped**, piece colours, and the downloads (OBJ,
zip with the result and the images). Backend: `GET /api/simulation/info`, `POST /api/simulation/start` (the stitching
body plus `options`), `GET|DELETE /api/simulation/{id}`, `GET /api/simulation/{id}/embedding|latest|frames|result`,
`GET /api/simulation/{id}/download?format=obj|zip`, `GET /api/simulation/{id}/images/{name}`
(`src/web/backend/simulation.py`). Simulations are cached in the build by the stitching result and the options.

## What is not simulated

* The avatar is static (the model's T-pose); no posing or animation.
* Seam allowance, fabric thickness and the stiffness of a sewn seam; interfacing only as stiffer bending.
* Buttons: the centre front is sewn shut (as in the stitching file). Pleats and the sleeve slit's folded plackets are
  sewn as the stitching file says (a placket along one edge), so a placket hangs free where a real one is topstitched.
* The fabric presets are typical values for each class of cloth, not measurements of a particular fabric.
* SMPL-X is the realistic body; without its model files the app and the tests use the synthetic body (primitives).

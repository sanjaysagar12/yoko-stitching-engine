# Pattern fit viewer

Turns a Seamly2D pattern plus body measurements into **real-scale** pieces and an **SMPL-X avatar**, lets you place a
piece on the avatar at CLO3D-style arrangement points, and shows the result in a browser. Everything is in
**metres**, y up, feet on y = 0. Nothing is ever scaled to fit: a 60 cm piece is 0.60 in every file and in the browser.

| Component | What it does | README |
|---|---|---|
| `sm2d_engine` | `.sm2d` + `.smis` to one mesh per piece (`*.obj`, `*.piece.json`, `report.json`) | [src/sm2d_engine/README.md](src/sm2d_engine/README.md) |
| `avatar_engine` | body measurements JSON to an avatar, bounding volumes and arrangement points | [src/avatar_engine/README.md](src/avatar_engine/README.md) |
| `alignment_engine` | places pieces on arrangement points (offset, rotation, mirror, wrap), stitch and ring reports | [src/alignment_engine/README.md](src/alignment_engine/README.md) |
| `web` | FastAPI backend and Vite + three.js frontend | [src/web/README.md](src/web/README.md) |

The three engines import nothing from each other or from `web`; they exchange files that follow the JSON Schemas in
[contracts/](contracts/). `tests/test_architecture.py` enforces this. Design notes: [docs/](docs/).

## Coordinate convention (also shown in the UI)

**+x = the wearer's left, +y = up, +z = front of the body**, metres, feet on y = 0. The avatar is in the model's zero
pose (a T-pose). Viewed from the front (camera on +z) the wearer's left is on the viewer's right.

## Quick start (Windows, PowerShell)

Requirements: Python 3.10+, Node 20+, and SMPL-X model files in `models/smplx/` (`SMPLX_MALE.npz`, `SMPLX_FEMALE.npz`,
`SMPLX_NEUTRAL.npz`; **never downloaded or committed**, see the licence note below).

```powershell
./scripts/start.ps1            # installs, builds the frontend, starts http://127.0.0.1:8000
./scripts/start.ps1 -Port 9000 # another port
```

That runs, in order: `pip install -e ".[web,dev]"`, `npm install` and `npm run build` in `src/web/frontend`, then
`python -m web.backend`. Open the printed address, pick a job in the drop-down (`male_shirt`, `aldrich_skirt`,
`trousers`), wait for the avatar, click a piece in the list and then a coloured point on the avatar.

Without the web app, each engine has its own CLI:

```powershell
sm2d-engine run examples/job.json --out outputs/pieces
avatar-engine build male_173 --out outputs/avatar/male_173
alignment-engine run --avatar outputs/avatar/male_173 --pieces outputs/pieces --spec examples/placement.json --out outputs/aligned
```

## Tests

```powershell
python -m pytest                       # engines, contracts, backend API, architecture rules, browser e2e
python -m mypy                         # strict, src/
ruff check src tests scripts
cd src/web/frontend; npm test; npm run lint; npm run build
python -m playwright install chromium # once, for tests/e2e
```

The browser test is skipped when the frontend has not been built or Chromium is not installed; tests that need the
avatar are skipped when `models/smplx` is absent.

## The scale guarantee

What is tested: pattern pieces match their formulas and the reference sizes (shirt, skirt) to 0.01 cm; an avatar of
173 cm has a bounding box 1.730 m high (+-1 mm) with its feet on y = 0; circumferences hit their targets to 0.2 mm after
ring correction; rigid placement preserves all pairwise vertex distances (1e-9 m) and the piece area; wrapping
preserves arc length along the piece x direction (0.1 %) and distances along y; the size the browser reports for a
piece is the backend's flat bounding box and is unchanged by placement.

What is not tested or guaranteed: that the avatar is *the person* (it is a statistical body fitted to 12
measurements; other dimensions follow the model); seam allowance (ignored); fabric behaviour (no simulation);
collisions between a placed piece and the body (only the nearest-vertex distance is reported).

## Licence note: SMPL and SMPL-X

SMPL and SMPL-X are licensed **for non-commercial research use only**. Replace them, or obtain a commercial licence,
before any commercial use. `avatar_engine` hides the body model behind the `BodyModel` protocol
(`avatar_engine/bodymodel.py`), so another body model can be swapped in without touching measurements, fitting,
volumes or points. `models/` and `outputs/` are in `.gitignore`.

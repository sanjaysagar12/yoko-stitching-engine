"""Simulate a stitched garment: scene -> solver -> draped pieces, checks, images, ``simulation.result.json``.

``simulate`` is what the CLI (``python -m simulator run``) and the web backend call. The folder it writes:

    scene.npz, scene.json      the solver's input and what it means (scene.py)
    solver/                    params.json, the runner's log, frames.npz, latest.npy while running
    draped.npz                 per piece: every vertex of its placed mesh, draped (metres, world frame)
    draped_m.obj               the draped garment (placed meshes) + the avatar
    simulation.result.json     inputs, fabric, run, checks, summary (contracts/simulation_result.schema.json)
    images/                    sim_<view>.png, sim_stretch_<front|back>.png, sim_placed_front.png, sim_sheet.png
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from stitcher.errors import StitchingError
from stitcher.load import Run, rel, sha256, validate

from .errors import SimulationError
from .fabrics import DEFAULT_QUALITY, Fabric, fabric, mesh_size
from .result import SCHEMA_VERSION, SceneArrays, check, summary, verdict, write_draped, write_obj
from .scene import build, save
from .solver import SolverRun, solver_info

Event = Callable[[dict[str, Any]], None]
BODY_FRICTION = 0.8  # skin; the solver takes the smaller of the two coefficients, so the fabric's one decides


@dataclass(frozen=True)
class Options:
    """What a simulation can be asked for (the defaults are the tested ones)."""

    fabric: str = "cotton_shirting"
    quality: str = DEFAULT_QUALITY
    frames: int = 240  # at 60 fps: 4 s
    sew_time: float = 0.25  # s of calm sewing (inactive momentum, 1 ms steps; the shirt closes in 0.1 s), then dynamics
    dt_after: float = 0.01
    keep_every: int = 4  # frames kept for the animation

    @classmethod
    def of(cls, d: dict[str, Any] | None) -> Options:
        d = dict(d or {})
        unknown = sorted(set(d) - set(cls.__dataclass_fields__))
        if unknown:
            raise SimulationError(f"unknown simulation options {unknown}; options: {sorted(cls.__dataclass_fields__)}")
        o = cls(**d)
        fabric(o.fabric)
        mesh_size(o.quality)
        if not 10 <= o.frames <= 1200:
            raise SimulationError(f"frames must be 10..1200, got {o.frames}")
        if not 0.05 <= o.sew_time <= 5.0:
            raise SimulationError(f"sew_time must be 0.05..5 s, got {o.sew_time}")
        if not 1e-3 <= o.dt_after <= 0.02:
            raise SimulationError(f"dt_after must be 0.001..0.02 s, got {o.dt_after}")
        if not 1 <= o.keep_every <= 60:
            raise SimulationError(f"keep_every must be 1..60, got {o.keep_every}")
        return o

    def key(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


def solver_params(o: Options, fab: Fabric, app: str, cache_dir: Path) -> dict[str, Any]:
    g = fab.solver_params()
    return {
        "app": app,
        "cache_dir": str(cache_dir),
        "frames": o.frames,
        "sew_time": o.sew_time,
        "dt_after": o.dt_after,
        "keep_every": o.keep_every,
        "garment": g,
        "stiff_bend": g["bend"] * fab.interfaced_bend_factor,
        "body": {"friction": BODY_FRICTION},
    }


def folder_key(result_path: Path, o: Options) -> str:
    """The simulation folder's name: the stitching result's content and the options."""
    h = hashlib.sha256(Path(result_path).read_bytes())
    h.update(o.key().encode())
    h.update((Path(__file__).parent / "ppf_runner.py").read_bytes())
    return h.hexdigest()[:16]


def simulate(
    result_path: Path,
    run: Run,
    out: Path,
    o: Options,
    on_event: Event | None = None,
    started: Callable[[SolverRun], None] | None = None,
    render: bool = True,
) -> dict[str, Any]:
    """Simulate the stitching result at ``result_path`` (of ``run``) into ``out``; the result document."""
    emit = on_event or (lambda e: None)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    result = json.loads(Path(result_path).read_text(encoding="utf-8"))
    fab = fabric(o.fabric)
    emit({"event": "stage", "stage": "building the simulation mesh"})
    sc = build(result, run, mesh_size(o.quality), fab, cache=run.folder.parent / "_sim_flat")
    save(sc, out)
    emit({"event": "stage", "stage": "starting the solver", "vertices": int(len(sc.vertices))})
    app = "yoko-" + hashlib.sha1(str(out.resolve()).encode()).hexdigest()[:10]
    solver = out / "solver"
    sr = SolverRun((out / "scene.npz").resolve(), solver_params(o, fab, app, (solver / "ppf").resolve()),
                   solver.resolve(), emit)  # fmt: skip
    if started:
        started(sr)
    done = sr.wait()
    return finish(out, result_path, run, o, done, emit, render)


def finish(
    out: Path, result_path: Path, run: Run, o: Options, done: dict[str, Any], emit: Event, render: bool = True
) -> dict[str, Any]:
    """Map the solver's frames back, check them, write the outputs (also for a stopped run: its last frame)."""
    out = Path(out)
    fab = fabric(o.fabric)
    sa = SceneArrays(out)
    frames_path = out / "solver" / "frames.npz"
    if not frames_path.is_file():
        raise SimulationError(f"the solver wrote no frames ({frames_path}); log: {(done.get('log') or [])[-5:]}")
    fz = np.load(frames_path)
    x = fz["X"][-1].astype(np.float64)
    emit({"event": "stage", "stage": "checking the result"})
    rows = check(sa, x, done, fab.strain_limit)
    write_draped(sa, x, out)
    disp = sa.display(x)
    faces = {p: np.asarray(run.placements[p]["wrapped_faces"], dtype=np.int64) for p in disp}
    bv, bf = sa.body
    write_obj(out / "draped_m.obj", [*((p, disp[p], faces[p]) for p in disp), ("avatar", bv, bf)])
    garment = str((_read(result_path) or {}).get("garment", "garment"))
    images: dict[str, str] = {}
    if render:
        emit({"event": "stage", "stage": "rendering images"})
        from .render import write_images

        strain_v = _vertex_strain(sa, x)
        strain = {}
        for p in disp:
            idx, w = sa.embedding(p)
            strain[p] = np.einsum("nk,nk->n", w, strain_v[idx])
        placed = {p: (np.asarray(run.placements[p]["wrapped_vertices"], dtype=np.float64), faces[p]) for p in disp}
        title = f"{garment} ({'synthetic body' if _synthetic(run) else 'SMPL-X'}, {fab.label})"
        paths = write_images(out, (bv, bf), {p: (disp[p], faces[p]) for p in disp}, placed, strain,
                             fab.strain_limit, title)  # fmt: skip
        images = {k: v.relative_to(out).as_posix() for k, v in paths.items()}
    doc = {
        "schema_version": SCHEMA_VERSION,
        "garment": garment,
        "units": "m",
        "convention": "world: metres, y up, +z front, +x the wearer's left (the alignment result's frame); draped "
        "pieces vertex for vertex in their placed mesh's order (wrapped_vertices)",
        "inputs": {
            "stitching_result": rel(result_path),
            "stitching_result_sha256": sha256(Path(result_path).read_bytes()),
            "run": rel(run.folder),
            "alignment_result_sha256": sha256((run.folder / "result.json").read_bytes()),
            "avatar": rel(run.avatar_dir),
            "synthetic": _synthetic(run),
        },
        "solver": {**solver_info(), "runner": "simulator/ppf_runner.py"},
        "options": asdict(o),
        "fabric": fab.doc(),
        "scene": {k: v for k, v in sa.doc.items() if k != "seams"},
        "run": {k: done.get(k) for k in ("finished", "cancelled", "frames", "wall_s", "ms_per_frame")},
        "closure": done.get("closure") or [],
        "frames_kept": fz["frame"].tolist(),
        "checks": rows,
        "pass": verdict(rows),
        "summary": summary(sa, x, rows, done),
        "files": {"draped": "draped.npz", "obj": "draped_m.obj", "frames": "solver/frames.npz", "images": images},
        "log": done.get("log") or [],
    }
    try:
        validate("simulation_result", doc)
    except StitchingError as e:
        raise SimulationError(f"the simulation result does not follow its contract: {e}") from None
    (out / "simulation.result.json").write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8", newline="\n")
    emit({"event": "finished", "pass": doc["pass"], "summary": doc["summary"]})
    return doc


def _synthetic(run: Run) -> bool:
    """The test body (tests/synthetic_avatar.py), not SMPL-X: from the run, or from the avatar bundle (web builds)."""
    if (run.verify or {}).get("inputs", {}).get("synthetic", False):
        return True
    bundle = _read(run.avatar_dir / "avatar_bundle.json") or {}
    return str(bundle.get("model_file", "")).startswith("synthetic")


def _read(p: Path) -> dict[str, Any] | None:
    try:
        out: dict[str, Any] = json.loads(Path(p).read_text(encoding="utf-8"))
        return out
    except (OSError, json.JSONDecodeError):
        return None


def _vertex_strain(sa: SceneArrays, x: Any) -> Any:
    """Per simulation vertex: the largest stretch of its edges against the pattern."""
    from .geometry import edge_lengths

    e, now = edge_lengths(x, sa.F)
    rest = np.linalg.norm(sa.UV[e[:, 0]] - sa.UV[e[:, 1]], axis=1)
    s = now / np.maximum(rest, 1e-12) - 1.0
    out = np.full(len(x), -1.0)
    np.maximum.at(out, e[:, 0], s)
    np.maximum.at(out, e[:, 1], s)
    return np.clip(out, 0.0, None)

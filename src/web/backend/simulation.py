"""Simulation of the layout on screen: stitch it (``stitching.run``, cached), then sew and drape it with the cloth
solver (``simulator``, docs/simulation.md) in the background.

One simulation runs at a time (one GPU; the solver itself refuses a second run). Jobs are cached in the build by the
stitching result's content and the options (``<build>/simulation/<key>``), and found again after a restart from
their folder. While a job runs the viewer polls its status and the newest frame of the simulation mesh; it gets the
embedding of the placed meshes once and maps every frame onto them itself.
"""

from __future__ import annotations

import base64
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from assembler import webio
from simulator import FABRICS, QUALITIES, Options, SimulationError, folder_key, simulate, solver_info
from simulator.result import SceneArrays
from simulator.solver import SolverRun

from . import stitching
from .builds import Build
from .jobs import JobError


def _b64(a: Any, dtype: str) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype=dtype).tobytes()).decode("ascii")


@dataclass
class SimJob:
    """State of one simulation."""

    id: str
    build: str
    garment: str
    folder: Path
    options: dict[str, Any]
    status: str = "queued"  # queued | running | done | failed | cancelled
    stage: str = "queued"
    frame: int = -1
    frames: int = 0
    gap_max_m: float | None = None
    wall_s: float | None = None
    error: dict[str, Any] | None = None
    summary: dict[str, Any] | None = None
    cached: bool = False
    runner: SolverRun | None = field(default=None, repr=False)
    cancelled: threading.Event = field(default_factory=threading.Event, repr=False)

    def public(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": self.id,
            "build": self.build,
            "garment": self.garment,
            "status": self.status,
            "stage": self.stage,
            "frame": self.frame,
            "frames": self.frames,
            "progress": round(max(self.frame, 0) / self.frames, 3) if self.frames else 0.0,
            "gap_max_m": self.gap_max_m,
            "wall_s": self.wall_s,
            "options": self.options,
            "cached": self.cached,
            "has_scene": (self.folder / "scene.npz").is_file(),
        }
        if self.summary is not None:
            d["summary"] = self.summary
        if self.error is not None:
            d["error"] = self.error
        return d


class Simulations:
    """Starts simulations one at a time and keeps their state. One instance per app."""

    def __init__(self) -> None:
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="simulation")
        self._jobs: dict[str, SimJob] = {}
        self._lock = threading.Lock()

    def shutdown(self) -> None:
        with self._lock:
            jobs = list(self._jobs.values())
        for j in jobs:
            if j.status in ("queued", "running"):
                self.cancel(j)
        self._pool.shutdown(wait=False, cancel_futures=True)

    @staticmethod
    def info() -> dict[str, Any]:
        """The solver and the choices a simulation offers."""
        return {
            "solver": solver_info(),
            "fabrics": [{"name": f.name, "label": f.label, "note": f.note} for f in FABRICS.values()],
            "qualities": [{"name": k, "mesh_mm": round(v * 1000)} for k, v in QUALITIES.items()],
            "defaults": json.loads(Options().key()),
        }

    def start(self, b: Build, body: dict[str, Any]) -> SimJob:
        """Stitch the layout in ``body`` (cached) and start (or reuse) its simulation."""
        try:
            o = Options.of(body.get("options"))
        except SimulationError as e:
            raise JobError("invalid_simulation", str(e)) from None
        info = solver_info()
        if not info["available"]:
            raise JobError("solver_not_found", info["message"])
        folder, result, _, run = stitching.run(b, body)
        if result["errors"]:
            bad = "; ".join(f"{e['seam']}: {e['error']}" for e in result["errors"][:5])
            raise JobError("stitching_failed", f"seams the stitcher could not resolve: {bad}")
        res_path = folder / "stitching.result.json"
        key = folder_key(res_path, o)
        jid = f"{b.id}-{key}"
        with self._lock:
            j = self._jobs.get(jid)
            if j is not None and j.status in ("queued", "running", "done"):
                return j
            job = SimJob(jid, b.id, result["garment"], b.dir / "simulation" / key, json.loads(o.key()))
            job.frames = o.frames
            if self._load_done(job):
                self._jobs[jid] = job
                return job
            self._jobs[jid] = job
        self._pool.submit(self._run, job, res_path, run, o)
        return job

    @staticmethod
    def _load_done(job: SimJob) -> bool:
        """A finished simulation already on disk (cancelled or failed runs are simulated again)."""
        p = job.folder / "simulation.result.json"
        if not p.is_file():
            return False
        doc = json.loads(p.read_text(encoding="utf-8"))
        if not (doc.get("run") or {}).get("finished"):
            return False
        job.status, job.stage, job.cached, job.summary = "done", "done", True, doc["summary"]
        job.frame = job.frames = int(doc["summary"].get("frames") or 0)
        job.wall_s = doc["summary"].get("wall_s")
        return True

    def _run(self, job: SimJob, res_path: Path, run: Any, o: Options) -> None:
        if job.cancelled.is_set():
            job.status = job.stage = "cancelled"
            return
        job.status, job.stage = "running", "starting"

        def on_event(ev: dict[str, Any]) -> None:
            if ev.get("event") == "stage":
                job.stage = str(ev.get("stage"))
            elif ev.get("event") == "progress":
                job.frame, job.gap_max_m, job.wall_s = int(ev["frame"]), ev.get("gap_max_m"), ev.get("wall_s")

        def started(sr: SolverRun) -> None:
            job.runner = sr
            if job.cancelled.is_set():
                sr.cancel()

        try:
            doc = simulate(res_path, run, job.folder, o, on_event, started)
            job.summary = doc["summary"]
            job.status = "cancelled" if job.cancelled.is_set() else "done"
            job.stage = job.status
        except Exception as e:  # noqa: BLE001 - every failure becomes a structured error on the job
            from .errors import to_error

            if job.cancelled.is_set():
                job.status = job.stage = "cancelled"
                return
            code = "simulation_failed" if isinstance(e, SimulationError) else to_error(e)[1]["error"]["code"]
            job.status, job.stage = "failed", "failed"
            job.error = {"code": code, "message": str(e)[-4000:]}
        finally:
            job.runner = None

    def get(self, jid: str, builds: Any) -> SimJob:
        """A job by id; one from an earlier server run is found again from its folder."""
        with self._lock:
            j = self._jobs.get(jid)
        if j is not None:
            return j
        bid, _, key = jid.rpartition("-")
        b = builds.get(bid) if bid else None
        if b is not None and key:
            job = SimJob(jid, bid, "garment", b.dir / "simulation" / key, {})
            p = job.folder / "simulation.result.json"
            if p.is_file():
                doc = json.loads(p.read_text(encoding="utf-8"))
                job.garment, job.options = doc.get("garment", "garment"), doc.get("options", {})
                job.status, job.stage, job.cached, job.summary = "done", "done", True, doc["summary"]
                job.frame = job.frames = int(doc["summary"].get("frames") or 0)
                with self._lock:
                    self._jobs[jid] = job
                return job
        raise JobError("simulation_not_found", f"no simulation {jid!r}")

    def cancel(self, job: SimJob) -> SimJob:
        job.cancelled.set()
        if job.status == "queued":
            job.status = job.stage = "cancelled"
        elif job.runner is not None:
            threading.Thread(target=job.runner.cancel, daemon=True).start()
            job.stage = "cancelling"
        return job

    # -- data for the viewer -------------------------------------------------------------------------------------------
    @staticmethod
    def embedding(job: SimJob) -> dict[str, Any]:
        """Per piece the simulation vertices and weights that carry every vertex of its placed mesh."""
        if not (job.folder / "scene.npz").is_file():
            raise JobError("simulation_not_ready", f"simulation {job.id} has no scene yet ({job.stage})")
        sa = SceneArrays(job.folder)
        n = len(sa.V)
        dtype = "<u2" if n < 65536 else "<u4"
        pieces = []
        for p in sa.doc["pieces"]:
            idx, w = sa.embedding(p["name"])
            pieces.append({"name": p["name"], "count": int(len(idx)), "idx": _b64(idx, dtype), "w": _b64(w, "<f4")})
        return {"sim_vertices": n, "idx_dtype": "uint16" if dtype == "<u2" else "uint32", "pieces": pieces,
                "start": _b64(sa.V, "<f4")}  # fmt: skip

    @staticmethod
    def latest(job: SimJob) -> dict[str, Any]:
        """The newest frame of the simulation mesh (while running: the solver's latest; after: the last one)."""
        frames = job.folder / "solver" / "frames.npz"
        if job.status == "done" and frames.is_file():
            z = np.load(frames)
            return {"frame": int(z["frame"][-1]), "x": _b64(z["X"][-1], "<f4")}
        p = job.folder / "solver" / "latest.npy"
        if not p.is_file():
            return {"frame": -1, "x": None}
        try:
            x = np.load(p)
        except (OSError, ValueError):  # being replaced right now
            return {"frame": -1, "x": None}
        return {"frame": job.frame, "x": _b64(x, "<f4")}

    @staticmethod
    def frames(job: SimJob) -> dict[str, Any]:
        """Every kept frame of the simulation mesh (T, N, 3), for the animation."""
        p = job.folder / "solver" / "frames.npz"
        if not p.is_file():
            raise JobError("simulation_not_ready", f"simulation {job.id} has no frames yet ({job.stage})")
        z = np.load(p)
        return {"frames": z["frame"].tolist(), "vertices": int(z["X"].shape[1]), "x": _b64(z["X"], "<f4")}

    @staticmethod
    def result_path(job: SimJob) -> Path:
        p = job.folder / "simulation.result.json"
        if not p.is_file():
            raise JobError("simulation_not_ready", f"simulation {job.id} is {job.status}")
        return p

    @staticmethod
    def zip(job: SimJob) -> Path:
        """``<garment>_simulation.zip``: the result, the draped OBJ, the draped meshes and every image."""
        res = Simulations.result_path(job)
        zp = job.folder / "simulation.zip"
        if not zp.is_file():
            files: dict[str, Path | bytes] = {"simulation.result.json": res}
            for name in ("draped_m.obj", "draped.npz", "scene.json"):
                if (job.folder / name).is_file():
                    files[name] = job.folder / name
            for img in sorted((job.folder / "images").glob("*.png")):
                files[f"images/{img.name}"] = img
            webio.write_zip(zp, files)
        return zp

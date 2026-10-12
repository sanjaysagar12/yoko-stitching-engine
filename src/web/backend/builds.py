"""Builds: pieces (sm2d_engine) + avatar (avatar_engine) for one job, cached by a hash of the inputs and versions."""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import avatar_engine
import sm2d_engine
from avatar_engine import AvatarEngine
from sm2d_engine import Engine, EngineOptions, EngineRequest

from .config import AppConfig
from .errors import error_body, piece_failure, to_error
from .jobs import JobFile, load_job, resolve_inputs

log = logging.getLogger(__name__)
_FORMAT = "build-format-1"


@dataclass
class Build:
    """State of one build. ``result`` is the JSON served to the frontend once ``status == "done"``."""

    id: str
    job: str
    dir: Path
    status: str = "queued"  # queued | running | done | failed
    stage: str = "queued"
    progress: float = 0.0
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    cached: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def public(self) -> dict[str, Any]:
        """Status document (``result`` only when done)."""
        d: dict[str, Any] = {
            "id": self.id,
            "job": self.job,
            "status": self.status,
            "stage": self.stage,
            "progress": round(self.progress, 3),
            "cached": self.cached,
        }
        if self.result is not None:
            d["result"] = self.result
        if self.error is not None:
            d.update(self.error)
        return d


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def input_hash(sm2d: bytes, smis: bytes, body: bytes, pieces: list[str] | None) -> str:
    """Content hash of the three input files plus the engine versions that produce the outputs."""
    h = hashlib.sha256()
    for part in (
        sm2d,
        smis,
        body,
        json.dumps(pieces).encode(),
        f"{_FORMAT}|sm2d {sm2d_engine.__version__}|avatar {avatar_engine.__version__}".encode(),
    ):
        h.update(len(part).to_bytes(8, "big"))
        h.update(part)
    return h.hexdigest()


class BuildManager:
    """Starts builds in a small thread pool and keeps their state. One instance per app (no globals)."""

    def __init__(self, cfg: AppConfig, workers: int = 2) -> None:
        self.cfg = cfg
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="build")
        self._builds: dict[str, Build] = {}
        self._lock = threading.Lock()

    def shutdown(self) -> None:
        """Stop accepting work (waits for running builds)."""
        self._pool.shutdown(wait=True)

    # -- API -------------------------------------------------------------------------------------------------------
    def start(self, job_name: str) -> Build:
        """Start (or reuse) the build of a job. Raises JobError for bad jobs / missing files."""
        job = load_job(self.cfg, job_name)
        sm2d_p, smis_p, body_p = resolve_inputs(self.cfg, job)
        digest = input_hash(sm2d_p.read_bytes(), smis_p.read_bytes(), body_p.read_bytes(), job.pieces)
        bid = f"{job.name}-{digest[:12]}"
        with self._lock:
            existing = self._builds.get(bid)
            if existing is not None and existing.status != "failed":
                return existing
            out = self.cfg.outputs_dir / job.name / digest[:16]
            b = Build(bid, job.name, out)
            cached = out / "build.json"
            if cached.is_file():
                b.result = json.loads(cached.read_text(encoding="utf-8"))
                b.status, b.stage, b.progress, b.cached = "done", "done", 1.0, True
            self._builds[bid] = b
            if not b.cached:
                self._pool.submit(self._run, b, job, sm2d_p, smis_p, body_p)
            return b

    def get(self, bid: str) -> Build | None:
        """Build by id (also builds restored from the cache after a restart, if their folder exists)."""
        with self._lock:
            b = self._builds.get(bid)
        if b is not None:
            return b
        job, _, short = bid.rpartition("-")
        for d in (self.cfg.outputs_dir / job).glob(f"{short}*") if job and short else []:
            if (d / "build.json").is_file():
                b = Build(
                    bid, job, d, "done", "done", 1.0, json.loads((d / "build.json").read_text("utf-8")), None, True
                )
                with self._lock:
                    self._builds[bid] = b
                return b
        return None

    # -- the build itself ---------------------------------------------------------------------------------------------
    def _set(self, b: Build, stage: str, progress: float) -> None:
        with b.lock:
            b.status, b.stage, b.progress = "running", stage, progress

    def _run(self, b: Build, job: JobFile, sm2d_p: Path, smis_p: Path, body_p: Path) -> None:
        try:
            self._set(b, "pieces", 0.05)
            pieces_dir, avatar_dir = b.dir / "pieces", b.dir / "avatar"
            res = Engine().run(
                EngineRequest(
                    sm2d_p.read_text(encoding="utf-8-sig"),
                    smis_p.read_text(encoding="utf-8-sig"),
                    EngineOptions(output_dir=pieces_dir, pieces=job.pieces, combined_layout=True),
                    sm2d_p.name,
                    smis_p.name,
                )
            )
            pieces = [self._piece_doc(p, pieces_dir, b.dir) for p in res.pieces]
            self._set(b, "avatar", 0.5)
            AvatarEngine().build(body_p.read_text(encoding="utf-8-sig"), self.cfg.models_dir, avatar_dir)
            self._set(b, "finishing", 0.95)
            bundle = _read_json(avatar_dir / "avatar_bundle.json")
            result = {
                "id": b.id,
                "job": job.name,
                "sm2d": sm2d_p.name,
                "smis": smis_p.name,
                "body": body_p.name,
                "pieces": pieces,
                "layout_file": next(
                    (f"pieces/{Path(x).name}" for x in res.output_paths if x.endswith("_layout_m.obj")), None
                ),
                "avatar": {
                    "bundle": bundle,
                    "mesh": f"avatar/{bundle['mesh']}",
                    "measurements": _read_json(avatar_dir / bundle["files"]["measurements_report"]),
                    "points": _read_json(avatar_dir / bundle["files"]["arrangement_points"])["points"],
                    "volumes": _read_json(avatar_dir / bundle["files"]["bounding_volumes"])["volumes"],
                },
                "warnings": res.warnings,
            }
            (b.dir / "build.json").write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
            with b.lock:
                b.result, b.status, b.stage, b.progress = result, "done", "done", 1.0
        except Exception as e:  # noqa: BLE001 - any failure becomes a structured error on the build
            _, body = to_error(e)
            with b.lock:
                b.status, b.stage, b.error = "failed", "failed", body

    @staticmethod
    def _piece_doc(p: Any, pieces_dir: Path, build_dir: Path) -> dict[str, Any]:
        if p.status != "ok":
            return {"name": p.name, "status": "failed", "reason": p.reason, **piece_failure(p.reason or "")}
        bundle_file = pieces_dir / str(p.file).replace("_m.obj", ".piece.json")
        bundle = json.loads(bundle_file.read_text(encoding="utf-8"))
        return {
            "name": p.name,
            "status": "ok",
            "width_cm": p.width_cm,
            "height_cm": p.height_cm,
            "area_cm2": p.area_cm2,
            "self_intersections": p.self_intersections,
            "warnings": p.warnings,
            "obj": f"pieces/{p.file}",
            "bundle": f"pieces/{bundle_file.name}",
            "named_points": [
                {"name": q["name"], "on_outline": q["on_outline"], "xy_m": q["xy_m"]} for q in bundle["named_points"]
            ],
            "default_reference": bundle["default_reference"]["xy_m"],
        }


__all__ = ["Build", "BuildManager", "error_body", "input_hash"]

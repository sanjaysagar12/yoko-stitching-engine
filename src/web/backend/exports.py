"""Image export of the layout on screen: one zip with every view, heat maps, sections, contact sheet and close-ups.

The request carries the placements with their ``solved`` blocks; the server only aligns them incrementally (nothing
moves, nothing is solved from scratch), renders with the assembler's own renderer and zips with the standard library.
Jobs run in a small thread pool, can be cancelled, and are cached by a hash of (build, layout, options).
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from alignment_engine import AlignmentEngine
from alignment_engine.levels import body_levels
from alignment_engine.loaders import read_avatar
from assembler import verify as V
from assembler import webio
from assembler.run import Inputs, _md, bundles_of, render_result, run_view, spec_of, verify_run

from .builds import Build
from .errors import to_error
from .jobs import JobError

_NAME = re.compile(r"^[A-Za-z0-9_.-]{1,40}$")
MAX_SHOT = 6_000_000  # bytes of one viewport screenshot
OPTION_KEYS = ("closeups", "heat", "verify", "assembly")


class Cancelled(Exception):
    """The user cancelled the export."""


@dataclass
class ExportJob:
    """State of one image export."""

    id: str
    build: str
    garment: str
    status: str = "queued"  # queued | running | done | failed | cancelled
    stage: str = "queued"
    progress: float = 0.0
    error: dict[str, Any] | None = None
    zip: Path | None = None
    files: int = 0
    failures: list[str] = field(default_factory=list)
    cancel: threading.Event = field(default_factory=threading.Event, repr=False)

    def public(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": self.id,
            "build": self.build,
            "garment": self.garment,
            "status": self.status,
            "stage": self.stage,
            "progress": round(self.progress, 3),
            "files": self.files,
            "failures": self.failures,
        }
        if self.error:
            d.update(self.error)
        return d


def options_of(body: dict[str, Any]) -> dict[str, bool]:
    """closeups / heat / verify / assembly switches (defaults on, on, off, on)."""
    given = body.get("options") or {}
    default = {"closeups": True, "heat": True, "verify": False, "assembly": True}
    return {k: bool(given.get(k, default[k])) for k in OPTION_KEYS}


def screenshots_of(body: dict[str, Any]) -> dict[str, bytes]:
    """Viewport screenshots ``{name: base64 png}`` (names and sizes checked; never trusted)."""
    out: dict[str, bytes] = {}
    for name, b64 in (body.get("viewport") or {}).items():
        if not _NAME.match(str(name)):
            raise JobError("invalid_request", f"bad screenshot name {name!r}")
        try:
            raw = base64.b64decode(str(b64).split(",")[-1], validate=True)
        except (binascii.Error, ValueError):
            raise JobError("invalid_request", f"screenshot {name!r} is not base64") from None
        if len(raw) > MAX_SHOT or not raw.startswith(b"\x89PNG"):
            raise JobError("invalid_request", f"screenshot {name!r} is not a PNG within {MAX_SHOT // 1_000_000} MB")
        out[str(name)] = raw
    return out


def cache_key(build_id: str, body: dict[str, Any], opts: dict[str, bool], shots: dict[str, bytes]) -> str:
    layout = {
        k: body.get(k) for k in ("placements", "derived_pieces", "arrange_options", "stitches", "rings", "expect")
    }
    layout["spec"] = spec_of(body)
    text = json.dumps(
        [build_id, layout, opts, body.get("garment"), body.get("notes"), body.get("mode")],
        sort_keys=True,
        default=str,
    )
    h = hashlib.sha256(text.encode())
    for n in sorted(shots):
        h.update(n.encode() + hashlib.sha256(shots[n]).digest())
    return h.hexdigest()[:16]


class ImageExports:
    """Starts, tracks and cancels image exports (one per app)."""

    def __init__(self, body_file_of: Any = None) -> None:
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="export")
        self._jobs: dict[str, ExportJob] = {}
        self._lock = threading.Lock()
        self._body_file_of = body_file_of  # job name -> body JSON dict (for the optional verification)

    def shutdown(self) -> None:
        for j in self._jobs.values():
            j.cancel.set()
        self._pool.shutdown(wait=True)

    def get(self, jid: str) -> ExportJob | None:
        with self._lock:
            return self._jobs.get(jid)

    def cancel(self, jid: str) -> ExportJob | None:
        j = self.get(jid)
        if j is not None and j.status in ("queued", "running"):
            j.cancel.set()
        return j

    def start(self, b: Build, body: dict[str, Any]) -> ExportJob:
        """Start (or reuse) the export of ``body``'s layout on build ``b``."""
        if not body.get("placements"):
            raise JobError("no_placements", "place at least one piece before exporting images")
        opts, shots = options_of(body), screenshots_of(body)
        garment = str(body.get("garment") or b.job)
        if not webio.SAFE.match(garment):
            raise JobError("invalid_request", "garment name may only use letters, digits, _ . -")
        jid = cache_key(b.id, body, opts, shots)
        with self._lock:
            old = self._jobs.get(jid)
            if old is not None and (
                old.status in ("queued", "running") or (old.status == "done" and old.zip and old.zip.is_file())
            ):
                return old
            job = ExportJob(jid, b.id, garment)
            self._jobs[jid] = job
            self._pool.submit(self._run, job, b, body, opts, shots)
            return job

    # -- the work ---------------------------------------------------------------------------------------------------
    def _set(self, j: ExportJob, stage: str, p: float) -> None:
        if j.cancel.is_set():
            raise Cancelled
        j.status, j.stage, j.progress = "running", stage, p

    def _run(
        self, j: ExportJob, b: Build, body: dict[str, Any], opts: dict[str, bool], shots: dict[str, bytes]
    ) -> None:
        root = b.dir / "export_images"
        work = root / j.id / "web"
        try:
            self._set(j, "aligning (nothing moves)", 0.02)
            spec = spec_of(body)
            res = AlignmentEngine().align(
                b.dir / "avatar",
                b.dir / "pieces",
                json.dumps(spec),
                derived_dir=b.dir / "derived",
                cache_dir=b.dir / "refined",
            )
            placements = [
                {**pl, "solved": d["solved"]} if d.get("solved") else pl
                for pl, d in zip(spec["placements"], res.doc["placements"], strict=True)
            ]
            doc = webio.export_doc(
                job=b.job,
                placements=placements,
                derived_pieces=spec.get("derived_pieces"),
                arrange_options=spec.get("arrange_options"),
                stitches=spec.get("stitches"),
                rings=spec.get("rings"),
                garment=j.garment,
                notes=str(body.get("notes") or ""),
                mode=str(body.get("mode") or "arrange"),
                include_solved=True,
                expect=body.get("expect") or None,
                views=body.get("views") or None,
            )
            inp = Inputs(b.dir / "pieces", b.dir / "avatar", None, "", {})

            def progress(name: str, frac: float) -> None:
                self._set(j, f"rendering {name}", 0.08 + 0.85 * frac)

            paths = render_result(
                work,
                doc,
                inp,
                res,
                all_views=True,
                closeups=opts["closeups"],
                heat=opts["heat"],
                progress=progress,
                failures=j.failures,
            )
            self._set(j, "packing", 0.95)
            files: dict[str, Path | bytes] = {}
            for p in paths.values():
                files[p.relative_to(work / "images" if p.is_relative_to(work / "images") else work).as_posix()] = p
            if opts["assembly"]:
                files["assembly.json"] = webio.dumps(doc).encode("utf-8")
            if opts["verify"]:
                self._verify(j, b, doc, spec, res, inp, files)
            for n, raw in shots.items():
                files[f"viewport/{n}.png"] = raw
            report = [f"rendered {len(paths)} images for {j.garment} on build {b.id}"]
            report += [f"FAILED {f}" for f in j.failures] or ["no failures"]
            files["export_report.txt"] = ("\n".join(report) + "\n").encode("utf-8")
            j.zip = webio.write_zip(root / f"{j.id}.zip", files)
            j.files = len(files)
            j.status, j.stage, j.progress = "done", "done", 1.0
        except Cancelled:
            j.status, j.stage = "cancelled", "cancelled"
        except Exception as e:  # noqa: BLE001 - any failure becomes a structured error on the job
            _, err = to_error(e)
            j.status, j.stage, j.error = "failed", "failed", err
        finally:
            shutil.rmtree(root / j.id, ignore_errors=True)

    def _verify(
        self,
        j: ExportJob,
        b: Build,
        doc: dict[str, Any],
        spec: dict[str, Any],
        res: Any,
        inp: Inputs,
        files: dict[str, Path | bytes],
    ) -> None:
        """Quick check of the primary layout (``verify.md`` / ``verify.json``); a failure is reported, not fatal."""
        try:
            self._set(j, "verifying", 0.93)
            avatar = read_avatar(inp.avatar)
            body = self._body_file_of(b.job) if self._body_file_of else None
            vin = Inputs(inp.pieces, inp.avatar, body, V.mesh_hash(avatar.vertices), {})
            points = {
                p["name"]: p for p in json.loads((inp.avatar / "arrangement_points.json").read_text("utf-8"))["points"]
            }
            rv = run_view("primary", res, spec, bundles_of(vin, b.dir / "derived"))
            extra = verify_run(
                rv,
                V.Surf(avatar.vertices, avatar.faces),
                points,
                body_levels(avatar),
                avatar,
                vin,
                doc.get("expect", {}),
            )
            rep = {
                "rows": rv.rows,
                "engine_valid": bool(res.doc["validity"]["valid"]),
                "fit": extra["fit"],
                "ease": extra["ease"],
            }
            files["verify.md"] = _md(
                j.garment, "web", {"primary": rep}, {"global": [], "warnings": res.warnings}
            ).encode()
            summary = {"garment": j.garment, "pass": V.verdict(rv.rows), "runs": {"primary": rep}}
            files["verify.json"] = (json.dumps(summary, indent=1, default=str) + "\n").encode()
        except Cancelled:
            raise
        except Exception as e:  # noqa: BLE001
            j.failures.append(f"verify: {type(e).__name__}: {e}")

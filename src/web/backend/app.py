"""FastAPI application: jobs, builds, alignment, placements, export. Thin: every calculation is an engine call."""

from __future__ import annotations

import base64
import hashlib
import json
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Any

import numpy as np
from fastapi import Body, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from alignment_engine import AlignmentEngine
from assembler import webio
from assembler.run import AssemblyError

from .builds import BuildManager
from .config import AppConfig
from .errors import ENGINE_ERRORS, error_body, to_error
from .exports import ImageExports
from .jobs import JobError, list_jobs, load_job, resolve_inputs

_SAFE = re.compile(r"^[A-Za-z0-9_.-]+$")


def _b64(a: Any) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode("ascii")


EMPTY_SPEC: dict[str, Any] = {"schema_version": 2, "placements": []}


def create_app(cfg: AppConfig | None = None) -> FastAPI:
    """Build the app. ``cfg`` defaults to the repository layout."""
    cfg = cfg or AppConfig.default()
    mgr = BuildManager(cfg)
    aligner = AlignmentEngine()

    def body_json(job: str) -> dict[str, Any]:
        loaded: dict[str, Any] = json.loads(resolve_inputs(cfg, load_job(cfg, job))[2].read_text(encoding="utf-8-sig"))
        return loaded

    exporter = ImageExports(body_json)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        exporter.shutdown()
        mgr.shutdown()

    app = FastAPI(title="Pattern fit viewer", version="0.2.0", lifespan=lifespan)
    app.state.cfg, app.state.builds = cfg, mgr

    # -- errors: always structured JSON ------------------------------------------------------------------------
    @app.exception_handler(StarletteHTTPException)
    async def http_error(_: Request, e: StarletteHTTPException) -> JSONResponse:
        code = {404: "not_found", 405: "method_not_allowed"}.get(e.status_code, "http_error")
        return JSONResponse(error_body(code, str(e.detail)), status_code=e.status_code)

    @app.exception_handler(RequestValidationError)
    async def bad_request(_: Request, e: RequestValidationError) -> JSONResponse:
        msg = "; ".join(f"{'/'.join(str(x) for x in err['loc'])}: {err['msg']}" for err in e.errors()[:5])
        return JSONResponse(error_body("invalid_request", msg), status_code=422)

    async def engine_error(_: Request, e: Exception) -> JSONResponse:
        status, body = to_error(e)
        return JSONResponse(body, status_code=status)

    # specific classes (handled inside the app, so they never become server errors) plus a catch-all
    for cls in (*ENGINE_ERRORS, Exception):
        app.add_exception_handler(cls, engine_error)

    def build_or_404(bid: str) -> Any:
        b = mgr.get(bid)
        if b is None:
            raise JobError("build_not_found", f"no build {bid!r}")
        return b

    @app.exception_handler(JobError)
    async def job_error(_: Request, e: JobError) -> JSONResponse:
        status = 404 if e.code in ("job_not_found", "build_not_found", "export_not_found") else 422
        extra = {"problems": e.problems} if e.problems else {}
        return JSONResponse(error_body(e.code, str(e), **extra), status_code=status)

    # -- jobs and builds -----------------------------------------------------------------------------------------------
    @app.get("/api/jobs")
    def jobs() -> list[dict[str, Any]]:
        return [j.model_dump() for j in list_jobs(cfg)]

    @app.post("/api/jobs/{name}/build")
    def start_build(name: str) -> dict[str, Any]:
        b = mgr.start(name)
        return {"build_id": b.id, "status": b.status, "cached": b.cached}

    @app.get("/api/builds/{bid}")
    def build_status(bid: str) -> dict[str, Any]:
        return build_or_404(bid).public()  # type: ignore[no-any-return]

    @app.get("/api/builds/{bid}/files/{path:path}")
    def build_file(bid: str, path: str) -> Response:
        b = build_or_404(bid)
        target = (b.dir / path).resolve()
        if not target.is_relative_to(b.dir.resolve()) or not target.is_file():
            raise JobError("file_not_found", f"no file {path!r} in build {bid}")
        return FileResponse(target)

    # -- alignment ---------------------------------------------------------------------------------------------------
    def spec_of(body: dict[str, Any]) -> dict[str, Any]:
        spec: dict[str, Any] = {"schema_version": int(body.get("schema_version", 5))}
        spec.update(
            {k: body[k] for k in ("placements", "stitches", "rings", "derived_pieces", "arrange_options") if k in body}
        )
        spec.setdefault("placements", [])
        return spec

    def ready_build(body: dict[str, Any]) -> Any:
        b = build_or_404(str(body.get("build", "")))
        if b.status != "done":
            raise JobError("build_not_ready", f"build {b.id} is {b.status}")
        return b

    def answer(res: Any, body: dict[str, Any]) -> Response:
        """The alignment result (always with ``validity`` and per piece ``size_check`` and ``solved``).

        With ``compact: true`` the rolled meshes travel as base64 (``wrapped_mesh``: float32 vertices, uint32 faces,
        ``mesh_id``); faces are left out for the ``known_meshes`` ids the browser already holds. Same numbers, about a
        tenth of the bytes of the JSON arrays.
        """
        if not body.get("compact"):
            return JSONResponse(res.doc)
        known = set(body.get("known_meshes") or [])
        doc = dict(res.doc)
        docs = []
        for d, p in zip(res.doc["placements"], res.placed, strict=True):
            d = dict(d)
            if d.get("wrapped_vertices") is not None:
                faces = np.ascontiguousarray(p.faces, dtype=np.uint32)
                mid = hashlib.sha1(faces.tobytes() + str(len(p.vertices)).encode()).hexdigest()[:16]
                mesh = {"mesh_id": mid, "vertices": _b64(np.asarray(p.vertices, dtype="<f4"))}
                if mid not in known:
                    mesh["faces"] = _b64(faces.astype("<u4"))
                d["wrapped_vertices"], d["wrapped_faces"], d["wrapped_mesh"] = None, None, mesh
            docs.append(d)
        doc["placements"] = docs
        return JSONResponse(doc)

    @app.post("/api/align")
    def align(body: Annotated[dict[str, Any], Body()]) -> Response:
        """Place the pieces collision-free and size-exact: ``{build, placements, derived_pieces, arrange_options?,
        compact?, known_meshes?}``. Placements carrying ``solved`` (from an earlier answer) stay where they are; the
        others are solved against them. The answer carries ``validity`` (never valid if anything collides or changed
        size) and per piece ``size_check`` and ``solved``."""
        b = ready_build(body)
        res = aligner.align(
            b.dir / "avatar",
            b.dir / "pieces",
            json.dumps(spec_of(body)),
            derived_dir=b.dir / "derived",
            cache_dir=b.dir / "refined",
        )
        return answer(res, body)

    @app.post("/api/arrange")
    def arrange(body: Annotated[dict[str, Any], Body()]) -> Response:
        """Re-solve every piece that is not locked (same body and answer as ``/api/align``)."""
        b = ready_build(body)
        res = aligner.arrange(
            b.dir / "avatar",
            b.dir / "pieces",
            json.dumps(spec_of(body)),
            derived_dir=b.dir / "derived",
            cache_dir=b.dir / "refined",
        )
        return answer(res, body)

    @app.post("/api/clone")
    def clone(body: Annotated[dict[str, Any], Body()]) -> dict[str, Any]:
        """Make a derived piece: ``{build, source, id, op?, axis?, derived_pieces?}`` -> the derived piece.

        ``op`` ``mirror`` (default, needs ``axis``) is a linked mirrored copy; ``copy`` an exact copy (no ``axis``).
        ``derived_pieces`` are the existing derived pieces, needed to copy one of them.
        """
        b = ready_build(body)
        op = body.get("op", "mirror")
        new = {"id": body.get("id"), "source": body.get("source"), "op": op}
        if op != "copy" or "axis" in body:  # an axis on a copy is rejected by the spec schema
            new["axis"] = body.get("axis")
        derived = [d for d in body.get("derived_pieces") or [] if d.get("id") != new["id"]] + [new]
        res = aligner.align(
            b.dir / "avatar",
            b.dir / "pieces",
            json.dumps(spec_of({"derived_pieces": derived})),
            derived_dir=b.dir / "derived",
        )
        d = res.doc["derived_pieces"][-1]
        bundle = json.loads((b.dir / "derived" / d["bundle"]).read_text(encoding="utf-8"))
        link = (
            {"clone_of": bundle["clone_of"], "mirror_of": None}
            if op == "copy"
            else {"mirror_of": bundle["mirror_of"], "axis": d["axis"]}
        )
        return {
            "id": d["id"],
            "name": d["id"],
            "status": "ok",
            "source": d["source"],
            "op": d["op"],
            **link,
            "origin_shift_m": d["origin_shift_m"],
            "width_cm": d["bbox_cm"]["width"],
            "height_cm": d["bbox_cm"]["height"],
            "obj": f"derived/{d['obj']}",
            "bundle": f"derived/{d['bundle']}",
            "named_points": [
                {"name": q["name"], "on_outline": q["on_outline"], "xy_m": q["xy_m"]} for q in bundle["named_points"]
            ],
            "default_reference": bundle["default_reference"]["xy_m"],
            "suggested_stitch": d.get("suggested_stitch"),
        }

    @app.post("/api/pair")
    def pair(body: Annotated[dict[str, Any], Body()]) -> dict[str, Any]:
        """Placements for a piece and its clone: ``{build, base, derived_id, mode, derived_pieces}``."""
        b = ready_build(body)
        warnings: list[str] = []
        placements = aligner.pair(
            b.dir / "avatar",
            b.dir / "pieces",
            list(body.get("derived_pieces", [])),
            dict(body.get("base", {})),
            str(body.get("derived_id", "")),
            str(body.get("mode", "same_anchor")),
            warnings,
        )
        return {"placements": placements, "warnings": warnings}

    # -- assembly files (import / export) and image export ------------------------------------------------------------
    def problems_error(e: webio.AssemblyProblems) -> JobError:
        return JobError(e.code, str(e), [{"code": c, "message": m} for c, m in e.problems])

    @app.post("/api/assembly/import")
    async def assembly_import(
        request: Request, job: str | None = None, build: str | None = None, skip_missing: bool = False
    ) -> dict[str, Any]:
        """Read an assembly file (the request body) into the payload the viewer applies. Without ``build`` only the file
        and its job are checked; with the id of the job's build every piece and point is checked too. Accepts the older
        ``{placements, derived_pieces}`` shape (name its job with ``?job=``). Lists every problem."""
        if int(request.headers.get("content-length") or 0) > webio.MAX_BYTES:
            raise JobError("invalid_assembly", "file is larger than 2 MB")
        raw = await request.body()
        try:
            doc, warns = webio.parse(raw, job)
            try:
                load_job(cfg, doc["job"])
            except JobError:
                raise webio.AssemblyProblems([("unknown_job", f"no job named {doc['job']!r} in this app")]) from None
            if build:
                b = ready_build({"build": build})
                if b.job != doc["job"]:
                    raise webio.AssemblyProblems(
                        [("unknown_job", f"build {build} is of job {b.job!r}, not {doc['job']!r}")]
                    )
                pieces = {p["name"] for p in b.result["pieces"] if p["status"] == "ok"}
                points = {p["name"] for p in b.result["avatar"]["points"]}
                doc, more = webio.check_build(doc, pieces, points, skip_missing)
                warns += more
        except webio.AssemblyProblems as e:
            raise problems_error(e) from None
        return webio.normalise(doc, warns)

    @app.post("/api/assembly/export")
    def assembly_export(body: Annotated[dict[str, Any], Body()]) -> Response:
        """The current layout as an assembly file: ``{job, placements, derived_pieces, arrange_options, stitches, rings,
        garment, notes, mode, include_solved, expect?, views?}``. Validated before it is returned."""
        load_job(cfg, str(body.get("job", "")))
        try:
            doc = webio.export_doc(
                job=str(body["job"]),
                placements=list(body.get("placements") or []),
                derived_pieces=body.get("derived_pieces"),
                arrange_options=body.get("arrange_options"),
                stitches=body.get("stitches"),
                rings=body.get("rings"),
                garment=body.get("garment") or None,
                notes=str(body.get("notes") or ""),
                mode=str(body.get("mode") or "arrange"),
                include_solved=bool(body.get("include_solved", True)),
                expect=body.get("expect") or None,
                views=body.get("views") or None,
            )
        except AssemblyError as e:
            raise JobError("invalid_assembly", str(e)) from None
        return Response(webio.dumps(doc), media_type="application/json")

    @app.post("/api/export/images")
    def export_images(body: Annotated[dict[str, Any], Body()]) -> dict[str, Any]:
        """Start the image zip of the layout on screen (see ``exports.py``); poll ``GET /api/export/images/{id}``."""
        return exporter.start(ready_build(body), body).public()

    def export_job(jid: str) -> Any:
        j = exporter.get(jid)
        if j is None:
            raise JobError("export_not_found", f"no image export {jid!r}")
        return j

    @app.get("/api/export/images/{jid}")
    def export_images_status(jid: str) -> dict[str, Any]:
        return export_job(jid).public()  # type: ignore[no-any-return]

    @app.delete("/api/export/images/{jid}")
    def export_images_cancel(jid: str) -> dict[str, Any]:
        export_job(jid)
        return exporter.cancel(jid).public()  # type: ignore[union-attr]

    @app.get("/api/export/images/{jid}/download")
    def export_images_download(jid: str) -> Response:
        j = export_job(jid)
        if j.status != "done" or j.zip is None or not j.zip.is_file():
            raise JobError("export_not_ready", f"image export {jid} is {j.status}")
        return FileResponse(j.zip, media_type="application/zip", filename=f"{j.garment}_images.zip")

    @app.get("/api/jobs/{name}/placements")
    def get_placements(name: str) -> dict[str, Any]:
        load_job(cfg, name)
        p = cfg.outputs_dir / name / "placements.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else dict(EMPTY_SPEC)

    @app.put("/api/jobs/{name}/placements")
    def put_placements(name: str, body: Annotated[dict[str, Any], Body()]) -> dict[str, Any]:
        load_job(cfg, name)
        doc = aligner.validate_spec(json.dumps({"schema_version": 2, **body}))
        p = cfg.outputs_dir / name / "placements.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8", newline="\n")
        return doc

    # -- export ------------------------------------------------------------------------------------------------------
    @app.get("/api/export/{bid}")
    def export(bid: str, format: str = "obj") -> Response:  # noqa: A002 - query parameter name is part of the API
        if format not in ("obj", "glb"):
            raise JobError("invalid_request", "format must be obj or glb")
        b = build_or_404(bid)
        if b.status != "done":
            raise JobError("build_not_ready", f"build {b.id} is {b.status}")
        saved = cfg.outputs_dir / b.job / "placements.json"
        spec = json.loads(saved.read_text(encoding="utf-8")) if saved.is_file() else dict(EMPTY_SPEC)
        out = b.dir / "export"
        aligner.align(b.dir / "avatar", b.dir / "pieces", json.dumps(spec), out, cache_dir=b.dir / "refined")
        scene = out / "scene_m.obj"
        if format == "obj":
            return FileResponse(scene, media_type="text/plain", filename=f"{b.job}_scene_m.obj")
        import trimesh  # glb conversion only; the geometry is already final

        glb = out / "scene.glb"
        geometry: Any = trimesh.load(str(scene), file_type="obj", process=False, group_material=False)
        geometry.export(str(glb))
        return FileResponse(glb, media_type="model/gltf-binary", filename=f"{b.job}_scene.glb")

    # -- the built frontend -------------------------------------------------------------------------------------------
    if cfg.frontend_dist.is_dir():
        app.mount("/", StaticFiles(directory=cfg.frontend_dist, html=True), name="frontend")
    return app

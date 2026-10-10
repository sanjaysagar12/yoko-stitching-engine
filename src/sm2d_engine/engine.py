"""Engine facade: orchestrates parse -> evaluate -> assemble -> mesh -> write. No maths of its own."""

from __future__ import annotations

import logging
import platform
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__, units
from .errors import EngineError, UnsupportedToolError
from .eval.scope import VariableScope
from .io.sm2d_reader import parse_sm2d
from .io.smis_reader import parse_smis
from .mesh.mesh import Mesh
from .mesh.placement import layout_offsets, to_origin
from .mesh.triangulate import EarcutTriangulator, Triangulator
from .mesh.validate import MeshValidator
from .mesh.writers import MeshWriter, writer_for
from .model.geometry_types import Polygon
from .model.pattern import PieceDef
from .pieces.assembler import assemble
from .reporting import EngineResult, PieceReport
from .tools import default_registry
from .tools.registry import Registry

log = logging.getLogger(__name__)


@dataclass
class EngineOptions:
    """Run options. ``flatten_tol_mm`` is the max chord deviation of flattened curves (mm)."""

    output_dir: str | Path = "out"
    pieces: list[str] | None = None
    format: str = "obj"
    combined_layout: bool = True
    layout_gap_cm: float = 5.0
    mirror_fold_pieces: bool = False
    flatten_tol_mm: float = 0.1


@dataclass
class EngineRequest:
    """Engine input: file *contents* (the engine knows nothing about where they live)."""

    sm2d_xml: str
    smis_xml: str
    options: EngineOptions = field(default_factory=EngineOptions)
    sm2d_name: str = "pattern.sm2d"
    smis_name: str = "measurements.smis"


@dataclass
class _Built:
    report: PieceReport
    poly_cm: Polygon
    mesh: Mesh
    header: list[str]
    slug: str = ""


def _slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_") or "piece"


def _version(pkg: str) -> str:
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return "unknown"


class Engine:
    """Pattern + measurements -> per-piece real-scale meshes.

    Collaborators are injected (``Triangulator``, writer factory, ``MeshValidator``) so tests can swap them.
    """

    def __init__(
        self,
        triangulator: Triangulator | None = None,
        writer_factory: Callable[[str], MeshWriter] = writer_for,
        validator: MeshValidator | None = None,
        registry: Registry | None = None,
    ) -> None:
        self.triangulator: Triangulator = triangulator or EarcutTriangulator()
        self.writer_factory = writer_factory
        self.validator = validator or MeshValidator()
        self.registry = registry or default_registry

    # -- public --------------------------------------------------------------
    def run(self, request: EngineRequest) -> EngineResult:
        """Evaluate and write meshes + report.json to ``options.output_dir``. Fatal errors raise EngineError."""
        return self._process(request, write=True)

    def inspect(self, request: EngineRequest) -> EngineResult:
        """Same evaluation as ``run`` but writes nothing."""
        return self._process(request, write=False)

    # -- internals -----------------------------------------------------------
    def _process(self, request: EngineRequest, write: bool) -> EngineResult:
        opt = request.options
        model = parse_sm2d(request.sm2d_xml)
        meas = parse_smis(request.smis_xml)
        scope = VariableScope(model, meas, self.registry)
        writer = self.writer_factory(opt.format) if write else None
        out_dir = Path(opt.output_dir)

        by_name: dict[str, PieceDef] = {}
        for p in model.pieces:
            by_name.setdefault(p.name, p)
        wanted = list(by_name) if opt.pieces is None else opt.pieces

        reports: list[PieceReport] = []
        built: list[_Built] = []
        used_slugs: set[str] = set()
        paths: list[str] = []
        for name in wanted:
            piece = by_name.get(name)
            if piece is None:
                reports.append(PieceReport(name, "failed", f"piece {name!r} not found in pattern"))
                continue
            try:
                b = self._build(piece, scope, request)
            except UnsupportedToolError as e:
                reports.append(PieceReport(name, "failed", str(e.with_piece(name))))
                log.warning("piece %s failed: %s", name, e)
                continue
            except EngineError as e:
                reports.append(PieceReport(name, "failed", f"{type(e).__name__}: {e}"))
                log.warning("piece %s failed: %s", name, e)
                continue
            except Exception as e:  # noqa: BLE001 - isolate any per-piece failure, keep the run going
                log.exception("piece %s: internal error", name)
                reports.append(PieceReport(name, "failed", f"internal error: {type(e).__name__}: {e}"))
                continue
            if writer is not None:
                slug, k = _slug(name), 2
                while slug in used_slugs:
                    slug, k = f"{_slug(name)}_{k}", k + 1
                used_slugs.add(slug)
                b.slug = slug
                path = writer.write_piece(out_dir, slug, b.mesh, b.header)
                b.report.file = path.name
                paths.append(str(path))
            reports.append(b.report)
            built.append(b)

        if writer is not None and opt.combined_layout and built:
            offs = layout_offsets([b.report.width_cm or 0.0 for b in built], opt.layout_gap_cm)
            meshes = []
            for b, off in zip(built, offs, strict=True):
                shifted = b.poly_cm + np.array([off, 0.0])
                meshes.append((b.slug, Mesh.from_polygon_cm(shifted, b.mesh.faces)))
            hdr = [
                f"{request.sm2d_name}: all pieces left to right, {opt.layout_gap_cm:g} cm apart",
                "units: metres (1 OBJ unit = 1 m); z = 0; x right, y up",
                f"source measurements: {request.smis_name}",
            ]
            paths.append(str(writer.write_combined(out_dir, Path(request.sm2d_name).stem, meshes, hdr)))

        result = EngineResult(
            pieces=reports,
            output_paths=paths,
            header={
                "pattern": request.sm2d_name,
                "measurements": request.smis_name,
                "units": {
                    "pattern": model.unit,
                    "measurements": meas.source_unit,
                    "report": "cm / cm^2",
                    "mesh": "metres",
                },
                "flatten_tol_mm": opt.flatten_tol_mm,
                "tool_versions": {
                    "sm2d_engine": __version__,
                    "python": platform.python_version(),
                    "numpy": np.__version__,
                    "mapbox_earcut": _version("mapbox_earcut"),
                },
            },
        )
        if writer is not None:
            paths.append(str(writer.write_report(out_dir, result.to_dict())))
        return result

    def _build(self, piece: PieceDef, scope: VariableScope, request: EngineRequest) -> _Built:
        opt = request.options
        outline = assemble(piece, scope, flatten_tol_cm=opt.flatten_tol_mm / 10.0)
        poly, w, h = to_origin(outline.points_cm)
        faces = self.triangulator.triangulate(poly * units.cm_to_m(1.0))
        mesh = Mesh.from_polygon_cm(poly, faces)
        v = self.validator.check(poly, mesh)

        warnings: list[str] = []
        if len(poly) < 3:
            warnings.append("fewer than 3 vertices: no faces written")
        qty, fold = piece.data.get("quantity"), piece.data.get("onFold")
        if qty is None:
            warnings.append("quantity not present in pattern (reported as null)")
        if fold is None:
            warnings.append("onFold not present in pattern (reported as null)")
        fold_pos = piece.data.get("foldPosition")
        if opt.mirror_fold_pieces:
            if (fold or "").lower() != "true":
                warnings.append("mirror_fold_pieces: piece has no fold information; not mirrored")
            elif fold_pos in (None, "", "Undefined"):
                warnings.append("mirror_fold_pieces: onFold is true but foldPosition is Undefined; not mirrored")
            else:
                warnings.append(f"mirror_fold_pieces: foldPosition={fold_pos!r} is not supported; not mirrored")
        elif (fold or "").lower() == "true":
            warnings.append("onFold is true: this is a half piece and is NOT mirrored")

        report = PieceReport(
            piece.name,
            width_cm=w,
            height_cm=h,
            area_cm2=v.area_cm2,
            perimeter_cm=v.perimeter_cm,
            mesh_area_cm2=v.mesh_area_cm2,
            area_diff_pct=v.area_diff_pct,
            vertices=len(poly),
            triangles=len(faces),
            self_intersections=v.self_intersections,
            gaps=outline.notes,
            warnings=warnings,
            quantity=qty,
            on_fold=fold,
        )
        header: list[str] = [
            f"piece: {piece.name}",
            "units: metres (1 OBJ unit = 1 m); z = 0; x right, y up",
            "origin: piece bounding-box bottom-left",
            f"size: {w:.3f} x {h:.3f} cm",
            f"source pattern: {request.sm2d_name}",
            f"source measurements: {request.smis_name}",
            f"quantity: {qty or 'null'}  onFold: {fold or 'null'}  (half pieces are NOT mirrored)",
        ]
        return _Built(report, poly, mesh, header)


__all__: list[Any] = ["Engine", "EngineOptions", "EngineRequest"]

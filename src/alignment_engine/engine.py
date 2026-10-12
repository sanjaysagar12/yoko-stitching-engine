"""Alignment engine facade: avatar folder + piece folder + placement spec -> placed pieces and fit numbers.

Everything is metres. The caller passes folders and the *text* of the placement spec; the engine writes outputs to
``out_dir`` when one is given (and writes nothing otherwise, which is what an interactive caller wants).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from . import contracts
from .derived import Derived, build_derived, derived_doc, write_derived
from .errors import UnknownMeasurementError, UnknownPieceError, UnknownPointError
from .loaders import Avatar, PieceInfo, PointRecord, read_avatar, read_pieces
from .pairs import place_pair
from .placement import apply_matrix, mirror_matrix, mirror_mesh, refine_mesh, rigid_matrix, target_frame
from .refine import RefineCache
from .sides import extends, lateral_offset, side_warnings
from .slice_wrap import TORSO_BVS, SliceInfo, TorsoSections, map_rows
from .stitch import stitch_report
from .widths import row_width
from .wrap import wrap_geometry, wrap_vertices

log = logging.getLogger(__name__)
F64 = NDArray[np.float64]
I64 = NDArray[np.int64]
WRAP_MAX_EDGE_M = 0.01  # wrapped pieces are subdivided to edges of at most 1 cm


@dataclass
class PlacedPiece:
    """One solved placement. ``vertices``/``faces`` are the placed mesh (world, metres)."""

    name: str
    point: str
    matrix: F64  # 4x4: original piece vertices -> world (includes the mirror when ``mirrored``)
    mirrored: bool
    vertices: F64
    faces: I64
    wrapped: bool
    reference_world: F64
    warnings: list[str]
    fit: dict[str, Any]
    transform: Callable[[F64], F64]  # piece-plane points (N,2) -> world (N,3), same mapping as the mesh
    stem: str
    extends: str = "centre"
    source: str = ""


@dataclass
class AlignmentResult:
    """``doc`` is the ``alignment_result`` document; ``placed`` the solved pieces; ``paths`` the files written."""

    doc: dict[str, Any]
    placed: list[PlacedPiece]
    paths: dict[str, Path] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def _r(x: Any, nd: int = 9) -> Any:
    if isinstance(x, np.ndarray):
        return [round(float(v), nd) for v in x.ravel()]
    return round(float(x), nd)


class AlignmentEngine:
    """Places pieces on arrangement points.

    Results depend only on the inputs of a call. The engine keeps one cache, the loaded avatars (and their torso
    sections) keyed by folder and file time, so interactive callers do not reload the avatar on every call.
    """

    def __init__(self) -> None:
        self._avatars: dict[tuple[str, int], Avatar] = {}
        self._refine: dict[str, RefineCache] = {}

    @staticmethod
    def validate_spec(spec_json_text: str) -> dict[str, Any]:
        """Parse and validate a placement spec (raises ContractError); returns the document."""
        doc: dict[str, Any] = contracts.loads("placement_spec", spec_json_text)
        return doc

    def _load(
        self, avatar_dir: Path, piece_dir: Path, derived_defs: list[dict[str, Any]], derived_dir: Path
    ) -> tuple[Avatar, dict[str, PieceInfo], dict[str, Derived]]:
        key = (str(Path(avatar_dir).resolve()), (Path(avatar_dir) / "avatar_bundle.json").stat().st_mtime_ns)
        avatar = self._avatars.get(key) or read_avatar(Path(avatar_dir))
        self._avatars = {key: avatar}  # keep only the latest avatar
        base = read_pieces(Path(piece_dir))
        derived = build_derived(derived_defs, base, derived_dir)
        return avatar, {**base, **{k: v.info for k, v in derived.items()}}, derived

    def align(
        self,
        avatar_dir: Path,
        piece_dir: Path,
        spec_json_text: str,
        out_dir: Path | None = None,
        derived_dir: Path | None = None,
    ) -> AlignmentResult:
        """Solve all placements of the spec.

        Derived pieces are built first. With ``out_dir`` the engine writes ``alignment_result.json``, placed OBJs, a
        scene OBJ and the derived pieces; with only ``derived_dir`` it writes just the derived pieces (what a viewer
        needs).
        """
        spec = contracts.loads("placement_spec", spec_json_text)
        where = Path(derived_dir or out_dir or ".")
        avatar, pieces, derived = self._load(avatar_dir, piece_dir, spec.get("derived_pieces", []), where)
        if derived and (derived_dir is not None or out_dir is not None):
            write_derived(where, derived)
        tree = cKDTree(avatar.vertices)

        placed: list[PlacedPiece] = []
        for pl in spec["placements"]:
            if pl["piece"] not in pieces:
                raise UnknownPieceError(pl["piece"], list(pieces))
            if pl["point"] not in avatar.points:
                raise UnknownPointError(pl["point"], list(avatar.points))
            src = derived[pl["piece"]].spec["source"] if pl["piece"] in derived else ""
            placed.append(self._place(pl, pieces[pl["piece"]], avatar, tree, src))

        by_name = {p.name: p for p in placed}  # a piece placed twice: the last placement wins for stitches
        bundles = {k: v.bundle for k, v in pieces.items()}
        transforms: dict[str, Callable[[F64], F64]] = {n: p.transform for n, p in by_name.items()}
        stitches = [stitch_report(s, bundles, transforms) for s in spec.get("stitches", [])]
        rings = [self._ring(r, pieces, avatar, derived) for r in spec.get("rings", [])]
        warns = [f"{p.name}: {w}" for p in placed for w in p.warnings]
        doc = {
            "schema_version": 2,
            "units": "m",
            "placements": [self._doc(p, pieces[p.name]) for p in placed],
            "stitch_reports": [_round_tree(s) for s in stitches],
            "ring_reports": rings,
            "derived_pieces": [derived_doc(d) for d in derived.values()],
            "warnings": warns,
        }
        result = AlignmentResult(doc, placed, {}, warns)
        if out_dir is not None:
            from .writer import write_outputs

            result.paths = write_outputs(Path(out_dir), doc, placed, avatar)
        return result

    def arrange(
        self,
        avatar_dir: Path,
        piece_dir: Path,
        spec_json_text: str,
        out_dir: Path | None = None,
        derived_dir: Path | None = None,
        cache_dir: Path | None = None,
    ) -> AlignmentResult:
        """Place the pieces on offset body shells, spread out so that nothing overlays (see ``arrange.py``).

        Non-locked pieces are mapped row by row onto the shell of their body region; locked pieces keep the
        placement the
        spec gives them (their ``wrap``) and act as obstacles. The result document (``schema_version`` 3) has, per
        placement, an ``arrange`` block, the world outline and the named points, and an ``arrangement`` block with the
        per-ring numbers (true ease, standoff), the seam candidates and the violations. ``cache_dir`` holds the refined
        piece meshes between runs.
        """
        from .arrange import ArrangeOptions, Arranger
        from .arrange_report import build_report

        t0 = time.perf_counter()
        spec = contracts.loads("placement_spec", spec_json_text)
        where = Path(derived_dir or out_dir or ".")
        avatar, pieces, derived = self._load(avatar_dir, piece_dir, spec.get("derived_pieces", []), where)
        if derived and (derived_dir is not None or out_dir is not None):
            write_derived(where, derived)
        for pl in spec["placements"]:
            if pl["piece"] not in pieces:
                raise UnknownPieceError(pl["piece"], list(pieces))
            if pl["point"] not in avatar.points:
                raise UnknownPointError(pl["point"], list(avatar.points))
        tree = cKDTree(avatar.vertices)
        cache = self._refine.setdefault(str(cache_dir or ""), RefineCache(cache_dir))
        sources = {k: v.spec["source"] for k, v in derived.items()}

        def place_locked(pl: dict[str, Any], info: PieceInfo, source: str) -> PlacedPiece:
            return self._place(pl, info, avatar, tree, source)

        ar = Arranger(
            avatar, pieces, spec, ArrangeOptions.from_doc(spec.get("arrange_options")), place_locked, cache, sources
        )
        ar.prepare()
        ar.solve()
        placed = ar.build_placed()
        extras, arrangement, warns_extra = build_report(ar, placed, time.perf_counter() - t0)
        by_name = {p.name: p for p in placed}
        bundles = {k: v.bundle for k, v in pieces.items()}
        transforms: dict[str, Callable[[F64], F64]] = {n: p.transform for n, p in by_name.items()}
        stitches = [stitch_report(s, bundles, transforms) for s in spec.get("stitches", [])]
        rings = [self._ring(r, pieces, avatar, derived) for r in spec.get("rings", [])]
        warns = [*warns_extra, *ar.warnings, *[f"{p.name}: {w}" for p in placed for w in p.warnings]]
        docs = []
        for p, ex in zip(placed, extras, strict=True):
            d = self._doc(p, pieces[p.name])
            d.update(ex)
            docs.append(d)
        arrangement["timing_s"] = round(time.perf_counter() - t0, 3)
        doc = {
            "schema_version": 3,
            "units": "m",
            "placements": docs,
            "stitch_reports": [_round_tree(s) for s in stitches],
            "ring_reports": rings,
            "derived_pieces": [derived_doc(d) for d in derived.values()],
            "arrangement": arrangement,
            "warnings": warns,
        }
        result = AlignmentResult(doc, placed, {}, warns)
        if out_dir is not None:
            from .writer import write_outputs

            result.paths = write_outputs(Path(out_dir), doc, placed, avatar)
        return result

    def pair(
        self,
        avatar_dir: Path,
        piece_dir: Path,
        derived_defs: list[dict[str, Any]],
        base_placement: dict[str, Any],
        derived_id: str,
        mode: str,
    ) -> list[dict[str, Any]]:
        """Placements ``[base, clone]`` for a piece and its derived clone (see ``pairs.place_pair``).

        For ``same_anchor`` the reference of ``base_placement`` must lie on the mirror axis (1 mm).
        """
        avatar, pieces, derived = self._load(avatar_dir, piece_dir, derived_defs, Path("."))
        if derived_id not in derived:
            raise UnknownPieceError(derived_id, list(derived))
        d = derived[derived_id]
        dist: float | None = None
        if mode == "same_anchor":
            base = pieces[d.spec["source"]]
            ref = base.reference(str(base_placement.get("reference", "bbox_centre")))[:2]
            p0, u = d.mirrored.line_source
            v = ref - p0
            dist = float(abs(v[0] * u[1] - v[1] * u[0]))
        return place_pair(base_placement, derived_id, mode, point_names=avatar.points, ref_axis_distance_m=dist)

    # -- one placement -----------------------------------------------------------------------------------------
    def _sections(self, avatar: Avatar) -> TorsoSections:
        if "sections" not in avatar.cache:
            avatar.cache["sections"] = TorsoSections.build(avatar)
        sections: TorsoSections = avatar.cache["sections"]
        return sections

    def _place(
        self, pl: dict[str, Any], piece: PieceInfo, avatar: Avatar, tree: cKDTree, source: str = ""
    ) -> PlacedPiece:
        point: PointRecord = avatar.points[pl["point"]]
        ref = piece.reference(pl.get("reference", "bbox_centre"))
        rot = float(pl.get("rotation_deg", 0.0))
        mirror = bool(pl.get("mirror", False))
        flip = bool(pl.get("flip_facing", False))
        offset = pl.get("offset_m")
        eff = point.offset_m if offset is None else float(offset)
        # the offset always pushes the piece away from the body (along the outward normal), even when it is flipped
        target = point.position + (eff - point.offset_m) * point.normal
        m = rigid_matrix(target_frame(point.normal, point.up, flip), rot, ref, target)
        total = m @ mirror_matrix(float(ref[0])) if mirror else m

        v, f = piece.mesh()
        if mirror:
            v, f = mirror_mesh(v, f, float(ref[0]))
        warnings: list[str] = []
        mode = str(pl.get("wrap", "flat"))
        if mode == "body_slice" and point.bv not in TORSO_BVS:
            warnings.append(
                f"WRAP_FALLBACK: body_slice needs a torso point (Hip, Waist, Chest, Neck), {point.name} is on "
                f"{point.bv}; wrapped around the body part instead"
            )
            mode = "around"
        fit: dict[str, Any] = {"mode": mode}

        def to_mirrored(p2: F64) -> F64:
            p3 = np.column_stack([p2, np.zeros(len(p2))])
            if mirror:
                p3[:, 0] = 2.0 * ref[0] - p3[:, 0]
            return p3

        reference_world = target
        if mode == "around":
            vol = avatar.volumes[point.bv]
            geom = wrap_geometry(target, vol.start, vol.end, point.up)
            v, f = refine_mesh(v, f, WRAP_MAX_EDGE_M)
            world, info = wrap_vertices(v, ref, geom, rot, flip)
            if info.warning:
                warnings.append(info.warning)

            def transform(p2: F64) -> F64:
                return wrap_vertices(to_mirrored(p2), ref, geom, rot, flip)[0]

            fit.update(
                wrap_radius_m=geom.radius,
                circumference_m=2 * np.pi * geom.radius,
                wrap_angle_deg=info.angle_deg,
                width_over_circumference=info.width_m / (2 * np.pi * geom.radius),
            )
        elif mode == "body_slice":
            sections = self._sections(avatar)
            v, f = refine_mesh(v, f, WRAP_MAX_EDGE_M)
            c0, _ = sections.profile(float(target[1]))
            anchor_deg = float(np.degrees(np.arctan2(target[0] - c0[0], target[2] - c0[1])) % 360.0)
            th = np.radians(rot)

            def slice_map(p3: F64) -> tuple[F64, SliceInfo]:
                d = p3 - ref
                s_ = np.cos(th) * d[:, 0] - np.sin(th) * d[:, 1]
                h_ = np.sin(th) * d[:, 0] + np.cos(th) * d[:, 1]
                return map_rows(sections, float(target[1]) + h_, s_, anchor_deg, eff, d[:, 2], flip)

            world, sinfo = slice_map(v)

            def transform(p2: F64) -> F64:
                return slice_map(to_mirrored(p2))[0]

            reference_world = slice_map(ref[None, :])[0][0]
            loop = sections.perimeter(float(target[1]))
            if sinfo.overrun:
                warnings.append(
                    f"WRAP_OVERRUN: the piece reaches {sinfo.max_overrun_m * 100:.1f} cm past the far side of the "
                    f"{loop * 100:.1f} cm body loop at the anchor row; clamped, not scaled"
                )
            if sinfo.out_of_range:
                warnings.append(
                    f"BODY_SLICE_RANGE: {sinfo.out_of_range} vertices lie above or below the torso sections; the "
                    "nearest section is used there"
                )
            fit.update(body_loop_m=loop, **self._rows(piece, ref, rot, target, sections, avatar, world))
        else:
            world = apply_matrix(m, v)

            def transform(p2: F64) -> F64:
                return apply_matrix(m, to_mirrored(p2))

        d, _ = tree.query(world)
        extent = (world.max(axis=0) - world.min(axis=0)) * 100.0
        fit.update(
            nearest_avatar_vertex_min_m=float(d.min()),
            nearest_avatar_vertex_mean_m=float(d.mean()),
            placed_extent_cm=[float(x) for x in extent],
            offset_m=eff,
        )
        bb = piece.bundle["bbox_m"]
        width_m = float(bb["max"][0] - bb["min"][0])
        lateral = lateral_offset(world, f, reference_world)
        ext = extends(lateral, width_m)
        fit["lateral_offset_m"] = lateral
        warnings += side_warnings(piece.name, source, point.name, ext)
        return PlacedPiece(
            piece.name,
            point.name,
            total,
            mirror,
            world,
            f,
            mode != "flat",
            reference_world,
            warnings,
            fit,
            transform,
            piece.stem,
            ext,
            source,
        )

    @staticmethod
    def _rows(
        piece: PieceInfo,
        ref: F64,
        rot: float,
        target: F64,
        sections: TorsoSections,
        avatar: Avatar,
        world: F64,
    ) -> dict[str, Any]:
        """Garment width against the body loop along rows of the placed piece (only without user rotation).

        ``ease_quarter_cm`` compares with a quarter of the loop (a half front or half back covers about that),
        ``ease_half_cm`` with half of it (a whole front or back).
        """
        if abs(rot) > 1e-9:
            return {"rows": [], "levels": {}}
        outline = np.array(piece.bundle["outline_m"])
        height = float(piece.bundle["bbox_m"]["max"][1])

        def entry(y_world: float) -> dict[str, Any] | None:
            local = float(ref[1] + (y_world - target[1]))
            if not 0.0 <= local <= height:
                return None
            w = row_width(outline, local) * 100.0
            loop = sections.perimeter(y_world) * 100.0
            return {
                "y_m": round(y_world, 4),
                "piece_width_cm": round(w, 3),
                "body_loop_cm": round(loop, 3),
                "ease_quarter_cm": round(w - loop / 4.0, 3),
                "ease_half_cm": round(w - loop / 2.0, 3),
            }

        rows = [
            e for y in np.linspace(world[:, 1].min() + 0.005, world[:, 1].max() - 0.005, 8) if (e := entry(float(y)))
        ]
        levels: dict[str, Any] = {}
        for name in ("Chest", "Waist", "Hip"):
            p = avatar.points.get(f"{name}_Front")
            if p is not None and (e := entry(float(p.position[1]))):
                levels[name] = e
        return {"rows": rows, "levels": levels}

    @staticmethod
    def _doc(p: PlacedPiece, piece: PieceInfo) -> dict[str, Any]:
        bb = piece.bundle["bbox_m"]
        return {
            "piece": p.name,
            "point": p.point,
            "matrix": _r(p.matrix),
            "mirrored": p.mirrored,
            "extends": p.extends,
            "wrapped_vertices": [[round(float(x), 6) for x in row] for row in p.vertices] if p.wrapped else None,
            "wrapped_faces": p.faces.tolist() if p.wrapped else None,
            "reference_position_m": _r(p.reference_world, 6),
            "bbox_cm": {
                "width": round((bb["max"][0] - bb["min"][0]) * 100.0, 4),
                "height": round((bb["max"][1] - bb["min"][1]) * 100.0, 4),
            },
            "warnings": p.warnings,
            "fit": {k: _round_any(v) for k, v in p.fit.items()},
        }

    @staticmethod
    def _width_cm(piece: PieceInfo, level_point: str | None, warns: list[str]) -> float:
        bb = piece.bundle["bbox_m"]
        if level_point:
            for q in piece.bundle["named_points"]:
                if q["name"] == level_point:
                    return row_width(np.array(piece.bundle["outline_m"]), float(q["xy_m"][1])) * 100.0
            warns.append(f"{piece.name}: no named point {level_point!r}; bbox width used")
        return float((bb["max"][0] - bb["min"][0]) * 100.0)

    def _ring(
        self, r: dict[str, Any], pieces: dict[str, PieceInfo], avatar: Avatar, derived: dict[str, Derived]
    ) -> dict[str, Any]:
        ach = avatar.bundle["achieved_cm"]
        if r["avatar_measurement"] not in ach:
            raise UnknownMeasurementError(r["avatar_measurement"], sorted(ach))
        names = list(r["pieces"])
        if r.get("include_clones", True):  # a listed piece also counts its mirrored clones (2 x front + 2 x back)
            for n in list(names):
                names += [d.id for d in derived.values() if d.spec["source"] == n and d.id not in names]
        warns: list[str] = []
        widths = []
        for n in names:
            if n not in pieces:
                raise UnknownPieceError(n, list(pieces))
            widths.append(self._width_cm(pieces[n], r.get("level_point"), warns))
        total, body = float(sum(widths)), float(ach[r["avatar_measurement"]])
        return {
            "name": r["name"],
            "pieces": names,
            "widths_cm": [round(w, 4) for w in widths],
            "sum_width_cm": round(total, 4),
            "avatar_measurement": r["avatar_measurement"],
            "avatar_cm": round(body, 4),
            "ease_cm": round(total - body, 4),
            "ease_pct": round(100.0 * (total - body) / body, 4),
            "warnings": warns,
        }


def _round_any(x: Any) -> Any:
    if isinstance(x, float):
        return round(x, 6)
    if isinstance(x, list):
        return [_round_any(v) for v in x]
    return x


def _round_tree(x: Any) -> Any:
    if isinstance(x, dict):
        return {k: _round_tree(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_round_tree(v) for v in x]
    if isinstance(x, float):
        return round(x, 6)
    return x

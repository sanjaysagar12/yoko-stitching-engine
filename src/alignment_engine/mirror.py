"""Clone & mirror: a pure operation that makes a linked, mirrored copy of a piece. Metres.

The copy is a separate piece (as in CLO3D: two pieces joined by a seam, never one merged polygon). A reflection reverses
orientation, so the outline order and the triangle winding are fixed: the outline becomes ``[0, n-1, ..., 1]`` (vertex 0
stays first, arc positions become ``perimeter - s``), the mesh vertices follow the same permutation (so the outline is
still exactly the mesh boundary) and every triangle swaps two corners (normals stay +z). The geometry is then shifted so
that the bounding box starts at (0, 0); the shift is recorded as ``origin_shift_m``.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .errors import AxisError, BundleError, UnknownNamedPointError

F64 = NDArray[np.float64]
I64 = NDArray[np.int64]


@dataclass(frozen=True)
class MirroredPiece:
    """Result of ``mirror_clone`` / ``copy_clone``: the new piece bundle, its mesh (z = 0) and the mirror line in
    source coordinates (None for a copy)."""

    bundle: dict[str, Any]
    vertices: F64
    faces: I64
    line_source: tuple[F64, F64] | None  # (point on the line, unit direction) in the SOURCE piece's coordinates


def _named(bundle: dict[str, Any], name: str) -> F64:
    for q in bundle["named_points"]:
        if q["name"] == name:
            return np.array(q["xy_m"], dtype=np.float64)
    raise UnknownNamedPointError(bundle["name"], name, [q["name"] for q in bundle["named_points"]])


def axis_line(bundle: dict[str, Any], axis: dict[str, Any]) -> tuple[F64, F64]:
    """(point on the line, unit direction) of a mirror axis, in the piece's own coordinates."""
    kind = axis["type"]
    if kind == "bbox_center":
        x = 0.5 * (bundle["bbox_m"]["min"][0] + bundle["bbox_m"]["max"][0])
        return np.array([x, 0.0]), np.array([0.0, 1.0])
    if kind == "x":
        return np.array([float(axis["value_m"]), 0.0]), np.array([0.0, 1.0])
    if kind == "edge":
        a, b = _named(bundle, axis["from_point"]), _named(bundle, axis["to_point"])
        d = b - a
        n = float(np.linalg.norm(d))
        if n < 1e-9:
            raise AxisError(f"mirror axis through {axis['from_point']!r} and {axis['to_point']!r}: the points coincide")
        return a, d / n
    raise AxisError(f"unknown mirror axis type {kind!r}")


def reflect(p: F64, p0: F64, d: F64) -> F64:
    """Reflect points ``p`` (N,2) across the line through ``p0`` with unit direction ``d``."""
    v = p - p0
    return np.asarray(p0 + 2.0 * (v @ d)[:, None] * d - v, dtype=np.float64)


def _edges(points: list[dict[str, Any]], perimeter: float) -> list[dict[str, Any]]:
    on = sorted((q for q in points if q["on_outline"]), key=lambda q: q["outline_s_m"])
    out: list[dict[str, Any]] = []
    for i, q in enumerate(on):
        nxt = on[(i + 1) % len(on)]
        length = (
            (nxt["outline_s_m"] - q["outline_s_m"])
            if i + 1 < len(on)
            else on[0]["outline_s_m"] + perimeter - q["outline_s_m"]
        )
        out.append({"from": q["name"], "to": nxt["name"], "length_m": float(length)})
    return out


def mirror_clone(bundle: dict[str, Any], vertices: F64, faces: I64, axis: dict[str, Any], new_id: str) -> MirroredPiece:
    """Mirror a piece across ``axis``.

    ``bundle`` is a piece bundle, ``vertices`` (N,3, z = 0) and ``faces`` (M,3) its mesh. Everything is reflected:
    mesh, ``outline_m``, ``named_points``, ``edges``, ``bbox_m``, ``default_reference`` and the grainline. Edge lengths,
    area and perimeter are preserved exactly. The clone keeps the named point names, records ``mirror_of`` and its
    own ``axis``
    (so mirroring it again with that axis restores the source), ``origin_shift_m`` and, for a vertical axis, the
    ``pattern_origin_m`` of the source minus that shift (heights in the drafting frame are kept).
    """
    outline = np.asarray(bundle["outline_m"], dtype=np.float64)
    n = len(outline)
    if n != len(vertices) or np.abs(np.asarray(vertices)[:, :2] - outline).max() > 1e-5:
        raise BundleError(f"piece {bundle['name']!r}: outline_m is not the mesh boundary; cannot mirror")
    p0, d = axis_line(bundle, axis)
    refl = reflect(outline, p0, d)
    perm = np.r_[0, np.arange(n - 1, 0, -1)].astype(np.int64)
    inv = np.argsort(perm)
    refl = refl[perm]
    shift = -refl.min(axis=0)
    refl = refl + shift
    per = float(np.linalg.norm(np.roll(refl, -1, axis=0) - refl, axis=1).sum())

    pts: list[dict[str, Any]] = []
    for q in bundle["named_points"]:
        xy = reflect(np.array([q["xy_m"]], dtype=np.float64), p0, d)[0] + shift
        s = q["outline_s_m"]
        pts.append(
            {
                "name": q["name"],
                "label": q["label"],
                "xy_m": [float(xy[0]), float(xy[1])],
                "on_outline": q["on_outline"],
                "outline_s_m": None if s is None else float(np.mod(per - s, per)),
            }
        )
    kind = axis["type"]
    if kind == "edge":
        own_axis: dict[str, Any] = dict(axis)
    else:  # a vertical line stays where it is, then moves with the shift
        x0 = float(p0[0] + shift[0])
        own_axis = {"type": "x", "value_m": x0}
    grain = copy.deepcopy(bundle.get("grainline"))
    if grain is not None:
        a = reflect(np.array([grain["anchor_xy_m"]], dtype=np.float64), p0, d)[0] + shift
        alpha = float(np.degrees(np.arctan2(d[1], d[0])))
        grain["anchor_xy_m"] = [float(a[0]), float(a[1])]
        grain["rotation_deg"] = float(np.mod(2.0 * alpha - grain["rotation_deg"], 360.0))
    hi = refl.max(axis=0)
    src_origin = bundle.get("pattern_origin_m")
    origin = None  # a vertical mirror keeps heights: the clone sits on the same drafting base lines, x mirrored
    if src_origin is not None and abs(float(d[0])) < 1e-9:
        origin = [float(src_origin[0] - shift[0]), float(src_origin[1] - shift[1])]
    out = {
        "schema_version": 3,
        "name": new_id,
        "units": "m",
        "obj": f"{new_id}_m.obj",
        "bbox_m": {"min": [0.0, 0.0], "max": [float(hi[0]), float(hi[1])]},
        "outline_m": [[float(x), float(y)] for x, y in refl],
        "named_points": pts,
        "edges": _edges(pts, per),
        "fold_edge": copy.deepcopy(bundle.get("fold_edge")),
        "grainline": grain,
        "default_reference": {"type": "bbox_centre", "xy_m": [float(hi[0] / 2), float(hi[1] / 2)]},
        "mirror_of": bundle["name"],
        "axis": own_axis,
        "origin_shift_m": [float(shift[0]), float(shift[1])],
        "pattern_origin_m": origin,
    }
    v3 = np.column_stack([refl, np.zeros(n)])
    f = np.ascontiguousarray(inv[np.asarray(faces)][:, [0, 2, 1]])
    return MirroredPiece(out, v3, f, (p0, d))


def copy_clone(bundle: dict[str, Any], vertices: F64, faces: I64, new_id: str) -> MirroredPiece:
    """An exact copy of a piece (same orientation; CLO3D "Duplicate"): a separate piece, never a mirror half.

    Mesh, outline, named points, edges, fold edge, grainline, bbox, default reference and ``pattern_origin_m`` are the
    source's; the copy records ``copy_of``, has no ``mirror_of`` / ``axis`` and ``origin_shift_m = [0, 0]``.
    """
    out = copy.deepcopy(bundle)
    for k in ("mirror_of", "axis", "copy_of"):
        out.pop(k, None)
    out.update(
        {
            "schema_version": 3,
            "name": new_id,
            "obj": f"{new_id}_m.obj",
            "copy_of": bundle["name"],
            "origin_shift_m": [0.0, 0.0],
        }
    )
    out.setdefault("pattern_origin_m", None)
    v = np.array(vertices, dtype=np.float64)
    f = np.array(faces, dtype=np.int64)
    return MirroredPiece(out, v, f, None)

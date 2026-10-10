"""Turn a PieceDef plus the evaluated scope into an outline polygon (seam allowance is ignored)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .. import units
from ..errors import GeometryError, PatternError, UnsupportedToolError
from ..eval.scope import VariableScope
from ..geometry.bezier import Curve
from ..model.geometry_types import Point2, Polygon
from ..model.pattern import PieceDef
from .orientation import place_curve

_TRIM_TOL_CM = 0.05  # a point node trims a neighbouring curve if it lies within 0.5 mm of it


@dataclass
class Outline:
    """Piece outline: ``points_cm`` is (N,2) in cm, y down (Qt), implicitly closed."""

    name: str
    points_cm: Polygon
    notes: list[str] = field(default_factory=list)


@dataclass
class _Item:
    point: Point2 | None = None
    curve: Curve | None = None
    pts: np.ndarray | None = None
    us: np.ndarray | None = None
    reverse: bool = False


def assemble(piece: PieceDef, scope: VariableScope, flatten_tol_cm: float = 0.01) -> Outline:
    """Evaluate the main (sewing) path of ``piece`` into an outline in cm.

    Curve nodes are flattened with chord deviation <= ``flatten_tol_cm`` and trimmed between their
    neighbouring point nodes; excluded nodes are skipped.
    """
    tol = scope.from_cm(flatten_tol_cm)
    items: list[_Item] = []
    for n in piece.nodes:
        if n.excluded:
            continue
        if n.type == "NodePoint":
            items.append(_Item(point=scope.point(n.id_object)))
        elif n.type in ("NodeSpline", "NodeSplinePath"):
            c = scope.curve(n.id_object)
            pts, us = c.flatten(tol)
            items.append(_Item(curve=c, pts=pts, us=us, reverse=n.reverse))
        else:
            raise UnsupportedToolError(n.type, piece.name, "piece node type")
    if not items:
        raise PatternError(f"piece {piece.name!r} has no nodes")

    trim = scope.from_cm(_TRIM_TOL_CM)
    out: list[Point2] = []
    notes: list[str] = []
    for i, it in enumerate(items):
        if it.point is not None:
            out.append(it.point)
            continue
        assert it.curve is not None and it.pts is not None and it.us is not None
        prev, nxt = items[i - 1], items[(i + 1) % len(items)]
        try:
            seg, gap = place_curve(
                it.curve, it.pts, it.us, prev.point, nxt.point, out[-1] if out else None, it.reverse, trim
            )
        except ValueError as e:
            raise GeometryError(f"piece {piece.name!r}, node {i}: {e}") from None
        if units.to_cm(gap, scope.unit) > 1e-3:
            notes.append(f"node {i}: gap {units.to_cm(gap, scope.unit):.4f} cm to previous point")
        out += seg

    clean: list[Point2] = []
    for p in out:
        if not clean or math.dist(clean[-1], p) > 1e-9:
            clean.append(p)
    if len(clean) > 1 and math.dist(clean[0], clean[-1]) < 1e-9:
        clean.pop()
    return Outline(piece.name, np.array(clean, dtype=np.float64) * units.to_cm(1.0, scope.unit), notes)

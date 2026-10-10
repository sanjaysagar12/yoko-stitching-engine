"""Curve tools: cubicBezier, cubicBezierPath, simpleInteractive, pathInteractive."""

from __future__ import annotations

import numpy as np

from ...errors import PatternError
from ...eval.scope import VariableScope
from ...geometry.bezier import Curve
from ...geometry.primitives import move
from ...model.geometry_types import Point2
from ...model.pattern import ToolDef
from ..base import Tool, req
from ..registry import register


def _seg(*pts: Point2) -> np.ndarray:
    return np.array(pts, dtype=np.float64)


@register("cubicBezier")
class CubicBezier(Tool):
    """Bezier with four explicit control points."""

    def evaluate(self, d: ToolDef, ctx: VariableScope) -> Curve:
        return Curve([_seg(*(ctx.point(req(d, f"point{i}")) for i in (1, 2, 3, 4)))])


@register("cubicBezierPath")
class CubicBezierPath(Tool):
    """Chain of Beziers over explicit control points (3n+1 of them)."""

    def evaluate(self, d: ToolDef, ctx: VariableScope) -> Curve:
        pts = [ctx.point(p.get("pSpline", "")) for p in d.path_points]
        if len(pts) < 4:
            raise PatternError(f"cubicBezierPath id {d.id}: needs at least 4 control points")
        return Curve([_seg(*pts[i : i + 4]) for i in range(0, len(pts) - 3, 3)])


@register("simpleInteractive")
class SimpleInteractive(Tool):
    """Spline with P2 = P1 + length1@angle1 and P3 = P4 + length2@angle2."""

    def evaluate(self, d: ToolDef, ctx: VariableScope) -> Curve:
        p1, p4 = ctx.point(req(d, "point1")), ctx.point(req(d, "point4"))
        p2 = move(p1, ctx.eval(req(d, "length1")), ctx.eval(req(d, "angle1")))
        p3 = move(p4, ctx.eval(req(d, "length2")), ctx.eval(req(d, "angle2")))
        return Curve([_seg(p1, p2, p3, p4)])


@register("pathInteractive")
class PathInteractive(Tool):
    """Spline path: segment i uses angle2/length2 of point i and angle1/length1 of point i+1."""

    def evaluate(self, d: ToolDef, ctx: VariableScope) -> Curve:
        pp = d.path_points
        pts = [ctx.point(p.get("pSpline", "")) for p in pp]
        segs = []
        for i in range(len(pp) - 1):
            p2 = move(pts[i], ctx.eval(pp[i]["length2"]), ctx.eval(pp[i]["angle2"]))
            p3 = move(pts[i + 1], ctx.eval(pp[i + 1]["length1"]), ctx.eval(pp[i + 1]["angle1"]))
            segs.append(_seg(pts[i], p2, p3, pts[i + 1]))
        if not segs:
            raise PatternError(f"pathInteractive id {d.id}: needs at least 2 points")
        return Curve(segs)

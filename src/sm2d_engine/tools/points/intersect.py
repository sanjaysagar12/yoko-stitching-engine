"""Intersection-style point tools: height, lineIntersect, lineIntersectAxis, curveIntersectAxis, shoulder."""

from __future__ import annotations

import math

from ...errors import GeometryError
from ...eval.scope import VariableScope
from ...geometry.primitives import circle_line, line_line, move, qt_angle
from ...model.geometry_types import Point2
from ...model.pattern import ToolDef
from ..base import Tool, req
from ..registry import register


@register("height")
class Height(Tool):
    """Foot of the perpendicular from basePoint onto line p1Line-p2Line."""

    def evaluate(self, d: ToolDef, ctx: VariableScope) -> Point2:
        base, a, b = (ctx.point(req(d, k)) for k in ("basePoint", "p1Line", "p2Line"))
        dx, dy = b[0] - a[0], b[1] - a[1]
        den = dx * dx + dy * dy
        if den == 0:
            raise GeometryError(f"height {d.name!r} (id {d.id}): zero-length line")
        k = ((base[0] - a[0]) * dx + (base[1] - a[1]) * dy) / den
        return (a[0] + k * dx, a[1] + k * dy)


@register("lineIntersect")
class LineIntersect(Tool):
    """Intersection of the (infinite) lines p1Line1-p1Line2 and p2Line1-p2Line2."""

    def evaluate(self, d: ToolDef, ctx: VariableScope) -> Point2:
        a1, a2, b1, b2 = (ctx.point(req(d, k)) for k in ("p1Line1", "p1Line2", "p2Line1", "p2Line2"))
        p = line_line(a1, a2, b1, b2)
        if p is None:
            raise GeometryError(f"lineIntersect {d.name!r} (id {d.id}): lines are parallel")
        return p


@register("lineIntersectAxis")
class LineIntersectAxis(Tool):
    """Axis (basePoint, angle) intersected with the infinite line p1Line-p2Line."""

    def evaluate(self, d: ToolDef, ctx: VariableScope) -> Point2:
        base, a, b = (ctx.point(req(d, k)) for k in ("basePoint", "p1Line", "p2Line"))
        far = move(base, 1.0, ctx.eval(req(d, "angle")))
        p = line_line(base, far, a, b)
        if p is None:
            raise GeometryError(f"lineIntersectAxis {d.name!r} (id {d.id}): axis parallel to line")
        return p


@register("curveIntersectAxis")
class CurveIntersectAxis(Tool):
    """Nearest crossing of ray (basePoint, angle) with a curve; if none, of the opposite ray."""

    def evaluate(self, d: ToolDef, ctx: VariableScope) -> Point2:
        base = ctx.point(req(d, "basePoint"))
        ang = ctx.eval(req(d, "angle"))
        c = ctx.curve(req(d, "curve"))
        hits = c.ray_hits(base, ang) or c.ray_hits(base, ang + 180)
        if not hits:
            raise GeometryError(f"curveIntersectAxis {d.name!r} (id {d.id}): no intersection")
        return min(hits, key=lambda h: math.dist(h, base))


@register("shoulder")
class Shoulder(Tool):
    """Circle(pShoulder, length) meets ray p1Line->p2Line beyond p2Line; else p2Line."""

    def evaluate(self, d: ToolDef, ctx: VariableScope) -> Point2:
        p1, p2, ps = (ctx.point(req(d, k)) for k in ("p1Line", "p2Line", "pShoulder"))
        length = ctx.eval(req(d, "length"))
        if length <= 0 or p1 == p2:
            return p2
        base_len = math.dist(p1, p2)
        base_ang = round(qt_angle(p1, p2))
        ts = circle_line(p1, p2, ps, length)
        if ts is None:
            return p2
        for t in ts:
            q = (p1[0] + (p2[0] - p1[0]) / base_len * t, p1[1] + (p2[1] - p1[1]) / base_len * t)
            if t > base_len and round(qt_angle(p1, q)) == base_ang:
                return q
        return p2

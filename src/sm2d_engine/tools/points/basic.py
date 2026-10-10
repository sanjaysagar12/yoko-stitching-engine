"""Point tools built from a base point plus formulas: single, endLine, alongLine, normal, intersectXY."""

from __future__ import annotations

import math

from ...eval.scope import VariableScope
from ...geometry.primitives import move, qt_angle
from ...model.geometry_types import Point2
from ...model.pattern import ToolDef
from ..base import Tool, req
from ..registry import register


@register("single")
class Single(Tool):
    """Absolute point; x/y in pattern unit. ``mx``/``my`` are label offsets and ignored."""

    def evaluate(self, d: ToolDef, ctx: VariableScope) -> Point2:
        return (float(req(d, "x")), float(req(d, "y")))


@register("endLine")
class EndLine(Tool):
    """``basePoint + length @ angle`` (Qt angle)."""

    def evaluate(self, d: ToolDef, ctx: VariableScope) -> Point2:
        base = ctx.point(req(d, "basePoint"))
        return move(base, ctx.eval(req(d, "length")), ctx.eval(req(d, "angle")))


def _two(d: ToolDef, ctx: VariableScope) -> tuple[Point2, Point2, float]:
    a, b = ctx.point(req(d, "firstPoint")), ctx.point(req(d, "secondPoint"))
    return a, b, math.dist(a, b)


@register("alongLine")
class AlongLine(Tool):
    """Point at ``length`` from firstPoint toward secondPoint; ``CurrentLength`` = line length."""

    def evaluate(self, d: ToolDef, ctx: VariableScope) -> Point2:
        a, b, n = _two(d, ctx)
        length = ctx.eval(req(d, "length"), n)
        if n == 0:  # Qt: QLineF::setLength is a no-op on a null line, so the result is the first point
            return a
        return (a[0] + (b[0] - a[0]) / n * length, a[1] + (b[1] - a[1]) / n * length)


@register("normal")
class Normal(Tool):
    """Perpendicular to first->second (Qt ``normalVector() = (dy, -dx)``) plus extra angle."""

    def evaluate(self, d: ToolDef, ctx: VariableScope) -> Point2:
        a, b, n = _two(d, ctx)
        length = ctx.eval(req(d, "length"), n)
        extra = ctx.eval(d.attrs.get("angle") or "0", n)
        if n == 0:  # null base line: Qt leaves the point at the first point
            return a
        ang = qt_angle((0.0, 0.0), (b[1] - a[1], -(b[0] - a[0]))) + extra
        return move(a, length, ang)


@register("intersectXY")
class IntersectXY(Tool):
    """``(first.x, second.y)``."""

    def evaluate(self, d: ToolDef, ctx: VariableScope) -> Point2:
        a, b, _ = _two(d, ctx)
        return (a[0], b[1])

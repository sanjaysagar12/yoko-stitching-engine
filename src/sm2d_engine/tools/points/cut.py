"""Point-on-curve tools: cutSpline, cutSplinePath."""

from __future__ import annotations

from ...eval.scope import VariableScope
from ...model.geometry_types import Point2
from ...model.pattern import ToolDef
from ..base import Tool, req
from ..registry import register

_CLAMP_CM = 0.1  # Seamly keeps the cut at least 1 mm from either end


def _cut(d: ToolDef, ctx: VariableScope, curve_attr: str) -> Point2:
    c = ctx.curve(req(d, curve_attr))
    full = c.length()
    eps = ctx.from_cm(_CLAMP_CM)
    length = min(max(ctx.eval(req(d, "length")), eps), full - eps)
    return c.point_at_length(length)


@register("cutSpline")
class CutSpline(Tool):
    """Point at arc length ``length`` along a spline, clamped to [1 mm, full-1 mm]."""

    def evaluate(self, d: ToolDef, ctx: VariableScope) -> Point2:
        return _cut(d, ctx, "spline")


@register("cutSplinePath")
class CutSplinePath(Tool):
    """Same as cutSpline over a spline path."""

    def evaluate(self, d: ToolDef, ctx: VariableScope) -> Point2:
        return _cut(d, ctx, "splinePath")

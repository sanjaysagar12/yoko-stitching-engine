"""Operation tool: rotation. Copies points and curves rotated about ``center`` by ``angle`` degrees (Qt sense)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ...errors import UnsupportedToolError
from ...eval.scope import VariableScope
from ...geometry.bezier import Curve
from ...geometry.primitives import move, qt_angle
from ...model.geometry_types import Point2
from ...model.pattern import ToolDef
from ..base import Tool, req
from ..registry import register


def rotate(center: Point2, p: Point2, angle: float) -> Point2:
    """Rotate ``p`` about ``center`` (QLineF(center,p).setAngle(angle0 + angle))."""
    r = ((p[0] - center[0]) ** 2 + (p[1] - center[1]) ** 2) ** 0.5
    if r == 0:
        return p
    return move(center, r, qt_angle(center, p) + angle)


@register("rotation")
class Rotation(Tool):
    """Destination objects are copies of the source objects (paired by position), rotated."""

    def evaluate(self, d: ToolDef, ctx: VariableScope) -> Any:
        raise UnsupportedToolError("rotation", detail="operations are evaluated per destination item")

    def apply(self, d: ToolDef, src_id: str, ctx: VariableScope) -> Any:
        center = ctx.point(req(d, "center"))
        angle = ctx.eval(req(d, "angle"))
        kind = ctx.kind(src_id)
        if kind == "point":
            return rotate(center, ctx.point(src_id), angle)
        if kind == "curve":
            segs = []
            for s in ctx.curve(src_id).segments:
                segs.append(np.array([rotate(center, (float(q[0]), float(q[1])), angle) for q in s]))
            return Curve(segs)
        raise UnsupportedToolError("rotation", detail=f"cannot rotate object {src_id} of kind {kind}")

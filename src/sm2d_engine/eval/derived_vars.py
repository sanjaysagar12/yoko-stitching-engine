"""Derived formula variables: Line_A_B, AngleLine_A_B, Spl_/SplPath_ lengths and sub-curve lengths.

Lengths are in the pattern unit, angles in degrees (Qt sense).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from ..geometry.primitives import qt_angle

if TYPE_CHECKING:
    from .scope import VariableScope

_POINT_CURVES = ("cubicBezier", "simpleInteractive")
_PATH_CURVES = ("cubicBezierPath", "pathInteractive")
_CUT_TYPES = ("curveIntersectAxis", "cutSpline", "cutSplinePath")


class DerivedVars:
    """Lazy resolver of names that are not measurements or increments."""

    def __init__(self, scope: VariableScope) -> None:
        self.s = scope
        self._names: dict[str, str] | None = None
        self._curves: dict[str, str] | None = None
        self._subs: dict[str, tuple[str, str, bool]] | None = None

    def _point_names(self) -> dict[str, str]:
        if self._names is None:
            self._names = {t.name: i for i, t in self.s.model.tools.items() if t.kind == "point" and t.name}
        return self._names

    def _ends(self, cid: str) -> tuple[str, str, str] | None:
        """(prefix, start point name, end point name) of a spline object."""
        tools = self.s.model.tools
        c = tools.get(cid)
        if c is None or c.kind != "curve":
            return None
        if c.type in _POINT_CURVES:
            a, b, pre = c.attrs.get("point1", ""), c.attrs.get("point4", ""), "Spl_"
        elif c.type in _PATH_CURVES and c.path_points:
            a, b, pre = c.path_points[0].get("pSpline", ""), c.path_points[-1].get("pSpline", ""), "SplPath_"
        else:
            return None
        if a not in tools or b not in tools:
            return None
        return pre, tools[a].name, tools[b].name

    def _curve_index(self) -> dict[str, str]:
        if self._curves is None:
            self._curves = {}
            for cid in self.s.model.tools:
                e = self._ends(cid)
                if e:
                    self._curves[f"{e[0]}{e[1]}_{e[2]}"] = cid
        return self._curves

    def _sub_index(self) -> dict[str, tuple[str, str, bool]]:
        """Seamly names the halves of a curve split by a point: <start>_<pt> (True) and <pt>_<end> (False)."""
        if self._subs is None:
            self._subs = {}
            m = self.s.model
            for pid, t in m.tools.items():
                if t.kind != "point" or t.type not in _CUT_TYPES:
                    continue
                cid = m.resolve(t.attrs.get("curve") or t.attrs.get("spline") or t.attrs.get("splinePath") or "")
                e = self._ends(cid)
                if e is None:
                    continue
                pre, start, end = e
                self._subs[f"{pre}{start}_{t.name}"] = (cid, pid, True)
                self._subs[f"{pre}{t.name}_{end}"] = (cid, pid, False)
        return self._subs

    def resolve(self, name: str) -> float | None:
        """Value of derived variable ``name`` or None if ``name`` is not one."""
        s = self.s
        cid = self._curve_index().get(name)
        if cid is not None:
            return s.curve(cid).length()
        sub = self._sub_index().get(name)
        if sub is not None:
            c = s.curve(sub[0])
            u, _ = c.project(s.point(sub[1]))
            at = c.length_at_u(u)
            return at if sub[2] else c.length() - at
        for pre, fn in (("AngleLine_", qt_angle), ("Line_", math.dist)):
            if name.startswith(pre):
                parts = name[len(pre) :].split("_")
                names = self._point_names()
                for i in range(1, len(parts)):
                    a, b = "_".join(parts[:i]), "_".join(parts[i:])
                    if a in names and b in names:
                        return float(fn(s.point(names[a]), s.point(names[b])))
        return None

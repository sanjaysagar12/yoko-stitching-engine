"""Variable scope: measurements, increments, derived variables, lazily evaluated points and curves.

All values handled here are in the **pattern unit** (``model.unit``); measurements are converted from cm on
lookup (units.py). Every lookup is lazy and guarded against cycles.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Protocol

from .. import units
from ..errors import CyclicDependencyError, FormulaError, MissingMeasurementError, PatternError
from ..geometry.bezier import Curve
from ..model.geometry_types import Point2
from ..model.measurements import Measurements
from ..model.pattern import PatternModel, ToolDef
from . import expression
from .derived_vars import DerivedVars


class ToolImpl(Protocol):
    """What the scope needs from a registered tool."""

    def evaluate(self, d: ToolDef, ctx: VariableScope) -> Any: ...

    def apply(self, d: ToolDef, src_id: str, ctx: VariableScope) -> Any: ...


class ToolLookup(Protocol):
    """Registry protocol: tool type -> implementation (raises UnsupportedToolError)."""

    def get(self, tool_type: str) -> ToolImpl: ...


class VariableScope:
    """Evaluation context for one pattern + one measurement set (pattern unit)."""

    def __init__(self, model: PatternModel, measurements: Measurements, registry: ToolLookup) -> None:
        self.model = model
        self.measurements = measurements
        self.registry = registry
        self.unit = model.unit
        self._stack: list[str] = []
        self._points: dict[str, Point2] = {}
        self._curves: dict[str, Curve] = {}
        self._incr: dict[str, float] = {}
        self._derived = DerivedVars(self)

    # -- units -------------------------------------------------------------
    def from_cm(self, value_cm: float) -> float:
        """Convert cm to the pattern unit."""
        return units.from_cm(value_cm, self.unit)

    # -- cycle guard -------------------------------------------------------
    @contextmanager
    def guard(self, key: str) -> Iterator[None]:
        """Mark ``key`` as being evaluated; re-entry raises CyclicDependencyError."""
        if key in self._stack:
            raise CyclicDependencyError([*self._stack[self._stack.index(key) :], key])
        self._stack.append(key)
        try:
            yield
        finally:
            self._stack.pop()

    # -- formulas ----------------------------------------------------------
    def eval(self, formula: str, current_length: float | None = None) -> float:
        """Evaluate a formula (pattern unit). ``current_length`` feeds ``CurrentLength``."""

        def look(name: str) -> float:
            return self.value(name, current_length)

        return expression.evaluate(formula, look)

    def value(self, name: str, current_length: float | None = None) -> float:
        """Resolve a variable name: CurrentLength, measurement, increment or derived variable."""
        if name == "CurrentLength":
            if current_length is None:
                raise FormulaError(name, "CurrentLength is not available in this formula")
            return current_length
        if name in self.measurements:
            return self.from_cm(self.measurements.get(name))
        inc = self.model.increments.get(name)
        if inc is not None:
            if name not in self._incr:
                with self.guard(f"increment {name}"):
                    self._incr[name] = self.eval(inc.formula)
            return self._incr[name]
        v = self._derived.resolve(name)
        if v is None:
            raise MissingMeasurementError(name)
        return v

    # -- objects -----------------------------------------------------------
    def kind(self, id_: str) -> str:
        """'point' | 'curve' | other, following modeling copies and operation destinations."""
        id_ = self.model.resolve(id_)
        if id_ in self.model.op_dest:
            return self.kind(self.model.op_dest[id_][1])
        t = self.model.tools.get(id_)
        return t.kind if t else "unknown"

    def _run(self, id_: str, want: str) -> Any:
        with self.guard(f"{want} {id_}"):
            if id_ in self.model.op_dest:
                op_id, src = self.model.op_dest[id_]
                op = self.model.tools[op_id]
                return self.registry.get(op.type).apply(op, src, self)
            d = self.model.tools.get(id_)
            if d is None:
                raise PatternError(f"reference to unknown object id {id_}")
            if d.kind != want:
                raise PatternError(f"object {id_} ({d.type}) is a {d.kind}, expected {want}")
            return self.registry.get(d.type).evaluate(d, self)

    def point(self, id_: str) -> Point2:
        """Point by id (modeling ids accepted), evaluated lazily and cached (pattern unit)."""
        key = self.model.resolve(id_)
        if key not in self._points:
            self._points[key] = self._run(key, "point")
        return self._points[key]

    def curve(self, id_: str) -> Curve:
        """Curve by id (modeling ids accepted), evaluated lazily and cached (pattern unit)."""
        key = self.model.resolve(id_)
        if key not in self._curves:
            self._curves[key] = self._run(key, "curve")
        return self._curves[key]

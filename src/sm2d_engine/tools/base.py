"""Tool strategy base class. A tool turns a ``ToolDef`` into a point or curve, pulling dependencies lazily."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..errors import PatternError, UnsupportedToolError
from ..eval.scope import VariableScope
from ..model.pattern import ToolDef


def req(d: ToolDef, key: str) -> str:
    """Required attribute of a tool definition."""
    v = d.attrs.get(key)
    if v is None:
        raise PatternError(f"tool {d.type!r} (id {d.id}): missing attribute {key!r}")
    return v


class Tool(ABC):
    """Strategy for one ``type`` attribute value. Units: pattern unit (see VariableScope)."""

    type: str = ""

    @abstractmethod
    def evaluate(self, d: ToolDef, ctx: VariableScope) -> Any:
        """Return a Point2 (point tools) or Curve (curve tools)."""

    def apply(self, d: ToolDef, src_id: str, ctx: VariableScope) -> Any:
        """Operation tools only: produce the copy of source object ``src_id``."""
        raise UnsupportedToolError(d.type, detail="not an operation")

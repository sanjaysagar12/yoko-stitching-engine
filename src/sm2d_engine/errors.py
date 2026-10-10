"""Error hierarchy. Every error names the offending id / name."""

from __future__ import annotations


class EngineError(Exception):
    """Base class for all engine errors."""


class UnitError(EngineError):
    """Unknown length unit."""

    def __init__(self, unit: str) -> None:
        super().__init__(f"unknown unit {unit!r}")
        self.unit = unit


class UnsupportedToolError(EngineError):
    """A tool / node type the engine does not implement. ``piece`` is filled in by the engine."""

    def __init__(self, tool_type: str, piece: str | None = None, detail: str = "") -> None:
        self.tool_type = tool_type
        self.piece = piece
        self.detail = detail
        super().__init__(self._msg())

    def _msg(self) -> str:
        where = f" (piece {self.piece!r})" if self.piece else ""
        extra = f": {self.detail}" if self.detail else ""
        return f"unsupported tool {self.tool_type!r}{where}{extra}"

    def with_piece(self, piece: str) -> UnsupportedToolError:
        """Return a copy that names ``piece``."""
        return UnsupportedToolError(self.tool_type, piece, self.detail)


class MissingMeasurementError(EngineError):
    """A formula references a variable that is not a measurement, increment or derived variable."""

    def __init__(self, name: str) -> None:
        super().__init__(f"missing measurement or variable {name!r}")
        self.name = name


class FormulaError(EngineError):
    """A formula could not be parsed or evaluated."""

    def __init__(self, formula: str, reason: str) -> None:
        super().__init__(f"formula {formula!r}: {reason}")
        self.formula = formula
        self.reason = reason


class CyclicDependencyError(EngineError):
    """Objects depend on each other in a loop."""

    def __init__(self, chain: list[str]) -> None:
        super().__init__("cyclic dependency: " + " -> ".join(chain))
        self.chain = chain


class GeometryError(EngineError):
    """Degenerate geometry (parallel lines, no intersection, ...)."""


class PatternError(EngineError):
    """Malformed or inconsistent pattern / measurement file."""


class UnsupportedFormatError(EngineError):
    """Output format without a writer."""

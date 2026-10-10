"""Result types and the JSON report. Lengths in cm, areas in cm^2."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class PieceReport:
    """Per-piece outcome. ``status`` is "ok" or "failed" (then only ``name``/``reason`` are meaningful)."""

    name: str
    status: str = "ok"
    reason: str | None = None
    width_cm: float | None = None
    height_cm: float | None = None
    area_cm2: float | None = None
    perimeter_cm: float | None = None
    mesh_area_cm2: float | None = None
    area_diff_pct: float | None = None
    vertices: int | None = None
    triangles: int | None = None
    self_intersections: int | None = None
    gaps: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    quantity: str | None = None
    on_fold: str | None = None
    file: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready dict (floats rounded to 6 decimals)."""
        d = asdict(self)
        return {k: round(v, 6) if isinstance(v, float) else v for k, v in d.items()}


@dataclass
class EngineResult:
    """What ``Engine.run`` returns: per-piece reports, global warnings, written paths and report header."""

    pieces: list[PieceReport]
    warnings: list[str] = field(default_factory=list)
    output_paths: list[str] = field(default_factory=list)
    header: dict[str, Any] = field(default_factory=dict)

    @property
    def failed(self) -> list[PieceReport]:
        """Pieces with status "failed"."""
        return [p for p in self.pieces if p.status != "ok"]

    def to_dict(self) -> dict[str, Any]:
        """The report.json content."""
        return {**self.header, "warnings": self.warnings, "pieces": [p.to_dict() for p in self.pieces]}

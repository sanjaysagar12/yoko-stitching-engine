"""Seamly2D pattern + measurements -> real-scale per-piece meshes (1 OBJ unit = 1 m)."""

__version__ = "0.2.0"

from .contracts import ContractError  # noqa: E402
from .engine import Engine, EngineOptions, EngineRequest  # noqa: E402
from .errors import EngineError, MissingMeasurementError, UnsupportedToolError  # noqa: E402
from .reporting import EngineResult, PieceReport  # noqa: E402

__all__ = [
    "ContractError",
    "Engine",
    "EngineError",
    "EngineOptions",
    "EngineRequest",
    "EngineResult",
    "MissingMeasurementError",
    "PieceReport",
    "UnsupportedToolError",
    "__version__",
]

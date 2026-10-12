"""Alignment engine: places flat pattern pieces onto avatar arrangement points."""

__version__ = "0.2.0"

from .contracts import ContractError  # noqa: E402
from .engine import AlignmentEngine, AlignmentResult  # noqa: E402
from .errors import (  # noqa: E402
    AlignmentError,
    NoMirroredPointError,
    ReferenceNotOnAxisError,
    UnknownPieceError,
    UnknownPointError,
)

__all__ = [
    "AlignmentEngine",
    "AlignmentError",
    "AlignmentResult",
    "ContractError",
    "NoMirroredPointError",
    "ReferenceNotOnAxisError",
    "UnknownPieceError",
    "UnknownPointError",
    "__version__",
]

"""Avatar engine: SMPL-family avatar from body measurements, bounding volumes and arrangement points."""

__version__ = "0.2.1"

from .contracts import ContractError  # noqa: E402
from .engine import AvatarEngine, AvatarResult  # noqa: E402
from .errors import (  # noqa: E402
    AvatarEngineError,
    FitDidNotConvergeError,
    ModelFilesNotFoundError,
)

__all__ = [
    "AvatarEngine",
    "AvatarEngineError",
    "AvatarResult",
    "ContractError",
    "FitDidNotConvergeError",
    "ModelFilesNotFoundError",
    "__version__",
]

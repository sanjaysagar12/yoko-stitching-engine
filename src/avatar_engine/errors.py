"""Specific errors of the avatar engine. Every message names the offending file / measurement / point."""

from __future__ import annotations

from pathlib import Path


class AvatarEngineError(Exception):
    """Base class for all avatar engine errors."""


class ModelFilesNotFoundError(AvatarEngineError):
    """No usable SMPL / SMPL-X model file for the requested gender and type."""

    def __init__(self, model_dir: Path, gender: str, model_type: str, searched: list[str]) -> None:
        self.model_dir = model_dir
        self.gender = gender
        self.model_type = model_type
        self.searched = searched
        super().__init__(
            f"no {model_type} model file for gender {gender!r} under {model_dir} "
            f"(searched: {', '.join(searched) if searched else 'nothing'}). "
            "Place SMPL_*.npz|pkl or SMPLX_*.npz in that folder (nothing is downloaded)."
        )


class ModelFormatError(AvatarEngineError):
    """A model file exists but does not contain the expected arrays."""


class MeasurementDefinitionError(AvatarEngineError):
    """A measurement cannot be defined on the template (no loop found, missing landmark, unknown name)."""

    def __init__(self, name: str, reason: str) -> None:
        self.name = name
        self.reason = reason
        super().__init__(f"measurement {name!r}: {reason}")


class FitDidNotConvergeError(AvatarEngineError):
    """The shape fit could not reach the required accuracy inside the beta bounds."""

    def __init__(self, name: str, achieved_cm: float, target_cm: float, tol_cm: float) -> None:
        self.name = name
        super().__init__(
            f"shape fit: {name!r} achieved {achieved_cm:.3f} cm, target {target_cm:.3f} cm "
            f"(tolerance {tol_cm:.3f} cm); not reachable inside the beta bounds"
        )


class UnknownArrangementPointError(AvatarEngineError):
    """A point name that is not in the arrangement point set."""

    def __init__(self, name: str, known: list[str]) -> None:
        self.name = name
        super().__init__(f"unknown arrangement point {name!r}; known: {', '.join(sorted(known)[:12])} ...")


class PointConfigError(AvatarEngineError):
    """The default_points.yaml configuration is malformed or references an unknown bounding volume."""


class RingCorrectionWarning(UserWarning):
    """A circumference correction needed more than +-10 % radial scale."""

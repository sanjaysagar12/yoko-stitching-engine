"""Engine errors -> structured JSON ``{"error": {"code", "message", "piece"?, "tool"?}}``, never a traceback."""

from __future__ import annotations

import logging
import re
from typing import Any

import alignment_engine
import avatar_engine
import sm2d_engine

from .jobs import JobError

log = logging.getLogger(__name__)
CONTRACT_ERRORS = (sm2d_engine.ContractError, avatar_engine.ContractError, alignment_engine.ContractError)
ENGINE_ERRORS: tuple[type[Exception], ...] = (
    sm2d_engine.EngineError,
    avatar_engine.AvatarEngineError,
    alignment_engine.AlignmentError,
    *CONTRACT_ERRORS,
)


def error_body(code: str, message: str, **extra: Any) -> dict[str, Any]:
    """The error envelope."""
    return {"error": {"code": code, "message": message, **{k: v for k, v in extra.items() if v is not None}}}


def _code(exc: Exception) -> str:
    """``UnknownNamedPointError`` -> ``unknown_named_point`` (the base ``AlignmentError`` stays ``alignment_error``)."""
    name = type(exc).__name__
    if name == "AlignmentError":
        return "alignment_error"
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name.removesuffix("Error")).lower()


def piece_failure(reason: str) -> dict[str, Any]:
    """``{code, tool?}`` for a failed piece reason string produced by the sm2d engine."""
    if reason.startswith("unsupported tool"):
        tool = reason.split("'")[1] if "'" in reason else None
        return {"code": "unsupported_tool", "tool": tool}
    if reason.startswith("MissingMeasurementError"):
        return {"code": "missing_measurement"}
    return {"code": "piece_failed"}


def to_error(exc: BaseException) -> tuple[int, dict[str, Any]]:
    """HTTP status and body for any exception raised by the engines or the backend."""
    if isinstance(exc, JobError):
        return (404 if exc.code == "job_not_found" else 422), error_body(exc.code, str(exc))
    if isinstance(exc, sm2d_engine.UnsupportedToolError):
        return 422, error_body("unsupported_tool", str(exc), piece=exc.piece, tool=exc.tool_type)
    if isinstance(exc, sm2d_engine.MissingMeasurementError):
        return 422, error_body("missing_measurement", str(exc))
    if isinstance(exc, sm2d_engine.EngineError):
        return 422, error_body("pattern_error", str(exc))
    if isinstance(exc, avatar_engine.ModelFilesNotFoundError):
        return 500, error_body("model_files_not_found", str(exc))
    if isinstance(exc, avatar_engine.FitDidNotConvergeError):
        return 422, error_body("fit_did_not_converge", str(exc))
    if isinstance(exc, avatar_engine.AvatarEngineError):
        return 422, error_body("avatar_error", str(exc))
    if isinstance(exc, alignment_engine.UnknownPieceError):
        return 422, error_body("unknown_piece", str(exc), piece=exc.name)
    if isinstance(exc, alignment_engine.UnknownPointError):
        return 422, error_body("unknown_point", str(exc))
    if isinstance(exc, alignment_engine.AlignmentError):
        return 422, error_body(_code(exc), str(exc), piece=getattr(exc, "piece", None) or None)
    if isinstance(exc, CONTRACT_ERRORS):
        return 422, error_body("invalid_document", str(exc))
    log.exception("unexpected error", exc_info=exc)
    return 500, error_body("internal_error", "unexpected server error; see the server log")

"""Side awareness: which way a placed piece extends from its reference, and warnings for suspicious placements."""

from __future__ import annotations

import re

import numpy as np
from numpy.typing import NDArray

F64 = NDArray[np.float64]
I64 = NDArray[np.int64]


def lateral_offset(world: F64, faces: I64, reference: F64) -> float:
    """Signed x offset (m, +x = the wearer's left) of the placed piece centroid (area weighted) from its reference."""
    t = world[faces]
    area = 0.5 * np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1)
    if area.sum() <= 0:
        return 0.0
    centroid = (t.mean(axis=1) * area[:, None]).sum(axis=0) / area.sum()
    return float(centroid[0] - reference[0])


def extends(lateral_m: float, width_m: float) -> str:
    """``left`` / ``right`` (wearer's side) or ``centre`` if within 1 cm or 5 % of the piece width."""
    if abs(lateral_m) < max(0.01, 0.05 * width_m):
        return "centre"
    return "left" if lateral_m > 0 else "right"


def side_warnings(piece_name: str, source_name: str, point: str, ext: str) -> list[str]:
    """``SIDE_UNLIKELY`` / ``SIDE_MISMATCH`` messages. ``source_name`` is the base piece name for a clone."""
    out: list[str] = []
    label = source_name or piece_name
    is_front = bool(re.search("front", label, re.I)) and not re.search("back", label, re.I)
    is_back = bool(re.search("back", label, re.I)) and not re.search("front", label, re.I)
    if is_front and "Back" in point:
        out.append(f"SIDE_UNLIKELY: {piece_name!r} looks like a front piece but is placed on {point}")
    if is_back and "Front" in point:
        out.append(f"SIDE_UNLIKELY: {piece_name!r} looks like a back piece but is placed on {point}")
    if (point.endswith("_R") and ext == "left") or (point.endswith("_L") and ext == "right"):
        side = "right" if point.endswith("_R") else "left"
        out.append(
            f"SIDE_MISMATCH: {piece_name!r} is placed on the {side} point {point} but extends toward the wearer's {ext}"
        )
    return out

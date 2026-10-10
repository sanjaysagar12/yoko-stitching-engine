"""Placement: y-up, origin at bounding-box bottom-left, optional left-to-right layout. Units: cm."""

from __future__ import annotations

import numpy as np

from ..model.geometry_types import Polygon


def signed_area(p: Polygon) -> float:
    """Shoelace area (positive for counter-clockwise in a y-up frame), cm^2."""
    x, y = p[:, 0], p[:, 1]
    return float(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def to_origin(points_cm_ydown: Polygon) -> tuple[Polygon, float, float]:
    """Flip y (Qt y-down -> y-up), move the bbox bottom-left to (0,0), make the winding counter-clockwise.

    Returns (points in cm, width_cm, height_cm).
    """
    x = points_cm_ydown[:, 0] - points_cm_ydown[:, 0].min()
    y = points_cm_ydown[:, 1].max() - points_cm_ydown[:, 1]
    p = np.column_stack([x, y])
    if len(p) >= 3 and signed_area(p) < 0:
        p = p[::-1].copy()
    return p, float(x.max()), float(y.max())


def layout_offsets(widths_cm: list[float], gap_cm: float) -> list[float]:
    """x offsets (cm) placing pieces left to right, ``gap_cm`` apart."""
    offs, cur = [], 0.0
    for w in widths_cm:
        offs.append(cur)
        cur += w + gap_cm
    return offs

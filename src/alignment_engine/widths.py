"""Widths of a flat piece outline along horizontal rows. Metres, piece coordinates."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

F64 = NDArray[np.float64]


def row_width(outline: F64, y: float) -> float:
    """Total length of the part of the horizontal line at ``y`` that lies inside the closed polygon ``outline`` (N,2).

    Even-odd rule: the crossings of the line with the polygon edges are paired from left to right.
    """
    a, b = outline, np.roll(outline, -1, axis=0)
    cond = (a[:, 1] <= y) != (b[:, 1] <= y)
    if not cond.any():
        return 0.0
    t = (y - a[cond, 1]) / (b[cond, 1] - a[cond, 1])
    xs = np.sort(a[cond, 0] + t * (b[cond, 0] - a[cond, 0]))
    return float(np.sum(xs[1::2] - xs[0::2])) if len(xs) % 2 == 0 else 0.0

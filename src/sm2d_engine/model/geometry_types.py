"""Basic geometry types. Units are those of the caller (pattern unit while evaluating, cm after assembly)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Point2 = tuple[float, float]
Polyline = NDArray[np.float64]  # shape (N, 2)
Polygon = NDArray[np.float64]  # shape (N, 2), implicitly closed

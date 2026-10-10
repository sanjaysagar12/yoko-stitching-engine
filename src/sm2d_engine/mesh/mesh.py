"""Mesh value type. Vertices are in **metres**, z = 0."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .. import units


@dataclass
class Mesh:
    """Triangle mesh: ``vertices_m`` (N,3) metres, ``faces`` (M,3) zero-based vertex indices."""

    vertices_m: NDArray[np.float64]
    faces: NDArray[np.int64]

    @classmethod
    def from_polygon_cm(cls, poly_cm: NDArray[np.float64], faces: NDArray[np.int64]) -> Mesh:
        """Build from a 2D polygon in cm (the cm -> m conversion happens here)."""
        v = np.zeros((len(poly_cm), 3), dtype=np.float64)
        v[:, :2] = poly_cm * units.cm_to_m(1.0)
        return cls(v, faces)

    def area_cm2(self) -> float:
        """Sum of triangle areas, cm^2 (computed from the metre vertices)."""
        if len(self.faces) == 0:
            return 0.0
        a, b, c = (self.vertices_m[self.faces[:, k], :2] for k in range(3))
        u, w = b - a, c - a
        return float(np.abs(u[:, 0] * w[:, 1] - u[:, 1] * w[:, 0]).sum() / 2 * 1e4)

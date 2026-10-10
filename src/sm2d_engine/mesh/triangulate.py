"""Triangulation. ``Triangulator`` is the extension point (e.g. interior refinement later)."""

from __future__ import annotations

from typing import Protocol

import mapbox_earcut as earcut
import numpy as np
from numpy.typing import NDArray

from ..model.geometry_types import Polygon


class Triangulator(Protocol):
    """Turns a closed 2D polygon into triangles over *its own vertices* (indices into the polygon)."""

    def triangulate(self, polygon: Polygon) -> NDArray[np.int64]:
        """Return faces (M,3) of vertex indices."""
        ...


class EarcutTriangulator:
    """Boundary-only ear clipping (mapbox_earcut); no interior vertices are added."""

    def triangulate(self, polygon: Polygon) -> NDArray[np.int64]:
        """Faces (M,3); empty for fewer than 3 vertices."""
        if len(polygon) < 3:
            return np.zeros((0, 3), dtype=np.int64)
        v = np.ascontiguousarray(polygon, dtype=np.float64)
        tri = earcut.triangulate_float64(v, np.array([len(v)], dtype=np.uint32))
        return np.asarray(tri, dtype=np.int64).reshape(-1, 3)

"""Mesh validation. Polygon coordinates in cm."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..model.geometry_types import Polygon
from .mesh import Mesh
from .placement import signed_area


def perimeter_cm(p: Polygon) -> float:
    """Closed perimeter, cm."""
    return float(np.hypot(*(np.roll(p, -1, axis=0) - p).T).sum()) if len(p) > 1 else 0.0


def self_intersections(p: Polygon) -> int:
    """Count properly crossing pairs of non-adjacent edges (O(n^2), vectorised per edge)."""
    n = len(p)
    if n < 4:
        return 0
    a, b = p, np.roll(p, -1, axis=0)

    def ccw(u: np.ndarray, v: np.ndarray, w: np.ndarray) -> np.ndarray:
        d1 = (w[..., 1] - u[..., 1]) * (v[..., 0] - u[..., 0])
        d2 = (v[..., 1] - u[..., 1]) * (w[..., 0] - u[..., 0])
        return np.asarray(d1 - d2)

    cnt = 0
    for i in range(n - 2):
        hi = n - 1 if i == 0 else n  # edge n-1 is adjacent to edge 0
        c, d = a[i + 2 : hi], b[i + 2 : hi]
        if len(c) == 0:
            continue
        ai, bi = a[i], b[i]
        ok = (ccw(ai, c, d) * ccw(bi, c, d) < 0) & (ccw(ai, bi, c) * ccw(ai, bi, d) < 0)
        cnt += int(ok.sum())
    return cnt


@dataclass(frozen=True)
class ValidationResult:
    """Outcome of validating one piece mesh against its outline."""

    area_cm2: float
    mesh_area_cm2: float
    area_diff_pct: float
    self_intersections: int
    perimeter_cm: float


class MeshValidator:
    """Checks mesh area against outline area and counts outline self-intersections."""

    def check(self, outline_cm: Polygon, mesh: Mesh) -> ValidationResult:
        """Validate ``mesh`` (built from ``outline_cm``, cm)."""
        area = abs(signed_area(outline_cm)) if len(outline_cm) >= 3 else 0.0
        marea = mesh.area_cm2()
        diff = abs(area - marea) / area * 100 if area > 0 else 0.0
        return ValidationResult(area, marea, diff, self_intersections(outline_cm), perimeter_cm(outline_cm))

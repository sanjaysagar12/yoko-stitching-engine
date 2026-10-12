"""Exact circumferences: radial scaling of the surface around each ring, smooth along the body part axis.

For a ring with axis point ``c`` and direction ``a``, every vertex moves by ``(s - 1) * w(v) * r(v)`` where ``r`` is the
radial offset from the axis, and ``w = cos^2(pi/2 * |d|/D) * (skinning weight of the body part)`` with ``d`` the
distance along the axis from the ring plane and ``D`` the falloff half-length. ``s`` is solved so the ring length hits
the target. Only the ring's own body part moves, so arms stay put when the chest is corrected.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from .bodymodel import BodyModel
from .measurements import MeasurementDef, RingDef
from .slicing import Ring

F64 = NDArray[np.float64]
MAX_SCALE_DEVIATION = 0.10


@dataclass
class CorrectionResult:
    """Corrected vertices and the radial scale applied per ring (1.0 = untouched)."""

    vertices: F64
    scales: dict[str, float]
    warnings: list[str] = field(default_factory=list)


def _displacement_field(
    model: BodyModel, v: F64, j: F64, d: RingDef, ring: Ring, parts: tuple[int, ...], mirrored: bool
) -> F64:
    """Unit-scale displacement ``w(v) * r(v)`` (N,3) for one ring."""
    pts = ring.points(v)
    if d.vertical:
        axis = np.array([0.0, 1.0, 0.0])
        c = pts.mean(axis=0)
    else:
        ia, ib = (model.joint_mirror[d.axis_a], model.joint_mirror[d.axis_b]) if mirrored else (d.axis_a, d.axis_b)
        a, b = j[ia], j[ib]
        axis = (b - a) / np.linalg.norm(b - a)
        c = pts.mean(axis=0)
        c = a + float((c - a) @ axis) * axis
    rel = v - c
    along = rel @ axis
    radial = rel - along[:, None] * axis
    fall = np.cos(0.5 * np.pi * np.clip(np.abs(along) / d.falloff_m, 0.0, 1.0)) ** 2
    part = model.skinning_weights[:, list(parts)].sum(axis=1).clip(0.0, 1.0)
    return np.asarray(radial * (fall * part)[:, None], dtype=np.float64)


class RingCorrector:
    """Scales rings to exact target lengths. ``tol_m`` is the stopping tolerance on every ring length."""

    def __init__(self, tol_m: float = 1e-5, max_passes: int = 6) -> None:
        self.tol_m = tol_m
        self.max_passes = max_passes

    def correct(
        self,
        model: BodyModel,
        v: F64,
        j: F64,
        defs: dict[str, MeasurementDef],
        targets_m: dict[str, float],
    ) -> CorrectionResult:
        """Return corrected vertices; joints ``j`` are used for the limb axes and are not modified."""
        rings = {k: d for k, d in defs.items() if isinstance(d, RingDef) and k in targets_m}
        out = v.copy()
        total = {k: 1.0 for k in rings}
        for _ in range(self.max_passes):
            worst = 0.0
            for k, d in rings.items():
                length = d.ring.length(out)
                worst = max(worst, abs(length - targets_m[k]))
                if abs(length - targets_m[k]) <= self.tol_m:
                    continue
                field_main = _displacement_field(model, out, j, d, d.ring, d.parts, False)
                field_mir = (
                    _displacement_field(model, out, j, d, d.mirror_ring, d.mirror_parts, True)
                    if d.mirror_ring is not None
                    else None
                )
                total_field = field_main if field_mir is None else field_main + field_mir

                def at(s: float, f: F64 = total_field, ring: Ring = d.ring, base: F64 = out) -> float:
                    return ring.length(base + (s - 1.0) * f)

                s = _solve_scale(at, targets_m[k])
                out = out + (s - 1.0) * total_field
                total[k] *= s
            if worst <= self.tol_m:
                break
        warns = [
            f"{k}: radial correction {100 * (s - 1):+.1f} % exceeds +-{100 * MAX_SCALE_DEVIATION:.0f} %"
            for k, s in total.items()
            if abs(s - 1.0) > MAX_SCALE_DEVIATION
        ]
        return CorrectionResult(out, total, warns)


def _solve_scale(length_at: Callable[[float], float], target: float) -> float:
    """Secant solve of ``length_at(s) = target`` starting from s = 1."""
    s0, s1 = 1.0, target / max(length_at(1.0), 1e-9)
    f0, f1 = length_at(s0) - target, length_at(s1) - target
    for _ in range(30):
        if abs(f1) < 1e-9 or f1 == f0:
            break
        s0, s1, f0 = s1, s1 - f1 * (s1 - s0) / (f1 - f0), f1
        f1 = length_at(s1) - target
    return float(s1)

"""Curve orientation and trimming for piece assembly. Units: pattern unit throughout."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from ..geometry.bezier import Curve
from ..model.geometry_types import Point2

_EPS_U = 1e-9


def place_curve(
    curve: Curve,
    pts: NDArray[np.float64],
    us: NDArray[np.float64],
    prev: Point2 | None,
    nxt: Point2 | None,
    ref: Point2 | None,
    reverse_flag: bool,
    tol: float,
) -> tuple[list[Point2], float]:
    """Pick the curve direction by geometry and trim it between its neighbouring point nodes.

    ``pts``/``us`` is the flattened curve (vertices and their parameters). ``prev``/``nxt`` are the
    neighbouring point-node positions (or None if the neighbour is not a point node); a neighbour only
    trims the curve if it lies within ``tol`` of it. ``ref`` is the last outline vertex. The direction
    with the smallest gap to ``ref`` wins; ``reverse_flag`` is only a tie-break.

    Returns (vertices to append, gap between ``ref`` and the chosen start; 0 if trimmed at the start).
    """
    umax = curve.u_max
    pu = curve.project(prev) if prev is not None else None
    nu = curve.project(nxt) if nxt is not None else None
    u_prev = pu[0] if pu is not None and pu[1] < tol else None
    u_next = nu[0] if nu is not None and nu[1] < tol else None
    best: tuple[float, list[Point2], float] | None = None
    for rev in (False, True):
        q = pts[::-1] if rev else pts
        uq = umax - us[::-1] if rev else us
        u0 = 0.0 if u_prev is None else (umax - u_prev if rev else u_prev)
        u1 = umax if u_next is None else (umax - u_next if rev else u_next)
        if u0 > u1 + _EPS_U:
            continue
        start_free = u0 <= _EPS_U
        gap = 0.0 if ref is None or not start_free else math.dist(ref, (float(q[0, 0]), float(q[0, 1])))
        cost = gap + (0.0 if rev == reverse_flag else 1e-7)
        if best is not None and cost >= best[0]:
            continue
        inner = (uq > u0 + _EPS_U) & (uq < u1 - _EPS_U)
        seg = [(float(x), float(y)) for x, y in q[inner]]
        if start_free:
            seg.insert(0, (float(q[0, 0]), float(q[0, 1])))
        if u1 >= umax - _EPS_U:
            seg.append((float(q[-1, 0]), float(q[-1, 1])))
        best = (cost, seg, gap)
    if best is None:
        raise ValueError("cannot orient curve between its neighbouring points")
    return best[1], best[2]

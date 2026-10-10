"""Pure 2D primitives in Qt convention (y down, angles counter-clockwise on screen). Units: caller's."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from ..model.geometry_types import Point2


def qt_angle(p: Point2, q: Point2) -> float:
    """Angle of line p->q in degrees [0, 360), y axis pointing up on screen (QLineF::angle)."""
    return math.degrees(math.atan2(-(q[1] - p[1]), q[0] - p[0])) % 360.0


def move(p: Point2, length: float, angle: float) -> Point2:
    """Point at ``length`` from ``p`` in direction ``angle`` degrees (Qt sense)."""
    a = math.radians(angle)
    return (p[0] + length * math.cos(a), p[1] - length * math.sin(a))


def dist(p: Point2, q: Point2) -> float:
    """Euclidean distance."""
    return math.dist(p, q)


def line_line(a1: Point2, a2: Point2, b1: Point2, b2: Point2) -> Point2 | None:
    """Intersection of the infinite lines a1-a2 and b1-b2, or None if parallel."""
    d = (a2[0] - a1[0], a2[1] - a1[1])
    e = (b2[0] - b1[0], b2[1] - b1[1])
    den = d[0] * e[1] - d[1] * e[0]
    if abs(den) < 1e-12 * max(1.0, math.hypot(*d) * math.hypot(*e)):
        return None
    w = (b1[0] - a1[0], b1[1] - a1[1])
    t = (w[0] * e[1] - w[1] * e[0]) / den
    return (a1[0] + t * d[0], a1[1] + t * d[1])


def circle_line(p1: Point2, p2: Point2, center: Point2, r: float) -> tuple[float, float] | None:
    """Distances t1<=t2 from p1 along the direction p1->p2 where the line meets the circle, or None."""
    n = math.dist(p1, p2)
    d = ((p2[0] - p1[0]) / n, (p2[1] - p1[1]) / n)
    w = (p1[0] - center[0], p1[1] - center[1])
    b = w[0] * d[0] + w[1] * d[1]
    c = w[0] ** 2 + w[1] ** 2 - r * r
    disc = b * b - c
    if disc < 0:
        return None
    s = math.sqrt(disc)
    return (-b - s, -b + s)


def ray_polyline(base: Point2, angle: float, pts: NDArray[np.float64]) -> list[Point2]:
    """All intersections of the ray (base, angle) with the polyline ``pts`` (N,2)."""
    d = (math.cos(math.radians(angle)), -math.sin(math.radians(angle)))
    a, b = pts[:-1], pts[1:]
    e = b - a
    den = d[0] * e[:, 1] - d[1] * e[:, 0]
    w = a - np.asarray(base)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (w[:, 0] * e[:, 1] - w[:, 1] * e[:, 0]) / den
        u = (w[:, 0] * d[1] - w[:, 1] * d[0]) / den
    ok = (np.abs(den) >= 1e-14) & (t >= -1e-12) & (u >= -1e-12) & (u <= 1 + 1e-12)
    return [(base[0] + float(tt) * d[0], base[1] + float(tt) * d[1]) for tt in t[ok]]

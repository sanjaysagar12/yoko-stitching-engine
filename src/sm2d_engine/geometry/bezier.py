"""Cubic Bezier chains: arc length, projection, splitting, adaptive flattening. Units: caller's."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from ..model.geometry_types import Point2
from .primitives import ray_polyline

F64 = NDArray[np.float64]
_DENSE = 2000  # samples per segment for lengths / projection (error ~1e-8 of length)


def bez_eval(seg: F64, t: F64) -> F64:
    """Evaluate the cubic ``seg`` (4,2) at parameters ``t`` (K,) -> (K,2)."""
    u = (1 - t)[:, None]
    tt = t[:, None]
    out: F64 = u**3 * seg[0] + 3 * u * u * tt * seg[1] + 3 * u * tt * tt * seg[2] + tt**3 * seg[3]
    return out


def split(seg: F64, t: float) -> tuple[F64, F64]:
    """de Casteljau split of a cubic at ``t``."""
    p01 = seg[0] + t * (seg[1] - seg[0])
    p12 = seg[1] + t * (seg[2] - seg[1])
    p23 = seg[2] + t * (seg[3] - seg[2])
    a = p01 + t * (p12 - p01)
    b = p12 + t * (p23 - p12)
    m = a + t * (b - a)
    return np.array([seg[0], p01, a, m]), np.array([m, b, p23, seg[3]])


def _restrict(seg: F64, ta: float, tb: float) -> F64:
    r = seg if ta <= 0 else split(seg, ta)[1]
    if tb >= 1:
        return r
    return split(r, (tb - ta) / (1 - ta))[0]


def _extrema(seg: F64) -> list[float]:
    """Parameters in (0,1) where x' or y' vanishes."""
    ts: list[float] = []
    for k in (0, 1):
        a = seg[1, k] - seg[0, k]
        b = seg[2, k] - seg[1, k]
        c = seg[3, k] - seg[2, k]
        qa, qb, qc = a - 2 * b + c, 2 * (b - a), a
        if abs(qa) < 1e-14:
            roots = [-qc / qb] if abs(qb) > 1e-14 else []
        else:
            disc = qb * qb - 4 * qa * qc
            roots = [] if disc < 0 else [(-qb - math.sqrt(disc)) / (2 * qa), (-qb + math.sqrt(disc)) / (2 * qa)]
        ts += [r for r in roots if 1e-9 < r < 1 - 1e-9]
    return sorted(set(ts))


def _flat_enough(seg: F64, tol: float) -> bool:
    p0, p3 = seg[0], seg[3]
    d = p3 - p0
    n = math.hypot(d[0], d[1])
    if n < 1e-12:
        dev = max(math.hypot(*(seg[1] - p0)), math.hypot(*(seg[2] - p0)))
    else:
        dev = max(abs(d[0] * (q[1] - p0[1]) - d[1] * (q[0] - p0[0])) / n for q in (seg[1], seg[2]))
    return 0.75 * dev <= tol


def _flatten(seg: F64, tol: float, ta: float, tb: float, out: list[tuple[float, float, float]], depth: int = 0) -> None:
    """Append the vertices (x, y, t) after the start of ``seg`` (covering [ta,tb]) to ``out``."""
    if depth >= 24 or _flat_enough(seg, tol):
        out.append((float(seg[3, 0]), float(seg[3, 1]), tb))
        return
    left, right = split(seg, 0.5)
    tm = (ta + tb) / 2
    _flatten(left, tol, ta, tm, out, depth + 1)
    _flatten(right, tol, tm, tb, out, depth + 1)


class Curve:
    """Chain of cubic Bezier segments, each a (4,2) array. Parameter u = segment index + t."""

    def __init__(self, segments: list[F64]) -> None:
        self.segments = [np.asarray(s, dtype=np.float64) for s in segments]
        self._dense: tuple[F64, F64, F64] | None = None

    def dense(self) -> tuple[F64, F64, F64]:
        """Dense sampling: (points (K,2), u (K,), cumulative arc length (K,))."""
        if self._dense is None:
            t = np.linspace(0.0, 1.0, _DENSE + 1)
            pts, us = [], []
            for i, s in enumerate(self.segments):
                p = bez_eval(s, t)
                pts.append(p if i == 0 else p[1:])
                us.append(i + (t if i == 0 else t[1:]))
            P, U = np.concatenate(pts), np.concatenate(us)
            S = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(P, axis=0).T))])
            self._dense = (P, U, S)
        return self._dense

    def length(self) -> float:
        """Arc length (caller's unit)."""
        return float(self.dense()[2][-1])

    @property
    def u_max(self) -> float:
        """Largest parameter value (= number of segments)."""
        return float(len(self.segments))

    def length_at_u(self, u: float) -> float:
        """Arc length from the start up to parameter ``u``."""
        _, U, S = self.dense()
        return float(np.interp(u, U, S))

    def point_at_u(self, u: float) -> Point2:
        """Curve point at parameter ``u``."""
        i = min(int(math.floor(u)), len(self.segments) - 1)
        p = bez_eval(self.segments[i], np.array([u - i]))[0]
        return (float(p[0]), float(p[1]))

    def point_at_length(self, length: float) -> Point2:
        """Point at arc length ``length`` from the start (clamped to the curve)."""
        _, U, S = self.dense()
        return self.point_at_u(float(np.interp(length, S, U)))

    def project(self, q: Point2) -> tuple[float, float]:
        """Closest point of the curve to ``q`` -> (parameter u, distance)."""
        P, U, _ = self.dense()
        a, e = P[:-1], P[1:] - P[:-1]
        ll = e[:, 0] ** 2 + e[:, 1] ** 2
        w = np.asarray(q) - a
        with np.errstate(divide="ignore", invalid="ignore"):
            k = np.where(ll > 0, (w[:, 0] * e[:, 0] + w[:, 1] * e[:, 1]) / ll, 0.0)
        k = np.clip(k, 0.0, 1.0)
        d = np.hypot(w[:, 0] - k * e[:, 0], w[:, 1] - k * e[:, 1])
        i = int(np.argmin(d))
        return float(U[i] + k[i] * (U[i + 1] - U[i])), float(d[i])

    def ray_hits(self, base: Point2, angle: float) -> list[Point2]:
        """Intersections of the ray (base, angle degrees Qt) with the curve."""
        return ray_polyline(base, angle, self.dense()[0])

    def flatten(self, tol: float) -> tuple[F64, F64]:
        """Adaptive polyline with chord deviation <= ``tol`` (caller's unit); curve extrema are vertices.

        Returns (points (K,2), parameters u (K,)).
        """
        first = self.segments[0][0]
        rows: list[tuple[float, float, float]] = [(float(first[0]), float(first[1]), 0.0)]
        for i, s in enumerate(self.segments):
            cuts = [0.0, *_extrema(s), 1.0]
            for ta, tb in zip(cuts[:-1], cuts[1:], strict=True):
                part: list[tuple[float, float, float]] = []
                _flatten(_restrict(s, ta, tb), tol, ta, tb, part)
                rows += [(x, y, i + t) for x, y, t in part]
        arr = np.array(rows, dtype=np.float64)
        return arr[:, :2].copy(), arr[:, 2].copy()

"""Rigid placement maths. Pure functions, metres, angles in degrees at the interface.

Piece frame: x right, y up, z out of the flat face. Target frame from an arrangement point: ``n`` = normal,
``u`` = up projected perpendicular to ``n``, ``t = u x n``. The rigid map sends piece x -> t, piece y -> u, piece z -> n
(a proper rotation), then rotates by the user's angle about ``n``, then moves the reference point to the target.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

F64 = NDArray[np.float64]
I64 = NDArray[np.int64]


@dataclass(frozen=True)
class Frame:
    """Orthonormal target frame (unit vectors): ``t = u x n``."""

    n: F64
    u: F64
    t: F64


def _unit(v: F64) -> F64:
    return np.asarray(v / np.linalg.norm(v), dtype=np.float64)


def target_frame(normal: F64, up: F64, flip: bool = False) -> Frame:
    """Frame at a point. ``flip`` turns the piece face to the other side (n -> -n); ``u`` stays."""
    n = _unit(np.asarray(normal, dtype=np.float64))
    u0 = np.asarray(up, dtype=np.float64)
    u = u0 - (u0 @ n) * n
    if np.linalg.norm(u) < 1e-9:  # up parallel to the normal: any perpendicular, preferring world up
        u = np.array([0.0, 1.0, 0.0]) - n[1] * n
        if np.linalg.norm(u) < 1e-9:
            u = np.array([0.0, 0.0, 1.0]) - n[2] * n
    u = _unit(u)
    if flip:
        n = -n
    return Frame(n, u, np.cross(u, n))


def rodrigues(axis: F64, deg: float) -> F64:
    """Rotation matrix about unit ``axis`` by ``deg`` degrees (counter-clockwise, right-hand rule)."""
    a = _unit(np.asarray(axis, dtype=np.float64))
    th = np.radians(deg)
    k = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    return np.asarray(np.eye(3) + np.sin(th) * k + (1.0 - np.cos(th)) * (k @ k), dtype=np.float64)


def rigid_matrix(frame: Frame, rotation_deg: float, ref_local: F64, target: F64) -> F64:
    """4x4 (column vectors) mapping piece points to world: the reference lands on ``target``."""
    base = np.column_stack([frame.t, frame.u, frame.n])
    rot = rodrigues(frame.n, rotation_deg) @ base
    m = np.eye(4)
    m[:3, :3] = rot
    m[:3, 3] = np.asarray(target) - rot @ np.asarray(ref_local)
    return m


def mirror_matrix(ref_x: float) -> F64:
    """Reflection across the plane x = ``ref_x`` as a 4x4."""
    m = np.eye(4)
    m[0, 0] = -1.0
    m[0, 3] = 2.0 * ref_x
    return m


def apply_matrix(m: F64, v: F64) -> F64:
    """Apply a 4x4 to points (N,3)."""
    return np.asarray(v @ m[:3, :3].T + m[:3, 3], dtype=np.float64)


def mirror_mesh(v: F64, f: I64, ref_x: float) -> tuple[F64, I64]:
    """Mirror vertices across x = ``ref_x`` and flip the triangle winding so normals keep their direction."""
    out = v.copy()
    out[:, 0] = 2.0 * ref_x - v[:, 0]
    return out, np.ascontiguousarray(f[:, [0, 2, 1]])


def refine_mesh(v: F64, f: I64, max_edge: float) -> tuple[F64, I64]:
    """Conforming midpoint subdivision until every edge is <= ``max_edge`` (metres). Boundary and area are unchanged.

    Used before wrapping so large flat triangles do not cut through the body.
    """
    v, f = v.copy(), f.copy()
    for _ in range(12):
        e = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
        uniq, inv = np.unique(e, axis=0, return_inverse=True)
        long_ = np.linalg.norm(v[uniq[:, 0]] - v[uniq[:, 1]], axis=1) > max_edge
        if not long_.any():
            break
        mid_id = np.full(len(uniq), -1, dtype=np.int64)
        idx = np.flatnonzero(long_)
        mid_id[idx] = len(v) + np.arange(len(idx))
        v = np.vstack([v, 0.5 * (v[uniq[idx, 0]] + v[uniq[idx, 1]])])
        em = mid_id[inv].reshape(3, len(f)).T  # midpoint id per face edge (ab, bc, ca) or -1
        out: list[tuple[int, int, int]] = []
        for (a, b, c), (m_ab, m_bc, m_ca) in zip(f.tolist(), em.tolist(), strict=True):
            out += _split(v, a, b, c, m_ab, m_bc, m_ca)
        f = np.array(out, dtype=np.int64)
    return v, f


def _split(v: F64, a: int, b: int, c: int, mab: int, mbc: int, mca: int) -> list[tuple[int, int, int]]:
    n = (mab >= 0) + (mbc >= 0) + (mca >= 0)
    if n == 0:
        return [(a, b, c)]
    if n == 3:
        return [(a, mab, mca), (mab, b, mbc), (mca, mbc, c), (mab, mbc, mca)]
    # rotate (a, b, c) so that the split edges come first: ab (one split) or ab and bc (two splits)
    for _ in range(3):
        if (n == 1 and mab >= 0) or (n == 2 and mab >= 0 and mbc >= 0):
            break
        a, b, c, mab, mbc, mca = b, c, a, mbc, mca, mab
    if n == 1:
        return [(a, mab, c), (mab, b, c)]
    # two split edges ab, bc: corner triangle at b plus a quad (a, mab, mbc, c) split on its shorter diagonal
    out = [(mab, b, mbc)]
    if np.linalg.norm(v[a] - v[mbc]) <= np.linalg.norm(v[mab] - v[c]):
        out += [(a, mab, mbc), (a, mbc, c)]
    else:
        out += [(a, mab, c), (mab, mbc, c)]
    return out

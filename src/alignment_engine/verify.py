"""Distances between meshes: exact point-to-triangle distance, avatar surface queries, piece samples. Metres, pure
functions."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

F64 = NDArray[np.float64]
I64 = NDArray[np.int64]


def point_triangle_distance(p: F64, tri: F64) -> F64:
    """Exact distance from points ``p`` (M,3) to triangles ``tri`` (M,3,3), pairwise (Ericson, closest point on a
    triangle)."""
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = np.einsum("ij,ij->i", ab, ap), np.einsum("ij,ij->i", ac, ap)
    bp = p - b
    d3, d4 = np.einsum("ij,ij->i", ab, bp), np.einsum("ij,ij->i", ac, bp)
    cp = p - c
    d5, d6 = np.einsum("ij,ij->i", ab, cp), np.einsum("ij,ij->i", ac, cp)
    vc, vb, va = d1 * d4 - d3 * d2, d5 * d2 - d1 * d6, d3 * d6 - d5 * d4
    out = np.empty((len(p), 3))
    with np.errstate(divide="ignore", invalid="ignore"):
        # default: interior of the face
        den = va + vb + vc
        v_, w_ = vb / den, vc / den
        out[:] = a + ab * v_[:, None] + ac * w_[:, None]
        # edge ac, vertex c, edge bc, vertex b, edge ab, vertex a (later assignments win, as in the reference algorithm)
        m = (va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0)  # edge bc
        t = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        out[m] = (b + (c - b) * t[:, None])[m]
        m = (vb <= 0) & (d2 >= 0) & (d6 <= 0)  # edge ac
        t = d2 / (d2 - d6)
        out[m] = (a + ac * t[:, None])[m]
        m = (d6 >= 0) & (d5 <= d6)  # vertex c
        out[m] = c[m]
        m = (vc <= 0) & (d1 >= 0) & (d3 <= 0)  # edge ab
        t = d1 / (d1 - d3)
        out[m] = (a + ab * t[:, None])[m]
        m = (d3 >= 0) & (d4 <= d3)  # vertex b
        out[m] = b[m]
        m = (d1 <= 0) & (d2 <= 0)  # vertex a
        out[m] = a[m]
    return np.asarray(np.linalg.norm(p - out, axis=1), dtype=np.float64)


class AvatarSurface:
    """The avatar surface for distance queries: a KD-tree of dense samples (about ``spacing`` apart) and the
    triangles."""

    def __init__(self, vertices: F64, faces: I64, spacing: float = 0.003) -> None:
        self.vertices, self.faces = vertices, faces
        tri = vertices[faces]
        e = np.linalg.norm(tri - np.roll(tri, -1, axis=1), axis=2).max(axis=1)
        k = np.clip(np.ceil(e / spacing).astype(int), 1, 8)
        pts, owner = [vertices], [_vertex_owner(faces, len(vertices))]
        for kk in np.unique(k):
            sel = np.flatnonzero(k == kk)
            ii, jj = np.meshgrid(np.arange(kk + 1), np.arange(kk + 1))
            keep = ii + jj <= kk
            w = np.column_stack([ii[keep], jj[keep], kk - ii[keep] - jj[keep]]) / kk
            pts.append((w[None, :, :, None] * tri[sel][:, None, :, :]).sum(axis=2).reshape(-1, 3))
            owner.append(np.repeat(sel, len(w)))
        self.samples = np.vstack(pts)
        self.owner = np.concatenate(owner)
        self.tree = cKDTree(self.samples)
        self.tri = tri
        fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        if float(np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum()) < 0:
            fn = -fn  # inward winding: flip so that normals point out of the body
        self.normals = fn / np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-18)

    def distance(self, points: F64, exact_within: float = 0.06) -> F64:
        """Distance of ``points`` (K,3) to the surface (exact for points within ``exact_within`` of it)."""
        d, idx = self.tree.query(points, k=6, workers=-1 if len(points) > 8000 else 1)
        out = d[:, 0].copy()
        near = np.flatnonzero(d[:, 0] < exact_within)
        if len(near):
            cand = self.owner[idx[near]]  # (n,6) candidate triangles from the 6 nearest samples
            pp = np.repeat(points[near], cand.shape[1], axis=0)
            dd = point_triangle_distance(pp, self.tri[cand.ravel()]).reshape(cand.shape)
            out[near] = dd.min(axis=1)
        return np.asarray(out, dtype=np.float64)

    def signed_distance(self, points: F64, exact_within: float = 0.06) -> F64:
        """Distance to the surface, negative for points inside the body (side of the nearest triangle's
        outward normal)."""
        d, idx = self.tree.query(points, k=6, workers=-1 if len(points) > 8000 else 1)
        cand = self.owner[idx]
        out = d[:, 0].copy()
        best = cand[:, 0].copy()
        near = np.flatnonzero(d[:, 0] < exact_within)
        if len(near):
            pp = np.repeat(points[near], cand.shape[1], axis=0)
            dd = point_triangle_distance(pp, self.tri[cand[near].ravel()]).reshape(len(near), -1)
            j = dd.argmin(axis=1)
            out[near] = dd[np.arange(len(near)), j]
            best[near] = cand[near, j]
        side = np.einsum("ij,ij->i", points - self.tri[best, 0], self.normals[best])
        return np.asarray(np.where(side < 0, -out, out), dtype=np.float64)


def _vertex_owner(faces: I64, n: int) -> I64:
    owner = np.zeros(n, dtype=np.int64)
    owner[faces[::-1, 0]] = np.arange(len(faces))[::-1]
    owner[faces[::-1, 1]] = np.arange(len(faces))[::-1]
    owner[faces[::-1, 2]] = np.arange(len(faces))[::-1]
    return owner


def surface_samples(vertices: F64, faces: I64, spacing: float = 0.003) -> F64:
    """Vertices plus points along every triangle edge at most ``spacing`` apart (the boundary and the mesh lines)."""
    e = np.unique(np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1), axis=0)
    a, b = vertices[e[:, 0]], vertices[e[:, 1]]
    n = np.maximum(np.ceil(np.linalg.norm(b - a, axis=1) / spacing).astype(int), 1)
    t = np.concatenate([np.arange(1, k) / k for k in n if k > 1] or [np.zeros(0)])
    idx = np.repeat(np.arange(len(e)), np.maximum(n - 1, 0))
    mid = a[idx] + (b[idx] - a[idx]) * t[:, None]
    return np.vstack([vertices, mid])


def self_overlaps(vertices: F64, faces: I64, radius: float = 0.002) -> int:
    """Pairs of non-adjacent triangles whose centroids are closer than ``radius``: a proxy for a mesh that folds
    over itself."""
    cen = vertices[faces].mean(axis=1)
    pairs = cKDTree(cen).query_pairs(radius, output_type="ndarray")
    if len(pairs) == 0:
        return 0
    fa, fb = faces[pairs[:, 0]], faces[pairs[:, 1]]
    share = (fa[:, :, None] == fb[:, None, :]).any(axis=(1, 2))
    return int((~share).sum())


def min_distance_between(a: F64, b: F64, tree_b: Any = None) -> float:
    """Smallest distance between two point sets."""
    t = tree_b if tree_b is not None else cKDTree(b)
    return float(t.query(a)[0].min())

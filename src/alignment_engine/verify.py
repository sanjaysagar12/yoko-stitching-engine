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
    """Exact distance from points ``p`` (M,3) to triangles ``tri`` (M,3,3), pairwise."""
    return np.asarray(np.linalg.norm(p - closest_on_triangle(p, tri), axis=1), dtype=np.float64)


def closest_on_triangle(p: F64, tri: F64) -> F64:
    """Closest points (M,3) on triangles ``tri`` (M,3,3) to points ``p`` (M,3), pairwise (Ericson)."""
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
    return out


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
        # every surface point has a sample of its own triangle within this distance (sub-triangle edge / sqrt 3)
        self.cover = float((e / k).max()) / np.sqrt(3.0)
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

    def closest(self, points: F64, within: float) -> tuple[F64, F64, NDArray[np.bool_]]:
        """(signed distance, unit direction away from the body, mask of points closer than ``within``) for ``points``.

        Only points within ``within`` of the surface are measured exactly (the others get ``inf`` and a zero direction):
        the direction is from the closest surface point to the point (its negative inside the body), or the outward
        triangle normal when the point is on the surface.
        """
        d, idx = self.tree.query(points, k=8, distance_upper_bound=within + 0.01, workers=-1)
        near = np.flatnonzero(np.isfinite(d[:, 0]))
        sd = np.full(len(points), np.inf)
        direction = np.zeros((len(points), 3))
        if len(near):
            cand = self.owner[np.where(np.isfinite(d[near]), idx[near], idx[near, :1])]
            pp = np.repeat(points[near], cand.shape[1], axis=0)
            q = closest_on_triangle(pp, self.tri[cand.ravel()]).reshape(len(near), cand.shape[1], 3)
            dd = np.linalg.norm(points[near, None, :] - q, axis=2)
            j = dd.argmin(axis=1)
            rows = np.arange(len(near))
            best, qn, dist = cand[rows, j], q[rows, j], dd[rows, j]
            nrm = self.normals[best]
            side = np.einsum("ij,ij->i", points[near] - self.tri[best, 0], nrm)
            sign = np.where(side < 0, -1.0, 1.0)
            vec = (points[near] - qn) * sign[:, None]
            ok = dist > 1e-6
            direction[near] = np.where(ok[:, None], vec / np.maximum(dist, 1e-12)[:, None], nrm)
            sd[near] = sign * dist
        far = np.flatnonzero(~np.isfinite(d[:, 0]))
        if len(far):  # deep inside the body is not "far": at least ``within`` deep, pushed out along the vertex normal
            deep, vn = self.inside(points[far])
            sd[far[deep]] = -within
            direction[far[deep]] = vn[deep]
        return sd, direction, sd < within

    def inside(self, points: F64) -> tuple[NDArray[np.bool_], F64]:
        """(inside the body, outward normal of the nearest vertex) for points: the side of the nearest vertex's
        tangent plane (exact enough for points more than a centimetre from the surface)."""
        if not hasattr(self, "_vtree"):
            n = np.zeros_like(self.vertices)
            fn = self.normals * np.linalg.norm(
                np.cross(self.tri[:, 1] - self.tri[:, 0], self.tri[:, 2] - self.tri[:, 0]), axis=1, keepdims=True
            )
            for k in range(3):
                np.add.at(n, self.faces[:, k], fn)
            self._vnormals = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-18)
            self._vtree = cKDTree(self.vertices)
        _, j = self._vtree.query(points, workers=-1)
        vn = self._vnormals[j]
        return np.einsum("ij,ij->i", points - self.vertices[j], vn) < 0, vn


class MeshProx:
    """Exact distances from points to a triangle mesh: a KD-tree of the vertices selects the candidate triangles
    (those around the nearest vertices), then the exact point-triangle distance is taken. Valid when the queried
    distances are larger than about one edge length, which holds for gaps of centimetres on 5 mm meshes."""

    def __init__(self, vertices: F64, faces: I64, adjacency: I64) -> None:
        self.vertices, self.faces, self.adj = vertices, faces, adjacency
        self.tree = cKDTree(vertices)
        self.tri = vertices[faces]

    def query(self, points: F64, upper: float, k: int = 3) -> tuple[F64, F64, I64, I64]:
        """(distance, closest point, triangle, nearest vertex) per point; points farther than ``upper`` from every
        vertex get distance ``inf`` (and -1 indices)."""
        d, idx = self.tree.query(points, k=k, distance_upper_bound=upper)
        n = len(points)
        dist, close = np.full(n, np.inf), np.zeros((n, 3))
        tri, vert = np.full(n, -1, dtype=np.int64), np.full(n, -1, dtype=np.int64)
        near = np.flatnonzero(np.isfinite(d[:, 0]))
        if not len(near):
            return dist, close, tri, vert
        vi = np.where(np.isfinite(d[near]), idx[near], idx[near, :1])
        cand = self.adj[vi].reshape(len(near), -1)  # (m, k * valence), -1 padded
        valid = cand >= 0
        safe = np.where(valid, cand, 0)
        pp = np.repeat(points[near], cand.shape[1], axis=0)
        q = closest_on_triangle(pp, self.tri[safe.ravel()]).reshape(len(near), cand.shape[1], 3)
        dd = np.where(valid, np.linalg.norm(points[near, None, :] - q, axis=2), np.inf)
        j = dd.argmin(axis=1)
        rows = np.arange(len(near))
        dist[near], close[near], tri[near], vert[near] = dd[rows, j], q[rows, j], safe[rows, j], vi[:, 0]
        return dist, close, tri, vert

    def min_distance(self, points: F64, upper: float) -> tuple[float, int, F64, int]:
        """Exact smallest distance from ``points`` to the mesh (``inf`` if all are farther than ``upper``).

        Returns (distance, index of the point, closest point, triangle). If the closest point ``q`` of ``p`` lies
        inside a face or an edge, ``p - q`` is perpendicular to it, so a vertex of that face is within
        ``sqrt(D^2 + R^2)`` of ``p`` (``R`` the largest circumradius). Hence only points whose nearest vertex is within
        ``sqrt(U^2 + R^2)`` (``U`` the best exact distance so far) can attain the minimum, and for each of them every
        vertex within that radius is a candidate (``k`` grows until the ball is covered).
        """
        d, _ = self.tree.query(points, distance_upper_bound=upper, workers=-1)
        fin = np.flatnonzero(np.isfinite(d))
        if not len(fin):
            return np.inf, -1, np.zeros(3), -1
        radius = self._radius()
        order = fin[np.argsort(d[fin], kind="stable")]
        best = (np.inf, -1, np.zeros(3), -1)
        for start in range(0, len(order), 2048):
            chunk = order[start : start + 2048]
            if np.isfinite(best[0]):
                chunk = chunk[d[chunk] ** 2 - radius**2 <= best[0] ** 2]
                if not len(chunk):
                    break
            dist, close, tri = self._exact(points[chunk], radius, best[0])
            k = int(np.argmin(dist))
            if dist[k] < best[0]:
                best = (float(dist[k]), int(chunk[k]), close[k], int(tri[k]))
        return best

    def _radius(self) -> float:
        """Largest distance from a point of a triangle to its nearest corner: at most the longest edge over sqrt(3)
        (the circumradius of an acute triangle, half the longest edge of an obtuse one), slivers included."""
        if not hasattr(self, "_r"):
            t = self.tri
            longest = np.max(np.linalg.norm(t - np.roll(t, -1, axis=1), axis=2))
            self._r = float(longest) / float(np.sqrt(3.0))
        return float(self._r)

    def _exact(self, pts: F64, radius: float, best: float) -> tuple[F64, F64, I64]:
        """Exact distance of each point: candidate triangles around every vertex within ``sqrt(d^2 + R^2)``."""
        n = len(pts)
        dist, close, tri = np.full(n, np.inf), np.zeros((n, 3)), np.full(n, -1, dtype=np.int64)
        todo = np.arange(n)
        k = 4
        while len(todo):
            kk = min(k, len(self.vertices))
            dv, iv = self.tree.query(pts[todo], k=kk, workers=-1)
            dv, iv = dv.reshape(len(todo), kk), iv.reshape(len(todo), kk)
            cand = self.adj[iv].reshape(len(todo), -1)
            valid = cand >= 0
            safe = np.where(valid, cand, 0)
            pp = np.repeat(pts[todo], cand.shape[1], axis=0)
            q = closest_on_triangle(pp, self.tri[safe.ravel()]).reshape(len(todo), cand.shape[1], 3)
            dd = np.where(valid, np.linalg.norm(pts[todo, None, :] - q, axis=2), np.inf)
            j = dd.argmin(axis=1)
            rows = np.arange(len(todo))
            dist[todo], close[todo], tri[todo] = dd[rows, j], q[rows, j], safe[rows, j]
            ball = np.sqrt(np.minimum(dist[todo], best) ** 2 + radius**2)
            covered = (dv[:, -1] > ball) | (kk >= len(self.vertices))
            todo = todo[~covered]
            k *= 3
        return dist, close, tri


def vertex_faces(faces: I64, n: int) -> I64:
    """Faces around every vertex, (n, max valence) padded with -1."""
    flat = faces.ravel()
    order = np.argsort(flat, kind="stable")
    counts = np.bincount(flat, minlength=n)
    width = int(counts.max()) if len(counts) else 1
    out = np.full((n, width), -1, dtype=np.int64)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    slot = np.arange(len(flat)) - np.repeat(starts, counts)
    out[flat[order], slot] = order // 3
    return out


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

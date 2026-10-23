"""Mesh geometry of the simulation: flat coordinates of a placed mesh, the simulation mesh of a piece, point location.

* ``flat_coords``: the alignment result stores a placed (rolled) mesh in world coordinates only. Placement is isometric
  (README, the scale guarantee: at most 0.05 %), so its flat (pattern) coordinates follow from the outline and a
  harmonic extension: the boundary loop gets its flat positions from the outline (``world_outline_m`` and the bundle's
  ``outline_m`` share their index; between two outline vertices by the fraction of world arc length), the interior from
  the cotangent Laplace equation. Linear functions are cotangent-harmonic on a flat mesh and the cotangent weights
  depend on edge lengths only, which the placement keeps, so the solve gives back the flat mesh to the placement's own
  accuracy (``flat_error``).
* ``sim_mesh``: a conforming Delaunay triangulation of a flat outline with edges of about ``h`` (boundary resampled at
  ``h``, corners kept; a hexagonal grid inside), the method of the alignment engine's refinement written again here
  (the simulator imports no engine).
* ``locate2d``: containing triangle and barycentric weights of flat points (clamped onto the nearest triangle).
"""

from __future__ import annotations

import numpy as np
import shapely
from numpy.typing import NDArray
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu
from scipy.spatial import Delaunay, cKDTree
from shapely.geometry import Polygon

from .errors import SimulationError

F64 = NDArray[np.float64]
I64 = NDArray[np.int64]

CORNER_DEG = 25.0  # an outline vertex turning more than this is a corner: the resampled boundary keeps it
MATCH_M = 2e-5  # an outline vertex of world_outline_m is a vertex of the placed mesh's boundary (files keep 1 um)
GAP_MIN_M = 0.004  # boundary parts closer than this (a slit drafted 0.1 mm wide) are opened to it in the sim mesh
ARAP_TRIGGER_M = 5e-5  # a harmonic solve whose edges are off by more than this is refined (sliver meshes)


def boundary_edges(faces: I64) -> I64:
    """Directed boundary edges (E,2) of a triangle mesh (edges used by one triangle, in that triangle's direction)."""
    f = np.asarray(faces, dtype=np.int64)
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    _, inv, cnt = np.unique(np.sort(e, axis=1), axis=0, return_inverse=True, return_counts=True)
    return np.asarray(e[cnt[inv.ravel()] == 1], dtype=np.int64)


def _shortest(adj: dict[int, list[tuple[int, float]]], a: int, b: int) -> list[int] | None:
    """Shortest path a -> b along directed edges (Dijkstra), or None."""
    import heapq

    dist, prev, heap = {a: 0.0}, {a: -1}, [(0.0, a)]
    while heap:
        d, u = heapq.heappop(heap)
        if u == b:
            path = [b]
            while prev[path[-1]] != -1:
                path.append(prev[path[-1]])
            return path[::-1]
        if d > dist.get(u, np.inf):
            continue
        for w, ln in adj.get(u, []):
            nd = d + ln
            if nd < dist.get(w, np.inf):
                dist[w], prev[w] = nd, u
                heapq.heappush(heap, (nd, w))
    return None


def outline_paths(world: F64, faces: I64, outline_world: F64) -> list[list[int]]:
    """For every outline segment (vertex i -> i + 1 of ``world_outline_m``) the boundary vertices of the placed mesh it
    runs through, both ends included. Each outline vertex is a boundary vertex of the mesh; between two of them the
    path is the shortest along the boundary edges, which decides the only places with a choice: a pinch vertex where
    the outline touches itself (the shirt sleeve's placket slit, drafted 0.1 mm wide). Every boundary edge is passed
    exactly once, or the mesh does not belong to the outline."""
    be = boundary_edges(faces)
    verts = np.unique(be)
    d, k = cKDTree(world[verts]).query(outline_world)
    if float(d.max()) > MATCH_M:
        raise SimulationError(
            f"outline vertex {int(d.argmax())} is {d.max() * 1000:.3f} mm off the placed mesh's boundary "
            "(the alignment result and its mesh do not belong together)"
        )
    ov = verts[k]
    ln = np.linalg.norm(world[be[:, 0]] - world[be[:, 1]], axis=1)
    m = len(ov)
    for edges in (be, be[:, ::-1]):  # the outline runs with the faces' boundary direction, or against it
        adj: dict[int, list[tuple[int, float]]] = {}
        for (a, b), w in zip(edges.tolist(), ln.tolist(), strict=True):
            adj.setdefault(a, []).append((b, w))
        paths: list[list[int]] = []
        for i in range(m):
            a, b = int(ov[i]), int(ov[(i + 1) % m])
            p = [a] if a == b else _shortest(adj, a, b)
            if p is None:
                break
            paths.append(p)
        else:
            passed = sum(len(p) - 1 for p in paths)
            if passed == len(be):
                return paths
    raise SimulationError(
        "the outline vertices (world_outline_m) do not walk the placed mesh's boundary once round "
        "(the alignment result and its mesh do not belong together)"
    )


def _cotan_laplacian(v: F64, f: I64) -> coo_matrix:
    """Cotangent weights w_ij (symmetric, off-diagonal) of the triangle mesh ``v, f``."""
    rows, cols, vals = [], [], []
    for k in range(3):
        i, j, o = f[:, (k + 1) % 3], f[:, (k + 2) % 3], f[:, k]
        a, b = v[i] - v[o], v[j] - v[o]
        cross = np.linalg.norm(np.cross(a, b), axis=1)
        cot = np.einsum("ij,ij->i", a, b) / np.maximum(cross, 1e-18)
        rows += [i, j]
        cols += [j, i]
        vals += [0.5 * cot, 0.5 * cot]
    n = len(v)
    return coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))


def outline_arcs(outline: F64) -> tuple[F64, float]:
    """Arc position of every outline vertex from vertex 0 (the bundle's ``outline_s_m`` frame) and the perimeter."""
    seg = np.linalg.norm(np.roll(outline, -1, axis=0) - outline, axis=1)
    return np.r_[0.0, np.cumsum(seg)[:-1]], float(seg.sum())


def flat_coords(world: F64, faces: I64, outline_flat: F64, outline_world: F64) -> tuple[F64, I64, F64]:
    """(flat coordinates (N,2) of every vertex of a placed mesh, its boundary loop in outline order (a pinch vertex
    twice), the outline arc of every loop vertex) (see the module docstring)."""
    paths = outline_paths(world, faces, outline_world)
    o_arc, per = outline_arcs(outline_flat)
    o_len = np.linalg.norm(np.roll(outline_flat, -1, axis=0) - outline_flat, axis=1)
    m = len(paths)
    loop_l: list[int] = []
    flat_l: list[F64] = []
    arc_l: list[F64] = []
    for i, p in enumerate(paths):
        idx = np.asarray(p, dtype=np.int64)
        cum = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(world[idx], axis=0), axis=1))]
        t = cum / max(float(cum[-1]), 1e-15)
        p0, p1 = outline_flat[i], outline_flat[(i + 1) % m]
        loop_l += p[:-1]  # the end is the next segment's start
        flat_l.append(p0[None] + t[:-1, None] * (p1 - p0)[None])
        arc_l.append(o_arc[i] + t[:-1] * o_len[i])
    loop = np.asarray(loop_l, dtype=np.int64)
    uv = np.zeros((len(world), 2))
    uv[loop] = np.vstack(flat_l)
    arc_b = np.concatenate(arc_l)
    # degenerate triangles (the slit's 0.1 mm sliver) carry no cotangent weight; a vertex left in none is solved last
    f = np.asarray(faces, dtype=np.int64)
    a, b, c = world[f[:, 0]], world[f[:, 1]], world[f[:, 2]]
    good = f[0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1) > 1e-12]
    fixed = np.unique(loop)  # a pinch vertex is on the loop twice, at one flat position
    interior = np.setdiff1d(np.unique(good), fixed)
    if len(interior):
        w = _cotan_laplacian(world, good).tocsr()
        deg = np.asarray(w.sum(axis=1)).ravel()
        lap = (coo_matrix((deg, (np.arange(len(world)), np.arange(len(world)))), shape=w.shape).tocsr() - w).tocsc()
        lii = lap[interior][:, interior].tocsc()
        lib = lap[interior][:, fixed]
        lu = splu(lii)
        for k in range(2):
            uv[interior, k] = lu.solve(-(lib @ uv[fixed, k]))
    solved = np.union1d(fixed, interior)
    rest = np.setdiff1d(np.arange(len(world)), solved)
    if len(rest):  # in degenerate triangles only, or in none: start from the nearest solved vertex
        uv[rest] = uv[solved[cKDTree(world[solved]).query(world[rest])[1]]]
    free = np.setdiff1d(np.unique(f), fixed)
    e, lw = edge_lengths(world, good)
    if len(free) and float(np.abs(np.linalg.norm(uv[e[:, 0]] - uv[e[:, 1]], axis=1) - lw).max()) > ARAP_TRIGGER_M:
        uv = _arap(world, f, uv, fixed, free)
    return uv, loop, np.mod(arc_b, per)


def _arap(world: F64, f: I64, uv: F64, fixed: I64, free: I64, iters: int = 60, tol: float = 2e-7) -> F64:
    """As-rigid-as-possible refinement of flat coordinates (Liu et al. 2008, local/global) with uniform edge weights,
    the boundary held. For a developable, isometric mesh its minimum is the exact flat mesh whatever the triangle
    shapes; the cotangent solve that starts it is ill-conditioned on slivers (the alignment engine's fallback mesh of
    a self-touching outline, such as the shirt sleeve's)."""
    from scipy.sparse import csr_matrix

    a, b, c = world[f[:, 0]], world[f[:, 1]], world[f[:, 2]]
    e1 = b - a
    l1 = np.maximum(np.linalg.norm(e1, axis=1), 1e-15)
    x1 = e1 / l1[:, None]
    e2 = c - a
    px = np.einsum("ij,ij->i", e2, x1)
    py = np.linalg.norm(e2 - px[:, None] * x1, axis=1)
    loc = np.zeros((len(f), 3, 2))  # each triangle laid flat in its own frame
    loc[:, 1, 0], loc[:, 2, 0], loc[:, 2, 1] = l1, px, py
    t = uv[f]
    sa = (t[:, 1, 0] - t[:, 0, 0]) * (t[:, 2, 1] - t[:, 0, 1]) - (t[:, 1, 1] - t[:, 0, 1]) * (t[:, 2, 0] - t[:, 0, 0])
    if np.sum(sa < 0) > np.sum(sa > 0):  # the faces wind clockwise in the flat frame: mirror the local frames
        loc[:, :, 1] *= -1
    he = [(0, 1), (1, 2), (2, 0)]
    i_all = np.concatenate([f[:, i] for i, _ in he])
    j_all = np.concatenate([f[:, j] for _, j in he])
    d_loc = np.concatenate([loc[:, j] - loc[:, i] for i, j in he])  # (3T,2)
    tri_of = np.tile(np.arange(len(f)), 3)
    n = len(world)
    ones = np.ones(len(i_all))
    adj = csr_matrix((np.r_[ones, ones], (np.r_[i_all, j_all], np.r_[j_all, i_all])), shape=(n, n))
    lap = (csr_matrix((np.asarray(adj.sum(axis=1)).ravel(), (np.arange(n), np.arange(n))), shape=(n, n)) - adj).tocsc()
    lu = splu(lap[free][:, free].tocsc())
    lib = lap[free][:, fixed]
    nt = len(f)
    for _ in range(iters):
        d_uv = uv[j_all] - uv[i_all]
        # M = sum of d_uv d_loc^T per triangle; the best rotation of a 2x2 M has angle atan2(m10 - m01, m00 + m11)
        m = [np.bincount(tri_of, d_uv[:, p] * d_loc[:, q], minlength=nt) for p in (0, 1) for q in (0, 1)]
        th = np.arctan2(m[2] - m[1], m[0] + m[3])
        co, si = np.cos(th)[tri_of], np.sin(th)[tri_of]
        rd = np.column_stack([co * d_loc[:, 0] - si * d_loc[:, 1], si * d_loc[:, 0] + co * d_loc[:, 1]])
        rhs = np.column_stack(
            [np.bincount(j_all, rd[:, k], minlength=n) - np.bincount(i_all, rd[:, k], minlength=n) for k in (0, 1)]
        )
        new = uv.copy()
        for k in range(2):
            new[free, k] = lu.solve(rhs[free, k] - lib @ uv[fixed, k])
        step = float(np.abs(new - uv).max())
        uv = new
        if step < tol:
            break
    return uv


def isometric_fit(x0: F64, uv: F64, f: I64, anchor: float = 0.02, iters: int = 40) -> F64:
    """``x0`` (N,3) moved as little as possible so that every triangle has its flat shape ``uv`` (3D as-rigid-as-
    possible: per triangle the best orthonormal 3x2 map of its flat edges, then a global solve anchored to ``x0``).
    The solver takes each triangle's rest shape from its start position, so the start must be the pattern."""
    from scipy.sparse import csr_matrix, identity

    he = [(0, 1), (1, 2), (2, 0)]
    i_all = np.concatenate([f[:, i] for i, _ in he])
    j_all = np.concatenate([f[:, j] for _, j in he])
    d = uv[j_all] - uv[i_all]  # (3T,2) flat edge vectors
    tri_of = np.tile(np.arange(len(f)), 3)
    n, nt = len(x0), len(f)
    ones = np.ones(len(i_all))
    adj = csr_matrix((np.r_[ones, ones], (np.r_[i_all, j_all], np.r_[j_all, i_all])), shape=(n, n))
    lap = csr_matrix((np.asarray(adj.sum(axis=1)).ravel(), (np.arange(n), np.arange(n))), shape=(n, n)) - adj
    lu = splu((lap + anchor * identity(n, format="csr")).tocsc())
    x = x0.copy()
    for _ in range(iters):
        e = x[j_all] - x[i_all]  # (3T,3)
        m = np.zeros((nt, 3, 2))
        for a in range(3):
            for b in range(2):
                m[:, a, b] = np.bincount(tri_of, e[:, a] * d[:, b], minlength=nt)
        u, _, vt = np.linalg.svd(m, full_matrices=False)  # m = U S V^T, U (3x2): the best map is U V^T
        r = np.einsum("tij,tjk->tik", u, vt)
        rd = np.einsum("tij,tj->ti", r[tri_of], d)
        rhs = np.column_stack(
            [np.bincount(j_all, rd[:, k], minlength=n) - np.bincount(i_all, rd[:, k], minlength=n) for k in range(3)]
        )
        new = np.column_stack([lu.solve(rhs[:, k] + anchor * x0[:, k]) for k in range(3)])
        step = float(np.abs(new - x).max())
        x = new
        if step < 1e-7:
            break
    return x


def edge_lengths(v: F64, f: I64) -> tuple[I64, F64]:
    e = np.unique(np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1), axis=0)
    return e, np.linalg.norm(v[e[:, 0]] - v[e[:, 1]], axis=1)


def flat_error(world: F64, uv: F64, faces: I64) -> float:
    """Largest relative edge length difference between the placed mesh and its flat coordinates."""
    e, lw = edge_lengths(world, faces)
    lf = np.linalg.norm(uv[e[:, 0]] - uv[e[:, 1]], axis=1)
    return float(np.max(np.abs(lw - lf) / np.maximum(lf, 1e-12)))


def _signed_area(p: F64) -> float:
    return float(0.5 * np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1]))


def _resample(outline: F64, h: float, corner_deg: float = CORNER_DEG) -> tuple[F64, F64]:
    """The closed outline resampled at about ``h`` between its corners (vertices turning more than ``corner_deg``,
    kept), with the outline arc of every point (the bundle's ``outline_s_m`` frame). A lower ``corner_deg`` keeps
    more of the outline's vertices (for an outline whose resampled chords would cut across a small notch)."""
    arcs, per = outline_arcs(outline)
    keep = np.r_[True, np.linalg.norm(np.diff(outline, axis=0), axis=1) > 1e-7]
    p, pa = outline[keep], arcs[keep]
    if len(p) > 1 and np.linalg.norm(p[0] - p[-1]) <= 1e-7:
        p, pa = p[:-1], pa[:-1]
    a, b = p - np.roll(p, 1, axis=0), np.roll(p, -1, axis=0) - p
    ang = np.degrees(np.abs(np.arctan2(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0], np.einsum("ij,ij->i", a, b))))
    corners = np.flatnonzero(ang > corner_deg)
    if len(corners) == 0:
        corners = np.array([0])
    out: list[F64] = []
    out_s: list[F64] = []
    n = len(p)
    for i, c in enumerate(corners):
        nxt = corners[(i + 1) % len(corners)]
        span = (nxt - c) % n or n
        idx = (c + np.arange(span + 1)) % n
        run = p[idx]
        cum = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(run, axis=0), axis=1))]
        cnt = max(1, int(np.ceil(cum[-1] / h - 0.25)))
        s = np.linspace(0.0, cum[-1], cnt + 1)[:-1]
        out.append(np.column_stack([np.interp(s, cum, run[:, 0]), np.interp(s, cum, run[:, 1])]))
        out_s.append(np.mod(pa[c] + s, per))
    pts, s_all = np.vstack(out), np.concatenate(out_s)
    # neighbours closer than 1 mm (a 0.1 mm stretch of hem between two corners) are one point
    d = np.linalg.norm(pts - np.roll(pts, 1, axis=0), axis=1)
    far = d > 1e-3
    far[0] = True
    return pts[far], s_all[far]


def _arc_dist(a: F64, b: F64, per: float) -> F64:
    d = np.abs(a - b) % per
    return np.minimum(d, per - d)


def _open_gaps(p: F64, s: F64, per: float, gap: float) -> tuple[F64, int]:
    """Points of a counter-clockwise boundary closer than ``gap`` to a part of the boundary that is far along the
    outline (the two sides of a slit drafted 0.1 mm wide) move inward, both sides, until the gap is ``gap``. The
    solver refuses a cloth that starts closer to itself than its contact gap. Returns the points and how many moved."""
    moved = np.zeros(len(p), dtype=bool)
    for _ in range(6):
        a, b = p, np.roll(p, -1, axis=0)
        t = b - a
        ln = np.maximum(np.linalg.norm(t, axis=1), 1e-15)
        u = np.clip(np.einsum("pij,ij->pi", p[:, None, :] - a[None], t) / (ln**2)[None], 0.0, 1.0)
        q = a[None] + u[..., None] * t[None]
        dist = np.linalg.norm(p[:, None, :] - q, axis=2)
        near_arc = np.minimum(_arc_dist(s[:, None], s[None, :], per), _arc_dist(s[:, None], np.roll(s, -1)[None], per))
        dist[near_arc < 3 * gap] = np.inf  # the point's own neighbourhood along the outline
        need = gap - dist.min(axis=1)
        push = need > 1e-6
        if not push.any():
            break
        tn = t / ln[:, None]
        nrm_e = np.column_stack([-tn[:, 1], tn[:, 0]])  # inward for a counter-clockwise loop
        nrm = nrm_e + np.roll(nrm_e, 1, axis=0)
        nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-15)
        p = p.copy()
        p[push] += (0.5 * need[push] + 1e-5)[:, None] * nrm[push]
        moved |= push
    return p, int(moved.sum())


def min_angle_deg(v: F64, f: I64) -> float:
    """Smallest angle of the triangles ``f`` of the flat mesh ``v``."""
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    out = np.inf
    for p, q, r in ((a, b, c), (b, c, a), (c, a, b)):
        u, w = q - p, r - p
        cos = np.einsum("ij,ij->i", u, w) / np.maximum(np.linalg.norm(u, axis=1) * np.linalg.norm(w, axis=1), 1e-30)
        out = min(out, float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))).min()))
    return out


MIN_ANGLE_DEG = 12.0  # a simulation mesh with a smaller angle (a sliver the strain limiter cannot move) is redone


def sim_mesh(outline: F64, h: float, lines: list[F64] | None = None) -> tuple[F64, I64, F64, int]:
    """(vertices (N,2), triangles (M,3) counter-clockwise, outline arc of every boundary vertex, number of boundary
    vertices moved to open a slit) of a flat outline in the bundle's frame: a conforming Delaunay triangulation with
    edges of about ``h``; the first ``len(arcs)`` vertices are the boundary loop (counter-clockwise).

    The boundary keeps the outline's corners; when the resampled boundary is not a simple polygon (a chord across a
    small notch) or the mesh has a sliver (three almost straight boundary points), more of the outline's vertices
    are kept (corner threshold 25, 12, 5, 0 degrees) and the best mesh is used.

    ``lines`` (flat polylines inside the piece: the legs and the fold line of a dart drawn on the piece) become edges
    of the mesh, so the cloth can fold along them as a sewn dart folds into a flap."""
    outline = np.asarray(outline, dtype=np.float64)
    best: tuple[float, tuple[F64, I64, F64, int]] | None = None
    for corner in (CORNER_DEG, 12.0, 5.0, -1.0):
        try:
            mesh = _triangulate(outline, h, corner, [np.array(x) for x in lines or []])
        except SimulationError:
            continue
        q = min_angle_deg(mesh[0], mesh[1])
        if best is None or q > best[0]:
            best = (q, mesh)
        if q >= MIN_ANGLE_DEG:
            break
    if best is None:
        raise SimulationError("the outline is not a simple polygon (after resampling and opening its slits)")
    return best[1]


def _cut_spikes(p: F64, s: F64, max_deg: float = 15.0, max_m: float = GAP_MIN_M) -> tuple[F64, F64]:
    """Boundary vertices at the tip of a needle (an angle under ``max_deg`` that stands less than ``max_m`` off the
    chord of its neighbours: the collar stand's 2.8 x 0.4 mm button-end tip) are dropped: such a tip can only be
    meshed with a sliver, which the solver's strain limiter cannot move."""
    while len(p) > 3:
        u, w = np.roll(p, 1, axis=0) - p, np.roll(p, -1, axis=0) - p
        cos = np.einsum("ij,ij->i", u, w) / np.maximum(np.linalg.norm(u, axis=1) * np.linalg.norm(w, axis=1), 1e-30)
        ang = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
        a, b = np.roll(p, 1, axis=0), np.roll(p, -1, axis=0)
        t = np.clip(np.einsum("ij,ij->i", p - a, b - a) / np.maximum(np.einsum("ij,ij->i", b - a, b - a), 1e-30), 0, 1)
        off = np.linalg.norm(p - (a + t[:, None] * (b - a)), axis=1)
        bad = np.flatnonzero((ang < max_deg) & (off < max_m))
        if not len(bad):
            break
        k = bad[np.argmin(ang[bad])]
        p, s = np.delete(p, k, axis=0), np.delete(s, k)
    return p, s


def _prepare_lines(lines: list[F64], bnd: F64, poly: Polygon, h: float) -> list[F64]:
    """Lines inside a piece (dart legs and fold lines) as polylines with points at most ``h`` apart. An end on a
    boundary vertex is snapped onto it; points closer than 0.4 h to the boundary otherwise are dropped (a vertex that
    close to an edge it is not on would make a sliver)."""
    out = []
    tree = cKDTree(bnd)
    for line in lines:
        p = np.asarray(line, dtype=np.float64)
        dense = [p[0]]
        for a, b in zip(p[:-1], p[1:], strict=True):
            n = max(1, int(np.ceil(np.linalg.norm(b - a) / h)))
            dense += [a + (b - a) * k / n for k in range(1, n + 1)]
        q = np.asarray(dense)
        d_v, i_v = tree.query(q)
        snapped = d_v < 1e-6
        q[snapped] = bnd[i_v[snapped]]
        near = shapely.distance(poly.exterior, shapely.points(q)) < 0.4 * h
        q = q[snapped | ~near]
        if len(q) >= 2:
            out.append(q)
    return out


def _triangulate(outline: F64, h: float, corner_deg: float, lines: list[F64] | None = None) -> tuple[F64, I64, F64, int]:
    _, per = outline_arcs(outline)
    bnd, arcs = _resample(outline, h, corner_deg)
    if _signed_area(bnd) < 0:
        bnd, arcs = bnd[::-1].copy(), arcs[::-1].copy()
    bnd, arcs = _cut_spikes(bnd, arcs)
    bnd, opened = _open_gaps(bnd, arcs, per, GAP_MIN_M)
    poly = Polygon(bnd)
    if not poly.is_valid:
        raise SimulationError("the resampled outline is not a simple polygon")
    cons = _prepare_lines(lines or [], bnd, poly, h)
    lo, hi = bnd.min(axis=0), bnd.max(axis=0)
    dy = h * np.sqrt(3.0) / 2.0
    ys = np.arange(lo[1] + dy / 2, hi[1], dy)
    grid = []
    for r, y in enumerate(ys):
        xs = np.arange(lo[0] + (h / 2 if r % 2 else 0.0), hi[0], h)
        grid.append(np.column_stack([xs, np.full(len(xs), y)]))
    pts = np.vstack(grid) if grid else np.zeros((0, 2))
    if len(pts):
        inside = shapely.contains_xy(poly, pts[:, 0], pts[:, 1])
        pts = pts[inside]
        far = shapely.distance(poly.exterior, shapely.points(pts)) > 0.55 * h
        pts = pts[far]
        if cons and len(pts):  # keep the grid off the inner lines, as off the outline
            far = shapely.distance(shapely.MultiLineString([c.tolist() for c in cons]), shapely.points(pts)) > 0.55 * h
            pts = pts[far]
    for _ in range(16):
        # the inner lines' points after the boundary's (a point shared by lines, such as a dart apex, once)
        nb = len(bnd)
        btree = cKDTree(bnd)
        extra: dict[tuple[float, float], int] = {}
        cidx: list[list[int]] = []
        for c in cons:
            ids = []
            for q in c:
                d, i = btree.query(q)
                if d < 1e-9:
                    ids.append(int(i))
                    continue
                key = (round(float(q[0]), 9), round(float(q[1]), 9))
                ids.append(nb + extra.setdefault(key, len(extra)))
            cidx.append(ids)
        cpts = np.array(list(extra), dtype=np.float64).reshape(-1, 2)
        allp = np.vstack([bnd, cpts, pts])
        tri = Delaunay(allp).simplices.astype(np.int64)
        e = np.sort(np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]), axis=1)
        have = {(int(a), int(b)) for a, b in e}
        missing = [i for i in range(nb) if (min(i, (i + 1) % nb), max(i, (i + 1) % nb)) not in have]
        miss_c = False
        for k, ids in enumerate(cidx):  # split the inner-line segments the triangulation lacks
            c = cons[k]
            add = [j for j in range(len(ids) - 1) if (min(ids[j], ids[j + 1]), max(ids[j], ids[j + 1])) not in have]
            if add:
                miss_c = True
                cons[k] = np.insert(c, [j + 1 for j in add], 0.5 * (c[add] + c[[j + 1 for j in add]]), axis=0)
        if not missing and not miss_c:
            break
        if not missing:
            continue
        nb_pts: list[F64] = []
        nb_arc: list[float] = []
        for i in range(nb):
            nb_pts.append(bnd[i])
            nb_arc.append(float(arcs[i]))
            if i in missing:  # split the boundary edge the triangulation lacks (a conforming Delaunay mesh)
                j = (i + 1) % nb
                ds = ((arcs[j] - arcs[i] + per / 2) % per) - per / 2
                nb_pts.append(0.5 * (bnd[i] + bnd[j]))
                nb_arc.append(float(np.mod(arcs[i] + ds / 2, per)))
        bnd, arcs = np.asarray(nb_pts), np.asarray(nb_arc)
    else:
        raise SimulationError("the simulation mesh does not conform to the outline after 16 refinements")
    cent = allp[tri].mean(axis=1)
    keep = shapely.contains_xy(Polygon(bnd), cent[:, 0], cent[:, 1])
    tri = tri[keep]
    a, b, c = allp[tri[:, 0]], allp[tri[:, 1]], allp[tri[:, 2]]
    area = 0.5 * ((b - a)[:, 0] * (c - a)[:, 1] - (b - a)[:, 1] * (c - a)[:, 0])
    tri = np.where((area < 0)[:, None], tri[:, [0, 2, 1]], tri)
    tri = tri[np.abs(area) > 1e-12]
    used = np.unique(tri)
    remap = np.full(len(allp), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    if (remap[: len(bnd)] < 0).any():
        raise SimulationError("a boundary vertex of the simulation mesh is in no triangle")
    return allp[used], remap[tri], arcs, opened


def locate_arc(arcs: F64, per: float, s: F64) -> tuple[I64, I64, F64]:
    """For outline arcs ``s``: the boundary edge of the simulation mesh (vertex ``i`` -> ``j``, the loop being the
    first ``len(arcs)`` vertices) and the fraction ``t`` along it. Points on the outline are found by their arc, never
    by position: the two sides of a slit lie 0.1 mm apart in the pattern."""
    n = len(arcs)
    order = np.argsort(arcs)
    sa = arcs[order]
    s = np.mod(np.asarray(s, dtype=np.float64), per)
    pos = np.searchsorted(sa, s, side="right") - 1  # -1: before the smallest arc, i.e. on the wrap-around edge
    lo = order[pos % n]
    hi = order[(pos + 1) % n]
    a0 = sa[pos % n] - np.where(pos < 0, per, 0.0)
    a1 = sa[(pos + 1) % n] + np.where(pos + 1 >= n, per, 0.0)
    t = np.clip((s - a0) / np.maximum(a1 - a0, 1e-15), 0.0, 1.0)
    return lo, hi, t


def locate2d(v: F64, f: I64, p: F64, k: int = 12) -> tuple[I64, F64]:
    """Triangle (index into ``f``) and barycentric weights of each flat point ``p`` (N,2); points outside the mesh go
    to the closest point of the nearest triangle."""
    tri = v[f]
    cand = cKDTree(tri.mean(axis=1)).query(p, k=min(k, len(f)))[1]
    cand = np.asarray(cand).reshape(len(p), -1)
    a, b, c = tri[cand, 0], tri[cand, 1], tri[cand, 2]
    q = p[:, None, :]
    v0, v1, v2 = b - a, c - a, q - a
    den = v0[..., 0] * v1[..., 1] - v0[..., 1] * v1[..., 0]
    den = np.where(np.abs(den) < 1e-30, 1e-30, den)
    w1 = (v2[..., 0] * v1[..., 1] - v2[..., 1] * v1[..., 0]) / den
    w2 = (v0[..., 0] * v2[..., 1] - v0[..., 1] * v2[..., 0]) / den
    bary = np.stack([1.0 - w1 - w2, w1, w2], axis=-1)
    # clamp onto the triangle and measure how far the point is from it
    cl = np.clip(bary, 0.0, None)
    cl /= np.maximum(cl.sum(axis=-1, keepdims=True), 1e-30)
    proj = cl[..., 0:1] * a + cl[..., 1:2] * b + cl[..., 2:3] * c
    dist = np.linalg.norm(proj - q, axis=-1)
    best = dist.argmin(axis=1)
    r = np.arange(len(p))
    return cand[r, best], cl[r, best]

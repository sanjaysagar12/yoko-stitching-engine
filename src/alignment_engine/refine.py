"""Refine a piece mesh with interior vertices (a conforming, well shaped triangulation of the outline). Metres.

The pieces from ``sm2d_engine`` are triangulated from the outline only, so a few huge triangles span the whole piece;
when
such a mesh is wrapped, only the outline moves and every interior triangle stays a flat chord that cuts through the
body.
``refine_piece`` adds interior vertices (a regular hexagonal grid) and long outline edges are split (collinear points,
the boundary itself does not change). The triangulation is the Delaunay triangulation of boundary and interior points,
restricted to the outline; outline edges that Delaunay does not contain are split until all of them are present
(a conforming Delaunay triangulation), so there are no T-junctions.

Why this method: ``scipy.spatial.Delaunay`` (Qhull) and ``shapely`` ship Windows wheels, so no compiler is needed. A
constrained-Delaunay library such as ``triangle`` has to be compiled; ``mapbox_earcut`` cannot add vertices. A plain
midpoint subdivision of the earcut triangles (what the first wrapping code used) gives 35 000-vertex meshes of slivers.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import shapely
from numpy.typing import NDArray
from scipy.spatial import Delaunay
from shapely.geometry import Polygon

from .errors import BundleError

F64 = NDArray[np.float64]
I64 = NDArray[np.int64]
_BOUNDARY_FACTOR = 0.75  # outline edges are split into pieces of at most this fraction of max_edge
_GRID_FACTOR = 0.7  # spacing of the interior grid, as a fraction of max_edge
_MARGIN_FACTOR = 0.5  # interior points keep this many grid spacings away from the outline


@dataclass(frozen=True)
class RefinedPiece:
    """A refined piece. ``vertices`` (N,3, z = 0) start with the boundary loop (counter-clockwise), then interior
    points.

    ``boundary`` indexes the boundary loop in order (original outline vertices plus the split points),
    ``original_index``
    the original outline vertices inside ``vertices``. ``bundle`` is the input bundle with ``outline_m`` replaced by the
    refined boundary (same points, plus the split points).
    """

    bundle: dict[str, Any]
    vertices: F64
    faces: I64
    boundary: I64
    original_index: I64
    fallback: bool = False  # True: invalid (self-touching) outline, midpoint subdivision of the source mesh was used


def _signed_area(p: F64) -> float:
    return float(0.5 * np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1]))


def _densify(outline: F64, step: float) -> tuple[F64, I64]:
    """Split long outline edges; returns (points, index of each original vertex in them)."""
    pts: list[F64] = []
    orig: list[int] = []
    for a, b in zip(outline, np.roll(outline, -1, axis=0), strict=True):
        orig.append(len(pts))
        pts.append(a)
        n = int(np.ceil(np.linalg.norm(b - a) / step))
        for k in range(1, n):
            pts.append(a + (b - a) * (k / n))
    return np.array(pts), np.array(orig, dtype=np.int64)


def refine_piece(
    bundle: dict[str, Any], max_edge_m: float = 0.01, fallback_mesh: tuple[F64, I64] | None = None
) -> RefinedPiece:
    """Triangulate the outline of ``bundle`` with every edge at most ``max_edge_m`` (+ a few per cent) long.

    Outline vertices stay exactly where they are; no change to the boundary, the area or the named points. An
    outline that
    touches itself (the placket slit of ``FullSleeve``) is not a valid polygon; with ``fallback_mesh`` (the source mesh)
    its triangles are subdivided at the midpoints of long edges instead (``fallback=True``), otherwise BundleError.
    """
    outline = np.asarray(bundle["outline_m"], dtype=np.float64)
    if len(outline) < 3 or abs(_signed_area(outline)) < 1e-12:
        raise BundleError(f"piece {bundle.get('name', '?')!r}: the outline is degenerate, cannot refine")
    if _signed_area(outline) < 0:
        raise BundleError(f"piece {bundle.get('name', '?')!r}: the outline must be counter-clockwise")
    poly = Polygon(outline)
    if not poly.is_valid:
        if fallback_mesh is None:
            raise BundleError(f"piece {bundle.get('name', '?')!r}: the outline is not a valid polygon")
        return _fallback(bundle, outline, fallback_mesh, max_edge_m)
    bnd, orig = _densify(outline, _BOUNDARY_FACTOR * max_edge_m)

    g = _GRID_FACTOR * max_edge_m
    x0, y0, x1, y1 = poly.bounds
    ys = np.arange(y0, y1 + g, g * np.sqrt(3) / 2)
    cand: list[F64] = []
    for r, y in enumerate(ys):
        xs = np.arange(x0 + (g / 2 if r % 2 else 0.0), x1 + g, g)
        cand.append(np.column_stack([xs, np.full_like(xs, y)]))
    grid = np.vstack(cand)
    inside = shapely.contains_xy(poly, grid[:, 0], grid[:, 1])
    grid = grid[inside]
    if len(grid):
        far = shapely.distance(poly.exterior, shapely.points(grid)) >= _MARGIN_FACTOR * g
        grid = grid[far]

    for _ in range(8):  # conforming Delaunay: split outline edges that the triangulation does not contain
        pts = np.vstack([bnd, grid])
        tri = Delaunay(pts).simplices
        c = pts[tri].mean(axis=1)
        keep = shapely.contains_xy(poly, c[:, 0], c[:, 1])
        tri = tri[keep]
        a, b, c3 = pts[tri[:, 0]], pts[tri[:, 1]], pts[tri[:, 2]]
        area2 = (b[:, 0] - a[:, 0]) * (c3[:, 1] - a[:, 1]) - (c3[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1])
        tri = np.where((area2 < 0)[:, None], tri[:, [0, 2, 1]], tri)
        tri = tri[np.abs(area2) > 1e-14]
        e = np.sort(np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]), axis=1)
        have = {(int(p), int(q)) for p, q in e}
        nb = len(bnd)
        missing = [i for i in range(nb) if tuple(sorted((i, (i + 1) % nb))) not in have]
        loop_edges = {tuple(sorted((i, (i + 1) % nb))) for i in range(nb)}
        long_ = [
            (int(p), int(q))
            for p, q in have
            if (p, q) not in loop_edges and np.linalg.norm(pts[p] - pts[q]) > max_edge_m
        ]
        if not missing and not long_:
            break
        if long_:  # Delaunay refinement: add the midpoints of interior edges that are too long
            mids = np.array([0.5 * (pts[p] + pts[q]) for p, q in long_])
            mids = mids[shapely.contains_xy(poly, mids[:, 0], mids[:, 1])]
            if len(mids):  # a midpoint close to the outline would make a sliver: skip those
                mids = mids[shapely.distance(poly.exterior, shapely.points(mids)) >= 0.3 * g]
            if len(mids):
                grid = np.vstack([grid, mids]) if len(grid) else mids
        if not missing:
            continue
        order: list[tuple[int, F64]] = []
        for i in missing:
            order.append((i, 0.5 * (bnd[i] + bnd[(i + 1) % nb])))
        # insert the midpoints into the boundary loop (back to front keeps the indices valid)
        new_bnd = list(bnd)
        shift = 0
        for i, m in sorted(order):
            new_bnd.insert(i + 1 + shift, m)
            shift += 1
        orig = np.array([int(o + np.sum([1 for i, _ in order if i < o])) for o in orig], dtype=np.int64)
        bnd = np.array(new_bnd)
    else:
        raise BundleError(f"piece {bundle.get('name', '?')!r}: could not make the triangulation conform to the outline")

    # Laplacian smoothing of the interior points (the outline is fixed) improves the angles; keep it only if
    # still conforming
    for _ in range(6):
        if len(pts) == len(bnd):
            break
        nbr_sum = np.zeros_like(pts)
        cnt = np.zeros(len(pts))
        for a, b in ((0, 1), (1, 2), (2, 0)):
            np.add.at(nbr_sum, tri[:, a], pts[tri[:, b]])
            np.add.at(nbr_sum, tri[:, b], pts[tri[:, a]])
            np.add.at(cnt, tri[:, a], 1)
            np.add.at(cnt, tri[:, b], 1)
        new_pts = pts.copy()
        new_pts[len(bnd) :] = 0.5 * pts[len(bnd) :] + 0.5 * (nbr_sum / np.maximum(cnt, 1)[:, None])[len(bnd) :]
        d = Delaunay(new_pts).simplices
        c = new_pts[d].mean(axis=1)
        d = d[shapely.contains_xy(poly, c[:, 0], c[:, 1])]
        a_, b_, c_ = new_pts[d[:, 0]], new_pts[d[:, 1]], new_pts[d[:, 2]]
        ar = (b_[:, 0] - a_[:, 0]) * (c_[:, 1] - a_[:, 1]) - (c_[:, 0] - a_[:, 0]) * (b_[:, 1] - a_[:, 1])
        d = np.where((ar < 0)[:, None], d[:, [0, 2, 1]], d)
        d = d[np.abs(ar) > 1e-14]
        ee = np.sort(np.concatenate([d[:, [0, 1]], d[:, [1, 2]], d[:, [2, 0]]]), axis=1)
        have2 = {(int(p), int(q)) for p, q in ee}
        nb2 = len(bnd)
        longest = max(np.linalg.norm(new_pts[p] - new_pts[q]) for p, q in have2)
        if (
            longest <= max_edge_m
            and all(tuple(sorted((i, (i + 1) % nb2))) in have2 for i in range(nb2))
            and len(np.unique(d)) == len(new_pts)
        ):
            pts, tri = new_pts, d
        else:
            break

    verts = np.column_stack([pts, np.zeros(len(pts))])
    out = dict(bundle)
    out["outline_m"] = [[float(x), float(y)] for x, y in bnd]
    return RefinedPiece(out, verts, tri.astype(np.int64), np.arange(len(bnd), dtype=np.int64), orig)


def _fallback(bundle: dict[str, Any], outline: F64, mesh: tuple[F64, I64], max_edge_m: float) -> RefinedPiece:
    from .placement import refine_mesh

    v0 = mesh[0].copy()
    v0[: len(outline), :2] = outline  # the OBJ is rounded to 1 micrometre, the bundle outline is exact
    v, f = refine_mesh(v0, mesh[1], max_edge_m)
    n = len(outline)  # refine_mesh appends, so the first n vertices are the outline vertices
    de = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])  # directed edges of counter-clockwise triangles
    have = {(int(a), int(b)) for a, b in de}
    nxt: dict[int, list[int]] = {}
    for a, b in have:
        if (b, a) not in have:  # a free edge, directed along the counter-clockwise boundary
            nxt.setdefault(a, []).append(b)
    loop: list[int] = []
    for i in range(n):  # original outline edge i -> i+1 became a chain of collinear free edges
        goal = (i + 1) % n
        cur = i
        loop.append(cur)
        while cur != goal and len(loop) <= len(v):
            cands = nxt.get(cur, [])
            if not cands:
                break
            d = v[goal, :2] - v[cur, :2]
            cur = max(
                cands,
                key=lambda c: float((v[c, :2] - v[cur, :2]) @ d) / (np.linalg.norm(v[c, :2] - v[cur, :2]) + 1e-18),
            )
            if cur != goal:
                loop.append(cur)
    out = dict(bundle)
    out["outline_m"] = [[float(x), float(y)] for x, y in v[loop, :2]]
    return RefinedPiece(out, v, f, np.array(loop, dtype=np.int64), np.arange(n, dtype=np.int64), True)


class RefineCache:
    """Refined meshes cached by the content of the outline and ``max_edge_m``: in memory, and in ``directory``
    (npz) if given."""

    def __init__(self, directory: Path | None = None) -> None:
        self.directory = Path(directory) if directory else None
        self._mem: dict[str, RefinedPiece] = {}
        self.hits = 0

    @staticmethod
    def key(bundle: dict[str, Any], max_edge_m: float) -> str:
        h = hashlib.sha1(np.asarray(bundle["outline_m"], dtype=np.float64).tobytes())
        h.update(f"{max_edge_m:.9f}|v1".encode())
        return h.hexdigest()[:20]

    def get(
        self, bundle: dict[str, Any], max_edge_m: float = 0.01, fallback_mesh: tuple[F64, I64] | None = None
    ) -> RefinedPiece:
        """The refined piece, from memory, from disk or freshly computed."""
        k = self.key(bundle, max_edge_m)
        if k in self._mem:
            self.hits += 1
            return self._mem[k]
        path = self.directory / f"{k}.npz" if self.directory else None
        if path is not None and path.is_file():
            z = np.load(path, allow_pickle=False)
            r = RefinedPiece(
                {**bundle, "outline_m": z["vertices"][z["boundary"], :2].tolist()},
                z["vertices"],
                z["faces"],
                z["boundary"],
                z["original_index"],
                bool(z["fallback"]),
            )
        else:
            r = refine_piece(bundle, max_edge_m, fallback_mesh)
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez(
                    path,
                    vertices=r.vertices,
                    faces=r.faces,
                    boundary=r.boundary,
                    original_index=r.original_index,
                    fallback=np.array(r.fallback),
                )
        self._mem[k] = r
        return r

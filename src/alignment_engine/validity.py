"""Checks run after every solve, on the placed meshes, independent of the solver. Metres.

* ``size_check``: the placed piece against its flat pattern (edges, outline, area, named-point distances). A change of
  more than ``SIZE_TOLERANCE_PCT`` raises ``SizeChangedError`` unless distortion was explicitly allowed.
* ``layout_check``: exact distances of every piece to the avatar and of every pair of pieces (samples of one mesh:
  vertices plus the midpoint of every edge, so at most 2.5 mm apart on 5 mm meshes, against the exact triangles of the
  other, both ways).
* ``pair_violations`` / ``concentric_warnings``: mirror pairs are symmetric and side by side; side-by-side pieces of
  one ring share its radius (up to a layer gap).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .errors import SizeChangedError
from .verify import AvatarSurface, MeshProx

F64 = NDArray[np.float64]
I64 = NDArray[np.int64]
SIZE_TOLERANCE_PCT = 0.05
OUTLINE_STEP = 0.002  # outline and named-point segments are measured on 3D samples this far apart (flat)
TOLERANCE = 0.0005  # a distance this far below its requirement is a violation
REPORT_RANGE = 0.06  # pairs closer than this get an exact distance in the report


def unique_edges(faces: I64) -> I64:
    """Undirected edges (E,2) of a triangle mesh, sorted."""
    e = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    n = int(faces.max()) + 1 if len(faces) else 1
    code = np.unique(e[:, 0] * n + e[:, 1])
    return np.column_stack([code // n, code % n]).astype(np.int64)


def _area(v: F64, f: I64) -> float:
    t = v[f]
    if t.shape[2] == 2:
        u, w = t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]
        return float(0.5 * np.abs(u[:, 0] * w[:, 1] - u[:, 1] * w[:, 0]).sum())
    return float(0.5 * np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1).sum())


def _polyline_points(a: F64, b: F64) -> tuple[F64, I64]:
    """Samples every ``OUTLINE_STEP`` along segments a->b (both (N,2)); returns points and the segment of each."""
    n = np.maximum(np.ceil(np.linalg.norm(b - a, axis=1) / OUTLINE_STEP).astype(int), 1)
    seg = np.repeat(np.arange(len(a)), n + 1)
    t = np.concatenate([np.linspace(0.0, 1.0, k + 1) for k in n])
    return a[seg] + (b[seg] - a[seg]) * t[:, None], seg


def size_check(
    flat: F64,
    faces: I64,
    world: F64,
    transform: Callable[[F64], F64],
    outline: F64,
    named: F64,
    edges: I64 | None = None,
) -> dict[str, Any]:
    """Size of a placed piece against the flat pattern, in per cent (all >= 0).

    ``flat`` (N,2) and ``world`` (N,3) are the same mesh flat and placed; ``transform`` maps piece points to world (the
    same mapping as the mesh); ``outline`` (M,2) the pattern outline; ``named`` (K,2) the named points. Measured: every
    mesh edge (3D chord against the flat length), the outline length (3D samples every 2 mm), the area (3D triangles),
    and for every pair of named points the length of the placed image of the straight segment between them (a
    geodesic on a developable surface, so an isometry keeps it exactly).
    """
    e = unique_edges(faces) if edges is None else edges
    fl = np.linalg.norm(flat[e[:, 0]] - flat[e[:, 1]], axis=1)
    wl = np.linalg.norm(world[e[:, 0]] - world[e[:, 1]], axis=1)
    edge = float((np.abs(wl / np.maximum(fl, 1e-12) - 1.0)).max() * 100.0)
    closed = np.vstack([outline, outline[:1]])
    pts, _ = _polyline_points(closed[:-1], closed[1:])
    flat_per = float(np.linalg.norm(np.diff(closed, axis=0), axis=1).sum())
    w = transform(pts)
    seg_len = np.linalg.norm(np.diff(w, axis=0), axis=1)
    # consecutive samples of different segments are the same point (segment ends): their distance is 0
    per = float(seg_len.sum())
    area_flat = _area(flat, faces)
    area = _area(world, faces)
    named_pct = 0.0
    if len(named) > 1:
        i, j = np.triu_indices(len(named), 1)
        d_flat = np.linalg.norm(named[i] - named[j], axis=1)
        ok = d_flat > 0.005
        if ok.any():
            p, seg = _polyline_points(named[i[ok]], named[j[ok]])
            wp = transform(p)
            step = np.linalg.norm(np.diff(wp, axis=0), axis=1)
            same = seg[1:] == seg[:-1]
            lengths = np.bincount(seg[1:][same], weights=step[same], minlength=int(ok.sum()))
            named_pct = float((np.abs(lengths / d_flat[ok] - 1.0)).max() * 100.0)
    max_mesh_edge = float(fl.max()) if len(fl) else 0.0
    mismatch = float(np.abs(transform(flat) - world).max()) if len(flat) else 0.0
    return {
        "max_edge_change_pct": edge,
        "perimeter_change_pct": abs(per / flat_per - 1.0) * 100.0,
        "area_change_pct": abs(area / area_flat - 1.0) * 100.0,
        "named_point_change_pct": named_pct,
        "max_mesh_edge_m": max_mesh_edge,
        "mesh_map_mismatch_m": mismatch,
    }


def enforce_size(name: str, check: dict[str, Any], allowed: bool) -> bool:
    """True if the piece changed size; raises ``SizeChangedError`` when that is not ``allowed``."""
    worst = max(
        check["max_edge_change_pct"],
        check["perimeter_change_pct"],
        check["area_change_pct"],
        check["named_point_change_pct"],
    )
    changed = bool(worst > SIZE_TOLERANCE_PCT or check["mesh_map_mismatch_m"] > 1e-7)
    if changed and not allowed:
        raise SizeChangedError(name, check, SIZE_TOLERANCE_PCT)
    return changed


@dataclass
class Placed:
    """What the layout check needs of a placed piece."""

    name: str
    world: F64
    faces: I64
    edges: I64
    normal: Callable[[F64], F64] | None  # outward normal at world points (None: from the triangles)
    cylinder: tuple[F64, F64, float, float] | None = None  # (axis point, axis direction, radius, sagitta) if rolled


@dataclass
class LayoutCheck:
    """Exact distances of a layout. Pair keys are indices into the list of pieces."""

    body: dict[int, tuple[float, F64]]  # piece -> (smallest signed distance to the avatar, where)
    pairs: dict[tuple[int, int], tuple[float, F64, bool]]  # (i, j) -> (distance, where, facing)


def fine_samples(p: Placed) -> F64:
    """Vertices plus the midpoint of every edge."""
    return np.vstack([p.world, 0.5 * (p.world[p.edges[:, 0]] + p.world[p.edges[:, 1]])])


def layout_check(
    pieces: list[Placed],
    body: AvatarSurface,
    clearance: float,
    adjacency: list[I64],
    cache: dict[Any, Any] | None = None,
    keys: list[bytes] | None = None,
    required: float = np.inf,
) -> LayoutCheck:
    """Exact distances to the avatar and between pieces (pairs closer than ``REPORT_RANGE`` by bounding box).

    Two phases keep it exact and fast: the distance from each sample to the nearest vertex of the other mesh is an
    upper bound of its true distance and exceeds it by at most one edge length, so only samples whose nearest-vertex
    distance is within one edge length of the smallest one can attain the minimum; those get the exact point-triangle
    distance. ``cache`` (optional) remembers results by the content ``keys`` of the pieces involved, so unchanged
    pieces and pairs are not measured again.
    """
    cache = cache if cache is not None else {}
    samples = [fine_samples(p) for p in pieces]
    boxes = [(p.world.min(axis=0), p.world.max(axis=0)) for p in pieces]
    out = LayoutCheck({}, {})
    slack = 0.012  # samples of the avatar are 6 mm apart: the nearest one is at most this much farther
    bound = clearance + slack  # farther than this is reported as this (a lower bound): only near points matter
    for i, s in enumerate(samples):
        ck = ("body", keys[i]) if keys else None
        if ck is not None and ck in cache:
            out.body[i] = cache[ck]
            continue
        res = _body_min(body, s, bound, slack, clearance)
        out.body[i] = res
        if ck is not None:
            cache[ck] = res
    edge_max = [float(np.linalg.norm(p.world[p.edges[:, 0]] - p.world[p.edges[:, 1]], axis=1).max()) for p in pieces]
    prox: dict[int, MeshProx] = {}

    def get_prox(b: int) -> MeshProx:
        if b not in prox:
            ck = ("prox", keys[b]) if keys else None
            if ck is not None and ck in cache:
                prox[b] = cache[ck]
            else:
                prox[b] = MeshProx(pieces[b].world, pieces[b].faces, adjacency[b])
                if ck is not None:
                    cache[ck] = prox[b]
        return prox[b]

    for i in range(len(pieces)):
        for j in range(i + 1, len(pieces)):
            lo = np.maximum(boxes[i][0], boxes[j][0])
            hi = np.minimum(boxes[i][1], boxes[j][1])
            if np.any(lo - hi > REPORT_RANGE):
                continue
            ck2 = ("pair", keys[i], keys[j]) if keys else None
            if ck2 is not None and ck2 in cache:
                if cache[ck2] is not None:
                    out.pairs[(i, j)] = cache[ck2]
                continue
            best: tuple[float, F64, bool] = (np.inf, np.zeros(3), False)
            lower = _coaxial_bound(pieces[i], pieces[j])
            if lower is not None and lower >= required:  # rolled on one axis: farther apart than required, exactly
                best = (float(lower), 0.5 * (pieces[i].world[0] + pieces[j].world[0]), True)
                out.pairs[(i, j)] = best
                if ck2 is not None:
                    cache[ck2] = best
                continue
            for a, b in ((i, j), (j, i)):
                pts = samples[a]
                pts = pts[np.all((pts > boxes[b][0] - REPORT_RANGE) & (pts < boxes[b][1] + REPORT_RANGE), axis=1)]
                if not len(pts):
                    continue
                d, k, q, tri = get_prox(b).min_distance(pts, REPORT_RANGE + edge_max[b])
                if d < best[0]:
                    p = pts[k]
                    normal = pieces[b].normal
                    if normal is not None:
                        nb = normal(q[None, :])[0]
                    else:
                        t = pieces[b].world[pieces[b].faces[tri]]
                        nb = np.cross(t[1] - t[0], t[2] - t[0])
                        nb = nb / max(float(np.linalg.norm(nb)), 1e-18)
                    facing = bool(d < 1e-3 or abs(float((p - q) @ nb)) > 0.8 * d)
                    best = (d, 0.5 * (p + q), facing)
            res2 = best if np.isfinite(best[0]) else None
            if res2 is not None:
                out.pairs[(i, j)] = res2
            if ck2 is not None:
                cache[ck2] = res2
    return out


def _body_min(body: AvatarSurface, s: F64, bound: float, slack: float, clearance: float) -> tuple[float, F64]:
    """Smallest signed distance of samples ``s`` to the avatar (exact below ``bound``; ``bound`` itself means "at
    least that far"). Points inside the body count, also deep ones.

    As for pieces: a point at distance ``D`` has an avatar sample within ``sqrt(D^2 + c^2)`` (``c`` the cover radius of
    the samples), so only points whose nearest sample is within ``sqrt(U^2 + c^2)`` of the best exact value ``U`` are
    measured exactly; points on the inner side of their nearest triangle always are.
    """
    d1, i1 = body.tree.query(s, distance_upper_bound=bound, workers=-1)
    near1 = np.isfinite(d1)
    if (~near1).any():
        deep, _ = body.inside(s[~near1])
        if deep.any():
            k = int(np.flatnonzero(~near1)[np.flatnonzero(deep)[0]])
            return (-bound, s[k])
    if not near1.any():
        return (bound, s[0])
    owner_t = body.owner[np.where(near1, i1, 0)]
    inside = near1 & (np.einsum("ij,ij->i", s - body.tri[owner_t, 0], body.normals[owner_t]) < 0)
    dn = np.where(near1, d1, np.inf)
    first = np.argsort(dn, kind="stable")[:64]
    first = first[np.isfinite(dn[first])]
    sd0, _, nr0 = body.closest(s[first], bound)
    u = float(np.min(np.where(nr0, sd0, bound)))
    cand = np.flatnonzero(inside | (dn**2 <= max(u, 0.0) ** 2 + body.cover**2 + 1e-12))
    cand = np.union1d(cand, first)
    sd, _, near = body.closest(s[cand], float(dn[cand].max()) + slack)
    k = int(np.argmin(np.where(near, sd, np.inf)))
    return (float(sd[k]), s[cand][k])


PAIR_TOLERANCE = 0.001  # the halves of a mirror pair agree in radius, axial shift and mirrored slide to 1 mm
CONCENTRIC_SLACK = 0.005  # side-by-side pieces of one ring may differ in radius by gap_layer plus this


def arc(world: F64, axis_point: F64, axis_dir: F64) -> tuple[float, float]:
    """(start, width) in radians of the smallest arc about the axis that holds every point (wrap-around aware)."""
    a = axis_dir / np.linalg.norm(axis_dir)
    front = np.array([0.0, 0.0, 1.0]) - a[2] * a
    if np.linalg.norm(front) < 1e-6:
        front = np.array([0.0, 1.0, 0.0]) - a[1] * a
    front /= np.linalg.norm(front)
    q = world - axis_point
    ang = np.sort(np.arctan2(q @ np.cross(a, front), q @ front))
    gaps = np.diff(np.append(ang, ang[0] + 2.0 * np.pi))
    k = int(np.argmax(gaps))
    return float(ang[(k + 1) % len(ang)]), float(2.0 * np.pi - gaps[k])


def arc_overlap(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Angle (radians) shared by two arcs ``(start, width)``."""
    best = 0.0
    for k in (-1, 0, 1):
        s = b[0] + 2.0 * np.pi * k
        best = max(best, min(a[0] + a[1], s + b[1]) - max(a[0], s))
    return best


@dataclass
class PairView:
    """One half of a mirror pair: its parameters ``(r, s, t)`` and, when rolled, its arc and radius."""

    name: str
    theta: F64
    arc: tuple[float, float] | None


def pair_violations(pairs: list[tuple[PairView, PairView]]) -> list[dict[str, Any]]:
    """Mirror pairs must be one rigid pair: equal radius and axial shift, mirrored slide (``PAIR_ASYMMETRY``), and side
    by side, never stacked: their arcs about the ring axis may not overlap (``PAIR_STACKED``)."""
    out: list[dict[str, Any]] = []
    for a, b in pairs:
        dev = float(max(abs(a.theta[0] - b.theta[0]), abs(a.theta[2] - b.theta[2]), abs(a.theta[1] + b.theta[1])))
        if dev > PAIR_TOLERANCE:
            out.append(
                {
                    "kind": "PAIR_ASYMMETRY",
                    "pieces": [a.name, b.name],
                    "value_m": round(dev, 6),
                    "required_m": PAIR_TOLERANCE,
                    "at_m": None,
                    "limit": "pair",
                }
            )
        if a.arc is not None and b.arc is not None:
            shared = arc_overlap(a.arc, b.arc) * float(min(a.theta[0], b.theta[0]))
            if shared > PAIR_TOLERANCE:
                out.append(
                    {
                        "kind": "PAIR_STACKED",
                        "pieces": [a.name, b.name],
                        "value_m": round(shared, 6),
                        "required_m": 0.0,
                        "at_m": None,
                        "limit": "pair",
                    }
                )
    return out


def concentric_warnings(
    ring: str, pieces: list[tuple[str, float, tuple[float, float], tuple[float, float]]], gap_layer: float
) -> list[str]:
    """``NOT_CONCENTRIC`` for pieces of one ring that lie side by side (arcs apart, heights overlapping) but differ in
    radius by more than ``gap_layer`` + ``CONCENTRIC_SLACK``. ``pieces``: (name, radius, arc, (y min, y max))."""
    out: list[str] = []
    for i, (na, ra, aa, ya) in enumerate(pieces):
        for nb, rb, ab, yb in pieces[i + 1 :]:
            if min(ya[1], yb[1]) <= max(ya[0], yb[0]) or arc_overlap(aa, ab) > 0.0:
                continue
            if abs(ra - rb) > gap_layer + CONCENTRIC_SLACK:
                out.append(
                    f"NOT_CONCENTRIC: {na} (r {ra * 100:.1f} cm) and {nb} (r {rb * 100:.1f} cm) lie side by side on "
                    f"the {ring} ring but {abs(ra - rb) * 100:.1f} cm apart radially"
                )
    return out


def _coaxial_bound(a: Placed, b: Placed) -> float | None:
    """Lower bound of the distance between two pieces rolled on the same axis: every vertex of a piece lies on its
    cylinder and every triangle within its chord sagitta inside it, so the meshes are at least ``|r_a - r_b|`` minus
    the sagittas apart. None if they are not coaxial cylinders."""
    if a.cylinder is None or b.cylinder is None:
        return None
    pa, da, ra, sa = a.cylinder
    pb, db, rb, sb = b.cylinder
    if np.linalg.norm(np.cross(da, db)) > 1e-9:
        return None
    d = pb - pa
    if np.linalg.norm(d - (d @ da) * da) > 1e-9:
        return None
    return float(abs(ra - rb) - (sb if rb > ra else sa))

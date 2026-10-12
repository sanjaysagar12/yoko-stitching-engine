"""Plane slices of a triangle mesh and the ordered rings they define. Metres; pure functions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

F64 = NDArray[np.float64]
I64 = NDArray[np.int64]


@dataclass(frozen=True)
class Ring:
    """Closed loop of points, each on a mesh edge: ``p = v[ia] + t * (v[ib] - v[ia])`` (linear in the vertices).

    The edge ids and weights are fixed once on the template; for any body shape the ring is the same loop of the
    mesh, so its length is cheap to evaluate and changes smoothly with the shape.
    """

    ia: I64
    ib: I64
    t: F64

    def points(self, v: F64) -> F64:
        """Ring points (K,3) for vertices ``v``."""
        return v[self.ia] + self.t[:, None] * (v[self.ib] - v[self.ia])

    def length(self, v: F64) -> float:
        """Closed polyline length through the ring points, metres."""
        p = self.points(v)
        return float(np.linalg.norm(np.roll(p, -1, axis=0) - p, axis=1).sum())

    def mirrored(self, mirror_map: I64) -> Ring:
        """The same loop on the other side of x = 0 (vertex ids mapped through ``mirror_map``)."""
        return Ring(mirror_map[self.ia], mirror_map[self.ib], self.t.copy())


class MeshTopology:
    """Unique edges and face-edge incidence of a triangle mesh (computed once)."""

    def __init__(self, faces: I64) -> None:
        e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
        e = np.sort(e, axis=1)
        self.edges, inv = np.unique(e, axis=0, return_inverse=True)
        self.face_edges = inv.reshape(3, len(faces)).T

    def slice(self, v: F64, point: F64, normal: F64) -> list[Ring]:
        """All closed loops where the plane (``point``, unit ``normal``) cuts the mesh."""
        s = (v - point) @ normal
        s[s == 0.0] = 1e-12
        a, b = self.edges[:, 0], self.edges[:, 1]
        sa, sb = s[a], s[b]
        cross = sa * sb < 0
        t_edge = np.where(cross, sa / np.where(cross, sa - sb, 1.0), 0.0)
        fc = cross[self.face_edges]
        two = np.flatnonzero(fc.sum(axis=1) == 2)
        adj: dict[int, list[int]] = {}
        for f in two:
            e0, e1 = self.face_edges[f][fc[f]]
            adj.setdefault(int(e0), []).append(int(e1))
            adj.setdefault(int(e1), []).append(int(e0))
        seen: set[int] = set()
        loops: list[Ring] = []
        for start in sorted(adj):
            if start in seen:
                continue
            chain, cur, prev = [start], start, -1
            seen.add(start)
            while True:
                nxt = [n for n in adj[cur] if n != prev]
                if not nxt:
                    break
                cur, prev = nxt[0], cur
                if cur == start:
                    break
                seen.add(cur)
                chain.append(cur)
            if len(chain) >= 3:
                ids = np.array(chain)
                loops.append(Ring(a[ids].astype(np.int64), b[ids].astype(np.int64), t_edge[ids]))
        return loops

    def loop_around(self, v: F64, point: F64, normal: F64, enclose: F64) -> Ring | None:
        """The largest slice loop that encloses ``enclose`` (a point on the plane), or None."""
        e1 = np.cross(normal, [1.0, 0.0, 0.0] if abs(normal[0]) < 0.9 else [0.0, 1.0, 0.0])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(normal, e1)
        best: tuple[float, Ring] | None = None
        for ring in self.slice(v, point, normal):
            p = ring.points(v)
            xy = np.column_stack([(p - enclose) @ e1, (p - enclose) @ e2])
            if not _contains_origin(xy):
                continue
            area = abs(0.5 * float(np.sum(xy[:, 0] * np.roll(xy[:, 1], -1) - np.roll(xy[:, 0], -1) * xy[:, 1])))
            if best is None or area > best[0]:
                best = (area, ring)
        return None if best is None else best[1]


def _contains_origin(xy: F64) -> bool:
    """Even-odd test whether the origin is inside the closed polygon ``xy``."""
    a, b = xy, np.roll(xy, -1, axis=0)
    cond = (a[:, 1] > 0) != (b[:, 1] > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        xs = a[:, 0] + (0 - a[:, 1]) * (b[:, 0] - a[:, 0]) / (b[:, 1] - a[:, 1])
    return bool(np.count_nonzero(cond & (0 < xs)) % 2 == 1)

"""Named-point chains on a placed piece's outline -> ordered 3D polylines (and the flat piece-space curve).

A piece's outline is a cycle of its on-outline named points, ordered by their arc position ``outline_s_m`` in the
bundle. A chain ``[p0, ..., pk]`` walks that cycle from ``p0`` to ``pk`` the way that passes every listed point in
order (wrap-around and zero-length edges included); if both ways do, the chain is ambiguous unless it lists exactly
the points of one way.

Geometry: arc length is measured on the flat outline (the truth: pieces are never stretched). World positions come
from the alignment result, never from the placed mesh vertices: ``world_outline_m`` (every outline vertex, same index
as the bundle's ``outline_m``) and ``world_edges`` polylines (arc-length samples between named points). Both are the
placement's own mapping of exact outline points; between these knots the curve is interpolated linearly, so a point
at any arc length is found to sub-vertex accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .errors import StitchingError

F64 = NDArray[np.float64]
ARC_EPS = 1e-9


def _outline_arcs(p: F64) -> tuple[F64, float]:
    seg = np.linalg.norm(np.roll(p, -1, axis=0) - p, axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)[:-1]]), float(seg.sum())


@dataclass
class Piece:
    """A placed piece: cyclic knots (flat arc s, world xyz, flat xy) and its outline named points."""

    name: str
    per: float  # flat perimeter (m)
    knot_s: F64
    knot_w: F64
    knot_f: F64
    s_of: dict[str, float]  # on-outline named point -> arc position
    cycle: list[str]  # on-outline named points in outline order
    edge_ids: dict[tuple[str, str], str]  # (from, to) of the alignment result's world_edges -> id
    world_named: dict[str, F64]
    off_outline: list[str]
    winding: int = 1  # +1: the stored flat outline runs counter-clockwise (interior on the left), -1 clockwise
    flat_named: dict[str, F64] | None = None  # every named point (on and off the outline), flat

    def at(self, s: F64) -> tuple[F64, F64]:
        """World and flat points at arc positions ``s`` (any real; taken modulo the perimeter)."""
        s = np.mod(np.asarray(s, dtype=np.float64), self.per)
        ks = np.concatenate([self.knot_s, [self.knot_s[0] + self.per]])
        kw = np.vstack([self.knot_w, self.knot_w[:1]])
        kf = np.vstack([self.knot_f, self.knot_f[:1]])
        w = np.column_stack([np.interp(s, ks, kw[:, d]) for d in range(3)])
        f = np.column_stack([np.interp(s, ks, kf[:, d]) for d in range(2)])
        return w, f

    def edge_id(self, a: str, b: str) -> str:
        return self.edge_ids.get((a, b)) or self.edge_ids.get((b, a)) or f"{a}-{b}"

    def edges(self) -> list[tuple[str, str, str, float]]:
        """Every outline edge between consecutive named points: (id, from, to, flat length m)."""
        out = []
        n = len(self.cycle)
        for i in range(n):
            a, b = self.cycle[i], self.cycle[(i + 1) % n]
            ln = (self.s_of[b] - self.s_of[a]) % self.per if n > 1 else self.per
            out.append((self.edge_id(a, b), a, b, ln))
        return out


def build_piece(placement: dict[str, Any], bundle: dict[str, Any]) -> Piece:
    """The ``Piece`` of one placement of the alignment result and its piece bundle."""
    name = placement["piece"]
    flat = np.asarray(bundle["outline_m"], dtype=np.float64)
    world = np.asarray(placement["world_outline_m"], dtype=np.float64)
    if world.shape != (len(flat), 3):
        raise StitchingError(f"{name}: world_outline_m has {len(world)} points, the bundle outline {len(flat)}")
    cum, per = _outline_arcs(flat)
    s_of = {q["name"]: float(q["outline_s_m"]) % per for q in bundle["named_points"] if q["on_outline"]}
    s_list, w_list, f_list = [cum], [world], [flat]
    edge_ids: dict[tuple[str, str], str] = {}
    for e in placement.get("world_edges", []):
        edge_ids[(e["from"], e["to"])] = e["id"]
        if e["from"] not in s_of or e["to"] not in s_of:
            continue
        poly = np.asarray(e["polyline_m"], dtype=np.float64)
        s0, s1 = s_of[e["from"]], s_of[e["to"]]
        fwd = (s1 - s0) % per
        span = fwd if fwd <= per - fwd else -(per - fwd)  # the alignment engine samples the shorter way round
        ss = np.mod(s0 + np.linspace(0.0, span, len(poly)), per)
        s_list.append(ss)
        w_list.append(poly)
        f_list.append(np.asarray([[np.nan, np.nan]] * len(poly)))
    s_all = np.concatenate(s_list)
    w_all = np.vstack(w_list)
    f_all = np.vstack(f_list)
    order = np.lexsort((np.isnan(f_all[:, 0]), s_all))  # outline vertices first among equal arcs
    s_all, w_all, f_all = s_all[order], w_all[order], f_all[order]
    keep = np.r_[True, np.diff(s_all) > ARC_EPS]
    s_all, w_all, f_all = s_all[keep], w_all[keep], f_all[keep]
    # flat point of an edge sample: on the flat outline at its arc
    miss = np.isnan(f_all[:, 0])
    if miss.any():
        ks = np.concatenate([cum, [per]])
        kf = np.vstack([flat, flat[:1]])
        f_all[miss] = np.column_stack([np.interp(s_all[miss], ks, kf[:, d]) for d in range(2)])
    names = [q["name"] for q in bundle["named_points"] if q["on_outline"]]
    cycle = sorted(names, key=lambda n: (s_of[n], names.index(n)))
    world_named = {q["name"]: np.asarray(q["world_m"], dtype=np.float64) for q in placement["world_named_points"]}
    off = [q["name"] for q in bundle["named_points"] if not q["on_outline"]]
    area = float(np.sum(flat[:, 0] * np.roll(flat[:, 1], -1) - np.roll(flat[:, 0], -1) * flat[:, 1]))
    flat_named = {q["name"]: np.asarray(q["xy_m"], dtype=np.float64) for q in bundle["named_points"]}
    return Piece(
        name, per, s_all, w_all, f_all, s_of, cycle, edge_ids, world_named, off, 1 if area > 0 else -1, flat_named
    )


@dataclass
class Path:
    """A resolved chain: from arc ``s0`` walking ``direction`` (+1 along the outline, -1 against) for ``length``."""

    piece: Piece
    names: list[str]  # every named point passed, in sewing order
    s0: float
    direction: int
    length: float  # flat (true) length, m

    @property
    def interior(self) -> int:
        """+1: walking the chain, the piece lies on the left (flat, stored frame); -1: on the right."""
        return self.direction * self.piece.winding

    @property
    def edge_ids(self) -> list[str]:
        return [self.piece.edge_id(a, b) for a, b in zip(self.names[:-1], self.names[1:], strict=True)]

    trim: tuple[float, float] = (0.0, 0.0)  # cut off the start / end of the chain (m), as listed

    def reversed(self) -> Path:
        e = self.s0 + self.direction * self.length
        return Path(self.piece, self.names[::-1], e, -self.direction, self.length, self.trim[::-1])

    def trimmed(self, start: float, end: float) -> Path:
        """The chain without ``start`` m at its start and ``end`` m at its end (a seam ending between points)."""
        if start < 0 or end < 0 or start + end >= self.length - 1e-4:
            raise StitchingError(
                f"{self.piece.name}: trims {start * 100:.1f} + {end * 100:.1f} cm leave nothing of the "
                f"{self.length * 100:.1f} cm chain {self.names}"
            )
        return Path(
            self.piece, self.names, self.s0 + self.direction * start, self.direction, self.length - start - end,
            (self.trim[0] + start, self.trim[1] + end),
        )  # fmt: skip

    def at(self, t: F64) -> tuple[F64, F64]:
        """World and flat points at distances ``t`` (0..length) along the chain."""
        return self.piece.at(self.s0 + self.direction * np.asarray(t, dtype=np.float64))

    def knots_t(self) -> F64:
        """Distances along the chain of every knot inside it, plus both ends (sorted)."""
        t = np.mod(self.direction * (self.piece.knot_s - self.s0), self.piece.per)
        inner = t[(t > ARC_EPS) & (t < self.length - ARC_EPS)]
        return np.unique(np.concatenate([[0.0], inner, [self.length]]))

    def interval(self) -> tuple[float, float]:
        """The arc interval [start, start + length] it covers along the outline direction (start modulo perimeter)."""
        start = self.s0 if self.direction > 0 else self.s0 - self.length
        return float(np.mod(start, self.piece.per)), self.length


def resolve(piece: Piece, chain: list[str]) -> Path:
    """Resolve a named-point chain on ``piece`` (errors name the valid points)."""
    bad = [n for n in chain if n not in piece.s_of]
    if bad:
        off = [n for n in bad if n in piece.off_outline]
        why = f" ({off} are not on the outline)" if off else ""
        raise StitchingError(f"{piece.name}: unknown outline point(s) {bad}{why}; outline points: {piece.cycle}")
    if len(chain) < 2:
        raise StitchingError(f"{piece.name}: a chain needs at least two points, got {chain}")
    n = len(piece.cycle)
    i0, ik = piece.cycle.index(chain[0]), piece.cycle.index(chain[-1])
    ways: dict[int, list[str]] = {}
    for d in (1, -1):
        walk = [piece.cycle[i0]]
        i = i0
        while True:
            i = (i + d) % n
            walk.append(piece.cycle[i])
            if i == ik or len(walk) > n:
                break
        ways[d] = walk
    ok = {d: w for d, w in ways.items() if _subsequence(chain, w)}
    exact = [d for d, w in ok.items() if _solid(piece, w, chain) == chain]
    if len(exact) == 1:
        d = exact[0]
    elif len(ok) == 1:
        d = next(iter(ok))
    elif not ok:
        raise StitchingError(
            f"{piece.name}: chain {chain} is not contiguous along the outline (in order the outline runs "
            f"{' '.join(ways[1])} one way and {' '.join(ways[-1])} the other)"
        )
    else:
        raise StitchingError(
            f"{piece.name}: chain {chain} is ambiguous: both ways round from {chain[0]} to {chain[-1]} pass its "
            "points; "
            f"list a point between them ({ways[1][1:-1][:3]} one way, {ways[-1][1:-1][:3]} the other)"
        )
    walk = ways[d]
    s0, s1 = piece.s_of[walk[0]], piece.s_of[walk[-1]]
    length = (d * (s1 - s0)) % piece.per
    if walk[0] == walk[-1]:
        length = piece.per
    return Path(piece, walk, s0, d, float(length))


def _subsequence(sub: list[str], seq: list[str]) -> bool:
    it = iter(seq)
    return all(any(x == y for y in it) for x in sub)


def _solid(piece: Piece, walk: list[str], keep: list[str]) -> list[str]:
    """The walk without points that coincide with their neighbour (zero-length edges), unless listed in ``keep``."""

    def same(a: str, b: str) -> bool:
        return abs(piece.s_of[a] - piece.s_of[b]) <= ARC_EPS

    return [
        n
        for i, n in enumerate(walk)
        if n in keep or not ((i > 0 and same(n, walk[i - 1])) or (i + 1 < len(walk) and same(n, walk[i + 1])))
    ]


@dataclass
class Line:
    """A straight line (polyline) through named points inside a piece (a dart leg to its apex): flat exact; world
    linear between the named points' world positions, put back onto the placed surface by ``snap``."""

    piece: Piece
    names: list[str]
    flat: F64
    world: F64
    snap: Any = None
    trim: tuple[float, float] = (0.0, 0.0)
    direction: int = 0
    interior: int = 0
    edge_ids: list[str] = field(default_factory=list)

    @property
    def length(self) -> float:
        return float(np.linalg.norm(np.diff(self.flat, axis=0), axis=1).sum())

    def _arcs(self) -> F64:
        return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(self.flat, axis=0), axis=1))])

    def at(self, t: F64) -> tuple[F64, F64]:
        a = self._arcs()
        t = np.asarray(t, dtype=np.float64)
        f = np.column_stack([np.interp(t, a, self.flat[:, d]) for d in range(2)])
        w = np.column_stack([np.interp(t, a, self.world[:, d]) for d in range(3)])
        return (self.snap(w) if self.snap is not None else w), f

    def knots_t(self) -> F64:
        a = self._arcs()  # every ~5 mm, so the snapped world line follows the surface
        return np.unique(np.concatenate([a, np.linspace(0.0, a[-1], max(2, int(a[-1] / 0.005) + 1))]))

    def reversed(self) -> Line:
        return Line(self.piece, self.names[::-1], self.flat[::-1], self.world[::-1], self.snap)

    def interval(self) -> None:
        return None


def line(piece: Piece, names: list[str], snap: Any = None) -> Line:
    """A ``Line`` through named points of ``piece`` (on or off the outline)."""
    assert piece.flat_named is not None
    bad = [n for n in names if n not in piece.flat_named or n not in piece.world_named]
    if bad or len(names) < 2:
        raise StitchingError(
            f"{piece.name}: line {names}: unknown point(s) {bad}; named points: {sorted(piece.flat_named)}"
        )
    return Line(
        piece, list(names), np.array([piece.flat_named[n] for n in names]),
        np.array([piece.world_named[n] for n in names]), snap,
    )  # fmt: skip

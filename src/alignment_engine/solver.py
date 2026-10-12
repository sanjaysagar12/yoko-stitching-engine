"""Collision solver: moves pieces rigidly along their own size-exact surfaces until nothing touches. Metres.

Every piece has three parameters ``theta = (r, s, t)`` (``roll.Surface``): distance from the body (radius), shift
across and shift along. Any value is an isometry of the flat piece, so the solver can never change a size.

Constraints, all exact and recomputed from the current positions in every iteration:

* body: every sample of a piece is at least ``clearance`` from the avatar (exact point-triangle distance). Against its
  own body part (the region of its ring) a piece may move out; against another part (the torso under a sleeve cap, the
  arm beside a torso panel) it may only slide (``Piece.other_tris``): a larger radius cannot clear a part that is not
  under the cylinder's axis, it only lifts the piece off its point;
* pieces: **every pair** of pieces (any ring, any layer, fixed pieces too) is at least ``gap`` apart, measured from the
  samples of one piece to the exact triangles of the other, both ways. There is no layer exemption.

A violated constraint at sample ``p`` (closest point ``q`` on the other piece) asks for a displacement of ``p`` along a
direction: ``p - q`` when the pieces are side by side, or the other piece's outward normal when they face each other
(one lies over the other, or they cross). Which of two facing pieces is the outer one is decided once per pair: the
side most of the piece is already on, else the piece being placed (incremental), else the smaller piece. The
displacements are turned into parameter changes by a weighted Kaczmarz projection on the linearised constraints
(radial moves cost more than sliding), applied, clamped to the limits, and the loop repeats. Nothing is skipped: a
facing pair is separated radially.

Groups move together: the two copies of a centre-line or side pair share their parameters with ``s`` mirrored, so a
symmetric body gives a symmetric result. The copies are mirror images about the plane x = 0, so they are ``gap`` apart
exactly when every vertex of one copy is at least ``gap / 2`` from that plane on its own side (``Piece.pair_side``).
That bound replaces the collision rows between the two copies: it is enforced after every step by sliding the pair
outward (``s`` only, never the radius), so a pair never crosses its mirror plane (``pair_slide``).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from .roll import Surface
from .verify import AvatarSurface, MeshProx, closest_on_triangle

log = logging.getLogger(__name__)
F64 = NDArray[np.float64]
I64 = NDArray[np.int64]
RADIAL_WEIGHT = 3.0  # a radial move costs this much more than a slide of the same length
MAX_STEP = 0.015  # metres per parameter per iteration (stay where the linearisation holds)
ROWS_PER_PAIR = 10  # worst samples per pair and direction that enter one iteration
ROWS_PER_BODY = 16
SWEEPS = 6
PATIENCE = 8  # iterations without a 2 % smaller total deficit end a stage
MIN_DEFICIT = 1e-5  # a constraint short by less than this (0.01 mm, inside the margin) is met


@dataclass
class Piece:
    """One placed piece as the solver sees it. ``local`` are the mesh vertices in local coordinates (N,2)."""

    name: str
    order: int
    surface: Surface | None  # None: a fixed mesh without a size-exact surface (a distorting wrap, never moved)
    local: F64
    faces: I64
    adjacency: I64
    area: float
    theta: F64
    theta0: F64
    r_min: float
    r_max: float
    moving: bool
    iter_idx: I64
    group: int = -1
    sign: F64 = field(default_factory=lambda: np.ones(3))
    slide_weight: float = 1.0  # cost of a slide across (s); a centre-line pair keeps its gap by sliding very little
    x_mid: float = 0.0
    world: F64 = field(default_factory=lambda: np.zeros((0, 3)))
    tree2d: cKDTree | None = None
    prox: MeshProx | None = None
    limits_hit: set[str] = field(default_factory=set)
    other_tris: NDArray[np.bool_] | None = None  # avatar triangles of other body parts (slide away, never grow)
    pair_side: float = 0.0  # lead of a pair: the side of x = 0 (world, +1 / -1) the piece must stay on; 0: no pair
    pair_room: tuple[float, float] | None = None  # (needed, available) outward slide of a pair that has no room

    def update(self) -> None:
        """Map the mesh at the current parameters."""
        if self.surface is not None:
            self.world = self.surface.map(self.local, self.theta)

    def normal(self, world: F64, tri: I64) -> F64:
        """Outward normal at points of the piece (``tri``: the triangle each point lies on)."""
        if self.surface is not None:
            return self.surface.normal(world)
        t = self.world[self.faces[tri]]
        n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
        return np.asarray(n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-18))

    def bbox(self) -> tuple[F64, F64]:
        """World bounding box (min, max)."""
        return self.world.min(axis=0), self.world.max(axis=0)


@dataclass(frozen=True)
class Requirements:
    """What the layout must satisfy. ``gap_layer`` (None: equal to ``gap``) applies to pairs facing each other."""

    clearance: float
    gap: float
    gap_layer: float | None
    max_shift: float
    margin: float = 0.0003  # aim this far beyond the requirement (the final check tolerates 0.5 mm)

    def gap_for(self, facing: bool) -> float:
        """Required distance of a pair."""
        return self.gap_layer if (facing and self.gap_layer is not None) else self.gap


@dataclass
class SolveStats:
    """What a solve did."""

    iterations: int = 0
    converged: bool = False
    rows_last: int = 0


class Solver:
    """Iterative projection on exact distance constraints (see the module docstring)."""

    def __init__(self, pieces: list[Piece], body: AvatarSurface, req: Requirements) -> None:
        self.pieces, self.body, self.req = pieces, body, req
        self.side: dict[tuple[int, int], int] = {}  # (i, j) -> +1 if piece i is outside piece j (decided once)
        self.layered: set[tuple[int, int]] = set()  # coaxial pairs separated radially (|r_i - r_j| >= gap, exact)
        for p in pieces:
            if p.surface is None:
                if p.prox is None:
                    p.prox = MeshProx(p.world, p.faces, p.adjacency)
            elif p.tree2d is None:
                p.tree2d = cKDTree(p.local)
            if not len(p.world):
                p.update()
        self._sync_groups()
        self.half_gap = pair_half_gap(req)

    # -- groups ----------------------------------------------------------------------------------------------------
    def _lead(self, g: int) -> Piece:
        return next(p for p in self.pieces if p.group == g and np.all(p.sign > 0))

    def _sync_groups(self) -> None:
        for p in self.pieces:
            if p.group >= 0 and not np.all(p.sign > 0):
                lead = self._lead(p.group)
                p.theta = lead.theta * p.sign
                p.update()

    def _project_pairs(self, max_shift: float) -> None:
        """Part every moving pair whose halves are closer than the gap (see ``pair_slide``)."""
        for p in self.pieces:
            if p.moving and p.pair_side and p.surface is not None and p.group >= 0 and np.all(p.sign > 0):
                p.theta[1] = pair_slide(p, p.theta, max_shift, self.half_gap)
                p.update()
                for q in self.pieces:
                    if q.group == p.group and q is not p:
                        q.limits_hit, q.pair_room = set(p.limits_hit), p.pair_room
        self._sync_groups()

    def block(self, i: int) -> int:
        """Parameter block of piece ``i`` (see ``_key``)."""
        return self._key(i)

    def _key(self, i: int) -> int:
        """Parameter block of piece ``i``: ``group + 1`` for a group, ``-1 - i`` for a single piece (0 means none)."""
        p = self.pieces[i]
        return p.group + 1 if p.group >= 0 else -1 - i

    # -- queries ---------------------------------------------------------------------------------------------------
    def query(self, q: Piece, pts: F64, upper: float) -> tuple[F64, F64, I64, I64]:
        """(distance, closest point, nearest vertex, triangle) from world points to the triangles of piece ``q``.

        The candidate triangles are found in the developed (unrolled) coordinates of ``q``, where its KD-tree never
        changes; the distance itself is the exact 3D point-triangle distance.
        """
        if q.surface is None:
            assert q.prox is not None
            d, c, t, v = q.prox.query(pts, upper)
            return d, c, v, t
        n = len(pts)
        dist, close, vert = np.full(n, np.inf), np.zeros((n, 3)), np.full(n, -1, dtype=np.int64)
        tri_id = np.full(n, -1, dtype=np.int64)
        dev = q.surface.develop(pts, q.theta, q.x_mid)
        keep = np.flatnonzero(np.abs(dev[:, 2]) < upper)
        if not len(keep):
            return dist, close, vert, tri_id
        assert q.tree2d is not None
        d2, idx = q.tree2d.query(dev[keep, :2], k=3, distance_upper_bound=upper + 0.006)
        ok = np.isfinite(d2[:, 0])
        keep, d2, idx = keep[ok], d2[ok], idx[ok]
        if not len(keep):
            return dist, close, vert, tri_id
        vi = np.where(np.isfinite(d2), idx, idx[:, :1])
        cand = q.adjacency[vi].reshape(len(keep), -1)
        valid = cand >= 0
        safe = np.where(valid, cand, 0)
        tri = q.world[q.faces[safe.ravel()]]
        pp = np.repeat(pts[keep], cand.shape[1], axis=0)
        cq = closest_on_triangle(pp, tri).reshape(len(keep), cand.shape[1], 3)
        dd = np.where(valid, np.linalg.norm(pts[keep, None, :] - cq, axis=2), np.inf)
        j = dd.argmin(axis=1)
        rows = np.arange(len(keep))
        dist[keep], close[keep], vert[keep], tri_id[keep] = dd[rows, j], cq[rows, j], vi[:, 0], safe[rows, j]
        return dist, close, vert, tri_id

    def decide_side(self, i: int, j: int, off: F64) -> int:
        """+1 if piece ``i`` is to be outside piece ``j`` (decided once per pair, see the module docstring).

        The decision is about one subject: the piece being moved (else the smaller one). Pieces of one ring: the
        subject goes on top unless it is already further in. Other pairs: the side of the other piece's surface most of
        the subject lies on (median offset of all its vertices), so a sleeve whose cap dips under a torso panel still
        counts as outside it. ``off`` (offsets of the conflicting samples of ``i``) decides only when the other piece
        has no surface. Ties: the subject goes outside.
        """
        key = (min(i, j), max(i, j))
        if key not in self.side:
            a, b = self.pieces[i], self.pieces[j]
            if a.moving != b.moving:
                subj, other = (i, j) if a.moving else (j, i)
            elif abs(a.area - b.area) > 1e-9:
                subj, other = (i, j) if a.area < b.area else (j, i)
            else:
                subj, other = (i, j) if a.order > b.order else (j, i)
            ps, po = self.pieces[subj], self.pieces[other]
            if self.coaxial(subj, other):
                dr = float(ps.theta[0] - po.theta[0])
                out = -1 if (dr < -0.002 and ps.moving == po.moving) else 1
            elif po.surface is not None:
                pts = ps.world[ps.iter_idx] if len(ps.iter_idx) else ps.world
                med = float(np.median(po.surface.develop(pts, po.theta, po.x_mid)[:, 2]))
                out = -1 if med < -0.002 else 1
            else:
                med = float(np.median(off)) if len(off) else 0.0
                med = med if subj == i else -med
                out = -1 if med < -0.002 else 1
            s_subj = out  # +1: the subject is outside the other piece
            s_i = s_subj if subj == i else -s_subj
            self.side[key] = s_i if i == key[0] else -s_i
        s0 = self.side[key]
        return s0 if i == key[0] else -s0

    def coaxial(self, i: int, j: int) -> bool:
        """Both pieces roll on the same axis (then ``|r_i - r_j|`` is a lower bound of their distance)."""
        a, b = self.pieces[i].surface, self.pieces[j].surface
        if a is None or b is None or a.kind != "cylinder" or b.kind != "cylinder":
            return False
        if np.linalg.norm(np.cross(a.e_y, b.e_y)) > 1e-9:
            return False
        d = b.origin - a.origin
        return bool(np.linalg.norm(d - (d @ a.e_y) * a.e_y) < 1e-9)

    def layer_colliding(self) -> bool:
        """Turn every still-violated coaxial pair into a layered pair and restart its moving pieces from their
        anchors (slides are reset, radii kept). Returns whether anything changed."""
        new: list[tuple[int, int]] = []
        for i in range(len(self.pieces)):
            for j in range(i + 1, len(self.pieces)):
                a, b = self.pieces[i], self.pieces[j]
                if (i, j) in self.layered or not (a.moving or b.moving) or (a.group >= 0 and a.group == b.group):
                    continue
                if self.coaxial(i, j) and (self._pair_rows(i, j, False) or self._pair_rows(j, i, False)):
                    new.append((i, j))
        if not new:
            return False
        reset: set[int] = set()
        for i, j in new:
            self.layered.add((i, j))
            reset |= {k for k in (i, j) if self.pieces[k].moving}
        keys = {self._key(k) for k in reset}
        for k, p in enumerate(self.pieces):
            if self._key(k) in keys:
                p.theta[1:] = p.theta0[1:]
                p.limits_hit.clear()
                p.update()
        return True

    def _layer_rows(self, i: int, j: int) -> list[tuple[int, F64, int, F64, float]]:
        """The single exact constraint of a layered pair: the outer radius exceeds the inner one by the gap."""
        a, b = self.pieces[i], self.pieces[j]
        dr = float(a.theta[0] - b.theta[0])
        sigma = self.decide_side(i, j, np.array([dr])) if abs(dr) > 0.002 else self.decide_side(i, j, np.zeros(0))
        need = self.req.gap_for(True) + self.req.margin - sigma * dr
        if need <= MIN_DEFICIT:
            return []
        outer, inner = (i, j) if sigma > 0 else (j, i)
        po, pi = self.pieces[outer], self.pieces[inner]
        g = np.array([1.0, 0.0, 0.0])
        if po.moving:
            return [(self._key(outer), g * po.sign, 0, np.zeros(3), need)]
        return [(self._key(inner), -g * pi.sign, 0, np.zeros(3), need)]

    # -- constraints -----------------------------------------------------------------------------------------------
    def _grad(self, p: Piece, idx: I64, d: F64) -> F64:
        """Gradient (n,3) of ``d . point`` with respect to the parameters of the block of ``p``."""
        assert p.surface is not None
        jac = p.surface.jacobian(p.local[idx], p.theta)
        return np.asarray(np.einsum("nij,ni->nj", jac, d) * p.sign)

    def rows(self, fine: bool) -> list[tuple[int, F64, int, F64, float]]:
        """Linearised violated constraints: (block a, gradient a, block b or 0, gradient b, required increase)."""
        req = self.req
        out: list[tuple[int, F64, int, F64, float]] = []
        clear = req.clearance + req.margin
        for i, p in enumerate(self.pieces):
            if not p.moving:
                continue
            idx = np.arange(len(p.local)) if fine else p.iter_idx
            sd, dirs, near = self.body.closest(p.world[idx], clear + 0.002)
            bad = np.flatnonzero(near & (sd < clear - MIN_DEFICIT))
            if len(bad):
                bad = bad[np.argsort(sd[bad], kind="stable")[:ROWS_PER_BODY]]
                tri = self.body.last_tri[bad]
                g = self._grad(p, idx[bad], dirs[bad])
                if p.other_tris is not None:
                    g[(tri >= 0) & p.other_tris[np.maximum(tri, 0)], 0] = 0.0
                for k, row in enumerate(bad):
                    out.append((self._key(i), g[k], 0, np.zeros(3), float(clear - sd[row])))
        boxes = [p.bbox() for p in self.pieces]
        for i, a in enumerate(self.pieces):
            for j in range(i + 1, len(self.pieces)):
                b = self.pieces[j]
                if not (a.moving or b.moving) or a.group == b.group >= 0:  # a pair's halves: ``pair_slide``
                    continue
                gap = max(req.gap, req.gap_layer or 0.0) + req.margin
                lo = np.maximum(boxes[i][0], boxes[j][0])
                hi = np.minimum(boxes[i][1], boxes[j][1])
                if np.any(lo - hi > gap):
                    continue
                if (i, j) in self.layered:
                    out += self._layer_rows(i, j)
                    continue
                out += self._pair_rows(i, j, fine)
                out += self._pair_rows(j, i, fine)
        return out

    def _pair_rows(self, i: int, j: int, fine: bool) -> list[tuple[int, F64, int, F64, float]]:
        """Constraints from the samples of piece ``i`` against the triangles of piece ``j``."""
        req = self.req
        p, q = self.pieces[i], self.pieces[j]
        idx = np.arange(len(p.local)) if fine else p.iter_idx
        pts = p.world[idx]
        upper = max(req.gap, req.gap_layer or 0.0) + req.margin
        lo, hi = q.bbox()
        inside = np.all((pts > lo - upper) & (pts < hi + upper), axis=1)
        if not inside.any():
            return []
        idx, pts = idx[inside], pts[inside]
        dist, close, vert, tri = self.query(q, pts, upper)
        cand = np.flatnonzero(np.isfinite(dist))
        if not len(cand):
            return []
        nq = q.normal(close[cand], tri[cand])
        vec = pts[cand] - close[cand]
        off = np.einsum("ij,ij->i", vec, nq)
        dd = dist[cand]
        facing = (dd < 1e-3) | (np.abs(off) > 0.8 * dd)
        need = np.where(facing, req.gap_for(True), req.gap) + req.margin
        viol = dd < need - MIN_DEFICIT
        if not viol.any():
            return []
        sigma = self.decide_side(i, j, off[facing])
        facing |= sigma * off < 0  # a sample on the wrong side is pushed through, never further under the other piece
        need = np.where(facing, req.gap_for(True), req.gap) + req.margin
        # required increase along the chosen direction, per sample
        delta = np.where(facing, need - sigma * off, need - dd)
        direction = np.where(facing[:, None], sigma * nq, vec / np.maximum(dd, 1e-12)[:, None])
        sel = np.flatnonzero(viol & (delta > MIN_DEFICIT))
        if not len(sel):
            return []
        sel = sel[np.argsort(-delta[sel], kind="stable")[:ROWS_PER_PAIR]]
        ga = self._grad(p, idx[cand[sel]], direction[sel]) if p.moving else np.zeros((len(sel), 3))
        gb = -self._grad(q, vert[cand[sel]], direction[sel]) if q.moving else np.zeros((len(sel), 3))
        out: list[tuple[int, F64, int, F64, float]] = []
        for k, si in enumerate(sel.tolist()):
            fac = bool(facing[si])
            # facing pieces: only the outer one moves (outward), unless it cannot move
            a_moves = p.moving and (not fac or sigma > 0 or not q.moving)
            b_moves = q.moving and (not fac or sigma < 0 or not p.moving)
            if not (a_moves or b_moves):
                continue
            out.append(
                (
                    self._key(i) if a_moves else 0,
                    ga[k] if a_moves else np.zeros(3),
                    self._key(j) if b_moves else 0,
                    gb[k] if b_moves else np.zeros(3),
                    float(delta[si]),
                )
            )
        return out

    # -- step ------------------------------------------------------------------------------------------------------
    def step(self, rows: list[tuple[int, F64, int, F64, float]]) -> dict[int, F64]:
        """Kaczmarz sweeps over the rows: the weighted smallest parameter change that satisfies them (linearised)."""
        weights = {self._key(i): p.slide_weight for i, p in enumerate(self.pieces) if p.moving}
        zero = np.zeros(3)

        def wv(key: int) -> F64:
            return np.array([1.0 / RADIAL_WEIGHT, 1.0 / weights.get(key, 1.0), 1.0])

        delta: dict[int, F64] = {}
        for _ in range(SWEEPS):
            changed = False
            for ka, ga, kb, gb, need in rows:
                if ka and ka == kb:  # both pieces of one group: a single block
                    ka, ga, kb, gb = ka, ga + gb, 0, zero
                da, db = delta.get(ka, zero), delta.get(kb, zero)
                cur = (float(ga @ da) if ka else 0.0) + (float(gb @ db) if kb else 0.0)
                den = (float(ga @ (wv(ka) * ga)) if ka else 0.0) + (float(gb @ (wv(kb) * gb)) if kb else 0.0)
                if den < 1e-12 or cur >= need - 1e-7:
                    continue
                lam = (need - cur) / den
                if ka:
                    delta[ka] = da + lam * wv(ka) * ga
                if kb:
                    delta[kb] = db + lam * wv(kb) * gb
                changed = True
            if not changed:
                break
        return delta

    def apply(self, delta: dict[int, F64], max_shift: float) -> float:
        """Apply parameter changes (capped per iteration), clamp to the limits; returns the largest change."""
        moved = 0.0
        for key, d in delta.items():
            d = np.clip(d, -MAX_STEP, MAX_STEP)
            members = [p for i, p in enumerate(self.pieces) if self._key(i) == key]
            lead = next(p for p in members if np.all(p.sign > 0)) if len(members) > 1 else members[0]
            th = lead.theta + d
            lead.limits_hit.discard("max_shift")
            lead.limits_hit.discard("max_standoff")
            sh = th[1:] - lead.theta0[1:]
            n = float(np.linalg.norm(sh))
            if n > max_shift:
                th[1:] = lead.theta0[1:] + sh * (max_shift / n)
                lead.limits_hit.add("max_shift")
            r_lo = max(p.r_min for p in members)
            r_hi = min(p.r_max for p in members)
            if th[0] > r_hi:
                lead.limits_hit.add("max_standoff")
            th[0] = float(np.clip(th[0], r_lo, max(r_hi, r_lo)))
            if lead.pair_side and lead.surface is not None:  # a pair never closes up or crosses its mirror plane
                th[1] = pair_slide(lead, th, max_shift, self.half_gap)
            moved = max(moved, float(np.abs(th - lead.theta).max()))
            for p in members:
                p.theta = th * p.sign
                p.limits_hit = set(lead.limits_hit)
                p.pair_room = lead.pair_room
                p.update()
        return moved

    def run(self, max_iter: int, max_shift: float, fine: bool = False) -> SolveStats:
        """Iterate until no constraint is violated (with the margin), ``max_iter``, or no progress for
        ``PATIENCE`` iterations (the next, wider stage takes over)."""
        st = SolveStats()
        self._project_pairs(max_shift)  # start (and restart after layering or a larger radius) with the pairs apart
        best, since = np.inf, 0
        for it in range(1, max_iter + 1):
            rows = self.rows(fine)
            st.iterations, st.rows_last = it, len(rows)
            if not rows:
                st.converged = True
                return st
            total = sum(r[4] for r in rows)
            if total < best * 0.98:
                best, since = total, 0
            else:
                since += 1
                if since >= PATIENCE:
                    break
            moved = self.apply(self.step(rows), max_shift)
            log.debug("iteration %d: %d rows, deficit %.5f, moved %.5f", it, len(rows), total, moved)
            if moved < 1e-7:  # stuck at the limits: more iterations do not help
                break
        st.converged = not self.rows(fine)
        return st


def pair_half_gap(req: Requirements) -> float:
    """How far every vertex of a pair half stays from the mirror plane (world x is a chord, the slide an arc: half the
    margin is plenty)."""
    return 0.5 * (req.gap + req.margin)


def _pair_clear(p: Piece, th: F64, half_gap: float) -> float:
    """Smallest distance of the lead's vertices from x = 0 on its own side, minus the half gap (>= 0: the pair is
    ``gap`` apart). World x depends on local x only (the axis has no x component), so one row of samples does."""
    assert p.surface is not None
    xs = p.local[p.iter_idx, 0] if len(p.iter_idx) else p.local[:, 0]
    x = p.surface.map(np.column_stack([xs, np.zeros_like(xs)]), th)[:, 0]
    return float((p.pair_side * x).min()) - half_gap


def pair_slide(p: Piece, th: F64, max_shift: float, half_gap: float) -> float:
    """The slide ``s`` of a pair lead at ``th``: unchanged if its halves are apart, else the smallest outward slide
    (away from the mirror plane, from world geometry: the direction that moves its innermost vertex away) that parts
    them, within ``max_shift`` of the anchor. Without room it stops at the limit, still on its own side, and records
    ``pair_room`` (needed, available outward slide from the anchor) and the limit ``pair_no_room``."""
    p.pair_room = None
    p.limits_hit.discard("pair_no_room")
    s0 = float(th[1])

    def clear(s: float) -> float:
        return _pair_clear(p, np.array([th[0], s, th[2]]), half_gap)

    if clear(s0) >= -MIN_DEFICIT:
        return s0
    out = float(np.sign(clear(s0 + 1e-4) - clear(s0 - 1e-4)))
    if out == 0.0:
        assert p.surface is not None
        out = float(np.sign(p.pair_side * p.surface.e_x[0])) or 1.0
    room = float(np.sqrt(max(max_shift**2 - (th[2] - p.theta0[2]) ** 2, 0.0)))
    lim = max(float(p.theta0[1]) + out * room, s0, key=lambda v: out * v)  # never back toward the plane
    if clear(lim) < -MIN_DEFICIT:
        far = s0 + out * 1.0  # how far it would have to go (reported); a metre is past any torso
        need = _bisect(clear, lim, far) if clear(far) >= 0.0 else far
        p.pair_room = (abs(need - float(p.theta0[1])), room)
        p.limits_hit.add("pair_no_room")
        return lim
    return _bisect(clear, s0, lim)


def _bisect(f: Callable[[float], float], lo: float, hi: float) -> float:
    """Where ``f`` turns non-negative between ``lo`` (f < 0) and ``hi`` (f >= 0), to 0.01 mm."""
    while abs(hi - lo) > 1e-5:
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if f(mid) >= 0.0 else (mid, hi)
    return hi


def iteration_samples(local: F64, boundary: I64, spacing: float = 0.01) -> I64:
    """Vertex indices used while iterating: the whole boundary plus interior vertices thinned to about ``spacing``."""
    mask = np.ones(len(local), dtype=bool)
    mask[boundary] = False
    interior = np.flatnonzero(mask)
    cell = np.floor(local[interior] / spacing).astype(np.int64)
    _, first = np.unique(cell, axis=0, return_index=True)
    return np.unique(np.concatenate([boundary, interior[np.sort(first)]])).astype(np.int64)

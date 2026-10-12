"""Auto-arrange: pieces on offset body shells, spread out so that nothing overlays and seams are visible. Metres.

Each non-locked piece is mapped *row by row* onto the offset curve of the body cross-section of its region (``shells``):
a point at flat offset ``(x, y)`` from the piece reference goes to axial position ``p0 + y`` and to arc distance ``x``
from the anchor ray of that row, so arc length along a row and distance along the axis are exact (nothing is scaled).
The standoff ``h`` (distance of the shell from the body) is a *display* quantity: it makes room for the pieces. It is
never ease. Ease is reported separately as garment loop minus skin loop (``arrange_report``).

Order of work: refine (cached) -> rings and layers -> standoff per ring (row sums, then a numeric clearance search
against
the avatar) -> relaxation of piece shifts until every pair is at least ``gap_piece`` apart -> check, raise the standoff
where something is still violated (at most a few rounds) -> meshes and report.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree
from shapely.affinity import translate
from shapely.geometry import Polygon

from .errors import RingNotFoundError
from .loaders import PieceInfo, PointRecord
from .refine import RefineCache, RefinedPiece, _densify
from .shells import Shells
from .stack import RingSurface, _angle
from .verify import AvatarSurface, surface_samples
from .widths import row_width

log = logging.getLogger(__name__)
F64 = NDArray[np.float64]
I64 = NDArray[np.int64]
TOL = 1e-4  # a pair may be this far below its target before it counts as a deficit
MAX_ROUNDS = 4
RELAX_ITERATIONS = 30
RELAX_DAMPING = 0.8  # fraction of the computed push applied per iteration (the nearest points move as pieces move)
RELAX_MAX_STEP = 0.02  # metres per iteration
RANGE_MARGIN = 0.005  # rows this close to the end of a body region still count as inside it
PRIORITY = {"torso": 0, "neck": 1, "leg_L": 1, "leg_R": 1, "arm_L": 2, "arm_R": 2}  # who yields across rings
STEEP_PERIMETER_SLOPE = 1.5  # |d perimeter / d axis| above this marks shoulder-like rows (reported separately)


@dataclass(frozen=True)
class ArrangeOptions:
    """Options of ``arrange`` (defaults from the specification)."""

    clearance_body: float = 0.03
    gap_piece: float = 0.03
    gap_layer: float = 0.015
    max_standoff: float = 0.15
    max_shift: float = 0.06
    refine_max_edge: float = 0.01
    standoff_mode: str = "uniform"

    @classmethod
    def from_doc(cls, d: dict[str, Any] | None) -> ArrangeOptions:
        """Options with the values of ``arrange_options`` of a placement spec."""
        return cls(**(d or {}))

    def as_dict(self) -> dict[str, Any]:
        """The options as a JSON-ready dict."""
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


@dataclass
class Item:
    """One placement being arranged."""

    index: int
    pl: dict[str, Any]
    info: PieceInfo
    point: PointRecord
    source: str
    locked: bool = False
    placed: Any = None  # PlacedPiece of a locked piece
    region: str = ""
    layer: int = 0
    user_layer: bool = False
    refined: RefinedPiece | None = None
    mirror: bool = False
    ref: F64 = field(default_factory=lambda: np.zeros(2))
    theta: float = 0.0
    p0: float = 0.0
    psi: float = 0.0
    uv: F64 = field(
        default_factory=lambda: np.zeros((0, 2))
    )  # unrolled vertices (X around, P along), from the reference
    uo: F64 = field(default_factory=lambda: np.zeros((0, 2)))  # unrolled refined outline
    ub: F64 = field(default_factory=lambda: np.zeros((0, 2)))  # unrolled outline sampled every 5 mm
    clear_idx: I64 = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    shift0: F64 = field(default_factory=lambda: np.zeros(2))
    delta: F64 = field(default_factory=lambda: np.zeros(2))
    widths: F64 = field(default_factory=lambda: np.zeros(0))  # flat row width every 5 mm from P = pmin
    pmin: float = 0.0
    sec: F64 = field(
        default_factory=lambda: np.ones(0)
    )  # 1 / cos(tilt of the shell) on every row, over the item's span
    clear_uv: F64 = field(default_factory=lambda: np.zeros((0, 2)))  # unrolled sample points of the clearance search

    @property
    def name(self) -> str:
        """Piece name."""
        return self.info.name


@dataclass
class RingState:
    """Standoff state of one region (layer 0): required (uncapped) and used standoff rows."""

    region: str
    required: float = 0.0
    used: F64 = field(default_factory=lambda: np.zeros(0))
    exceeded: bool = False
    proxy_rows_m: F64 = field(default_factory=lambda: np.zeros(0))
    proxy_need: F64 = field(default_factory=lambda: np.zeros(0))
    proxy_h: float = 0.0  # standoff of the row-sum estimate
    clear_h: float = 0.0  # standoff after the clearance search


def _wrap(x: F64 | float, length: F64 | float) -> F64 | float:
    return (x + 0.5 * length) % length - 0.5 * length


class Arranger:
    """One run of ``arrange`` (all state lives here; nothing is global)."""

    def __init__(
        self,
        avatar: Any,
        pieces: dict[str, PieceInfo],
        spec: dict[str, Any],
        options: ArrangeOptions,
        place_locked: Callable[[dict[str, Any], PieceInfo, str], Any],
        refine_cache: RefineCache,
        derived_sources: dict[str, str],
    ) -> None:
        self.avatar, self.pieces, self.spec, self.o = avatar, pieces, spec, options
        self.refine_cache = refine_cache
        self.window = options.clearance_body + 0.01
        if "shells" not in avatar.cache:
            avatar.cache["shells"] = Shells.from_avatar(avatar)
        self.shells: Shells = avatar.cache["shells"]
        if "surface" not in avatar.cache:
            avatar.cache["surface"] = AvatarSurface(avatar.vertices, avatar.faces, 0.008)
        self.body: AvatarSurface = avatar.cache["surface"]
        self.items: list[Item] = []
        self.surf: dict[str, RingSurface] = {}
        self.rings: dict[str, RingState] = {}
        self._stab: dict[tuple[int, int], F64] = {}
        self._ell: dict[int, F64] = {}
        self._locked: dict[int, cKDTree] | None = None
        self.notes: list[str] = []
        self.warnings: list[str] = []
        self.place_locked = place_locked
        self.derived_sources = derived_sources

    # -- preparation ----------------------------------------------------------------------------------------------
    def prepare(self) -> None:
        """Create the items: refined meshes, anchors, unrolled coordinates, layers."""
        from .placement import target_frame

        for k, pl in enumerate(self.spec["placements"]):
            info = self.pieces[pl["piece"]]
            point = self.avatar.points[pl["point"]]
            it = Item(k, pl, info, point, self.derived_sources.get(pl["piece"], ""))
            it.locked = bool(pl.get("locked", False))
            it.user_layer = "layer" in pl
            it.layer = int(pl.get("layer", 0))
            ring = pl.get("ring")
            if ring is not None and ring not in self.shells.regions:
                raise RingNotFoundError(f"unknown ring {ring!r}; rings: {sorted(self.shells.regions)}")
            if it.locked:
                it.placed = self.place_locked(pl, info, it.source)
                try:
                    it.region = ring or self.shells.ring_of_bv(point.bv)
                except RingNotFoundError:
                    it.region = ""
                self.items.append(it)
                continue
            it.region = ring or self.shells.ring_of_bv(point.bv)
            region = self.shells.regions[it.region]
            it.refined = self.refine_cache.get(info.bundle, self.o.refine_max_edge, info.mesh())
            it.mirror = bool(pl.get("mirror", False))
            it.ref = info.reference(str(pl.get("reference", "bbox_centre")))[:2]
            it.shift0 = np.array(pl.get("shift_m", [0.0, 0.0]), dtype=np.float64)
            it.delta = it.shift0.copy()
            # anchor: axial position and ray angle of the arrangement point in the region frame
            p0, xy0 = region.coords(point.position)
            it.p0 = float(p0[0])
            hull = region.hull_at(float(np.clip(it.p0, *region.p_range)), self.window)[0]
            c = hull.centroid
            it.psi = float(np.arctan2(xy0[0, 1] - c[1], xy0[0, 0] - c[0]))
            if region.symmetric and abs(float(point.position[0])) < 0.002:  # on the centre line: exactly front or back
                it.psi = 0.0 if xy0[0, 0] > c[0] else float(np.pi)
            frame = target_frame(point.normal, point.up, False)
            e_phi = np.cross(region.axis, region.vector_of(np.array([it.psi]))[0])
            it.theta = float(np.arctan2(frame.t @ region.axis, frame.t @ e_phi)) + np.radians(
                float(pl.get("rotation_deg", 0.0))
            )
            r = it.refined
            it.uv = self._unroll(it, r.vertices[:, :2])
            it.uo = self._unroll(it, r.vertices[r.boundary, :2])
            it.ub = _densify(it.uo, 0.0075)[0]
            nb = len(r.boundary)
            interior = np.arange(len(r.vertices))
            interior = interior[~np.isin(interior, r.boundary)]
            it.clear_idx = np.concatenate([r.boundary[::3], interior[::14]]).astype(np.int64)
            it.clear_uv = it.uv[it.clear_idx]
            it.pmin = float(it.uo[:, 1].min())
            ps = np.arange(it.pmin, float(it.uo[:, 1].max()) + 0.005, 0.005)
            it.widths = np.array([row_width(it.uo, float(y)) for y in ps])
            del nb
            self.items.append(it)

    @staticmethod
    def _unroll(it: Item, q: F64) -> F64:
        x = (2.0 * it.ref[0] - q[:, 0]) if it.mirror else q[:, 0]
        d = np.column_stack([x - it.ref[0], q[:, 1] - it.ref[1]])
        c, s = np.cos(it.theta), np.sin(it.theta)
        return np.column_stack([c * d[:, 0] - s * d[:, 1], s * d[:, 0] + c * d[:, 1]])

    def free(self, region: str | None = None, layer: int | None = None) -> list[Item]:
        """Non-locked items, optionally of one region and layer."""
        return [
            i
            for i in self.items
            if not i.locked and (region is None or i.region == region) and (layer is None or i.layer == layer)
        ]

    def build_surfaces(self) -> None:
        """One ``RingSurface`` per region in use, covering all rows the pieces (and their shifts) can reach."""
        margin = self.o.max_shift + 0.03
        for region in sorted({i.region for i in self.free()}):
            its = self.free(region)
            lo = min(i.p0 + i.shift0[1] + float(i.uo[:, 1].min()) for i in its) - margin
            hi = max(i.p0 + i.shift0[1] + float(i.uo[:, 1].max()) for i in its) + margin
            self.surf[region] = RingSurface(self.shells.regions[region], lo, hi, self.window)

    def assign_layers(self) -> None:
        """Layer 0 for the big pieces; a piece lying mostly on top of a lower piece gets that piece's layer + 1."""
        for region in sorted(self.surf):
            its = self.free(region)
            if len(its) < 2:
                continue
            surf = self.surf[region]
            h = np.full(1, 0.05)
            r0 = surf.row_of(np.array([its[0].p0]))
            length = float(surf.stack.length(r0, h)[0])
            s_ref = float(surf.stack.arc_of_ray(r0, np.array([its[0].psi]), h)[0])
            polys: dict[int, Polygon] = {}
            for it in its:
                s = float(surf.stack.arc_of_ray(r0, np.array([it.psi]), h)[0])
                rel = float(_wrap(s - s_ref, length))
                pts = it.uo + np.array([rel, it.p0 + it.shift0[1]])
                poly = Polygon(pts)
                polys[it.index] = (
                    poly if poly.is_valid else poly.convex_hull
                )  # (the placket slit of a sleeve; make_valid is slow)
            order = sorted(its, key=lambda i: (-polys[i.index].area, i.index))
            done: list[Item] = []
            for it in order:
                if not it.user_layer:
                    best = 0
                    pi = polys[it.index]
                    for jt in done:
                        cover = max(
                            pi.intersection(translate(polys[jt.index], k * length, 0.0)).area for k in (-1, 0, 1)
                        ) / max(pi.area, 1e-12)
                        if cover >= 0.4:
                            best = max(best, jt.layer + 1)
                    it.layer = best
                done.append(it)

    # -- surfaces and mapping -------------------------------------------------------------------------------------
    def hrows(self, it: Item) -> F64:
        """Standoff of an item on every row: the ring standoff plus ``layer * gap_layer`` measured along the 3D normal.

        The rows are horizontal (cross-section) offsets, so on a tilted shell (shoulders) the 3D distance is the
        row offset times ``cos(tilt)``; ``it.sec`` undoes that, so the layers are ``gap_layer`` apart in 3D.
        """
        used = self.rings[it.region].used
        if it.layer == 0 or len(it.sec) != len(used):
            return np.asarray(used)
        return np.asarray(used + it.layer * self.o.gap_layer * it.sec)

    def tilt_secant(self, it: Item) -> F64:
        """``sec`` of the shell tilt on every row, for the normal directions the item covers around the ring."""
        surf = self.surf[it.region]
        st = surf.stack
        r = surf.row_of(np.array([it.p0 + it.shift0[1]]))
        h = self.rings[it.region].used[r]
        s_anchor = surf.ray_table(it.psi, self.rings[it.region].used)[r[0]]
        xs = np.linspace(float(it.uo[:, 0].min()), float(it.uo[:, 0].max()), 32)
        _, nu = _angle(st, np.full(32, r[0]), s_anchor + xs, np.full(32, h[0]))
        dirs = np.radians(np.arange(0.0, 360.0, 5.0))
        diff = np.abs(_wrap(dirs[:, None] - nu[None, :], 2.0 * np.pi)).min(axis=1)
        sel = diff < np.radians(6.0) if np.ptp(nu) < 2.0 * np.pi - 0.2 else np.ones(len(dirs), dtype=bool)
        sup = st.support(np.column_stack([np.cos(dirs), np.sin(dirs)]))[:, sel]
        step = 5
        slope = np.zeros(sup.shape)
        slope[step:-step] = (sup[2 * step :] - sup[: -2 * step]) / (2 * step * 0.001)
        slope[:step], slope[-step:] = slope[step], slope[-step - 1]
        return np.asarray(np.sqrt(1.0 + (np.abs(slope).max(axis=1)) ** 2))

    def stab(self, it: Item) -> F64:
        """Arc coordinate of the anchor ray of the item on every row (cached until the standoff changes)."""
        key = (it.index, 0)
        if key not in self._stab:
            self._stab[key] = self.surf[it.region].ray_table(it.psi, self.hrows(it))
        return self._stab[key]

    def ell(self, it: Item) -> F64:
        """Surface length along the axis up to every row, averaged over the arc positions the piece covers.

        Rows are 1 mm apart in axial position. On a tilted shell (hips, bust, shoulders) the distance along
        the surface is
        longer than the axial one, so ``P`` of the piece is converted to an axial position through this
        length: a vertical
        piece edge keeps its length instead of stretching by ``1 / cos(tilt)``. Rows stay horizontal.
        """
        if it.index not in self._ell:
            surf = self.surf[it.region]
            h = self.hrows(it)
            step = 5
            sub = np.unique(np.r_[np.flatnonzero((surf.k0 + np.arange(surf.rows)) % step == 0), 0, surf.rows - 1])
            pc = surf.p_of_row[sub]
            xs = np.linspace(float(it.uo[:, 0].min()), float(it.uo[:, 0].max()), 13)
            w = np.zeros(len(sub) - 1)
            for x in xs:
                pts = surf.world(pc, self.stab(it), np.full(len(pc), x), h, False)
                w += np.linalg.norm(np.diff(pts, axis=0), axis=1) / np.diff(pc)
            cum = np.concatenate([[0.0], np.cumsum(w / len(xs) * np.diff(pc))])
            self._ell[it.index] = np.interp(surf.p_of_row, pc, cum)
        return self._ell[it.index]

    def p_of(self, it: Item, big_p: F64) -> F64:
        """Axial positions of unrolled ``P`` values (from the reference) at the current shift of the item."""
        surf = self.surf[it.region]
        ell, rows = self.ell(it), surf.p_of_row
        q = float(np.interp(it.p0 + it.delta[1], rows, ell)) + big_p
        p = np.interp(q, ell, rows)
        p = np.where(q < ell[0], rows[0] + (q - ell[0]), p)
        return np.asarray(np.where(q > ell[-1], rows[-1] + (q - ell[-1]), p))

    def map_unrolled(self, it: Item, uv: F64, interpolate: bool) -> F64:
        """World points of unrolled coordinates ``uv`` (N,2) of an item at its current shift."""
        surf = self.surf[it.region]
        return surf.world(self.p_of(it, uv[:, 1]), self.stab(it), uv[:, 0] + it.delta[0], self.hrows(it), interpolate)

    def _drop(self, region: str) -> None:
        """Forget the cached anchor tables of a region (its standoff changed)."""
        self._stab = {k: v for k, v in self._stab.items() if self.items[k[0]].region != region}
        self._ell = {k: v for k, v in self._ell.items() if self.items[k].region != region}

    def set_standoff(self, region: str, rows: F64) -> None:
        """Set the (uncapped is kept elsewhere) used standoff rows of a region and drop the cached anchor tables."""
        self.rings[region].used = rows
        self._drop(region)

    # -- standoff -------------------------------------------------------------------------------------------------
    def row_sums(self, region: str, initial: bool = False) -> tuple[F64, F64, F64]:
        """Axial rows (every 5 mm), the sum of the flat widths of the layer-0 pieces of the ring on each, and
        how many."""
        its = self.free(region, 0)
        shift = {i.index: (i.shift0 if initial else i.delta)[1] for i in its}
        lo = min(i.p0 + shift[i.index] + i.pmin for i in its)
        hi = max(i.p0 + shift[i.index] + float(i.uo[:, 1].max()) for i in its)
        ps = np.arange(lo, hi, 0.005)
        total = np.zeros(len(ps))
        count = np.zeros(len(ps))
        for it in its:
            idx = np.rint((ps - it.p0 - shift[it.index] - it.pmin) / 0.005).astype(int)
            ok = (idx >= 0) & (idx < len(it.widths))
            w = np.where(ok, it.widths[np.clip(idx, 0, len(it.widths) - 1)], 0.0)
            total += w
            count += w > 1e-4
        return ps, total, count

    def proxy(self, region: str) -> None:
        """Row-sum estimate of the standoff: all pieces of the ring side by side need ``sum(w) + n gap <=
        perimeter + 2 pi h``."""
        ps, total, count = self.row_sums(region)
        per = self.surf[region].perimeter(ps)
        need = np.where(count > 0, (total + count * self.o.gap_piece - per) / (2.0 * np.pi), -np.inf)
        st = self.rings[region]
        st.proxy_rows_m, st.proxy_need = ps, need

    def min_clearance(self, region: str, h: F64) -> float:
        """Smallest signed distance to the avatar over sample points of the layer-0 pieces (rows inside the region)."""
        surf = self.surf[region]
        reg = self.shells.regions[region]
        lo, hi = reg.p_range
        old = self.rings[region].used
        self.rings[region].used = h
        self._drop(region)
        worst = np.inf
        try:
            for it in self.free(region, 0):
                uv = it.clear_uv
                p = self.p_of(it, uv[:, 1])
                keep = (p >= lo - RANGE_MARGIN) & (p <= hi + RANGE_MARGIN)
                if not keep.any():
                    continue
                w = surf.world(p[keep], self.stab(it), uv[keep, 0] + it.delta[0], h, False)
                worst = min(worst, float(self.body.signed_distance(w).min()))
        finally:
            self.rings[region].used = old
            self._drop(region)
        return worst

    def solve_standoff(self, region: str, extra: float = 0.0) -> None:
        """Standoff of a region at the current positions: the larger of the row-sum need and the standoff that clears
        the body, plus ``extra`` (what the gaps of slanted edges needed in earlier rounds)."""
        o = self.o
        surf = self.surf[region]
        st = self.rings[region]
        self.proxy(region)
        need = st.proxy_need[np.isfinite(st.proxy_need)]
        h_gap = max(float(need.max()) if len(need) else 0.0, o.clearance_body)
        target = o.clearance_body + 0.0005
        h = max(h_gap, st.clear_h if st.clear_h > 0 else h_gap)  # start where the last round ended
        f = self.min_clearance(region, np.full(surf.rows, h)) - target
        slope, last = 1.3, None  # distance to the body per unit standoff (about 1 to 1.5), refined by the secant
        for _ in range(10):
            if abs(f) < 3e-5 or (f > 0 and h <= h_gap + 1e-12) or h >= 0.6:
                break
            if last is not None and abs(h - last[0]) > 1e-9:
                slope = float(np.clip((f - last[1]) / (h - last[0]), 0.5, 3.0))
            last = (h, f)
            h = float(np.clip(h - f / slope, h_gap, 0.6))
            f = self.min_clearance(region, np.full(surf.rows, h)) - target
        st.proxy_h, st.clear_h = h_gap, h
        st.required = h + extra
        self.cap(region)

    def cap(self, region: str) -> None:
        """Use the required standoff, but never more than ``max_standoff`` (the shortfall is reported, not hidden)."""
        st = self.rings[region]
        surf = self.surf[region]
        st.exceeded = st.required > self.o.max_standoff + 1e-9
        self.set_standoff(region, np.full(surf.rows, min(st.required, self.o.max_standoff)))

    # -- relaxation -----------------------------------------------------------------------------------------------
    def boundary_world(self, it: Item) -> F64:
        """World points of the outline of a free item (every 5 mm)."""
        return self.map_unrolled(it, it.ub, False)

    def locked_samples(self) -> dict[int, cKDTree]:
        """KD-trees of the surface of locked pieces (obstacles); built once, locked pieces never move."""
        if self._locked is None:
            self._locked = {
                it.index: cKDTree(surface_samples(it.placed.vertices, it.placed.faces, 0.006))
                for it in self.items
                if it.locked
            }
        return self._locked

    def relax(self, warm: bool = False) -> tuple[int, dict[tuple[int, int], float]]:
        """Push pieces of the same layer apart (along their own surfaces) until every pair is ``gap_piece`` apart.

        Starts from the initial shifts, at most ``RELAX_ITERATIONS`` rounds, each piece stays within
        ``max_shift`` of its
        initial shift, locked pieces stay. Returns the number of rounds used and the shortfall of every pair that
        still is below the gap (empty when converged).
        """
        gap = self.o.gap_piece
        free = self.free()
        if not warm:
            for it in free:
                it.delta = it.shift0.copy()
        locked = self.locked_samples()
        pairs = [
            (a, b)
            for a in self.items
            for b in self.items
            if a.index < b.index and a.layer == b.layer and not (a.locked and b.locked)
        ]
        rounds = 0
        short: dict[tuple[int, int], float] = {}
        cache: dict[int, tuple[F64, F64]] = {}
        for rounds in range(1, RELAX_ITERATIONS + 1):  # noqa: B007 - the last value is returned
            for it in free:  # only pieces that moved are mapped again
                c = cache.get(it.index)
                if c is None or not np.array_equal(c[0], it.delta):
                    cache[it.index] = (it.delta.copy(), self.boundary_world(it))
            bw = {k: v[1] for k, v in cache.items()}
            lo = {k: v.min(axis=0) for k, v in bw.items()}
            hi = {k: v.max(axis=0) for k, v in bw.items()}
            for k, tr in locked.items():
                lo[k], hi[k] = tr.mins, tr.maxes
            ids = list(lo)
            pos = {k: i for i, k in enumerate(ids)}
            lo_a, hi_a = np.array([lo[k] for k in ids]), np.array([hi[k] for k in ids])
            sep = np.linalg.norm(
                np.maximum(0.0, np.maximum(lo_a[:, None, :] - hi_a[None, :, :], lo_a[None, :, :] - hi_a[:, None, :])),
                axis=2,
            )  # distance between the bounding boxes of every two pieces
            trees: dict[int, cKDTree] = dict(locked)
            push = {it.index: np.zeros(2) for it in free}
            events: dict[int, list[tuple[int, F64, float]]] = {}
            worst = 0.0
            short = {}
            for a, b in pairs:
                if a.locked:
                    a, b = b, a  # the free one first
                if sep[pos[a.index], pos[b.index]] > gap:
                    continue  # the bounding boxes are farther apart than the gap
                if b.index not in trees:
                    trees[b.index] = cKDTree(bw[b.index])
                tb = trees[b.index]
                ba = bw[a.index]
                d, j = tb.query(ba)
                k = int(np.argmin(d))
                deficit = gap + TOL - float(d[k])
                if deficit <= TOL:
                    continue
                worst = max(worst, deficit)
                short[(min(a.index, b.index), max(a.index, b.index))] = gap - float(d[k])
                n3 = ba[k] - tb.data[j[k]]
                nn = float(np.linalg.norm(n3))
                if nn < 1e-6:  # touching or overlapping: push along the line between the piece centres
                    n3 = ba.mean(axis=0) - (tb.data.mean(axis=0) if b.index in locked else bw[b.index].mean(axis=0))
                    nn = float(np.linalg.norm(n3))
                if nn < 1e-9:
                    continue
                n3 = n3 / nn
                # across rings the piece of the lower-priority region (arm, neck) gives way, the
                # torso does not move for it
                pa, pb = (0, 0) if a.locked or b.locked else (PRIORITY.get(a.region, 1), PRIORITY.get(b.region, 1))
                a_moves = not a.locked and pa >= pb
                b_moves = not b.locked and pb >= pa
                share = deficit * (1.0 if not (a_moves and b_moves) else 0.5)
                if a_moves:
                    events.setdefault(a.index, []).append((k, n3, share))
                if b_moves:
                    events.setdefault(b.index, []).append((int(j[k]), -n3, share))
            if worst <= TOL:
                short = {}
                break
            for idx, ev in events.items():
                push[idx] += self._push(self.items[idx], ev)
            for it in free:
                step = it.delta + push[it.index] - it.shift0
                n = float(np.linalg.norm(step))
                if n > self.o.max_shift:
                    step *= self.o.max_shift / n
                it.delta = it.shift0 + step
        return rounds, short

    def _alone(self, it: Item) -> bool:
        """Whether the item is the only free piece of its ring and layer."""
        return len(self.free(it.region, it.layer)) == 1

    def _push(self, it: Item, events: list[tuple[int, F64, float]]) -> F64:
        """The shift of ``it`` that moves its boundary points by the wanted amounts along the wanted directions.

        Each event is (boundary sample, unit direction, distance); the shift is taken along the surface of the piece:
        the arc direction and the axis of its shell at that sample.
        """
        surf = self.surf[it.region]
        k = np.array([e[0] for e in events])
        n3 = np.array([e[1] for e in events])
        share = np.array([e[2] for e in events])
        ub = it.ub
        e_arc, axis = surf.frame_at(self.p_of(it, ub[k, 1]), self.stab(it), ub[k, 0] + it.delta[0], self.hrows(it))
        v = np.column_stack([np.einsum("ij,ij->i", n3, e_arc), n3 @ axis])
        if self._alone(
            it
        ):  # the only piece of its ring (a sleeve on its arm): it slides along the axis, never around it
            v[:, 0] = 0.0
        nv2 = (v * v).sum(axis=1)
        ok = (
            nv2 >= 0.15**2
        )  # pieces facing each other across the surface normal cannot be moved apart along the surface
        step = np.asarray((share[ok, None] * v[ok] / nv2[ok, None]).sum(axis=0)) * RELAX_DAMPING
        n = float(np.linalg.norm(step))
        return step * (RELAX_MAX_STEP / n) if n > RELAX_MAX_STEP else step

    # -- checks and rounds ----------------------------------------------------------------------------------------
    def pair_distances(self) -> dict[tuple[int, int], float]:
        """Smallest distance between the outlines of every pair of pieces of the same layer (locked ones:
        their surface)."""
        locked = self.locked_samples()
        bw = {it.index: self.boundary_world(it) for it in self.free()}
        trees = {k: cKDTree(v) for k, v in bw.items()}
        out: dict[tuple[int, int], float] = {}
        for a in self.items:
            for b in self.items:
                if a.index >= b.index or a.layer != b.layer or (a.locked and b.locked):
                    continue
                if a.locked:
                    d = (
                        float(trees[b.index].query(locked[a.index].data)[0].min())
                        if False
                        else float(locked[a.index].query(bw[b.index])[0].min())
                    )
                elif b.locked:
                    d = float(locked[b.index].query(bw[a.index])[0].min())
                else:
                    d = min(
                        float(trees[b.index].query(bw[a.index])[0].min()),
                        float(trees[a.index].query(bw[b.index])[0].min()),
                    )
                out[(a.index, b.index)] = d
        return out

    def gap_deficits(self, short: dict[tuple[int, int], float]) -> dict[str, float]:
        """Sum of the gap shortfalls of the pairs of each ring (layer 0) that relaxation could not close."""
        out: dict[str, float] = {}
        for region in sorted(self.surf):
            ids = {i.index for i in self.free(region, 0)}
            out[region] = sum(v for (a, b), v in short.items() if a in ids and b in ids and v > TOL)
        return out

    def solve(self) -> None:
        """The whole solver: surfaces, layers, rounds of (standoff -> relaxation).

        The standoff depends on where the pieces are (rows, clearance) and the positions depend on the standoff, so the
        rounds iterate to a fixed point: the final positions are the ones the final standoff was computed for. Gap
        shortfalls of slanted edges that the row sums cannot see are added to the standoff of their ring (``extra``) by
        a secant iteration, so the standoff is the smallest one that works (not the first one that did).
        """
        self.build_surfaces()
        self.assign_layers()
        regions = sorted(self.surf)
        extra = {r: 0.0 for r in regions}
        for region in regions:
            self.rings[region] = RingState(region)
        self._rounds(MAX_ROUNDS, extra)
        # settle: start again from the shifts just found, as a rerun with them would. The standoff then no
        # longer carries
        # the history of the first rounds (the extra for gaps), so a rerun from the result changes nothing.
        for it in self.free():
            it.shift0 = it.delta.copy()
        self._rounds(2, {r: 0.0 for r in regions})
        self.refine_layers()

    def _rounds(self, limit: int, extra: dict[str, float]) -> None:
        """Rounds of (standoff at the current positions -> relaxation) until the standoff and the positions
        are stable."""
        regions = sorted(self.surf)
        last: dict[str, tuple[float, float]] = {}
        for rnd in range(limit + 1):
            before = {it.index: it.delta.copy() for it in self.free()}
            h_before = {r: float(self.rings[r].used[0]) if len(self.rings[r].used) else 0.0 for r in regions}
            for region in regions:
                if self.free(region, 0):
                    self.solve_standoff(region, extra[region])
                else:
                    self.rings[region].required = self.o.clearance_body
                    self.cap(region)
            for it in self.free():
                if (
                    len(it.sec) != self.surf[it.region].rows
                ):  # once: the tilt of the shell does not depend on the standoff
                    it.sec = self.tilt_secant(it) if it.layer > 0 else np.ones(self.surf[it.region].rows)
            self._stab.clear()
            self._ell.clear()
            dh = max((abs(float(self.rings[r].used[0]) - h_before[r]) for r in regions), default=0.0)
            if rnd > 0 and dh < 3e-5:
                break  # the standoff did not change: the last relaxation is still valid
            _, short = self.relax()
            deficit = self.gap_deficits(short)
            moved = max((float(np.abs(it.delta - before[it.index]).max()) for it in self.free()), default=0.0)
            open_rings = [r for r in regions if deficit[r] > 2 * TOL and not self.rings[r].exceeded]
            log.debug(
                "round %d: h %s extra %s deficit %s moved %.5f dh %.5f",
                rnd,
                {r: round(float(self.rings[r].used[0]), 5) for r in regions},
                {r: round(v, 5) for r, v in extra.items()},
                {r: round(v, 5) for r, v in deficit.items()},
                moved,
                dh,
            )
            if rnd == limit or (not open_rings and moved < 1e-3 and (rnd == 0 or dh < 3e-4)):
                break
            for r in open_rings:  # the ring is (2 pi dh) longer per standoff dh: first order, then secant
                slope = 2.0 * np.pi
                if r in last and last[r][0] - deficit[r] > 1e-9 and extra[r] - last[r][1] > 1e-9:
                    slope = float(
                        np.clip((last[r][0] - deficit[r]) / (extra[r] - last[r][1]), 0.5 * np.pi, 2.0 * np.pi)
                    )
                last[r] = (deficit[r], extra[r])
                extra[r] += deficit[r] / slope

    def refine_layers(self) -> None:
        """Closed loop on the layer offsets: measure the 3D distance to the lower layers per row and scale the
        row offset.

        The first guess ``tilt_secant`` is analytic; this makes the distance ``gap_layer`` where the pieces really are.
        """
        gap = self.o.gap_layer
        for it in sorted((i for i in self.free() if i.layer > 0), key=lambda i: (i.layer, i.index)):
            below = [j for j in self.items if j.layer < it.layer]
            if not below:
                continue
            base = np.vstack([self._surface_points(j) for j in below])
            tree = cKDTree(base)
            surf = self.surf[it.region]
            for _ in range(4):
                world = self.map_unrolled(it, it.uv, False)
                d = tree.query(world)[0]
                rows = surf.row_of(self.p_of(it, it.uv[:, 1]))
                worst = np.full(surf.rows, np.inf)
                np.minimum.at(worst, rows, d)
                bad = worst < gap - TOL
                if not bad.any():
                    break
                f = np.ones(surf.rows)
                f[bad] = (gap + 0.0003) / np.maximum(worst[bad], 0.002)
                k = 5
                padded = np.pad(f, k, mode="edge")
                f = np.max([padded[i : i + len(f)] for i in range(2 * k + 1)], axis=0)
                it.sec = it.sec * np.minimum(f, 3.0)
                self._stab.pop((it.index, 0), None)
                self._ell.pop(it.index, None)

    def _surface_points(self, it: Item) -> F64:
        if it.locked:
            return surface_samples(it.placed.vertices, it.placed.faces, 0.006)
        return self.map_unrolled(it, it.uv, False)

    # -- results --------------------------------------------------------------------------------------------------
    def build_placed(self) -> list[Any]:
        """PlacedPiece objects in spec order (locked pieces as placed by their own wrap)."""
        from .engine import PlacedPiece
        from .placement import mirror_matrix, rigid_matrix, target_frame
        from .sides import extends, lateral_offset, side_warnings

        out: list[Any] = []
        for it in self.items:
            if it.locked:
                out.append(it.placed)
                continue
            r = it.refined
            assert r is not None
            world = self.map_unrolled(it, it.uv, True)
            faces = np.ascontiguousarray(r.faces[:, [0, 2, 1]]) if it.mirror else r.faces

            def transform(p2: F64, it: Item = it) -> F64:
                return self.map_unrolled(it, self._unroll(it, np.asarray(p2, dtype=np.float64)), True)

            ref_world = self.map_unrolled(it, np.zeros((1, 2)), True)[0]
            frame = target_frame(it.point.normal, it.point.up, False)
            ref3 = np.array([it.ref[0], it.ref[1], 0.0])
            m = rigid_matrix(frame, float(it.pl.get("rotation_deg", 0.0)), ref3, it.point.position)
            total = m @ mirror_matrix(float(it.ref[0])) if it.mirror else m
            bb = it.info.bundle["bbox_m"]
            lateral = lateral_offset(world, faces, ref_world)
            ext = extends(lateral, float(bb["max"][0] - bb["min"][0]))
            warns = side_warnings(it.name, it.source, it.point.name, ext)
            fit = {"mode": "arrange", "ring": it.region, "layer": it.layer}
            out.append(
                PlacedPiece(
                    it.name,
                    it.point.name,
                    total,
                    it.mirror,
                    world,
                    faces,
                    True,
                    ref_world,
                    warns,
                    fit,
                    transform,
                    it.info.stem,
                    ext,
                    it.source,
                )
            )
        return out

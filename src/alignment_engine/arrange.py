"""Collision-free, size-exact placement of all pieces of a spec (used by ``align`` and ``arrange``). Metres.

Every piece is placed by an isometry (``roll.Surface``: plane or cylinder) and then moved rigidly along that surface
or pushed away from the body (``solver``) until no piece is closer than ``clearance_body`` to the avatar and no two
pieces are closer than ``gap_piece``. Nothing is ever scaled or bent out of its developable surface.

* ``roll`` pieces of one ring (torso, neck, an arm, a leg) start on one cylinder about the ring axis whose radius is
  the smallest that fits the ring's pieces side by side with gaps and keeps every vertex of each piece ``clearance``
  from the body region under it; pieces lying over others (in the same angular sector) start one gap further out.
  That radius is display space, never ease (ease is reported separately as garment loop minus skin loop). A torso
  panel on a neck point (taller than the neck) rolls on the torso ring; collars stay on the neck ring.
* A piece and its mirrored clone are one rigid pair (``s`` mirrored, radius and ``t`` shared), also when one of them
  was placed earlier: the new copy is derived from the placed one and the pair moves together (see
  ``docs/alignment_pairs.md``). Torso panels of one pattern keep their drafting heights (``pattern_origin_m``).
* ``align`` is incremental: pieces that carry ``solved`` parameters (from an earlier answer) stay where they are and
  are obstacles; every other piece is solved in spec order against what is already placed. Exceptions: the placed half
  of a pair whose other half is new, and a ring that must grow to fit a new piece side by side (its placed pieces move
  out together, ``s`` and ``t`` kept); both come back with new ``solved`` blocks.
* ``arrange`` re-solves every piece that is not locked, jointly (a ``solved`` block is only a warm start).
* After solving, ``validity`` measures everything again on the placed meshes: sizes, clearances, all pair gaps. What
  cannot be met inside ``max_standoff`` and ``max_shift`` is returned with ``valid: false`` and the exact violations.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import shapely
from numpy.typing import NDArray
from scipy.spatial import cKDTree
from shapely.affinity import translate
from shapely.geometry import Polygon

from .errors import RingNotFoundError, UnknownNamedPointError
from .levels import level_height
from .loaders import Avatar, PieceInfo, PointRecord
from .orient import edge_point, orient_rotation, registration_spec, tilt_angles, tilt_matrix
from .pairs import counterpart_point
from .refine import RefineCache, RefinedPiece
from .roll import (
    RING_BVS,
    SYMMETRIC_RINGS,
    Surface,
    cylinder_surface,
    flat_surface,
    local_coords,
    ring_axis,
    ring_of_bv,
)
from .shells import Shells
from .slice_wrap import TORSO_BVS
from .solver import Piece, Requirements, Solver, iteration_samples, pair_half_gap, pair_slide
from .validity import TOLERANCE, LayoutCheck, Placed, layout_check, unique_edges
from .verify import AvatarSurface, vertex_faces
from .widths import row_width

log = logging.getLogger(__name__)
F64 = NDArray[np.float64]
I64 = NDArray[np.int64]
ROW_STEP = 0.005
COVER = 0.4  # a piece whose footprint is covered this much by a larger piece of its ring starts one layer out
STAGE1_SHIFT = 0.05  # first stage: small slides only; later stages widen to max_shift and then the radius
STAGE1_ITER, STAGE2_ITER, FINE_ITER = 30, 80, 15
CENTRE_SLIDE_WEIGHT = 1.0e4  # a slide is used only when nothing else can meet a constraint
RADIUS_STEP, RADIUS_ROUNDS, RADIUS_ITER = 0.02, 6, 20
GROW_TOL = 0.003  # a ring grows (its placed pieces move out) only when it lacks more than this
DATUM_HEIGHT_RATIO = 0.5  # torso pieces at least this similar in height share the drafting datum (not pockets)
LIMB_RINGS = ("arm_L", "arm_R", "leg_L", "leg_R")
REGISTERED_AXIAL_WEIGHT = 1.0e4  # a registered piece slides along the axis only when nothing else can clear it
ANCHOR_STEP, ANCHOR_RANGE = 0.01, 0.8  # limb registration: search step and reach along the limb (metres)


@dataclass(frozen=True)
class ArrangeOptions:
    """Options (``arrange_options`` of a placement spec). ``gap_layer`` None means equal to ``gap_piece``."""

    clearance_body: float = 0.03
    gap_piece: float = 0.03
    gap_layer: float | None = None
    max_standoff: float = 0.30
    max_shift: float = 0.25
    max_standoff_from_point: float = 0.12  # farther from its point than this (and than its ring needs): not valid
    refine_max_edge: float = 0.005
    incremental: bool = True
    allow_distortion: bool = False
    standoff_mode: str = "uniform"  # accepted for old specs; display only (the roll radius is the standoff)

    @classmethod
    def from_doc(cls, d: dict[str, Any] | None) -> ArrangeOptions:
        """Options with the values of ``arrange_options`` of a placement spec."""
        return cls(**(d or {}))

    def as_dict(self) -> dict[str, Any]:
        """The options as a JSON-ready dict (``gap_layer`` resolved)."""
        out = {k: getattr(self, k) for k in self.__dataclass_fields__}
        out["gap_layer"] = self.gap_layer if self.gap_layer is not None else self.gap_piece
        return out

    def requirements(self) -> Requirements:
        """The solver's requirements."""
        gl = self.gap_layer if self.gap_layer is not None and self.gap_layer != self.gap_piece else None
        return Requirements(self.clearance_body, self.gap_piece, gl, self.max_shift)


@dataclass
class Item:
    """One placement."""

    index: int
    pl: dict[str, Any]
    info: PieceInfo
    point: PointRecord
    source: str
    mode: str
    ring: str = ""
    locked: bool = False
    solve: bool = False
    warnings: list[str] = field(default_factory=list)
    refined: RefinedPiece | None = None
    ref: F64 = field(default_factory=lambda: np.zeros(3))
    mirror: bool = False
    rotation: float = 0.0
    surface: Surface | None = None
    piece: Piece | None = None
    placed: Any = None  # PlacedPiece of a distorting wrap (body_slice)
    user_layer: int | None = None
    layer: int = 0
    axial0: float = 0.0  # axial position of the anchor along the ring axis
    needed_radius: float = 0.0
    regrown: bool = False  # a placed piece moved out with its ring (incremental): it gets a new ``solved`` block
    layer_reason: str = ""
    partner: str = ""  # the other half of its mirror pair
    anchor: dict[str, Any] | None = None  # limb pieces: axial registration (mode, shift_m, reason)
    edges: I64 = field(default_factory=lambda: np.zeros((0, 2), dtype=np.int64))
    outline: F64 = field(default_factory=lambda: np.zeros((0, 2)))  # pattern outline in local coordinates
    geom_key: tuple[Any, ...] = ()
    orient_note: str = ""
    tilt: tuple[float, float] = (0.0, 0.0)
    registration: dict[str, Any] | None = None  # level registration: spec + the heights (filled by ``_register``)
    axis_up: F64 | None = None  # the up the placement asks for (point up / ring axis), before any tilt

    @property
    def name(self) -> str:
        """Piece name."""
        return self.info.name


class Layout:
    """One solve of a spec (all state lives here)."""

    def __init__(
        self,
        avatar: Avatar,
        pieces: dict[str, PieceInfo],
        spec: dict[str, Any],
        options: ArrangeOptions,
        refine_cache: RefineCache,
        derived_sources: dict[str, str],
        place_distorting: Callable[[dict[str, Any], PieceInfo, str], Any],
        check_cache: dict[Any, Any] | None = None,
        geom_cache: dict[Any, Any] | None = None,
    ) -> None:
        self.avatar, self.pieces, self.spec, self.o = avatar, pieces, spec, options
        self.cache, self.sources, self.place_distorting = refine_cache, derived_sources, place_distorting
        self.check_cache = check_cache if check_cache is not None else {}
        self.geom_cache = geom_cache if geom_cache is not None else {}
        if "surface" not in avatar.cache:
            avatar.cache["surface"] = AvatarSurface(avatar.vertices, avatar.faces, 0.006)
        self.body: AvatarSurface = avatar.cache["surface"]
        self.items: list[Item] = []
        self.warnings: list[str] = []
        self.notes: list[str] = []
        self.groups: dict[int, list[Item]] = {}
        self.pairs: list[tuple[Item, Item, bool]] = []  # (source, clone, same anchor) of every recognised pair
        self.ring_radius: dict[str, float] = {}
        self.check: LayoutCheck | None = None
        self.iterations = 0
        self.version = int(spec.get("schema_version", 1))

    # -- preparation -----------------------------------------------------------------------------------------------
    def shells(self) -> Shells:
        """Body regions (for the clearance radius and the skin loops); cached on the avatar."""
        if "shells" not in self.avatar.cache:
            self.avatar.cache["shells"] = Shells.from_avatar(self.avatar)
        shells: Shells = self.avatar.cache["shells"]
        return shells

    def prepare(self, incremental: bool) -> None:
        """Items, surfaces, start parameters, groups."""
        o = self.o
        new = {
            pl["piece"]
            for pl in self.spec["placements"]
            if not isinstance(pl.get("solved"), dict) and not pl.get("locked")
        }
        for k, pl in enumerate(self.spec["placements"]):
            ov = pl.get("over")
            if ov and incremental and ov.get("piece") in new and isinstance(pl.get("solved"), dict):
                pl = {key: v for key, v in pl.items() if key != "solved"}  # its host is new: it moves onto the host
            info = self.pieces[pl["piece"]]
            point = self.avatar.points[pl["point"]]
            it = Item(k, pl, info, point, self.sources.get(pl["piece"], ""), str(pl.get("wrap", "flat")))
            it.locked = bool(pl.get("locked", False))
            it.user_layer = int(pl["layer"]) if "layer" in pl else None
            self.items.append(it)
            if it.mode == "body_slice":
                if point.bv in TORSO_BVS:
                    it.placed = self.place_distorting(pl, info, it.source)
                    continue
                it.warnings.append(
                    f"WRAP_FALLBACK: body_slice needs a torso point (Hip, Waist, Chest, Neck), {point.name} is on "
                    f"{point.bv}; rolled around the body part instead"
                )
                it.mode = "around"
            if it.mode == "roll":
                try:
                    it.ring = str(pl.get("ring") or ring_of_bv(point.bv))
                    if it.ring == "neck" and not pl.get("ring"):
                        it.ring = self._neck_or_torso(it)
                    if it.ring not in RING_BVS:
                        raise RingNotFoundError(f"unknown ring {it.ring!r}; rings: {sorted(RING_BVS)}")
                except RingNotFoundError:
                    if pl.get("ring"):
                        raise
                    it.warnings.append(
                        f"WRAP_FALLBACK: roll needs a torso, neck, arm or leg point; {point.name} is on {point.bv}, "
                        "rolled around its body part instead"
                    )
                    it.mode = "around"
            self._surface(it)
            has_solved = isinstance(pl.get("solved"), dict)
            it.solve = not it.locked and (not incremental or not has_solved or not o.incremental)
        self.pairs = self._find_pairs()
        for src, clone, _ in self.pairs:
            src.partner, clone.partner = clone.name, src.name
            if not (src.locked or clone.locked) and src.solve != clone.solve:
                src.solve = clone.solve = True  # the new half is derived from the placed one; they move as one
        self._datum()
        self._groups()
        self._lone_halves()
        self._over()
        self._start_radii(incremental)
        self._reregister()
        self._over()

    def _over(self) -> None:
        """``over: {piece, point}``: a piece laid over another one (a pocket on a front, an interfacing on a collar)
        starts where its named ``point`` coincides with the same named point of the other piece (same arrangement point
        and wrap, so the same surface); the layering then lifts it one gap out. A placed piece keeps its place."""
        for it in self.items:
            ov = it.pl.get("over")
            p = it.piece
            if not ov or p is None or np.any(p.sign < 0):
                continue
            placed = isinstance(it.pl.get("solved"), dict)
            tgt = next((j for j in self.items if j.name == ov["piece"] and j is not it), None)
            if tgt is None or tgt.piece is None:  # placed before its host (one piece at a time): on its own point
                it.warnings.append(f"OVER_PENDING: {ov['piece']} is not placed; {it.name} stays on {it.point.name}")
                continue
            if tgt.point.name != it.point.name or tgt.mode != it.mode:
                it.warnings.append(
                    f"OVER_IGNORED: {it.name} is over {tgt.name} only on the same point and wrap "
                    f"({tgt.point.name}, {tgt.mode})"
                )
                continue

            def local(x: Item, name: str = str(ov["point"])) -> F64:
                q = next((q for q in x.info.bundle["named_points"] if q["name"] == name), None)
                if q is None:
                    raise UnknownNamedPointError(x.name, name, [q["name"] for q in x.info.bundle["named_points"]])
                xy = local_coords(np.array([q["xy_m"]], dtype=np.float64), x.ref, x.rotation, x.mirror)
                return np.asarray(xy[0], dtype=np.float64)

            d = tgt.piece.theta[1:] + local(tgt) - local(it)
            p.theta0[1:] = d  # the anchor; a placed piece keeps its place (its slide is measured from here)
            if not placed:
                p.theta[1:] = d
            p.update()
            for g in self.groups.values():  # a pair's clone follows its lead
                if g[0] is it and not placed:
                    g[1].piece.theta = p.theta * g[1].piece.sign  # type: ignore[union-attr]
                    g[1].piece.theta0 = p.theta0 * g[1].piece.sign  # type: ignore[union-attr]
                    g[1].piece.update()  # type: ignore[union-attr]

    def _reregister(self) -> None:
        """Registration again at the start radius (a radial move shifts heights a little on a tilted axis); a pair's
        clone follows its lead."""
        for it in self.items:
            p = it.piece
            if it.registration is None or p is None or p.surface is None or not it.solve or np.any(p.sign < 0):
                continue
            if isinstance(it.pl.get("solved"), dict):
                continue
            user = float(it.pl.get("shift_m", [0.0, 0.0])[1])
            th = p.theta.copy()
            th[2] = user
            p.theta[2] = p.theta0[2] = self._register(it, p.surface, th)
            p.update()
        for lead, clone in self.groups.values():
            assert lead.piece is not None and clone.piece is not None
            clone.piece.theta = lead.piece.theta * clone.piece.sign
            clone.piece.theta0 = lead.piece.theta0 * clone.piece.sign
            clone.piece.update()

    def _neck_or_torso(self, it: Item) -> str:
        """A piece on a neck point that is taller than the neck (a front or back panel cut up to the neckline) rolls
        on the torso ring with the rest of the garment; a collar stays on the neck ring."""
        vol = self.avatar.volumes.get("Neck")
        if vol is None:
            return "neck"
        outline = np.asarray(it.info.bundle["outline_m"], dtype=np.float64)
        ref = it.info.reference(str(it.pl.get("reference", "bbox_centre")))
        height = float(np.ptp(local_coords(outline, ref, self._rotation(it)[0], False)[:, 1]))
        neck = float(np.linalg.norm(vol.end - vol.start))
        if height <= neck:
            return "neck"
        it.warnings.append(
            f"RING_TORSO: {it.name} ({height * 100:.0f} cm tall) is taller than the neck ({neck * 100:.0f} cm); it "
            'rolls on the torso ring with the other panels (set "ring" to override)'
        )
        return "torso"

    def _rotation(self, it: Item) -> tuple[float, str]:
        """(in-plane rotation: ``rotation_deg`` plus ``orient``, why the orient part)."""
        extra, why = orient_rotation(it.info.bundle, it.pl.get("orient", "auto" if self.version >= 5 else "none"))
        return float(it.pl.get("rotation_deg", 0.0)) + extra, why

    def _surface(self, it: Item) -> None:
        o, pl, point = self.o, it.pl, it.point
        it.refined = self.cache.get(it.info.bundle, o.refine_max_edge, it.info.mesh())
        r = it.refined
        it.ref = it.info.reference(str(pl.get("reference", "bbox_centre")))
        it.mirror = bool(pl.get("mirror", False))
        it.rotation, it.orient_note = self._rotation(it)
        flip = bool(pl.get("flip_facing", False))
        g = self._geometry(it, r)
        local, faces = g["local"], g["faces"]
        eff = point.offset_m if pl.get("offset_m") is None else float(pl["offset_m"])
        if it.mode == "flat":
            surf, r0 = flat_surface(point, flip)
            r0 = eff
            r_min, r_max = 0.0, o.max_standoff
        else:
            if it.mode == "roll":
                c, a = ring_axis(self.avatar, it.ring)
                centre = it.ring in SYMMETRIC_RINGS and abs(float(point.position[0])) < 0.002
            else:
                vol = self.avatar.volumes[point.bv]
                c, a = vol.start, vol.end - vol.start
                centre = False
            surf, radius = cylinder_surface(point, c, a, flip, centre)
            r0 = radius + (eff - point.offset_m)
            it.axial0 = float((surf.origin - c) @ surf.e_y)
            gap = max(o.gap_piece, o.gap_layer or 0.0)
            r_min = (float(np.ptp(local[:, 0])) + gap) / (2.0 * np.pi)  # never wide enough to overlap itself
            r_max = surf.r_skin + o.max_standoff
            r0 = max(r0, r_min)
        shift = np.array(pl.get("shift_m", [0.0, 0.0]), dtype=np.float64)
        theta0 = np.array([r0, shift[0], shift[1]])
        it.tilt = tilt_angles(pl)
        it.axis_up = surf.e_y.copy()
        if any(it.tilt):  # rotate the frame (and a cylinder's axis) about the reference
            pivot = surf.map(np.zeros((1, 2)), theta0)[0]
            rm = tilt_matrix(surf.e_n, surf.e_x, surf.e_y, *it.tilt)
            surf = Surface(
                surf.kind, pivot + rm @ (surf.origin - pivot), rm @ surf.e_n, rm @ surf.e_x, rm @ surf.e_y, surf.r_skin
            )
        it.surface = surf
        it.registration = None if pl.get("over") else registration_spec(pl, it.info.bundle, self.version, point.bv)
        if it.registration is not None:
            theta0[2] = self._register(it, surf, theta0)
        theta = theta0.copy()
        if isinstance(pl.get("solved"), dict):
            sv = pl["solved"]
            theta = np.array([float(sv["radius_m"]), float(sv["shift_m"][0]), float(sv["shift_m"][1])])
            it.needed_radius = float(sv.get("needed_radius_m", 0.0))  # why it sits there (kept with the piece)
        it.piece = Piece(
            it.name,
            it.index,
            surf,
            local,
            faces,
            g["adjacency"],
            g["area"],
            theta,
            theta0,
            r_min,
            r_max,
            False,
            g["iter_idx"],
            x_mid=g["x_mid"],
            tree2d=g["tree2d"],
        )
        it.edges, it.outline = g["edges"], g["outline"]
        ring = self._ring_of(it)
        if ring in LIMB_RINGS or ring == "torso":  # R2: a limb slides off every other part, a torso piece off an arm
            names, tri_region = self._tri_regions()
            if ring == "torso":
                it.piece.other_tris = np.isin(tri_region, [names.index(a) for a in ("arm_L", "arm_R")])
            else:
                it.piece.other_tris = tri_region != names.index(ring)
        if it.registration is not None:
            it.piece.axial_weight = REGISTERED_AXIAL_WEIGHT
            if ring in LIMB_RINGS:  # a registered limb piece wraps the torso too: only the arms and the other limb
                names, tri_region = self._tri_regions()  # are slid away from
                other = {"leg_L": "leg_R", "leg_R": "leg_L", "arm_L": "arm_R", "arm_R": "arm_L"}[ring]
                it.piece.other_tris = np.isin(tri_region, [names.index(a) for a in ("arm_L", "arm_R", other)])
        it.piece.update()

    def _register(self, it: Item, surf: Surface, theta0: F64) -> float:
        """Axial shift ``t`` that puts the registered piece point (or edge) at its level height; records the heights
        in ``it.registration``. The user's ``shift_m`` along the axis is added on top."""
        reg = it.registration
        assert reg is not None
        if "piece_point" in reg:
            q = next((p for p in it.info.bundle["named_points"] if p["name"] == reg["piece_point"]), None)
            if q is None:
                raise UnknownNamedPointError(
                    it.name, str(reg["piece_point"]), [p["name"] for p in it.info.bundle["named_points"]]
                )
            xy = local_coords(np.array([q["xy_m"]], dtype=np.float64), it.ref, it.rotation, it.mirror)
            what = str(reg["piece_point"])
        else:
            edge = str(reg.get("edge", "top"))
            ol = local_coords(np.asarray(it.info.bundle["outline_m"], dtype=np.float64), it.ref, it.rotation, it.mirror)
            xy = edge_point(ol, edge)[None, :]
            what = f"{edge} edge"
        level = reg.get("level")
        target = level_height(self.avatar, str(level)) if level else float(it.point.position[1])
        target += float(reg.get("offset_m", 0.0))
        user = float(theta0[2])
        th = theta0.copy()
        for _ in range(3):  # world y is linear in t (slope e_y.y); repeated only for rounding
            y = float(surf.map(xy, th)[0, 1])
            th[2] += (target - y) / max(float(surf.e_y[1]), 1e-6)
        th[2] += user  # the user's own shift on top
        it.registration = {
            **reg,
            "registered": what,
            "level_y_m": target,
            "local_xy_m": [float(xy[0, 0]), float(xy[0, 1])],
            "shift_m": float(th[2] - user),
        }
        return float(th[2])

    def registration_offset(self, it: Item) -> float | None:
        """Height of the registered point now minus its level height (None: not registered)."""
        if it.registration is None or it.piece is None or it.piece.surface is None:
            return None
        xy = np.array([it.registration["local_xy_m"]])
        y = float(it.piece.surface.map(xy, it.piece.theta)[0, 1])
        return y - float(it.registration["level_y_m"])

    def _ring_of(self, it: Item) -> str:
        """The body region a rolled piece wraps (its ring; for ``around`` the ring of its point's volume), else ''."""
        if it.mode == "roll":
            return it.ring
        if it.mode == "around":
            try:
                return ring_of_bv(it.point.bv)
            except RingNotFoundError:
                return ""
        return ""

    def _tri_regions(self) -> tuple[list[str], I64]:
        """(region names, region index of every avatar triangle, -1 for head, hands and feet); cached on the avatar."""
        if "tri_regions" not in self.avatar.cache:
            regions = self.shells().regions
            names = sorted(regions)
            vr = np.full(len(self.avatar.vertices), -1, dtype=np.int64)
            for k, n in enumerate(names):
                vr[regions[n].vertex_ids] = k
            self.avatar.cache["tri_regions"] = (names, vr[self.avatar.faces[:, 0]])
        out: tuple[list[str], I64] = self.avatar.cache["tri_regions"]
        return out

    def _geometry(self, it: Item, r: RefinedPiece) -> dict[str, Any]:
        """Static data of a piece in local coordinates (cached by content: outline, refinement, reference, rotation,
        mirror), so that an incremental call does not rebuild it for the pieces that did not change."""
        key = (
            RefineCache.key(it.info.bundle, self.o.refine_max_edge),
            tuple(np.round(it.ref, 12)),
            round(it.rotation, 12),
            it.mirror,
        )
        it.geom_key = key
        g = self.geom_cache.get(key)
        if g is None:
            local = local_coords(r.vertices[:, :2], it.ref, it.rotation, it.mirror)
            faces = np.ascontiguousarray(r.faces[:, [0, 2, 1]]) if it.mirror else r.faces
            u, w = local[faces[:, 1]] - local[faces[:, 0]], local[faces[:, 2]] - local[faces[:, 0]]
            tri_area = u[:, 0] * w[:, 1] - u[:, 1] * w[:, 0]
            outline = np.asarray(it.info.bundle["outline_m"], dtype=np.float64)
            g = {
                "local": local,
                "faces": faces,
                "adjacency": vertex_faces(faces, len(local)),
                "area": float(0.5 * np.abs(tri_area).sum()),
                "iter_idx": iteration_samples(local, r.boundary),
                "x_mid": float(0.5 * (local[:, 0].min() + local[:, 0].max())),
                "tree2d": cKDTree(local),
                "edges": unique_edges(faces),
                "outline": local_coords(outline, it.ref, it.rotation, it.mirror),
            }
            if len(self.geom_cache) > 200:
                self.geom_cache.clear()
            self.geom_cache[key] = g
        return g

    def _find_pairs(self) -> list[tuple[Item, Item, bool]]:
        """A piece and its mirrored clone placed as a pair: on the same centre point (``same``) or on left/right
        counterpart points; same mode and reference, rotation equal (same point) or negated (counterparts)."""
        out: list[tuple[Item, Item, bool]] = []
        used: set[int] = set()
        for it in self.items:
            if it.piece is None or it.index in used or it.name not in self.sources.values():
                continue
            for clone in self.items:
                if clone.source != it.name or clone.piece is None or clone.index in used or clone.mode != it.mode:
                    continue
                if clone.pl.get("reference", "bbox_centre") != it.pl.get("reference", "bbox_centre"):
                    continue
                same = clone.point.name == it.point.name and abs(float(it.point.position[0])) < 0.002
                try:
                    side = clone.point.name == counterpart_point(it.point.name, self.avatar.points) != it.point.name
                except Exception:  # noqa: BLE001 - no counterpart: not a side pair
                    side = False
                if not (same and _same_angle(clone.rotation, it.rotation)) and not (
                    side and _same_angle(clone.rotation, -it.rotation)
                ):
                    continue
                used |= {it.index, clone.index}
                out.append((it, clone, same))
                break
        return out

    def _groups(self) -> None:
        """A pair moves as one, ``s`` mirrored (the clone follows its source exactly). When one half was placed
        earlier, the pair starts from that half's place: the new half is its mirror image."""
        for gid, (it, clone, same) in enumerate(p for p in self.pairs if p[0].solve and p[1].solve):
            assert it.piece is not None and clone.piece is not None
            it.piece.group = clone.piece.group = gid
            clone.piece.sign = np.array([1.0, -1.0, 1.0])
            src_solved, clone_solved = (isinstance(x.pl.get("solved"), dict) for x in (it, clone))
            if clone_solved and not src_solved:  # the clone was placed first: the source is its mirror image
                it.piece.theta = clone.piece.theta * clone.piece.sign
            if same:  # the gap across the centre line stays the gap: the pair moves out rather than apart
                it.piece.slide_weight = clone.piece.slide_weight = CENTRE_SLIDE_WEIGHT
                shift, direction = self._centre_shift(it)
                it.piece.theta0[1] += shift  # the two copies start half a gap away from the centre line each
                if not (src_solved or clone_solved) or direction * it.piece.theta[1] < direction * it.piece.theta0[1]:
                    it.piece.theta[1] = it.piece.theta0[1]  # new, or placed alone inside the half gap: opened
                assert it.surface is not None  # local +x runs along world e_x, which is +-x on the centre line
                it.piece.pair_side = direction * (1.0 if it.surface.e_x[0] >= 0 else -1.0)
            else:  # a side pair: the side of its point (the solver slides it outward if the halves overlap)
                it.piece.pair_side = 1.0 if float(it.point.position[0]) >= 0 else -1.0
            clone.piece.theta = it.piece.theta * clone.piece.sign
            clone.piece.theta0 = it.piece.theta0 * clone.piece.sign
            clone.piece.update()
            it.piece.update()
            self.groups[gid] = [it, clone]

    def _lone_halves(self) -> None:
        """A new half of a pair placed alone on a centre point leaves half a gap at the centre line already, so that
        its other half joins it later without moving it. A placed one keeps its place; only its anchor is set."""
        for it in self.items:
            p = it.piece
            if p is None or p.group >= 0:
                continue
            if abs(float(it.point.position[0])) >= 0.002 or not (it.source or it.name in self.sources.values()):
                continue
            new = it.solve and not isinstance(it.pl.get("solved"), dict)
            p.theta0[1] += self._centre_shift(it, warn=new)[0]
            if new:
                p.theta[1] = p.theta0[1]
                p.update()

    def _centre_shift(self, it: Item, warn: bool = True) -> tuple[float, float]:
        """(shift ``s``, direction): the shift that puts the centre-line edge of a pair half half a gap from the
        centre line, and the side the half lies on (+1: +x of the line, local coordinates).

        The side comes from the mirror axis of the pair (the half lies on one side of it), else from where most of
        the piece lies about its reference. A reference that is not on that edge is reported (REFERENCE_ADJUSTED):
        the pair is then placed by its edge, not by the reference."""
        assert it.piece is not None
        direction = self._join_side(it)
        if direction is None:
            direction = 1.0 if float(np.mean(it.piece.local[:, 0])) >= 0 else -1.0
        xs = it.outline[:, 0]
        inner = float(xs.min() if direction > 0 else xs.max())
        if abs(inner) > 0.002 and warn:
            ref = str(it.pl.get("reference", "bbox_centre"))
            it.warnings.append(
                f"REFERENCE_ADJUSTED: the reference {ref} of {it.name} is {abs(inner) * 100:.1f} cm from its "
                "centre-line edge; the pair is placed by that edge (pick a reference on the mirror axis to anchor it)"
            )
        return direction * (0.5 * self.o.gap_piece + self.o.requirements().margin) - inner, direction

    def _join_side(self, it: Item) -> float | None:
        """+1 / -1: the side (local x) of the pair's mirror axis this half lies on; None if the axis is unknown or
        runs through the piece (a ``bbox_center`` axis)."""
        from .mirror import axis_line

        if it.source:
            clone, sign = it.info.bundle, 1.0
        else:
            ids = sorted(k for k, v in self.sources.items() if v == it.name and k in self.pieces)
            if not ids:
                return None
            clone, sign = self.pieces[ids[0]].bundle, -1.0
        if not clone.get("axis") or abs(it.rotation) > 1e-9:
            return None
        try:
            p0, d = axis_line(clone, clone["axis"])
        except Exception:  # noqa: BLE001 - an axis that cannot be resolved tells nothing
            return None
        n = np.array([d[1], -d[0]])
        n = -n if n[0] < 0 else n
        side = float((np.mean(np.asarray(clone["outline_m"], dtype=np.float64), axis=0) - p0) @ n)
        return None if abs(side) < 0.01 else sign * float(np.sign(side))

    def _datum(self) -> None:
        """Torso panels of one pattern keep the heights they have in the drafting frame (``pattern_origin_m``).

        The first placed panel (else the tallest) is the datum; every new panel of similar height that spans the
        datum's drafting line starts with the axial shift that puts its drafting lines on the datum's, so front and back
        meet on the same chest line whatever their references. Pockets, yokes and other small pieces keep their
        reference (``DATUM_HEIGHT_RATIO``)."""
        cands = [
            it
            for it in self.items
            if it.mode == "roll"
            and it.ring == "torso"
            and it.piece is not None
            and it.info.bundle.get("pattern_origin_m") is not None
            and abs(it.rotation) < 1e-9
        ]
        if len(cands) < 2:
            return

        def height(x: Item) -> float:
            return float(np.ptp(x.outline[:, 1]))

        placed = [x for x in cands if isinstance(x.pl.get("solved"), dict) or not x.solve]
        registered = [x for x in cands if x.registration is not None]
        # a registered panel is the datum; else the first placed panel keeps it (a placed panel's anchor stays where it
        # was); else the tallest
        if registered:
            lead = registered[0]
        elif placed:
            lead = min(placed, key=lambda x: x.index)
        else:
            lead = max(cands, key=lambda x: (height(x), -x.index))
        assert lead.piece is not None
        y_datum = float(lead.info.bundle["pattern_origin_m"][1] + lead.ref[1])  # drafting height of its reference
        a_datum = lead.axial0 + float(lead.piece.theta[2])  # where that height lies along the axis
        for it in cands:
            p = it.piece
            assert p is not None
            if it is lead or it.registration is not None or it.pl.get("over"):
                continue
            if min(height(it), height(lead)) < DATUM_HEIGHT_RATIO * max(height(it), height(lead)):
                continue
            y_local = y_datum - float(it.info.bundle["pattern_origin_m"][1] + it.ref[1])  # the datum line, local y
            if not float(it.outline[:, 1].min()) <= y_local <= float(it.outline[:, 1].max()):
                continue
            dt = a_datum - it.axial0 - y_local  # on top of the user's own shift
            if abs(dt) < 1e-9:
                continue
            p.theta0[2] += dt  # the anchor; a placed piece keeps its place (its slide is measured from here)
            if not it.solve:
                continue
            if not isinstance(it.pl.get("solved"), dict):
                p.theta[2] = p.theta0[2]
                p.update()
            self.notes.append(
                f"DATUM: {it.name} moved {dt * 100:+.1f} cm along the body to keep its drafting heights with "
                f"{lead.name} (pattern_origin_m)"
            )

    # -- start radii (D2) ------------------------------------------------------------------------------------------
    def clear_radius(self, it: Item) -> float:
        """Smallest radius at which every avatar vertex under the piece's footprint (outline buffered by the clearance)
        is ``clearance`` inside the cylinder. A larger radius shrinks the angular footprint, so this clears a T-pose arm
        beside a torso panel; vertices still under the footprint at ``max_standoff`` (the torso under a sleeve cap)
        cannot be cleared by a radius and are left to the slides of the solver. A limb piece only counts its own limb:
        another body part under it (the torso at the arm root) is cleared by sliding along the limb (``_anchor``), never
        by a larger radius. Bisection."""
        p, surf = it.piece, it.surface
        assert p is not None and surf is not None and it.refined is not None
        ring = self._ring_of(it)
        verts = self.avatar.vertices
        if ring in LIMB_RINGS:
            regions = self.shells().regions
            ids = regions[ring].vertex_ids
            if it.registration is not None and "torso" in regions:  # a trouser panel at the waist wraps the torso
                ids = np.concatenate([ids, regions["torso"].vertex_ids])
            verts = verts[ids]
        c = self.o.clearance_body
        poly = _polygon(it.outline).buffer(c)

        def blocking(radius: float) -> NDArray[np.bool_]:
            dev = surf.develop(verts, np.array([radius, p.theta[1], p.theta[2]]), p.x_mid)
            inside = shapely.contains_xy(poly, dev[:, 0], dev[:, 1])
            return np.asarray(inside & (dev[:, 2] > -c))

        lo, hi = p.r_min, p.r_max
        stuck = blocking(hi)  # not clearable by any allowed radius
        if not (blocking(lo) & ~stuck).any():
            return lo
        for _ in range(24):
            mid = 0.5 * (lo + hi)
            lo, hi = (lo, mid) if not (blocking(mid) & ~stuck).any() else (mid, hi)
        return hi

    def _footprint(self, it: Item, scale: float) -> Polygon:
        """Developed footprint of a roll piece in ring coordinates (arc at radius ``scale`` from the ring's front)."""
        p, surf = it.piece, it.surface
        assert p is not None and surf is not None and it.refined is not None
        c, a = ring_axis(self.avatar, it.ring)
        front = np.array([0.0, 0.0, 1.0]) - a[2] * a
        if np.linalg.norm(front) < 1e-6:
            front = np.array([0.0, 1.0, 0.0]) - a[1] * a
        front /= np.linalg.norm(front)
        side = np.cross(a, front)
        psi = float(np.arctan2(surf.e_n @ side, surf.e_n @ front))
        sgn = 1.0 if surf.e_x @ np.cross(a, surf.e_n) > 0 else -1.0
        b = it.outline
        pts = np.column_stack([scale * psi + sgn * (b[:, 0] + p.theta[1]), it.axial0 + b[:, 1] + p.theta[2]])
        return _polygon(pts)

    def _anchor(self, it: Item, radius: float) -> None:
        """Axial registration of a limb piece (``anchor``: auto, point, root, tip; see ``docs/alignment_limbs.md``).

        ``root``: the piece slides along the limb (its top edge, the sleeve cap or the trouser waist, first) to the
        position closest to the torso where no torso vertex is under its footprint within ``clearance`` at ``radius``,
        and no further in than ``clearance`` out from the limb joint (shoulder, hip): the cap sits at the limb root.
        ``tip``: its bottom edge on the end of the limb (wrist, ankle). ``auto``: root when the piece is longer than the
        point's volume or reaches into the torso at the point, else point. The user's ``shift_m`` is added on top. Sets
        the anchor ``theta0[2]`` (and ``theta[2]`` unless the piece is placed)."""
        p, surf = it.piece, it.surface
        ring = self._ring_of(it)
        mode = str(it.pl.get("anchor", "auto"))
        if p is None or surf is None or ring not in LIMB_RINGS or mode == "point" or np.any(p.sign < 0):
            return
        if it.registration is not None or it.pl.get("over"):  # registered on a level, or over a piece: kept
            return
        t_user = float(it.pl.get("shift_m", [0.0, 0.0])[1])
        ys = it.outline[:, 1]
        extent = float(ys.max() - ys.min())
        vol = self.avatar.volumes[it.point.bv]
        bv_len = float(np.linalg.norm(vol.end - vol.start))
        limb = "arm" if ring.startswith("arm") else "leg"
        if mode == "tip":
            tip = self.avatar.volumes[RING_BVS[ring][-1]].end
            t = float((tip - surf.origin) @ surf.e_y) - float(ys.min())
            why = f"bottom edge at the end of the {limb} ({RING_BVS[ring][-1]})"
        else:
            torso = self.avatar.vertices[self.shells().regions["torso"].vertex_ids]
            c = self.o.clearance_body
            poly = _polygon(it.outline).buffer(c)

            def blocked(t: float) -> bool:
                dev = surf.develop(torso, np.array([radius, p.theta0[1], t]), p.x_mid)
                return bool((shapely.contains_xy(poly, dev[:, 0], dev[:, 1]) & (dev[:, 2] > -c)).any())

            at_point = blocked(t_user)
            if mode == "auto" and not at_point and extent <= bv_len + 1e-9:
                return
            n = int(ANCHOR_RANGE / ANCHOR_STEP)
            if at_point:  # away from the torso until clear
                k = next((k for k in range(1, n + 1) if not blocked(t_user - k * ANCHOR_STEP)), None)
                if k is None:
                    it.warnings.append(f"ANCHOR_FAILED: {it.name} reaches into the torso anywhere along the {limb}")
                    return
                lo, hi = t_user - k * ANCHOR_STEP, t_user - (k - 1) * ANCHOR_STEP
            else:  # toward the torso until it would touch it
                k = next((k for k in range(1, n + 1) if blocked(t_user + k * ANCHOR_STEP)), None)
                if k is None:
                    return
                lo, hi = t_user + (k - 1) * ANCHOR_STEP, t_user + k * ANCHOR_STEP
            for _ in range(12):
                mid = 0.5 * (lo + hi)
                lo, hi = (mid, hi) if not blocked(mid) else (lo, mid)
            # never inboard of the limb joint (shoulder, hip) offset outward by the clearance: the limb root
            joint = self.avatar.volumes[RING_BVS[ring][0]].start
            t = min(lo, float((joint - surf.origin) @ surf.e_y) - c - float(ys.max()))
            cause = (
                "at the point it reached into the torso"
                if at_point
                else f"{extent * 100:.0f} cm long, longer than {vol.name} ({bv_len * 100:.0f} cm)"
            )
            why = f"top edge at the {limb} root, {c * 100:.0f} cm out from the joint and the torso ({cause})"
        dt = t - t_user
        p.theta0[2] = t
        if not isinstance(it.pl.get("solved"), dict):
            p.theta[2] = t
            p.update()
        mode_out = "tip" if mode == "tip" else "root"
        it.anchor = {"mode": mode_out, "shift_m": round(dt, 6), "reason": f"{why}: {dt * 100:+.1f} cm along the {limb}"}

    def _sync_clones(self) -> None:
        """The clone of a pair follows its lead exactly (``s`` mirrored), also its anchor.

        A pair to be solved starts apart: if its halves overlap at the start they slide outward first (``pair_slide``),
        and the radius that clears the body is measured there, where the pair has to be (both halves share it; a
        placed half solved again with its new other half is measured too). The ring radius is raised to it, so the
        solver does not have to find it in coarse radius steps."""
        req = self.o.requirements()
        for lead, clone in self.groups.values():
            lp, cp = lead.piece, clone.piece
            assert lp is not None and cp is not None
            if lp.pair_side and lp.surface is not None:
                lp.theta[1] = pair_slide(lp, lp.theta, self.o.max_shift, pair_half_gap(req))
                lp.update()
                if lp.surface.kind == "cylinder":
                    need = max(self.clear_radius(lead), self.clear_radius(clone) if clone.surface else 0.0)
                    lead.needed_radius = clone.needed_radius = max(lead.needed_radius, clone.needed_radius, need)
                    if need > lp.theta[0] + 0.002:
                        self._set_radius(lead, need)
                        lp.theta[1] = pair_slide(lp, lp.theta, self.o.max_shift, pair_half_gap(req))
                        lp.update()
            cp.theta = lp.theta * cp.sign
            cp.theta0 = lp.theta0 * cp.sign
            cp.limits_hit, cp.pair_room = set(lp.limits_hit), lp.pair_room
            clone.anchor = lead.anchor
            cp.update()

    def _layers(self, its: list[Item], scale: float) -> None:
        """Start layers: a piece mostly covered by a larger piece of its ring starts one layer further out."""
        length = 2.0 * np.pi * scale
        polys = {it.index: self._footprint(it, scale) for it in its}
        done: list[Item] = []
        for it in sorted(its, key=lambda i: (-polys[i.index].area, i.index)):
            if it.user_layer is not None:
                it.layer, it.layer_reason = it.user_layer, "user" if it.user_layer else ""
            else:
                best = 0
                pi = polys[it.index]
                for jt in done:
                    cover = max(pi.intersection(translate(polys[jt.index], k * length, 0.0)).area for k in (-1, 0, 1))
                    if cover / max(pi.area, 1e-12) >= COVER and jt.layer + 1 > best:
                        best, it.layer_reason = jt.layer + 1, f"over {jt.name}"
                it.layer = best
            done.append(it)

    def ring_names(self) -> list[str]:
        """Rings of the rolled pieces."""
        return sorted({it.ring for it in self.items if it.mode == "roll"})

    def fit_radius(self, its: list[Item]) -> float:
        """Row sums: the pieces of layer 0 side by side with a gap each need ``sum(w) + n gap <= 2 pi R``.

        Half of a mirror pair whose other half is not placed yet counts twice on a symmetric ring: the ring is sized
        for the whole garment, so the other half joins later without anything climbing a layer."""
        base = [it for it in its if it.layer == 0]
        if not base:
            return 0.0
        rows: dict[int, list[float]] = {}
        for it in base:
            p = it.piece
            assert p is not None
            b = it.outline
            lone = it.ring in SYMMETRIC_RINGS and not it.partner and bool(it.source or it.name in self.sources.values())
            ys = np.arange(float(b[:, 1].min()), float(b[:, 1].max()) + ROW_STEP, ROW_STEP)
            for y in ys:
                w = row_width(b, float(y))
                if w > 1e-4:
                    rows.setdefault(int(round((it.axial0 + y + p.theta[2]) / ROW_STEP)), []).extend([w] * (1 + lone))
        need = max((sum(ws) + len(ws) * self.o.gap_piece for ws in rows.values()), default=0.0)
        return need / (2.0 * np.pi)

    def _start_radii(self, incremental: bool) -> None:
        """Start radius of the rolled pieces to be solved (``solved`` ones keep theirs). ``around`` pieces start at the
        smallest radius from their point's own that clears the body under them; ``roll`` pieces share their ring's.

        The ring radius fits every layer-0 piece of the ring side by side, placed ones included (``fit_radius``), and
        clears the body under the new ones. A placed layer-0 ring that is too small for that moves out as a whole
        (incremental): the garment stays concentric instead of the new pieces climbing a layer. The body alone never
        moves placed pieces: a new piece that needs more clearance sits further out by just that much."""
        o = self.o
        gl = o.gap_layer if o.gap_layer is not None else o.gap_piece
        for it in self.items:
            if it.mode == "around" and it.solve and it.piece is not None and not isinstance(it.pl.get("solved"), dict):
                r_est = max(it.piece.theta0[0], it.piece.r_min)
                self._anchor(it, r_est)
                radius = max(it.piece.theta0[0], self.clear_radius(it))
                if radius > r_est + 0.002:
                    self._anchor(it, radius)
                self._set_radius(it, radius)
        for it in self.items:  # placed limb pieces: the anchor their slide is measured from
            if it.piece is not None and (isinstance(it.pl.get("solved"), dict) or not it.solve):
                self._anchor(it, float(it.piece.theta[0]))
        rings = sorted({it.ring for it in self.items if it.mode == "roll"})
        for ring in rings:
            ring_items = [it for it in self.items if it.mode == "roll" and it.ring == ring and it.piece is not None]
            todo = [it for it in ring_items if it.solve and not isinstance(it.pl.get("solved"), dict)]
            if not todo:
                continue
            placed = [it for it in ring_items if not it.solve]  # fixed: solved earlier (incremental) or locked
            for it in ring_items:  # layers are judged where the pieces will sit: all side by side
                it.layer, it.layer_reason = 0, ""
            scale = max(
                [self.fit_radius(ring_items)]
                + [it.piece.theta0[0] for it in todo if it.piece is not None]
                + [min((it.piece.theta[0] for it in placed if it.piece is not None), default=0.0)]
            )
            self._layers(ring_items, scale)
            fit = self.fit_radius(ring_items)
            if ring in LIMB_RINGS:
                for it in todo:
                    assert it.piece is not None
                    self._anchor(it, max(fit, it.piece.theta0[0]))
            movable = [it for it in placed if not it.locked and it.piece is not None]
            base = [it.piece.theta[0] for it in movable if it.layer == 0 and it.piece is not None]
            if incremental and base and fit > min(base) + GROW_TOL:
                self._grow(ring, movable, fit - min(base))
                base = [r + fit - min(base) for r in base]
            radius = max(
                [fit, min(base, default=0.0)]
                + [max(self.clear_radius(it), it.piece.theta0[0]) for it in todo if it.layer == 0 and it.piece]
            )
            self.ring_radius[ring] = radius
            for it in todo:
                r = radius + it.layer * gl
                r = r if it.layer == 0 else max(r, self.clear_radius(it))
                if ring in LIMB_RINGS and r > fit + 0.002:
                    self._anchor(it, r)
                self._set_radius(it, r)
        self._sync_clones()

    def _grow(self, ring: str, its: list[Item], dr: float) -> None:
        """Move placed pieces of a ring out by ``dr`` together (``s`` and ``t`` kept, layers kept)."""
        for it in its:
            p = it.piece
            assert p is not None
            p.theta[0] = p.theta0[0] = min(float(p.theta[0]) + dr, max(p.r_max, p.r_min))
            p.update()
            it.regrown = True
        self.notes.append(
            f"RING_GROWN: the {ring} ring grew by {dr * 100:.1f} cm to fit the new pieces side by side; moved out: "
            + ", ".join(it.name for it in its)
        )
        for g in self.groups.values():  # a group shares the radius of its lead
            lead, clone = g
            assert lead.piece is not None and clone.piece is not None
            r = max(lead.piece.theta[0], clone.piece.theta[0])
            for it in g:
                assert it.piece is not None
                it.piece.theta[0] = it.piece.theta0[0] = r
                it.piece.update()

    def _set_radius(self, it: Item, radius: float) -> None:
        p = it.piece
        assert p is not None
        it.needed_radius = radius
        if radius > p.r_max + 1e-9:
            it.warnings.append(
                f"STANDOFF_EXCEEDED: {it.name} needs a radius of {radius * 100:.1f} cm, but max_standoff allows "
                f"{p.r_max * 100:.1f} cm; it starts at the limit"
            )
            p.limits_hit.add("max_standoff")
        p.theta[0] = p.theta0[0] = float(np.clip(radius, p.r_min, max(p.r_max, p.r_min)))
        p.update()

    # -- solving ---------------------------------------------------------------------------------------------------
    def solve(self, incremental: bool) -> None:
        """Solve the pieces marked ``solve``: one block at a time in spec order (incremental) or all together."""
        req = self.o.requirements()
        placed = [it for it in self.items if it.piece is not None and not it.solve]
        obstacles = [it.piece for it in placed if it.piece is not None] + self._distorting()
        if incremental:
            blocks: list[list[Item]] = []
            seen: set[int] = set()
            for it in self.items:
                if not it.solve or it.index in seen:
                    continue
                assert it.piece is not None
                block = self.groups[it.piece.group] if it.piece.group >= 0 else [it]
                seen.update(b.index for b in block)
                blocks.append(block)
            for block in blocks:
                for b in block:
                    assert b.piece is not None
                    b.piece.moving = True
                moving = [b.piece for b in block if b.piece is not None]
                self._run(obstacles + moving, req)
                for p in moving:
                    p.moving = False
                obstacles += moving
        else:
            moving = []
            for it in self.items:
                if it.solve and it.piece is not None:
                    it.piece.moving = True
                    moving.append(it.piece)
            if moving:
                self._run(obstacles + moving, req)
            for p in moving:
                p.moving = False
        for it in self.items:  # the parameters are returned rounded: the layout is the one a later call rebuilds
            if it.solve and it.piece is not None:
                it.piece.theta = np.round(it.piece.theta, 9)
                it.piece.update()

    def _distorting(self) -> list[Piece]:
        """Pieces placed by a distorting wrap (body_slice): fixed obstacles without a surface."""
        out = []
        for it in self.items:
            if it.placed is not None:
                v, f = it.placed.vertices, it.placed.faces
                out.append(
                    Piece(
                        it.name,
                        it.index,
                        None,
                        np.zeros((len(v), 2)),
                        f,
                        vertex_faces(f, len(v)),
                        0.0,
                        np.zeros(3),
                        np.zeros(3),
                        0.0,
                        0.0,
                        False,
                        np.zeros(0, dtype=np.int64),
                        world=v,
                    )
                )
        return out

    def _run(self, pieces: list[Piece], req: Requirements) -> None:
        """Widening stages until nothing is violated:

        1. slides within ``STAGE1_SHIFT`` (pieces stay near their anchors);
        2. pieces of one ring that still collide are layered (one rolls ``gap`` further out than the other, which on
           coaxial cylinders is an exact separation) and the moving ones restart from their anchors, slides up to
           ``max_shift``;
        3. larger radii, step by step up to ``max_standoff``;
        4. a pass with every vertex.
        Whatever still remains is reported by the layout check, with the limit that stopped the solver.
        """
        solver = Solver(pieces, self.body, req)
        st = solver.run(STAGE1_ITER, min(STAGE1_SHIFT, self.o.max_shift))
        self.iterations += st.iterations
        if not st.converged:
            if solver.layer_colliding():
                st = solver.run(STAGE2_ITER, self.o.max_shift)
                self.iterations += st.iterations
            else:
                st = solver.run(STAGE2_ITER, self.o.max_shift)
                self.iterations += st.iterations
        rounds = 0
        while not st.converged and rounds < RADIUS_ROUNDS:
            grown = False
            rows = solver.rows(False)  # only the pieces that still collide move out
            stuck = {r[0] for r in rows} | {r[2] for r in rows}
            for i, p in enumerate(pieces):
                if solver.block(i) not in stuck:
                    continue
                if p.moving and p.surface is not None and (p.group < 0 or np.all(p.sign > 0)) and p.theta[0] < p.r_max:
                    p.theta[0] = min(p.theta[0] + RADIUS_STEP, p.r_max)
                    if p.theta[0] >= p.r_max:
                        p.limits_hit.add("max_standoff")
                    grown = True
            if not grown:
                break
            solver._sync_groups()
            for p in pieces:
                p.update()
            st = solver.run(RADIUS_ITER, self.o.max_shift)
            self.iterations += st.iterations
            rounds += 1
        st = solver.run(FINE_ITER, self.o.max_shift, fine=True)
        self.iterations += st.iterations
        if not st.converged:
            for p in pieces:
                if p.moving and not p.limits_hit:
                    p.limits_hit.add("iterations")

    # -- checks ----------------------------------------------------------------------------------------------------
    def placed_views(self) -> list[Placed]:
        """Every placed mesh for the layout check (spec order)."""
        out = []
        for it in self.items:
            if it.piece is not None:
                p = it.piece
                cyl = None
                if p.surface is not None and p.surface.kind == "cylinder":
                    longest = float(np.linalg.norm(p.local[it.edges[:, 0]] - p.local[it.edges[:, 1]], axis=1).max())
                    cyl = (p.surface.origin, p.surface.e_y, float(p.theta[0]), longest**2 / (8.0 * float(p.theta[0])))
                out.append(Placed(it.name, p.world, p.faces, it.edges, p.surface.normal if p.surface else None, cyl))
            else:
                v, f = it.placed.vertices, it.placed.faces
                out.append(Placed(it.name, v, f, unique_edges(f), None))
        return out

    def run_check(self) -> LayoutCheck:
        """Exact distances of the final layout (cached by content between calls of one engine)."""
        views = self.placed_views()
        adj = [
            it.piece.adjacency if it.piece is not None else vertex_faces(v.faces, len(v.world))
            for it, v in zip(self.items, views, strict=True)
        ]
        keys = [_digest(v.world, v.faces, self.o.clearance_body) for v in views]
        need = max(self.o.gap_piece, self.o.gap_layer or 0.0)
        self.check = layout_check(views, self.body, self.o.clearance_body, adj, self.check_cache, keys, need)
        return self.check

    def piece_key(self, it: Item) -> bytes | None:
        """Content key of a placed piece (geometry, surface, parameters): equal keys mean identical placed meshes."""
        if it.piece is None or it.piece.surface is None:
            return None
        import hashlib

        sf, th = it.piece.surface, it.piece.theta
        h = hashlib.sha1(repr((it.geom_key, sf.kind, it.mode, it.ring, self.o.allow_distortion)).encode())
        for arr in (sf.origin, sf.e_n, sf.e_x, sf.e_y, th, np.array([sf.r_skin])):
            h.update(np.ascontiguousarray(arr, dtype=np.float64).tobytes())
        return h.digest()

    def required(self, facing: bool) -> float:
        """Required distance of a pair (``gap_layer`` only for facing pairs, and only if set lower)."""
        return self.o.requirements().gap_for(facing)

    def violations(self) -> list[dict[str, Any]]:
        """Clearance and gap violations of the final layout (with the limit that stopped the solver)."""
        assert self.check is not None
        o = self.o
        out: list[dict[str, Any]] = []
        for i, (d, at) in sorted(self.check.body.items()):
            if d < o.clearance_body - TOLERANCE:
                it = self.items[i]
                out.append(
                    {
                        "kind": "CLEARANCE_BODY",
                        "pieces": [it.name],
                        "value_m": round(d, 6),
                        "required_m": o.clearance_body,
                        "at_m": [round(float(x), 6) for x in at],
                        "limit": self._limit([it]),
                    }
                )
        for (i, j), (d, at, facing) in sorted(self.check.pairs.items()):
            need = self.required(facing)
            if d < need - TOLERANCE:
                a, b = self.items[i], self.items[j]
                out.append(
                    {
                        "kind": "GAP_PIECES",
                        "pieces": [a.name, b.name],
                        "value_m": round(d, 6),
                        "required_m": need,
                        "at_m": [round(float(x), 6) for x in at],
                        "limit": self._limit([a, b]),
                    }
                )
        return out

    @staticmethod
    def _limit(its: list[Item]) -> str:
        hits: set[str] = set()
        for it in its:
            if it.piece is not None and (it.solve or it.piece.limits_hit):
                hits |= it.piece.limits_hit
            elif it.piece is None or not it.solve:
                hits.add("fixed")
        order = ("max_standoff", "max_shift", "iterations", "fixed")
        return ",".join(h for h in order if h in hits) or "none"


def _same_angle(a: float, b: float) -> bool:
    """Equal directions (degrees, modulo 360: a mirrored quarter turn of 90 is 270)."""
    return abs((a - b + 180.0) % 360.0 - 180.0) < 1e-9


def _polygon(pts: F64) -> Polygon:
    """A valid polygon of an outline (a self-touching outline, like a placket slit, is repaired by ``buffer(0)``)."""
    poly = Polygon(pts)
    if not poly.is_valid:
        poly = poly.buffer(0)
        if poly.is_empty or not poly.is_valid or poly.geom_type != "Polygon":
            poly = Polygon(pts).convex_hull
    return poly


def _digest(world: F64, faces: I64, clearance: float) -> bytes:
    import hashlib

    h = hashlib.sha1(np.ascontiguousarray(np.round(world, 9)).tobytes())
    h.update(np.ascontiguousarray(faces).tobytes())
    h.update(f"{clearance:.6f}".encode())
    return h.digest()

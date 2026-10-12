"""Body shells: what pieces wrap onto. Metres.

A *region* (torso, neck, arm_L, arm_R, leg_L, leg_R) has an axis and a set of avatar vertices. For a position ``p``
along the
axis the cross-section is the **convex hull** of the region vertices within a window of +- ``window`` metres along the
axis.
Because the window is at least the clearance plus about one triangle edge, every body surface point outside it is
farther than the clearance (axially), and every
vertex inside it is inside the hull, so the hull offset outward by ``h >= clearance`` is at least ``h`` from the body.

The offset of a convex polygon by ``h`` is its edges moved out by ``h`` joined by circular arcs of radius ``h``; its
length is
``perimeter + 2 pi h`` (Steiner), so loop lengths, arc positions and the offset curve are exact and the offset never
self-intersects. Directions are expressed as the angle of the outward normal ``phi`` in the plane of the section.

Regions follow the T-pose: an arm is the straight axis shoulder -> wrist (upper arm and forearm merged, so a sleeve
may cross
the elbow), a leg hip -> ankle. They are found from the bounding-volume files: each avatar vertex belongs to the nearest
capsule (distance to the axis minus radius); the first fraction of the upper arm (the shoulder cap) counts as torso.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d, maximum_filter1d
from scipy.spatial import ConvexHull

from .errors import RingNotFoundError
from .loaders import Avatar

F64 = NDArray[np.float64]
I64 = NDArray[np.int64]
SHOULDER_CAP = 0.2  # fraction of the upper arm (from the shoulder joint) that counts as torso
_RINGS = {
    "Chest": "torso",
    "Waist": "torso",
    "Hip": "torso",
    "Neck": "neck",
    "UpperArm_L": "arm_L",
    "Forearm_L": "arm_L",
    "UpperArm_R": "arm_R",
    "Forearm_R": "arm_R",
    "Thigh_L": "leg_L",
    "Shin_L": "leg_L",
    "Thigh_R": "leg_R",
    "Shin_R": "leg_R",
}


class Hull:
    """A convex polygon (counter-clockwise, 2D) and its outward offset curves.

    The offset curve starts at the beginning of edge 0 (outward normal angle ``nu[0]``) and runs counter-clockwise.
    ``arc_of_angle`` and ``point_at`` give arc coordinate <-> position at any offset ``h``.
    """

    def __init__(self, pts: F64) -> None:
        self.pts = np.asarray(pts, dtype=np.float64)
        m = len(self.pts)
        e = np.roll(self.pts, -1, axis=0) - self.pts
        self.edge_len = np.linalg.norm(e, axis=1)
        self.tangent = e / np.maximum(self.edge_len, 1e-18)[:, None]
        nu = np.arctan2(-e[:, 0], e[:, 1])  # outward normal of a counter-clockwise polygon is (dy, -dx)
        for i in range(1, m):
            while nu[i] < nu[i - 1]:
                nu[i] += 2.0 * np.pi
        self.nu = nu
        self.dn = nu - nu[0]
        self.cum = np.concatenate([[0.0], np.cumsum(self.edge_len)])
        self.perimeter = float(self.cum[-1])
        # area centroid of the polygon: stable from row to row (the vertex mean jumps when a hull vertex appears)
        x, y = self.pts[:, 0], self.pts[:, 1]
        x1, y1 = np.roll(x, -1), np.roll(y, -1)
        cr = x * y1 - x1 * y
        a2 = float(cr.sum())
        self.centroid = (
            np.array([((x + x1) * cr).sum(), ((y + y1) * cr).sum()]) / (3.0 * a2)
            if abs(a2) > 1e-18
            else self.pts.mean(axis=0)
        )

    def length(self, h: float) -> float:
        """Length of the outward offset curve by ``h``: ``perimeter + 2 pi h``."""
        return self.perimeter + 2.0 * np.pi * h

    def arc_of_angle(self, phi: float, h: float) -> float:
        """Arc coordinate of the point of the offset curve whose outward normal has angle ``phi`` (radians)."""
        d = (phi - self.nu[0]) % (2.0 * np.pi)
        k = int(np.searchsorted(self.dn, d, side="left"))
        return float(self.cum[k] + h * d)

    def arc_of_ray(self, center: F64, psi: float, h: float) -> float:
        """Arc coordinate where the ray from ``center`` (a point inside the hull) at angle ``psi`` crosses the
        offset curve.

        Anchoring a piece on a ray from the section centre keeps a vertical line of the piece (the centre
        front) straight
        on the body; the support-point anchor of ``arc_of_angle`` follows the nearest hull vertex and wobbles.
        """
        d = np.array([np.cos(psi), np.sin(psi)])
        m = len(self.pts)
        best_t, best_arc = -1.0, 0.0
        nvec = np.column_stack([np.cos(self.nu), np.sin(self.nu)])
        a_pts = self.pts + h * nvec  # offset edge i runs from a_pts[i] to a_pts[i] + l_i * tangent[i]
        w = a_pts - center
        e = self.tangent * self.edge_len[:, None]
        den = d[0] * e[:, 1] - d[1] * e[:, 0]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (w[:, 0] * e[:, 1] - w[:, 1] * e[:, 0]) / den
            u = (w[:, 0] * d[1] - w[:, 1] * d[0]) / den
        ok = (np.abs(den) > 1e-14) & (t > 0) & (u >= -1e-12) & (u <= 1 + 1e-12)
        for i in np.flatnonzero(ok):
            if t[i] > best_t:
                best_t, best_arc = float(t[i]), float(self.cum[i] + h * self.dn[i] + u[i] * self.edge_len[i])
        if h > 1e-12:  # corner arcs: circles of radius h around the hull vertices
            for k in range(m):
                prev = m - 1 if k == 0 else k - 1
                f = center - self.pts[k]
                b = float(d @ f)
                c = float(f @ f) - h * h
                disc = b * b - c
                if disc < 0:
                    continue
                for tt in (-b + np.sqrt(disc),):
                    if tt <= 0 or tt <= best_t:
                        continue
                    q = f + tt * d
                    th = float(np.arctan2(q[1], q[0]))
                    lo = self.nu[prev]
                    rel = (th - lo) % (2.0 * np.pi)
                    span = (
                        (self.nu[k] - lo) % (2.0 * np.pi)
                        if k != 0
                        else (self.nu[0] + 2.0 * np.pi - self.nu[m - 1]) % (2.0 * np.pi)
                    )
                    if rel <= span + 1e-9:
                        base = (self.cum[m] if k == 0 else self.cum[k]) + h * self.dn[prev]
                        best_t, best_arc = float(tt), float(base + h * rel)
        return best_arc

    def _segments(self, h: float) -> tuple[F64, I64, I64]:
        m = len(self.pts)
        starts: list[float] = []
        kind: list[int] = []
        ref: list[int] = []
        for i in range(m):
            starts.append(self.cum[i] + h * self.dn[i])
            kind.append(0)
            ref.append(i)
            k = i + 1 if i + 1 < m else 0
            starts.append(self.cum[i + 1] + h * self.dn[i])  # the corner arc after edge i, at vertex k
            kind.append(1)
            ref.append(k)
        return np.array(starts), np.array(kind, dtype=np.int64), np.array(ref, dtype=np.int64)

    def point_at(self, arcs: F64, h: float) -> tuple[F64, F64]:
        """Positions (K,2) and outward normal angles (K,) of the offset curve at arc coordinates ``arcs``."""
        m = len(self.pts)
        a = np.mod(np.asarray(arcs, dtype=np.float64), self.length(h))
        starts, kind, ref = self._segments(h)
        idx = np.clip(np.searchsorted(starts, a, side="right") - 1, 0, len(starts) - 1)
        s = a - starts[idx]
        pos = np.zeros((len(a), 2))
        ang = np.zeros(len(a))
        ed = kind[idx] == 0
        i = ref[idx[ed]]
        n = np.column_stack([np.cos(self.nu[i]), np.sin(self.nu[i])])
        pos[ed] = self.pts[i] + h * n + s[ed, None] * self.tangent[i]
        ang[ed] = self.nu[i]
        ve = ~ed
        k = ref[idx[ve]]
        prev = np.where(k == 0, m - 1, k - 1)
        theta = self.nu[prev] + s[ve] / max(h, 1e-12)
        pos[ve] = self.pts[k] + h * np.column_stack([np.cos(theta), np.sin(theta)])
        ang[ve] = theta
        return pos, ang


@dataclass
class Region:
    """A body region with its axis frame ``(u, v, axis)`` (right-handed; ``u x v = axis``) and its vertices."""

    name: str
    origin: F64
    axis: F64
    u: F64
    v: F64
    vertex_ids: I64
    _xy: F64 = field(repr=False)
    _axial: F64 = field(repr=False)
    _tri_xy: F64 = field(repr=False)
    _tri_axial: F64 = field(repr=False)
    p_range: tuple[float, float] = (0.0, 0.0)
    symmetric: bool = (
        False  # the support function also covers the mirror image (x -> -x), so shells are left-right symmetric
    )
    _hulls: dict[tuple[int, int], tuple[Hull, bool]] = field(default_factory=dict, repr=False)
    _loops: dict[int, float] = field(default_factory=dict, repr=False)
    _support: dict[tuple[int, int, int], tuple[int, F64]] = field(default_factory=dict, repr=False)

    def coords(self, world: F64) -> tuple[F64, F64]:
        """(axial position (K,), plane coordinates (K,2)) of world points."""
        r = np.atleast_2d(world) - self.origin
        return r @ self.axis, np.column_stack([r @ self.u, r @ self.v])

    def to_world(self, p: float | F64, xy: F64) -> F64:
        """World points for plane coordinates ``xy`` (K,2) at axial position ``p``."""
        return np.asarray(self.origin + np.asarray(p)[..., None] * self.axis + xy @ np.vstack([self.u, self.v]))

    def angle_of(self, direction: F64) -> float:
        """Angle phi of a world direction projected into the plane of the sections (0 = +u, 90 degrees = +v)."""
        return float(np.arctan2(direction @ self.v, direction @ self.u))

    def vector_of(self, phi: F64) -> F64:
        """World unit vectors (K,3) for plane angles ``phi`` (K,)."""
        return np.asarray(np.cos(phi)[:, None] * self.u + np.sin(phi)[:, None] * self.v)

    def hull_at(self, p: float, window: float = 0.03) -> tuple[Hull, bool]:
        """The convex hull of the region at axial position ``p`` (cached per millimetre) and whether ``p`` is inside the
        region. Outside, the nearest section is used."""
        lo, hi = self.p_range
        ok = lo - 1e-9 <= p <= hi + 1e-9
        pc = float(np.clip(p, lo, hi))
        key = (int(round(pc * 1000)), int(round(window * 1000)))
        if key not in self._hulls:
            w = window
            for _ in range(5):
                sel = np.abs(self._axial - key[0] / 1000.0) <= w
                if sel.sum() >= 12:
                    pts = self._xy[sel]
                    try:
                        h = ConvexHull(pts)
                        self._hulls[key] = (Hull(pts[h.vertices]), True)
                        break
                    except Exception:  # noqa: BLE001 - Qhull: degenerate (collinear) input, widen the window
                        pass
                w *= 1.5
            else:
                raise RingNotFoundError(f"no cross-section of {self.name} at p = {p:.3f} m")
        hull, _ = self._hulls[key]
        return hull, ok

    def support_rows(self, keys: I64, window: float, sigma: float = 0.01, n_dir: int = 90) -> F64:
        """Smooth support functions (rows, ``n_dir``) of the cross-sections at the rows ``keys`` (millimetres
        along the axis).

        ``hull_at`` gives a different polygon on every row (a vertex of the sparse mesh enters or leaves the
        window), so the
        perimeter jumps by millimetres between rows and a piece mapped row by row shears. Here the *support function* of
        the region vertices is taken (largest ``x . n`` in ``n_dir`` equally spaced directions, the first one
        is ``+u``),
        enveloped over ``window + 1.5 sigma`` along the axis and blurred with a Gaussian of ``sigma`` (truncated at 1.5
        sigma). A blurred envelope of support functions is the support function of a convex body, so the cross-sections
        stay convex and their offset curves exact; every vertex within ``window`` of a row is still inside its
        cross-section, because each blurred value is an average of envelopes that all cover that window.
        """
        wk, sk = int(round(window * 1000)), int(round(sigma * 1000))
        if (wk, sk, n_dir) not in self._support:
            reach = wk + int(np.ceil(1.5 * sk))
            k_lo = int(np.floor(self.p_range[0] * 1000)) - reach - 2
            k_hi = int(np.ceil(self.p_range[1] * 1000)) + reach + 2
            ang = np.arange(n_dir) * (2.0 * np.pi / n_dir)
            dirs = np.column_stack([np.cos(ang), np.sin(ang)])
            tab = np.full((k_hi - k_lo + 1, n_dir), -np.inf)
            bins = np.rint(self._axial * 1000).astype(np.int64) - k_lo
            np.maximum.at(tab, bins, self._xy @ dirs.T)
            if self.symmetric:  # mirror across the sagittal plane: the plane coordinate ``v`` is the wearer's x
                np.maximum.at(tab, bins, (self._xy * np.array([1.0, -1.0])) @ dirs.T)
            env = maximum_filter1d(tab, size=2 * reach + 1, axis=0, mode="nearest")
            good = np.flatnonzero(np.isfinite(env).all(axis=1))  # rows beyond the vertices take the nearest covered row
            env = env[good[np.abs(np.arange(len(env))[:, None] - good[None, :]).argmin(axis=1)]]
            if sk > 0:
                env = gaussian_filter1d(env, sk, axis=0, truncate=1.5, mode="nearest")
            self._support[(wk, sk, n_dir)] = (k_lo, env)
        k_lo, env = self._support[(wk, sk, n_dir)]
        return np.asarray(env[np.clip(np.asarray(keys) - k_lo, 0, len(env) - 1)])

    def skin_loop(self, p: float) -> float:
        """Length of the actual skin section of the region at ``p`` (metres): a radial first-hit loop around the hull
        centroid (one ray per degree, gaps interpolated). This is the body loop used for the true ease."""
        key = int(round(float(np.clip(p, *self.p_range)) * 1000))
        if key in self._loops:
            return self._loops[key]
        pc = key / 1000.0
        hull, _ = self.hull_at(pc, 0.03)
        s = self._tri_axial - pc
        cross = s * np.roll(s, -1, axis=1) < 0
        two = cross.sum(axis=1) == 2
        if not two.any():
            self._loops[key] = hull.perimeter
            return hull.perimeter
        txy, ts, cc = self._tri_xy[two], s[two], cross[two]
        order = np.argsort(~cc, axis=1, kind="stable")[:, :2]
        rows = np.arange(len(txy))[:, None]
        a_xy, b_xy = txy[rows, order], txy[rows, (order + 1) % 3]
        sa, sb = ts[rows, order], ts[rows, (order + 1) % 3]
        pts = a_xy + (sa / (sa - sb))[..., None] * (b_xy - a_xy)
        p0, p1 = pts[:, 0], pts[:, 1]
        c = hull.centroid
        th = np.radians(np.arange(360.0))
        d = np.column_stack([np.cos(th), np.sin(th)])
        e = p1 - p0
        w = p0 - c
        den = d[:, 0:1] * e[None, :, 1] - d[:, 1:2] * e[None, :, 0]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (w[None, :, 0] * e[None, :, 1] - w[None, :, 1] * e[None, :, 0]) / den
            u = (w[None, :, 0] * d[:, 1:2] - w[None, :, 1] * d[:, 0:1]) / den
        ok = (np.abs(den) > 1e-12) & (t > 0.005) & (u >= 0) & (u <= 1)
        r = np.where(ok, t, np.inf).min(axis=1)
        good = np.isfinite(r)
        if good.sum() < 8:
            self._loops[key] = hull.perimeter
            return hull.perimeter
        idx = np.arange(360)
        r = np.interp(idx, np.r_[idx[good] - 360, idx[good], idx[good] + 360], np.r_[r[good], r[good], r[good]])
        r = gaussian_filter1d(r, 1.0, mode="wrap")
        poly = c + r[:, None] * d
        length = float(np.linalg.norm(np.roll(poly, -1, axis=0) - poly, axis=1).sum())
        self._loops[key] = length
        return length


class Shells:
    """The regions of one avatar."""

    def __init__(self, regions: dict[str, Region]) -> None:
        self.regions = regions

    @staticmethod
    def ring_of_bv(bv: str) -> str:
        """Default ring of an arrangement point's bounding volume (``RingNotFoundError`` for head, hands, feet)."""
        try:
            return _RINGS[bv]
        except KeyError:
            raise RingNotFoundError(
                f"bounding volume {bv!r} has no body shell (rings: {sorted(set(_RINGS.values()))})"
            ) from None

    @classmethod
    def from_avatar(cls, avatar: Avatar) -> Shells:
        """Build the six regions from the avatar mesh and its bounding volumes."""
        vols = avatar.volumes
        need = [
            "Hip",
            "Waist",
            "Chest",
            "Neck",
            "Head",
            "UpperArm_L",
            "UpperArm_R",
            "Forearm_L",
            "Forearm_R",
            "Thigh_L",
            "Thigh_R",
            "Shin_L",
            "Shin_R",
        ]
        missing = [n for n in need if n not in vols]
        if missing:
            raise RingNotFoundError(f"avatar has no bounding volumes {missing}; body shells need them")
        v, f = avatar.vertices, avatar.faces
        names = list(vols)
        score = np.empty((len(v), len(names)))
        for j, n in enumerate(names):
            vol = vols[n]
            d = vol.end - vol.start
            t = ((v - vol.start) @ d) / (d @ d)
            k = np.clip(t, 0.0, 1.0)
            # a volume only claims vertices next to its own axis segment (a fat chest must not claim the neck)
            score[:, j] = np.linalg.norm(v - (vol.start + k[:, None] * d), axis=1) - vol.radius_m
            if n in ("Hip", "Waist", "Chest", "Neck"):
                score[(t < -0.1) | (t > 1.1), j] += 10.0
        owner = np.array(names)[np.argmin(score, axis=1)]
        crotch = float(vols["Hip"].end[1])  # above the crotch the hips and buttocks are torso, not thigh
        owner[np.isin(owner, ["Thigh_L", "Thigh_R"]) & (v[:, 1] > crotch)] = "Hip"

        def ids(*bvs: str) -> I64:
            return np.flatnonzero(np.isin(owner, bvs)).astype(np.int64)

        regions: dict[str, Region] = {}
        torso = set(ids("Hip", "Waist", "Chest").tolist())
        cap: dict[str, set[int]] = {}
        for side in ("L", "R"):  # the shoulder cap (first fraction of the upper arm) belongs to the torso
            ua = vols[f"UpperArm_{side}"]
            d = ua.end - ua.start
            cap[side] = {int(i) for i in ids(f"UpperArm_{side}") if ((v[i] - ua.start) @ d) / (d @ d) < SHOULDER_CAP}
            torso |= cap[side]
        spec: dict[str, tuple[I64, F64, F64]] = {
            "torso": (np.array(sorted(torso), dtype=np.int64), np.zeros(3), np.array([0.0, 1.0, 0.0])),
            "neck": (ids("Neck"), vols["Neck"].start, vols["Neck"].end - vols["Neck"].start),
        }
        for s in ("L", "R"):
            arm = sorted(set(ids(f"UpperArm_{s}", f"Forearm_{s}").tolist()) - cap[s])
            spec[f"arm_{s}"] = (
                np.array(arm, dtype=np.int64),
                vols[f"UpperArm_{s}"].start,
                vols[f"Forearm_{s}"].end - vols[f"UpperArm_{s}"].start,
            )
            spec[f"leg_{s}"] = (
                ids(f"Thigh_{s}", f"Shin_{s}"),
                vols[f"Thigh_{s}"].start,
                vols[f"Shin_{s}"].end - vols[f"Thigh_{s}"].start,
            )
        z = np.array([0.0, 0.0, 1.0])
        for name, (vid, origin, axis_vec) in spec.items():
            a = axis_vec / np.linalg.norm(axis_vec)
            u = z - (z @ a) * a
            u /= np.linalg.norm(u)
            vv = np.cross(a, u)
            r = v[vid] - origin
            axial, xy = r @ a, np.column_stack([r @ u, r @ vv])
            inreg = np.zeros(len(v), dtype=bool)
            inreg[vid] = True
            tri = f[inreg[f].all(axis=1)]
            tr = v[tri] - origin
            tri_axial = tr @ a
            tri_xy = np.stack([tr @ u, tr @ vv], axis=-1)
            regions[name] = Region(
                name,
                np.asarray(origin, dtype=np.float64),
                a,
                u,
                vv,
                vid,
                xy,
                axial,
                tri_xy,
                tri_axial,
                (float(axial.min()), float(axial.max())),
                symmetric=name in ("torso", "neck"),
            )
        return cls(regions)

    def region_for_bv(self, bv: str) -> Region:
        """The region a bounding volume belongs to."""
        return self.regions[self.ring_of_bv(bv)]

"""Body-slice wrap: map a flat piece onto the torso using the avatar horizontal cross-sections. Metres.

The engine only sees contract files, so the torso is found geometrically. The torso axis is the polyline through the
Hip, Waist, Chest and Neck volume axes. Triangles inside an arm volume are left out (except the shoulder cap, the first
12 % of the upper arm), so the T-pose arms never enter a section. For every height from the crotch to mid neck the
remaining mesh is sliced horizontally and the section is stored as a radial profile ``r(theta)`` around the axis
(theta 0 = front, 90 = +x = the wearer's left, 1 degree steps; the first hit from the axis wins; gaps where an arm was
cut out are filled by interpolating in the angle; the profiles are smoothed).

A piece vertex at row ``h`` and horizontal offset ``s`` from the reference goes to world height ``y_anchor + h`` and to
the point of that height's loop whose arc length from the anchor angle is ``s`` (increasing theta = the point tangent).
The loop used is the skin loop pushed out by the offset along its horizontal normals, so arc length along every row
and vertical distance in world height are exact. The surface distance along steep parts (shoulder slope, bust) is
longer than the world height, and on a sloped surface the perpendicular distance to the skin is a little smaller than
the (horizontal) offset.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d

from .loaders import Avatar

F64 = NDArray[np.float64]
I64 = NDArray[np.int64]
N_THETA = 360
STEP_M = 0.01
TORSO_BVS = ("Hip", "Waist", "Chest", "Neck")
SMOOTH_THETA = 3.0  # Gaussian sigma of the radial profile along theta, in degrees
SHOULDER_CAP = 0.12  # fraction of the upper arm (from the shoulder joint) that still counts as torso


@dataclass(frozen=True)
class SliceInfo:
    """Outcome of one mapping: whether offsets were clamped (and by how much) and rows outside the torso range."""

    overrun: bool
    max_overrun_m: float
    out_of_range: int


def _seg_dist(p: F64, a: F64, b: F64) -> F64:
    d = b - a
    k = np.clip(((p - a) @ d) / (d @ d), 0.0, 1.0)
    return np.asarray(np.linalg.norm(p - (a + k[:, None] * d), axis=1), dtype=np.float64)


class TorsoSections:
    """Cross-sections of the torso at 1 cm spacing, interpolated linearly in height."""

    def __init__(self, ys: F64, centres: F64, radii: F64) -> None:
        self.ys, self.centres, self.radii = ys, centres, radii
        self.theta = np.radians(np.arange(N_THETA, dtype=np.float64))

    @property
    def y_range(self) -> tuple[float, float]:
        """Lowest and highest section height (metres)."""
        return float(self.ys[0]), float(self.ys[-1])

    @classmethod
    def build(cls, avatar: Avatar) -> TorsoSections:
        """Slice the avatar. Raises KeyError-free: missing torso volumes give a ValueError."""
        vols = avatar.volumes
        missing = [n for n in TORSO_BVS if n not in vols]
        if missing:
            raise ValueError(f"avatar has no bounding volumes {missing}; body_slice needs the torso volumes")
        pts = np.array(
            [
                vols["Hip"].end,
                vols["Hip"].start,
                vols["Waist"].start,
                vols["Waist"].end,
                vols["Chest"].end,
                vols["Neck"].end,
            ]
        )
        pts = pts[np.argsort(pts[:, 1])]
        y_lo = float(vols["Hip"].end[1]) + 0.01
        y_hi = float(vols["Neck"].start[1] + 0.5 * (vols["Neck"].end[1] - vols["Neck"].start[1]))
        ys = np.arange(y_lo, y_hi + 1e-9, STEP_M)
        cx = np.interp(ys, pts[:, 1], pts[:, 0])
        cz = np.interp(ys, pts[:, 1], pts[:, 2])

        v, f = avatar.vertices, avatar.faces
        arm = np.zeros(len(v), dtype=bool)
        for name, vol in vols.items():
            if name.startswith(("UpperArm", "Forearm", "Hand")):
                inside = _seg_dist(v, vol.start, vol.end) < vol.radius_m * 1.05
                if name.startswith("UpperArm"):  # keep the shoulder cap: the first 12 % of the upper arm
                    d = vol.end - vol.start
                    inside &= ((v - vol.start) @ d) / (d @ d) > SHOULDER_CAP
                arm |= inside
        tri = v[f[~arm[f].any(axis=1)]]  # triangles with no vertex in an arm
        tri = tri[(tri[:, :, 1].max(axis=1) >= y_lo - 0.02) & (tri[:, :, 1].min(axis=1) <= y_hi + 0.02)]

        theta = np.radians(np.arange(N_THETA, dtype=np.float64))
        d = np.column_stack([np.sin(theta), np.cos(theta)])  # (x, z) direction of each ray
        radii = np.full((len(ys), N_THETA), np.nan)
        for k, y in enumerate(ys):
            s = tri[:, :, 1] - y
            a_, b_ = s, np.roll(s, -1, axis=1)
            cross = a_ * b_ < 0
            two = cross.sum(axis=1) == 2
            if not two.any():
                continue
            tt, cc = tri[two], cross[two]
            order = np.argsort(~cc, axis=1, kind="stable")[:, :2]
            rows = np.arange(len(tt))[:, None]
            va = tt[rows, order]  # start vertex of the two crossing edges
            vb = tt[rows, (order + 1) % 3]
            sa = va[:, :, 1] - y
            sb = vb[:, :, 1] - y
            w = sa / (sa - sb)
            p = va + w[:, :, None] * (vb - va)
            p0, p1 = p[:, 0][:, [0, 2]], p[:, 1][:, [0, 2]]  # segment end points in (x, z)
            c = np.array([cx[k], cz[k]])
            e = p1 - p0
            wv = p0 - c
            den = d[:, 0:1] * e[None, :, 1] - d[:, 1:2] * e[None, :, 0]  # (J, S) cross(d, e)
            with np.errstate(divide="ignore", invalid="ignore"):
                t = (wv[None, :, 0] * e[None, :, 1] - wv[None, :, 1] * e[None, :, 0]) / den
                u = (wv[None, :, 0] * d[:, 1:2] - wv[None, :, 1] * d[:, 0:1]) / den
            ok = (np.abs(den) > 1e-12) & (t > 0.005) & (u >= 0) & (u <= 1)
            best = np.where(ok, t, np.inf).min(axis=1)  # the first skin crossing seen from the axis
            radii[k] = np.where(np.isfinite(best), best, np.nan)
        for k in range(len(ys)):  # fill angular gaps (where an arm was cut out) by periodic linear interpolation
            r = radii[k]
            good = np.isfinite(r)
            if good.sum() < 8:
                continue
            idx = np.arange(N_THETA)
            radii[k] = np.interp(
                idx, np.r_[idx[good] - N_THETA, idx[good], idx[good] + N_THETA], np.r_[r[good], r[good], r[good]]
            )
        valid = np.isfinite(radii).all(axis=1)
        if valid.sum() < 5:
            raise ValueError("could not slice the torso: too few valid cross-sections")
        ys, radii = ys[valid], radii[valid]
        # smooth the profiles (3 degrees along the loop, 1 cm between sections) so that interpolated gaps do not leave
        # concave kinks, which a 3 cm offset of the loop would fold over
        radii = gaussian_filter1d(
            gaussian_filter1d(radii, SMOOTH_THETA, axis=1, mode="wrap"), 1.0, axis=0, mode="nearest"
        )
        return cls(ys, np.column_stack([cx[valid], cz[valid]]), radii)

    # -- loops ---------------------------------------------------------------------------------------------------
    def profile(self, y: float) -> tuple[F64, F64]:
        """(centre (x, z), radii (360,)) at height ``y`` (clamped to the sampled range)."""
        k = float(np.clip((y - self.ys[0]) / STEP_M, 0.0, len(self.ys) - 1.0))
        i = int(min(np.floor(k), len(self.ys) - 2))
        f = k - i
        return (1 - f) * self.centres[i] + f * self.centres[i + 1], (1 - f) * self.radii[i] + f * self.radii[i + 1]

    def loop(self, y: float) -> tuple[F64, F64, float]:
        """(points (360,2) as (x, z), cumulative arc length from theta = 0 (360,), perimeter)."""
        c, r = self.profile(y)
        pts = c + r[:, None] * np.column_stack([np.sin(self.theta), np.cos(self.theta)])
        seg = np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1)
        cum = np.r_[0.0, np.cumsum(seg)[:-1]]
        return pts, cum, float(seg.sum())

    def perimeter(self, y: float) -> float:
        """Length of the loop at height ``y`` (metres)."""
        return self.loop(y)[2]


def _offset_loop(sections: TorsoSections, y: float, offset: float) -> tuple[F64, F64, F64, float]:
    """Skin loop at ``y`` pushed out by ``offset`` along its horizontal normals.

    Returns (offset points (360,2) as (x, z), unit outward normals (360,2), cumulative arc length of the *offset* loop
    from theta = 0, its perimeter). The fabric length of a row is measured on this loop (the garment sits on it).
    """
    pts, _, _ = sections.loop(y)
    c, _ = sections.profile(y)
    tang = np.roll(pts, -1, axis=0) - np.roll(pts, 1, axis=0)
    n = np.column_stack([tang[:, 1], -tang[:, 0]])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    flip = np.einsum("ij,ij->i", n, pts - c) < 0
    n[flip] *= -1.0
    off = pts + offset * n
    seg = np.linalg.norm(np.roll(off, -1, axis=0) - off, axis=1)
    return off, n, np.r_[0.0, np.cumsum(seg)[:-1]], float(seg.sum())


def map_rows(
    sections: TorsoSections,
    y_world: F64,
    s: F64,
    anchor_deg: float,
    offset: float,
    thickness: F64,
    flip: bool,
) -> tuple[F64, SliceInfo]:
    """World points for rows at heights ``y_world`` and horizontal offsets ``s`` (metres from the anchor angle).

    A row is placed on the skin loop pushed out by ``offset`` (plus the piece thickness along the normal) and ``s`` is
    the arc length along that pushed-out loop, so arc length along a row equals the piece x distance exactly.
    ``flip`` runs the offsets the other way round the body. Offsets beyond half the loop are clamped (``overrun``).
    """
    lo, hi = sections.y_range
    yq = np.round(np.clip(y_world, lo, hi) / 0.001).astype(np.int64)
    out = np.zeros((len(y_world), 3))
    out[:, 1] = y_world
    overrun = False
    worst = 0.0
    th = sections.theta
    for key in np.unique(yq):
        sel = np.flatnonzero(yq == key)
        y = float(key) * 0.001
        off_pts, nrm, cum, per = _offset_loop(sections, y, offset)
        ext = np.r_[cum - per, cum, cum + per]
        ang = np.r_[th - 2 * np.pi, th, th + 2 * np.pi]
        a0 = float(np.interp(np.radians(anchor_deg), ang, ext))
        d = (-1.0 if flip else 1.0) * s[sel]
        lim = 0.5 * per
        if np.abs(d).max() > lim:
            overrun = True
            worst = max(worst, float(np.abs(d).max() - lim))
            d = np.clip(d, -lim, lim)
        a = a0 + d

        def at(arr: F64, a: F64 = a, ext: F64 = ext) -> F64:
            return np.asarray(np.interp(a, ext, np.r_[arr, arr, arr]), dtype=np.float64)

        out[sel, 0] = at(off_pts[:, 0])
        out[sel, 2] = at(off_pts[:, 1])
        if np.any(thickness[sel] != 0.0):
            nx, nz = at(nrm[:, 0]), at(nrm[:, 1])
            nn = np.maximum(np.hypot(nx, nz), 1e-12)
            out[sel, 0] += thickness[sel] * nx / nn
            out[sel, 2] += thickness[sel] * nz / nn
    out_of_range = int(np.sum((y_world < lo - 1e-9) | (y_world > hi + 1e-9)))
    return out, SliceInfo(overrun, worst, out_of_range)

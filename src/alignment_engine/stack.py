"""Many convex cross-sections at once: vectorised offset curves (position and ray anchors) for whole meshes. Metres.

``HullStack`` is a padded array version of ``shells.Hull`` for a list of rows (one hull per row, 1 mm apart), so that
the
arc coordinate and the position of thousands of mesh points on their row's offset curve are computed in a few numpy
calls.
``RingSurface`` adds the region frame and the mapping of unrolled piece coordinates (X around, P along the axis) to
world.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .shells import Hull, Region

F64 = NDArray[np.float64]
I64 = NDArray[np.int64]
TWO_PI = 2.0 * np.pi
ROW_KEY = (
    1000.0  # row r is searched at keys ``r * ROW_KEY + arc`` of one flat sorted array (every arc length is below this)
)


@dataclass(frozen=True)
class Offsets:
    """Edge start arcs of all rows for one standoff per row, as a flat sorted array (one ``searchsorted`` per query)."""

    h: F64
    length: F64
    flat: F64


class HullStack:
    """``rows`` hulls padded to the same vertex count. Padding edges have zero length and start at the end of the
    loop."""

    def __init__(self, hulls: list[Hull]) -> None:
        r, m = len(hulls), max(len(h.pts) for h in hulls)
        n = np.array([len(h.pts) for h in hulls], dtype=np.int64)
        pts, tan = np.zeros((r, m, 2)), np.zeros((r, m, 2))
        el, nu, cum = np.zeros((r, m)), np.zeros((r, m)), np.zeros((r, m + 1))
        for i, h in enumerate(hulls):
            k = len(h.pts)
            pts[i, :k], pts[i, k:] = h.pts, h.pts[0]
            tan[i, :k], tan[i, k:] = h.tangent, h.tangent[0]
            el[i, :k] = h.edge_len
            nu[i, :k], nu[i, k:] = h.nu, h.nu[0] + TWO_PI
            cum[i, : k + 1], cum[i, k + 1 :] = h.cum, h.cum[k]
        self._setup(n, pts, tan, el, nu, cum, np.array([h.centroid for h in hulls]))

    @classmethod
    def from_support(cls, env: F64) -> HullStack:
        """Rows of convex polygons given by support values ``env`` (R,K) in K equally spaced directions from ``+u``.

        The polygon of row r is the intersection of the half planes ``x . n_k <= env[r, k]``; edge k lies on line k, so
        the outward normal angle of edge k is ``2 pi k / K`` and zero-length edges (directions that only touch a vertex)
        are harmless.
        """
        r, k = env.shape
        ang = np.arange(k) * (TWO_PI / k)
        n = np.column_stack([np.cos(ang), np.sin(ang)])
        t = np.column_stack([-np.sin(ang), np.cos(ang)])
        n1 = np.roll(n, -1, axis=0)
        det = n[:, 0] * n1[:, 1] - n[:, 1] * n1[:, 0]
        s1 = np.roll(env, -1, axis=1)
        v = np.stack(
            [(env * n1[:, 1] - s1 * n[:, 1]) / det, (n[:, 0] * s1 - n1[:, 0] * env) / det], axis=-1
        )  # v_k = lines k, k+1
        pts = np.roll(v, 1, axis=1)  # pts[k] = v_{k-1}: edge k runs from pts[k] to pts[k+1] on line k
        nxt = np.roll(pts, -1, axis=1)
        el = np.maximum(np.einsum("rkc,kc->rk", nxt - pts, t), 0.0)
        cum = np.concatenate([np.zeros((r, 1)), np.cumsum(el, axis=1)], axis=1)
        area2 = pts[..., 0] * nxt[..., 1] - nxt[..., 0] * pts[..., 1]
        a = area2.sum(axis=1)
        cen = np.stack(
            [((pts[..., 0] + nxt[..., 0]) * area2).sum(axis=1), ((pts[..., 1] + nxt[..., 1]) * area2).sum(axis=1)],
            axis=1,
        )
        obj = cls.__new__(cls)
        obj._setup(
            np.full(r, k, dtype=np.int64),
            pts,
            np.broadcast_to(t, (r, k, 2)).copy(),
            el,
            np.broadcast_to(ang, (r, k)).copy(),
            cum,
            cen / (3.0 * a)[:, None],
        )
        return obj

    def _setup(self, n: I64, pts: F64, tan: F64, el: F64, nu: F64, cum: F64, center: F64) -> None:
        r, m = nu.shape
        self.n, self.pts, self.tan, self.el, self.nu, self.cum = n, pts, tan, el, nu, cum
        self.dn = self.nu - self.nu[:, :1]
        self.perim = self.cum[:, -1].copy()
        self.center = center
        self.nvec = np.stack([np.cos(self.nu), np.sin(self.nu)], axis=-1)
        idx = np.arange(m)[None, :]
        prev = np.where(idx == 0, self.n[:, None] - 1, idx - 1)
        rows = np.arange(r)[:, None]
        self.lo = self.nu[rows, prev]  # normal angle at the start of the corner arc at vertex k
        self.span = np.mod(self.nu - self.lo, TWO_PI)
        self.dn_prev = self.dn[rows, prev]
        self.cum_c = np.where(idx == 0, self.cum[rows, self.n[:, None]], self.cum[:, :m])
        self.valid = idx < self.n[:, None]
        # one gather per query: [pts(2), tangent(2), normal(2), nu, cum at the end of the edge, dn]
        self.pack = np.concatenate([pts, tan, self.nvec, nu[..., None], cum[:, 1:, None], self.dn[..., None]], axis=-1)

    def support(self, dirs: F64) -> F64:
        """Support function of every row hull in the directions ``dirs`` (K,2): the largest ``pts . dir`` (R,K)."""
        dots = np.einsum("rmc,kc->rmk", self.pts, dirs)
        dots = np.where(self.valid[..., None], dots, -np.inf)
        return np.asarray(dots.max(axis=1), dtype=np.float64)

    def length(self, rows: I64, h: F64) -> F64:
        """Length of the offset curve of ``rows`` by ``h``."""
        return np.asarray(self.perim[rows] + TWO_PI * h, dtype=np.float64)

    def offsets(self, h_rows: F64) -> Offsets:
        """Search table for the standoff ``h_rows`` (one value per row)."""
        edge_start = self.cum[:, : self.pts.shape[1]] + h_rows[:, None] * self.dn
        flat = (edge_start + np.arange(len(h_rows))[:, None] * ROW_KEY).ravel()
        return Offsets(h_rows, self.perim + TWO_PI * h_rows, flat)

    def point_at_table(self, off: Offsets, rows: I64, arcs: F64) -> F64:
        """Same as ``point_at`` for the standoff of ``off`` (faster for many points: no per-point row copies)."""
        m = self.pts.shape[1]
        h = off.h[rows]
        a = np.mod(arcs, off.length[rows])
        idx = np.searchsorted(off.flat, rows * ROW_KEY + a, side="right") - 1
        ie = np.clip(idx - rows * m, 0, m - 1)
        start = off.flat[rows * m + ie] - rows * ROW_KEY
        g = self.pack[rows, ie]
        c = g[:, 7] + h * g[:, 8]
        edge = g[:, 0:2] + h[:, None] * g[:, 4:6] + (a - start)[:, None] * g[:, 2:4]
        th = g[:, 6] + (a - c) / np.maximum(h, 1e-12)
        corner = self.pack[rows, (ie + 1) % self.n[rows], 0:2] + h[:, None] * np.stack(
            [np.cos(th), np.sin(th)], axis=-1
        )
        return np.asarray(np.where((a < c)[:, None], edge, corner), dtype=np.float64)

    def point_at(self, rows: I64, arcs: F64, h: F64) -> F64:
        """Positions (N,2) at arc coordinates ``arcs`` of the offset curves of ``rows`` (``h`` per point)."""
        n = len(rows)
        m = self.pts.shape[1]
        a = np.mod(arcs, self.perim[rows] + TWO_PI * h)
        cum, dn = self.cum[rows], self.dn[rows]
        e = cum[:, :m] + h[:, None] * dn
        ie = np.clip((e <= a[:, None]).sum(axis=1) - 1, 0, m - 1)
        ar = np.arange(n)
        c = cum[ar, ie + 1] + h * dn[ar, ie]
        nu = self.nu[rows, ie]
        on_edge = a < c
        nvec = np.stack([np.cos(nu), np.sin(nu)], axis=-1)
        edge = self.pts[rows, ie] + h[:, None] * nvec + (a - e[ar, ie])[:, None] * self.tan[rows, ie]
        k = (ie + 1) % self.n[rows]
        th = nu + (a - c) / np.maximum(h, 1e-12)
        corner = self.pts[rows, k] + h[:, None] * np.stack([np.cos(th), np.sin(th)], axis=-1)
        return np.asarray(np.where(on_edge[:, None], edge, corner), dtype=np.float64)

    def arc_of_ray(self, rows: I64, psi: F64, h: F64) -> F64:
        """Arc coordinate where the ray from each row's hull centre at angle ``psi`` crosses the offset curve."""
        d = np.stack([np.cos(psi), np.sin(psi)], axis=-1)
        d0, d1 = d[:, 0:1], d[:, 1:2]
        hh = h[:, None]
        cen = self.center[rows][:, None, :]
        pts = self.pts[rows]
        w = pts + hh[..., None] * self.nvec[rows] - cen
        e = self.tan[rows] * self.el[rows][..., None]
        den = d0 * e[..., 1] - d1 * e[..., 0]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (w[..., 0] * e[..., 1] - w[..., 1] * e[..., 0]) / den
            u = (w[..., 0] * d1 - w[..., 1] * d0) / den
        ok = (np.abs(den) > 1e-14) & (t > 0) & (u >= -1e-12) & (u <= 1 + 1e-12)
        with np.errstate(invalid="ignore"):
            arc_e = self.cum[rows][:, :-1] + hh * self.dn[rows] + u * self.el[rows]
        t_e = np.where(ok, t, -1.0)
        f = cen - pts
        b = f[..., 0] * d0 + f[..., 1] * d1
        disc = b * b - ((f * f).sum(axis=-1) - hh * hh)
        tt = -b + np.sqrt(np.maximum(disc, 0.0))
        q = f + tt[..., None] * d[:, None, :]
        rel = np.mod(np.arctan2(q[..., 1], q[..., 0]) - self.lo[rows], TWO_PI)
        ok_c = (disc >= 0) & (tt > 0) & (rel <= self.span[rows] + 1e-9) & self.valid[rows] & (hh > 1e-12)
        arc_c = self.cum_c[rows] + hh * self.dn_prev[rows] + hh * rel
        t_c = np.where(ok_c, tt, -1.0)
        t_all = np.concatenate([t_e, t_c], axis=1)
        arc_all = np.concatenate([arc_e, arc_c], axis=1)
        best = np.argmax(t_all, axis=1)
        out = arc_all[np.arange(len(rows)), best]
        return np.asarray(np.where(t_all[np.arange(len(rows)), best] > 0, out, 0.0), dtype=np.float64)


class RingSurface:
    """The offset surfaces of one region over a range of axial positions: rows every millimetre.

    ``h_rows`` arguments are standoff arrays with one value per row (a constant array for a uniform standoff).
    """

    def __init__(self, region: Region, p_lo: float, p_hi: float, window: float) -> None:
        self.region = region
        self.k0 = int(np.floor(p_lo * 1000.0)) - 1
        k1 = int(np.ceil(p_hi * 1000.0)) + 1
        self.rows = k1 - self.k0 + 1
        self._offsets: dict[bytes, Offsets] = {}
        self.stack = HullStack.from_support(region.support_rows(np.arange(self.k0, k1 + 1), window))
        self.p_of_row = (self.k0 + np.arange(self.rows)) / 1000.0

    def row_of(self, p: F64) -> I64:
        """Nearest row of axial positions ``p``."""
        return np.asarray(np.clip(np.rint(np.asarray(p) * 1000.0).astype(np.int64) - self.k0, 0, self.rows - 1))

    def perimeter(self, p: F64) -> F64:
        """Perimeter of the hull at ``p`` (no standoff)."""
        return np.asarray(self.stack.perim[self.row_of(p)])

    def ray_table(self, psi: float, h_rows: F64) -> F64:
        """Arc coordinate of the ray at angle ``psi`` on every row (anchor of a piece)."""
        sub = np.unique(np.r_[np.flatnonzero((self.k0 + np.arange(self.rows)) % 4 == 0), 0, self.rows - 1])
        s = self.stack.arc_of_ray(sub, np.full(len(sub), psi), h_rows[sub])
        return np.asarray(np.interp(np.arange(self.rows), sub, s))

    def length_rows(self, h_rows: F64) -> F64:
        """Length of the offset curve on every row."""
        return self.stack.length(np.arange(self.rows), h_rows)

    def world(self, p: F64, arcs_of_row: F64, x: F64, h_rows: F64, interpolate: bool = True) -> F64:
        """World points for axial positions ``p`` (N,), arc offsets ``x`` (N,) from the ray anchor of each row.

        ``arcs_of_row`` is ``ray_table(psi, h_rows)``. With ``interpolate`` the point is the linear blend of its two
        neighbouring rows (smooth under 1 mm), otherwise the nearest row is used (faster).
        """
        p = np.asarray(p, dtype=np.float64)
        kf = np.clip(p * 1000.0 - self.k0, 0.0, self.rows - 1.0)
        key = h_rows.tobytes()
        if key not in self._offsets:
            if len(self._offsets) > 12:
                self._offsets.clear()
            self._offsets[key] = self.stack.offsets(np.asarray(h_rows, dtype=np.float64))
        off = self._offsets[key]
        if not interpolate:
            r = np.rint(kf).astype(np.int64)
            xy = self.stack.point_at_table(off, r, arcs_of_row[r] + x)
            return self.region.to_world(p, xy)
        i0 = np.minimum(np.floor(kf).astype(np.int64), self.rows - 2)
        f = (kf - i0)[:, None]
        xy0 = self.stack.point_at_table(off, i0, arcs_of_row[i0] + x)
        xy1 = self.stack.point_at_table(off, i0 + 1, arcs_of_row[i0 + 1] + x)
        return self.region.to_world(p, (1.0 - f) * xy0 + f * xy1)

    def frame_at(self, p: F64, arcs_of_row: F64, x: F64, h_rows: F64) -> tuple[F64, F64]:
        """World unit vectors (e_arc (N,3), axis (3,)) at surface points: used to move a piece along its own surface."""
        r = self.row_of(np.atleast_1d(p))
        _, ang = _angle(self.stack, r, arcs_of_row[r] + np.atleast_1d(x), h_rows[r])
        rad = self.region.vector_of(ang)
        return np.cross(self.region.axis, rad), self.region.axis


def _angle(stack: HullStack, rows: I64, arcs: F64, h: F64) -> tuple[F64, F64]:
    """Outward normal angle (radians) of the offset curve at ``arcs`` (companion of ``HullStack.point_at``)."""
    m = stack.pts.shape[1]
    a = np.mod(arcs, stack.perim[rows] + TWO_PI * h)
    cum, dn = stack.cum[rows], stack.dn[rows]
    e = cum[:, :m] + h[:, None] * dn
    ie = np.clip((e <= a[:, None]).sum(axis=1) - 1, 0, m - 1)
    ar = np.arange(len(rows))
    c = cum[ar, ie + 1] + h * dn[ar, ie]
    nu = stack.nu[rows, ie]
    return a, np.where(a < c, nu, nu + (a - c) / np.maximum(h, 1e-12))

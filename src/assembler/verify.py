"""Independent verifier of an assembly run (section 4 of docs/assembly_loop.md). Computed from the world meshes.

Every check is a row ``{id, status, value, limit, piece, hint, run}``; ``status`` is ``pass``, ``fail``, ``check``
(reported fit data worth a look, never failing) or ``info``. Distances are measured here, with this module's own
nearest-triangle search; only the body level heights are taken from ``alignment_engine.levels`` (the same landmark
definitions the registration uses, so a level is defined once).
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

F64 = NDArray[np.float64]
I64 = NDArray[np.int64]
SIZE_TOL = 0.001  # 0.1 %
DIST_TOL = 0.0005
PAIR_TOL = 0.001
REG_TOL = 0.015
LEVEL_CHECK = 0.08
ANGLE_TOL = 5.0
REGION_REACH = 0.12
CONCENTRIC_SLACK = 0.005
LEVEL_LINES = (
    (r"waist ?line", "Waist"),
    (r"hip ?line", "Hip"),
    (r"(bust|chest) ?line", "Chest"),
    (r"crotch ?line", "Crotch"),
    (r"knee ?line", "Knee"),
    (r"heel ?line", "Ankle"),
)


@dataclass
class PieceView:
    """What the verifier needs of a placed piece: the world mesh and the same mesh flat (vertex for vertex)."""

    name: str
    point: str
    world: F64
    flat: F64
    faces: I64
    bundle: dict[str, Any]
    placement: dict[str, Any]
    doc: dict[str, Any]  # its entry of the engine result


@dataclass
class RunView:
    """One engine run to verify."""

    name: str
    pieces: list[PieceView]
    result: dict[str, Any]
    options: dict[str, Any]
    rows: list[dict[str, Any]] = field(default_factory=list)


def _row(
    rows: list[dict[str, Any]],
    cid: str,
    ok: bool | None,
    value: Any,
    limit: Any,
    piece: str = "",
    hint: str = "",
    status: str | None = None,
) -> None:
    st = status or ("pass" if ok else "fail")
    v = round(float(value), 6) if isinstance(value, (float, np.floating)) else value
    lim = round(float(limit), 6) if isinstance(limit, (float, np.floating)) else limit
    rows.append({"id": cid, "status": st, "value": v, "limit": lim, "piece": piece, "hint": hint})


# -- geometry helpers -------------------------------------------------------------------------------------------------
def _closest_on_tris(p: F64, a: F64, b: F64, c: F64) -> F64:
    """Closest points on triangles (a, b, c) to points p (all (M,3)): plane projection when inside, else the nearest
    of the three edges."""
    n = np.cross(b - a, c - a)
    nn = np.einsum("ij,ij->i", n, n)
    nn = np.where(nn < 1e-30, 1e-30, nn)
    q = p - (np.einsum("ij,ij->i", p - a, n) / nn)[:, None] * n
    inside = np.ones(len(p), dtype=bool)
    for u, v in ((a, b), (b, c), (c, a)):
        inside &= np.einsum("ij,ij->i", np.cross(v - u, q - u), n) >= 0
    best = q.copy()
    bd = np.where(inside, np.linalg.norm(p - q, axis=1), np.inf)
    for u, v in ((a, b), (b, c), (c, a)):
        d = v - u
        t = np.clip(np.einsum("ij,ij->i", p - u, d) / np.maximum(np.einsum("ij,ij->i", d, d), 1e-30), 0.0, 1.0)
        e = u + t[:, None] * d
        de = np.linalg.norm(p - e, axis=1)
        better = de < bd
        best[better], bd[better] = e[better], de[better]
    return np.asarray(best, dtype=np.float64)


def split_long(tri: F64, max_edge: float = 0.02) -> F64:
    """Triangles (T,3,3) split 4-way until no edge is longer than ``max_edge``. The set of surface points is unchanged
    (T-junctions do not matter to distance queries), so distances stay exact while the candidate search stays local
    (the synthetic body has 40 cm triangles)."""
    out = []
    while len(tri):
        e = np.stack([np.linalg.norm(tri[:, k] - tri[:, (k + 1) % 3], axis=1) for k in range(3)], axis=1).max(axis=1)
        out.append(tri[e <= max_edge])
        t = tri[e > max_edge]
        a, b, c = t[:, 0], t[:, 1], t[:, 2]
        ab, bc, ca = 0.5 * (a + b), 0.5 * (b + c), 0.5 * (c + a)
        tri = np.concatenate([np.stack(x, axis=1) for x in ((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca))])
    return np.concatenate(out) if out else np.zeros((0, 3, 3))


class Surf:
    """Nearest-triangle queries on a mesh: exact point-triangle distances, candidates from bounding spheres of locally
    split triangles."""

    def __init__(self, v: F64, f: I64) -> None:
        self.v, self.f = v, f
        self.wtri = v[f]
        self.tri = split_long(self.wtri)
        self.cent = self.tri.mean(axis=1)
        self.rad = np.linalg.norm(self.tri - self.cent[:, None, :], axis=2).max(axis=1)
        self.rmax = float(self.rad.max())
        self.tree = cKDTree(self.cent)
        corners = self.tri.reshape(-1, 3)
        self.vtree = cKDTree(corners)
        self.corner_pts = corners
        self.vface = np.repeat(np.arange(len(self.tri)), 3)
        self.edge = 2.0 * self.rmax
        n = np.cross(self.tri[:, 1] - self.tri[:, 0], self.tri[:, 2] - self.tri[:, 0])
        self.n = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-30)
        w = self.wtri
        vol = float(np.einsum("ij,ij->i", w[:, 0], np.cross(w[:, 1], w[:, 2])).sum())
        self.orient = -1.0 if vol < 0 else 1.0

    def closest(self, p: F64, near: float = 0.045, chunk: int = 20000) -> tuple[F64, F64, I64]:
        """(distance, closest point, triangle) per point. Exact for points within ``near`` of the surface; farther
        points get their nearest vertex (only near points decide a clearance or a gap)."""
        d0, i0 = self.vtree.query(p)  # nearest vertex: an upper bound, within one edge of the exact distance
        d_out, q_out = d0.copy(), self.corner_pts[i0]
        t_out = self.vface[i0]
        sel = np.flatnonzero(d0 < near)  # beyond: the exact distance is at least d0 - rmax (> any clearance tested)
        if len(sel):
            d1, q1, t1 = self._exact(p[sel], chunk)
            d_out[sel], q_out[sel], t_out[sel] = d1, q1, t1
        return d_out, q_out, t_out

    def _exact(self, p: F64, chunk: int) -> tuple[F64, F64, I64]:
        """Exact distances: every triangle whose bounding sphere (centroid, radius) comes within the nearest-vertex
        distance (an upper bound) is a candidate, so the true closest triangle is always tested, however long the
        triangles are."""
        d_out, q_out, t_out = np.empty(len(p)), np.empty((len(p), 3)), np.empty(len(p), dtype=np.int64)
        upper, _ = self.vtree.query(p)
        for s in range(0, len(p), chunk):
            pp, up = p[s : s + chunk], upper[s : s + chunk]
            k = min(32, len(self.cent))
            dk, ik = self.tree.query(pp, k=k)
            dk, ik = dk.reshape(len(pp), -1), ik.reshape(len(pp), -1)
            full = dk[:, -1] <= up + self.rmax  # 32 candidates do not cover the bound: all of them, exactly
            row = np.repeat(np.arange(len(pp)), k)[~np.repeat(full, k)]
            idx = ik[~full].ravel()
            if full.any():
                lists = self.tree.query_ball_point(pp[full], up[full] + self.rmax + 1e-9)
                cnt = np.array([len(x) for x in lists])
                row = np.concatenate([row, np.repeat(np.flatnonzero(full), cnt)])
                idx = np.concatenate([idx, *[np.asarray(x, dtype=np.int64) for x in lists]])
            ok = np.linalg.norm(pp[row] - self.cent[idx], axis=1) <= up[row] + self.rad[idx] + 1e-9
            row, idx = row[ok], idx[ok]
            t = self.tri[idx]
            q = _closest_on_tris(pp[row], t[:, 0], t[:, 1], t[:, 2])
            dist = np.linalg.norm(pp[row] - q, axis=1)
            order = np.lexsort((dist, row))
            first = order[np.r_[True, row[order][1:] != row[order][:-1]]]
            d_out[s + row[first]] = dist[first]
            q_out[s + row[first]] = q[first]
            t_out[s + row[first]] = idx[first]
        return d_out, q_out, t_out

    def signed(self, p: F64) -> tuple[F64, F64]:
        """(signed distance: negative inside the body, closest point): the side of the closest triangle's outward
        normal. Outward is fixed once for the whole mesh by its signed volume (the synthetic body is wound inward)."""
        d, q, t = self.closest(p)
        side = np.einsum("ij,ij->i", p - q, self.n[t]) * self.orient
        return np.where(side < -1e-9, -d, d), q


def edges_of(f: I64) -> I64:
    e = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
    return np.unique(e, axis=0)


def samples(v: F64, f: I64) -> F64:
    """Vertices and edge midpoints."""
    e = edges_of(f)
    return np.vstack([v, 0.5 * (v[e[:, 0]] + v[e[:, 1]])])


def boundary_edges(f: I64) -> I64:
    """Directed edges (E,2) used by one triangle only."""
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    _, inv, cnt = np.unique(np.sort(e, axis=1), axis=0, return_inverse=True, return_counts=True)
    return np.asarray(e[cnt[inv.ravel()] == 1], dtype=np.int64)


def outline_position(outline: F64, pts: F64) -> F64:
    """Arc position along the closed outline (from vertex 0) of points lying on it."""
    a, b = outline, np.roll(outline, -1, axis=0)
    d = b - a
    ln = np.linalg.norm(d, axis=1)
    cum = np.r_[0.0, np.cumsum(ln)[:-1]]
    t = np.clip(np.einsum("pij,ij->pi", pts[:, None, :] - a[None], d) / np.maximum(ln**2, 1e-30), 0.0, 1.0)
    q = a[None] + t[:, :, None] * d[None]
    k = np.argmin(np.linalg.norm(pts[:, None, :] - q, axis=2), axis=1)
    r = np.arange(len(pts))
    return np.asarray(cum[k] + t[r, k] * ln[k], dtype=np.float64)


def jacobians(world: F64, flat: F64, f: I64) -> tuple[F64, F64]:
    """Per triangle: J (T,3,2) with world edge = J @ flat edge, and the flat area (T,)."""
    ef = np.stack([flat[f[:, 1]] - flat[f[:, 0]], flat[f[:, 2]] - flat[f[:, 0]]], axis=2)  # (T,2,2)
    ew = np.stack([world[f[:, 1]] - world[f[:, 0]], world[f[:, 2]] - world[f[:, 0]]], axis=2)  # (T,3,2)
    det = ef[:, 0, 0] * ef[:, 1, 1] - ef[:, 0, 1] * ef[:, 1, 0]
    ok = np.abs(det) > 1e-14
    inv = np.zeros_like(ef)
    inv[ok, 0, 0], inv[ok, 1, 1] = ef[ok, 1, 1] / det[ok], ef[ok, 0, 0] / det[ok]
    inv[ok, 0, 1], inv[ok, 1, 0] = -ef[ok, 0, 1] / det[ok], -ef[ok, 1, 0] / det[ok]
    return np.einsum("tij,tjk->tik", ew, inv), 0.5 * np.abs(det)


def flat_to_world(p: PieceView, xy: F64) -> F64:
    """World points of flat points (barycentric in the flat mesh, applied to the world mesh)."""
    t = p.flat[p.faces]
    out = np.empty((len(xy), 3))
    for i, q in enumerate(np.asarray(xy, dtype=np.float64)):
        a, b, c = t[:, 0], t[:, 1], t[:, 2]
        den = (b[:, 1] - c[:, 1]) * (a[:, 0] - c[:, 0]) + (c[:, 0] - b[:, 0]) * (a[:, 1] - c[:, 1])
        den = np.where(np.abs(den) < 1e-30, 1e-30, den)
        l1 = ((b[:, 1] - c[:, 1]) * (q[0] - c[:, 0]) + (c[:, 0] - b[:, 0]) * (q[1] - c[:, 1])) / den
        l2 = ((c[:, 1] - a[:, 1]) * (q[0] - c[:, 0]) + (a[:, 0] - c[:, 0]) * (q[1] - c[:, 1])) / den
        l3 = 1 - l1 - l2
        k = int(np.argmax(np.minimum(np.minimum(l1, l2), l3)))
        w = p.world[p.faces[k]]
        out[i] = l1[k] * w[0] + l2[k] * w[1] + l3[k] * w[2]
    return out


def mesh_hash(v: F64) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.round(v, 9)).tobytes()).hexdigest()[:16]


# -- checks ------------------------------------------------------------------------------------------------------------
def check_size(run: RunView) -> None:
    rows = run.rows
    if run.options.get("allow_distortion"):
        _row(rows, "size.allow_distortion", False, True, False, hint="allow_distortion must stay false")
    for p in run.pieces:
        if p.doc.get("changes_piece_size"):
            _row(rows, "size.changes_piece_size", False, True, False, p.name)
        j, area_f = jacobians(p.world, p.flat, p.faces)
        tf = p.flat[p.faces]
        longest = np.max([np.linalg.norm(tf[:, k] - tf[:, (k + 1) % 3], axis=1) for k in range(3)], axis=0)
        good = 2.0 * area_f / np.maximum(longest, 1e-12) > 0.0005  # slivers (a self-crossing outline) amplify rounding
        sv = np.linalg.svd(j[good], compute_uv=False)
        worst = float(np.abs(sv - 1.0).max()) if len(sv) else 0.0
        _row(
            rows,
            "size.singular_values",
            worst <= SIZE_TOL,
            worst,
            SIZE_TOL,
            p.name,
            "" if worst <= SIZE_TOL else "a triangle is stretched: the placement is not an isometry",
        )
        tw = p.world[p.faces]
        area_w = 0.5 * np.linalg.norm(np.cross(tw[:, 1] - tw[:, 0], tw[:, 2] - tw[:, 0]), axis=1).sum()
        da = abs(area_w / area_f.sum() - 1.0)
        _row(rows, "size.area", da <= SIZE_TOL, da, SIZE_TOL, p.name)
        be = boundary_edges(p.faces)
        lw = np.linalg.norm(p.world[be[:, 1]] - p.world[be[:, 0]], axis=1)
        lf = np.linalg.norm(p.flat[be[:, 1]] - p.flat[be[:, 0]], axis=1)
        dp = abs(lw.sum() / lf.sum() - 1.0)
        _row(rows, "size.perimeter", dp <= SIZE_TOL, dp, SIZE_TOL, p.name)
        # every outline edge between named outline points: the boundary edges lying on it (by arc position along the
        # pattern outline; a slit or pinch point is no problem), world against flat length
        outline = np.asarray(p.bundle["outline_m"], dtype=np.float64)
        s_mid = outline_position(outline, 0.5 * (p.flat[be[:, 0]] + p.flat[be[:, 1]]))
        per = float(np.linalg.norm(np.roll(outline, -1, axis=0) - outline, axis=1).sum())
        on = {q["name"]: q["outline_s_m"] for q in p.bundle["named_points"] if q.get("on_outline")}
        worst_e = 0.0
        for e in p.bundle.get("edges", []):
            if e["from"] not in on or e["to"] not in on or on[e["from"]] is None or on[e["to"]] is None:
                continue
            a, b = float(on[e["from"]]), float(on[e["to"]])
            sel = ((s_mid - a) % per) < ((b - a) % per)
            if lf[sel].sum() < 1e-6:
                continue
            worst_e = max(worst_e, abs(lw[sel].sum() / lf[sel].sum() - 1.0))
        _row(rows, "size.outline_edges", worst_e <= SIZE_TOL, worst_e, SIZE_TOL, p.name)
        m = np.asarray(p.doc.get("matrix", np.eye(4).ravel()), dtype=np.float64).reshape(4, 4)[:3, :3]
        sv_m = np.linalg.svd(m, compute_uv=False)
        dm = float(np.abs(sv_m - 1.0).max())
        _row(rows, "size.matrix_scale", dm <= 1e-6, dm, 1e-6, p.name)


def check_avatar(rows: list[dict[str, Any]], avatar_v: F64, expected_hash: str, height_cm: float | None) -> None:
    h = mesh_hash(avatar_v)
    _row(
        rows,
        "size.avatar_mesh",
        h == expected_hash,
        h,
        expected_hash,
        hint="" if h == expected_hash else "the avatar used is not the freshly built one",
    )
    if height_cm is not None:
        dh = abs(float(avatar_v[:, 1].max()) * 100.0 - height_cm)
        _row(rows, "size.avatar_height", dh <= 0.5, dh, 0.5, hint="cm from the body JSON height")
    _row(rows, "size.avatar_floor", abs(float(avatar_v[:, 1].min())) < 1e-6, float(avatar_v[:, 1].min()), 0.0)


def check_clearance(run: RunView, body: Surf, points: dict[str, Any], expect: dict[str, Any]) -> dict[str, F64]:
    rows, o = run.rows, run.options
    clear, gap = float(o.get("clearance_body", 0.03)), float(o.get("gap_piece", 0.03))
    gl = o.get("gap_layer")
    gap_need = min(gap, float(gl)) if gl is not None else gap
    far_ok = set(expect.get("far_ok", []))
    near: dict[str, F64] = {}
    for p in run.pieces:
        s = samples(p.world, p.faces)
        d, q = body.signed(s)
        near[p.name] = q[: len(p.world)]
        dmin = float(d.min())
        _row(
            rows,
            "clearance.body",
            dmin >= clear - DIST_TOL,
            dmin,
            clear,
            p.name,
            "" if dmin >= clear - DIST_TOL else "closer to the avatar than clearance_body (negative: inside)",
        )
        floor = float(p.world[:, 1].min())
        _row(
            rows,
            "clearance.floor",
            floor >= clear - DIST_TOL,
            floor,
            clear,
            p.name,
            "" if floor >= clear - DIST_TOL else "below the floor clearance",
        )
        maxs = float(o.get("max_standoff", 0.30))
        _row(rows, "clearance.standoff", dmin <= maxs + DIST_TOL, dmin, maxs, p.name)
        pt = points.get(p.point)
        host = next((h for h in run.pieces if h.name == (p.placement.get("over") or {}).get("piece")), None)
        if host is not None:  # laid over another piece: its region is that piece
            dh = float(cKDTree(host.world).query(p.world)[0].min())
            _row(
                rows,
                "clearance.region",
                dh <= REGION_REACH,
                dh,
                REGION_REACH,
                p.name,
                "" if dh <= REGION_REACH else f"{dh * 100:.0f} cm from {host.name}, the piece it lies over",
            )
        elif pt is not None:
            surf = np.asarray(pt["position_m"]) - float(pt["offset_m"]) * np.asarray(pt["normal"])
            dist = float(np.linalg.norm(p.world - surf, axis=1).min())
            lim = max(dmin, 0.0) + REGION_REACH
            ok = dist <= lim + DIST_TOL or p.name in far_ok
            _row(
                rows,
                "clearance.region",
                ok,
                dist,
                lim,
                p.name,
                "" if ok else f"{dist * 100:.0f} cm from its point {p.point}: not in its target region",
            )
        # self intersection: world vertices that meet although far apart on the pattern
        tw = cKDTree(p.world)
        pairs = tw.query_pairs(0.0015, output_type="ndarray")
        bad = 0
        if len(pairs):
            bad = int((np.linalg.norm(p.flat[pairs[:, 0]] - p.flat[pairs[:, 1]], axis=1) > 0.01).sum())
        _row(rows, "clearance.self", bad == 0, bad, 0, p.name)
    surfs = {p.name: Surf(p.world, p.faces) for p in run.pieces}
    boxes = {p.name: (p.world.min(axis=0), p.world.max(axis=0)) for p in run.pieces}
    for i, a in enumerate(run.pieces):
        for b in run.pieces[i + 1 :]:
            lo = np.maximum(boxes[a.name][0], boxes[b.name][0])
            hi = np.minimum(boxes[a.name][1], boxes[b.name][1])
            if np.any(lo - hi > gap + 0.01):
                continue
            d1 = float(surfs[b.name].closest(samples(a.world, a.faces))[0].min())
            d2 = float(surfs[a.name].closest(samples(b.world, b.faces))[0].min())
            dd = d1 if d1 < d2 else d2
            _row(rows, "clearance.pieces", bool(dd >= gap_need - DIST_TOL), dd, gap_need, f"{a.name} | {b.name}")
    return near


def _mean_dir(p: PieceView, d2: F64) -> F64:
    j, a = jacobians(p.world, p.flat, p.faces)
    v = np.einsum("tij,j->ti", j, d2)
    m = (v * a[:, None]).sum(axis=0)
    return np.asarray(m / max(float(np.linalg.norm(m)), 1e-18), dtype=np.float64)


def _angle(u: F64, v: F64, line: bool = False) -> float:
    c = float(u @ v) / max(float(np.linalg.norm(u) * np.linalg.norm(v)), 1e-18)
    if line:
        c = abs(c)
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def axis_up(pt: dict[str, Any], volumes: dict[str, Any]) -> F64:
    """The up a piece on this point must have: its body volume's axis, upright on the torso (never the point's own up,
    which is tilted by the surface normal and was upside down on the Hip points), toward the root on a limb."""
    up = np.asarray(pt["up"], dtype=np.float64)
    bv = str(pt.get("bv", ""))
    v = volumes.get(bv)
    if v is None:
        return up
    a = np.asarray(v["axis_end"], dtype=np.float64) - np.asarray(v["axis_start"], dtype=np.float64)
    side = bv[-2:] if bv[-2:] in ("_L", "_R") else ""
    if not side:  # torso, neck, head: upright, whatever way the volume's axis runs or the point's up says
        a /= max(float(np.linalg.norm(a)), 1e-18)
        return a if a[1] >= 0 else -a
    for first, last in (("UpperArm", "Forearm"), ("Thigh", "Shin")):  # a limb: the whole limb, root to end
        if side and bv[:-2] in (first, last) and f"{first}{side}" in volumes and f"{last}{side}" in volumes:
            a = np.asarray(volumes[f"{last}{side}"]["axis_end"]) - np.asarray(volumes[f"{first}{side}"]["axis_start"])
    a /= max(float(np.linalg.norm(a)), 1e-18)
    return a if a @ up >= 0 else -a


def check_orientation(
    run: RunView, points: dict[str, Any], near: dict[str, F64], expect: dict[str, Any], volumes: dict[str, Any]
) -> None:
    rows = run.rows
    grain_exp = expect.get("grain", {})
    for p in run.pieces:
        pt = points.get(p.point)
        if pt is None:
            continue
        up = axis_up(pt, volumes)
        pl = p.placement
        tilt = pl.get("tilt") or {}
        allow = ANGLE_TOL + float(np.hypot(tilt.get("pitch_deg", 0.0), tilt.get("roll_deg", 0.0)))
        explicit = bool(pl.get("rotation_deg")) or not isinstance(pl.get("orient", "auto"), str)
        top = _mean_dir(p, np.array([0.0, 1.0]))
        a = _angle(top, up)
        if explicit:
            _row(rows, "orientation.top_up", None, a, allow, p.name, "rotation asked for by the placement", "info")
        else:
            _row(
                rows,
                "orientation.top_up",
                a <= allow,
                a,
                allow,
                p.name,
                "" if a <= allow else "the pattern's top is not up (toward the limb root on a limb)",
            )
        g = p.bundle.get("grainline")
        mode = grain_exp.get(p.name, "vertical")
        if g is not None and mode != "any" and not explicit:
            th = np.radians(float(g["rotation_deg"]))
            gw = _mean_dir(p, np.array([np.cos(th), np.sin(th)]))
            ag = _angle(gw, up, line=True)
            _row(
                rows,
                "orientation.grain",
                ag <= allow,
                ag,
                allow,
                p.name,
                "" if ag <= allow else "grainline not parallel to the ring / limb axis",
            )
        elif g is None and not explicit:
            bb = p.bundle["bbox_m"]
            if (bb["max"][1] - bb["min"][1]) >= (bb["max"][0] - bb["min"][0]):  # tall piece: long axis is the grain
                _row(
                    rows,
                    "orientation.grain",
                    a <= allow,
                    a,
                    allow,
                    p.name,
                    "" if a <= allow else "the long axis of a tall piece is not along the axis",
                )
        # face: the flat +z (pattern front) must point away from the avatar
        j, area = jacobians(p.world, p.flat, p.faces)
        nw = np.cross(j[:, :, 0], j[:, :, 1])
        nw /= np.maximum(np.linalg.norm(nw, axis=1, keepdims=True), 1e-18)
        c = p.world[p.faces].mean(axis=1)
        q = near[p.name][p.faces].mean(axis=1)
        out = c - q
        out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-18)
        frac = float((area * (np.einsum("ij,ij->i", nw, out) > 0)).sum() / area.sum())
        flipped = bool(pl.get("flip_facing")) != bool(pl.get("mirror"))
        val = 1.0 - frac if flipped else frac
        _row(
            rows,
            "orientation.face_out",
            val >= 0.5,
            val,
            0.5,
            p.name,
            "" if val >= 0.5 else "the pattern face points into the body",
        )


def check_registration(run: RunView, levels: dict[str, float], expect: dict[str, Any]) -> list[dict[str, Any]]:
    rows = run.rows
    by = {p.name: p for p in run.pieces}
    for r in expect.get("registration", []):
        p = by.get(r["piece"])
        if p is None:
            _row(rows, "registration", False, None, REG_TOL, r["piece"], "piece not placed")
            continue
        if "piece_point" in r:
            q = next(q for q in p.bundle["named_points"] if q["name"] == r["piece_point"])
            y = float(flat_to_world(p, np.array([q["xy_m"]]))[0, 1])
        else:  # the same edge point as the engine's (one defined point; a band on a tilted ring is not level)
            from alignment_engine.orient import edge_point

            xy = edge_point(np.asarray(p.bundle["outline_m"], dtype=np.float64), r.get("edge", "top"))
            y = float(flat_to_world(p, xy[None, :])[0, 1])
        off = y - levels[r["level"]] - float(r.get("offset_m", 0.0))
        _row(
            rows,
            "registration",
            abs(off) <= REG_TOL,
            off,
            REG_TOL,
            p.name,
            f"{r.get('piece_point', r.get('edge', 'top'))} at {y:.3f} m, {r['level']} at {levels[r['level']]:.3f} m",
        )
    fit = []
    for p in run.pieces:
        for q in p.bundle["named_points"]:
            for pat, lv in LEVEL_LINES:
                if lv in levels and re.match(rf"^{pat}$", q["name"].replace("_", ""), re.IGNORECASE):
                    y = float(flat_to_world(p, np.array([q["xy_m"]]))[0, 1])
                    off = y - levels[lv]
                    st = "check" if abs(off) > LEVEL_CHECK else "info"
                    _row(
                        rows,
                        "levels.offset",
                        None,
                        off,
                        LEVEL_CHECK,
                        p.name,
                        f"{q['name']} at {y:.3f} m vs body {lv} {levels[lv]:.3f} m",
                        st,
                    )
                    fit.append(
                        {
                            "piece": p.name,
                            "line": q["name"],
                            "level": lv,
                            "y_m": round(y, 4),
                            "body_y_m": round(levels[lv], 4),
                            "offset_m": round(off, 4),
                        }
                    )
    return fit


def _boundary_dense(p: PieceView, step: float = 0.001) -> F64:
    be = boundary_edges(p.faces)
    out = []
    for a, b in zip(p.world[be[:, 0]], p.world[be[:, 1]], strict=True):
        n = max(int(np.ceil(np.linalg.norm(b - a) / step)), 1)
        out.append(a + np.linspace(0.0, 1.0, n, endpoint=False)[:, None] * (b - a))
    return np.vstack(out)


def check_pairs(run: RunView, points: dict[str, Any], expect: dict[str, Any]) -> None:
    rows = run.rows
    by = {p.name: p for p in run.pieces}
    done: set[tuple[str, str]] = set()
    for p in run.pieces:
        other = p.doc.get("arrange", {}).get("pair_with")
        if not other or other not in by:
            continue
        key = tuple(sorted((p.name, other)))
        if key in done:
            continue
        done.add((key[0], key[1]))
        a, b = by[key[0]], by[key[1]]
        sa, sb = a.doc.get("solved") or {}, b.doc.get("solved") or {}
        if sa and sb:
            dev = max(
                abs(sa["radius_m"] - sb["radius_m"]),
                abs(sa["shift_m"][1] - sb["shift_m"][1]),
                abs(sa["shift_m"][0] + sb["shift_m"][0]),
            )
            _row(rows, "pairs.parameters", dev <= PAIR_TOL, dev, PAIR_TOL, f"{a.name} | {b.name}")
        ma = _boundary_dense(a) * np.array([-1.0, 1.0, 1.0])
        db = _boundary_dense(b)
        cd = max(float(cKDTree(db).query(ma)[0].max()), float(cKDTree(ma).query(db)[0].max()))
        _row(
            rows,
            "pairs.mirror_image",
            cd <= PAIR_TOL + 0.0005,
            cd,
            PAIR_TOL + 0.0005,
            f"{a.name} | {b.name}",
            "" if cd <= PAIR_TOL + 0.0005 else "the halves are not mirror images about x = 0",
        )
        xa, xb = float(a.world[:, 0].mean()), float(b.world[:, 0].mean())
        pa = points.get(a.point)
        if pa is not None and abs(pa["position_m"][0]) > 0.002:  # a side pair: each on its own point's side
            ok = np.sign(xa) == np.sign(pa["position_m"][0])
        else:  # a centre-line pair: on opposite sides, apart by at least the gap
            gap = float(run.options.get("gap_piece", 0.03))
            ok = np.sign(xa) != np.sign(xb) and min(np.abs(a.world[:, 0]).min(), np.abs(b.world[:, 0]).min()) >= (
                0.5 * gap - DIST_TOL
            )
        _row(
            rows,
            "pairs.sides",
            bool(ok),
            round(xa, 4),
            "own side",
            f"{a.name} | {b.name}",
            "" if ok else "a half crossed to the other side (swapped or stacked)",
        )
    flags = [w for w in run.result.get("warnings", []) if w.startswith(("NOT_CONCENTRIC", "PAIR_"))]
    flags += [v["kind"] for v in run.result["validity"]["violations"] if v["kind"].startswith("PAIR_")]
    _row(rows, "pairs.engine_flags", not flags, len(flags), 0, hint="; ".join(flags)[:300])
    gl = run.options.get("gap_layer")
    gl = float(gl if gl is not None else run.options.get("gap_piece", 0.03))
    for ring in expect.get("rings", []):
        radii = [by[n].doc["arrange"].get("radius_m") for n in ring["pieces"] if n in by]
        radii = [r for r in radii if r is not None]
        if len(radii) >= 2 and ring.get("concentric", True):
            spread = max(radii) - min(radii)
            _row(rows, "pairs.concentric", spread <= gl + CONCENTRIC_SLACK, spread, gl + CONCENTRIC_SLACK, ring["name"])


def _slice_len(v: F64, f: I64, y: float) -> tuple[float, F64]:
    from .render import slice_segments

    s = slice_segments(v, f, y)
    return float(np.linalg.norm(s[:, 1] - s[:, 0], axis=1).sum()) if len(s) else 0.0, s


def check_semantics(
    run: RunView, levels: dict[str, float], avatar_v: F64, avatar_f: I64, expect: dict[str, Any]
) -> list[dict[str, Any]]:
    rows = run.rows
    names = {p.name for p in run.pieces}
    for n in expect.get("required_pieces", []):
        _row(rows, "semantics.required", n in names, n in names, True, n)
    by = {p.name: p for p in run.pieces}
    ease = []
    gap = float(run.options.get("gap_piece", 0.03))
    for ring in expect.get("rings", []):
        for lv in ring.get("levels", []):
            if lv not in levels:
                continue
            y = levels[lv] + float(ring.get("dy_m", 0.0))
            centre = np.asarray(ring.get("centre_xz", [0.0, 0.0]), dtype=float)
            arcs, total = [], 0.0
            for n in ring["pieces"]:
                if n not in by:
                    continue
                ln, segs = _slice_len(by[n].world, by[n].faces, y)
                if not ln:
                    continue
                total += ln
                pts = segs.reshape(-1, 2) - centre
                ang = np.sort(np.arctan2(pts[:, 0], pts[:, 1]))
                gaps = np.diff(np.r_[ang, ang[0] + 2 * np.pi])
                k = int(np.argmax(gaps))
                start, width = float(ang[(k + 1) % len(ang)]), float(2 * np.pi - gaps[k])
                arcs.append((start, width, float(np.linalg.norm(pts, axis=1).mean()), n))
            a_len, a_segs = _slice_len(avatar_v, avatar_f, y)
            if a_len and len(a_segs):
                mids = a_segs.mean(axis=1) - centre
                keep = np.linalg.norm(mids, axis=1) < float(ring.get("body_radius_m", 0.35))
                a_len = float(np.linalg.norm(a_segs[keep, 1] - a_segs[keep, 0], axis=1).sum())
            cov = sum(w for _, w, _, _ in arcs) / (2 * np.pi)
            arcs.sort()
            min_gap = np.inf
            for i, (s0, w0, r0, _) in enumerate(arcs):
                s1, _, r1, _ = arcs[(i + 1) % len(arcs)]
                g = (s1 - (s0 + w0)) % (2 * np.pi)
                min_gap = min(min_gap, g * min(r0, r1))
            if len(arcs) >= 2:
                _row(
                    rows,
                    "semantics.ring_gaps",
                    min_gap >= gap - 0.002,
                    float(min_gap),
                    gap,
                    ring["name"],
                    f"{lv}: {len(arcs)} pieces cover {cov * 100:.0f} % of the ring",
                )
            ease.append(
                {
                    "ring": ring["name"],
                    "level": lv,
                    "y_m": round(y, 4),
                    "pieces_cut": len(arcs),
                    "garment_width_m": round(total, 4),
                    "body_girth_m": round(a_len, 4),
                    "ease_m": round(total - a_len, 4),
                    "coverage": round(cov, 3),
                }
            )
            _row(
                rows,
                "semantics.ease",
                None,
                round(total - a_len, 4),
                None,
                ring["name"],
                f"{lv}: garment {total * 100:.1f} cm vs body {a_len * 100:.1f} cm",
                "info",
            )
    return ease


def verdict(rows: list[dict[str, Any]]) -> bool:
    return not any(r["status"] == "fail" for r in rows)


def check_honesty(run: RunView) -> None:
    engine = bool(run.result["validity"]["valid"])
    mine = verdict(run.rows)
    _row(
        run.rows,
        "engine.honesty",
        engine == mine,
        engine,
        mine,
        hint="" if engine == mine else f"engine says valid={engine}, the independent checks say {mine}",
    )

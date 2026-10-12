"""Arrangement points: CLO3D-style anchors on bounding volumes, resolved to the avatar surface. Metres.

A point is defined by (bounding volume, angle around its axis, t along its axis, offset, wrap direction). Its
position is found by casting a ray from the axis out to the surface and adding ``offset_m`` along the surface normal.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from numpy.typing import NDArray

from .errors import PointConfigError, UnknownArrangementPointError
from .volumes import BoundingVolume

F64 = NDArray[np.float64]
I64 = NDArray[np.int64]
DEFAULT_CONFIG = Path(__file__).with_name("default_points.yaml")


@dataclass(frozen=True)
class PointSpec:
    """One configured point (no coordinates): where it sits relative to a bounding volume."""

    name: str
    bv: str
    angle_deg: float
    t: float
    offset_m: float
    wrap_direction: str


@dataclass(frozen=True)
class ArrangementPoint:
    """A resolved point. ``position_m`` already includes the offset; ``normal``/``up``/``tangent`` are unit vectors."""

    spec: PointSpec
    position_m: F64
    normal: F64
    up: F64
    tangent: F64


def load_specs(path: Path | None = None) -> list[PointSpec]:
    """Expand the YAML configuration into point specs (``_L`` / ``_R`` pairs are generated as mirror pairs)."""
    p = Path(path) if path else DEFAULT_CONFIG
    try:
        cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
        offs = cfg["offset_m"]
        entries = cfg["points"]
    except (OSError, yaml.YAMLError, KeyError, TypeError) as e:
        raise PointConfigError(f"{p}: cannot read point configuration: {e}") from None
    out: list[PointSpec] = []
    for entry in entries:
        try:
            bv, t, wrap, angles = entry["bv"], float(entry["t"]), entry["wrap"], entry["angles"]
            sym = bool(entry.get("symmetric", False))
            paired = bool(entry.get("paired", False))
        except (KeyError, TypeError, ValueError) as err:
            raise PointConfigError(f"{p}: bad point entry {entry!r}: {err}") from None
        if wrap not in ("around", "flat") or not 0.0 <= t <= 1.0:
            raise PointConfigError(f"{p}: entry {bv}: wrap must be around|flat and t within 0..1")
        off = float(entry.get("offset_m", offs["limb" if sym else "central"]))
        for label, ang in angles.items():
            if sym or paired:
                bv_l, bv_r = (f"{bv}_L", f"{bv}_R") if sym else (bv, bv)
                out.append(PointSpec(f"{bv}_{label}_L", bv_l, float(ang) % 360.0, t, off, wrap))
                out.append(PointSpec(f"{bv}_{label}_R", bv_r, (360.0 - float(ang)) % 360.0, t, off, wrap))
            else:
                out.append(PointSpec(f"{bv}_{label}", bv, float(ang) % 360.0, t, off, wrap))
    names = [s.name for s in out]
    if len(set(names)) != len(names):
        raise PointConfigError(f"{p}: duplicate point names")
    return out


def vertex_normals(v: F64, faces: I64) -> F64:
    """Area-weighted, outward-pointing vertex normals (outward decided by the sign of the enclosed volume)."""
    a, b, c = v[faces[:, 0]], v[faces[:, 1]], v[faces[:, 2]]
    fn = np.cross(b - a, c - a)
    vol = float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum())
    if vol < 0:
        fn = -fn
    n = np.zeros_like(v)
    for k in range(3):
        np.add.at(n, faces[:, k], fn)
    return np.asarray(n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12), dtype=np.float64)


def ray_frame(volume: BoundingVolume) -> tuple[F64, F64, F64]:
    """(axis, front, left-ish) unit vectors of a volume; angle theta points along cos(theta)*front + sin(theta)*side."""
    a = volume.axis
    for ref in (np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0])):
        f = ref - (ref @ a) * a
        if np.linalg.norm(f) > 1e-3:
            f /= np.linalg.norm(f)
            side = np.cross(a, f)
            # central volumes: 90 deg is always +x (the wearer's left), even if the axis points down (hips)
            return a, f, (-side if (not volume.limb and a[1] < 0) else side)
    raise PointConfigError(f"bounding volume {volume.name}: degenerate axis")


def _ray_hit(origin: F64, d: F64, tri: F64) -> tuple[float, int, F64] | None:
    """Nearest hit of the ray with triangles ``tri`` (M,3,3): (distance, triangle index, barycentric)."""
    e1, e2 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    p = np.cross(d, e2)
    det = np.einsum("ij,ij->i", e1, p)
    ok = np.abs(det) > 1e-14
    inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    s = origin - tri[:, 0]
    u = np.einsum("ij,ij->i", s, p) * inv
    q = np.cross(s, e1)
    w = (q @ d) * inv
    t = np.einsum("ij,ij->i", e2, q) * inv
    hit = ok & (u >= -1e-9) & (w >= -1e-9) & (u + w <= 1 + 1e-9) & (t > 1e-6)
    if not hit.any():
        return None
    idx = np.flatnonzero(hit)
    i = int(idx[np.argmin(t[idx])])
    return float(t[i]), i, np.array([1.0 - u[i] - w[i], u[i], w[i]])


def _dist_point_triangles(p: F64, tri: F64) -> float:
    """Exact distance from ``p`` to the nearest of the triangles (M,3,3)."""
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    ab, ac, ap = b - a, c - a, p - a
    n = np.cross(ab, ac)
    nn = np.einsum("ij,ij->i", n, n)
    ok = nn > 1e-18
    plane = np.where(ok, np.abs(np.einsum("ij,ij->i", ap, n)) / np.sqrt(np.where(ok, nn, 1.0)), np.inf)
    proj = ap - (np.einsum("ij,ij->i", ap, n) / np.where(ok, nn, 1.0))[:, None] * n
    d00, d01, d11 = np.einsum("ij,ij->i", ab, ab), np.einsum("ij,ij->i", ab, ac), np.einsum("ij,ij->i", ac, ac)
    d20, d21 = np.einsum("ij,ij->i", proj, ab), np.einsum("ij,ij->i", proj, ac)
    den = np.where(ok, d00 * d11 - d01 * d01, 1.0)
    vv, ww = (d11 * d20 - d01 * d21) / den, (d00 * d21 - d01 * d20) / den
    inside = ok & (vv >= 0) & (ww >= 0) & (vv + ww <= 1)
    best = np.where(inside, plane, np.inf)
    out = ~inside
    if out.any():
        for u0, u1 in ((a, b), (b, c), (c, a)):
            e = u1[out] - u0[out]
            k = np.clip(
                np.einsum("ij,ij->i", p - u0[out], e) / np.maximum(np.einsum("ij,ij->i", e, e), 1e-18), 0.0, 1.0
            )
            best[out] = np.minimum(best[out], np.linalg.norm(p - (u0[out] + e * k[:, None]), axis=1))
    return float(best.min())


def _near_axis(v: F64, bv: BoundingVolume) -> NDArray[np.bool_]:
    """Vertices within 1.6 radii (+2 cm) of the volume axis: the surface a ray from the axis can reach first."""
    d = bv.end - bv.start
    k = np.clip(((v - bv.start) @ d) / (d @ d), 0.0, 1.0)
    dist = np.linalg.norm(v - (bv.start + k[:, None] * d), axis=1)
    return np.asarray(dist <= 1.6 * bv.radius_m + 0.02)


class PointBuilder:
    """Resolves point specs against the avatar surface."""

    def __init__(self, specs: list[PointSpec]) -> None:
        self.specs = specs

    def build(self, volumes: list[BoundingVolume], v: F64, faces: I64) -> list[ArrangementPoint]:
        """Resolve every spec. Raises PointConfigError if a spec references an unknown volume or misses the surface."""
        by_name = {b.name: b for b in volumes}
        normals = vertex_normals(v, faces)
        tri_all = v[faces]
        out: list[ArrangementPoint] = []
        for s in self.specs:
            bv = by_name.get(s.bv)
            if bv is None:
                raise PointConfigError(f"point {s.name}: unknown bounding volume {s.bv!r}")
            a, f, side = ray_frame(bv)
            th = np.radians(s.angle_deg)
            d = np.cos(th) * f + np.sin(th) * side
            origin = bv.point_at(s.t)
            mine = _near_axis(v, bv)
            mine[bv.vertex_ids] = True
            tris = np.flatnonzero(mine[faces].any(axis=1))
            hit = _ray_hit(origin, d, tri_all[tris])
            if hit is None:
                raise PointConfigError(
                    f"point {s.name}: ray from {s.bv} axis at {s.angle_deg:g} deg, t={s.t:g} misses the surface"
                )
            dist, k, bary = hit
            face = faces[tris[k]]
            n = bary @ normals[face]
            n /= np.linalg.norm(n)
            if n @ d < 0:
                n = -n
            surf = origin + dist * d
            near = np.flatnonzero(np.linalg.norm(v - surf, axis=1) < s.offset_m + 0.15)
            tri_near = tri_all[np.flatnonzero(np.isin(faces, near).any(axis=1))]
            delta = s.offset_m
            for _ in range(8):  # make the point sit exactly offset_m from the surface, also in concave places
                dmin = _dist_point_triangles(surf + delta * n, tri_near)
                if abs(dmin - s.offset_m) < 1e-5:
                    break
                delta += s.offset_m - dmin
            up = a if not bv.limb else -a
            up = up - (up @ n) * n
            if np.linalg.norm(up) < 1e-6:
                up = np.array([0.0, 1.0, 0.0]) - n[1] * n
            up /= np.linalg.norm(up)
            out.append(ArrangementPoint(s, surf + delta * n, n, up, np.cross(up, n)))
        return out


def find_point(points: list[ArrangementPoint], name: str) -> ArrangementPoint:
    """Point by name or UnknownArrangementPointError."""
    for p in points:
        if p.spec.name == name:
            return p
    raise UnknownArrangementPointError(name, [p.spec.name for p in points])

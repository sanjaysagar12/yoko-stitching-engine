"""Size-exact placement surfaces: a plane (``flat``) or a cylinder (``roll``, ``around``). Pure functions, metres.

A piece is placed by an **isometry** of its flat pattern, nothing else: every edge length, angle, area and every
distance measured inside the piece stays what the pattern says.

* ``flat``: rigid. A local point ``(x, y)`` (piece coordinates relative to the reference, after mirror and rotation)
  goes to ``origin + r n + (x + s) e_x + (y + t) e_y``.
* cylinder: rolled about an axis. ``(x, y)`` goes to angle ``phi = (x + s) / r`` about the axis, axial position
  ``y + t`` and radius ``r``. A cylinder is developable, so arc length along rows, distance along the axis and all
  angles on the surface are exact (the cylinder unrolls onto the plane without stretch).

The three numbers ``theta = (r, s, t)`` are what the collision solver may change: ``r`` the distance from the body
(flat: along the point normal from the surface point; cylinder: the radius), ``s`` the shift across (flat: along
``e_x``; cylinder: arc length at radius ``r``), ``t`` the shift along ``e_y`` / the axis. Every value of ``theta`` is an
isometry; the solver never touches the piece shape.

``roll`` uses the axis of the body region (ring) of the point and a radius shared by the ring; ``around`` uses the
axis of the point's bounding volume and starts at the point's own radius.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .errors import RingNotFoundError
from .loaders import Avatar, PointRecord
from .placement import target_frame

F64 = NDArray[np.float64]

# bounding volumes of each ring (the region a ``roll`` wraps around); head, hands and feet have none
RING_BVS: dict[str, tuple[str, ...]] = {
    "torso": ("Hip", "Waist", "Chest"),
    "neck": ("Neck",),
    "arm_L": ("UpperArm_L", "Forearm_L"),
    "arm_R": ("UpperArm_R", "Forearm_R"),
    "leg_L": ("Thigh_L", "Shin_L"),
    "leg_R": ("Thigh_R", "Shin_R"),
}
SYMMETRIC_RINGS = ("torso", "neck")


def ring_of_bv(bv: str) -> str:
    """Ring of a bounding volume (``RingNotFoundError`` for head, hands, feet)."""
    for ring, bvs in RING_BVS.items():
        if bv in bvs:
            return ring
    raise RingNotFoundError(f"bounding volume {bv!r} belongs to no ring (rings: {sorted(RING_BVS)})")


def ring_axis(avatar: Avatar, ring: str) -> tuple[F64, F64]:
    """(point on the axis, unit direction) of a ring, from the bounding-volume axes.

    Torso and neck: the least-squares line through the axis end points of their volumes, kept in the sagittal plane
    (x = 0) so that a symmetric body gives symmetric placements. Arms and legs: the straight line from the first joint
    (shoulder, hip) to the last (wrist, ankle).
    """
    if ring not in RING_BVS:
        raise RingNotFoundError(f"unknown ring {ring!r}; rings: {sorted(RING_BVS)}")
    vols = [avatar.volumes[n] for n in RING_BVS[ring] if n in avatar.volumes]
    if not vols:
        raise RingNotFoundError(f"avatar has no bounding volumes for ring {ring!r}")
    if ring in SYMMETRIC_RINGS:
        pts = np.array([p for v in vols for p in (v.start, v.end)])
        c = pts.mean(axis=0)
        d = np.linalg.svd(pts - c)[2][0]
        c[0], d[0] = 0.0, 0.0
        d = d / np.linalg.norm(d)
        return c, (d if d[1] >= 0 else -d)
    a, b = vols[0].start, vols[-1].end
    return np.asarray(a, dtype=np.float64), (b - a) / np.linalg.norm(b - a)


@dataclass(frozen=True)
class Surface:
    """Where a piece lies (see the module docstring). ``kind`` is ``flat`` or ``cylinder``.

    flat: ``origin`` is the surface point under the arrangement point, ``e_n`` the outward normal, ``e_x``/``e_y`` the
    directions of piece x / y. cylinder: ``origin`` is the point of the axis at the anchor, ``e_n`` the radial unit
    toward the anchor, ``e_y`` the axis (oriented with the point's up), ``e_x`` the direction of increasing piece x
    around the axis. ``r_skin`` is the distance from the body surface used for the standoff (flat: 0; cylinder: the
    radius of the skin under the anchor).
    """

    kind: str
    origin: F64
    e_n: F64
    e_x: F64
    e_y: F64
    r_skin: float

    def map(self, xy: F64, theta: F64) -> F64:
        """World points (N,3) of local points ``xy`` (N,2) at parameters ``theta = (r, s, t)``."""
        r, s, t = float(theta[0]), float(theta[1]), float(theta[2])
        x, y = xy[:, 0] + s, xy[:, 1] + t
        if self.kind == "flat":
            return np.asarray(self.origin + r * self.e_n + np.outer(x, self.e_x) + np.outer(y, self.e_y))
        phi = x / r
        return np.asarray(
            self.origin
            + np.outer(y, self.e_y)
            + r * (np.outer(np.cos(phi), self.e_n) + np.outer(np.sin(phi), self.e_x))
        )

    def jacobian(self, xy: F64, theta: F64) -> F64:
        """d world / d theta for local points (N,3,3): ``[:, :, k]`` is the derivative along parameter k."""
        n = len(xy)
        out = np.empty((n, 3, 3))
        if self.kind == "flat":
            out[:, :, 0], out[:, :, 1], out[:, :, 2] = self.e_n, self.e_x, self.e_y
            return out
        r, s = float(theta[0]), float(theta[1])
        phi = (xy[:, 0] + s) / r
        rho = np.outer(np.cos(phi), self.e_n) + np.outer(np.sin(phi), self.e_x)
        tau = np.outer(-np.sin(phi), self.e_n) + np.outer(np.cos(phi), self.e_x)
        out[:, :, 0] = rho - phi[:, None] * tau
        out[:, :, 1] = tau
        out[:, :, 2] = self.e_y
        return out

    def normal(self, world: F64) -> F64:
        """Outward unit normal of the surface at world points (N,3) (away from the body)."""
        if self.kind == "flat":
            return np.tile(self.e_n, (len(world), 1))
        q = world - self.origin
        q = q - np.outer(q @ self.e_y, self.e_y)
        return np.asarray(q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12))

    def develop(self, world: F64, theta: F64, x_mid: float = 0.0) -> F64:
        """Inverse of ``map``: (N,3) local ``x``, ``y`` and the offset from the surface (along the normal).

        For a cylinder the angle is taken within half a turn of local ``x_mid`` (the middle of the piece).
        """
        r, s, t = float(theta[0]), float(theta[1]), float(theta[2])
        q = world - self.origin
        y = q @ self.e_y - t
        if self.kind == "flat":
            return np.column_stack([q @ self.e_x - s, y, q @ self.e_n - r])
        a, b = q @ self.e_n, q @ self.e_x
        mid = (x_mid + s) / r
        phi = mid + (np.arctan2(b, a) - mid + np.pi) % (2.0 * np.pi) - np.pi
        return np.column_stack([r * phi - s, y, np.hypot(a, b) - r])

    def standoff(self, theta: F64) -> float:
        """Distance of the surface from the body under the anchor (display space, not ease)."""
        return float(theta[0]) - self.r_skin


def local_coords(xy: F64, ref: F64, rotation_deg: float, mirror: bool) -> F64:
    """Piece coordinates (N,2) -> local coordinates: mirror across x = ref.x, then about the reference, rotated."""
    x = 2.0 * ref[0] - xy[:, 0] if mirror else xy[:, 0]
    d = np.column_stack([x - ref[0], xy[:, 1] - ref[1]])
    c, s = np.cos(np.radians(rotation_deg)), np.sin(np.radians(rotation_deg))
    return np.column_stack([c * d[:, 0] - s * d[:, 1], s * d[:, 0] + c * d[:, 1]])


def flat_surface(point: PointRecord, flip: bool) -> tuple[Surface, float]:
    """Plane of an arrangement point and the start ``r`` (the point's own offset)."""
    fr = target_frame(point.normal, point.up, flip)
    n = point.normal / np.linalg.norm(point.normal)
    origin = point.position - point.offset_m * n
    return Surface("flat", origin, n, fr.t, fr.u, 0.0), float(point.offset_m)


def cylinder_surface(
    point: PointRecord, axis_point: F64, axis_dir: F64, flip: bool, centre: bool
) -> tuple[Surface, float]:
    """Cylinder about the axis through the arrangement point and its radius (axis to the point).

    ``centre``: the point lies on the sagittal plane of a symmetric ring, so the radial direction is kept in it.
    """
    a = axis_dir / np.linalg.norm(axis_dir)
    if a @ point.up < 0:
        a = -a
    q = axis_point + ((point.position - axis_point) @ a) * a
    rv = point.position - q
    if centre:
        rv = rv - np.array([rv[0], 0.0, 0.0])
    radius = float(np.linalg.norm(rv))
    e_n = rv / radius
    e_x = np.cross(a, e_n)
    skin = max(radius - point.offset_m, 1e-3)
    return Surface("cylinder", np.asarray(q), e_n, -e_x if flip else e_x, a, skin), radius

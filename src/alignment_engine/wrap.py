"""Wrapping a flat piece around a body part. Pure functions, metres.

Piece offsets from the reference are mapped to a cylinder about the bounding-volume axis: the offset ``s`` along the
x axis of the piece becomes the angle ``phi = s / R`` (so arc length along x is preserved), the offset ``h`` along y
stays the distance along the axis (preserved), and z becomes the radial thickness. Nothing is scaled.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .errors import WrapWidthExceededWarning

F64 = NDArray[np.float64]


@dataclass(frozen=True)
class WrapGeometry:
    """Cylinder the piece is wrapped on.

    ``axis_dir`` points toward the piece "up"; ``radial`` is the unit vector from the axis to the target;
    ``axis_point`` is the point of the axis at the height of the target; ``radius`` is the distance from the axis to
    the target (surface plus offset).
    """

    axis_point: F64
    axis_dir: F64
    radial: F64
    radius: float


@dataclass(frozen=True)
class WrapInfo:
    """What happened while wrapping: piece width (after rotation), angle spanned, clamping and the warning text."""

    width_m: float
    angle_deg: float
    clamped: bool
    warning: str


def wrap_geometry(target: F64, vol_start: F64, vol_end: F64, up: F64) -> WrapGeometry:
    """Cylinder through ``target`` about the axis ``vol_start -> vol_end`` (direction chosen to match ``up``)."""
    a = vol_end - vol_start
    a = a / np.linalg.norm(a)
    if a @ up < 0:
        a = -a
    q = vol_start + ((target - vol_start) @ a) * a
    r = target - q
    rho = float(np.linalg.norm(r))
    return WrapGeometry(np.asarray(q, dtype=np.float64), np.asarray(a, dtype=np.float64), r / rho, rho)


def wrap_vertices(local: F64, ref: F64, geom: WrapGeometry, rotation_deg: float, flip: bool) -> tuple[F64, WrapInfo]:
    """Wrap piece points ``local`` (N,3, already mirrored if requested) around ``geom``.

    The reference point ``ref`` maps to ``axis_point + radius * radial``. ``rotation_deg`` turns the piece in its own
    plane (counter-clockwise about its normal) before wrapping; ``flip`` reverses the direction of x around the body.
    """
    th = np.radians(rotation_deg)
    d = local - ref
    s = np.cos(th) * d[:, 0] - np.sin(th) * d[:, 1]
    h = np.sin(th) * d[:, 0] + np.cos(th) * d[:, 1]
    half = np.pi * geom.radius
    width = float(s.max() - s.min()) if len(s) else 0.0
    clamped = bool(width > 2.0 * half + 1e-9)
    warning = ""
    if clamped:
        text = f"piece is {width * 100:.1f} cm wide but the circumference is {2 * half * 100:.1f} cm: clamped"
        warning = f"{WrapWidthExceededWarning.__name__}: {text}"
        s = np.clip(s, -half, half)
    phi = (-s if flip else s) / geom.radius
    tang = np.cross(geom.axis_dir, geom.radial)
    rad = geom.radius + d[:, 2]
    out = (
        geom.axis_point
        + h[:, None] * geom.axis_dir
        + rad[:, None] * (np.cos(phi)[:, None] * geom.radial + np.sin(phi)[:, None] * tang)
    )
    return np.asarray(out, dtype=np.float64), WrapInfo(width, float(np.degrees(width / geom.radius)), clamped, warning)

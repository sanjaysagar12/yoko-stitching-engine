"""In-plane orientation (``orient``), tilt of the placement frame (``tilt``) and level registration (``anchor`` with a
``level``). All are isometries: they choose where a rigid piece goes, never its shape. Metres, degrees.

* ``orient``: ``"none"`` (only ``rotation_deg``), ``"auto"`` (the grainline, else the long axis of a tall piece,
  parallel to the point's up / the ring axis, top up; a grain more than 45 degrees off the pattern's vertical is a
  deliberate cross-grain cut and is kept), ``"grain"`` (the grainline parallel to the axis whatever its angle; without
  one like ``none``) or a quarter turn ``0 | 90 | 180 | 270``. The result is added to ``rotation_deg``.
* ``tilt``: ``{pitch_deg, roll_deg}`` rotates the placement frame (and a cylinder's axis) about the reference: pitch
  leans the top of the piece away from the body (+) or toward it (-), roll leans the top toward the piece's +x (+).
  Each is clamped to ``TILT_LIMIT``.
* ``anchor: {"level": L, "piece_point": P}`` puts the named piece point ``P`` at the height of body level ``L``
  (``levels.py``); ``{"level": L, "edge": "top"|"bottom"}`` the top / bottom of the outline; ``{"edge": ...}`` without a
  level uses the height of the arrangement point; ``offset_m`` puts it that far above the level. With
  ``schema_version`` 5 a bottom (a piece on a waist, hip or leg point) without ``anchor`` is registered by default
  (``registration_spec``).
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .placement import rodrigues

F64 = NDArray[np.float64]
TILT_LIMIT = 45.0
GRAIN_AUTO_LIMIT = 45.0
# named level lines of a pattern (case and separators ignored) -> body level; earlier entries win
LEVEL_LINES: tuple[tuple[str, str], ...] = (
    (r"waist ?line", "Waist"),
    (r"hip ?line", "Hip"),
    (r"(bust|chest) ?line", "Chest"),
)


def _norm_angle(a: float) -> float:
    """Angle in (-180, 180]."""
    a = float(np.mod(a + 180.0, 360.0) - 180.0)
    return 180.0 if a == -180.0 else a


def orient_rotation(bundle: dict[str, Any], orient: Any) -> tuple[float, str]:
    """(extra in-plane rotation in degrees, why) for an ``orient`` value."""
    if orient is None or orient == "none":
        return 0.0, "none"
    if isinstance(orient, (int, float)) and not isinstance(orient, bool):
        return float(orient), f"quarter turn {int(orient)}"
    grain = bundle.get("grainline")
    if grain is not None:
        rho = _norm_angle(90.0 - float(grain["rotation_deg"]))
        if rho > 90.0:  # the grain is a line: pick the sense nearest to the pattern's up
            rho -= 180.0
        elif rho <= -90.0:
            rho += 180.0
        if orient == "auto" and abs(rho) > GRAIN_AUTO_LIMIT:
            return 0.0, f"grainline {float(grain['rotation_deg']):.0f} deg is a cross-grain cut: kept"
        return rho, f"grainline {float(grain['rotation_deg']):.1f} deg -> vertical"
    if orient == "grain":
        return 0.0, "no grainline: kept"
    bb = bundle["bbox_m"]
    w, h = bb["max"][0] - bb["min"][0], bb["max"][1] - bb["min"][1]
    return 0.0, ("long axis already vertical" if h >= w else "wide piece: kept")


def tilt_angles(pl: dict[str, Any]) -> tuple[float, float]:
    """(pitch, roll) in degrees, clamped to ``TILT_LIMIT``."""
    t = pl.get("tilt") or {}
    pitch = float(np.clip(float(t.get("pitch_deg", 0.0)), -TILT_LIMIT, TILT_LIMIT))
    roll = float(np.clip(float(t.get("roll_deg", 0.0)), -TILT_LIMIT, TILT_LIMIT))
    return pitch, roll


def tilt_matrix(e_n: F64, e_x: F64, e_y: F64, pitch: float, roll: float) -> F64:
    """Rotation (3x3): pitch turns ``e_y`` toward ``e_n`` (top away from the body), then roll toward ``e_x``."""
    r = np.eye(3)
    if pitch:
        r = rodrigues(np.cross(e_y, e_n), pitch) @ r
    if roll:
        y1 = r @ e_y
        r = rodrigues(np.cross(y1, r @ e_x), roll) @ r
    return r


def default_registration(bundle: dict[str, Any]) -> dict[str, str] | None:
    """``{"level", "piece_point"}`` from the first level line of the piece (``LEVEL_LINES``), else None."""
    names = [q["name"] for q in bundle["named_points"]]
    for pattern, level in LEVEL_LINES:
        rx = re.compile(rf"^{pattern}$", re.IGNORECASE)
        for n in names:
            if rx.match(n.replace("_", "").replace("-", "")):
                return {"level": level, "piece_point": n}
    return None


BOTTOM_BVS = ("Waist", "Hip", "Thigh_L", "Thigh_R", "Shin_L", "Shin_R")


def registration_spec(
    pl: dict[str, Any], bundle: dict[str, Any], schema_version: int, bv: str = ""
) -> dict[str, Any] | None:
    """The level registration of a placement: an explicit ``anchor`` object; else, from ``schema_version`` 5 on and only
    for bottoms (a piece on a waist, hip or leg point), the piece's level line, or for a piece without one on a waist
    or hip point its top edge at the waist (a skirt hangs from the waist, not from the middle of its point). Tops keep
    their point: a shirt's waist line is drafted for a body whose waist is lower than SMPL's narrowest girth, and on
    male_173 it would lift the back panel above the shoulders."""
    a = pl.get("anchor")
    if isinstance(a, dict):
        return dict(a)
    if a is not None or schema_version < 5 or bv not in BOTTOM_BVS:
        return None
    reg = default_registration(bundle)
    if reg is None and bv in ("Waist", "Hip"):
        reg = {"level": "Waist", "edge": "top"}
    return reg


def edge_point(outline: F64, edge: str) -> F64:
    """The registration point of an edge: among the outline points within 1 mm of the extreme height (``top``: the
    highest), the one nearest the middle of the piece across. A band rolled on a slightly tilted ring is not level, so
    one well-defined point is registered, not the lowest corner wherever it ends up."""
    y = outline[:, 1]
    ext = float(y.max() if edge == "top" else y.min())
    cand = outline[np.abs(y - ext) <= 0.001]
    mid = 0.5 * float(outline[:, 0].min() + outline[:, 0].max())
    return np.asarray(cand[int(np.argmin(np.abs(cand[:, 0] - mid)))], dtype=np.float64)

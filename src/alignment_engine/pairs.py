"""Pair placement: place a piece and its mirrored clone together. Pure functions on placement dicts."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .errors import NoMirroredPointError, ReferenceNotOnAxisError

AXIS_TOLERANCE_M = 1e-3  # the reference of a centre-line pair must lie within 1 mm of the mirror axis
MODES = ("same_anchor", "mirror_anchor")
_SWAPS = (("_L", "_R"), ("_Left", "_Right"))


def counterpart_point(name: str, known: Iterable[str]) -> str:
    """Left/right counterpart of an arrangement point (``_L`` / ``_R`` or ``_Left`` / ``_Right``).

    A point without a side suffix is a centre point and maps to itself. Raises NoMirroredPointError if the counterpart
    does not exist.
    """
    names = set(known)
    for a, b in _SWAPS:
        for x, y in ((a, b), (b, a)):
            if name.endswith(x):
                other = name[: -len(x)] + y
                if other not in names:
                    raise NoMirroredPointError(name, other)
                return other
    return name


def place_pair(
    base: dict[str, Any],
    derived_id: str,
    mode: str,
    *,
    point_names: Iterable[str],
    ref_axis_distance_m: float | None = None,
) -> list[dict[str, Any]]:
    """Two placements ``[base, clone]``.

    * ``same_anchor`` (centre-line pair): same point, reference, offset, wrap, rotation. If ``ref_axis_distance_m`` (the
      distance of the reference from the mirror axis, in the base piece) is given and exceeds 1 mm, raises
      ReferenceNotOnAxisError: only then do the two copies meet exactly along the centre line.
    * ``mirror_anchor`` (side pair): the clone goes on the counterpart point, same offset / wrap / reference, rotation
      negated (the frame at the other side is the mirror of this one, so the clone is the exact mirror image).
    """
    if mode not in MODES:
        raise ValueError(f"unknown pair mode {mode!r}; use one of {MODES}")
    first = dict(base)
    clone = dict(base)
    clone["piece"] = derived_id
    if mode == "same_anchor":
        if ref_axis_distance_m is not None and ref_axis_distance_m > AXIS_TOLERANCE_M:
            raise ReferenceNotOnAxisError(str(base.get("reference", "bbox_centre")), ref_axis_distance_m)
    else:
        clone["point"] = counterpart_point(str(base["point"]), point_names)
        clone["rotation_deg"] = -float(base.get("rotation_deg", 0.0))
    return [first, clone]

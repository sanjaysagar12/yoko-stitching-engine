"""Body levels: heights (world y, metres) of the landmarks a pattern is registered on, found on the avatar itself.

The engine only sees contract files, so every level is measured on the mesh and the bounding volumes, with the
definitions of ``avatar_engine.measurements``:

* ``Waist``: the narrowest torso section between the Waist volume's end points (spine1 .. spine2, natural waist);
* ``Hip``: going down from the pelvis joint, the first torso section as long as the measured hip girth (``_hip``);
* ``Chest``: the widest torso section in the lower 60 % of the Chest volume (bust level, below the arm pits);
* ``Crotch``: the lowest midline vertex (x = 0, where the legs join) between the knees and the top of the Hip volume
  (the inseam landmark);
* ``Neck``: the base of the neck (start of the Neck volume); ``Shoulder``: the shoulder joints (upper arm starts);
* ``Knee`` / ``Ankle``: the knee / ankle joints (ends of the thigh / shin volumes), mean of left and right.

Sections are the torso cross-sections of ``slice_wrap`` (arms cut out). Results are cached on the avatar.
"""

from __future__ import annotations

import numpy as np

from .errors import UnknownLevelError
from .loaders import Avatar
from .slice_wrap import TorsoSections

LEVELS = ("Neck", "Shoulder", "Chest", "Waist", "Hip", "Crotch", "Knee", "Ankle")
_STEP = 0.005


def _sections(avatar: Avatar) -> TorsoSections:
    if "sections" not in avatar.cache:
        avatar.cache["sections"] = TorsoSections.build(avatar)
    s: TorsoSections = avatar.cache["sections"]
    return s


def _extreme(sec: TorsoSections, lo: float, hi: float, largest: bool) -> float:
    a, b = sec.y_range
    ys = np.arange(max(lo, a), min(hi, b) + 1e-9, _STEP)
    if not len(ys):
        return float(np.clip(0.5 * (lo + hi), a, b))
    per = np.array([sec.perimeter(float(y)) for y in ys])
    return float(ys[int(np.argmax(per) if largest else np.argmin(per))])


def _hip(sec: TorsoSections, avatar: Avatar, crotch: float) -> float:
    """Going down from the pelvis joint: the first height whose torso girth reaches the avatar's measured hip girth.
    Close to the crotch the sections run round both thighs and only grow, so "the widest" is not usable there; without
    the measurement, the widest section in the upper 3/4 between the crotch and the pelvis."""
    top = float(avatar.volumes["Hip"].start[1])
    target = avatar.bundle.get("achieved_cm", {}).get("hip_circ")
    if target is not None:
        for y in np.arange(top, crotch, -_STEP):
            if sec.perimeter(float(y)) >= float(target) / 100.0:
                return float(y)
    return _extreme(sec, crotch + 0.25 * (top - crotch), top, largest=True)


def body_levels(avatar: Avatar) -> dict[str, float]:
    """Height (world y) of every level in ``LEVELS`` that the avatar has the volumes for."""
    if "levels" in avatar.cache:
        out: dict[str, float] = avatar.cache["levels"]
        return out
    vols, v = avatar.volumes, avatar.vertices
    out = {}

    def mean_y(names: tuple[str, ...], end: bool) -> float | None:
        ys = [float((vols[n].end if end else vols[n].start)[1]) for n in names if n in vols]
        return float(np.mean(ys)) if ys else None

    knee = mean_y(("Thigh_L", "Thigh_R"), True)
    ankle = mean_y(("Shin_L", "Shin_R"), True)
    if "Neck" in vols:
        out["Neck"] = float(vols["Neck"].start[1])
    shoulder = mean_y(("UpperArm_L", "UpperArm_R"), False)
    if shoulder is not None:
        out["Shoulder"] = shoulder
    centre = v[(np.abs(v[:, 0]) < 1e-4) & (v[:, 1] > (knee if knee is not None else 0.3))]
    hip_top = float(vols["Hip"].start[1]) if "Hip" in vols else None
    if len(centre) and hip_top is not None:
        below = centre[centre[:, 1] < hip_top]
        if len(below):
            out["Crotch"] = float(below[:, 1].min())
    try:
        sec = _sections(avatar)
    except ValueError:
        sec = None
    if sec is not None:
        if "Waist" in vols:
            w = vols["Waist"]
            out["Waist"] = _extreme(sec, min(w.start[1], w.end[1]), max(w.start[1], w.end[1]), largest=False)
        if "Hip" in vols:
            out["Hip"] = _hip(sec, avatar, out.get("Crotch", float(min(vols["Hip"].start[1], vols["Hip"].end[1]))))
        if "Chest" in vols:
            c = vols["Chest"]
            lo, hi = float(min(c.start[1], c.end[1])), float(max(c.start[1], c.end[1]))
            out["Chest"] = _extreme(sec, lo, lo + 0.6 * (hi - lo), largest=True)
    if knee is not None:
        out["Knee"] = knee
    if ankle is not None:
        out["Ankle"] = ankle
    avatar.cache["levels"] = out
    return out


def level_height(avatar: Avatar, name: str) -> float:
    """Height of one level (``UnknownLevelError`` for a name that is not in ``LEVELS`` or not on this avatar)."""
    lv = body_levels(avatar)
    if name not in lv:
        raise UnknownLevelError(name, sorted(lv))
    return lv[name]

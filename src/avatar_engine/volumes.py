"""Bounding volumes: one oriented capsule per body part, fitted from the vertices the skinning weights assign to it.

A vertex belongs to the body part of its dominant joint. The capsule axis runs from a joint to its child joint (or to
a virtual landmark for the head, hips); the radius is the 90th percentile of the distance of the part's vertices to the
axis. All coordinates are metres.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .bodymodel import BodyModel
from .errors import PointConfigError

F64 = NDArray[np.float64]
I64 = NDArray[np.int64]


@dataclass(frozen=True)
class BoundingVolume:
    """Oriented capsule. ``axis`` runs from ``start`` to ``end`` (metres); ``vertex_ids`` are the part's vertices."""

    name: str
    kind: str
    joint: str
    child_joint: str
    start: F64
    end: F64
    radius_m: float
    vertex_ids: I64
    limb: bool  # limbs: "up" points toward the torso; others: up = start -> end

    @property
    def length_m(self) -> float:
        """Axis length."""
        return float(np.linalg.norm(self.end - self.start))

    @property
    def axis(self) -> F64:
        """Unit axis start -> end."""
        d = self.end - self.start
        return d / np.linalg.norm(d)

    def point_at(self, t: float) -> F64:
        """Point on the axis, t in [0, 1] from start to end."""
        return self.start + t * (self.end - self.start)


# base name -> (joint, child joint or virtual landmark, dominant-joint set, limb)
_SYMMETRIC = {
    "UpperArm": ("shoulder", "elbow", ["shoulder"], True),
    "Forearm": ("elbow", "wrist", ["elbow"], True),
    "Hand": ("wrist", "<hand>", ["wrist", "<fingers>"], True),
    "Thigh": ("hip", "knee", ["hip"], True),
    "Shin": ("knee", "ankle", ["knee"], True),
    "Foot": ("ankle", "foot", ["ankle", "foot"], True),
}
_CENTRAL = {
    "Head": ("head", "<head_top>", ["head", "jaw", "left_eye", "right_eye"], False),
    "Neck": ("neck", "head", ["neck"], False),
    "Chest": ("spine2", "neck", ["spine2", "spine3"], False),
    "Waist": ("spine1", "spine2", ["spine1"], False),
    "Hip": ("pelvis", "<crotch>", ["pelvis"], False),
}


class VolumeBuilder:
    """Builds the bounding volumes of a model for given vertices ``v`` and joints ``j``."""

    def __init__(self, model: BodyModel) -> None:
        self.m = model
        self.jn = {n: i for i, n in enumerate(model.joint_names)}
        self.dominant = np.argmax(model.skinning_weights, axis=1)

    def _joints(self, names: list[str], side: str | None) -> list[int]:
        out: list[int] = []
        for n in names:
            if n == "<fingers>":
                out += [
                    i
                    for k, i in self.jn.items()
                    if side
                    and k.startswith(side + "_")
                    and k.split("_", 1)[1][:3] in ("ind", "mid", "pin", "rin", "thu", "han")
                ]
                continue
            full = f"{side}_{n}" if side else n
            if full in self.jn:
                out.append(self.jn[full])
        return out

    def _landmark(self, tag: str, side: str | None, v: F64, j: F64, owned: I64) -> F64:
        if tag == "<head_top>":
            top = owned[np.argmax(v[owned, 1])]
            h = j[self.jn["head"]]
            return np.array([h[0], v[top, 1], h[2]])
        if tag == "<crotch>":
            mid = np.flatnonzero(np.abs(v[:, 0]) < 1e-4)
            knee, pel = j[self.jn["left_knee"]][1], j[self.jn["pelvis"]][1]
            c = mid[(v[mid, 1] > knee) & (v[mid, 1] < pel)]
            low = c[np.argmin(v[c, 1])]
            return np.array([0.0, v[low, 1], j[self.jn["pelvis"]][2]])
        if tag == "<hand>":  # tip of the hand chain: middle finger last joint (SMPL-X) or the hand joint (SMPL)
            for cand in (f"{side}_middle3", f"{side}_hand"):
                if cand in self.jn:
                    return np.asarray(j[self.jn[cand]], dtype=np.float64)
        raise PointConfigError(f"unknown landmark {tag!r}")

    def build(self, v: F64, j: F64) -> list[BoundingVolume]:
        """All volumes, central parts first, then limbs as ``<Base>_L`` / ``<Base>_R``."""
        out: list[BoundingVolume] = []
        specs: list[tuple[str, str | None, tuple[str, str, list[str], bool]]] = [
            (k, None, s) for k, s in _CENTRAL.items()
        ]
        for sd in ("left", "right"):
            specs += [(f"{k}_{'L' if sd == 'left' else 'R'}", sd, s) for k, s in _SYMMETRIC.items()]
        for name, side, (a, b, parts, limb) in specs:
            ja = self.jn[f"{side}_{a}" if side else a]
            ids = self._joints(parts, side)
            owned = np.flatnonzero(np.isin(self.dominant, ids)).astype(np.int64)
            if len(owned) == 0:
                raise PointConfigError(f"bounding volume {name}: no vertices are dominated by joints {parts}")
            start = j[ja]
            if b.startswith("<"):
                end = self._landmark(b, side, v, j, owned)
                child = b.strip("<>")
            else:
                jb = self.jn[f"{side}_{b}" if side else b]
                end, child = j[jb], self.m.joint_names[jb]
            axis = end - start
            ln = float(np.linalg.norm(axis))
            u = axis / ln
            rel = v[owned] - start
            t = rel @ u
            rad = np.linalg.norm(rel - np.outer(t, u), axis=1)
            keep = (t >= -0.1 * ln) & (t <= 1.1 * ln)
            radius = float(np.percentile(rad[keep] if keep.any() else rad, 90))
            out.append(BoundingVolume(name, "capsule", self.m.joint_names[ja], child, start, end, radius, owned, limb))
        return out

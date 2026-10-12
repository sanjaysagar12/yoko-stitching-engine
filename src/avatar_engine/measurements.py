"""Measurement definitions, derived procedurally from the template (no hard-coded vertex indices).

Every definition is evaluated from the vertices ``v`` (N,3) and joints ``J`` (K,3) in metres, so it works for any
body shape of the same model, and for SMPL (6890 vertices) as well as SMPL-X (10475). See ``docs/measurements.md``.

Circumferences are rings of mesh-edge points (see ``slicing.Ring``) fixed once on the template at a landmark level.
Lengths are distances between fixed vertices / joints, or along a fixed vertex path.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from .bodymodel import BodyModel
from .errors import MeasurementDefinitionError
from .slicing import MeshTopology, Ring

F64 = NDArray[np.float64]
I64 = NDArray[np.int64]

KNOWN = (
    "height",
    "neck_circ",
    "shoulder_width",
    "chest_circ",
    "waist_circ",
    "hip_circ",
    "upper_arm_circ",
    "wrist_circ",
    "thigh_circ",
    "arm_length",
    "inseam",
    "neck_back_to_waist",
)


class MeasurementDef(ABC):
    """One measurement. ``evaluate`` returns metres."""

    name: str
    description: str

    @abstractmethod
    def evaluate(self, v: F64, j: F64) -> float:
        """Value in metres for vertices ``v`` and joints ``j``."""


@dataclass(frozen=True)
class Landmark:
    """A vertex (``kind == "v"``) or a joint (``"j"``) of the model."""

    kind: str
    idx: int

    def at(self, v: F64, j: F64) -> F64:
        """Position (3,) in metres."""
        return np.asarray(v[self.idx] if self.kind == "v" else j[self.idx], dtype=np.float64)


@dataclass(frozen=True)
class HeightDef(MeasurementDef):
    """Top-most minus bottom-most vertex (feet are on y = 0, so this is the top of the head)."""

    name: str = "height"
    description: str = "highest vertex above the floor (top of the head, standing, zero pose)"

    def evaluate(self, v: F64, j: F64) -> float:
        return float(v[:, 1].max() - v[:, 1].min())


@dataclass(frozen=True)
class DistanceDef(MeasurementDef):
    """Straight distance between two landmarks (``vertical_only``: only the y difference)."""

    name: str
    description: str
    a: Landmark
    b: Landmark
    vertical_only: bool = False

    def evaluate(self, v: F64, j: F64) -> float:
        d = self.a.at(v, j) - self.b.at(v, j)
        return float(abs(d[1]) if self.vertical_only else np.linalg.norm(d))


@dataclass(frozen=True)
class SumDef(MeasurementDef):
    """Sum of several measurements (e.g. arm length = shoulder-elbow + elbow-wrist)."""

    name: str
    description: str
    parts: tuple[MeasurementDef, ...]

    def evaluate(self, v: F64, j: F64) -> float:
        return sum(p.evaluate(v, j) for p in self.parts)


@dataclass(frozen=True)
class PathDef(MeasurementDef):
    """Length of the polyline through fixed vertices (a surface path chosen once on the template)."""

    name: str
    description: str
    vertex_ids: I64

    def evaluate(self, v: F64, j: F64) -> float:
        p = v[self.vertex_ids]
        return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())


@dataclass(frozen=True)
class RingDef(MeasurementDef):
    """A circumference. ``mirror_ring`` is the same loop on the other side for limbs (symmetric model).

    The remaining fields describe how ``RingCorrector`` may deform the surface to hit an exact length.
    """

    name: str
    description: str
    ring: Ring
    axis_a: int  # joint ids of the axis through the ring plane (axis point = a + frac * (b - a))
    axis_b: int
    frac: float
    vertical: bool  # torso/neck rings: plane normal = +y; limb rings: plane normal = limb axis
    falloff_m: float  # half-length of the smooth radial deformation along the axis
    parts: tuple[int, ...]  # joints whose skinning weight limits the deformation to this body part
    mirror_ring: Ring | None = None
    mirror_parts: tuple[int, ...] = ()
    level_m: float = field(default=0.0)  # height (torso) of the ring on the template, information only

    def evaluate(self, v: F64, j: F64) -> float:
        return self.ring.length(v)


# -- construction -----------------------------------------------------------------------------------------------
class _Builder:
    def __init__(self, model: BodyModel) -> None:
        self.m = model
        self.v, self.j = model.rest(np.zeros(1))
        self.topo = MeshTopology(model.faces)
        self.jn = {n: i for i, n in enumerate(model.joint_names)}
        n = len(self.v)
        if model.mirror_map is not None:
            self.mid = np.flatnonzero(model.mirror_map == np.arange(n))
        else:
            self.mid = np.flatnonzero(np.abs(self.v[:, 0]) < 1e-4)

    def J(self, name: str) -> F64:
        return np.asarray(self.j[self.jn[name]], dtype=np.float64)

    def loop(self, name: str, c: F64, normal: F64) -> Ring:
        r = self.topo.loop_around(self.v, c, normal, c)
        if r is None:
            raise MeasurementDefinitionError(name, f"no closed loop around the axis at {np.round(c, 3).tolist()}")
        return r

    def extreme_level(self, name: str, a: str, b: str, lo: float, hi: float, vertical: bool, largest: bool) -> float:
        """Fraction in [lo, hi] along joint a -> b where the ring length is smallest (or largest)."""
        A, B = self.J(a), self.J(b)
        n = np.array([0.0, 1.0, 0.0]) if vertical else (B - A) / np.linalg.norm(B - A)
        best: tuple[float, float] | None = None
        for f in np.linspace(lo, hi, 41):
            c = A + f * (B - A)
            r = self.topo.loop_around(self.v, c, n, c)
            if r is None:
                continue
            L = r.length(self.v)
            if best is None or (L > best[0] if largest else L < best[0]):
                best = (L, float(f))
        if best is None:
            raise MeasurementDefinitionError(name, f"no loop found between joints {a} and {b}")
        return best[1]

    def ring_def(
        self,
        name: str,
        desc: str,
        a: str,
        b: str,
        frac: float,
        vertical: bool,
        falloff: float,
        parts: list[str],
        limb: bool = False,
    ) -> RingDef:
        A, B = self.J(a), self.J(b)
        c = A + frac * (B - A)
        n = np.array([0.0, 1.0, 0.0]) if vertical else (B - A) / np.linalg.norm(B - A)
        ring = self.loop(name, c, n)
        mj = self.m.joint_mirror
        pid = tuple(self.jn[p] for p in parts)
        return RingDef(
            name=name,
            description=desc,
            ring=ring,
            axis_a=self.jn[a],
            axis_b=self.jn[b],
            frac=frac,
            vertical=vertical,
            falloff_m=falloff,
            parts=pid,
            mirror_ring=ring.mirrored(self.m.mirror_map) if limb and self.m.mirror_map is not None else None,
            mirror_parts=tuple(int(mj[p]) for p in pid) if limb else (),
            level_m=float(c[1]),
        )


def build_definitions(model: BodyModel, names: list[str] | None = None) -> dict[str, MeasurementDef]:
    """Definitions for ``names`` (default: all of ``KNOWN``) derived from the model's template (zero betas)."""
    want = list(KNOWN) if names is None else names
    unknown = [n for n in want if n not in KNOWN]
    if unknown:
        raise MeasurementDefinitionError(unknown[0], f"unknown measurement; known: {', '.join(KNOWN)}")
    b = _Builder(model)
    out: dict[str, MeasurementDef] = {}
    jn, v = b.jn, b.v
    have_hands = "left_wrist" in jn

    def need_ring(
        nm: str,
        desc: str,
        a: str,
        bb: str,
        vertical: bool,
        falloff: float,
        parts: list[str],
        *,
        frac: float | None = None,
        search: tuple[float, float] | None = None,
        largest: bool = False,
        limb: bool = False,
    ) -> None:
        if nm not in want:
            return
        f = frac if frac is not None else b.extreme_level(nm, a, bb, search[0], search[1], vertical, largest)  # type: ignore[index]
        out[nm] = b.ring_def(nm, desc, a, bb, f, vertical, falloff, parts, limb)

    need_ring(
        "neck_circ",
        "narrowest horizontal girth of the neck between the neck and head joints",
        "neck",
        "head",
        True,
        0.04,
        ["neck", "head", "spine3", "left_collar", "right_collar"],
        search=(0.25, 0.75),
    )
    need_ring(
        "chest_circ",
        "horizontal girth of the torso half way between the spine2 and spine3 joints (bust level, below the armpits)",
        "spine2",
        "spine3",
        True,
        0.10,
        ["spine1", "spine2", "spine3"],
        frac=0.5,
    )
    need_ring(
        "waist_circ",
        "narrowest horizontal torso girth between the spine1 and spine2 joints (natural waist)",
        "spine1",
        "spine2",
        True,
        0.10,
        ["pelvis", "spine1", "spine2"],
        search=(0.0, 1.0),
    )
    # hip: widest girth between the pelvis joint and just above the crotch (where the legs split)
    crotch = _crotch_vertex(b)
    y_crotch = float(v[crotch, 1])
    if "hip_circ" in want:
        a, bb = "pelvis", "left_hip"
        pa = b.J(a)
        levels = np.linspace(pa[1], y_crotch + 0.03, 41)
        best: tuple[float, float] | None = None
        for y in levels:
            c = np.array([0.0, y, pa[2]])
            r = b.topo.loop_around(v, c, np.array([0.0, 1.0, 0.0]), c)
            if r is not None and (best is None or r.length(v) > best[0]):
                best = (r.length(v), float(y))
        if best is None:
            raise MeasurementDefinitionError("hip_circ", "no loop between the pelvis and the crotch")
        # express the level as a fraction of the pelvis -> hip-joint segment's vertical drop (axis stays vertical)
        c = np.array([0.0, best[1], pa[2]])
        ring = b.loop("hip_circ", c, np.array([0.0, 1.0, 0.0]))
        out["hip_circ"] = RingDef(
            "hip_circ",
            "widest horizontal girth between the pelvis joint and the crotch (hips / buttocks)",
            ring,
            b.jn[a],
            b.jn[bb],
            float((pa[1] - best[1]) / max(pa[1] - b.J(bb)[1], 1e-9)),
            True,
            0.12,
            tuple(jn[p] for p in ("pelvis", "spine1", "left_hip", "right_hip")),
            level_m=best[1],
        )
    if have_hands:
        need_ring(
            "upper_arm_circ",
            "girth of the upper arm half way between shoulder and elbow joints (perpendicular to the arm)",
            "left_shoulder",
            "left_elbow",
            False,
            0.11,
            ["left_shoulder", "left_elbow"],
            frac=0.5,
            limb=True,
        )
        need_ring(
            "wrist_circ",
            "narrowest girth of the forearm between 60 % and 100 % from elbow to wrist joint",
            "left_elbow",
            "left_wrist",
            False,
            0.09,
            ["left_elbow", "left_wrist"],
            search=(0.6, 1.0),
            limb=True,
        )
    need_ring(
        "thigh_circ",
        "girth of the thigh 35 % of the way from the hip to the knee joint (perpendicular to the leg)",
        "left_hip",
        "left_knee",
        False,
        0.15,
        ["left_hip", "left_knee"],
        frac=0.35,
        limb=True,
    )

    if "height" in want:
        out["height"] = HeightDef()
    if "inseam" in want:
        out["inseam"] = DistanceDef(
            "inseam",
            "height of the crotch (lowest midline vertex between the legs) above the floor",
            Landmark("v", crotch),
            Landmark("v", int(np.argmin(v[:, 1]))),
            vertical_only=True,
        )
    acro = _acromion(b)
    if "shoulder_width" in want:
        out["shoulder_width"] = DistanceDef(
            "shoulder_width",
            "straight distance between the left and right acromion points "
            "(top of the shoulder, 3.5 cm outside the shoulder joint)",
            Landmark("v", acro),
            Landmark("v", int(model.mirror_map[acro]) if model.mirror_map is not None else acro),
        )
    if "arm_length" in want and have_hands:
        sh = Landmark("v", acro)
        el, wr = Landmark("j", jn["left_elbow"]), Landmark("j", jn["left_wrist"])
        out["arm_length"] = SumDef(
            "arm_length",
            "left arm: acromion to elbow joint to wrist joint (two straight segments)",
            (DistanceDef("a", "", sh, el), DistanceDef("b", "", el, wr)),
        )
    if "neck_back_to_waist" in want:
        out["neck_back_to_waist"] = _back_path(b, out.get("waist_circ"))
    return out


def _crotch_vertex(b: _Builder) -> int:
    """Lowest midline vertex above the knees: the perineum, where the legs join."""
    knee = b.J("left_knee")[1]
    cand = b.mid[(b.v[b.mid, 1] > knee) & (b.v[b.mid, 1] < b.J("pelvis")[1])]
    if len(cand) == 0:
        raise MeasurementDefinitionError("inseam", "no midline vertices between knee and pelvis")
    return int(cand[np.argmin(b.v[cand, 1])])


def _acromion(b: _Builder) -> int:
    sj = b.J("left_shoulder")
    near = np.hypot(b.v[:, 0] - (sj[0] + 0.035), b.v[:, 2] - sj[2]) < 0.02
    sel = np.flatnonzero(near & (b.v[:, 1] > sj[1] - 0.1))
    if len(sel) == 0:
        raise MeasurementDefinitionError("shoulder_width", "no vertices above the shoulder joint")
    return int(sel[np.argmax(b.v[sel, 1])])


def _back_path(b: _Builder, waist: MeasurementDef | None) -> PathDef:
    """Back midline vertices from C7 (just above the neck joint) down to the waist level."""
    y_c7 = float(b.J("neck")[1]) + 0.02
    if isinstance(waist, RingDef):
        y_w = waist.level_m
    else:
        y_w = float(b.J("spine1")[1] + 0.4 * (b.J("spine2")[1] - b.J("spine1")[1]))
    mid = b.mid
    back = mid[(b.v[mid, 2] < b.J("spine2")[2] - 0.03) & (b.v[mid, 1] <= y_c7 + 0.01) & (b.v[mid, 1] >= y_w - 0.01)]
    if len(back) < 3:
        raise MeasurementDefinitionError("neck_back_to_waist", "no back midline vertices between neck and waist")
    order = back[np.argsort(-b.v[back, 1])]
    return PathDef(
        "neck_back_to_waist",
        "length along the back midline surface from C7 (neck joint + 2 cm) down to the waist ring level",
        order.astype(np.int64),
    )

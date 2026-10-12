"""A synthetic, mirror-symmetric body for tests that must run without the SMPL-X files.

``SyntheticBodyModel`` implements the ``avatar_engine.bodymodel.BodyModel`` protocol: a torso ellipsoid, a neck, a head,
T-pose arms and legs as capsules (about 3000 vertices, metres, y up, feet on y = 0, +x = the wearer's left), with the
SMPL-X joint names and one-hot skinning weights. ``build_avatar`` runs the real ``avatar_engine`` volume and point code
on it and writes an avatar folder in exactly the format of ``AvatarEngine.build`` (no fitting: the shape is fixed).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from avatar_engine import __version__ as avatar_version
from avatar_engine.bodymodel import SMPLX_JOINTS, joint_mirror_map, vertex_mirror_map
from avatar_engine.points import PointBuilder, load_specs
from avatar_engine.volumes import VolumeBuilder
from avatar_engine.writer import CONVENTION, points_doc, volumes_doc, write_all

F64 = NDArray[np.float64]
I64 = NDArray[np.int64]

# left side and centre joints (the right side is the mirror image); metres, before the feet are put on y = 0
_JOINTS = {
    "pelvis": (0.0, 0.95, 0.0),
    "spine1": (0.0, 1.06, 0.0),
    "spine2": (0.0, 1.20, 0.0),
    "spine3": (0.0, 1.30, 0.0),
    "neck": (0.0, 1.43, 0.0),
    "head": (0.0, 1.58, 0.0),
    "jaw": (0.0, 1.62, 0.06),
    "left_eye": (0.03, 1.68, 0.08),
    "left_collar": (0.07, 1.39, 0.0),
    "left_shoulder": (0.17, 1.40, 0.0),
    "left_elbow": (0.43, 1.40, 0.0),
    "left_wrist": (0.68, 1.40, 0.0),
    "left_hip": (0.09, 0.88, 0.0),
    "left_knee": (0.10, 0.48, 0.0),
    "left_ankle": (0.10, 0.09, 0.0),
    "left_foot": (0.10, 0.03, 0.13),
}
# primitives: (kind, a, b, radius, joint that owns the vertices). Ellipsoid: centre a, semi-axes b.
_TORSO = ("ellipsoid", (0.0, 1.10, 0.0), (0.18, 0.37, 0.13))
_CAPSULES = [
    ("neck", (0.0, 1.40, 0.0), (0.0, 1.57, 0.0), 0.055),
    ("head", (0.0, 1.62, 0.0), (0.0, 1.70, 0.0), 0.095),
    ("left_shoulder", (0.12, 1.40, 0.0), (0.43, 1.40, 0.0), 0.05),
    ("left_elbow", (0.43, 1.40, 0.0), (0.68, 1.40, 0.0), 0.04),
    ("left_wrist", (0.68, 1.40, 0.0), (0.84, 1.40, 0.0), 0.035),
    ("left_hip", (0.09, 0.84, 0.0), (0.10, 0.48, 0.0), 0.075),
    ("left_knee", (0.10, 0.48, 0.0), (0.10, 0.09, 0.0), 0.05),
    ("left_ankle", (0.10, 0.05, -0.02), (0.10, 0.04, 0.17), 0.04),
]


def _uv_sphere(n_lat: int, n_lon: int) -> tuple[F64, I64]:
    """Unit sphere, poles on +-y, longitudes symmetric about x = 0 (n_lon a multiple of 4)."""
    th = np.linspace(0.0, np.pi, n_lat + 1)[1:-1]
    ph = 2.0 * np.pi * np.arange(n_lon) / n_lon
    t, p = np.meshgrid(th, ph, indexing="ij")
    v = np.column_stack([(np.sin(t) * np.sin(p)).ravel(), np.cos(t).ravel(), (np.sin(t) * np.cos(p)).ravel()])
    v = np.vstack([[0.0, 1.0, 0.0], v, [0.0, -1.0, 0.0]])
    f: list[tuple[int, int, int]] = []
    ring = lambda i, j: 1 + i * n_lon + (j % n_lon)  # noqa: E731
    for j in range(n_lon):
        f.append((0, ring(0, j + 1), ring(0, j)))
        f.append((len(v) - 1, ring(n_lat - 2, j), ring(n_lat - 2, j + 1)))
    for i in range(n_lat - 2):
        for j in range(n_lon):
            a, b, c, d = ring(i, j), ring(i, j + 1), ring(i + 1, j), ring(i + 1, j + 1)
            # the diagonal of the mirrored quad is the mirror of this one: an exactly symmetric mesh
            f += [(a, b, d), (a, d, c)] if j < n_lon // 2 else [(a, b, c), (b, d, c)]
    return v, np.array(f, dtype=np.int64)


def _capsule(a: F64, b: F64, r: float, n: int = 20) -> tuple[F64, I64]:
    """Capsule between ``a`` and ``b``: a unit sphere split at its equator and stretched along the axis."""
    v, f = _uv_sphere(n // 2 + 2, n)
    ax = b - a
    length = float(np.linalg.norm(ax))
    u = ax / length
    ref = np.array([0.0, 0.0, 1.0]) if abs(u[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = ref - (ref @ u) * u
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(u, e1)
    along = v[:, 1] * r + np.where(v[:, 1] >= 0, length, 0.0)
    out = a + np.outer(along, u) + r * (np.outer(v[:, 2], e1) + np.outer(v[:, 0], e2))
    if np.linalg.det(np.column_stack([e2, u, e1])) < 0:  # keep the triangles counter-clockwise from outside
        f = f[:, [0, 2, 1]]
    return out, f


def _inside(p: F64, prim: tuple[str, F64, F64, float]) -> NDArray[np.bool_]:
    kind, a, b, r = prim
    if kind == "ellipsoid":
        return np.asarray((((p - a) / b) ** 2).sum(axis=1) < 1.0 - 1e-6)
    d = b - a
    t = np.clip(((p - a) @ d) / (d @ d), 0.0, 1.0)
    return np.asarray(np.linalg.norm(p - (a + t[:, None] * d), axis=1) < r - 1e-6)


def _mirror(v: F64, f: I64) -> tuple[F64, I64]:
    return v * np.array([-1.0, 1.0, 1.0]), np.ascontiguousarray(f[:, [0, 2, 1]])


class SyntheticBodyModel:
    """Fixed-shape, mirror-symmetric body (``BodyModel`` protocol). ``betas`` are accepted and ignored."""

    model_type = "smplx"  # SMPL-X joint names and order, so that the avatar_engine volume code applies unchanged
    gender = "male"
    source_file = "synthetic (tests/synthetic_avatar.py)"

    def __init__(self) -> None:
        self.joint_names = list(SMPLX_JOINTS)
        jn = {n: i for i, n in enumerate(self.joint_names)}
        j = np.zeros((len(self.joint_names), 3))
        for name, xyz in _JOINTS.items():
            j[jn[name]] = xyz
            if name.startswith("left_"):
                j[jn["right_" + name[5:]]] = (-xyz[0], xyz[1], xyz[2])
        for i, name in enumerate(self.joint_names):  # fingers: along the hand
            if name[5:].startswith(("index", "middle", "pinky", "ring", "thumb")) or name[6:].startswith(
                ("index", "middle", "pinky", "ring", "thumb")
            ):
                side = 1.0 if name.startswith("left_") else -1.0
                k = int(name[-1])
                j[i] = (side * (0.72 + 0.04 * k), 1.40, 0.0)
        prims: list[tuple[str, F64, F64, float, str]] = []
        prims.append(("ellipsoid", np.array(_TORSO[1]), np.array(_TORSO[2]), 0.0, "torso"))
        for owner, a, b, r in _CAPSULES:
            prims.append(("capsule", np.array(a), np.array(b), r, owner))
            if owner.startswith("left_"):
                m = np.array([-1.0, 1.0, 1.0])
                prims.append(("capsule", np.array(a) * m, np.array(b) * m, r, "right_" + owner[5:]))
        verts: list[F64] = []
        faces: list[I64] = []
        owners: list[NDArray[np.int64]] = []
        off = 0
        for k, (kind, a, b, r, owner) in enumerate(prims):
            if kind == "ellipsoid":
                v, f = _uv_sphere(36, 48)
                v = a + v * b
            elif owner.startswith("right_"):
                left = next(p for p in prims if p[4] == "left_" + owner[6:])
                v, f = _mirror(*_capsule(left[1], left[2], left[3]))
            else:
                v, f = _capsule(a, b, r)
            # drop the triangles buried inside another primitive: no hidden surfaces for distance queries
            cen = v[f].mean(axis=1)
            buried = np.zeros(len(f), dtype=bool)
            for kk, other in enumerate(prims):
                if kk != k:
                    buried |= _inside(cen, (other[0], other[1], other[2], other[3]))
            f = f[~buried]
            if kind == "ellipsoid":
                y = v[:, 1]
                part = np.where(
                    y < 1.0,
                    jn["pelvis"],
                    np.where(y < 1.15, jn["spine1"], np.where(y < 1.30, jn["spine2"], jn["spine3"])),
                )
            else:
                part = np.full(len(v), jn[owner])
            verts.append(v)
            faces.append(f + off)
            owners.append(part)
            off += len(v)
        v_all = np.vstack(verts)
        f_all = np.vstack(faces)
        own = np.concatenate(owners)
        used = np.unique(f_all)  # vertices of dropped triangles only: removed, so that every vertex is on the surface
        remap = np.full(len(v_all), -1, dtype=np.int64)
        remap[used] = np.arange(len(used))
        v_all, f_all, own = v_all[used], remap[f_all], own[used]
        # exact mirror symmetry (the right limbs are mirrored copies; the central parts are symmetric by construction)
        dy = -float(v_all[:, 1].min())
        self._v = v_all + np.array([0.0, dy, 0.0])
        self._j = j + np.array([0.0, dy, 0.0])
        self.faces = f_all
        self.skinning_weights = np.zeros((len(v_all), len(self.joint_names)))
        self.skinning_weights[np.arange(len(v_all)), own] = 1.0
        self.num_betas = 1
        self.mirror_map = vertex_mirror_map(self._v)
        self.joint_mirror = joint_mirror_map(self.joint_names)

    def rest(self, betas: F64) -> tuple[F64, F64]:
        """(vertices, joints), feet on y = 0."""
        return self._v.copy(), self._j.copy()

    def vertices(self, betas: F64) -> F64:
        """Vertices (N,3)."""
        return self._v.copy()

    def joints(self, betas: F64) -> F64:
        """Joints (K,3)."""
        return self._j.copy()


def _loop_length(v: F64, f: I64, y: float) -> float:
    """Perimeter of the convex hull of the horizontal section of the torso at height ``y`` (central part only)."""
    from scipy.spatial import ConvexHull

    tri = v[f]
    s = tri[:, :, 1] - y
    cross = (s * np.roll(s, -1, axis=1)) < 0
    pts = []
    for t_, c_ in zip(tri[cross.any(axis=1)], cross[cross.any(axis=1)], strict=True):
        for k in np.flatnonzero(c_):
            a, b = t_[k], t_[(k + 1) % 3]
            pts.append(a + (y - a[1]) / (b[1] - a[1]) * (b - a))
    p = np.array(pts)
    p = p[np.abs(p[:, 0]) < 0.25][:, [0, 2]]
    h = ConvexHull(p)
    return float(h.area)


def build_avatar(out_dir: Path, name: str = "synthetic_male") -> Path:
    """Write an avatar folder (bundle, ``avatar.obj``, volumes, points, report) for the synthetic body."""
    model = SyntheticBodyModel()
    v, j = model.rest(np.zeros(1))
    volumes = VolumeBuilder(model).build(v, j)  # type: ignore[arg-type]
    points = PointBuilder(load_specs()).build(volumes, v, model.faces)
    chest = next(b for b in volumes if b.name == "Chest").point_at(0.3)[1]
    achieved = {
        "chest_circ": round(_loop_length(v, model.faces, float(chest)) * 100, 4),
        "height": round(float(v[:, 1].max()) * 100, 4),
    }
    bundle = {
        "schema_version": 1,
        "units": "m",
        "convention": CONVENTION,
        "name": name,
        "model_type": model.model_type,
        "model_file": model.source_file,
        "gender": model.gender,
        "betas": [0.0],
        "num_betas_requested": 1,
        "num_betas_used": 1,
        "target_cm": dict(achieved),
        "achieved_cm": achieved,
        "mesh": "avatar.obj",
        "files": {
            "bounding_volumes": "bounding_volumes.json",
            "arrangement_points": "arrangement_points.json",
            "measurements_report": "measurements_report.json",
        },
        "bbox_m": {
            "min": [round(float(x), 6) for x in v.min(axis=0)],
            "max": [round(float(x), 6) for x in v.max(axis=0)],
        },
        "engine_version": avatar_version,
    }
    report = {"schema_version": 1, "units": "cm", "synthetic": True, "measurements": {}, "warnings": []}
    header = [f"avatar: {name}", f"units: metres (1 OBJ unit = 1 m); {CONVENTION}", f"model: {model.source_file}"]
    write_all(Path(out_dir), bundle, volumes_doc(volumes), points_doc(points), report, v, model.faces, header)
    return Path(out_dir)

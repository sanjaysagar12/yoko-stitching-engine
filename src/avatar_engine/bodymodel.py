"""Body model protocol and the numpy implementation shared by SMPL and SMPL-X (zero pose only).

Convention of the returned geometry (metres): y up, feet on y = 0, +z = front of the body, **+x = the wearer's left**.
Both SMPL and SMPL-X already use these axes, so no axis conversion is needed; the only change is a vertical
translation that puts the lowest vertex on y = 0.

At zero pose linear blend skinning is the identity and the pose blend shapes vanish, so
``v = v_template + sum_k beta_k * shapedirs[:, :, k]`` and ``J = J_regressor @ v``.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from .errors import ModelFormatError

F64 = NDArray[np.float64]
I64 = NDArray[np.int64]

_BODY22 = (
    "pelvis left_hip right_hip spine1 left_knee right_knee spine2 left_ankle right_ankle spine3 left_foot right_foot "
    "neck left_collar right_collar head left_shoulder right_shoulder left_elbow right_elbow left_wrist right_wrist"
).split()
_FINGERS = [f"{f}{i}" for f in ("index", "middle", "pinky", "ring", "thumb") for i in (1, 2, 3)]
SMPLX_JOINTS = (
    _BODY22 + ["jaw", "left_eye", "right_eye"] + [f"left_{n}" for n in _FINGERS] + [f"right_{n}" for n in _FINGERS]
)
SMPL_JOINTS = _BODY22 + ["left_hand", "right_hand"]
_MAX_SHAPE = {"smpl": 300, "smplx": 300}  # SMPL-X files carry 300 shape + 100 expression components


class BodyModel(Protocol):
    """A parametric body in rest pose. All coordinates metres, see module docstring for the axes."""

    model_type: str
    gender: str
    source_file: str
    faces: I64
    skinning_weights: F64
    joint_names: list[str]
    num_betas: int  # number of shape components available
    mirror_map: I64 | None  # vertex i <-> mirror_map[i] across x = 0 (None if the model is not symmetric)
    joint_mirror: I64  # joint i <-> joint_mirror[i]

    def rest(self, betas: F64) -> tuple[F64, F64]:
        """(vertices (N,3), joints (K,3)) for ``betas`` (any length <= num_betas), feet on y = 0."""
        ...

    def vertices(self, betas: F64) -> F64:
        """Vertices (N,3), metres."""
        ...

    def joints(self, betas: F64) -> F64:
        """Joints (K,3), metres."""
        ...


def joint_mirror_map(names: list[str]) -> I64:
    """Index of the mirrored joint (left_x <-> right_x); centre joints map to themselves."""
    idx = {n: i for i, n in enumerate(names)}
    out = np.arange(len(names), dtype=np.int64)
    for n, i in idx.items():
        if n.startswith("left_") and "right_" + n[5:] in idx:
            out[i] = idx["right_" + n[5:]]
        elif n.startswith("right_") and "left_" + n[6:] in idx:
            out[i] = idx["left_" + n[6:]]
    return out


def vertex_mirror_map(v: F64) -> I64 | None:
    """Vertex permutation of the mirror x -> -x, or None if the template is not mirror symmetric."""
    _, j = cKDTree(v).query(v * np.array([-1.0, 1.0, 1.0]))
    j = j.astype(np.int64)
    if len(set(j.tolist())) != len(v) or not np.array_equal(j[j], np.arange(len(v))):
        return None
    return np.asarray(j, dtype=np.int64)


class SmplLikeModel:
    """SMPL or SMPL-X from numpy arrays. ``symmetrize`` removes the left/right asymmetric shape components."""

    def __init__(
        self,
        model_type: str,
        gender: str,
        source_file: str,
        v_template: F64,
        shapedirs: F64,
        j_regressor: F64,
        weights: F64,
        faces: I64,
        joint_names: list[str],
        symmetrize: bool = True,
    ) -> None:
        n = len(v_template)
        if shapedirs.shape[:2] != (n, 3) or j_regressor.shape[1] != n or weights.shape[0] != n:
            raise ModelFormatError(f"{source_file}: inconsistent array shapes (N={n})")
        if j_regressor.shape[0] != len(joint_names) or weights.shape[1] != len(joint_names):
            raise ModelFormatError(
                f"{source_file}: {j_regressor.shape[0]} joints in J_regressor, "
                f"expected {len(joint_names)} for {model_type}"
            )
        self.model_type, self.gender, self.source_file = model_type, gender, source_file
        self.faces = np.asarray(faces, dtype=np.int64)
        self.skinning_weights = np.asarray(weights, dtype=np.float64)
        self.joint_names = joint_names
        self.joint_mirror = joint_mirror_map(joint_names)
        self._jreg = np.asarray(j_regressor, dtype=np.float64)
        self.num_betas = min(shapedirs.shape[2], _MAX_SHAPE[model_type])
        v0 = np.asarray(v_template, dtype=np.float64)
        s = np.ascontiguousarray(np.asarray(shapedirs[:, :, : self.num_betas], dtype=np.float64).transpose(2, 0, 1))
        self.mirror_map = vertex_mirror_map(v0) if symmetrize else None
        self.symmetrized = self.mirror_map is not None
        if self.mirror_map is not None:
            flip = np.array([-1.0, 1.0, 1.0])
            v0 = 0.5 * (v0 + v0[self.mirror_map] * flip)
            s = 0.5 * (s + s[:, self.mirror_map, :] * flip)
        self._v0 = v0
        self._s = s.reshape(self.num_betas, -1)  # (C, 3N)

    def rest(self, betas: F64) -> tuple[F64, F64]:
        """See ``BodyModel.rest``."""
        b = np.asarray(betas, dtype=np.float64)
        if len(b) > self.num_betas:
            raise ValueError(f"{len(b)} betas given, model has {self.num_betas}")
        v = self._v0 + (b @ self._s[: len(b)]).reshape(-1, 3) if len(b) else self._v0.copy()
        j = self._jreg @ v
        if self.mirror_map is not None:  # joint regressors are not symmetric: symmetrise the joints
            flip = np.array([-1.0, 1.0, 1.0])
            j = 0.5 * (j + j[self.joint_mirror] * flip)
        dy = -float(v[:, 1].min())
        shift = np.array([0.0, dy, 0.0])
        return v + shift, j + shift

    def vertices(self, betas: F64) -> F64:
        """See ``BodyModel.vertices``."""
        return self.rest(betas)[0]

    def joints(self, betas: F64) -> F64:
        """See ``BodyModel.joints``."""
        return self.rest(betas)[1]

    def joint_index(self, name: str) -> int:
        """Index of joint ``name``."""
        return self.joint_names.index(name)

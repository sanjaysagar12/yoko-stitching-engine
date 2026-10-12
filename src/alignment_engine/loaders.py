"""Read the avatar folder and the piece folder (validated against the contracts). Metres."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from . import contracts
from .errors import BundleError

F64 = NDArray[np.float64]
I64 = NDArray[np.int64]


@dataclass(frozen=True)
class PointRecord:
    """An arrangement point as written by the avatar engine."""

    name: str
    bv: str
    offset_m: float
    wrap_direction: str
    position: F64
    normal: F64
    up: F64
    tangent: F64


@dataclass(frozen=True)
class VolumeRecord:
    """A bounding volume axis (start -> end) and radius, metres."""

    name: str
    start: F64
    end: F64
    radius_m: float


@dataclass
class Avatar:
    """Everything the alignment needs from an avatar folder."""

    bundle: dict[str, Any]
    points: dict[str, PointRecord]
    volumes: dict[str, VolumeRecord]
    vertices: F64
    faces: I64
    cache: dict[str, Any] = field(default_factory=dict)  # derived data (torso sections), filled lazily


@dataclass
class PieceInfo:
    """A piece bundle plus its mesh (loaded lazily)."""

    bundle: dict[str, Any]
    obj_path: Path
    _mesh: tuple[F64, I64] | None = field(default=None, repr=False)

    @property
    def name(self) -> str:
        """Piece name."""
        return str(self.bundle["name"])

    @property
    def stem(self) -> str:
        """File stem used for output names (the OBJ name without ``_m.obj``)."""
        return self.obj_path.name.removesuffix("_m.obj").removesuffix(".obj")

    def mesh(self) -> tuple[F64, I64]:
        """(vertices (N,3), faces (M,3)) in metres, z = 0."""
        if self._mesh is None:
            self._mesh = read_obj(self.obj_path)
        return self._mesh

    def reference(self, name: str) -> F64:
        """Reference point (3,) in piece coordinates: ``bbox_centre`` or a named point."""
        if name in ("bbox_centre", "default_reference"):
            x, y = self.bundle["default_reference"]["xy_m"]
            return np.array([x, y, 0.0])
        for q in self.bundle["named_points"]:
            if q["name"] == name:
                return np.array([q["xy_m"][0], q["xy_m"][1], 0.0])
        from .errors import UnknownNamedPointError

        raise UnknownNamedPointError(self.name, name, [q["name"] for q in self.bundle["named_points"]])


def read_obj(path: Path) -> tuple[F64, I64]:
    """Minimal OBJ reader (``v`` and triangular ``f`` lines)."""
    if not path.is_file():
        raise BundleError(f"mesh file not found: {path}")
    v: list[list[float]] = []
    f: list[list[int]] = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        if ln.startswith("v "):
            v.append([float(x) for x in ln.split()[1:4]])
        elif ln.startswith("f "):
            f.append([int(x.split("/")[0]) - 1 for x in ln.split()[1:4]])
    return np.array(v, dtype=np.float64).reshape(-1, 3), np.array(f, dtype=np.int64).reshape(-1, 3)


def read_avatar(folder: Path) -> Avatar:
    """Load ``avatar_bundle.json`` and the files it names."""
    folder = Path(folder)
    bpath = folder / "avatar_bundle.json"
    if not bpath.is_file():
        raise BundleError(f"avatar_bundle.json not found in {folder}")
    bundle = contracts.read_json(bpath, "avatar_bundle")
    pts = contracts.read_json(folder / bundle["files"]["arrangement_points"], "arrangement_points")
    vols = contracts.read_json(folder / bundle["files"]["bounding_volumes"], "bounding_volumes")
    verts, faces = read_obj(folder / bundle["mesh"])
    points = {
        p["name"]: PointRecord(
            p["name"],
            p["bv"],
            float(p["offset_m"]),
            p["wrap_direction"],
            np.array(p["position_m"]),
            np.array(p["normal"]),
            np.array(p["up"]),
            np.array(p["tangent"]),
        )
        for p in pts["points"]
    }
    volumes = {
        v["name"]: VolumeRecord(v["name"], np.array(v["axis_start"]), np.array(v["axis_end"]), float(v["radius_m"]))
        for v in vols["volumes"]
    }
    return Avatar(bundle, points, volumes, verts, faces)


def read_pieces(folder: Path) -> dict[str, PieceInfo]:
    """Load every ``*.piece.json`` of ``folder`` (validated)."""
    folder = Path(folder)
    out: dict[str, PieceInfo] = {}
    for p in sorted(folder.glob("*.piece.json")):
        b = contracts.read_json(p, "piece_bundle")
        out[b["name"]] = PieceInfo(b, folder / b["obj"])
    if not out:
        raise BundleError(f"no *.piece.json files in {folder}")
    return out

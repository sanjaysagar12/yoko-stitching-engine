"""Avatar output files. Floats are written with fixed precision so two builds are byte-identical."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from . import contracts
from .points import ArrangementPoint
from .volumes import BoundingVolume

F64 = NDArray[np.float64]
CONVENTION = "metres; y up, feet on y=0; +z = front of the body; +x = the wearer's left; zero pose (T-pose)"


def _r(x: Any, nd: int = 6) -> Any:
    if isinstance(x, np.ndarray):
        return [round(float(v), nd) for v in x]
    return round(float(x), nd)


def write_obj(path: Path, vertices: F64, faces: NDArray[np.int64], header: list[str]) -> Path:
    """Wavefront OBJ, vertices in metres with 6 decimals (1 micrometre)."""
    lines = [f"# {h}" for h in header]
    lines += [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in vertices]
    lines += [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in faces]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return path


def volumes_doc(volumes: list[BoundingVolume]) -> dict[str, Any]:
    """``bounding_volumes`` document."""
    return {
        "schema_version": 1,
        "units": "m",
        "convention": CONVENTION,
        "volumes": [
            {
                "name": b.name,
                "kind": b.kind,
                "joint": b.joint,
                "child_joint": b.child_joint,
                "axis_start": _r(b.start),
                "axis_end": _r(b.end),
                "centre": _r(0.5 * (b.start + b.end)),
                "radius_m": _r(b.radius_m),
                "length_m": _r(b.length_m),
                "num_vertices": int(len(b.vertex_ids)),
            }
            for b in volumes
        ],
    }


def points_doc(points: list[ArrangementPoint]) -> dict[str, Any]:
    """``arrangement_points`` document."""
    return {
        "schema_version": 1,
        "units": "m",
        "convention": CONVENTION,
        "points": [
            {
                "name": p.spec.name,
                "bv": p.spec.bv,
                "angle_deg": _r(p.spec.angle_deg, 4),
                "t": _r(p.spec.t, 4),
                "offset_m": _r(p.spec.offset_m),
                "wrap_direction": p.spec.wrap_direction,
                "position_m": _r(p.position_m),
                "normal": _r(p.normal, 6),
                "up": _r(p.up, 6),
                "tangent": _r(p.tangent, 6),
            }
            for p in points
        ],
    }


def write_report(path: Path, doc: dict[str, Any]) -> Path:
    """``measurements_report.json`` (not part of the exchanged contracts)."""
    path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    return path


def write_all(
    out_dir: Path,
    bundle: dict[str, Any],
    vol_doc: dict[str, Any],
    pts_doc: dict[str, Any],
    report: dict[str, Any],
    vertices: F64,
    faces: NDArray[np.int64],
    header: list[str],
) -> dict[str, Path]:
    """Write the five avatar files into ``out_dir`` (created); every contract file is validated first."""
    out_dir.mkdir(parents=True, exist_ok=True)
    files = bundle["files"]
    return {
        "mesh": write_obj(out_dir / bundle["mesh"], vertices, faces, header),
        "bounding_volumes": contracts.write_json(out_dir / files["bounding_volumes"], vol_doc, "bounding_volumes"),
        "arrangement_points": contracts.write_json(
            out_dir / files["arrangement_points"], pts_doc, "arrangement_points"
        ),
        "measurements_report": write_report(out_dir / files["measurements_report"], report),
        "bundle": contracts.write_json(out_dir / "avatar_bundle.json", bundle, "avatar_bundle"),
    }

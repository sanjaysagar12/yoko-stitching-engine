"""Alignment output files: result JSON, one placed OBJ per piece, one combined scene OBJ. Metres, fixed precision."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from . import contracts
from .loaders import Avatar

if TYPE_CHECKING:
    from .engine import PlacedPiece

F64 = NDArray[np.float64]


def _obj_lines(v: F64, f: NDArray[np.int64], offset: int = 0) -> tuple[list[str], list[str]]:
    return (
        [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in v],
        [f"f {a + 1 + offset} {b + 1 + offset} {c + 1 + offset}" for a, b, c in f],
    )


def write_outputs(out_dir: Path, doc: dict[str, Any], placed: list[PlacedPiece], avatar: Avatar) -> dict[str, Path]:
    """Write ``alignment_result.json``, ``<piece>_placed_m.obj`` per placement and ``scene_m.obj``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    header = ["units: metres (1 OBJ unit = 1 m); y up, feet on y=0, +z front, +x the wearer's left"]
    scene_v: list[str] = []
    scene_f: list[str] = []
    off = 0
    groups = [("avatar", avatar.vertices, avatar.faces)] + [(p.name, p.vertices, p.faces) for p in placed]
    for name, v, f in groups:
        vl, fl = _obj_lines(v, f, off)
        scene_v += vl
        scene_f += [f"g {name}", *fl]
        off += len(v)
    for p in placed:
        vl, fl = _obj_lines(p.vertices, p.faces)
        lines = [f"# piece {p.name} placed on {p.point}", *[f"# {h}" for h in header], *vl, *fl]
        path = out_dir / f"{p.stem}_placed_m.obj"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
        paths[f"placed:{p.name}"] = path
    scene = out_dir / "scene_m.obj"
    scene.write_text(
        "\n".join([*[f"# {h}" for h in header], *scene_v, *scene_f]) + "\n", encoding="utf-8", newline="\n"
    )
    paths["scene"] = scene
    if "arrangement" in doc:  # files are byte-identical for identical inputs: the run time is only in the API answer
        doc = {**doc, "arrangement": {**doc["arrangement"], "timing_s": 0.0}}
    paths["result"] = contracts.write_json(out_dir / "alignment_result.json", doc, "alignment_result")
    return paths

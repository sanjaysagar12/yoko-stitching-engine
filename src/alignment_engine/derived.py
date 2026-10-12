"""Derived pieces (linked mirrored clones): build them from a placement spec and cache them as files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from . import contracts
from .errors import DerivedOfDerivedError, DuplicatePieceError, UnknownPieceError
from .loaders import PieceInfo
from .mirror import MirroredPiece, mirror_clone

F64 = NDArray[np.float64]


@dataclass
class Derived:
    """A built derived piece: its definition from the spec, the mirrored piece and a loader-compatible PieceInfo."""

    spec: dict[str, Any]
    mirrored: MirroredPiece
    info: PieceInfo

    @property
    def id(self) -> str:
        """Piece id."""
        return str(self.spec["id"])

    def suggested_stitch(self) -> dict[str, Any] | None:
        """For an ``edge`` axis: the seam between that edge on the source and on the clone (equal lengths)."""
        axis = self.spec["axis"]
        if axis["type"] != "edge":
            return None
        return {
            "a": {"piece": self.spec["source"], "from_point": axis["from_point"], "to_point": axis["to_point"]},
            "b": {"piece": self.id, "from_point": axis["from_point"], "to_point": axis["to_point"]},
        }


def build_derived(defs: list[dict[str, Any]], pieces: dict[str, PieceInfo], out_dir: Path) -> dict[str, Derived]:
    """Build every derived piece of ``defs`` (derived-of-derived and duplicate ids are errors).

    ``pieces`` is the dict of base pieces; ``out_dir`` is where the files will be (only used for the file names).
    """
    built: dict[str, Derived] = {}
    for d in defs:
        pid, src = d["id"], d["source"]
        if pid in pieces or pid in built:
            raise DuplicatePieceError(pid)
        if src in built:
            raise DerivedOfDerivedError(pid, src)
        if src not in pieces:
            raise UnknownPieceError(src, list(pieces))
        base = pieces[src]
        v, f = base.mesh()
        m = mirror_clone(base.bundle, v, f, d["axis"], pid)
        info = PieceInfo(m.bundle, Path(out_dir) / m.bundle["obj"], (m.vertices, m.faces))
        built[pid] = Derived(d, m, info)
    return built


def write_derived(out_dir: Path, derived: dict[str, Derived]) -> list[Path]:
    """Write ``<id>_m.obj`` and ``<id>.piece.json`` for each derived piece (validated, deterministic)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for d in derived.values():
        b = d.mirrored.bundle
        lines = [
            f"# piece: {b['name']} (mirror of {b['mirror_of']})",
            "# units: metres (1 OBJ unit = 1 m); z = 0; x right, y up",
            "# origin: piece bounding-box bottom-left",
        ]
        lines += [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in d.mirrored.vertices]
        lines += [f"f {a + 1} {c + 1} {e + 1}" for a, c, e in d.mirrored.faces]
        obj = out_dir / b["obj"]
        obj.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
        paths.append(obj)
        doc = _rounded(b)
        paths.append(contracts.write_json(out_dir / f"{b['name']}.piece.json", doc, "piece_bundle"))
    return paths


def _rounded(x: Any) -> Any:
    if isinstance(x, float):
        return round(x, 12)
    if isinstance(x, dict):
        return {k: _rounded(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_rounded(v) for v in x]
    return x


def derived_doc(d: Derived) -> dict[str, Any]:
    """The ``derived_pieces`` entry of the alignment result."""
    b = d.mirrored.bundle
    w, h = b["bbox_m"]["max"]
    stitch = d.suggested_stitch()
    return {
        "id": d.id,
        "source": d.spec["source"],
        "axis": b["axis"] if d.spec["axis"]["type"] != "edge" else d.spec["axis"],
        "origin_shift_m": [round(x, 9) for x in b["origin_shift_m"]],
        "obj": b["obj"],
        "bundle": f"{b['name']}.piece.json",
        "bbox_cm": {"width": round(w * 100.0, 4), "height": round(h * 100.0, 4)},
        "suggested_stitch": stitch,
    }

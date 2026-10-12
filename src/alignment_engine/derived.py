"""Derived pieces (linked mirrored clones and plain copies): build them from a placement spec, cache them as files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from . import contracts
from .errors import AxisError, DerivedCycleError, DerivedOfDerivedError, DuplicatePieceError, UnknownPieceError
from .loaders import PieceInfo
from .mirror import MirroredPiece, copy_clone, mirror_clone

F64 = NDArray[np.float64]


@dataclass
class Derived:
    """A built derived piece: its definition from the spec, the new piece (mirrored or copied) and a loader-compatible
    PieceInfo."""

    spec: dict[str, Any]
    mirrored: MirroredPiece
    info: PieceInfo

    @property
    def id(self) -> str:
        """Piece id."""
        return str(self.spec["id"])

    @property
    def op(self) -> str:
        """``mirror`` (the default, also for old specs without ``op``) or ``copy``."""
        return str(self.spec.get("op", "mirror"))

    def suggested_stitch(self) -> dict[str, Any] | None:
        """For an ``edge`` axis: the seam between that edge on the source and on the clone (equal lengths)."""
        axis = self.spec.get("axis")
        if axis is None or axis["type"] != "edge":
            return None
        return {
            "a": {"piece": self.spec["source"], "from_point": axis["from_point"], "to_point": axis["to_point"]},
            "b": {"piece": self.id, "from_point": axis["from_point"], "to_point": axis["to_point"]},
        }


def build_derived(defs: list[dict[str, Any]], pieces: dict[str, PieceInfo], out_dir: Path) -> dict[str, Derived]:
    """Build every derived piece of ``defs`` (any order; duplicate ids, unknown sources and cycles are errors).

    ``op: "mirror"`` (default) needs an ``axis`` and a base piece (mirroring a derived piece is an error).
    ``op: "copy"`` has no ``axis`` and may copy a base piece or a derived piece: a copy of a copy is a copy of the
    root (``clone_of`` names it), a copy of a mirror clone copies the mirrored geometry (``clone_of`` = the clone).
    ``pieces`` is the dict of base pieces; ``out_dir`` is where the files will be (only used for the file names).
    """
    by_id: dict[str, dict[str, Any]] = {}
    for d in defs:
        if d["id"] in pieces or d["id"] in by_id:
            raise DuplicatePieceError(d["id"])
        by_id[d["id"]] = d
    built: dict[str, Derived] = {}

    def build(pid: str) -> Derived:
        if pid in built:
            return built[pid]
        d = by_id[pid]
        src, op = d["source"], d.get("op", "mirror")
        if src not in pieces and src not in by_id:
            raise UnknownPieceError(src, [*pieces, *by_id])
        if op == "copy":
            if d.get("axis") is not None:
                raise AxisError(f"derived piece {pid!r}: a copy has no mirror axis (remove 'axis' or use op 'mirror')")
            root, seen = src, [pid]  # follow copies of copies to the root: a base piece or a mirror clone
            while root in by_id and by_id[root].get("op", "mirror") == "copy":
                if root in seen:
                    raise DerivedCycleError([*seen, root])
                seen.append(root)
                root = by_id[root]["source"]
                if root not in pieces and root not in by_id:
                    raise UnknownPieceError(root, [*pieces, *by_id])
            base = pieces[root] if root in pieces else build(root).info
            v, f = base.mesh()
            m = copy_clone(base.bundle, v, f, pid)
        else:
            if d.get("axis") is None:
                raise AxisError(f"derived piece {pid!r}: a mirror clone needs an 'axis'")
            if src in by_id:
                raise DerivedOfDerivedError(pid, src)
            base = pieces[src]
            v, f = base.mesh()
            m = mirror_clone(base.bundle, v, f, d["axis"], pid)
        info = PieceInfo(m.bundle, Path(out_dir) / m.bundle["obj"], (m.vertices, m.faces))
        built[pid] = Derived(d, m, info)
        return built[pid]

    for pid in by_id:
        build(pid)
    return {pid: built[pid] for pid in by_id}


def write_derived(out_dir: Path, derived: dict[str, Derived]) -> list[Path]:
    """Write ``<id>_m.obj`` and ``<id>.piece.json`` for each derived piece (validated, deterministic)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for d in derived.values():
        b = d.mirrored.bundle
        lines = [
            f"# piece: {b['name']} (copy of {b['clone_of']})"
            if d.op == "copy"
            else f"# piece: {b['name']} (mirror of {b['mirror_of']})",
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
    doc: dict[str, Any] = {"id": d.id, "source": d.spec["source"], "op": d.op}
    if d.op == "copy":
        return {
            **doc,
            "clone_of": b["clone_of"],
            "origin_shift_m": [0.0, 0.0],
            "obj": b["obj"],
            "bundle": f"{b['name']}.piece.json",
            "bbox_cm": {"width": round(w * 100.0, 4), "height": round(h * 100.0, 4)},
            "suggested_stitch": None,
        }
    stitch = d.suggested_stitch()
    return {
        **doc,
        "axis": b["axis"] if d.spec["axis"]["type"] != "edge" else d.spec["axis"],
        "origin_shift_m": [round(x, 9) for x in b["origin_shift_m"]],
        "obj": b["obj"],
        "bundle": f"{b['name']}.piece.json",
        "bbox_cm": {"width": round(w * 100.0, 4), "height": round(h * 100.0, 4)},
        "suggested_stitch": stitch,
    }

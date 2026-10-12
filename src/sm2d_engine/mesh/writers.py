"""Mesh writers. All coordinates are written in metres."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

from ..contracts import write_json
from ..errors import UnsupportedFormatError
from .mesh import Mesh


class MeshWriter(Protocol):
    """Output sink. Implementations decide the file format; paths are returned for the report."""

    def write_piece(self, out_dir: Path, slug: str, mesh: Mesh, header: list[str]) -> Path:
        """Write one piece (metres); return the file path."""
        ...

    def write_piece_bundle(self, out_dir: Path, slug: str, bundle: dict[str, Any]) -> Path:
        """Write ``<slug>.piece.json`` (metres, validated against the piece_bundle schema); return the path."""
        ...

    def write_combined(self, out_dir: Path, stem: str, meshes: list[tuple[str, Mesh]], header: list[str]) -> Path:
        """Write all pieces in one file, one group per piece; return the file path."""
        ...

    def write_report(self, out_dir: Path, report: dict[str, Any]) -> Path:
        """Write the JSON report; return the file path."""
        ...


def _write_report(out_dir: Path, report: dict[str, Any]) -> Path:
    path = out_dir / "report.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    return path


class ObjWriter:
    """Wavefront OBJ, ``%.6f`` metres, LF newlines, deterministic."""

    @staticmethod
    def _verts(mesh: Mesh) -> list[str]:
        return [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in mesh.vertices_m]

    def write_piece(self, out_dir: Path, slug: str, mesh: Mesh, header: list[str]) -> Path:
        """``<slug>_m.obj``."""
        out_dir.mkdir(parents=True, exist_ok=True)
        lines = [f"# {h}" for h in header] + self._verts(mesh)
        lines += [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in mesh.faces]
        path = out_dir / f"{slug}_m.obj"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
        return path

    def write_piece_bundle(self, out_dir: Path, slug: str, bundle: dict[str, Any]) -> Path:
        """``<slug>.piece.json``."""
        return write_json(out_dir / f"{slug}.piece.json", bundle, "piece_bundle")

    def write_combined(self, out_dir: Path, stem: str, meshes: list[tuple[str, Mesh]], header: list[str]) -> Path:
        """``<stem>_all_pieces_layout_m.obj`` with a ``g <name>`` group per piece."""
        out_dir.mkdir(parents=True, exist_ok=True)
        lines = [f"# {h}" for h in header]
        for _, m in meshes:
            lines += self._verts(m)
        off = 0
        for name, m in meshes:
            lines.append(f"g {name}")
            lines += [f"f {a + 1 + off} {b + 1 + off} {c + 1 + off}" for a, b, c in m.faces]
            off += len(m.vertices_m)
        path = out_dir / f"{stem}_all_pieces_layout_m.obj"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
        return path

    def write_report(self, out_dir: Path, report: dict[str, Any]) -> Path:
        """``report.json``."""
        out_dir.mkdir(parents=True, exist_ok=True)
        return _write_report(out_dir, report)


class PlyWriter:
    """Stub: PLY output is not implemented."""

    def _no(self) -> UnsupportedFormatError:
        return UnsupportedFormatError("PLY output is not implemented")

    def write_piece(self, out_dir: Path, slug: str, mesh: Mesh, header: list[str]) -> Path:
        """Not implemented."""
        raise self._no()

    def write_piece_bundle(self, out_dir: Path, slug: str, bundle: dict[str, Any]) -> Path:
        """Not implemented."""
        raise self._no()

    def write_combined(self, out_dir: Path, stem: str, meshes: list[tuple[str, Mesh]], header: list[str]) -> Path:
        """Not implemented."""
        raise self._no()

    def write_report(self, out_dir: Path, report: dict[str, Any]) -> Path:
        """Not implemented."""
        raise self._no()


def writer_for(fmt: str) -> MeshWriter:
    """Writer for an output format name."""
    if fmt == "obj":
        return ObjWriter()
    if fmt == "ply":
        return PlyWriter()
    raise UnsupportedFormatError(f"unknown output format {fmt!r}")

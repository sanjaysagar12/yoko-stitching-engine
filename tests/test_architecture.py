"""Dependency rules, enforced by parsing imports (no import-linter needed)."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
ENGINES = ["sm2d_engine", "avatar_engine", "alignment_engine"]


def imports(py: Path) -> list[tuple[str, int]]:
    """(absolute module name, line) of every non-relative import in ``py``."""
    out: list[tuple[str, int]] = []
    for node in ast.walk(ast.parse(py.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            out += [(a.name, node.lineno) for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            out.append((node.module, node.lineno))
    return out


@pytest.mark.parametrize("engine", ENGINES)
def test_engines_import_nothing_from_each_other_or_web(engine: str) -> None:
    forbidden = {e for e in ENGINES if e != engine} | {"web"}
    bad = [
        f"{py.relative_to(SRC)}:{line} imports {mod}"
        for py in (SRC / engine).rglob("*.py")
        for mod, line in imports(py)
        if mod.split(".")[0] in forbidden
    ]
    assert not bad, "\n".join(bad)


def test_backend_uses_only_engine_facades() -> None:
    backend = SRC / "web" / "backend"
    if not backend.is_dir():
        pytest.skip("web backend not written yet")
    bad = [
        f"{py.relative_to(SRC)}:{line} imports {mod}"
        for py in backend.rglob("*.py")
        for mod, line in imports(py)
        if mod.split(".")[0] in ENGINES and "." in mod
    ]
    assert not bad, "only `from <engine> import <Facade>` is allowed:\n" + "\n".join(bad)


def test_geometry_packages_do_not_import_web_frameworks() -> None:
    bad = [
        f"{py.relative_to(SRC)}:{line} imports {mod}"
        for e in ENGINES
        for py in (SRC / e).rglob("*.py")
        for mod, line in imports(py)
        if mod.split(".")[0] in {"fastapi", "uvicorn", "starlette"}
    ]
    assert not bad, "\n".join(bad)

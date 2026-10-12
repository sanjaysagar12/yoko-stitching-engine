from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

from sm2d_engine.engine import Engine, EngineOptions, EngineRequest
from sm2d_engine.eval.scope import VariableScope
from sm2d_engine.io.sm2d_reader import parse_sm2d
from sm2d_engine.io.smis_reader import parse_smis
from sm2d_engine.tools import default_registry

ROOT = Path(__file__).resolve().parents[1]
SM2D = ROOT / "assert" / "sm2d"
SMIS = ROOT / "assert" / "smis"
SKIRT = ("Aldrich-Womens-6th-Ed-Basic-Blocks.sm2d", "aldrich_skirt_size12_h164.smis")
SKIRT_PIECES = ["A - Skirt Back", "A - Skirt Front"]


def read(p: Path) -> str:
    return p.read_text(encoding="utf-8-sig")


def set_measurement(smis_xml: str, name: str, delta: float) -> str:
    """Return smis text with measurement ``name`` increased by ``delta`` (file unit)."""
    pat = re.compile(rf'(<m name="{name}" value=")([-\d.]+)(")')
    assert pat.search(smis_xml), name
    return pat.sub(lambda m: f"{m.group(1)}{float(m.group(2)) + delta!r}{m.group(3)}", smis_xml)


def inspect(sm2d: str, smis: str, pieces: list[str] | None = None, smis_xml: str | None = None) -> dict[str, Any]:
    """Run the engine without writing; return reports by piece name."""
    req = EngineRequest(read(SM2D / sm2d), smis_xml or read(SMIS / smis), EngineOptions(pieces=pieces))
    return {p.name: p for p in Engine().inspect(req).pieces}


def make_scope(sm2d: str, smis: str, smis_xml: str | None = None) -> VariableScope:
    return VariableScope(parse_sm2d(read(SM2D / sm2d)), parse_smis(smis_xml or read(SMIS / smis)), default_registry)


def point_by_name(scope: VariableScope, name: str) -> tuple[float, float]:
    ids = [i for i, t in scope.model.tools.items() if t.kind == "point" and t.name == name]
    assert len(ids) == 1, name
    return scope.point(ids[0])


@pytest.fixture(scope="session")
def shirt_reports() -> dict[str, Any]:
    return inspect("male_shirt.sm2d", "male_shirt.smis")


@pytest.fixture(scope="session")
def scene(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    from avatar_engine import AvatarEngine
    from sm2d_engine.engine import Engine, EngineOptions, EngineRequest

    base = tmp_path_factory.mktemp("scene")
    models = ROOT / "models" / "smplx"
    if not (models / "SMPLX_MALE.npz").is_file():
        pytest.skip("models/smplx not present")
    AvatarEngine().build((ROOT / "assert/body/male_173.json").read_text(), models, base / "avatar")
    sm2d = (ROOT / "assert/sm2d/male_shirt.sm2d").read_text(encoding="utf-8")
    smis = (ROOT / "assert/smis/male_shirt.smis").read_text(encoding="utf-8")
    Engine().run(
        EngineRequest(sm2d, smis, EngineOptions(output_dir=base / "pieces"), "male_shirt.sm2d", "male_shirt.smis")
    )
    return {"avatar": base / "avatar", "pieces": base / "pieces", "out": base / "out"}

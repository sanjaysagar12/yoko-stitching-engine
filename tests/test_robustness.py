"""Error handling, laziness / cycle detection, jackets, DI and options."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from sm2d_engine.engine import Engine, EngineOptions, EngineRequest
from sm2d_engine.errors import (
    CyclicDependencyError,
    FormulaError,
    MissingMeasurementError,
    UnsupportedFormatError,
    UnsupportedToolError,
)
from sm2d_engine.eval.scope import VariableScope
from sm2d_engine.io.sm2d_reader import parse_sm2d
from sm2d_engine.io.smis_reader import parse_smis
from sm2d_engine.tools import default_registry

from . import synthetic as syn
from .conftest import SM2D, SMIS, read


def scope_for(sm2d: str) -> VariableScope:
    return VariableScope(parse_sm2d(sm2d), parse_smis(syn.smis()), default_registry)


def test_cycle_detected() -> None:
    body = syn.single("1", 0, 0) + syn.end_line("2", "3", "1", "0") + syn.end_line("3", "2", "1", "0")
    with pytest.raises(CyclicDependencyError) as e:
        scope_for(syn.sm2d("cm", body, "")).point("2")
    assert e.value.chain[0] == e.value.chain[-1] and "point 2" in e.value.chain


def test_increment_cycle() -> None:
    inc = '<increment name="#a" formula="#b+1"/><increment name="#b" formula="#a+1"/>'
    with pytest.raises(CyclicDependencyError):
        scope_for(syn.sm2d("cm", "", "", inc)).value("#a")


def test_missing_measurement_and_formula_errors() -> None:
    body = syn.single("1", 0, 0) + syn.end_line("2", "1", "waist_nope", "0")
    with pytest.raises(MissingMeasurementError) as e:
        scope_for(syn.sm2d("cm", body, "")).point("2")
    assert "waist_nope" in str(e.value)
    body = syn.single("1", 0, 0) + syn.end_line("2", "1", "1+", "0")
    with pytest.raises(FormulaError):
        scope_for(syn.sm2d("cm", body, "")).point("2")


def test_document_order_is_not_dependency_order() -> None:
    fwd = syn.single("1", 0, 0) + syn.end_line("2", "1", "10", "0") + syn.end_line("3", "2", "5", "270")
    rev = syn.end_line("3", "2", "5", "270") + syn.end_line("2", "1", "10", "0") + syn.single("1", 0, 0)
    assert scope_for(syn.sm2d("cm", fwd, "")).point("3") == scope_for(syn.sm2d("cm", rev, "")).point("3")


def test_unsupported_tool_names_tool_and_piece() -> None:
    err = UnsupportedToolError("bisector", "Bodice")
    assert "bisector" in str(err) and "Bodice" in str(err)
    with pytest.raises(UnsupportedToolError, match="nopeTool"):
        default_registry.get("nopeTool")


@pytest.mark.parametrize("n", range(1, 7))
def test_jackets_fail_per_piece_without_crashing(n: int) -> None:
    f = next(SM2D.glob(f"jacket{n}_*.sm2d"))
    res = Engine().inspect(EngineRequest(read(f), read(SMIS / "male_shirt.smis")))
    assert res.header["units"]["pattern"] == "mm"
    assert res.pieces
    for p in res.pieces:
        if p.status == "failed":
            r = p.reason or ""
            assert ("unsupported tool '" in r) or ("missing measurement or variable '" in r), r


def test_pieces_filter_and_unknown_piece() -> None:
    xml = syn.sm2d("cm", syn.single("1", 0, 0), syn.piece("A", ["1"]))
    res = Engine().inspect(EngineRequest(xml, syn.smis(), EngineOptions(pieces=["A", "Nope"])))
    assert [(p.name, p.status) for p in res.pieces] == [("A", "ok"), ("Nope", "failed")]


def test_mirror_fold_warns_without_fold_data() -> None:
    body = syn.single("1", 0, 0) + syn.end_line("2", "1", "10", "0") + syn.end_line("3", "2", "5", "270")
    xml = syn.sm2d(
        "cm",
        body,
        '<piece name="A"><nodes><node idObject="1" type="NodePoint"/>'
        '<node idObject="2" type="NodePoint"/><node idObject="3" type="NodePoint"/></nodes></piece>',
    )
    res = Engine().inspect(EngineRequest(xml, syn.smis(), EngineOptions(mirror_fold_pieces=True)))
    w = " ".join(res.pieces[0].warnings)
    assert "quantity not present" in w and "no fold information; not mirrored" in w
    assert res.pieces[0].quantity is None and res.pieces[0].on_fold is None


def test_ply_is_fatal(tmp_path: Path) -> None:
    xml = syn.line_piece("cm", "1")
    with pytest.raises(UnsupportedFormatError):
        Engine().run(EngineRequest(xml, syn.smis(), EngineOptions(output_dir=tmp_path, format="ply")))


def test_triangulator_is_injected() -> None:
    class Fan:
        def triangulate(self, polygon: Any) -> Any:
            n = len(polygon)
            return np.array([[0, i, i + 1] for i in range(1, n - 1)], dtype=np.int64).reshape(-1, 3)

    body = (
        syn.single("1", 0, 0)
        + syn.end_line("2", "1", "10", "0")
        + syn.end_line("3", "2", "5", "270")
        + syn.end_line("4", "1", "5", "270")
    )
    xml = syn.sm2d("cm", body, syn.piece("R", ["1", "2", "3", "4"]))
    res = Engine(triangulator=Fan()).inspect(EngineRequest(xml, syn.smis()))
    p = res.pieces[0]
    assert p.triangles == 2 and p.area_cm2 == pytest.approx(50.0) and p.mesh_area_cm2 == pytest.approx(50.0)

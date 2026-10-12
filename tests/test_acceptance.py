"""The 8 acceptance tests from the spec."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from sm2d_engine.cli import main
from sm2d_engine.engine import Engine, EngineOptions, EngineRequest
from sm2d_engine.eval.scope import VariableScope

from . import synthetic as syn
from .conftest import (
    ROOT,
    SKIRT,
    SKIRT_PIECES,
    SM2D,
    SMIS,
    inspect,
    make_scope,
    point_by_name,
    read,
    set_measurement,
)


def run_to(tmp: Path, sm2d: str, smis: str, **opt: Any) -> Path:
    Engine().run(EngineRequest(sm2d, smis, EngineOptions(output_dir=tmp, **opt), "t.sm2d", "t.smis"))
    return tmp


def verts(text: str) -> np.ndarray:
    return np.array([ln.split()[1:] for ln in text.splitlines() if ln.startswith("v ")], dtype=float)


def faces(text: str) -> list[str]:
    return [ln for ln in text.splitlines() if ln.startswith("f ")]


# 1 ----------------------------------------------------------------------------------------------
def test_1_unit_round_trip(tmp_path: Path) -> None:
    run_to(tmp_path, syn.line_piece("cm", "10"), syn.smis())
    text = (tmp_path / "Line_m.obj").read_text()
    v = verts(text)
    assert len(v) == 2
    assert abs(v[1, 0] - v[0, 0]) == pytest.approx(0.100000, abs=1e-12)
    assert "v 0.100000 0.000000 0.000000" in text and "v 0.000000 0.000000 0.000000" in text


# 2 ----------------------------------------------------------------------------------------------
def test_2_skirt_formulas_and_sizes() -> None:
    sc = make_scope(*SKIRT)
    m = sc.measurements.get
    cm = sc.value("#CM")
    assert cm == pytest.approx(m("height") / 166)

    def p(n: str) -> tuple[float, float]:
        return point_by_name(sc, n)

    tol = 1e-3  # 0.01 mm in cm
    assert math.dist(p("A"), p("A1")) == pytest.approx(5 * cm, abs=tol)
    assert abs(p("A7")[0] - p("A1")[0]) == pytest.approx(m("hip_circ") / 4 + 1.5 * cm, abs=tol)
    assert math.dist(p("A1"), p("A3")) == pytest.approx(m("height_waist_side_to_knee"), abs=tol)
    assert math.dist(p("A1"), p("A9")) == pytest.approx(m("waist_circ") / 4 + 4 * cm, abs=tol)

    r = inspect(*SKIRT, SKIRT_PIECES)
    exp = {"A - Skirt Back": (30.457, 59.971, 1574.76), "A - Skirt Front": (28.957, 59.95, 1484.99)}
    for name, (w, h, a) in exp.items():
        assert r[name].width_cm == pytest.approx(w, abs=0.001)
        assert r[name].height_cm == pytest.approx(h, abs=0.001)
        # 0.1 mm chord flattening makes the polygon up to ~0.03 cm2 smaller than the true curve area
        assert r[name].area_cm2 == pytest.approx(a, abs=0.05)


# 3 ----------------------------------------------------------------------------------------------
SHIRT = {
    "FrontPanel": (37.25, 74.40),
    "BackPanel": (34.75, 68.50),
    "Yoke": (25.67, 11.46),
    "ShortSleeve": (52.30, 34.00),
    "FullSleeve": (52.30, 62.55),
    "Cuff": (29, 6),
    "CollarBase": (22.80, 4.30),
    "PlacketTop": (19.71, 6.00),
}


def test_3_shirt_regression(shirt_reports: dict[str, Any]) -> None:
    assert len(shirt_reports) == 17
    for name, (w, h) in SHIRT.items():
        assert shirt_reports[name].width_cm == pytest.approx(w, abs=0.01), name
        assert shirt_reports[name].height_cm == pytest.approx(h, abs=0.01), name
    for name, p in shirt_reports.items():
        assert p.status == "ok", name
        assert p.gaps == [], name
        assert p.area_diff_pct < 0.001, name
        assert p.self_intersections <= (1 if name == "FullSleeve" else 0), name


# 4 ----------------------------------------------------------------------------------------------
def test_4_measurement_sensitivity() -> None:
    xml = set_measurement(read(SMIS / SKIRT[1]), "hip_circ", 4.0)

    def hip_width(sc: VariableScope) -> float:
        return abs(point_by_name(sc, "A7")[0] - point_by_name(sc, "A1")[0])

    assert hip_width(make_scope(*SKIRT, smis_xml=xml)) - hip_width(make_scope(*SKIRT)) == pytest.approx(1.0, abs=1e-9)


# 5 ----------------------------------------------------------------------------------------------
def _objs(tmp: Path) -> dict[str, str]:
    return {p.name: p.read_text() for p in sorted(tmp.glob("*.obj"))}


def test_5a_smis_in_mm_or_cm_identical(tmp_path: Path) -> None:
    xml = syn.rect_with_curve(1, "cm")
    a = _objs(run_to(tmp_path / "cm", xml, syn.smis(100, "cm")))
    b = _objs(run_to(tmp_path / "mm", xml, syn.smis(1000, "mm")))
    assert a and a == b


def test_5b_mm_pattern_with_cm_smis_equals_cm_pattern(tmp_path: Path) -> None:
    a = _objs(run_to(tmp_path / "cm", syn.rect_with_curve(1, "cm"), syn.smis(100, "cm")))
    b = _objs(run_to(tmp_path / "mm", syn.rect_with_curve(10, "mm"), syn.smis(100, "cm")))
    assert a and a.keys() == b.keys()
    for k in a:
        va, vb = verts(a[k]), verts(b[k])
        assert va.shape == vb.shape and np.abs(va - vb).max() < 2e-6, k  # same mesh to 2 micrometres
        assert faces(a[k]) == faces(b[k])


# 6 ----------------------------------------------------------------------------------------------
def _isolation_pattern() -> str:
    body = (
        syn.single("1", 0, 0)
        + syn.end_line("2", "1", "10", "0")
        + syn.end_line("3", "2", "5", "270")
        + syn.end_line("4", "1", "5", "270")
        + '<point id="5" name="P5" type="trueDarts" basePoint="1"/>'
    )
    pieces = syn.piece("Good", ["1", "2", "3", "4"]) + syn.piece("Bad", ["1", "2", "5"])
    return syn.sm2d("cm", body, pieces)


def test_6_isolation_engine(tmp_path: Path) -> None:
    res = Engine().run(EngineRequest(_isolation_pattern(), syn.smis(), EngineOptions(output_dir=tmp_path)))
    by = {p.name: p for p in res.pieces}
    assert by["Good"].status == "ok" and by["Bad"].status == "failed"
    assert "trueDarts" in (by["Bad"].reason or "") and "Bad" in (by["Bad"].reason or "")
    assert (tmp_path / "Good_m.obj").exists() and not (tmp_path / "Bad_m.obj").exists()
    report = json.loads((tmp_path / "report.json").read_text())
    assert [p["status"] for p in report["pieces"]] == ["ok", "failed"]


def test_6_isolation_exit_code(tmp_path: Path) -> None:
    (tmp_path / "p.sm2d").write_text(_isolation_pattern())
    (tmp_path / "m.smis").write_text(syn.smis())
    job = tmp_path / "job.json"
    job.write_text(json.dumps({"sm2d": "p.sm2d", "smis": "m.smis", "output_dir": str(tmp_path / "out")}))
    assert main(["run", str(job), "--sm2d-dir", str(tmp_path), "--smis-dir", str(tmp_path)]) == 1
    assert (tmp_path / "out" / "Good_m.obj").exists()


def test_exit_code_fatal(tmp_path: Path) -> None:
    job = tmp_path / "job.json"
    job.write_text(json.dumps({"sm2d": "x.sm2d", "smis": "y.smis", "bogus": 1}))
    assert main(["run", str(job)]) == 2  # unknown key
    job.write_text(json.dumps({"sm2d": "nope.sm2d", "smis": "nope.smis"}))
    assert main(["run", str(job), "--sm2d-dir", str(tmp_path), "--smis-dir", str(tmp_path)]) == 2  # missing files


# 7 ----------------------------------------------------------------------------------------------
def test_7_cli_e2e(tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(ROOT)
    out = tmp_path / "meshes"
    assert main(["run", "examples/job.json", "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "17/17 pieces ok" in printed and "FrontPanel" in printed
    files = {p.name for p in out.iterdir()}
    assert len([f for f in files if f.endswith("_m.obj")]) == 17 + 1  # pieces + combined layout
    assert {"FrontPanel_m.obj", "male_shirt_all_pieces_layout_m.obj", "report.json"} <= files
    assert (out / "male_shirt_all_pieces_layout_m.obj").read_text().count("\ng ") == 17
    rep = json.loads((out / "report.json").read_text())
    assert rep["pattern"] == "male_shirt.sm2d" and rep["measurements"] == "male_shirt.smis"
    assert rep["units"]["pattern"] == "cm" and len(rep["pieces"]) == 17
    fp = next(p for p in rep["pieces"] if p["name"] == "FrontPanel")
    keys = {"width_cm", "height_cm", "area_cm2", "perimeter_cm", "mesh_area_cm2", "vertices", "triangles"}
    assert keys | {"self_intersections", "gaps", "status", "warnings"} <= fp.keys()
    header = (out / "FrontPanel_m.obj").read_text().splitlines()[:8]
    assert any("units: metres" in h for h in header) and any("male_shirt.smis" in h for h in header)


def test_7_inspect_writes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(ROOT)
    job = tmp_path / "job.json"
    spec = {"sm2d": "male_shirt.sm2d", "smis": "male_shirt.smis", "output_dir": str(tmp_path / "o")}
    job.write_text(json.dumps(spec))
    assert main(["inspect", str(job)]) == 0
    assert not (tmp_path / "o").exists()
    assert "Yoke" in capsys.readouterr().out


# 8 ----------------------------------------------------------------------------------------------
def test_8_determinism(tmp_path: Path) -> None:
    sm2d, smis = read(SM2D / "male_shirt.sm2d"), read(SMIS / "male_shirt.smis")
    a = run_to(tmp_path / "a", sm2d, smis)
    b = run_to(tmp_path / "b", sm2d, smis)
    names = sorted(p.name for p in a.iterdir())
    assert (
        names == sorted(p.name for p in b.iterdir()) and len(names) == 17 * 2 + 2
    )  # OBJ + piece.json per piece, combined, report
    for n in names:
        assert (a / n).read_bytes() == (b / n).read_bytes(), n

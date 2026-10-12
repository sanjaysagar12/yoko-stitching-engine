"""The piece bundle JSON written next to each piece OBJ."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from sm2d_engine.contracts import read_json
from sm2d_engine.engine import Engine, EngineOptions, EngineRequest
from tests.conftest import SKIRT, SKIRT_PIECES, SM2D, SMIS, read


def run(tmp: Path, sm2d: str, smis: str, pieces: list[str] | None = None) -> dict[str, dict[str, Any]]:
    res = Engine().run(
        EngineRequest(read(SM2D / sm2d), read(SMIS / smis), EngineOptions(output_dir=tmp, pieces=pieces), sm2d, smis)
    )
    out: dict[str, dict[str, Any]] = {}
    for p in res.pieces:
        if p.status == "ok":
            f = tmp / str(p.file).replace("_m.obj", ".piece.json")
            out[p.name] = {"bundle": read_json(f, "piece_bundle"), "report": p, "file": f}
    return out


@pytest.fixture(scope="module")
def shirt(tmp_path_factory: pytest.TempPathFactory) -> dict[str, dict[str, Any]]:
    return run(tmp_path_factory.mktemp("shirt"), "male_shirt.sm2d", "male_shirt.smis")


def obj_vertices(path: Path) -> np.ndarray:
    return np.array([ln.split()[1:3] for ln in path.read_text().splitlines() if ln.startswith("v ")], dtype=float)


def test_one_bundle_per_piece_and_validates(shirt: dict[str, dict[str, Any]]) -> None:
    assert len(shirt) == 17
    for name, d in shirt.items():
        b = d["bundle"]
        assert b["name"] == name and b["units"] == "m"
        assert (d["file"].parent / b["obj"]).is_file()


def test_outline_is_the_mesh_boundary(shirt: dict[str, dict[str, Any]]) -> None:
    for name, d in shirt.items():
        v = obj_vertices(d["file"].parent / d["bundle"]["obj"])
        o = np.array(d["bundle"]["outline_m"])
        assert o.shape == v.shape and np.abs(o - v).max() < 1e-6, name


def test_bbox_and_default_reference(shirt: dict[str, dict[str, Any]]) -> None:
    for name, d in shirt.items():
        b, r = d["bundle"], d["report"]
        assert b["bbox_m"]["min"] == pytest.approx([0, 0], abs=1e-9)
        assert b["bbox_m"]["max"] == pytest.approx([r.width_cm / 100, r.height_cm / 100], abs=1e-6), name
        assert b["default_reference"]["xy_m"] == pytest.approx([r.width_cm / 200, r.height_cm / 200], abs=1e-6), name


def test_named_points_inside_bbox(shirt: dict[str, dict[str, Any]]) -> None:
    total = 0
    for name, d in shirt.items():
        (x0, y0), (x1, y1) = d["bundle"]["bbox_m"]["min"], d["bundle"]["bbox_m"]["max"]
        for p in d["bundle"]["named_points"]:
            x, y = p["xy_m"]
            assert x0 - 3e-4 <= x <= x1 + 3e-4 and y0 - 3e-4 <= y <= y1 + 3e-4, (name, p["name"])
            total += 1
    assert total > 50


def test_edge_lengths_sum_to_perimeter(shirt: dict[str, dict[str, Any]]) -> None:
    checked = 0
    for name, d in shirt.items():
        b = d["bundle"]
        if any(p["on_outline"] for p in b["named_points"]):
            total = sum(e["length_m"] for e in b["edges"])
            assert total == pytest.approx(d["report"].perimeter_cm / 100, abs=1e-6), name
            checked += 1
        else:
            assert b["edges"] == []
    assert checked >= 15


def test_edges_connect_on_outline_points(shirt: dict[str, dict[str, Any]]) -> None:
    b = shirt["Cuff"]["bundle"]
    on = [p for p in b["named_points"] if p["on_outline"]]
    assert len(on) >= 3 and len(b["edges"]) == len(on)
    names = {p["name"] for p in on}
    assert all(e["from"] in names and e["to"] in names for e in b["edges"])
    per = shirt["Cuff"]["report"].perimeter_cm / 100
    assert per == pytest.approx(0.70, abs=1e-6)
    assert all(0 <= p["outline_s_m"] <= per + 1e-9 for p in on)


def test_fold_and_grainline_null_when_absent(shirt: dict[str, dict[str, Any]]) -> None:
    for d in shirt.values():
        assert d["bundle"]["fold_edge"] is None and d["bundle"]["grainline"] is None


def test_grainline_when_present(tmp_path: Path) -> None:
    r = run(tmp_path, *SKIRT, SKIRT_PIECES)
    g = r["A - Skirt Back"]["bundle"]["grainline"]
    assert g is not None and g["length_m"] == pytest.approx(0.15) and g["rotation_deg"] == 90
    # onFold is true but foldPosition is Undefined: the fold edge is not guessed
    assert r["A - Skirt Back"]["bundle"]["fold_edge"] is None


def test_trousers_rotated_pieces_have_points(tmp_path: Path) -> None:
    r = run(tmp_path, "trousers.sm2d", "trousers.smis", ["Back", "Front_Pleats"])
    assert set(r) == {"Back", "Front_Pleats"}
    for d in r.values():
        assert sum(p["on_outline"] for p in d["bundle"]["named_points"]) >= 5


def test_report_names_bundle_neighbour(tmp_path: Path) -> None:
    run(tmp_path, "male_shirt.sm2d", "male_shirt.smis", ["Cuff"])
    rep = json.loads((tmp_path / "report.json").read_text())
    assert rep["pieces"][0]["file"] == "Cuff_m.obj"
    assert (tmp_path / "Cuff.piece.json").is_file()

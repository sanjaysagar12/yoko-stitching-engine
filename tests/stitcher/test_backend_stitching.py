"""/api/stitching/*: import (schema problems), run on the layout on screen (nothing moves), result and zip export.

Real backend, synthetic avatar, the skirt (the quickest garment)."""

from __future__ import annotations

import dataclasses
import io
import json
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from tests.web_build import prebuild
from web.backend.app import create_app
from web.backend.config import AppConfig

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def env(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[TestClient, dict[str, Any]]]:
    out = tmp_path_factory.mktemp("outputs")
    bid = prebuild(out, "aldrich_skirt")
    cfg = dataclasses.replace(AppConfig.default(), outputs_dir=out)
    asm = json.loads((ROOT / "assert/assemblies/aldrich_skirt.assembly.json").read_text(encoding="utf-8"))
    st = json.loads((ROOT / "assert/stitchings/aldrich_skirt.stitching.json").read_text(encoding="utf-8"))
    with TestClient(create_app(cfg)) as c:
        r = c.post("/api/arrange", json={"build": bid, "placements": asm["placements"],
                                        "derived_pieces": asm["derived_pieces"]})  # fmt: skip
        assert r.status_code == 200, r.text
        solved = [{**p, "solved": d["solved"]} for p, d in zip(asm["placements"], r.json()["placements"], strict=True)]
        yield c, {"build": bid, "placements": solved, "derived_pieces": asm["derived_pieces"], "stitching": st}


def test_import_lists_schema_problems(env: tuple[TestClient, dict[str, Any]]) -> None:
    c, body = env
    ok = c.post("/api/stitching/import", content=json.dumps(body["stitching"]))
    assert ok.status_code == 200 and ok.json()["stitching"]["garment"] == "aldrich_skirt"
    bad = c.post("/api/stitching/import", content=json.dumps({**body["stitching"], "seams": [{"id": "x"}]}))
    assert bad.status_code == 422 and bad.json()["error"]["code"] == "invalid_stitching"


def test_run_stitches_the_layout_on_screen(env: tuple[TestClient, dict[str, Any]]) -> None:
    c, body = env
    r = c.post("/api/stitching/run", json=body)
    assert r.status_code == 200, r.text
    a = r.json()
    assert a["pass"], a["failures"]
    assert {s["id"] for s in a["seams"]} == {s["id"] for s in body["stitching"]["seams"]}
    assert all(s["status"] == "pass" and len(s["pairs"]) >= 2 for s in a["seams"])
    res = c.post("/api/stitching/result", json=body).json()
    assert res["untouched"] and res["summary"]["unstitched_edges"] == 0


def test_unknown_point_is_reported(env: tuple[TestClient, dict[str, Any]]) -> None:
    c, body = env
    st = json.loads(json.dumps(body["stitching"]))
    st["seams"][0]["a"]["chain"][0] = "Z99"
    a = c.post("/api/stitching/run", json={**body, "stitching": st}).json()
    assert not a["pass"] and any("Z99" in f for f in a["failures"])


def test_export_zip(env: tuple[TestClient, dict[str, Any]]) -> None:
    c, body = env
    r = c.post("/api/stitching/export", json=body)
    assert r.status_code == 200
    names = zipfile.ZipFile(io.BytesIO(r.content)).namelist()
    assert {
        "stitching.result.json",
        "stitch_verify.md",
        "images/stitch_sheet.png",
        "images/stitch_unstitched.png",
    } <= set(names)
    assert any(n.startswith("images/stitch_seam_") for n in names)

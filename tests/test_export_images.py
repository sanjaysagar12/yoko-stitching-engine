"""POST /api/export/images: one zip of every view, sections, contact sheet and close-ups of the layout on screen.

Real backend, synthetic avatar (no SMPL-X files needed).
"""

from __future__ import annotations

import base64
import dataclasses
import io
import json
import time
import zipfile
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from tests.web_build import prebuild
from web.backend.app import create_app
from web.backend.config import AppConfig

THREE = [
    {"piece": "FrontPanel", "point": "Chest_Front", "reference": "A32", "wrap": "roll"},
    {"piece": "BackPanel", "point": "Chest_Back", "reference": "A24", "wrap": "roll"},
    {"piece": "Yoke", "point": "Chest_Back", "wrap": "roll"},
]
ANGLES = ["front", "back", "left", "right", "top", "front_left", "front_right", "back_left", "back_right"]
REQUIRED = {f"{a}.png" for a in ANGLES} | {
    "points_front.png",
    "heat_front.png",
    "heat_back.png",
    "sections.png",
    "sheet.png",
}


@pytest.fixture(scope="module")
def env(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[TestClient, str, list[dict[str, Any]]]]:
    out = tmp_path_factory.mktemp("outputs")
    bid = prebuild(out)
    cfg = dataclasses.replace(AppConfig.default(), outputs_dir=out)
    with TestClient(create_app(cfg)) as c:
        res = c.post("/api/align", json={"build": bid, "placements": THREE}).json()
        solved = [{**pl, "solved": d["solved"]} for pl, d in zip(THREE, res["placements"], strict=True)]
        yield c, bid, solved


def wait(c: TestClient, jid: str, until: tuple[str, ...] = ("done", "failed", "cancelled")) -> dict[str, Any]:
    for _ in range(400):
        st: dict[str, Any] = c.get(f"/api/export/images/{jid}").json()
        if st["status"] in until:
            return st
        time.sleep(0.2)
    raise AssertionError(f"export did not finish: {st}")


def start(c: TestClient, bid: str, pls: list[dict[str, Any]], **kw: Any) -> dict[str, Any]:
    r = c.post("/api/export/images", json={"build": bid, "placements": pls, "garment": "shirt", **kw})
    assert r.status_code == 200, r.text
    out: dict[str, Any] = r.json()
    return out


def zip_of(c: TestClient, jid: str) -> zipfile.ZipFile:
    r = c.get(f"/api/export/images/{jid}/download")
    assert r.status_code == 200 and r.headers["content-type"] == "application/zip", r.text
    assert "shirt_images.zip" in r.headers["content-disposition"]
    return zipfile.ZipFile(io.BytesIO(r.content))


def test_zip_has_every_view_the_sheet_and_the_sections(env: tuple[TestClient, str, list[dict[str, Any]]]) -> None:
    c, bid, pls = env
    st = wait(c, start(c, bid, pls)["id"])
    assert st["status"] == "done" and st["failures"] == [], st
    z = zip_of(c, st["id"])
    names = set(z.namelist())
    assert REQUIRED <= names and "assembly.json" in names and "export_report.txt" in names
    assert {f"closeups/{p['piece']}.png" for p in pls} <= names
    assert "verify.md" not in names and len(names) >= 12
    for n in names:
        if n.endswith(".png"):
            im = Image.open(io.BytesIO(z.read(n)))
            im.load()
            assert im.size[0] > 50 and im.getbbox() is not None
    doc = json.loads(z.read("assembly.json"))
    assert [p["solved"] for p in doc["placements"]] == [p["solved"] for p in pls]  # nothing moved, nothing re-solved
    assert doc["expect"]["required_pieces"] == [p["piece"] for p in pls]


def test_options_verification_and_screenshots(
    env: tuple[TestClient, str, list[dict[str, Any]]], monkeypatch: pytest.MonkeyPatch
) -> None:
    from web.backend import exports

    # the independent verifier takes minutes on a real body (tested in test_assembler); here only the wiring counts
    monkeypatch.setattr(exports, "verify_run", lambda *a, **k: {"near": {}, "fit": [], "ease": []})
    c, bid, pls = env
    shot = base64.b64encode(_png()).decode()
    st = wait(
        c,
        start(
            c,
            bid,
            pls,
            options={"closeups": False, "heat": False, "verify": True, "assembly": False},
            viewport={"front": shot},
        )["id"],
    )
    assert st["status"] == "done", st
    names = set(zip_of(c, st["id"]).namelist())
    assert {"verify.md", "verify.json", "viewport/front.png", "sheet.png", "sections.png"} <= names
    assert not any(n.startswith("closeups/") or n.startswith("heat_") or n == "assembly.json" for n in names)
    bad = c.post("/api/export/images", json={"build": bid, "placements": pls, "viewport": {"../x": shot}})
    assert bad.status_code == 422
    bad = c.post("/api/export/images", json={"build": bid, "placements": pls, "viewport": {"a": "bm90IGEgcG5n"}})
    assert bad.status_code == 422


def _png() -> bytes:
    b = io.BytesIO()
    Image.new("RGB", (8, 8), (200, 10, 10)).save(b, "PNG")
    return b.getvalue()


def test_cache_determinism_and_errors(env: tuple[TestClient, str, list[dict[str, Any]]]) -> None:
    c, bid, pls = env
    a = start(c, bid, pls, options={"assembly": False}, notes="one")
    sa = wait(c, a["id"])
    again = start(c, bid, pls, options={"assembly": False}, notes="one")
    assert again["id"] == a["id"] and again["status"] == "done"  # cached
    other = start(c, bid, pls, options={"assembly": False}, notes="two")  # same pictures, other cache key
    assert other["id"] != a["id"]
    wait(c, other["id"])
    assert zip_of(c, a["id"]).namelist() == zip_of(c, other["id"]).namelist()
    ra = c.get(f"/api/export/images/{a['id']}/download").content
    rb = c.get(f"/api/export/images/{other['id']}/download").content
    assert ra == rb and sa["status"] == "done"  # same layout -> same bytes
    assert c.get("/api/export/images/nope").status_code == 404
    assert (
        c.post("/api/export/images", json={"build": bid, "placements": []}).json()["error"]["code"] == "no_placements"
    )
    assert (
        c.post("/api/export/images", json={"build": "x-1", "placements": pls}).json()["error"]["code"]
        == "build_not_found"
    )
    assert c.post("/api/export/images", json={"build": bid, "placements": pls, "garment": "a/b"}).status_code == 422


def test_cancel_and_progress(env: tuple[TestClient, str, list[dict[str, Any]]]) -> None:
    c, bid, pls = env
    first = start(c, bid, pls, notes="busy")  # occupies the single worker
    second = start(c, bid, pls, notes="queued")
    assert second["status"] == "queued" or second["status"] == "running"
    c.delete(f"/api/export/images/{second['id']}")
    st = wait(c, second["id"])
    assert st["status"] == "cancelled"
    assert c.get(f"/api/export/images/{second['id']}/download").status_code == 422
    mid = c.get(f"/api/export/images/{first['id']}").json()
    assert 0 <= mid["progress"] <= 1 and mid["stage"]
    assert wait(c, first["id"])["status"] == "done"
    # a cancelled export can be started again
    assert wait(c, start(c, bid, pls, notes="queued")["id"])["status"] == "done"


def test_a_failing_view_is_reported_and_the_rest_is_kept(
    env: tuple[TestClient, str, list[dict[str, Any]]], monkeypatch: pytest.MonkeyPatch
) -> None:
    from assembler import render as R

    c, bid, pls = env
    real = R.render_view

    def flaky(scene: Any, view: Any, title: str, *a: Any, **k: Any) -> Any:
        if view.name == "back" and "clearance" not in title:
            raise RuntimeError("boom")
        return real(scene, view, title, *a, **k)

    monkeypatch.setattr(R, "render_view", flaky)
    st = wait(c, start(c, bid, pls, notes="flaky")["id"])
    assert st["status"] == "done" and len(st["failures"]) == 1 and "back" in st["failures"][0]
    z = zip_of(c, st["id"])
    assert "back.png" not in z.namelist() and "front.png" in z.namelist() and "sheet.png" in z.namelist()
    assert "boom" in z.read("export_report.txt").decode()

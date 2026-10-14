"""/api/assembly/import and /api/assembly/export: normalised payloads, every problem listed, round trip, Save/Load.

Real backend, synthetic avatar (no SMPL-X files needed).
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from assembler.run import load_assembly
from tests.web_build import prebuild
from web.backend.app import create_app
from web.backend.config import AppConfig

ROOT = Path(__file__).resolve().parents[1]
ASSEMBLIES = ROOT / "assert" / "assemblies"
GARMENTS = ("male_shirt", "aldrich_skirt", "trousers")


@pytest.fixture(scope="module")
def env(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[TestClient, dict[str, str]]]:
    out = tmp_path_factory.mktemp("outputs")
    ids = {g: prebuild(out, g) for g in GARMENTS}
    cfg = dataclasses.replace(AppConfig.default(), outputs_dir=out)
    with TestClient(create_app(cfg)) as c:
        yield c, ids


def asm(g: str = "male_shirt") -> dict[str, Any]:
    return json.loads((ASSEMBLIES / f"{g}.assembly.json").read_text(encoding="utf-8"))


def imp(c: TestClient, doc: Any, **q: Any) -> Any:
    raw = doc if isinstance(doc, bytes | str) else json.dumps(doc)
    return c.post("/api/assembly/import", content=raw, params=q)


def codes(r: Any) -> set[str]:
    return {p["code"] for p in r.json()["error"]["problems"]}


@pytest.mark.parametrize("g", GARMENTS)
def test_import_of_the_committed_assemblies(env: tuple[TestClient, dict[str, str]], g: str) -> None:
    c, ids = env
    r = imp(c, asm(g), build=ids[g])
    assert r.status_code == 200, r.text
    p = r.json()
    assert p["garment"] == g and p["job"] == g and p["mode"] == "arrange"
    assert {
        "derived_pieces",
        "placements",
        "arrange_options",
        "stitches",
        "rings",
        "expect",
        "views",
        "notes",
        "warnings",
    } <= set(p)
    assert len(p["placements"]) == len(asm(g)["placements"])
    assert p["notes"] == asm(g).get("notes", "") and p["expect"] == asm(g).get("expect", {})
    assert imp(c, asm(g)).status_code == 200  # without a build only the file and its job are checked


def test_bad_files_give_specific_codes_and_list_every_problem(env: tuple[TestClient, dict[str, str]]) -> None:
    c, ids = env
    bid = ids["male_shirt"]
    r = imp(c, b"{not json")
    assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_assembly"
    assert codes(imp(c, {"schema_version": 7, "garment": "x"})) == {"invalid_assembly", "unknown_job"}
    assert imp(c, {**asm(), "job": "nope"}).json()["error"]["code"] == "unknown_job"
    for path_like in ("../../etc/passwd", "a/b", "C:\\x"):
        assert imp(c, {**asm(), "job": path_like}).json()["error"]["code"] == "unknown_job"
    big = imp(c, " " * 2_100_000)
    assert big.status_code == 422 and "2 MB" in big.json()["error"]["message"]
    d = asm()
    own = [pl for pl in d["placements"] if not pl["piece"].endswith("_L")]
    d["placements"] = [
        {**own[0], "piece": "Ghost"},
        {**own[1], "point": "No_Such_Point"},
        {**own[2], "wrap": "body_slice"},
    ]
    d["arrange_options"] = {"allow_distortion": True}
    r = imp(c, d, build=bid)
    assert r.status_code == 422
    assert codes(r) == {"unsupported_option"}  # schema-level problems stop before the build checks
    d["placements"][2]["wrap"] = "roll"
    d["arrange_options"] = {}
    d["derived_pieces"] = [{"id": "X_L", "source": "Nothing", "op": "mirror", "axis": {"type": "bbox_center"}}]
    r = imp(c, d, build=bid)
    problems = r.json()["error"]["problems"]
    assert {"unknown_piece", "unknown_point"} <= codes(r) and len(problems) >= 3
    assert (
        imp(c, {**d, "derived_pieces": []}, build=bid, skip_missing=True).status_code == 422
    )  # the point stays an error
    d["placements"][1]["point"] = own[1]["point"]
    ok = imp(c, d, build=bid, skip_missing=True)
    assert ok.status_code == 200 and any("Ghost" in w for w in ok.json()["warnings"])
    assert all(pl["piece"] != "Ghost" for pl in ok.json()["placements"])
    assert imp(c, asm(), build=ids["trousers"]).json()["error"]["code"] == "unknown_job"  # a build of another job


def test_legacy_placements_json_is_upgraded(env: tuple[TestClient, dict[str, str]]) -> None:
    c, ids = env
    d = asm()
    legacy = {"schema_version": 2, "placements": d["placements"][:3], "derived_pieces": d["derived_pieces"][:1]}
    assert imp(c, legacy).json()["error"]["code"] == "unknown_job"
    r = imp(c, legacy, job="male_shirt", build=ids["male_shirt"])
    assert r.status_code == 200, r.text
    p = r.json()
    assert p["job"] == "male_shirt" and len(p["placements"]) == 3 and any("upgraded" in w for w in p["warnings"])


def test_utf8_bom_is_accepted(env: tuple[TestClient, dict[str, str]]) -> None:
    c, _ = env
    assert imp(c, b"\xef\xbb\xbf" + json.dumps(asm()).encode()).status_code == 200


def assemble(c: TestClient, bid: str, payload: dict[str, Any]) -> dict[str, Any]:
    body = {
        "build": bid,
        "placements": payload["placements"],
        "derived_pieces": payload["derived_pieces"],
        "arrange_options": payload["arrange_options"],
        "stitches": payload["stitches"],
        "rings": payload["rings"],
    }
    url = {"arrange": "/api/arrange", "align": "/api/align", "incremental": "/api/align"}[payload["run"]]
    r = c.post(url, json=body)
    assert r.status_code == 200, r.text
    out: dict[str, Any] = r.json()
    return out


def export(c: TestClient, payload: dict[str, Any], res: dict[str, Any], **kw: Any) -> str:
    pls = [{**pl, "solved": d["solved"]} for pl, d in zip(payload["placements"], res["placements"], strict=True)]
    body = {
        "job": payload["job"],
        "placements": pls,
        "derived_pieces": payload["derived_pieces"],
        "arrange_options": payload["arrange_options"],
        "stitches": payload["stitches"],
        "rings": payload["rings"],
        "garment": payload["garment"],
        "notes": payload["notes"],
        "mode": payload["mode"],
        "expect": payload["expect"],
        "views": payload["views"],
        **kw,
    }
    r = c.post("/api/assembly/export", json=body)
    assert r.status_code == 200, r.text
    return r.text


def verts(res: dict[str, Any]) -> list[np.ndarray]:
    return [np.array(p["wrapped_vertices"]) for p in res["placements"]]


def test_round_trip_shirt(env: tuple[TestClient, dict[str, str]], tmp_path: Path) -> None:
    c, ids = env
    bid = ids["male_shirt"]
    p1 = imp(c, asm(), build=bid).json()
    res1 = assemble(c, bid, p1)
    assert res1["validity"]["valid"] is True
    # with solved poses: identical
    text = export(c, p1, res1)
    assert text.endswith("\n") and "\r" not in text and text.startswith('{\n  "schema_version": 1')
    f = tmp_path / "out.assembly.json"
    f.write_text(text, encoding="utf-8", newline="\n")
    load_assembly(f)  # python -m assembler check
    doc = json.loads(text)
    assert doc["expect"] == asm()["expect"] and doc["notes"] == asm()["notes"]
    assert text == export(c, p1, res1)  # deterministic, stable key order
    p2 = imp(c, text, build=bid).json()
    assert p2["run"] == "align"
    res2 = assemble(c, bid, p2)
    for a, b in zip(verts(res1), verts(res2), strict=True):
        assert np.abs(a - b).max() < 1e-6
    # without solved poses: the same garment again within 1 mm
    bare = export(c, p1, res1, include_solved=False)
    assert '"solved"' not in bare
    p3 = imp(c, bare, build=bid).json()
    assert p3["run"] == "arrange"
    res3 = assemble(c, bid, p3)
    for a, b in zip(verts(res1), verts(res3), strict=True):
        assert np.abs(a - b).max() < 1e-3


def test_export_without_import_makes_a_minimal_expect_and_refuses_distortion(
    env: tuple[TestClient, dict[str, str]],
) -> None:
    c, _ = env
    pls = [
        {"piece": "FrontPanel", "point": "Chest_Front", "reference": "A32", "wrap": "roll", "tilt": {"pitch_deg": 5}}
    ]
    r = c.post("/api/assembly/export", json={"job": "male_shirt", "placements": pls, "notes": "hi", "garment": "g1"})
    doc = r.json()
    assert doc["expect"] == {"required_pieces": ["FrontPanel"]} and doc["garment"] == "g1" and doc["mode"] == "arrange"
    assert "/" not in r.text.replace("//", "") or "\\" not in r.text  # no absolute paths
    bad = c.post(
        "/api/assembly/export",
        json={"job": "male_shirt", "placements": pls, "arrange_options": {"allow_distortion": True}},
    )
    assert bad.status_code == 422 and bad.json()["error"]["code"] == "invalid_assembly"
    assert c.post("/api/assembly/export", json={"job": "../x", "placements": pls}).status_code == 404


def test_copy_and_mirror_round_trip_and_new_fields_survive(env: tuple[TestClient, dict[str, str]]) -> None:
    c, ids = env
    bid = ids["male_shirt"]
    d = asm()
    d["derived_pieces"] = [
        {"id": "Pocket_C", "source": "Pocket", "op": "copy"},
        {
            "id": "FrontPanel_L",
            "source": "FrontPanel",
            "op": "mirror",
            "axis": {"type": "edge", "from_point": "A33", "to_point": "A32"},
        },
    ]
    d["placements"] = [
        {
            "piece": "FrontPanel",
            "point": "Chest_Front",
            "reference": "A32",
            "wrap": "roll",
            "orient": 0,
            "tilt": {"pitch_deg": 4.0},
            "anchor": "auto",
            "rotation_deg": 1.5,
            "offset_m": 0.01,
        },
        {"piece": "Pocket", "point": "Chest_FrontSide_L", "wrap": "roll", "layer": 1},
        {"piece": "Pocket_C", "point": "Chest_FrontSide_R", "wrap": "roll", "layer": 1, "flip_facing": True},
    ]
    d["expect"] = {"required_pieces": ["FrontPanel", "Pocket", "Pocket_C"]}
    p = imp(c, d, build=bid).json()
    res = assemble(c, bid, p)
    text = export(c, p, res)
    back = json.loads(text)
    assert back["derived_pieces"] == d["derived_pieces"] or [x["id"] for x in back["derived_pieces"]] == [
        "Pocket_C",
        "FrontPanel_L",
    ]
    first = back["placements"][0]
    assert first["orient"] == 0 and first["tilt"] == {"pitch_deg": 4.0} and first["anchor"] == "auto"
    assert first["rotation_deg"] == 1.5 and first["offset_m"] == 0.01 and back["placements"][2]["flip_facing"] is True
    res2 = assemble(c, bid, imp(c, text, build=bid).json())
    for a, b in zip(verts(res), verts(res2), strict=True):
        assert np.abs(a - b).max() < 1e-6
    assert {x["op"] for x in res2["derived_pieces"]} == {"copy", "mirror"}


def test_save_and_load_keep_the_new_fields(env: tuple[TestClient, dict[str, str]]) -> None:
    c, _ = env
    pls = [
        {
            "piece": "FrontPanel",
            "point": "Chest_Front",
            "wrap": "roll",
            "orient": 90,
            "tilt": {"roll_deg": 3.0},
            "anchor": {"level": "Chest", "edge": "top"},
            "ring": "torso",
            "layer": 0,
            "locked": False,
            "shift_m": [0.0, 0.01],
        },
        {"piece": "Pocket_C", "point": "Chest_Front", "wrap": "roll", "over": {"piece": "FrontPanel", "point": "A32"}},
    ]
    derived = [{"id": "Pocket_C", "source": "Pocket", "op": "copy"}]
    r = c.put("/api/jobs/male_shirt/placements", json={"placements": pls, "derived_pieces": derived})
    assert r.status_code == 200, r.text
    got = c.get("/api/jobs/male_shirt/placements").json()
    assert got["placements"] == pls and got["derived_pieces"] == derived

"""Web backend API tests (FastAPI TestClient, real engines, real SMPL-X files; skipped if models are absent)."""

from __future__ import annotations

import dataclasses
import json
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from web.backend.app import create_app
from web.backend.config import AppConfig

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not (ROOT / "models/smplx/SMPLX_MALE.npz").is_file(), reason="models/smplx not present")


def make_cfg(tmp: Path) -> AppConfig:
    return dataclasses.replace(AppConfig.default(), outputs_dir=tmp / "outputs")


def wait_done(c: TestClient, bid: str, timeout: float = 90) -> dict[str, Any]:
    t0 = time.time()
    while time.time() - t0 < timeout:
        st = c.get(f"/api/builds/{bid}").json()
        if st["status"] in ("done", "failed"):
            return st  # type: ignore[no-any-return]
        time.sleep(0.2)
    raise AssertionError("build did not finish")


@pytest.fixture(scope="module")
def client(tmp_path_factory: pytest.TempPathFactory) -> Iterator[TestClient]:
    with TestClient(create_app(make_cfg(tmp_path_factory.mktemp("web")))) as c:
        yield c


@pytest.fixture(scope="module")
def shirt(client: TestClient) -> dict[str, Any]:
    r = client.post("/api/jobs/male_shirt/build")
    assert r.status_code == 200
    st = wait_done(client, r.json()["build_id"])
    assert st["status"] == "done", st
    return st


def test_jobs_lists_the_three_jobs(client: TestClient) -> None:
    names = [j["name"] for j in client.get("/api/jobs").json()]
    assert names == ["aldrich_skirt", "male_shirt", "trousers"]
    j = {x["name"]: x for x in client.get("/api/jobs").json()}["aldrich_skirt"]
    assert j["pieces"] == ["A - Skirt Back", "A - Skirt Front"] and j["body"].startswith("female")


def test_build_result_has_pieces_avatar_points_and_report(shirt: dict[str, Any]) -> None:
    r = shirt["result"]
    assert len(r["pieces"]) == 17 and all(p["status"] == "ok" for p in r["pieces"])
    fp = next(p for p in r["pieces"] if p["name"] == "FrontPanel")
    assert fp["width_cm"] == pytest.approx(37.25, abs=0.01) and fp["height_cm"] == pytest.approx(74.40, abs=0.01)
    assert len(fp["named_points"]) > 5 and fp["default_reference"] == pytest.approx([0.18625, 0.372], abs=1e-4)
    assert len(r["avatar"]["points"]) == 80 and len(r["avatar"]["volumes"]) == 17
    assert r["avatar"]["bundle"]["units"] == "m"
    assert r["avatar"]["measurements"]["measurements"]["height"]["post_error_cm"] == pytest.approx(0.0, abs=0.1)


def test_files_are_served_and_traversal_is_refused(client: TestClient, shirt: dict[str, Any]) -> None:
    r = client.get(f"/api/builds/{shirt['id']}/files/{shirt['result']['avatar']['mesh']}")
    assert r.status_code == 200 and r.text.count("\nv ") > 10000
    r = client.get(f"/api/builds/{shirt['id']}/files/pieces/FrontPanel_m.obj")
    assert r.status_code == 200 and "units: metres" in r.text
    bad = client.get(f"/api/builds/{shirt['id']}/files/..%2F..%2Fbuild.json")
    assert bad.status_code in (404, 422) and "error" in bad.json()
    assert client.get(f"/api/builds/{shirt['id']}/files/../../../pyproject.toml").status_code in (404, 422)


def test_build_is_cached_by_content_hash(client: TestClient, shirt: dict[str, Any], tmp_path: Path) -> None:
    again = client.post("/api/jobs/male_shirt/build").json()
    assert again["build_id"] == shirt["id"] and again["status"] == "done"
    # a fresh app on the same output folder reuses the files without rebuilding
    with TestClient(
        create_app(dataclasses.replace(AppConfig.default(), outputs_dir=client.app.state.cfg.outputs_dir))
    ) as c2:  # type: ignore[attr-defined]
        r = c2.post("/api/jobs/male_shirt/build").json()
        assert r["build_id"] == shirt["id"] and r["cached"] is True and r["status"] == "done"


def test_align_places_the_front_panel_at_chest_front(client: TestClient, shirt: dict[str, Any]) -> None:
    pts = {p["name"]: p for p in shirt["result"]["avatar"]["points"]}
    body = {"build": shirt["id"], "placements": [{"piece": "FrontPanel", "point": "Chest_Front"}]}
    r = client.post("/api/align", json=body)
    assert r.status_code == 200, r.text
    doc = r.json()
    pl = doc["placements"][0]
    # pushed out from the point where the flat panel would touch the body, never in; the answer says it is valid
    off = np.array(pl["reference_position_m"]) - np.array(pts["Chest_Front"]["position_m"])
    assert off @ np.array(pts["Chest_Front"]["normal"]) >= -1e-6 and doc["validity"]["valid"] is True
    assert pl["bbox_cm"] == {"width": pytest.approx(37.25, abs=0.01), "height": pytest.approx(74.40, abs=0.01)}
    m = np.array(pl["matrix"]).reshape(4, 4)
    assert m[:3, 2] == pytest.approx(pts["Chest_Front"]["normal"], abs=1e-5)  # the piece faces along the point normal
    # many placements at once, with wrap
    two = {
        "build": shirt["id"],
        "placements": [
            {"piece": "FrontPanel", "point": "Chest_Front", "wrap": "around"},
            {"piece": "BackPanel", "point": "Chest_Back", "mirror": True, "offset_m": 0.05},
        ],
    }
    out = client.post("/api/align", json=two).json()
    assert [p["piece"] for p in out["placements"]] == ["FrontPanel", "BackPanel"]
    assert out["placements"][0]["wrapped_vertices"] is not None and out["placements"][1]["mirrored"] is True


def test_errors_are_structured_json_without_stack_traces(client: TestClient, shirt: dict[str, Any]) -> None:
    cases = [
        (client.post("/api/jobs/nope/build"), 404, "job_not_found"),
        (client.get("/api/builds/nope-123"), 404, "build_not_found"),
        (
            client.post(
                "/api/align", json={"build": shirt["id"], "placements": [{"piece": "X", "point": "Chest_Front"}]}
            ),
            422,
            "unknown_piece",
        ),
        (
            client.post(
                "/api/align", json={"build": shirt["id"], "placements": [{"piece": "Cuff", "point": "Nowhere"}]}
            ),
            422,
            "unknown_point",
        ),
        (
            client.post(
                "/api/align",
                json={"build": shirt["id"], "placements": [{"piece": "Cuff", "point": "Chest_Front", "wrap": "x"}]},
            ),
            422,
            "invalid_document",
        ),
        (client.get("/api/nothing"), 404, "not_found"),
        (client.get("/api/export/" + shirt["id"] + "?format=stl"), 422, "invalid_request"),
    ]
    for resp, status, code in cases:
        assert resp.status_code == status, resp.text
        err = resp.json()["error"]
        assert err["code"] == code and err["message"] and "Traceback" not in resp.text
    assert (
        client.post(
            "/api/align", json={"build": shirt["id"], "placements": [{"piece": "X", "point": "Chest_Front"}]}
        ).json()["error"]["piece"]
        == "X"
    )


def test_placements_save_and_load(client: TestClient, shirt: dict[str, Any]) -> None:
    assert client.get("/api/jobs/male_shirt/placements").json()["placements"] == []
    spec = {"placements": [{"piece": "Cuff", "point": "Hand_Outer_L", "rotation_deg": 15, "wrap": "around"}]}
    saved = client.put("/api/jobs/male_shirt/placements", json=spec).json()
    assert saved["schema_version"] == 2 and saved == client.get("/api/jobs/male_shirt/placements").json()
    bad = client.put("/api/jobs/male_shirt/placements", json={"placements": [{"piece": "Cuff"}]})
    assert bad.status_code == 422 and bad.json()["error"]["code"] == "invalid_document"


def test_export_obj_and_glb(client: TestClient, shirt: dict[str, Any]) -> None:
    obj = client.get(f"/api/export/{shirt['id']}?format=obj")
    assert obj.status_code == 200 and "\ng avatar" in obj.text and "\ng Cuff" in obj.text
    glb = client.get(f"/api/export/{shirt['id']}?format=glb")
    assert glb.status_code == 200 and glb.content[:4] == b"glTF"


def test_a_failed_piece_does_not_fail_the_build(tmp_path: Path) -> None:
    cfg = make_cfg(tmp_path)
    jobs = tmp_path / "jobs"
    jobs.mkdir()
    (jobs / "jacket.json").write_text(
        json.dumps(
            {"name": "jacket", "sm2d": "jacket3_40-146.sm2d", "smis": "male_shirt.smis", "body": "male_173.json"}
        )
    )
    with TestClient(create_app(dataclasses.replace(cfg, jobs_dir=jobs))) as c:
        bid = c.post("/api/jobs/jacket/build").json()["build_id"]
        st = wait_done(c, bid)
        assert st["status"] == "done"
        pieces = st["result"]["pieces"]
        assert pieces and all(p["status"] == "failed" and p["reason"] for p in pieces)
        assert {p["code"] for p in pieces} <= {"unsupported_tool", "missing_measurement"}
        assert len(st["result"]["avatar"]["points"]) == 80  # the avatar was still built


def test_missing_input_file_is_a_structured_error(tmp_path: Path) -> None:
    cfg = make_cfg(tmp_path)
    jobs = tmp_path / "jobs"
    jobs.mkdir()
    (jobs / "broken.json").write_text(
        json.dumps({"name": "broken", "sm2d": "nope.sm2d", "smis": "x.smis", "body": "male_173.json"})
    )
    with TestClient(create_app(dataclasses.replace(cfg, jobs_dir=jobs))) as c:
        r = c.post("/api/jobs/broken/build")
        assert r.status_code == 422 and r.json()["error"]["code"] == "job_file_missing"


# half pieces: clone & mirror, pairs ---------------------------------------------------------------------------------
CF = {"type": "edge", "from_point": "A33", "to_point": "A32"}
DERIVED = [{"id": "FrontPanel_L", "source": "FrontPanel", "op": "mirror", "axis": CF}]


def test_clone_returns_the_derived_piece_and_its_files(client: TestClient, shirt: dict[str, Any]) -> None:
    r = client.post("/api/clone", json={"build": shirt["id"], "source": "FrontPanel", "axis": CF, "id": "FrontPanel_L"})
    assert r.status_code == 200, r.text
    p = r.json()
    assert p["name"] == "FrontPanel_L" and p["mirror_of"] == "FrontPanel" and p["axis"] == CF
    assert p["width_cm"] == pytest.approx(37.25, abs=0.01) and p["height_cm"] == pytest.approx(74.40, abs=0.01)
    assert p["obj"] == "derived/FrontPanel_L_m.obj" and p["bundle"] == "derived/FrontPanel_L.piece.json"
    assert any(q["name"] == "A33" for q in p["named_points"]) and p["suggested_stitch"]["b"]["piece"] == "FrontPanel_L"
    obj = client.get(f"/api/builds/{shirt['id']}/files/{p['obj']}")
    assert obj.status_code == 200 and "mirror of FrontPanel" in obj.text
    bundle = client.get(f"/api/builds/{shirt['id']}/files/{p['bundle']}").json()
    assert bundle["mirror_of"] == "FrontPanel" and len(bundle["outline_m"]) > 10


def test_pair_and_align_with_derived_pieces(client: TestClient, shirt: dict[str, Any]) -> None:
    base = {"piece": "FrontPanel", "point": "Chest_Front", "reference": "A33", "wrap": "roll"}
    r = client.post(
        "/api/pair",
        json={
            "build": shirt["id"],
            "base": base,
            "derived_id": "FrontPanel_L",
            "mode": "same_anchor",
            "derived_pieces": DERIVED,
        },
    )
    assert r.status_code == 200, r.text
    pair = r.json()["placements"]
    assert [p["piece"] for p in pair] == ["FrontPanel", "FrontPanel_L"] and pair[0]["reference"] == pair[1]["reference"]
    a = client.post("/api/align", json={"build": shirt["id"], "placements": pair, "derived_pieces": DERIVED})
    assert a.status_code == 200, a.text
    doc = a.json()
    p0, p1 = doc["placements"]
    r0, r1 = p0["reference_position_m"], p1["reference_position_m"]  # the same anchor, half a gap off the centre each
    assert abs(r0[0] + r1[0]) < 1e-5 and abs(r0[1] - r1[1]) < 1e-5 and doc["validity"]["valid"] is True
    assert {p0["extends"], p1["extends"]} == {"left", "right"}
    assert (
        p0["bbox_cm"]
        == p1["bbox_cm"]
        == {"width": pytest.approx(37.25, abs=0.01), "height": pytest.approx(74.40, abs=0.01)}
    )
    assert doc["derived_pieces"][0]["id"] == "FrontPanel_L" and doc["derived_pieces"][0]["obj"] == "FrontPanel_L_m.obj"
    assert doc["schema_version"] == 4


def test_half_piece_errors_have_specific_codes(client: TestClient, shirt: dict[str, Any]) -> None:
    bid = shirt["id"]
    bad_ref = client.post(
        "/api/pair",
        json={
            "build": bid,
            "base": {"piece": "FrontPanel", "point": "Chest_Front"},
            "derived_id": "FrontPanel_L",
            "mode": "same_anchor",
            "derived_pieces": DERIVED,
        },
    )
    # an off-axis reference is replaced by the named point on the mirror axis nearest in height (alignment_pairs.md)
    assert bad_ref.status_code == 200, bad_ref.text
    adjusted = bad_ref.json()
    assert any(w.startswith("REFERENCE_ADJUSTED") for w in adjusted["warnings"])
    assert (
        {p["reference"] for p in adjusted["placements"]} == {adjusted["placements"][0]["reference"]} != {"bbox_centre"}
    )
    no_mirror = client.post(
        "/api/pair",
        json={
            "build": bid,
            "base": {"piece": "FrontPanel", "point": "Nowhere_L"},
            "derived_id": "FrontPanel_L",
            "mode": "mirror_anchor",
            "derived_pieces": DERIVED,
        },
    )
    assert no_mirror.status_code == 422 and no_mirror.json()["error"]["code"] == "no_mirrored_point"
    dd = [*DERIVED, {"id": "FP_LL", "source": "FrontPanel_L", "op": "mirror", "axis": {"type": "bbox_center"}}]
    r = client.post("/api/align", json={"build": bid, "placements": [], "derived_pieces": dd})
    assert r.status_code == 422 and r.json()["error"]["code"] == "derived_of_derived"
    dup = client.post("/api/clone", json={"build": bid, "source": "FrontPanel", "axis": CF, "id": "BackPanel"})
    assert dup.status_code == 422 and dup.json()["error"]["code"] == "duplicate_piece"
    unk = client.post(
        "/api/clone",
        json={
            "build": bid,
            "source": "FrontPanel",
            "axis": {"type": "edge", "from_point": "A33", "to_point": "zzz"},
            "id": "X",
        },
    )
    assert unk.status_code == 422 and unk.json()["error"]["code"] == "unknown_named_point"
    assert "Traceback" not in unk.text


def test_derived_pieces_persist_and_are_exported(client: TestClient, shirt: dict[str, Any]) -> None:
    pair = [
        {"piece": "FrontPanel", "point": "Chest_Front", "reference": "A33"},
        {"piece": "FrontPanel_L", "point": "Chest_Front", "reference": "A33"},
    ]
    saved = client.put("/api/jobs/male_shirt/placements", json={"placements": pair, "derived_pieces": DERIVED}).json()
    assert saved["derived_pieces"] == DERIVED and saved["schema_version"] == 2
    assert client.get("/api/jobs/male_shirt/placements").json()["derived_pieces"] == DERIVED
    obj = client.get(f"/api/export/{shirt['id']}?format=obj")
    assert obj.status_code == 200 and "\ng FrontPanel_L" in obj.text and "\ng FrontPanel\n" in obj.text


def test_arrange_endpoint_returns_the_arrangement(client: TestClient, shirt: dict[str, Any]) -> None:
    body = {
        "build": shirt["id"],
        "placements": [
            {"piece": "BackPanel", "point": "Chest_Back", "reference": "A24", "wrap": "roll"},
            {"piece": "Pocket", "point": "Chest_FrontSide_L", "layer": 1, "wrap": "roll"},
        ],
        "arrange_options": {"gap_piece": 0.04},
    }
    r = client.post("/api/arrange", json=body)
    assert r.status_code == 200, r.text
    doc = r.json()
    assert doc["schema_version"] == 4 and doc["arrangement"]["options"]["gap_piece"] == 0.04
    assert doc["validity"]["valid"] is True and all("size_check" in p for p in doc["placements"])
    assert doc["arrangement"]["rings"][0]["ring"] == "torso"
    bad = client.post(
        "/api/arrange",
        json={**body, "placements": [{"piece": "BackPanel", "point": "Chest_Back", "ring": "tail", "wrap": "roll"}]},
    )
    assert bad.status_code == 422 and bad.json()["error"]["code"] == "ring_not_found"


def test_clone_copy_pair_rejection_and_job_round_trip(client: TestClient, shirt: dict[str, Any]) -> None:
    bid = shirt["id"]
    r = client.post("/api/clone", json={"build": bid, "source": "Pocket", "id": "Pocket_copy", "op": "copy"})
    assert r.status_code == 200, r.text
    p = r.json()
    assert p["id"] == p["name"] == "Pocket_copy" and p["op"] == "copy" and p["source"] == "Pocket"
    assert p["clone_of"] == "Pocket" and p["mirror_of"] is None and "axis" not in p and p["suggested_stitch"] is None
    assert "copy of Pocket" in client.get(f"/api/builds/{bid}/files/{p['obj']}").text
    copy_ = {"id": "Pocket_copy", "source": "Pocket", "op": "copy"}
    bad = client.post("/api/clone", json={"build": bid, "source": "Pocket", "id": "P2", "op": "copy", "axis": CF})
    assert bad.status_code == 422
    bad_id = client.post("/api/clone", json={"build": bid, "source": "Pocket", "id": "a b", "op": "copy"})
    assert bad_id.status_code == 422 and bad_id.json()["error"]["code"] == "invalid_document"
    pair = client.post(
        "/api/pair",
        json={
            "build": bid,
            "base": {"piece": "Pocket", "point": "Chest_FrontSide_R"},
            "derived_id": "Pocket_copy",
            "mode": "mirror_anchor",
            "derived_pieces": [copy_],
        },
    )
    assert pair.status_code == 422 and pair.json()["error"]["code"] == "not_a_mirror_clone", pair.text
    both = [*DERIVED, copy_, {"id": "FrontPanel_L_copy", "source": "FrontPanel_L", "op": "copy"}]
    pls = [{"piece": "Pocket", "point": "Chest_FrontSide_R"}, {"piece": "Pocket_copy", "point": "Chest_FrontSide_L"}]
    saved = client.put("/api/jobs/male_shirt/placements", json={"placements": pls, "derived_pieces": both}).json()
    assert saved["derived_pieces"] == both == client.get("/api/jobs/male_shirt/placements").json()["derived_pieces"]

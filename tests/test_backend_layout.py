"""/api/align and /api/arrange: validity, size checks, incremental placement and the compact mesh transport.

Runs the real backend on a male_shirt build with the synthetic avatar (no SMPL-X files needed).
"""

from __future__ import annotations

import base64
import dataclasses
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from tests.web_build import prebuild
from web.backend.app import create_app
from web.backend.config import AppConfig

CF = {"type": "edge", "from_point": "A33", "to_point": "A32"}
DERIVED = [{"id": "FrontPanel_L", "source": "FrontPanel", "op": "mirror", "axis": CF}]
THREE = [
    {"piece": "FrontPanel", "point": "Chest_Front", "reference": "A32", "wrap": "roll"},
    {"piece": "BackPanel", "point": "Chest_Back", "reference": "A24", "wrap": "roll"},
    {"piece": "Yoke", "point": "Chest_Back", "wrap": "roll"},
]


@pytest.fixture(scope="module")
def client(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[TestClient, str]]:
    out = tmp_path_factory.mktemp("outputs")
    bid = prebuild(out)
    cfg = dataclasses.replace(AppConfig.default(), outputs_dir=out)
    with TestClient(create_app(cfg)) as c:
        started = c.post("/api/jobs/male_shirt/build").json()
        assert started["build_id"] == bid and started["status"] == "done"
        yield c, bid


def post(c: TestClient, url: str, body: dict[str, Any]) -> dict[str, Any]:
    r = c.post(url, json=body)
    assert r.status_code == 200, r.text
    return r.json()  # type: ignore[no-any-return]


def test_align_returns_validity_size_check_and_solved(client: tuple[TestClient, str]) -> None:
    c, bid = client
    doc = post(c, "/api/align", {"build": bid, "placements": THREE})
    assert doc["schema_version"] == 4
    v = doc["validity"]
    assert v["valid"] is True and v["size_ok"] and v["clearance_ok"] and v["gaps_ok"] and v["violations"] == []
    assert v["min_distance_to_avatar_m"] >= 0.029 and v["min_distance_between_pieces_m"] is not None
    for p in doc["placements"]:
        sc = p["size_check"]
        assert (
            max(
                sc["max_edge_change_pct"],
                sc["perimeter_change_pct"],
                sc["area_change_pct"],
                sc["named_point_change_pct"],
            )
            <= 0.05
        )
        assert p["changes_piece_size"] is False and p["solved"]["radius_m"] > 0
    # incremental: the first two keep their place exactly, only the changed yoke is solved again
    again = [{**pl, "solved": d["solved"]} for pl, d in zip(THREE, doc["placements"], strict=True)]
    again[2] = {**THREE[2], "rotation_deg": 10.0}
    doc2 = post(c, "/api/align", {"build": bid, "placements": again})
    for a, b in zip(doc["placements"][:2], doc2["placements"][:2], strict=True):
        assert a["wrapped_vertices"] == b["wrapped_vertices"]
    assert doc2["validity"]["valid"] is True


def test_compact_transport_carries_the_same_mesh(client: tuple[TestClient, str]) -> None:
    c, bid = client
    plain = post(c, "/api/align", {"build": bid, "placements": THREE[:1]})["placements"][0]
    doc = post(c, "/api/align", {"build": bid, "placements": THREE[:1], "compact": True})
    p = doc["placements"][0]
    assert p["wrapped_vertices"] is None and p["wrapped_faces"] is None
    m = p["wrapped_mesh"]
    v = np.frombuffer(base64.b64decode(m["vertices"]), dtype="<f4").reshape(-1, 3)
    f = np.frombuffer(base64.b64decode(m["faces"]), dtype="<u4").reshape(-1, 3)
    assert np.abs(v - np.array(plain["wrapped_vertices"])).max() < 1e-6 and np.array_equal(
        f, np.array(plain["wrapped_faces"])
    )
    known = post(
        c, "/api/align", {"build": bid, "placements": THREE[:1], "compact": True, "known_meshes": [m["mesh_id"]]}
    )
    assert "faces" not in known["placements"][0]["wrapped_mesh"]  # faces travel once per mesh


def test_arrange_returns_validity_and_the_arrangement(client: tuple[TestClient, str]) -> None:
    c, bid = client
    body = {
        "build": bid,
        "placements": [
            {"piece": "FrontPanel", "point": "Chest_Front", "reference": "A32", "wrap": "roll"},
            {"piece": "FrontPanel_L", "point": "Chest_Front", "reference": "A32", "wrap": "roll"},
            {"piece": "Pocket", "point": "Chest_FrontSide_L", "wrap": "roll"},
        ],
        "derived_pieces": DERIVED,
        "arrange_options": {"gap_piece": 0.04},
    }
    doc = post(c, "/api/arrange", body)
    assert doc["validity"]["valid"] is True and doc["arrangement"]["options"]["gap_piece"] == 0.04
    assert doc["validity"]["min_distance_between_pieces_m"] >= 0.04 - 0.0005
    assert all("size_check" in p and p["changes_piece_size"] is False for p in doc["placements"])
    bad = c.post(
        "/api/arrange",
        json={**body, "placements": [{"piece": "BackPanel", "point": "Chest_Back", "wrap": "roll", "ring": "tail"}]},
    )
    assert bad.status_code == 422 and bad.json()["error"]["code"] == "ring_not_found"


def test_invalid_layouts_are_reported_and_stretching_is_refused(client: tuple[TestClient, str]) -> None:
    c, bid = client
    doc = post(c, "/api/align", {"build": bid, "placements": THREE[:2], "arrange_options": {"max_standoff": 0.005}})
    v = doc["validity"]
    assert v["valid"] is False and v["violations"] and all(x["pieces"] and x["limit"] for x in v["violations"])
    r = c.post(
        "/api/align",
        json={"build": bid, "placements": [{"piece": "BackPanel", "point": "Chest_Back", "wrap": "body_slice"}]},
    )
    assert r.status_code == 422 and r.json()["error"]["code"] == "size_changed" and "Traceback" not in r.text

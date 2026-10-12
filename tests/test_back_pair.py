"""Mirror pairs placed one at a time (the web app's incremental ``align``) on the synthetic body.

The shirt's front and back halves are placed piece by piece in several orders, ``solved`` carried over as the web app
does. Every order must give one rigid, symmetric pair per panel (side by side, never stacked), a concentric garment
(front and back on one ring radius up to a layer gap) and front and back on the same drafting chest line.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from alignment_engine import AlignmentEngine
from alignment_engine.loaders import read_avatar
from alignment_engine.validity import arc, arc_overlap
from alignment_engine.verify import AvatarSurface
from tests.test_layout import CLEAR, GAP, body_lower_bound, pair_lower_bound

CF = {"type": "edge", "from_point": "A33", "to_point": "A32"}
DERIVED = [
    {"id": "FrontPanel_L", "source": "FrontPanel", "op": "mirror", "axis": CF},
    {"id": "BackPanel_L", "source": "BackPanel", "op": "mirror", "axis": {"type": "x", "value_m": 0.0}},
]
PAIRS = (("FrontPanel", "FrontPanel_L"), ("BackPanel", "BackPanel_L"))
CHEST = {"FrontPanel": "A1", "FrontPanel_L": "A1", "BackPanel": "BustLine", "BackPanel_L": "BustLine"}  # y 0.493
GAP_LAYER = 0.03


def pl(piece: str, point: str, ref: str | None = None) -> dict[str, Any]:
    d: dict[str, Any] = {"piece": piece, "point": point, "wrap": "roll"}
    if ref:
        d["reference"] = ref
    return d


def front(point: str = "Chest_Front") -> list[dict[str, Any]]:
    return [pl("FrontPanel", point, "A32"), pl("FrontPanel_L", point, "A32")]


SCENARIOS = {
    "A_tests_layout": front() + [pl("BackPanel", "Chest_Back", "A24"), pl("BackPanel_L", "Chest_Back", "A24")],
    "B_back_bbox_centre": front() + [pl("BackPanel", "Chest_Back"), pl("BackPanel_L", "Chest_Back")],
    "C_front_on_neck": front("Neck_Front") + [pl("BackPanel", "Chest_Back", "A24"), pl("BackPanel_L", "Chest_Back", "A24")],
    "D_back_on_sides": front()
    + [pl("BackPanel", "Chest_BackSide_R", "A24"), pl("BackPanel_L", "Chest_BackSide_L", "A24")],
    "F_back_first": [pl("BackPanel", "Chest_Back", "A24"), pl("BackPanel_L", "Chest_Back", "A24"), *front()],
    "clone_first": [
        pl("FrontPanel_L", "Chest_Front", "A32"),
        pl("FrontPanel", "Chest_Front", "A32"),
        pl("BackPanel_L", "Chest_Back", "A24"),
        pl("BackPanel", "Chest_Back", "A24"),
    ],
}


@pytest.fixture(scope="module")
def ctx(shirt_pieces: Path, refine_cache: Path, tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    from tests.synthetic_avatar import build_avatar

    d = tmp_path_factory.mktemp("back_pair")
    av = build_avatar(d / "avatar")
    a = read_avatar(av)
    return {"avatar": av, "pieces": shirt_pieces, "derived": d / "derived", "cache": refine_cache, "av": a,
            "surface": AvatarSurface(a.vertices, a.faces)}


def call(ctx: dict[str, Any], method: str, pls: list[dict[str, Any]], eng: AlignmentEngine | None = None) -> Any:
    text = json.dumps({"schema_version": 4, "placements": pls, "derived_pieces": DERIVED})
    return getattr(eng or AlignmentEngine(), method)(
        ctx["avatar"], ctx["pieces"], text, derived_dir=ctx["derived"], cache_dir=ctx["cache"]
    )


def one_by_one(ctx: dict[str, Any], pls: list[dict[str, Any]], eng: AlignmentEngine | None = None) -> Any:
    """Place the pieces one at a time, carrying every returned ``solved`` block (all of them: a pair may move both)."""
    eng = eng or AlignmentEngine()
    done: list[dict[str, Any]] = []
    res = None
    for new in pls:
        res = call(ctx, "align", [*done, new], eng)
        docs = {d["piece"]: d for d in res.doc["placements"]}
        done = [{**p, "solved": docs[p["piece"]]["solved"]} for p in [*done, new]]
    return res


@pytest.fixture(scope="module")
def results(ctx: dict[str, Any]) -> dict[str, Any]:
    return {k: one_by_one(ctx, v) for k, v in SCENARIOS.items()}


def by_name(res: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    return {p.name: p for p in res.placed}, {d["piece"]: d for d in res.doc["placements"]}


def chest_y(res: Any, name: str) -> float:
    placed, docs = by_name(res)
    q = next(q for q in docs[name]["world_named_points"] if q["name"] == CHEST[name])
    return float(q["world_m"][1])


@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_pairs_are_rigid_and_side_by_side(ctx: dict[str, Any], results: dict[str, Any], scenario: str) -> None:
    res = results[scenario]
    placed, docs = by_name(res)
    for a, b in PAIRS:
        ra, rb = docs[a]["arrange"], docs[b]["arrange"]
        (sa, ta), (sb, tb) = ra["shift_m"], rb["shift_m"]
        assert abs(ra["radius_m"] - rb["radius_m"]) < 0.001, (scenario, a)
        assert abs(ta - tb) < 0.001 and abs(sa + sb) < 0.001, (scenario, a, ra["shift_m"], rb["shift_m"])
        assert ra["layer"] == rb["layer"] and ra["pair_with"] == b and rb["pair_with"] == a
        c, axis = np.array([0.0, 1.0, 0.0]), np.array([0.0, 1.0, 0.0])
        assert arc_overlap(arc(placed[a].vertices, c, axis), arc(placed[b].vertices, c, axis)) == 0.0, (scenario, a)
        assert pair_lower_bound(placed[a], placed[b]) >= GAP - 0.001, (scenario, a)


@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_garment_is_concentric_and_on_one_chest_line(results: dict[str, Any], scenario: str) -> None:
    res = results[scenario]
    _, docs = by_name(res)
    radii = {n: docs[n]["arrange"]["radius_m"] for n in CHEST}
    assert {docs[n]["arrange"]["ring"] for n in CHEST} == {"torso"}
    assert abs(radii["FrontPanel"] - radii["BackPanel"]) <= GAP_LAYER + 0.005, (scenario, radii)
    assert max(radii.values()) <= 0.29, (scenario, radii)  # never pushed out layer by layer
    assert abs(chest_y(res, "FrontPanel") - chest_y(res, "BackPanel")) < 0.005, scenario
    assert not [w for w in res.warnings if w.startswith("NOT_CONCENTRIC")]


@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_valid_sizes_and_clearances(ctx: dict[str, Any], results: dict[str, Any], scenario: str) -> None:
    res = results[scenario]
    v = res.doc["validity"]
    assert v["valid"] and v["size_ok"] and v["pairs_ok"], (scenario, v["violations"])
    assert v["min_distance_to_avatar_m"] >= CLEAR - 0.0005 and v["min_distance_between_pieces_m"] >= GAP - 0.0005
    for p in res.placed:
        assert body_lower_bound(ctx["surface"], p) >= CLEAR - 0.001, (scenario, p.name)
    for d in res.doc["placements"]:
        assert d["changes_piece_size"] is False and d["size_check"]["area_change_pct"] < 0.05


def test_incremental_matches_arrange(ctx: dict[str, Any], results: dict[str, Any]) -> None:
    inc = by_name(results["A_tests_layout"])[1]
    arr_res = call(ctx, "arrange", SCENARIOS["A_tests_layout"])
    arr = by_name(arr_res)[1]
    assert arr_res.doc["validity"]["valid"]
    for n in CHEST:
        assert abs(inc[n]["arrange"]["radius_m"] - arr[n]["arrange"]["radius_m"]) <= GAP_LAYER + 0.005, n
    assert abs(chest_y(arr_res, "FrontPanel") - chest_y(arr_res, "BackPanel")) < 0.005


def test_deterministic(ctx: dict[str, Any], results: dict[str, Any]) -> None:
    again = one_by_one(ctx, SCENARIOS["A_tests_layout"])
    first = results["A_tests_layout"]
    assert [d["solved"] for d in again.doc["placements"]] == [d["solved"] for d in first.doc["placements"]]


def test_validity_catches_a_stacked_pair(ctx: dict[str, Any]) -> None:
    """The old failure: the clone one layer out over its source. Locked, so nothing re-solves it."""
    pls = [
        {**pl("FrontPanel", "Chest_Front", "A32"), "locked": True, "solved": {"radius_m": 0.21, "shift_m": [0.0, 0.0]}},
        {**pl("FrontPanel_L", "Chest_Front", "A32"), "locked": True, "solved": {"radius_m": 0.24, "shift_m": [0.05, 0.0]}},
    ]
    res = call(ctx, "align", pls)
    v = res.doc["validity"]
    kinds = {x["kind"] for x in v["violations"]}
    assert v["valid"] is False and v["pairs_ok"] is False and {"PAIR_ASYMMETRY", "PAIR_STACKED"} <= kinds

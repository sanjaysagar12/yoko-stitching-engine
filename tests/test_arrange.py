"""Auto-arrange: what the arrangement reports besides collisions and sizes (those are in ``test_layout.py``).

True ease against the display standoff, locked pieces, a pocket layered over its panel, seam candidates, report fields.
Runs on the synthetic body and, when ``models/smplx`` exists, on SMPL-X.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy.spatial import cKDTree

from alignment_engine import AlignmentEngine
from alignment_engine.verify import surface_samples

CF = {"type": "edge", "from_point": "A33", "to_point": "A32"}
DERIVED = [
    {"id": "FrontPanel_L", "source": "FrontPanel", "op": "mirror", "axis": CF},
    {"id": "BackPanel_L", "source": "BackPanel", "op": "mirror", "axis": {"type": "x", "value_m": 0.0}},
]
TORSO = ["FrontPanel", "FrontPanel_L", "BackPanel", "BackPanel_L"]


def placements(extra: dict[str, dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    pl: list[dict[str, Any]] = [
        {"piece": "FrontPanel", "point": "Chest_Front", "reference": "A32", "wrap": "roll"},
        {"piece": "FrontPanel_L", "point": "Chest_Front", "reference": "A32", "wrap": "roll"},
        {"piece": "BackPanel", "point": "Chest_Back", "reference": "A24", "wrap": "roll"},
        {"piece": "BackPanel_L", "point": "Chest_Back", "reference": "A24", "wrap": "roll"},
        {"piece": "Pocket", "point": "Chest_FrontSide_L", "wrap": "roll"},
    ]
    for p in pl:
        p.update((extra or {}).get(p["piece"], {}))
    return pl


def spec(pl: list[dict[str, Any]] | None = None, **opts: Any) -> str:
    doc: dict[str, Any] = {
        "schema_version": 4,
        "placements": pl if pl is not None else placements(),
        "derived_pieces": DERIVED,
    }
    if opts:
        doc["arrange_options"] = opts
    return json.dumps(doc)


@pytest.fixture(scope="module")
def ctx(
    avatar: dict[str, Any], shirt_pieces: Path, refine_cache: Path, tmp_path_factory: pytest.TempPathFactory
) -> dict[str, Any]:
    d = tmp_path_factory.mktemp(f"arrange_{avatar['kind']}")
    return {
        "avatar": avatar["dir"],
        "pieces": shirt_pieces,
        "derived": d / "derived",
        "cache": refine_cache,
        "engine": AlignmentEngine(),
    }


def arrange(ctx: dict[str, Any], text: str) -> Any:
    return ctx["engine"].arrange(ctx["avatar"], ctx["pieces"], text, derived_dir=ctx["derived"], cache_dir=ctx["cache"])


@pytest.fixture(scope="module")
def run(ctx: dict[str, Any]) -> Any:
    return arrange(ctx, spec())


def by_name(res: Any) -> dict[str, Any]:
    return {p.name: p for p in res.placed}


def pair_distance(a: Any, b: Any) -> float:
    sa, sb = surface_samples(a.vertices, a.faces, 0.003), surface_samples(b.vertices, b.faces, 0.003)
    return float(cKDTree(sb).query(sa)[0].min())


def test_5_true_ease_is_not_the_standoff(ctx: dict[str, Any], run: Any) -> None:
    other = arrange(ctx, spec(clearance_body=0.04, gap_piece=0.045))
    ra = {r["ring"]: r for r in run.doc["arrangement"]["rings"]}["torso"]
    rb = {r["ring"]: r for r in other.doc["arrangement"]["rings"]}["torso"]
    assert ra["true_ease_m"] == pytest.approx(ra["garment_loop_m"] - ra["body_loop_m"], abs=1e-9)
    assert ra["true_ease_m"] == rb["true_ease_m"] and ra["body_loop_m"] == rb["body_loop_m"]  # no option changes it
    assert rb["standoff_used_m"] > ra["standoff_used_m"]  # the display standoff does
    # the shirt is much wider than the body: 2 x 37.25 + 2 x 34.75 = 144 cm of garment for the torso loop
    assert 1.2 < ra["garment_loop_m"] < 1.6 and ra["true_ease_m"] > 0.3  # the binding row depends on the body
    assert "Pocket" not in ra["pieces"] and set(ra["pieces"]) == set(TORSO)  # the pocket lies over the ring


def test_6_locked_pieces_do_not_move_and_are_obstacles(ctx: dict[str, Any], run: Any) -> None:
    solved = {d["piece"]: d["solved"] for d in run.doc["placements"]}
    locked = placements({"BackPanel": {"locked": True, "solved": solved["BackPanel"]}})
    first = arrange(ctx, spec(locked))
    second = arrange(ctx, spec(locked, gap_piece=0.05))
    a, b = by_name(first)["BackPanel"], by_name(second)["BackPanel"]
    assert np.abs(a.vertices - b.vertices).max() < 1e-9  # the locked piece keeps its place
    assert np.abs(a.vertices - by_name(run)["BackPanel"].vertices).max() < 1e-9
    assert second.doc["validity"]["valid"] is True, second.doc["validity"]["violations"]
    for n in ("FrontPanel", "FrontPanel_L", "BackPanel_L"):  # an obstacle: the others keep the larger gap from it
        assert pair_distance(by_name(second)[n], b) >= 0.05 - 0.001, n
    assert first.doc["placements"][2]["arrange"]["locked"] is True
    assert run.doc["placements"][2]["arrange"]["locked"] is False


def test_9_a_pocket_is_layered_over_its_panel(run: Any) -> None:
    placed = by_name(run)
    arr = {p["piece"]: p["arrange"] for p in run.doc["placements"]}
    assert arr["Pocket"]["radius_m"] >= arr["FrontPanel"]["radius_m"] + 0.03 - 1e-6  # a gap further out
    assert arr["Pocket"]["layer"] >= 1 and arr["FrontPanel"]["layer"] == 0
    assert pair_distance(placed["Pocket"], placed["FrontPanel"]) >= 0.03 - 0.001


def test_10_seam_candidates_and_world_geometry(run: Any) -> None:
    cands = run.doc["arrangement"]["seam_candidates"]
    pairs = {frozenset((c["a"]["piece"], c["b"]["piece"])): c for c in cands}
    for p in ({"FrontPanel", "FrontPanel_L"}, {"BackPanel", "BackPanel_L"}):
        assert frozenset(p) in pairs, (p, list(pairs))
    for c in cands:
        assert 0.0 < c["distance_m"] < 3 * 0.03 and c["a"]["polyline_m"] and c["b"]["polyline_m"]
    assert pairs[frozenset(("FrontPanel", "FrontPanel_L"))]["distance_m"] == pytest.approx(0.03, abs=0.002)
    p0 = run.doc["placements"][0]
    assert len(p0["world_outline_m"]) > 50 and {q["name"] for q in p0["world_named_points"]} >= {"A32", "A33"}


def test_report_fields(run: Any) -> None:
    doc = run.doc
    assert doc["schema_version"] == 4
    arr = doc["arrangement"]
    assert arr["options"]["clearance_body"] == 0.03 and arr["options"]["gap_layer"] == 0.03
    assert arr["timing_s"] >= 0 and "violations" in arr and arr["rings"]
    for p in doc["placements"]:
        for k in ("ring", "mode", "layer", "locked", "radius_m", "standoff_m", "shift_m", "min_distance_to_avatar_m"):
            assert k in p["arrange"], k
        assert p["arrange"]["min_distance_to_avatar_m"] >= 0.03 - 0.001
        assert set(p["size_check"]) >= {
            "max_edge_change_pct",
            "perimeter_change_pct",
            "area_change_pct",
            "named_point_change_pct",
        }
        assert p["changes_piece_size"] is False and p["solved"]["radius_m"] > 0
    v = doc["validity"]
    assert v["valid"] and v["size_ok"] and v["clearance_ok"] and v["gaps_ok"] and v["violations"] == []

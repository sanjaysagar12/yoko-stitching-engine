"""Orientation tools: ``orient`` (grain along the axis), ``rotation_deg`` and ``tilt`` (isometries, mirror pairs) and
the answer fields the web UI shows. Synthetic avatar."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from alignment_engine import AlignmentEngine
from alignment_engine.contracts import validate
from alignment_engine.orient import edge_point, orient_rotation, registration_spec, tilt_matrix
from assembler import verify as V

from .conftest import ROOT

TOL_PCT = 0.05


@pytest.fixture(scope="module")
def ctx(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    from assembler.run import build_inputs
    from tests.synthetic_avatar import build_avatar

    base = tmp_path_factory.mktemp("orient")
    cache = base / "cache"
    out: dict[str, Any] = {"avatar": build_avatar(base / "avatar"), "base": base, "engine": AlignmentEngine()}
    for job in ("aldrich_skirt", "trousers", "male_shirt"):
        out[job] = build_inputs(job, True, cache).pieces
    return out


def align(
    ctx: dict[str, Any],
    job: str,
    placements: list[dict[str, Any]],
    derived: list[dict[str, Any]] | None = None,
    pieces: Path | None = None,
) -> Any:
    spec = {"schema_version": 5, "placements": placements, "derived_pieces": derived or []}
    return ctx["engine"].align(
        ctx["avatar"], pieces or ctx[job], json.dumps(spec), derived_dir=ctx["base"] / f"d_{job}"
    )


def bundle_file(folder: Path, name: str) -> Path:
    return next(f for f in folder.glob("*.piece.json") if json.loads(f.read_text("utf-8")).get("name") == name)


def worst_pct(d: dict[str, Any]) -> float:
    c = d["size_check"]
    return max(c["max_edge_change_pct"], c["perimeter_change_pct"], c["area_change_pct"], c["named_point_change_pct"])


def world_dir(p: Any, d2: tuple[float, float]) -> np.ndarray:
    pv = V.PieceView(p.name, p.point, p.vertices, p.flat, p.faces, {}, {}, {})
    return V._mean_dir(pv, np.array(d2))


def axis_of(ctx: dict[str, Any], point: str) -> np.ndarray:
    pts = {q["name"]: q for q in json.loads((ctx["avatar"] / "arrangement_points.json").read_text("utf-8"))["points"]}
    vols = {v["name"]: v for v in json.loads((ctx["avatar"] / "bounding_volumes.json").read_text("utf-8"))["volumes"]}
    return V.axis_up(pts[point], vols)


# -- pure functions ----------------------------------------------------------------------------------------------------
def _bundle(grain: float | None, w: float = 0.3, h: float = 0.6) -> dict[str, Any]:
    g = None if grain is None else {"anchor_xy_m": [0.1, 0.1], "rotation_deg": grain, "length_m": 0.1}
    return {"grainline": g, "bbox_m": {"min": [0.0, 0.0], "max": [w, h]}, "named_points": []}


@pytest.mark.parametrize(
    ("grain", "orient", "want"),
    [
        (90.0, "auto", 0.0),
        (80.0, "auto", 10.0),
        (100.0, "auto", -10.0),
        (270.0, "auto", 0.0),
        (0.0, "auto", 0.0),
        (0.0, "grain", 90.0),
        (None, "auto", 0.0),
        (80.0, "none", 0.0),
        (80.0, 90, 90.0),
        (80.0, 270, 270.0),
    ],
)
def test_orient_rotation(grain: float | None, orient: Any, want: float) -> None:
    assert orient_rotation(_bundle(grain), orient)[0] == pytest.approx(want)


def test_tilt_matrix_is_a_rotation_and_leans_the_top_out() -> None:
    e_n, e_x, e_y = np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    r = tilt_matrix(e_n, e_x, e_y, 10.0, -5.0)
    assert np.allclose(r @ r.T, np.eye(3)) and np.linalg.det(r) == pytest.approx(1.0)
    assert (r @ e_y) @ e_n > 0.15 and (r @ e_y) @ e_x < -0.05


def test_default_registration_only_for_bottoms() -> None:
    b = {"named_points": [{"name": "WaistLIne", "xy_m": [0.1, 1.0]}], "bbox_m": {"min": [0, 0], "max": [0.3, 1.1]}}
    assert registration_spec({}, b, 5, "Thigh_L") == {"level": "Waist", "piece_point": "WaistLIne"}
    assert registration_spec({}, b, 5, "Chest") is None  # a top keeps its point
    assert registration_spec({}, b, 4, "Thigh_L") is None  # old specs unchanged
    assert registration_spec({"anchor": "root"}, b, 5, "Thigh_L") is None
    nb = {"named_points": [], "bbox_m": b["bbox_m"]}
    assert registration_spec({}, nb, 5, "Waist") == {"level": "Waist", "edge": "top"}


def test_edge_point_is_the_middle_of_a_flat_edge() -> None:
    sq = np.array([[0.0, 0.0], [0.4, 0.0], [0.4, 0.04], [0.2, 0.04], [0.0, 0.04]])
    assert edge_point(sq, "top").tolist() == [0.2, 0.04]


# -- orient auto: grain along the ring / limb axis on the three garments -----------------------------------------------
@pytest.mark.parametrize(
    ("job", "piece", "point"),
    [
        ("aldrich_skirt", "A - Skirt Front", "Waist_Front"),
        ("trousers", "Back", "Thigh_Back_L"),
        ("male_shirt", "FrontPanel", "Chest_Front"),
        ("male_shirt", "FullSleeve", "UpperArm_Outer_R"),
    ],
)
def test_orient_auto_keeps_grain_along_the_axis(ctx: dict[str, Any], job: str, piece: str, point: str) -> None:
    res = align(ctx, job, [{"piece": piece, "point": point, "wrap": "roll"}])
    p, d = res.placed[0], res.doc["placements"][0]
    g = (res.doc and d["arrange"].get("applied_rotation_deg"), d["arrange"].get("orient"))
    assert g[1]
    ang = V._angle(world_dir(p, (0.0, 1.0)), axis_of(ctx, point))
    assert ang <= 5.0, (piece, ang)


def test_orient_auto_turns_an_off_grain_piece(ctx: dict[str, Any], tmp_path: Path) -> None:
    """A skirt front whose grainline is drafted 8 degrees off vertical is turned so that the grain hangs straight."""
    pieces = tmp_path / "pieces"
    shutil.copytree(ctx["aldrich_skirt"], pieces)
    f = bundle_file(pieces, "A - Skirt Front")
    b = json.loads(f.read_text(encoding="utf-8"))
    b["grainline"]["rotation_deg"] = 98.0
    f.write_text(json.dumps(b), encoding="utf-8")
    res = align(
        ctx, "aldrich_skirt", [{"piece": "A - Skirt Front", "point": "Waist_Front", "wrap": "roll"}], pieces=pieces
    )
    d, p = res.doc["placements"][0], res.placed[0]
    assert d["arrange"]["applied_rotation_deg"] == pytest.approx(-8.0)
    th = np.radians(98.0)
    assert V._angle(world_dir(p, (np.cos(th), np.sin(th))), axis_of(ctx, "Waist_Front"), line=True) <= 1.0
    assert worst_pct(d) <= TOL_PCT


# -- rotation and tilt are isometries; mirror pairs --------------------------------------------------------------------
@pytest.mark.parametrize("wrap", ["flat", "roll"])
def test_rotation_and_tilt_keep_every_size(ctx: dict[str, Any], wrap: str) -> None:
    pl = {
        "piece": "A - Skirt Front",
        "point": "Hip_Front",
        "wrap": wrap,
        "rotation_deg": 25.0,
        "tilt": {"pitch_deg": 12.0, "roll_deg": -7.0},
    }
    res = align(ctx, "aldrich_skirt", [pl])
    d = res.doc["placements"][0]
    assert worst_pct(d) <= TOL_PCT and d["changes_piece_size"] is False
    assert d["arrange"]["tilt_deg"] == {"pitch_deg": 12.0, "roll_deg": -7.0}
    assert d["arrange"]["applied_rotation_deg"] == pytest.approx(25.0)
    rv = V.RunView(
        "t",
        [
            V.PieceView(
                p.name,
                p.point,
                p.vertices,
                p.flat,
                p.faces,
                json.loads(bundle_file(ctx["aldrich_skirt"], "A - Skirt Front").read_text("utf-8")),
                pl,
                d,
            )
            for p in res.placed
        ],
        res.doc,
        {},
    )
    V.check_size(rv)
    assert all(r["status"] == "pass" for r in rv.rows), rv.rows


def test_tilt_clamped_to_45_degrees(ctx: dict[str, Any]) -> None:
    res = align(
        ctx,
        "aldrich_skirt",
        [{"piece": "A - Skirt Front", "point": "Hip_Front", "wrap": "flat", "tilt": {"pitch_deg": 80}}],
    )
    assert res.doc["placements"][0]["arrange"]["tilt_deg"]["pitch_deg"] == 45.0


def test_quarter_turn_side_pair_is_mirrored(ctx: dict[str, Any]) -> None:
    """A piece turned 90 on the right arm and its mirror clone turned 270 on the left are one mirror pair."""
    derived = [{"id": "PlacketTop_L", "source": "PlacketTop", "op": "mirror", "axis": {"type": "bbox_center"}}]
    res = align(
        ctx,
        "male_shirt",
        [
            {"piece": "PlacketTop", "point": "Forearm_Back_R", "wrap": "roll", "orient": 90},
            {"piece": "PlacketTop_L", "point": "Forearm_Back_L", "wrap": "roll", "orient": 270},
        ],
        derived,
    )
    a, b = res.doc["placements"]
    assert a["arrange"]["pair_with"] == "PlacketTop_L" and b["arrange"]["pair_with"] == "PlacketTop"
    pa, pb = res.placed
    ma = pa.vertices * np.array([-1.0, 1.0, 1.0])
    from scipy.spatial import cKDTree

    assert float(cKDTree(pb.vertices).query(ma)[0].max()) < 0.003
    assert abs(world_dir(pa, (1.0, 0.0))[0]) > 0.95  # turned a quarter: its long side runs along the arm


# -- answer fields the web UI shows ------------------------------------------------------------------------------------
def test_answer_fields_round_trip_through_the_contracts(ctx: dict[str, Any]) -> None:
    placements = [
        {
            "piece": "A - Skirt Front",
            "point": "Waist_Front",
            "wrap": "roll",
            "orient": "grain",
            "anchor": {"level": "Waist", "edge": "top", "offset_m": 0.01},
            "tilt": {"pitch_deg": 3.0},
        },
    ]
    spec = {"schema_version": 5, "placements": placements}
    validate("placement_spec", spec)
    res = align(ctx, "aldrich_skirt", placements)
    validate("alignment_result", res.doc)
    a = res.doc["placements"][0]["arrange"]
    assert {"applied_rotation_deg", "orient", "tilt_deg", "registration"} <= set(a)
    assert a["registration"]["level"] == "Waist" and abs(a["registration"]["offset_m"]) < 0.015
    solved = res.doc["placements"][0]["solved"]
    again = align(ctx, "aldrich_skirt", [{**placements[0], "solved": solved}])
    assert np.abs(again.placed[0].vertices - res.placed[0].vertices).max() < 1e-6


def test_frontend_types_know_the_new_fields() -> None:
    ts = (ROOT / "src/web/frontend/src/types.ts").read_text(encoding="utf-8")
    for name in ("orient?: Orient", "tilt?:", "LevelAnchor", "applied_rotation_deg", "registration?:", "over?:"):
        assert name in ts, name


def test_central_points_have_their_up_upward(ctx: dict[str, Any]) -> None:
    """The Hip volume runs pelvis -> crotch (downward); its points' up used to point to the floor, so every piece on a
    Hip point came out upside down (and the trouser panel on Hip_Left of the report too)."""
    pts = json.loads((ctx["avatar"] / "arrangement_points.json").read_text("utf-8"))["points"]
    central = [p for p in pts if not p["bv"].endswith(("_L", "_R"))]
    assert {p["bv"] for p in central} >= {"Hip", "Waist", "Chest", "Neck"}
    assert all(p["up"][1] > 0.5 for p in central), [(p["name"], p["up"]) for p in central if p["up"][1] <= 0.5]
    res = align(
        ctx, "aldrich_skirt", [{"piece": "A - Skirt Front", "point": "Hip_Front", "wrap": "roll", "anchor": "point"}]
    )
    assert world_dir(res.placed[0], (0.0, 1.0))[1] > 0.95  # the pattern's top is up

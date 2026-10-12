"""Auto-arrange around the body (alignment_engine Part C and D): standoff, gaps, layers, honesty about ease."""

from __future__ import annotations

import itertools
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy.spatial import cKDTree

from alignment_engine import AlignmentEngine
from alignment_engine.loaders import read_avatar
from alignment_engine.verify import AvatarSurface, surface_samples

CF = {"type": "edge", "from_point": "A33", "to_point": "A32"}
DERIVED = [
    {"id": "FrontPanel_L", "source": "FrontPanel", "op": "mirror", "axis": CF},
    {"id": "BackPanel_L", "source": "BackPanel", "op": "mirror", "axis": {"type": "x", "value_m": 0.0}},
    {"id": "ShortSleeve_L", "source": "ShortSleeve", "op": "mirror", "axis": {"type": "bbox_center"}},
]
TORSO = ["FrontPanel", "FrontPanel_L", "BackPanel", "BackPanel_L"]


def shirt_placements(extra: dict[str, dict[str, Any]] | None = None, pocket: bool = True) -> list[dict[str, Any]]:
    pl: list[dict[str, Any]] = [
        {"piece": "FrontPanel", "point": "Neck_Front", "reference": "A32", "ring": "torso"},
        {"piece": "FrontPanel_L", "point": "Neck_Front", "reference": "A32", "ring": "torso"},
        {"piece": "BackPanel", "point": "Chest_Back", "reference": "A24"},
        {"piece": "BackPanel_L", "point": "Chest_Back", "reference": "A24"},
        {"piece": "ShortSleeve", "point": "UpperArm_Outer_R", "shift_m": [0.0, 0.13]},
        {"piece": "ShortSleeve_L", "point": "UpperArm_Outer_L", "shift_m": [0.0, 0.13]},
    ]
    if pocket:
        pl.append({"piece": "Pocket", "point": "Chest_FrontSide_L"})
    for p in pl:
        p.update((extra or {}).get(p["piece"], {}))
    return pl


def spec(placements: list[dict[str, Any]] | None = None, **opts: Any) -> str:
    doc: dict[str, Any] = {
        "schema_version": 3,
        "placements": placements if placements is not None else shirt_placements(),
        "derived_pieces": DERIVED,
    }
    if opts:
        doc["arrange_options"] = opts
    return json.dumps(doc)


@pytest.fixture(scope="module")
def surface(scene: dict[str, Path]) -> AvatarSurface:
    av = read_avatar(scene["avatar"])
    return AvatarSurface(av.vertices, av.faces)


@pytest.fixture(scope="module")
def run(scene: dict[str, Path], tmp_path_factory: pytest.TempPathFactory) -> Any:
    d = tmp_path_factory.mktemp("derived")
    eng = AlignmentEngine()
    return eng.arrange(scene["avatar"], scene["pieces"], spec(), derived_dir=d, cache_dir=d / "cache")


def by_name(res: Any) -> dict[str, Any]:
    return {p.name: p for p in res.placed}


# 2 ------------------------------------------------------------------------------------------------------------
def test_2_every_piece_keeps_the_body_clearance_including_interior_vertices(run: Any, surface: AvatarSurface) -> None:
    placed = by_name(run)
    assert set(placed) == {
        "FrontPanel",
        "FrontPanel_L",
        "BackPanel",
        "BackPanel_L",
        "ShortSleeve",
        "ShortSleeve_L",
        "Pocket",
    }
    for name, p in placed.items():
        d = surface.distance(p.vertices)  # all vertices, interior ones included
        assert d.min() >= 0.03 - 0.001, (name, d.min())
        d_edges = surface.distance(surface_samples(p.vertices, p.faces, 0.004))
        assert d_edges.min() >= 0.03 - 0.001, (name, d_edges.min())
    assert not any(w.startswith("CLEARANCE_BODY") for w in run.warnings), run.warnings


# 3 ------------------------------------------------------------------------------------------------------------
_SAMPLES: dict[int, tuple[Any, Any]] = {}


def samples(p: Any) -> tuple[Any, Any]:
    """Surface samples (vertices and points along every triangle edge, 3 mm apart) and their KD-tree, cached per
    piece."""
    if id(p) not in _SAMPLES:
        pts = surface_samples(p.vertices, p.faces, 0.003)
        _SAMPLES[id(p)] = (pts, cKDTree(pts))
    return _SAMPLES[id(p)]


def pair_distance(a: Any, b: Any) -> float:
    return float(samples(b)[1].query(samples(a)[0])[0].min())


def test_3_pieces_on_a_ring_keep_the_gap(run: Any) -> None:
    placed = by_name(run)
    for a, b in itertools.combinations(TORSO, 2):
        d = pair_distance(placed[a], placed[b])
        assert d >= 0.03 - 0.001, (a, b, d)
    # the centre-line pair has the gap across the centre line
    assert pair_distance(placed["FrontPanel"], placed["FrontPanel_L"]) == pytest.approx(0.03, abs=0.001)
    assert pair_distance(placed["BackPanel"], placed["BackPanel_L"]) == pytest.approx(0.03, abs=0.001)
    assert not any(w.startswith("GAP_PIECES") for w in run.warnings), run.warnings


# 4 ------------------------------------------------------------------------------------------------------------
def test_4_real_size_is_preserved(run: Any) -> None:
    placed = by_name(run)
    for name in ("FrontPanel", "BackPanel", "ShortSleeve", "Pocket"):
        p = placed[name]
        flat_w = {"FrontPanel": 0.3725, "BackPanel": 0.3475, "ShortSleeve": 0.5230, "Pocket": 0.09}[name]
        # arc length along a row equals the flat distance (the rows are the offset curves of the body section)
        for y in (0.02, 0.05, 0.1, 0.2, 0.3):
            x = np.linspace(0.0, flat_w * 0.9, 600)
            w = p.transform(np.column_stack([x, np.full_like(x, y)]))
            assert np.linalg.norm(np.diff(w, axis=0), axis=1).sum() == pytest.approx(x[-1], rel=0.005), (name, y)
    for p_doc in run.doc["placements"]:
        arr = p_doc["arrange"]
        name = p_doc["piece"]
        assert abs(arr["area_ratio"] - 1.0) < 0.02, (name, arr["area_ratio"])
        assert arr["self_intersections"] == 0, name
        # A flat piece cannot be wrapped on a double-curved body without some stretch: the median edge is
        # within 1.5 %, 90 %
        # of the edges of the body rows within 12 %; the worst edges (reported separately) are at the
        # shoulder/underarm bands
        assert arr["edge_change_p50_pct"] < 1.5, (name, arr["edge_change_p50_pct"])
        assert arr["edge_change_p90_pct"] < 12.0, (name, arr["edge_change_p90_pct"])
        assert (
            arr["max_edge_change_body_pct"] is not None
            and arr["max_edge_change_pct"] >= arr["max_edge_change_body_pct"]
        )


# 5 ------------------------------------------------------------------------------------------------------------
def test_5_true_ease_is_not_the_standoff(scene: dict[str, Path], tmp_path: Path) -> None:
    eng = AlignmentEngine()
    a = eng.arrange(scene["avatar"], scene["pieces"], spec(clearance_body=0.03, gap_piece=0.03), derived_dir=tmp_path)
    b = eng.arrange(scene["avatar"], scene["pieces"], spec(clearance_body=0.04, gap_piece=0.045), derived_dir=tmp_path)
    ra = {r["ring"]: r for r in a.doc["arrangement"]["rings"]}["torso"]
    rb = {r["ring"]: r for r in b.doc["arrangement"]["rings"]}["torso"]
    assert ra["true_ease_m"] == pytest.approx(ra["garment_loop_m"] - ra["body_loop_m"], abs=1e-9)
    assert (
        ra["true_ease_m"] == rb["true_ease_m"] and ra["body_loop_m"] == rb["body_loop_m"]
    )  # not changed by the options
    assert rb["standoff_used_m"] > ra["standoff_used_m"] + 0.005  # the display standoff is
    # the shirt is much wider than the body: 2 x 37.25 + 2 x 34.75 = 144 cm of chest for a body loop of about 1 m
    assert ra["garment_loop_m"] == pytest.approx(1.44, abs=0.15) and ra["true_ease_m"] > 0.3
    assert ra["standoff_used_m"] > 0.03 and ra["binding_row_m"] is not None


# 6 ------------------------------------------------------------------------------------------------------------
def test_6_locked_pieces_do_not_move_and_are_obstacles(scene: dict[str, Path], tmp_path: Path) -> None:
    eng = AlignmentEngine()
    base = eng.arrange(scene["avatar"], scene["pieces"], spec(), derived_dir=tmp_path)
    locked_pl = shirt_placements({"BackPanel": {"locked": True, "wrap": "around"}})
    first = eng.arrange(scene["avatar"], scene["pieces"], spec(locked_pl), derived_dir=tmp_path)
    second = eng.arrange(scene["avatar"], scene["pieces"], spec(locked_pl, gap_piece=0.05), derived_dir=tmp_path)
    a, b = by_name(first)["BackPanel"], by_name(second)["BackPanel"]
    assert np.abs(a.vertices - b.vertices).max() < 1e-9  # the locked piece keeps its manual placement
    plain = eng.align(
        scene["avatar"],
        scene["pieces"],
        json.dumps({"schema_version": 3, "placements": [locked_pl[2]], "derived_pieces": DERIVED}),
        derived_dir=tmp_path,
    ).placed[0]
    assert np.abs(plain.vertices - a.vertices).max() < 1e-9
    # it is an obstacle: the other pieces stay the required gap away from it
    for n in ("FrontPanel", "FrontPanel_L", "BackPanel_L"):
        assert pair_distance(by_name(second)[n], b) >= 0.05 - 0.001, n
    assert first.doc["placements"][2]["arrange"]["locked"] is True
    assert base.doc["placements"][2]["arrange"]["locked"] is False


def test_6_standoff_exceeded_is_reported_not_clamped_silently(scene: dict[str, Path], tmp_path: Path) -> None:
    res = AlignmentEngine().arrange(scene["avatar"], scene["pieces"], spec(max_standoff=0.01), derived_dir=tmp_path)
    ring = {r["ring"]: r for r in res.doc["arrangement"]["rings"]}["torso"]
    assert any(w.startswith("STANDOFF_EXCEEDED") for w in res.warnings)
    assert ring["standoff_used_m"] == pytest.approx(0.01) and ring["standoff_required_m"] > 0.03  # the need is reported
    assert ring["violating_rows"]  # and the rows that need more are listed
    assert len(res.placed) == 7  # still the best arrangement within the limit


# 7 ------------------------------------------------------------------------------------------------------------
def test_7_deterministic_and_idempotent(scene: dict[str, Path], tmp_path: Path) -> None:
    eng = AlignmentEngine()
    r1 = eng.arrange(scene["avatar"], scene["pieces"], spec(), out_dir=tmp_path / "a", derived_dir=tmp_path / "d1")
    r2 = AlignmentEngine().arrange(
        scene["avatar"], scene["pieces"], spec(), out_dir=tmp_path / "b", derived_dir=tmp_path / "d2"
    )
    for f in sorted(p.name for p in (tmp_path / "a").iterdir()):
        if f == "alignment_result.json":  # identical except the run time
            da, db = (json.loads((tmp_path / d / f).read_text()) for d in "ab")
            da["arrangement"].pop("timing_s"), db["arrangement"].pop("timing_s")
            assert da == db
        else:
            assert (tmp_path / "a" / f).read_bytes() == (tmp_path / "b" / f).read_bytes(), f
    # run again starting from the shifts the first run found: nothing moves by more than 0.1 mm
    shifts = {p["piece"]: p["arrange"]["shift_m"] for p in r1.doc["placements"]}
    again = shirt_placements({k: {"shift_m": v} for k, v in shifts.items()})
    r3 = eng.arrange(scene["avatar"], scene["pieces"], spec(again), derived_dir=tmp_path / "d3")
    for p1, p3 in zip(r1.placed, r3.placed, strict=True):
        assert np.abs(p1.vertices - p3.vertices).max() < 1e-4, p1.name
    assert r2.doc["arrangement"]["rings"] == r1.doc["arrangement"]["rings"]


# 8 ------------------------------------------------------------------------------------------------------------
def test_8_symmetry_of_a_mirrored_pair(run: Any) -> None:
    placed = by_name(run)
    pts = np.column_stack([np.linspace(0.0, 0.3, 40), np.full(40, 0.25)])
    left = placed["FrontPanel"].transform(pts)
    flat_w = 0.3725
    right = placed["FrontPanel_L"].transform(
        np.column_stack([flat_w - pts[:, 0], pts[:, 1]])
    )  # mirrored piece, same distances from CF
    assert np.abs(left * [1, 1, 1] - right * [-1, 1, 1]).max() < 0.002
    sl, sr = placed["ShortSleeve"].vertices, placed["ShortSleeve_L"].vertices
    ref_l = run.doc["placements"][4]["reference_position_m"]
    ref_r = run.doc["placements"][5]["reference_position_m"]
    assert abs(ref_l[0] + ref_r[0]) < 0.002 and abs(ref_l[1] - ref_r[1]) < 0.002 and abs(ref_l[2] - ref_r[2]) < 0.002
    assert abs(sl[:, 0].min() + sr[:, 0].max()) < 0.003 and abs(sl[:, 1].mean() - sr[:, 1].mean()) < 0.002


# 9 ------------------------------------------------------------------------------------------------------------
def test_9_a_pocket_floats_over_the_panel(run: Any, surface: AvatarSurface) -> None:
    placed = by_name(run)
    arr = {p["piece"]: p["arrange"] for p in run.doc["placements"]}
    assert arr["Pocket"]["layer"] == 1 and arr["FrontPanel"]["layer"] == 0
    assert pair_distance(placed["Pocket"], placed["FrontPanel"]) >= 0.015 - 0.001
    assert surface.distance(placed["Pocket"].vertices).min() >= 0.03 - 0.001
    assert (
        arr["Pocket"]["standoff_m"] >= arr["FrontPanel"]["standoff_m"] + 0.015 - 1e-9
    )  # more on a tilted shell (shoulders)
    # a pocket is not part of the ring: it does not count for the gaps or for the garment loop
    ring = {r["ring"]: r for r in run.doc["arrangement"]["rings"]}["torso"]
    assert "Pocket" not in ring["pieces"] and set(ring["pieces"]) == set(TORSO)


# 10 -----------------------------------------------------------------------------------------------------------
def test_10_seam_candidates(run: Any) -> None:
    cands = run.doc["arrangement"]["seam_candidates"]
    pairs = {frozenset((c["a"]["piece"], c["b"]["piece"])): c for c in cands}
    for p in (
        {"FrontPanel", "FrontPanel_L"},
        {"BackPanel", "BackPanel_L"},
        {"FrontPanel", "BackPanel_L"},
        {"FrontPanel_L", "BackPanel"},
    ):
        assert frozenset(p) in pairs, (p, list(pairs))
    for c in cands:
        assert 0.0 < c["distance_m"] < 3 * 0.03 and c["a"]["polyline_m"] and c["b"]["polyline_m"]
        assert c["a"]["edge"] and c["b"]["edge"] and c["length_ratio"] > 0
    assert pairs[frozenset(("FrontPanel", "FrontPanel_L"))]["distance_m"] == pytest.approx(0.03, abs=0.002)
    # world outlines and named points are exported for picking edges
    p0 = run.doc["placements"][0]
    assert len(p0["world_outline_m"]) > 50 and {q["name"] for q in p0["world_named_points"]} >= {"A32", "A33"}


# report -------------------------------------------------------------------------------------------------------
def test_report_fields(run: Any) -> None:
    arr = run.doc["arrangement"]
    assert arr["options"]["clearance_body"] == 0.03 and arr["options"]["standoff_mode"] == "uniform"
    assert arr["timing_s"] >= 0 and "violations" in arr
    p = run.doc["placements"][0]["arrange"]
    for k in (
        "ring",
        "layer",
        "locked",
        "standoff_m",
        "shift_m",
        "min_distance_to_avatar_m",
        "min_distance_to_pieces_m",
        "area_ratio",
    ):
        assert k in p, k
    assert p["min_distance_to_avatar_m"] >= 0.03 - 0.001
    assert run.doc["schema_version"] == 3


def test_performance_for_the_17_piece_shirt(scene: dict[str, Path], tmp_path: Path) -> None:
    anchors = [
        ("FrontPanel", "Neck_Front", "A32", "torso"),
        ("BackPanel", "Chest_Back", "A24", None),
        ("Yoke", "Chest_Back", None, "torso"),
        ("Pocket", "Chest_FrontSide_L", None, None),
        ("PocketFlap", "Chest_FrontSide_L", None, None),
        ("PocketRound", "Chest_FrontSide_R", None, None),
        ("PocketFlapRound", "Chest_FrontSide_R", None, None),
        ("PlacketTop", "Chest_Front", None, None),
        ("PlacketUnder", "Waist_Front", None, None),
        ("CollarBase", "Neck_Front", None, None),
        ("CollarBaseInterface", "Neck_Front", None, None),
        ("CollarTop", "Neck_Back", None, None),
        ("CollarTopInterface", "Neck_Back", None, None),
        ("ShortSleeve", "UpperArm_Outer_R", None, None),
        ("FullSleeve", "UpperArm_Outer_L", None, None),
        ("Cuff", "Forearm_Outer_R", None, None),
        ("CuffInterface", "Forearm_Outer_L", None, None),
    ]
    pl: list[dict[str, Any]] = []
    for name, point, ref, ring in anchors:
        d: dict[str, Any] = {"piece": name, "point": point}
        if ref:
            d["reference"] = ref
        if ring:
            d["ring"] = ring
        if name in ("ShortSleeve", "FullSleeve"):
            d["shift_m"] = [0.0, 0.13]
        pl.append(d)
    eng = AlignmentEngine()
    text = json.dumps({"schema_version": 3, "placements": pl})
    eng.arrange(
        scene["avatar"], scene["pieces"], text, derived_dir=tmp_path, cache_dir=tmp_path / "c"
    )  # fills the caches
    t = time.time()
    res = eng.arrange(scene["avatar"], scene["pieces"], text, derived_dir=tmp_path, cache_dir=tmp_path / "c")
    assert len(res.placed) == 17
    assert time.time() - t < 10.0, time.time() - t

"""body_slice wrap (alignment_engine Part C; only with ``allow_distortion``) and ring multiplicity (A5)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from alignment_engine import AlignmentEngine
from avatar_engine.points import _dist_point_triangles

CF = {"type": "edge", "from_point": "A33", "to_point": "A32"}


def spec(*placements: dict[str, Any], **extra: Any) -> str:
    """body_slice stretches the piece (it is kept for comparison only), so these specs allow distortion explicitly."""
    return json.dumps(
        {"schema_version": 4, "placements": list(placements), "arrange_options": {"allow_distortion": True}, **extra}
    )


def avatar_points(scene: dict[str, Path]) -> dict[str, Any]:
    pts = json.loads((scene["avatar"] / "arrangement_points.json").read_text())["points"]
    return {p["name"]: p for p in pts}


def avatar_triangles(scene: dict[str, Path]) -> np.ndarray:
    lines = (scene["avatar"] / "avatar.obj").read_text().splitlines()
    v = np.array([ln.split()[1:4] for ln in lines if ln.startswith("v ")], dtype=float)
    f = np.array([[int(x) - 1 for x in ln.split()[1:4]] for ln in lines if ln.startswith("f ")])
    return v[f]  # (M,3,3)


def make_piece(
    folder: Path, name: str, outline: list[tuple[float, float]], named: dict[str, tuple[float, float]]
) -> None:
    """Write a piece folder entry (obj + piece.json) for a convex-ish polygon, triangulated as a fan."""
    n = len(outline)
    xs, ys = zip(*outline, strict=True)
    lines = [f"v {x:.6f} {y:.6f} 0.000000" for x, y in outline] + [f"f 1 {i + 1} {i + 2}" for i in range(1, n - 1)]
    (folder / f"{name}_m.obj").write_text("\n".join(lines) + "\n")
    pts = [
        {"name": k, "label": k, "xy_m": [x, y], "on_outline": False, "outline_s_m": None} for k, (x, y) in named.items()
    ]
    doc = {
        "schema_version": 2,
        "name": name,
        "units": "m",
        "obj": f"{name}_m.obj",
        "bbox_m": {"min": [0.0, 0.0], "max": [max(xs), max(ys)]},
        "outline_m": [list(p) for p in outline],
        "named_points": pts,
        "edges": [],
        "fold_edge": None,
        "grainline": None,
        "default_reference": {"type": "bbox_centre", "xy_m": [max(xs) / 2, max(ys) / 2]},
        "mirror_of": None,
        "axis": None,
        "origin_shift_m": None,
    }
    (folder / f"{name}.piece.json").write_text(json.dumps(doc))


@pytest.fixture(scope="module")
def panels(scene: dict[str, Path], tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A controlled half panel (30 x 30 cm, centre front on x = 0, CF point at mid height) and a 1.2 m wide strip."""
    d = tmp_path_factory.mktemp("panels")
    make_piece(d, "Panel", [(0, 0), (0.30, 0), (0.30, 0.30), (0, 0.30)], {"CF": (0.0, 0.15)})
    make_piece(d, "Wide", [(0, 0), (1.2, 0), (1.2, 0.2), (0, 0.2)], {})
    return d


def panel(point: str = "Chest_Front", **extra: Any) -> dict[str, Any]:
    return {"piece": "Panel", "point": point, "reference": "CF", "wrap": "body_slice", **extra}


# 8a -----------------------------------------------------------------------------------------------------------
def test_8_arc_length_along_each_row_equals_the_piece_x_distance(scene: dict[str, Path], panels: Path) -> None:
    res = AlignmentEngine().align(scene["avatar"], panels, spec(panel()))
    p = res.placed[0]
    assert p.wrapped and not any(w.startswith(("WRAP_OVERRUN", "BODY_SLICE_RANGE")) for w in p.warnings)
    for y in (0.0, 0.1, 0.2, 0.3):  # rows of the piece from bottom to top
        x = np.linspace(0.0, 0.30, 400)
        w = p.transform(np.column_stack([x, np.full_like(x, y)]))
        arc = np.linalg.norm(np.diff(w, axis=0), axis=1).sum()
        assert arc == pytest.approx(0.30, rel=0.005), y
        assert np.allclose(w[:, 1], w[0, 1], atol=1e-9)  # a piece row stays a horizontal row of the body
    a = p.transform(np.array([[0.1, 0.0]]))[0]
    b = p.transform(np.array([[0.1, 0.3]]))[0]
    assert b[1] - a[1] == pytest.approx(0.3, abs=1e-9)  # vertical distance is the piece y distance
    # the mesh itself: the bottom boundary row of the wrapped mesh is as long as the piece edge
    v = p.vertices
    row = v[np.abs(v[:, 1] - v[:, 1].min()) < 1e-9]
    row = row[np.argsort(np.arctan2(row[:, 0], row[:, 2]))]
    assert np.linalg.norm(np.diff(row, axis=0), axis=1).sum() == pytest.approx(0.30, rel=0.005)


def test_8_the_cf_line_stays_straight_on_the_body(scene: dict[str, Path], panels: Path) -> None:
    p = AlignmentEngine().align(scene["avatar"], panels, spec(panel())).placed[0]
    w = p.transform(np.column_stack([np.zeros(30), np.linspace(0, 0.3, 30)]))
    assert np.abs(w[:, 0]).max() < 5e-5  # x = 0 at every height (to 0.05 mm): the centre front
    assert (w[:, 2] > 0).all()


# 8b -----------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("level", ["Chest_Front", "Waist_Front", "Hip_Front"])
def test_8_wrapped_points_sit_at_the_offset_from_the_torso(scene: dict[str, Path], panels: Path, level: str) -> None:
    pts = avatar_points(scene)
    tri = avatar_triangles(scene)
    res = AlignmentEngine().align(scene["avatar"], panels, spec(panel(level, offset_m=0.03)))
    v = res.placed[0].vertices
    y0 = pts[level]["position_m"][1]
    sel = v[(np.abs(v[:, 1] - y0) < 0.03) & (np.abs(v[:, 0]) < 0.14) & (v[:, 2] > 0)]  # the front, around the CF
    assert len(sel) > 20
    for q in sel[:: max(1, len(sel) // 25)]:
        near = tri[np.linalg.norm(tri.mean(axis=1) - q, axis=1) < 0.25]
        assert _dist_point_triangles(q, near) == pytest.approx(0.03, abs=0.005), (level, q)


# 8c -----------------------------------------------------------------------------------------------------------
def test_8_a_piece_wider_than_the_loop_overruns(scene: dict[str, Path], panels: Path) -> None:
    res = AlignmentEngine().align(
        scene["avatar"], panels, spec({"piece": "Wide", "point": "Chest_Front", "wrap": "body_slice"})
    )
    r = res.doc["placements"][0]
    assert any(w.startswith("WRAP_OVERRUN") for w in r["warnings"])
    assert np.isfinite(res.placed[0].vertices).all()  # clamped, not scaled, never NaN
    assert r["bbox_cm"]["width"] == pytest.approx(120.0, abs=1e-6)


def test_8_a_half_panel_does_not_overrun_and_reports_ease(scene: dict[str, Path], panels: Path) -> None:
    r = AlignmentEngine().align(scene["avatar"], panels, spec(panel())).doc["placements"][0]
    assert not any("WRAP_OVERRUN" in w for w in r["warnings"])
    levels = r["fit"]["levels"]
    assert "Chest" in levels
    chest = levels["Chest"]
    assert chest["piece_width_cm"] == pytest.approx(30.0, abs=1e-6) and 80 < chest["body_loop_cm"] < 120
    assert chest["ease_quarter_cm"] == pytest.approx(30.0 - chest["body_loop_cm"] / 4, abs=0.01)
    assert chest["ease_half_cm"] == pytest.approx(30.0 - chest["body_loop_cm"] / 2, abs=0.01)
    assert len(r["fit"]["rows"]) >= 5


def test_a_real_shirt_front_panel_too_tall_for_the_torso_warns(scene: dict[str, Path]) -> None:
    """The 74 cm FrontPanel hung from the mid-neck point has its shoulders above the avatar shoulders: reported."""
    res = AlignmentEngine().align(
        scene["avatar"],
        scene["pieces"],
        spec({"piece": "FrontPanel", "point": "Neck_Front", "reference": "A32", "wrap": "body_slice"}),
    )
    r = res.doc["placements"][0]
    w = " ".join(r["warnings"])
    assert "WRAP_OVERRUN" in w or "BODY_SLICE_RANGE" in w
    assert np.isfinite(res.placed[0].vertices).all() and r["bbox_cm"]["height"] == pytest.approx(74.4, abs=0.01)


def test_body_slice_on_a_limb_point_falls_back_with_a_warning(scene: dict[str, Path]) -> None:
    res = AlignmentEngine().align(
        scene["avatar"],
        scene["pieces"],
        spec({"piece": "ShortSleeve", "point": "UpperArm_Outer_R", "wrap": "body_slice"}),
    )
    r = res.doc["placements"][0]
    assert any(w.startswith("WRAP_FALLBACK") for w in r["warnings"]) and r["fit"]["mode"] == "around"


def test_body_slice_is_symmetric_for_a_centre_line_pair(scene: dict[str, Path], panels: Path, tmp_path: Path) -> None:
    derived = [{"id": "Panel_L", "source": "Panel", "op": "mirror", "axis": {"type": "x", "value_m": 0.0}}]
    pair = AlignmentEngine().pair(scene["avatar"], panels, derived, panel(), "Panel_L", "same_anchor")
    res = AlignmentEngine().align(scene["avatar"], panels, spec(*pair, derived_pieces=derived), derived_dir=tmp_path)
    a, b = res.placed
    x = np.linspace(0, 0.30, 50)
    pa = a.transform(np.column_stack([x, np.full(50, 0.2)]))
    xb = float(json.loads((tmp_path / "Panel_L.piece.json").read_text())["bbox_m"]["max"][0])
    pb = b.transform(np.column_stack([xb - x, np.full(50, 0.2)]))  # the same distances from the CF edge, mirrored piece
    assert np.abs(pa[:, 0] + pb[:, 0]).max() < 1e-4  # the two halves are mirror images across x = 0 (0.1 mm)
    assert np.abs(pa[:, 1:] - pb[:, 1:]).max() < 1e-4
    assert {p.extends for p in res.placed} == {"left", "right"}


def test_body_slice_is_deterministic_and_fast(scene: dict[str, Path], panels: Path) -> None:
    eng = AlignmentEngine()
    t = time.time()
    a = eng.align(scene["avatar"], panels, spec(panel()))
    assert time.time() - t < 8.0
    t = time.time()
    b = eng.align(scene["avatar"], panels, spec(panel()))  # the torso sections are cached for the next call
    assert time.time() - t < 3.0
    assert np.array_equal(a.placed[0].vertices, b.placed[0].vertices)


# 9 (A5) -------------------------------------------------------------------------------------------------------
def test_9_ring_check_counts_derived_pieces_at_the_chest_height(scene: dict[str, Path], tmp_path: Path) -> None:
    # Front is wider at the chest line (y = 0.4) than at the shoulder; Back is a rectangle
    front = [(0, 0), (0.30, 0), (0.30, 0.5), (0.22, 0.6), (0.22, 0.7), (0, 0.7)]
    make_piece(tmp_path, "Front", front, {"Bust": (0.1, 0.65), "Chest": (0.1, 0.4)})
    make_piece(tmp_path, "Back", [(0, 0), (0.25, 0), (0.25, 0.7), (0, 0.7)], {"Chest": (0.1, 0.4), "Bust": (0.1, 0.4)})
    derived = [
        {"id": "Front_L", "source": "Front", "op": "mirror", "axis": {"type": "x", "value_m": 0.0}},
        {"id": "Back_L", "source": "Back", "op": "mirror", "axis": {"type": "x", "value_m": 0.0}},
    ]
    ring = {"name": "chest", "pieces": ["Front", "Back"], "avatar_measurement": "chest_circ"}
    eng = AlignmentEngine()

    def report(**extra: Any) -> dict[str, Any]:
        res = eng.align(scene["avatar"], tmp_path, spec(derived_pieces=derived, rings=[{**ring, **extra}]))
        rep: dict[str, Any] = res.doc["ring_reports"][0]
        return rep

    rep = report()
    assert rep["pieces"] == ["Front", "Back", "Front_L", "Back_L"]
    assert rep["sum_width_cm"] == pytest.approx(2 * 30 + 2 * 25, abs=1e-6)  # 2 x front + 2 x back (bbox widths)
    assert rep["avatar_cm"] == pytest.approx(100.0, abs=0.01) and rep["ease_cm"] == pytest.approx(10.0, abs=0.01)
    assert report(level_point="Chest")["sum_width_cm"] == pytest.approx(110.0, abs=1e-6)
    # at the bust line (y = 0.65) the front is only 22 cm wide
    assert report(level_point="Bust")["widths_cm"] == pytest.approx([22.0, 25.0, 22.0, 25.0], abs=1e-6)
    assert report(include_clones=False)["sum_width_cm"] == pytest.approx(55.0)  # only the listed pieces count
    miss = report(level_point="Nope")  # a missing level point falls back to the bbox width, with a warning
    assert miss["warnings"] and miss["sum_width_cm"] == pytest.approx(110.0)


def test_9_ring_check_with_the_real_shirt_clones(scene: dict[str, Path]) -> None:
    derived = [
        {"id": "FrontPanel_L", "source": "FrontPanel", "op": "mirror", "axis": CF},
        {"id": "BackPanel_L", "source": "BackPanel", "op": "mirror", "axis": {"type": "bbox_center"}},
    ]
    ring = {"name": "chest", "pieces": ["FrontPanel", "BackPanel"], "avatar_measurement": "chest_circ"}
    r = AlignmentEngine().align(scene["avatar"], scene["pieces"], spec(derived_pieces=derived, rings=[ring]))
    rep = r.doc["ring_reports"][0]
    assert rep["sum_width_cm"] == pytest.approx(2 * 37.25 + 2 * 34.75, abs=0.01)
    assert rep["ease_cm"] == pytest.approx(rep["sum_width_cm"] - 100.0, abs=0.01)

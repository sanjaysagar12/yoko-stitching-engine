"""alignment_engine acceptance tests 8 to 12, plus integration with a real avatar and the pieces of the shirt."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from alignment_engine import AlignmentEngine
from alignment_engine.errors import (
    UnknownMeasurementError,
    UnknownNamedPointError,
    UnknownPieceError,
    UnknownPointError,
)
from alignment_engine.placement import (
    apply_matrix,
    mirror_mesh,
    refine_mesh,
    rigid_matrix,
    target_frame,
)
from alignment_engine.stitch import edge_between, stitch_report
from alignment_engine.wrap import WrapGeometry, wrap_geometry, wrap_vertices

ROOT = Path(__file__).resolve().parents[1]


# -- synthetic piece: an L-shaped polygon (asymmetric, so mirrors and rotations are detectable) ----------------------
def l_piece() -> tuple[np.ndarray, np.ndarray]:
    outline = np.array([[0, 0], [0.30, 0], [0.30, 0.10], [0.10, 0.10], [0.10, 0.40], [0, 0.40]], dtype=float)
    # ear-clip by hand: three quads split in triangles, CCW
    v = np.column_stack([outline, np.zeros(len(outline))])
    f = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 5]], dtype=np.int64)
    return v, f


def areas(v: np.ndarray, f: np.ndarray) -> np.ndarray:
    return 0.5 * np.linalg.norm(np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]), axis=1)


def normals(v: np.ndarray, f: np.ndarray) -> np.ndarray:
    n = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    return n / np.linalg.norm(n, axis=1, keepdims=True)


POINT_N = np.array([0.0, 0.3, 0.954])
POINT_N /= np.linalg.norm(POINT_N)
POINT_UP = np.array([0.0, 1.0, 0.0])
POINT_POS = np.array([0.02, 1.25, 0.18])


def pdist(v: np.ndarray) -> np.ndarray:
    return np.linalg.norm(v[:, None] - v[None, :], axis=2)


# 8 ------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("rot,flip", [(0, False), (37.5, False), (-120, True), (90, False)])
def test_8_rigid_preserves_distances_and_area(rot: float, flip: bool) -> None:
    v, f = l_piece()
    frame = target_frame(POINT_N, POINT_UP, flip)
    m = rigid_matrix(frame, rot, np.array([0.15, 0.2, 0.0]), POINT_POS + 0.03 * POINT_N)
    w = apply_matrix(m, v)
    assert np.abs(pdist(w) - pdist(v)).max() < 1e-9
    assert areas(w, f).sum() == pytest.approx(areas(v, f).sum(), abs=1e-12)
    assert np.linalg.det(m[:3, :3]) == pytest.approx(1.0)  # a proper rotation, never a reflection


# 9 ------------------------------------------------------------------------------------------------------------
def test_9_reference_lands_on_point_and_face_normal_matches() -> None:
    v, f = l_piece()
    ref = np.array([0.05, 0.3, 0.0])
    off = 0.045
    m = rigid_matrix(target_frame(POINT_N, POINT_UP, False), 20.0, ref, POINT_POS + off * POINT_N)
    w = apply_matrix(m, v)
    assert apply_matrix(m, ref[None])[0] == pytest.approx(POINT_POS + off * POINT_N, abs=1e-6)
    assert np.abs(normals(w, f) - POINT_N).max() < 1e-9
    # piece "up" maps to the point's up, projected perpendicular to the normal (at zero user rotation)
    m0 = rigid_matrix(target_frame(POINT_N, POINT_UP, False), 0.0, ref, POINT_POS)
    up = m0[:3, :3] @ np.array([0.0, 1.0, 0.0])
    expect = POINT_UP - (POINT_UP @ POINT_N) * POINT_N
    assert up == pytest.approx(expect / np.linalg.norm(expect), abs=1e-9)


def test_9_flip_facing_reverses_the_face_normal() -> None:
    v, f = l_piece()
    m = rigid_matrix(target_frame(POINT_N, POINT_UP, True), 0.0, np.zeros(3), POINT_POS)
    assert np.abs(normals(apply_matrix(m, v), f) + POINT_N).max() < 1e-9


# 10 -----------------------------------------------------------------------------------------------------------
def cylinder_geometry(radius: float = 0.15, offset: float = 0.0) -> tuple[WrapGeometry, np.ndarray]:
    start, end = np.array([0.0, 1.0, 0.0]), np.array([0.0, 1.4, 0.0])
    target = np.array([0.0, 1.2, radius + offset])  # on the front of a vertical cylinder
    return wrap_geometry(target, start, end, np.array([0.0, 1.0, 0.0])), target


def test_10_wrap_preserves_arc_length_and_height() -> None:
    geom, target = cylinder_geometry(0.15, 0.01)
    assert geom.radius == pytest.approx(0.16)
    n = 400
    x = np.linspace(0.0, 0.60, n)  # 60 cm wide strip along x, well under the 2*pi*0.16 = 1.005 m circumference
    local = np.column_stack([x, np.full(n, 0.05), np.zeros(n)])
    ref = np.array([0.30, 0.05, 0.0])
    out, info = wrap_vertices(local, ref, geom, 0.0, False)
    assert not info.clamped
    arc = np.linalg.norm(np.diff(out, axis=0), axis=1).sum()
    assert arc == pytest.approx(0.60, rel=1e-3)  # arc length along x preserved (0.1 %)
    assert np.allclose(out[:, 1], target[1], atol=1e-12)  # constant piece-y stays at constant height
    r = np.linalg.norm(out[:, [0, 2]], axis=1)
    assert np.allclose(r, 0.16, atol=1e-9)  # all points on the cylinder of radius R
    assert out[n // 2][[0, 2]] == pytest.approx([0.0, 0.16], abs=2e-3)  # the reference column sits at the point
    # distances along y are preserved
    v = np.array([[0.30, 0.05, 0.0], [0.30, 0.35, 0.0]])
    o2, _ = wrap_vertices(v, ref, geom, 0.0, False)
    assert np.linalg.norm(o2[1] - o2[0]) == pytest.approx(0.30, abs=1e-12)


def test_10_wrap_warns_and_clamps_when_wider_than_circumference() -> None:
    geom, _ = cylinder_geometry(0.10)
    x = np.linspace(0.0, 1.0, 50)  # 1 m wide, circumference 0.628 m
    local = np.column_stack([x, np.zeros(50), np.zeros(50)])
    out, info = wrap_vertices(local, np.array([0.5, 0.0, 0.0]), geom, 0.0, False)
    assert info.clamped and "WrapWidthExceededWarning" in info.warning
    r = np.linalg.norm(out[:, [0, 2]], axis=1)
    assert np.allclose(r, 0.10, atol=1e-9)


def test_10_wrap_rotation_and_mirror_match_the_flat_orientation() -> None:
    geom, _ = cylinder_geometry(0.15)
    local = np.array([[0.1, 0.0, 0.0], [0.0, 0.1, 0.0]])
    flat = apply_matrix(
        rigid_matrix(
            target_frame(np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0])),
            0.0,
            np.zeros(3),
            geom.axis_point + geom.radius * geom.radial,
        ),
        local,
    )
    wrapped, _ = wrap_vertices(local, np.zeros(3), geom, 0.0, False)
    # +x of the piece goes toward the wearer's left (+x), +y goes up, in both modes
    assert (flat[0] - flat[1])[0] > 0 and (wrapped[0] - wrapped[1])[0] > 0
    assert wrapped[1][1] > wrapped[0][1]
    rot, _ = wrap_vertices(local, np.zeros(3), geom, 90.0, False)  # counter-clockwise quarter turn: +x piece -> +y
    assert rot[0][1] - geom.axis_point[1] == pytest.approx(0.1, abs=1e-9)


# 11 -----------------------------------------------------------------------------------------------------------
def test_11_mirror_is_a_mirror_image_with_consistent_winding() -> None:
    v, f = l_piece()
    mv, mf = mirror_mesh(v, f, 0.15)
    assert mv[:, 0] == pytest.approx(0.30 - v[:, 0])
    assert mv[:, 1:] == pytest.approx(v[:, 1:])
    assert np.abs(normals(mv, mf) - normals(v, f)).max() < 1e-12  # all triangle normals point the same way as before
    assert areas(mv, mf).sum() == pytest.approx(areas(v, f).sum())
    # placing the mirrored piece keeps the normal on the point's normal
    m = rigid_matrix(target_frame(POINT_N, POINT_UP, False), 0.0, np.array([0.15, 0.2, 0.0]), POINT_POS)
    assert np.abs(normals(apply_matrix(m, mv), mf) - POINT_N).max() < 1e-9


# refinement used for wrapping -------------------------------------------------------------------------------------
def test_refine_mesh_keeps_area_and_boundary_and_limits_edges() -> None:
    v, f = l_piece()
    rv, rf = refine_mesh(v, f, 0.02)
    assert areas(rv, rf).sum() == pytest.approx(areas(v, f).sum(), abs=1e-12)
    e = np.concatenate([rf[:, [0, 1]], rf[:, [1, 2]], rf[:, [2, 0]]])
    assert np.linalg.norm(rv[e[:, 0]] - rv[e[:, 1]], axis=1).max() <= 0.02 + 1e-12
    assert np.all(normals(rv, rf)[:, 2] > 0.999999)
    # no cracks: every edge is shared by at most two triangles, and boundary edges form closed loops
    es = np.sort(e, axis=1)
    uniq, counts = np.unique(es, axis=0, return_counts=True)
    assert counts.max() == 2
    boundary = uniq[counts == 1]
    _, deg = np.unique(boundary.ravel(), return_counts=True)
    assert np.all(deg == 2)


# 12 -----------------------------------------------------------------------------------------------------------
def rect_bundle(name: str, w: float, h: float, names: dict[str, float]) -> dict[str, Any]:
    """Piece bundle (rectangle w x h) with named points on the bottom edge at the given x positions."""
    outline = [[0, 0], [w, 0], [w, h], [0, h]]
    per = 2 * (w + h)
    pts = [{"name": k, "label": k, "xy_m": [x, 0.0], "on_outline": True, "outline_s_m": x} for k, x in names.items()]
    pts.append({"name": f"{name}_TR", "label": "tr", "xy_m": [w, h], "on_outline": True, "outline_s_m": w + h})
    return {
        "schema_version": 1,
        "name": name,
        "units": "m",
        "obj": f"{name}_m.obj",
        "bbox_m": {"min": [0, 0], "max": [w, h]},
        "outline_m": outline,
        "named_points": pts,
        "edges": [],
        "fold_edge": None,
        "grainline": None,
        "default_reference": {"type": "bbox_centre", "xy_m": [w / 2, h / 2]},
        "_perimeter": per,
    }


def test_12_stitch_equal_edges_report_zero_and_one_cm_mismatch_exactly() -> None:
    a = rect_bundle("A", 0.40, 0.30, {"P1": 0.05, "P2": 0.25})
    b = rect_bundle("B", 0.50, 0.30, {"Q1": 0.10, "Q2": 0.30})
    b2 = rect_bundle("C", 0.50, 0.30, {"R1": 0.10, "R2": 0.31})
    st = {
        "a": {"piece": "A", "from_point": "P1", "to_point": "P2"},
        "b": {"piece": "B", "from_point": "Q1", "to_point": "Q2"},
    }
    r = stitch_report(st, {"A": a, "B": b}, {})
    assert r["a_length_cm"] == pytest.approx(20.0, abs=1e-9) and r["b_length_cm"] == pytest.approx(20.0, abs=1e-9)
    assert r["diff_cm"] == pytest.approx(0.0, abs=1e-9) and r["diff_pct"] == pytest.approx(0.0, abs=1e-9)
    st2 = {"a": st["a"], "b": {"piece": "C", "from_point": "R1", "to_point": "R2"}}
    r2 = stitch_report(st2, {"A": a, "C": b2}, {})
    assert r2["diff_cm"] == pytest.approx(-1.0, abs=1e-9) and r2["b_length_cm"] == pytest.approx(21.0, abs=1e-9)
    assert r2["diff_pct"] == pytest.approx(-100 / 21, abs=1e-9)
    assert abs(r2["diff_cm"]) == pytest.approx(1.0)
    # the other way round the outline is longer, so the short arc is chosen
    pts, length = edge_between(a, "P2", "P1")
    assert length == pytest.approx(0.20) and pts[0] == pytest.approx([0.25, 0.0])
    with pytest.raises(UnknownNamedPointError):
        edge_between(a, "P1", "nope")


def test_12_stitch_samples_are_resampled_by_arc_length() -> None:
    a = rect_bundle("A", 0.40, 0.30, {"P1": 0.05, "P2": 0.25})
    pts, length = edge_between(a, "P1", "P2", samples=11)
    assert len(pts) == 11 and length == pytest.approx(0.20)
    steps = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    assert np.allclose(steps, 0.02, atol=1e-12)


# integration with the real avatar and shirt pieces ---------------------------------------------------------------
def spec(*placements: dict[str, Any], **extra: Any) -> str:
    return json.dumps({"schema_version": 1, "placements": list(placements), **extra})


def test_engine_places_front_panel_on_chest_front(scene: dict[str, Path]) -> None:
    res = AlignmentEngine().align(
        scene["avatar"], scene["pieces"], spec({"piece": "FrontPanel", "point": "Chest_Front"}), scene["out"]
    )
    pl = res.doc["placements"][0]
    pts = {p["name"]: p for p in json.loads((scene["avatar"] / "arrangement_points.json").read_text())["points"]}
    p = pts["Chest_Front"]
    ref = np.array(pl["reference_position_m"])
    assert ref == pytest.approx(p["position_m"], abs=1e-6)  # default offset: the point's own position
    assert pl["bbox_cm"]["width"] == pytest.approx(37.25, abs=0.01) and pl["bbox_cm"]["height"] == pytest.approx(
        74.40, abs=0.01
    )
    m = np.array(pl["matrix"]).reshape(4, 4)
    assert m[:3, 2] == pytest.approx(p["normal"], abs=1e-5)  # the piece z axis is the point normal
    assert (scene["out"] / "alignment_result.json").is_file() and (scene["out"] / "scene_m.obj").is_file()
    assert (scene["out"] / "FrontPanel_placed_m.obj").is_file()


def test_engine_offset_replaces_default_offset(scene: dict[str, Path]) -> None:
    a = AlignmentEngine().align(scene["avatar"], scene["pieces"], spec({"piece": "Cuff", "point": "Chest_Front"}))
    b = AlignmentEngine().align(
        scene["avatar"], scene["pieces"], spec({"piece": "Cuff", "point": "Chest_Front", "offset_m": 0.08})
    )
    p = {q["name"]: q for q in json.loads((scene["avatar"] / "arrangement_points.json").read_text())["points"]}[
        "Chest_Front"
    ]
    d = np.array(b.doc["placements"][0]["reference_position_m"]) - np.array(
        a.doc["placements"][0]["reference_position_m"]
    )
    assert d == pytest.approx((0.08 - p["offset_m"]) * np.array(p["normal"]), abs=1e-5)


def test_engine_wrap_on_the_torso(scene: dict[str, Path]) -> None:
    res = AlignmentEngine().align(
        scene["avatar"], scene["pieces"], spec({"piece": "BackPanel", "point": "Chest_Back", "wrap": "around"})
    )
    pl = res.doc["placements"][0]
    v = np.array(pl["wrapped_vertices"])
    f = np.array(pl["wrapped_faces"])
    assert len(v) > 500 and f.max() < len(v)
    assert pl["bbox_cm"]["width"] == pytest.approx(34.75, abs=0.01)  # the flat size is what is reported
    assert not any("WrapWidthExceeded" in w for w in pl["warnings"])
    # nothing is scaled: mesh area is the flat area
    assert areas(v, f).sum() > 0.2 and pl["fit"]["wrap_angle_deg"] > 30


def test_engine_errors_are_specific(scene: dict[str, Path]) -> None:
    eng = AlignmentEngine()
    with pytest.raises(UnknownPieceError, match="Nope"):
        eng.align(scene["avatar"], scene["pieces"], spec({"piece": "Nope", "point": "Chest_Front"}))
    with pytest.raises(UnknownPointError, match="Chest_Nowhere"):
        eng.align(scene["avatar"], scene["pieces"], spec({"piece": "Cuff", "point": "Chest_Nowhere"}))
    with pytest.raises(UnknownNamedPointError):
        eng.align(scene["avatar"], scene["pieces"], spec({"piece": "Cuff", "point": "Chest_Front", "reference": "zzz"}))
    with pytest.raises(UnknownMeasurementError):
        eng.align(
            scene["avatar"],
            scene["pieces"],
            spec(rings=[{"name": "x", "pieces": ["Cuff"], "avatar_measurement": "foo"}]),
        )


def test_engine_ring_check(scene: dict[str, Path]) -> None:
    res = AlignmentEngine().align(
        scene["avatar"],
        scene["pieces"],
        spec(rings=[{"name": "chest", "pieces": ["FrontPanel", "BackPanel"], "avatar_measurement": "chest_circ"}]),
    )
    r = res.doc["ring_reports"][0]
    assert r["sum_width_cm"] == pytest.approx(37.25 + 34.75, abs=0.01) and r["avatar_cm"] == pytest.approx(
        100.0, abs=0.01
    )
    assert r["ease_cm"] == pytest.approx(r["sum_width_cm"] - 100.0, abs=1e-9)


def test_engine_stitch_report_on_placed_pieces(scene: dict[str, Path]) -> None:
    bundle = json.loads((scene["pieces"] / "Cuff.piece.json").read_text())
    names = [p["name"] for p in bundle["named_points"] if p["on_outline"]]
    st = {
        "a": {"piece": "Cuff", "from_point": names[0], "to_point": names[1]},
        "b": {"piece": "CuffInterface", "from_point": names[0], "to_point": names[1]},
    }
    res = AlignmentEngine().align(
        scene["avatar"],
        scene["pieces"],
        spec(
            {"piece": "Cuff", "point": "Chest_Front"},
            {"piece": "CuffInterface", "point": "Chest_Front", "offset_m": 0.1},
            stitches=[st],
        ),
    )
    s = res.doc["stitch_reports"][0]
    assert abs(s["diff_cm"]) < 1e-6 and len(s["segments"]) > 5
    gap = np.linalg.norm(np.array(s["segments"])[:, 0] - np.array(s["segments"])[:, 1], axis=1)
    assert gap == pytest.approx(0.1 - 0.03, abs=1e-6)  # pieces sit 7 cm apart along the same normal


def test_engine_is_deterministic(scene: dict[str, Path], tmp_path: Path) -> None:
    s = spec(
        {"piece": "FrontPanel", "point": "Chest_Front", "wrap": "around"},
        {"piece": "Cuff", "point": "Hand_Outer_L", "mirror": True},
    )
    for d in ("a", "b"):
        AlignmentEngine().align(scene["avatar"], scene["pieces"], s, tmp_path / d)
    for f in sorted(p.name for p in (tmp_path / "a").iterdir()):
        assert (tmp_path / "a" / f).read_bytes() == (tmp_path / "b" / f).read_bytes(), f

"""Clone & mirror, derived pieces, pair placement and side checks (alignment_engine Part A)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy.spatial import cKDTree

from alignment_engine import AlignmentEngine
from alignment_engine.errors import (
    DerivedOfDerivedError,
    NoMirroredPointError,
    ReferenceNotOnAxisError,
    UnknownNamedPointError,
)
from alignment_engine.loaders import read_pieces
from alignment_engine.mirror import mirror_clone
from alignment_engine.pairs import counterpart_point, place_pair
from alignment_engine.verify import surface_samples

CF = {"type": "edge", "from_point": "A33", "to_point": "A32"}  # the centre-front edge of FrontPanel


def spec(*placements: dict[str, Any], derived: list[dict[str, Any]] | None = None, **extra: Any) -> str:
    doc: dict[str, Any] = {"schema_version": 2, "placements": list(placements), **extra}
    if derived is not None:
        doc["derived_pieces"] = derived
    return json.dumps(doc)


def signed_area(p: np.ndarray) -> float:
    return float(0.5 * np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1]))


def perimeter(p: np.ndarray) -> float:
    return float(np.linalg.norm(np.roll(p, -1, axis=0) - p, axis=1).sum())


def edge_lengths(bundle: dict[str, Any]) -> list[float]:
    """Edge lengths between consecutive on-outline named points, recomputed from the arc positions."""
    on = sorted(q["outline_s_m"] for q in bundle["named_points"] if q["on_outline"])
    per = perimeter(np.array(bundle["outline_m"]))
    if not on:
        return []
    return sorted([*(b - a for a, b in zip(on[:-1], on[1:], strict=True)), on[0] + per - on[-1]])


def piece(scene: dict[str, Path], name: str) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    info = read_pieces(scene["pieces"])[name]
    v, f = info.mesh()
    return info.bundle, v, f


def axes_for(bundle: dict[str, Any]) -> list[dict[str, Any]]:
    on = [q for q in bundle["named_points"] if q["on_outline"]]
    w = bundle["bbox_m"]["max"][0]
    out: list[dict[str, Any]] = [{"type": "bbox_center"}, {"type": "x", "value_m": 0.3 * w}]
    if len(on) >= 2:
        a = on[0]
        b = next(q for q in on[1:] if np.hypot(q["xy_m"][0] - a["xy_m"][0], q["xy_m"][1] - a["xy_m"][1]) > 0.01)
        out.append({"type": "edge", "from_point": a["name"], "to_point": b["name"]})
    return out


# 1, 2, 3 ------------------------------------------------------------------------------------------------------
def test_1_mirror_preserves_lengths_area_perimeter_for_all_pieces_and_axes(scene: dict[str, Path]) -> None:
    n = 0
    for name in read_pieces(scene["pieces"]):
        bundle, v, f = piece(scene, name)
        for axis in axes_for(bundle):
            m = mirror_clone(bundle, v, f, axis, f"{name}_M")
            src, dst = np.array(bundle["outline_m"]), np.array(m.bundle["outline_m"])
            assert abs(abs(signed_area(src)) - abs(signed_area(dst))) < 1e-12, (name, axis)
            assert abs(perimeter(src) - perimeter(dst)) < 1e-12, (name, axis)
            a, b = edge_lengths(bundle), edge_lengths(m.bundle)
            assert len(a) == len(b) and np.abs(np.array(a) - np.array(b)).max(initial=0.0) < 1e-12, (name, axis)
            assert sum(e["length_m"] for e in m.bundle["edges"]) == pytest.approx(perimeter(dst), abs=1e-12)
            # every pairwise vertex distance is preserved
            d0 = np.linalg.norm(src[:, None] - src[None, :], axis=2)
            perm = np.r_[0, np.arange(len(src) - 1, 0, -1)]
            d1 = np.linalg.norm(dst[:, None] - dst[None, :], axis=2)
            assert np.abs(d0[np.ix_(perm, perm)] - d1).max() < 1e-12
            assert m.bundle["bbox_m"]["min"] == [0.0, 0.0] and np.array(m.bundle["outline_m"]).min() >= 0.0
            n += 1
    assert n >= 17 * 2


def test_2_mirroring_twice_restores_the_original(scene: dict[str, Path]) -> None:
    for name in ("FrontPanel", "ShortSleeve", "BackPanel", "Yoke"):
        bundle, v, f = piece(scene, name)
        for axis in axes_for(bundle):
            m1 = mirror_clone(bundle, v, f, axis, "X")
            m2 = mirror_clone(m1.bundle, m1.vertices, m1.faces, m1.bundle["axis"], name)
            assert np.abs(np.array(m2.bundle["outline_m"]) - np.array(bundle["outline_m"])).max() < 1e-12, (name, axis)
            assert np.abs(m2.vertices[:, :2] - np.array(bundle["outline_m"])).max() < 1e-12
            assert np.array_equal(m2.faces, np.array(f))  # original winding and triangle order
            a = {q["name"]: q["xy_m"] for q in bundle["named_points"]}
            for q in m2.bundle["named_points"]:
                assert np.abs(np.array(q["xy_m"]) - np.array(a[q["name"]])).max() < 1e-12
            assert m2.bundle["mirror_of"] == "X"


def test_3_normals_of_the_clone_point_to_plus_z(scene: dict[str, Path]) -> None:
    for name in read_pieces(scene["pieces"]):
        bundle, v, f = piece(scene, name)
        m = mirror_clone(bundle, v, f, {"type": "bbox_center"}, f"{name}_M")
        t = m.vertices[m.faces]
        nz = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])[:, 2]
        assert np.all(nz >= -1e-15) and nz.sum() > 0, name
        assert np.all(m.vertices[:, 2] == 0.0)


def test_clone_records_its_link_and_axis(scene: dict[str, Path]) -> None:
    bundle, v, f = piece(scene, "FrontPanel")
    m = mirror_clone(bundle, v, f, CF, "FrontPanel_L")
    b = m.bundle
    assert b["name"] == "FrontPanel_L" and b["mirror_of"] == "FrontPanel" and b["obj"] == "FrontPanel_L_m.obj"
    assert b["axis"] == CF and b["schema_version"] == 2
    w = bundle["bbox_m"]["max"][0]
    assert b["origin_shift_m"] == pytest.approx(
        [w, 0.0], abs=1e-12
    )  # reflected across x = 0, shifted back by the width
    assert {q["name"] for q in b["named_points"]} == {q["name"] for q in bundle["named_points"]}
    centre = next(q for q in b["named_points"] if q["name"] == "A33")
    assert centre["xy_m"] == pytest.approx([w, 0.0], abs=1e-12)  # the CF edge is now on the right of the clone
    with pytest.raises(UnknownNamedPointError):
        mirror_clone(bundle, v, f, {"type": "edge", "from_point": "A33", "to_point": "zzz"}, "x")


def test_non_vertical_edge_axis_reflects_correctly() -> None:
    bundle = {
        "schema_version": 2,
        "name": "T",
        "units": "m",
        "obj": "T_m.obj",
        "bbox_m": {"min": [0, 0], "max": [0.2, 0.1]},
        "outline_m": [[0, 0], [0.2, 0], [0.2, 0.1], [0, 0.1]],
        "named_points": [
            {"name": "P", "label": "P", "xy_m": [0.0, 0.0], "on_outline": True, "outline_s_m": 0.0},
            {"name": "Q", "label": "Q", "xy_m": [0.2, 0.1], "on_outline": True, "outline_s_m": 0.3},
        ],
        "edges": [],
        "fold_edge": None,
        "grainline": None,
        "default_reference": {"type": "bbox_centre", "xy_m": [0.1, 0.05]},
        "mirror_of": None,
        "axis": None,
        "origin_shift_m": None,
    }
    v = np.array([[0, 0, 0], [0.2, 0, 0], [0.2, 0.1, 0], [0, 0.1, 0]], dtype=float)
    f = np.array([[0, 1, 2], [0, 2, 3]])
    m = mirror_clone(bundle, v, f, {"type": "edge", "from_point": "P", "to_point": "Q"}, "T2")  # the diagonal
    # reflection across the diagonal of a 2:1 rectangle: the other points land on a different rectangle
    assert np.array(m.bundle["outline_m"]).min() >= 0 and abs(
        signed_area(np.array(m.bundle["outline_m"]))
    ) == pytest.approx(0.02)
    assert signed_area(np.array(m.bundle["outline_m"])) > 0  # still counter-clockwise
    p = {q["name"]: q["xy_m"] for q in m.bundle["named_points"]}
    sx, sy = m.bundle["origin_shift_m"]
    assert np.array(p["P"]) == pytest.approx(np.array([0.0, 0.0]) + [sx, sy], abs=1e-12)  # points on the axis stay put
    assert np.array(p["Q"]) == pytest.approx(np.array([0.2, 0.1]) + [sx, sy], abs=1e-12)


# derived pieces in a spec -----------------------------------------------------------------------------------------
def test_derived_pieces_are_built_and_cached(scene: dict[str, Path], tmp_path: Path) -> None:
    res = AlignmentEngine().align(
        scene["avatar"],
        scene["pieces"],
        spec(
            {"piece": "FrontPanel_L", "point": "Chest_Front", "reference": "A33"},
            derived=[{"id": "FrontPanel_L", "source": "FrontPanel", "op": "mirror", "axis": CF}],
        ),
        tmp_path,
    )
    d = res.doc["derived_pieces"][0]
    assert d["id"] == "FrontPanel_L" and d["source"] == "FrontPanel" and d["axis"] == CF
    assert (tmp_path / "FrontPanel_L_m.obj").is_file() and (tmp_path / "FrontPanel_L.piece.json").is_file()
    cached = json.loads((tmp_path / "FrontPanel_L.piece.json").read_text())
    assert cached["mirror_of"] == "FrontPanel" and d["bbox_cm"] == {"width": 37.25, "height": 74.4}
    assert d["suggested_stitch"]["a"]["piece"] == "FrontPanel" and d["suggested_stitch"]["b"]["piece"] == "FrontPanel_L"


def test_derived_of_derived_and_duplicates_are_errors(scene: dict[str, Path]) -> None:
    base = {"id": "FP_L", "source": "FrontPanel", "op": "mirror", "axis": {"type": "bbox_center"}}
    with pytest.raises(DerivedOfDerivedError, match="FP_L"):
        AlignmentEngine().align(
            scene["avatar"], scene["pieces"], spec(derived=[base, {**base, "id": "FP_LL", "source": "FP_L"}])
        )
    with pytest.raises(Exception, match="FrontPanel"):  # id collides with a real piece
        AlignmentEngine().align(scene["avatar"], scene["pieces"], spec(derived=[{**base, "id": "FrontPanel"}]))


# 4 ------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("wrap", ["flat", "around", "roll"])
def test_4_centre_line_pair_keeps_the_gap_across_the_centre_line(
    scene: dict[str, Path], tmp_path: Path, wrap: str
) -> None:
    eng = AlignmentEngine()
    derived = [{"id": "FrontPanel_L", "source": "FrontPanel", "op": "mirror", "axis": CF}]
    base = {"piece": "FrontPanel", "point": "Chest_Front", "reference": "A33", "wrap": wrap}
    pair = eng.pair(scene["avatar"], scene["pieces"], derived, base, "FrontPanel_L", "same_anchor")
    assert [p["piece"] for p in pair] == ["FrontPanel", "FrontPanel_L"]
    assert all(p["point"] == "Chest_Front" and p["reference"] == "A33" and p["wrap"] == wrap for p in pair)
    res = eng.align(scene["avatar"], scene["pieces"], spec(*pair, derived=derived), derived_dir=tmp_path)
    placed = {p.name: p for p in res.placed}
    b = json.loads((scene["pieces"] / "FrontPanel.piece.json").read_text())
    c = json.loads((tmp_path / "FrontPanel_L.piece.json").read_text())
    for name in ("A33", "A32"):
        pa = next(q["xy_m"] for q in b["named_points"] if q["name"] == name)
        pb = next(q["xy_m"] for q in c["named_points"] if q["name"] == name)
        wa = placed["FrontPanel"].transform(np.array([pa]))[0]
        wb = placed["FrontPanel_L"].transform(np.array([pb]))[0]
        # the CF points mirror across the centre line, a gap apart (they never touch: room for the seam)
        assert 0.03 - 0.001 <= np.linalg.norm(wa - wb) <= 0.035, name
        assert np.abs(wa * [-1, 1, 1] - wb).max() < 1e-4, name  # the point is at x = 0 within 0.1 mm
    sa = surface_samples(placed["FrontPanel"].vertices, placed["FrontPanel"].faces, 0.003)
    sb = surface_samples(placed["FrontPanel_L"].vertices, placed["FrontPanel_L"].faces, 0.003)
    # the gap across the centre line (a flat pair may widen a little where its outer edge meets the T-pose arm)
    assert 0.03 - 0.001 <= float(cKDTree(sb).query(sa)[0].min()) <= 0.035
    ext = {p["piece"]: p["extends"] for p in res.doc["placements"]}
    assert ext["FrontPanel"] != ext["FrontPanel_L"] and {ext["FrontPanel"], ext["FrontPanel_L"]} == {"left", "right"}
    assert not any("SIDE_" in w for p in res.doc["placements"] for w in p["warnings"])
    assert res.doc["placements"][0]["bbox_cm"] == res.doc["placements"][1]["bbox_cm"]


# 5 ------------------------------------------------------------------------------------------------------------
def test_5_side_pair_for_sleeves(scene: dict[str, Path]) -> None:
    eng = AlignmentEngine()
    derived = [{"id": "ShortSleeve_L", "source": "ShortSleeve", "op": "mirror", "axis": {"type": "bbox_center"}}]
    base = {"piece": "ShortSleeve", "point": "UpperArm_Outer_R", "rotation_deg": 25.0, "offset_m": 0.04}
    pair = eng.pair(scene["avatar"], scene["pieces"], derived, base, "ShortSleeve_L", "mirror_anchor")
    assert pair[1]["point"] == "UpperArm_Outer_L" and pair[1]["rotation_deg"] == -25.0
    assert pair[1]["offset_m"] == 0.04 and pair[1].get("reference") == pair[0].get("reference")
    res = eng.align(scene["avatar"], scene["pieces"], spec(*pair, derived=derived))
    r, lf = res.doc["placements"]
    pr, pl = np.array(r["reference_position_m"]), np.array(lf["reference_position_m"])
    assert np.abs(pr - pl * np.array([-1.0, 1.0, 1.0])).max() < 0.005
    mr, ml = np.array(r["matrix"]).reshape(4, 4), np.array(lf["matrix"]).reshape(4, 4)
    n_r, n_l = mr[:3, 2], ml[:3, 2]
    assert np.abs(n_r - n_l * np.array([-1.0, 1.0, 1.0])).max() < 2e-3  # normals mirror across x = 0 (0.1 deg)
    # the whole placed sleeve is the mirror image of the other one: every named point (the clone keeps the names; its
    # mesh is refined on its own, so vertices are not compared one to one)
    a, b = res.placed
    src = json.loads((scene["pieces"] / "ShortSleeve.piece.json").read_text())
    na = {q["name"]: q["xy_m"] for q in src["named_points"]}
    clone = eng.align(scene["avatar"], scene["pieces"], spec(*pair, derived=derived), derived_dir=scene["out"] / "d5")
    dst = json.loads((scene["out"] / "d5" / "ShortSleeve_L.piece.json").read_text())
    nb = {q["name"]: q["xy_m"] for q in dst["named_points"]}
    names = sorted(set(na) & set(nb))
    wa = a.transform(np.array([na[n] for n in names]))
    wb = clone.placed[1].transform(np.array([nb[n] for n in names])) * np.array([-1.0, 1.0, 1.0])
    assert np.abs(wa - wb).max() < 0.005


# 6 ------------------------------------------------------------------------------------------------------------
def test_6_pair_errors(scene: dict[str, Path]) -> None:
    eng = AlignmentEngine()
    derived = [{"id": "FrontPanel_L", "source": "FrontPanel", "op": "mirror", "axis": CF}]
    with pytest.raises(ReferenceNotOnAxisError, match="A33|bbox"):
        eng.pair(
            scene["avatar"],
            scene["pieces"],
            derived,
            {"piece": "FrontPanel", "point": "Chest_Front"},
            "FrontPanel_L",
            "same_anchor",
        )
    ok = eng.pair(
        scene["avatar"],
        scene["pieces"],
        derived,
        {"piece": "FrontPanel", "point": "Chest_Front", "reference": "A32"},
        "FrontPanel_L",
        "same_anchor",
    )
    assert len(ok) == 2
    # a centre point maps to itself; sided points swap L/R
    mid = eng.pair(
        scene["avatar"],
        scene["pieces"],
        derived,
        {"piece": "FrontPanel", "point": "Chest_Front"},
        "FrontPanel_L",
        "mirror_anchor",
    )
    assert mid[1]["point"] == "Chest_Front" and mid[1]["rotation_deg"] == 0
    with pytest.raises(NoMirroredPointError, match="Chest_FrontSide_L"):
        place_pair(
            {"piece": "A", "point": "Chest_FrontSide_L"}, "B", "mirror_anchor", point_names=["Chest_FrontSide_L"]
        )


def test_counterpart_point_names() -> None:
    names = ["UpperArm_Outer_L", "UpperArm_Outer_R", "Chest_Left", "Chest_Right", "Chest_Front"]
    assert counterpart_point("UpperArm_Outer_L", names) == "UpperArm_Outer_R"
    assert counterpart_point("Chest_Right", names) == "Chest_Left"
    assert counterpart_point("Chest_Front", names) == "Chest_Front"
    with pytest.raises(NoMirroredPointError):
        counterpart_point("UpperArm_Outer_R", ["UpperArm_Outer_R"])


# 7 ------------------------------------------------------------------------------------------------------------
def codes(res: Any) -> list[str]:
    return [w.split(":")[0] for p in res.doc["placements"] for w in p["warnings"]]


def test_7_side_unlikely_front_piece_on_a_back_point(scene: dict[str, Path]) -> None:
    res = AlignmentEngine().align(
        scene["avatar"], scene["pieces"], spec({"piece": "FrontPanel", "point": "Chest_Back"})
    )
    assert "SIDE_UNLIKELY" in codes(res)
    res = AlignmentEngine().align(
        scene["avatar"], scene["pieces"], spec({"piece": "BackPanel", "point": "Chest_Front"})
    )
    assert "SIDE_UNLIKELY" in codes(res)
    ok = AlignmentEngine().align(
        scene["avatar"], scene["pieces"], spec({"piece": "FrontPanel", "point": "Chest_Front"})
    )
    assert "SIDE_UNLIKELY" not in codes(ok)


def test_7_side_unlikely_uses_the_source_name_for_clones(scene: dict[str, Path]) -> None:
    derived = [{"id": "Mirrored_1", "source": "FrontPanel", "op": "mirror", "axis": CF}]
    res = AlignmentEngine().align(
        scene["avatar"], scene["pieces"], spec({"piece": "Mirrored_1", "point": "Chest_Back"}, derived=derived)
    )
    assert "SIDE_UNLIKELY" in codes(res)


def test_7_side_mismatch_left_extending_piece_on_a_right_point(scene: dict[str, Path]) -> None:
    derived = [{"id": "FrontPanel_L", "source": "FrontPanel", "op": "mirror", "axis": CF}]
    # FrontPanel extends toward the wearer's left of its CF edge: on an _R point that is a mismatch
    r = AlignmentEngine().align(
        scene["avatar"],
        scene["pieces"],
        spec({"piece": "FrontPanel", "point": "Thigh_Front_R", "reference": "A33"}, derived=derived),
    )
    assert r.doc["placements"][0]["extends"] == "left" and "SIDE_MISMATCH" in codes(r)
    # its mirror image extends to the right: on an _L point that is a mismatch, on the _R point it is fine
    r = AlignmentEngine().align(
        scene["avatar"],
        scene["pieces"],
        spec({"piece": "FrontPanel_L", "point": "Thigh_Front_L", "reference": "A33"}, derived=derived),
    )
    assert r.doc["placements"][0]["extends"] == "right" and "SIDE_MISMATCH" in codes(r)
    r = AlignmentEngine().align(
        scene["avatar"],
        scene["pieces"],
        spec({"piece": "FrontPanel_L", "point": "Thigh_Front_R", "reference": "A33"}, derived=derived),
    )
    assert r.doc["placements"][0]["extends"] == "right" and "SIDE_MISMATCH" not in codes(r)

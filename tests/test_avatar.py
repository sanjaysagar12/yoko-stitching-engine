"""avatar_engine acceptance tests 1 to 7 (real SMPL-X files from models/smplx; skipped if they are absent)."""

from __future__ import annotations

import json
import pickle
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from avatar_engine import AvatarEngine
from avatar_engine.bodymodel import BodyModel
from avatar_engine.errors import FitDidNotConvergeError, ModelFilesNotFoundError
from avatar_engine.modelfiles import convert_smpl_pkl, load_model, locate
from avatar_engine.points import _dist_point_triangles

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models" / "smplx"
BODIES = ROOT / "assert" / "body"
needs_models = pytest.mark.skipif(not (MODELS / "SMPLX_MALE.npz").is_file(), reason="models/smplx not present")
pytestmark = needs_models

_MODEL_CACHE: dict[str, BodyModel] = {}


def cached_factory(model_dir: Path, gender: str, model_type: str) -> BodyModel:
    key = f"{model_dir}|{gender}|{model_type}"
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = load_model(locate(model_dir, gender, model_type))
    return _MODEL_CACHE[key]


def body_text(name: str = "male_173", **changes: Any) -> str:
    doc = json.loads((BODIES / f"{name}.json").read_text())
    for k, v in changes.items():
        if k == "options":
            doc["options"] = {**doc.get("options", {}), **v}
        else:
            doc["measurements"][k] = v
    return json.dumps(doc)


def build(text: str, out: Path, write: bool = True) -> Any:
    return AvatarEngine(model_factory=cached_factory).build(text, MODELS, out, write)


@pytest.fixture(scope="module")
def male(tmp_path_factory: pytest.TempPathFactory) -> Any:
    return build(body_text(), tmp_path_factory.mktemp("male"))


def obj_vertices(path: Path) -> np.ndarray:
    return np.array([ln.split()[1:4] for ln in path.read_text().splitlines() if ln.startswith("v ")], dtype=float)


# 1 ------------------------------------------------------------------------------------------------------------
def test_1_height_and_circumferences(male: Any) -> None:
    rows = male.report["measurements"]
    assert abs(rows["height"]["post_error_cm"]) < 0.1  # 1 mm
    for k, row in rows.items():
        if row["kind"] == "circumference" and row["target_cm"] is not None:
            assert abs(row["post_error_cm"]) < 0.2, k  # 2 mm
            assert row["pre_error_cm"] is not None  # the report lists pre-correction errors too


def test_1_ring_correction_does_the_work_with_few_betas(tmp_path: Path) -> None:
    r = build(body_text(options={"num_betas": 10}), tmp_path)
    rows = r.report["measurements"]
    assert r.report["num_betas_used"] == 10
    pre = [abs(rows[k]["pre_error_cm"]) for k in rows if rows[k]["kind"] == "circumference" and rows[k]["target_cm"]]
    post = [abs(rows[k]["post_error_cm"]) for k in rows if rows[k]["kind"] == "circumference" and rows[k]["target_cm"]]
    assert max(pre) > 0.3, "10 betas should leave visible circumference errors before correction"
    assert max(post) < 0.2
    assert abs(rows["height"]["post_error_cm"]) < 0.1


def test_1_components_used_is_reported(tmp_path: Path) -> None:
    import copy

    small = copy.copy(cached_factory(MODELS, "male", "auto"))  # a model that only provides 12 shape components
    small.num_betas = 12  # type: ignore[misc]
    eng = AvatarEngine(model_factory=lambda d, g, t: small)
    r = eng.build(body_text(options={"num_betas": 40}), MODELS, tmp_path, write=False)
    assert r.bundle["num_betas_requested"] == 40 and r.bundle["num_betas_used"] == 12
    assert len(r.bundle["betas"]) == 12 and r.report["num_betas_used"] == 12
    assert any("12 shape components" in w for w in r.warnings)


def test_1_unreachable_height_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(FitDidNotConvergeError, match="height"):
        build(body_text(height=260), tmp_path, write=False)


# 2 ------------------------------------------------------------------------------------------------------------
def test_2_scale_is_metres_and_feet_on_floor(male: Any) -> None:
    v = obj_vertices(male.paths["mesh"])
    assert v[:, 1].max() - v[:, 1].min() == pytest.approx(1.73, abs=0.001)
    assert v[:, 1].min() == pytest.approx(0.0, abs=1e-6)
    assert male.bundle["bbox_m"]["max"][1] == pytest.approx(1.73, abs=0.001)
    assert male.bundle["units"] == "m"


def test_2_axes_front_is_plus_z_left_is_plus_x(male: Any) -> None:
    pts = {p.spec.name: p for p in male.points}
    assert pts["Chest_Front"].position_m[2] > 0 > pts["Chest_Back"].position_m[2]
    assert pts["Chest_Left"].position_m[0] > 0 > pts["Chest_Right"].position_m[0]
    assert pts["UpperArm_Outer_L"].position_m[0] > 0 > pts["UpperArm_Outer_R"].position_m[0]
    assert pts["Hip_Left"].position_m[0] > 0, "hips: the volume axis points down but 90 deg is still +x"


# 3 ------------------------------------------------------------------------------------------------------------
def test_3_determinism(tmp_path: Path) -> None:
    a = build(body_text(), tmp_path / "a")
    b = build(body_text(), tmp_path / "b")
    assert a.paths.keys() == b.paths.keys() and len(a.paths) == 5
    for k in a.paths:
        assert a.paths[k].read_bytes() == b.paths[k].read_bytes(), k


# 4 ------------------------------------------------------------------------------------------------------------
def test_4_chest_circ_moves_chest_front(tmp_path: Path) -> None:
    a = build(body_text(), tmp_path / "a", write=False)
    b = build(body_text(chest_circ=110), tmp_path / "b", write=False)
    pa = {p.spec.name: p for p in a.points}
    pb = {p.spec.name: p for p in b.points}
    assert list(pa) == list(pb) and len(pa) == 80
    out = float(pb["Chest_Front"].position_m[2] - pa["Chest_Front"].position_m[2])
    assert out == pytest.approx(0.016, abs=0.005)


# 5 ------------------------------------------------------------------------------------------------------------
def test_5_points_sit_at_their_offset_and_face_outwards(male: Any) -> None:
    tri = male.vertices[male.faces]
    vols = {b.name: b for b in male.volumes}
    for p in male.points:
        near = tri[np.linalg.norm(tri.mean(axis=1) - p.position_m, axis=1) < 0.3]
        d = _dist_point_triangles(p.position_m, near)
        assert d == pytest.approx(p.spec.offset_m, abs=0.001), p.spec.name
        assert np.linalg.norm(p.normal) == pytest.approx(1.0) and np.linalg.norm(p.up) == pytest.approx(1.0)
        bv = vols[p.spec.bv]
        axis_pt = bv.start + float(np.clip((p.position_m - bv.start) @ bv.axis, 0, bv.length_m)) * bv.axis
        radial = p.position_m - axis_pt
        assert p.normal @ radial > 0, p.spec.name  # points away from the body axis
        assert abs(p.up @ p.normal) < 1e-6 and np.allclose(np.cross(p.up, p.normal), p.tangent, atol=1e-6)


# 6 ------------------------------------------------------------------------------------------------------------
def test_6_left_right_pairs_mirror(male: Any) -> None:
    pts = {p.spec.name: p for p in male.points}
    pairs = [(n, n[:-2] + "_R") for n in pts if n.endswith("_L")]
    pairs += [(f"{b}_Left", f"{b}_Right") for b in ("Head", "Neck", "Chest", "Waist", "Hip")]
    assert len(pairs) == 24 + 6 + 5  # limbs, FrontSide/BackSide of chest/waist/hip, Left/Right of the column
    for left, right in pairs:
        pl, pr = pts[left].position_m, pts[right].position_m
        assert np.abs(pl - pr * np.array([-1.0, 1.0, 1.0])).max() < 0.005, left
    # central front/back points lie on the mid plane
    for n in ("Chest_Front", "Waist_Back", "Neck_Front", "Head_Back"):
        assert abs(pts[n].position_m[0]) < 0.005


# 7 ------------------------------------------------------------------------------------------------------------
def test_7_smplx_loads() -> None:
    f = locate(MODELS, "female")
    assert f.model_type == "smplx" and f.path.name == "SMPLX_FEMALE.npz"
    m = load_model(f)
    assert m.model_type == "smplx" and len(m.rest(np.zeros(3))[0]) == 10475 and m.symmetrized


def test_7_missing_model_gives_clear_error(tmp_path: Path) -> None:
    (tmp_path / "readme.txt").write_text("no models here")
    with pytest.raises(ModelFilesNotFoundError) as ei:
        locate(tmp_path, "male")
    msg = str(ei.value)
    assert str(tmp_path) in msg and "SMPLX_MALE.npz" in msg and "SMPL_MALE" in msg and "nothing is downloaded" in msg
    with pytest.raises(ModelFilesNotFoundError):
        locate(tmp_path / "does_not_exist", "female", "smplx")


def test_7_gender_falls_back_to_neutral(tmp_path: Path) -> None:
    (tmp_path / "SMPLX_NEUTRAL.npz").write_bytes(b"x")
    f = locate(tmp_path, "male")
    assert f.gender == "neutral" and "using 'neutral'" in f.warnings[0]


def test_7_smpl_files_in_subfolder_are_found(tmp_path: Path) -> None:
    (tmp_path / "smpl").mkdir()
    (tmp_path / "smpl" / "SMPL_FEMALE.pkl").write_bytes(b"x")
    f = locate(tmp_path, "female", "smpl")
    assert f.model_type == "smpl" and f.path.name == "SMPL_FEMALE.pkl"
    f = locate(tmp_path, "female", "auto")
    assert f.model_type == "smpl"


def test_7_smpl_pkl_converts_without_chumpy(tmp_path: Path) -> None:
    """A pickle that references chumpy classes (module absent here) converts to a plain npz."""
    mod = types.ModuleType("chumpy.ch")

    class Ch:  # stand-in used only to *create* the pickle
        def __init__(self, x: Any) -> None:
            self.x = x

    Ch.__module__, Ch.__qualname__ = "chumpy.ch", "Ch"
    mod.Ch = Ch  # type: ignore[attr-defined]
    sys.modules["chumpy"], sys.modules["chumpy.ch"] = types.ModuleType("chumpy"), mod
    try:
        blob = pickle.dumps(
            {
                "v_template": Ch(np.zeros((4, 3))),
                "shapedirs": Ch(np.ones((4, 3, 10))),
                "J_regressor": np.eye(2, 4),
                "weights": np.ones((4, 2)) / 2,
                "f": np.array([[0, 1, 2]]),
            }
        )
    finally:
        del sys.modules["chumpy"], sys.modules["chumpy.ch"]
    pkl = tmp_path / "SMPL_MALE.pkl"
    pkl.write_bytes(blob)
    assert "chumpy" not in sys.modules
    npz = convert_smpl_pkl(pkl, tmp_path / "cache" / "SMPL_MALE.npz")
    z = np.load(npz, allow_pickle=False)
    assert z["shapedirs"].shape == (4, 3, 10) and z["J_regressor"].shape == (2, 4) and z["f"].shape == (1, 3)


def test_7_unsafe_pickle_is_refused(tmp_path: Path) -> None:
    pkl = tmp_path / "SMPL_MALE.pkl"
    pkl.write_bytes(pickle.dumps(eval))  # a builtin callable that must not be loaded
    with pytest.raises(pickle.UnpicklingError):
        convert_smpl_pkl(pkl, tmp_path / "x.npz")


def test_7_npz_is_opened_without_pickle(tmp_path: Path) -> None:
    arr = np.empty(1, dtype=object)
    arr[0] = {"a": 1}
    np.savez(tmp_path / "SMPLX_MALE.npz", part2num=arr)  # object array requires pickle: must not be loaded
    from avatar_engine.errors import ModelFormatError

    with pytest.raises(ModelFormatError, match="missing arrays"):
        load_model(locate(tmp_path, "male"))


def test_extra_torso_points_exist_with_the_specified_angles(male: Any) -> None:
    pts = {p.spec.name: p for p in male.points}
    want = {"FrontSide_L": 40.0, "FrontSide_R": 320.0, "BackSide_L": 140.0, "BackSide_R": 220.0}
    for bv in ("Chest", "Waist", "Hip"):
        for label, angle in want.items():
            p = pts[f"{bv}_{label}"]
            assert p.spec.bv == bv and p.spec.angle_deg == angle and p.spec.offset_m == 0.03
            assert (p.position_m[0] > 0) == label.endswith("_L")  # +x = the wearer's left
            assert (p.position_m[2] > 0) == label.startswith("Front")
        for n in ("Front", "Back", "Left", "Right"):
            assert f"{bv}_{n}" in pts  # the centre points are still there
    assert [p.spec.name for p in male.points].count("Chest_Front") == 1
    # a front-side point lies between the front and the side point of the same level, around the same axis
    f, s_, side = pts["Chest_Front"].position_m, pts["Chest_FrontSide_L"].position_m, pts["Chest_Left"].position_m
    assert f[0] < s_[0] < side[0] and f[2] > s_[2] > side[2] - 0.2

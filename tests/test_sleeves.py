"""Sleeves on the arm points (synthetic body), placed one at a time as the web app does.

The old failure: a 62 cm full sleeve centred on ``UpperArm_Outer`` (a 26 cm volume) reached into the torso, and the
solver cleared the torso by growing the radius to ``max_standoff``: a 35 cm arc over the shoulder, reported valid. Now
a limb piece is registered at the limb root (cap at the shoulder, ``anchor``), clears only its own limb by its radius
and slides away from every other body part; whatever still cannot sit at its point is a violation with a hint.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from alignment_engine import AlignmentEngine
from alignment_engine.loaders import read_avatar
from alignment_engine.verify import AvatarSurface
from tests.test_layout import CLEAR, body_lower_bound

CF = {"type": "edge", "from_point": "A33", "to_point": "A32"}
DERIVED = [
    {"id": "FrontPanel_L", "source": "FrontPanel", "op": "mirror", "axis": CF},
    {"id": "BackPanel_L", "source": "BackPanel", "op": "mirror", "axis": {"type": "x", "value_m": 0.0}},
    {"id": "ShortSleeve_L", "source": "ShortSleeve", "op": "mirror", "axis": {"type": "bbox_center"}},
    {"id": "FullSleeve_L", "source": "FullSleeve", "op": "mirror", "axis": {"type": "bbox_center"}},
]
FLIP = np.array([-1.0, 1.0, 1.0])


def pl(piece: str, point: str, ref: str | None = None, **kw: Any) -> dict[str, Any]:
    d: dict[str, Any] = {"piece": piece, "point": point, "wrap": "roll", **kw}
    if ref:
        d["reference"] = ref
    return d


PANELS = [
    pl("FrontPanel", "Chest_Front", "A32"),
    pl("FrontPanel_L", "Chest_Front", "A32"),
    pl("BackPanel", "Chest_Back", "A24"),
    pl("BackPanel_L", "Chest_Back", "A24"),
]
SCENARIOS = {
    "S1_short_pair": [pl("ShortSleeve", "UpperArm_Outer_R"), pl("ShortSleeve_L", "UpperArm_Outer_L")],
    "S1_short_pair_clone_first": [pl("ShortSleeve_L", "UpperArm_Outer_L"), pl("ShortSleeve", "UpperArm_Outer_R")],
    "S2_full_pair": [pl("FullSleeve", "UpperArm_Outer_R"), pl("FullSleeve_L", "UpperArm_Outer_L")],
    "S3_full_and_short": [pl("FullSleeve", "UpperArm_Outer_R"), pl("ShortSleeve_L", "UpperArm_Outer_L")],
    "S4_panels_first": [*PANELS, pl("FullSleeve", "UpperArm_Outer_R"), pl("FullSleeve_L", "UpperArm_Outer_L")],
    "S5_panels_after": [pl("FullSleeve", "UpperArm_Outer_R"), pl("FullSleeve_L", "UpperArm_Outer_L"), *PANELS],
}
SLEEVES = ("ShortSleeve", "ShortSleeve_L", "FullSleeve", "FullSleeve_L")


@pytest.fixture(scope="module")
def ctx(shirt_pieces: Path, refine_cache: Path, tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    from tests.synthetic_avatar import build_avatar

    d = tmp_path_factory.mktemp("sleeves")
    av = build_avatar(d / "avatar")
    a = read_avatar(av)
    return {
        "avatar": av,
        "pieces": shirt_pieces,
        "derived": d / "derived",
        "cache": refine_cache,
        "av": a,
        "surface": AvatarSurface(a.vertices, a.faces),
    }


def call(ctx: dict[str, Any], method: str, pls: list[dict[str, Any]], eng: AlignmentEngine | None = None) -> Any:
    text = json.dumps({"schema_version": 4, "placements": pls, "derived_pieces": DERIVED})
    return getattr(eng or AlignmentEngine(), method)(
        ctx["avatar"], ctx["pieces"], text, derived_dir=ctx["derived"], cache_dir=ctx["cache"]
    )


def one_by_one(ctx: dict[str, Any], pls: list[dict[str, Any]]) -> Any:
    """Place the pieces one at a time, carrying every returned ``solved`` block (a pair may move both halves)."""
    eng, done, res = AlignmentEngine(), [], None
    for new in pls:
        res = call(ctx, "align", [*done, new], eng)
        docs = {d["piece"]: d for d in res.doc["placements"]}
        done = [{**p, "solved": docs[p["piece"]]["solved"]} for p in [*done, new]]
    return res


@pytest.fixture(scope="module")
def results(ctx: dict[str, Any]) -> dict[str, Any]:
    return {k: one_by_one(ctx, v) for k, v in SCENARIOS.items()}


def docs(res: Any) -> dict[str, Any]:
    return {d["piece"]: d for d in res.doc["placements"]}


def arm_axis(ctx: dict[str, Any], side: str) -> tuple[np.ndarray, np.ndarray, float]:
    """(shoulder joint, unit direction to the wrist, arm length) of one arm."""
    ua, fa = ctx["av"].volumes[f"UpperArm_{side}"], ctx["av"].volumes[f"Forearm_{side}"]
    d = fa.end - ua.start
    return ua.start, d / np.linalg.norm(d), float(np.linalg.norm(d))


def check_sleeve(ctx: dict[str, Any], res: Any, name: str, overhang: float = 0.10) -> None:
    p = next(x for x in res.placed if x.name == name)
    d = docs(res)[name]
    side = d["point"][-1]
    start, u, length = arm_axis(ctx, side)
    assert d["arrange"]["radius_m"] <= 0.12, (name, d["arrange"]["radius_m"])  # around the arm, not the torso
    rel = p.vertices - start
    axial = rel @ u
    centroid = p.vertices.mean(axis=0) - start
    assert np.linalg.norm(centroid - (centroid @ u) * u) <= 0.08, name  # centred on the arm axis line
    assert axial.min() >= -0.12 and axial.max() <= length + overhang, (name, axial.min(), axial.max())
    # the cap (top edge of the pattern) is at the shoulder end, the hem toward the wrist
    q = {n["name"]: np.array(n["world_m"]) for n in d["world_named_points"]}
    folder = ctx["pieces"] if (ctx["pieces"] / f"{name}.piece.json").is_file() else ctx["derived"]
    y = {n["name"]: n["xy_m"][1] for n in json.loads((folder / f"{name}.piece.json").read_text())["named_points"]}
    top = max(q, key=lambda k: y[k])
    assert (q[top] - start) @ u < float(np.median(axial)), name
    assert body_lower_bound(ctx["surface"], p) >= CLEAR - 0.001, name
    assert d["arrange"]["limits"] == [] and d["arrange"]["flags"] == [], (name, d["arrange"])


@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_sleeves_wrap_the_arm_with_the_cap_at_the_shoulder(
    ctx: dict[str, Any], results: dict[str, Any], scenario: str
) -> None:
    res = results[scenario]
    assert res.doc["validity"]["valid"] is True, (scenario, res.doc["validity"]["violations"])
    for name in docs(res):
        if name in SLEEVES:
            check_sleeve(ctx, res, name)


@pytest.mark.parametrize("scenario", ["S1_short_pair", "S1_short_pair_clone_first", "S2_full_pair", "S4_panels_first"])
def test_side_pairs_are_mirror_images(results: dict[str, Any], scenario: str) -> None:
    res = results[scenario]
    by = docs(res)
    for a in ("ShortSleeve", "FullSleeve"):
        if a not in by:
            continue
        b = a + "_L"
        ra, rb = by[a]["arrange"], by[b]["arrange"]
        assert ra["pair_with"] == b and abs(ra["radius_m"] - rb["radius_m"]) < 1e-6
        assert abs(ra["shift_m"][0] + rb["shift_m"][0]) < 1e-6 and abs(ra["shift_m"][1] - rb["shift_m"][1]) < 1e-6
        wa = {n["name"]: np.array(n["world_m"]) for n in by[a]["world_named_points"]}
        wb = {n["name"]: np.array(n["world_m"]) * FLIP for n in by[b]["world_named_points"]}
        assert max(float(np.abs(wa[k] - wb[k]).max()) for k in wa) < 0.001, scenario
    assert not [v for v in res.doc["validity"]["violations"] if v["kind"] == "PAIR_STACKED"]


def test_sleeves_do_not_depend_on_the_order(results: dict[str, Any]) -> None:
    """Panels before or after, pair or single: a sleeve on a given point ends at the same place."""
    ref = docs(results["S2_full_pair"])["FullSleeve"]["solved"]
    for scenario in ("S3_full_and_short", "S4_panels_first", "S5_panels_after"):
        got = docs(results[scenario])["FullSleeve"]["solved"]
        assert got["radius_m"] == pytest.approx(ref["radius_m"], abs=1e-6), scenario
        assert np.allclose(got["shift_m"], ref["shift_m"], atol=1e-6), scenario
    short = [docs(results[s])["ShortSleeve_L"]["solved"] for s in ("S1_short_pair", "S3_full_and_short")]
    assert short[0] == pytest.approx(short[1])


@pytest.mark.parametrize("point", ["UpperArm_Back_R", "UpperArm_Front_L", "UpperArm_Inner_R", "Forearm_Outer_L"])
def test_full_sleeve_on_any_arm_point_wraps_the_arm(ctx: dict[str, Any], point: str) -> None:
    res = call(ctx, "align", [pl("FullSleeve", point)])
    assert res.doc["validity"]["valid"] is True, (point, res.doc["validity"]["violations"])
    check_sleeve(ctx, res, "FullSleeve", 0.17 if "Inner" in point else 0.10)  # cap apex at the armpit: onto the hand
    a = docs(res)["FullSleeve"]["arrange"]
    assert a["anchor"]["mode"] == "root" and "root" in a["anchor"]["reason"]


def test_anchor_modes(ctx: dict[str, Any]) -> None:
    """point: centred on the point (it reaches into the torso): the solver slides it out along the arm, never grows
    the radius over the shoulder; tip: hem at the wrist."""
    res = call(ctx, "align", [pl("FullSleeve", "UpperArm_Outer_R", anchor="point")])
    a = docs(res)["FullSleeve"]["arrange"]
    assert "anchor" not in a and a["radius_m"] <= 0.12, a
    if res.doc["validity"]["valid"]:
        check_sleeve(ctx, res, "FullSleeve")
    else:
        assert all(x.get("hint") for x in res.doc["validity"]["violations"] if x["kind"] == "NOT_AT_POINT")
    res = call(ctx, "align", [pl("FullSleeve", "UpperArm_Outer_R", anchor="tip")])
    assert res.doc["validity"]["valid"] is True, res.doc["validity"]["violations"]
    start, u, length = arm_axis(ctx, "R")
    axial = (res.placed[0].vertices - start) @ u
    # the hem edge on the wrist; this sleeve is longer than the arm, so the solver slides its cap off the torso
    assert length - 0.01 <= axial.max() <= length + 0.05
    assert docs(res)["FullSleeve"]["arrange"]["anchor"]["mode"] == "tip"


def test_a_cuff_keeps_its_point(ctx: dict[str, Any]) -> None:
    res = call(ctx, "align", [pl("Cuff", "Forearm_Outer_R")])
    a = docs(res)["Cuff"]["arrange"]
    assert res.doc["validity"]["valid"] is True and "anchor" not in a
    assert a["point_distance_m"] <= a["standoff_m"] + 0.01


def test_side_point_under_a_t_pose_arm_is_a_clear_violation(ctx: dict[str, Any]) -> None:
    res = call(ctx, "align", [pl("FrontPanel", "Chest_Right")])
    v = res.doc["validity"]
    assert v["valid"] is False and v["placement_ok"] is False
    hints = [x["hint"] for x in v["violations"] if "hint" in x]
    assert hints and "arm" in hints[0] and "Chest_FrontSide_R" in hints[0]


def test_deterministic(ctx: dict[str, Any], results: dict[str, Any]) -> None:
    again = one_by_one(ctx, SCENARIOS["S4_panels_first"])
    assert [d["solved"] for d in again.doc["placements"]] == [
        d["solved"] for d in results["S4_panels_first"].doc["placements"]
    ]


def test_arrange_matches_one_by_one(ctx: dict[str, Any], results: dict[str, Any]) -> None:
    res = call(ctx, "arrange", SCENARIOS["S4_panels_first"])
    assert res.doc["validity"]["valid"] is True, res.doc["validity"]["violations"]
    inc = docs(results["S4_panels_first"])
    for name in ("FullSleeve", "FullSleeve_L"):
        assert docs(res)[name]["solved"]["radius_m"] == pytest.approx(inc[name]["solved"]["radius_m"], abs=0.002)

"""Mirror pairs never swap: a piece on a right-side point and its mirrored clone on the left counterpart stay each on
their own side of the mirror plane, side by side with the gap, on the synthetic body (``docs/alignment_pairs.md``).

The regression: two wide halves overlapping at the centre line were pushed toward each other until ``max_shift`` and
ended swapped (source on the left, clone on the right, both invalid).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from alignment_engine import AlignmentEngine

GAP = 0.03
AXES = {"bbox": {"type": "bbox_center"}, "x0": {"type": "x", "value_m": 0.0}}
TORSO_SIDES = [
    "Waist_FrontSide",
    "Chest_BackSide",
    "Chest_FrontSide",
    "Hip_FrontSide",
    "Hip_BackSide",
    "Waist_BackSide",
]


@pytest.fixture(scope="module")
def ctx(shirt_pieces: Path, refine_cache: Path, tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    from tests.synthetic_avatar import build_avatar

    d = tmp_path_factory.mktemp("pair_swap")
    return {"avatar": build_avatar(d / "avatar"), "pieces": shirt_pieces, "cache": refine_cache, "tmp": d}


def call(ctx: dict[str, Any], method: str, pls: list[dict[str, Any]], axis: str = "bbox", **opts: Any) -> Any:
    derived = [
        {"id": f"{src}_R", "source": src, "op": "mirror", "axis": AXES[axis]}
        for src in ("BackPanel", "FrontPanel", "ShortSleeve")
    ]
    doc: dict[str, Any] = {"schema_version": 4, "placements": pls, "derived_pieces": derived}
    if opts:
        doc["arrange_options"] = opts
    derived_dir = ctx["tmp"] / f"derived_{axis}"
    return getattr(AlignmentEngine(), method)(
        ctx["avatar"], ctx["pieces"], json.dumps(doc), derived_dir=derived_dir, cache_dir=ctx["cache"]
    )


def pl(piece: str, point: str, **kw: Any) -> dict[str, Any]:
    return {"piece": piece, "point": point, "wrap": "roll", **kw}


def one_by_one(ctx: dict[str, Any], pls: list[dict[str, Any]], axis: str = "bbox", **opts: Any) -> Any:
    """The web app: one piece per call, every returned ``solved`` block carried over."""
    done: list[dict[str, Any]] = []
    res = None
    for new in pls:
        res = call(ctx, "align", [*done, new], axis, **opts)
        docs = {d["piece"]: d for d in res.doc["placements"]}
        done = [{**p, "solved": docs[p["piece"]]["solved"]} for p in [*done, new]]
    return res


def views(res: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    return {p.name: p for p in res.placed}, {d["piece"]: d for d in res.doc["placements"]}


def kinds(res: Any) -> set[str]:
    return {v["kind"] for v in res.doc["validity"]["violations"]}


def assert_pair(res: Any, src: str, clone: str, src_side: float) -> None:
    """Each half on its own side, mirror images, valid, the gap between them."""
    placed, docs = views(res)
    xs, xc = (float(placed[n].vertices[:, 0].mean()) for n in (src, clone))
    assert np.sign(xs) == src_side and np.sign(xc) == -src_side, (xs, xc)
    a, b = docs[src]["arrange"], docs[clone]["arrange"]
    assert abs(a["radius_m"] - b["radius_m"]) < 1e-3
    assert abs(a["shift_m"][1] - b["shift_m"][1]) < 1e-3 and abs(a["shift_m"][0] + b["shift_m"][0]) < 1e-3
    assert abs(xs + xc) < 1e-3
    assert res.doc["validity"]["valid"], res.doc["validity"]["violations"]
    assert a["min_distance_to_pieces_m"].get(clone, np.inf) >= GAP - 5e-4  # far apart pieces are not listed
    assert not kinds(res) & {"GAP_PIECES", "PAIR_STACKED", "PAIR_NO_ROOM"}


@pytest.mark.parametrize("axis", list(AXES))
@pytest.mark.parametrize("order", ["source_first", "clone_first", "arrange"])
def test_waist_side_pair_does_not_swap(ctx: dict[str, Any], axis: str, order: str) -> None:
    src, clone = pl("BackPanel", "Waist_FrontSide_R"), pl("BackPanel_R", "Waist_FrontSide_L")
    alone = call(ctx, "align", [src], axis)
    x_alone = float(alone.placed[0].vertices[:, 0].mean())
    s_alone = alone.doc["placements"][0]["arrange"]["shift_m"][0]
    if order == "arrange":
        res = call(ctx, "arrange", [src, clone], axis)
    else:
        res = one_by_one(ctx, [src, clone] if order == "source_first" else [clone, src], axis)
    assert_pair(res, "BackPanel", "BackPanel_R", -1.0)
    placed, docs = views(res)
    assert float(placed["BackPanel"].vertices[:, 0].mean()) < x_alone  # moved outward (right is -x), never inward
    slide = abs(docs["BackPanel"]["arrange"]["shift_m"][0] - s_alone)
    assert 0.04 < slide < 0.08, slide


@pytest.mark.parametrize("point", TORSO_SIDES)
def test_torso_side_pairs(ctx: dict[str, Any], point: str) -> None:
    res = one_by_one(ctx, [pl("BackPanel", f"{point}_R"), pl("BackPanel_R", f"{point}_L")])
    assert_pair(res, "BackPanel", "BackPanel_R", -1.0)
    assert "max_standoff" not in views(res)[1]["BackPanel"]["arrange"]["limits"]


@pytest.mark.parametrize("piece,point", [("BackPanel", "Chest_Back"), ("FrontPanel", "Chest_Front")])
def test_centre_pairs_with_bbox_reference(ctx: dict[str, Any], piece: str, point: str) -> None:
    """A centre-line pair whose reference (bbox centre) is not on the centre line: placed by its edge, apart."""
    res = one_by_one(ctx, [pl(piece, point), pl(f"{piece}_R", point)], axis="x0" if piece == "BackPanel" else "bbox")
    placed, _ = views(res)
    side = float(np.sign(placed[piece].vertices[:, 0].mean()))
    assert_pair(res, piece, f"{piece}_R", side)


def test_sleeve_pair_on_the_arms(ctx: dict[str, Any]) -> None:
    res = one_by_one(ctx, [pl("ShortSleeve", "UpperArm_Outer_R"), pl("ShortSleeve_R", "UpperArm_Outer_L")])
    assert_pair(res, "ShortSleeve", "ShortSleeve_R", -1.0)


def test_no_room_is_reported_never_crossed(ctx: dict[str, Any]) -> None:
    """Not enough slide allowed to part the halves: one PAIR_NO_ROOM violation, each half still on its own side."""
    res = call(
        ctx, "arrange", [pl("BackPanel", "Waist_FrontSide_R"), pl("BackPanel_R", "Waist_FrontSide_L")], max_shift=0.02
    )
    placed, _ = views(res)
    assert placed["BackPanel"].vertices[:, 0].mean() < 0 < placed["BackPanel_R"].vertices[:, 0].mean()
    assert not res.doc["validity"]["valid"]
    room = [v for v in res.doc["validity"]["violations"] if v["kind"] == "PAIR_NO_ROOM"]
    assert len(room) == 1 and room[0]["value_m"] > room[0]["required_m"], room
    assert "farther apart" in room[0]["hint"]
    assert not kinds(res) & {"GAP_PIECES", "PAIR_STACKED", "NOT_AT_POINT"}  # the one issue, not three


def test_idempotent_stable_and_deterministic(ctx: dict[str, Any]) -> None:
    pair = [pl("BackPanel", "Waist_FrontSide_R"), pl("BackPanel_R", "Waist_FrontSide_L")]
    res = one_by_one(ctx, pair)
    again = one_by_one(ctx, pair)
    _, d1 = views(res)
    _, d2 = views(again)
    assert all(d1[n]["solved"] == d2[n]["solved"] for n in d1)  # same input, same answer
    carried = [{**p, "solved": d1[p["piece"]]["solved"]} for p in pair]
    rerun = call(ctx, "align", carried)
    added = call(ctx, "align", [*carried, pl("Yoke", "Chest_Back")])
    for r in (rerun, added):
        placed, _ = views(r)
        for n in ("BackPanel", "BackPanel_R"):
            assert np.abs(placed[n].vertices - views(res)[0][n].vertices).max() < 1e-3, n
    # the clone deleted and placed again: each half stays on its own side
    src_only = call(ctx, "align", [carried[0]])
    readd = one_by_one(ctx, [{**carried[0], "solved": views(src_only)[1]["BackPanel"]["solved"]}, pair[1]])
    assert_pair(readd, "BackPanel", "BackPanel_R", -1.0)

"""Clone (plain copy, ``op: "copy"``): an exact, independent second copy of a piece; never half of a mirror pair.

Runs on the synthetic body (no SMPL-X) with the male_shirt pieces.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from alignment_engine import AlignmentEngine, contracts
from alignment_engine.derived import build_derived, derived_doc, write_derived
from alignment_engine.errors import (
    AlignmentError,
    AxisError,
    DerivedCycleError,
    DerivedOfDerivedError,
    DuplicatePieceError,
    UnknownPieceError,
)
from alignment_engine.loaders import read_pieces

GAP = 0.03
MIRROR = {"type": "bbox_center"}


@pytest.fixture(scope="module")
def ctx(shirt_pieces: Path, refine_cache: Path, tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    from tests.synthetic_avatar import build_avatar

    d = tmp_path_factory.mktemp("clone")
    return {"avatar": build_avatar(d / "avatar"), "pieces": shirt_pieces, "cache": refine_cache, "tmp": d}


def copy(pid: str, src: str) -> dict[str, Any]:
    return {"id": pid, "source": src, "op": "copy"}


def mirror(pid: str, src: str) -> dict[str, Any]:
    return {"id": pid, "source": src, "op": "mirror", "axis": MIRROR}


def perimeter(p: np.ndarray) -> float:
    return float(np.linalg.norm(np.roll(p, -1, axis=0) - p, axis=1).sum())


def area(p: np.ndarray) -> float:
    return float(0.5 * np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1]))


# build_derived ----------------------------------------------------------------------------------------------------
def test_copy_is_geometrically_identical(shirt_pieces: Path, tmp_path: Path) -> None:
    pieces = read_pieces(shirt_pieces)
    src = pieces["Pocket"]
    built = build_derived([copy("Pocket_copy", "Pocket")], pieces, tmp_path)
    c = built["Pocket_copy"]
    b, sb = c.info.bundle, src.bundle
    v, f = c.info.mesh()
    sv, sf = src.mesh()
    assert np.array_equal(v, sv) and np.array_equal(f, sf)  # no reflection, same winding
    for k in ("outline_m", "named_points", "edges", "bbox_m", "default_reference", "grainline", "fold_edge"):
        assert b[k] == sb[k], k
    assert b.get("pattern_origin_m") == sb.get("pattern_origin_m")
    o, so = np.array(b["outline_m"]), np.array(sb["outline_m"])
    assert area(o) == area(so) and perimeter(o) == perimeter(so)
    assert b["name"] == "Pocket_copy" and b["obj"] == "Pocket_copy_m.obj" and b["clone_of"] == "Pocket"
    assert b["mirror_of"] is None and b["axis"] is None and b["origin_shift_m"] == [0.0, 0.0]
    paths = write_derived(tmp_path, built)  # validated against piece_bundle on write
    assert (tmp_path / "Pocket_copy_m.obj").read_text().startswith("# piece: Pocket_copy (copy of Pocket)")
    assert contracts.loads("piece_bundle", paths[1].read_text())["clone_of"] == "Pocket"
    doc = derived_doc(c)
    assert doc["op"] == "copy" and doc["clone_of"] == "Pocket" and doc["suggested_stitch"] is None
    assert "axis" not in doc


def test_copy_chains_any_order(shirt_pieces: Path, tmp_path: Path) -> None:
    pieces = read_pieces(shirt_pieces)
    defs = [  # dependants first: resolution does not depend on the order
        copy("S_L_copy", "S_L"),
        copy("Pocket_copy2", "Pocket_copy"),
        copy("Pocket_copy", "Pocket"),
        mirror("S_L", "ShortSleeve"),
    ]
    built = build_derived(defs, pieces, tmp_path)
    assert list(built) == [d["id"] for d in defs]
    cc = built["Pocket_copy2"].info.bundle
    assert cc["clone_of"] == "Pocket" and cc["outline_m"] == pieces["Pocket"].bundle["outline_m"]  # copy of the root
    mc, sl = built["S_L_copy"].info.bundle, built["S_L"].info.bundle
    assert mc["clone_of"] == "S_L" and mc["outline_m"] == sl["outline_m"] and mc["mirror_of"] is None
    assert np.array_equal(built["S_L_copy"].info.mesh()[1], built["S_L"].info.mesh()[1])


@pytest.mark.parametrize(
    "defs,err",
    [
        ([copy("P2", "Pocket"), mirror("P3", "P2")], DerivedOfDerivedError),
        ([mirror("P2", "Pocket"), mirror("P3", "P2")], DerivedOfDerivedError),
        ([copy("P2", "Pocket"), copy("P2", "Cuff")], DuplicatePieceError),
        ([copy("Cuff", "Pocket")], DuplicatePieceError),
        ([copy("P2", "Nope")], UnknownPieceError),
        ([copy("A", "B"), copy("B", "A")], DerivedCycleError),
        ([copy("A", "A")], DerivedCycleError),
        ([{**copy("P2", "Pocket"), "axis": MIRROR}], AxisError),
        ([{"id": "P2", "source": "Pocket", "op": "mirror"}], AxisError),
    ],
)
def test_derived_errors(shirt_pieces: Path, tmp_path: Path, defs: list[dict[str, Any]], err: type) -> None:
    with pytest.raises(err):
        build_derived(defs, read_pieces(shirt_pieces), tmp_path)


def test_spec_schema_rejects_axis_on_copy() -> None:
    doc = {"schema_version": 4, "placements": [], "derived_pieces": [{**copy("P2", "Pocket"), "axis": MIRROR}]}
    with pytest.raises(contracts.ContractError):
        contracts.loads("placement_spec", json.dumps(doc))
    ok = {**doc, "derived_pieces": [copy("P2", "Pocket"), {"id": "P3", "source": "Pocket", "axis": MIRROR}]}
    contracts.loads("placement_spec", json.dumps(ok))  # op defaults to mirror (old specs)


# engine -----------------------------------------------------------------------------------------------------------
def align(ctx: dict[str, Any], pls: list[dict[str, Any]], derived: list[dict[str, Any]], method: str = "align") -> Any:
    doc = {"schema_version": 4, "placements": pls, "derived_pieces": derived}
    return getattr(AlignmentEngine(), method)(
        ctx["avatar"], ctx["pieces"], json.dumps(doc), derived_dir=ctx["tmp"] / "derived", cache_dir=ctx["cache"]
    )


def assert_unpaired(res: Any) -> None:
    v = res.doc["validity"]
    assert v["valid"] and v["size_ok"], v["violations"]
    assert not [x for x in v["violations"] if x["kind"].startswith("PAIR_") or x["kind"] == "NOT_CONCENTRIC"]
    for p in res.doc["placements"]:
        assert p["mirrored"] is False and not p["arrange"]["pair_with"], p["piece"]
        assert not [f for f in p["arrange"]["flags"] if f.startswith("PAIR_") or f == "NOT_CONCENTRIC"]
    (d,) = res.doc["derived_pieces"]
    assert d["op"] == "copy" and d["suggested_stitch"] is None


def test_copy_on_the_other_side_is_not_a_pair(ctx: dict[str, Any]) -> None:
    pls = [
        {"piece": "Pocket", "point": "Chest_FrontSide_R"},
        {"piece": "Pocket_copy", "point": "Chest_FrontSide_L"},
    ]
    res = align(ctx, pls, [copy("Pocket_copy", "Pocket")])
    assert_unpaired(res)
    gaps = res.doc["placements"][0]["arrange"]["min_distance_to_pieces_m"]
    assert gaps.get("Pocket_copy", np.inf) >= GAP - 5e-4
    placed = {p.name: p for p in res.placed}
    assert placed["Pocket"].vertices[:, 0].mean() < 0 < placed["Pocket_copy"].vertices[:, 0].mean()
    # same handedness: the copy is the source translated, not mirrored (the right-hand point frame is not the left's)
    alone = align(ctx, pls[:1], [copy("Pocket_copy", "Pocket")])
    assert np.abs(alone.placed[0].vertices - placed["Pocket"].vertices).max() < 1e-6  # the copy does not move it
    moved = align(ctx, [pls[0], {**pls[1], "point": "Waist_FrontSide_L"}], [copy("Pocket_copy", "Pocket")])
    assert np.abs({p.name: p for p in moved.placed}["Pocket"].vertices - placed["Pocket"].vertices).max() < 1e-6


@pytest.mark.parametrize("method", ["align", "arrange"])
def test_copy_on_the_same_point_is_valid_and_apart(ctx: dict[str, Any], method: str) -> None:
    pls = [{"piece": "Pocket", "point": "Chest_FrontSide_R"}, {"piece": "Pocket_copy", "point": "Chest_FrontSide_R"}]
    res = align(ctx, pls, [copy("Pocket_copy", "Pocket")], method)
    assert_unpaired(res)
    a = res.doc["placements"][0]["arrange"]
    need = a["required_gap_to_pieces_m"]["Pocket_copy"]  # side by side: gap_piece; layered: gap_layer
    assert a["min_distance_to_pieces_m"]["Pocket_copy"] >= need - 5e-4
    placed = {p.name: p for p in res.placed}
    assert np.abs(placed["Pocket"].vertices - placed["Pocket_copy"].vertices).max() > 0.02  # never stacked


def test_pair_rejects_a_copy(ctx: dict[str, Any]) -> None:
    with pytest.raises(AlignmentError, match="copy"):
        AlignmentEngine().pair(
            ctx["avatar"],
            ctx["pieces"],
            [copy("Pocket_copy", "Pocket")],
            {"piece": "Pocket", "point": "Chest_FrontSide_R"},
            "Pocket_copy",
            "mirror_anchor",
        )

"""Assembler (assembly JSON -> engine runs -> independent verification) on the synthetic avatar.

The three committed assemblies must pass on the synthetic body; the verifier must catch what it is there to catch
(an injected scale, a piece below the floor, a rotated piece, a swapped or stacked pair, a skirt centred on its point
instead of hanging from the waist). Golden numbers (registration offsets, ring radii) are in ``tests/golden``.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from alignment_engine import AlignmentEngine
from alignment_engine.levels import body_levels
from alignment_engine.loaders import read_avatar
from assembler import verify as V
from assembler.run import AssemblyError, Inputs, _solve, build_inputs, bundles_of, load_assembly, run, run_view, spec_of

ROOT = Path(__file__).resolve().parents[1]
ASSEMBLIES = ROOT / "assert" / "assemblies"
GARMENTS = ("aldrich_skirt", "trousers", "male_shirt")
GOLDEN = ROOT / "tests" / "golden"


@pytest.fixture(scope="module")
def cache(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("asm_cache")


@pytest.fixture(scope="module")
def runs(cache: Path, tmp_path_factory: pytest.TempPathFactory) -> dict[str, dict[str, Any]]:
    out = tmp_path_factory.mktemp("asm_runs")
    res = {}
    for g in GARMENTS:
        res[g] = run(ASSEMBLIES / f"{g}.assembly.json", out / g, synthetic=True, fast=True, render=False, cache=cache)
        res[g]["dir"] = out / g
    return res


# -- schema ------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("g", GARMENTS)
def test_committed_assemblies_validate(g: str) -> None:
    doc = load_assembly(ASSEMBLIES / f"{g}.assembly.json")
    assert doc["garment"] == g


def _bad(tmp: Path, patch: dict[str, Any] | None = None, drop: str | None = None, **kw: Any) -> Path:
    doc = json.loads((ASSEMBLIES / "aldrich_skirt.assembly.json").read_text(encoding="utf-8"))
    doc.update(patch or {})
    if drop:
        del doc[drop]
    for k, v in kw.items():
        doc["placements"][0][k] = v
    p = tmp / "bad.assembly.json"
    p.write_text(json.dumps(doc), encoding="utf-8")
    return p


@pytest.mark.parametrize(
    "case",
    [
        {"drop": "placements"},
        {"patch": {"schema_version": 2}},
        {"patch": {"arrange_options": {"allow_distortion": True}}},
        {"wrap": "body_slice"},
        {"anchor": {"level": "Belly"}},
        {"patch": {"unknown_key": 1}},
    ],
)
def test_bad_assemblies_are_rejected(tmp_path: Path, case: dict[str, Any]) -> None:
    with pytest.raises(AssemblyError):
        load_assembly(_bad(tmp_path, **case))


def test_inputs_are_cached_by_content(cache: Path) -> None:
    a = build_inputs("aldrich_skirt", True, cache)
    b = build_inputs("aldrich_skirt", True, cache)
    assert b.cached == {"pieces": True, "avatar": True}
    assert (a.pieces, a.avatar, a.avatar_hash) == (b.pieces, b.avatar, b.avatar_hash)


# -- end to end --------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("g", GARMENTS)
def test_assembly_passes_on_the_synthetic_avatar(runs: dict[str, dict[str, Any]], g: str) -> None:
    s = runs[g]
    fails = [r for r in s["runs"]["primary"]["rows"] if r["status"] == "fail"]
    assert s["pass"], fails[:10]
    assert (s["dir"] / "verify.md").is_file() and (s["dir"] / "scene.glb").stat().st_size > 1000


@pytest.mark.parametrize("g", GARMENTS)
def test_golden_numbers(runs: dict[str, dict[str, Any]], g: str) -> None:
    """Registration offsets and ring radii stay what they were when the assembly was accepted (1 mm)."""
    result = json.loads((runs[g]["dir"] / "result.json").read_text(encoding="utf-8"))
    now = {
        d["piece"]: {
            "radius_m": d["arrange"]["radius_m"],
            "registration_offset_m": (d["arrange"].get("registration") or {}).get("offset_m"),
        }
        for d in result["placements"]
    }
    path = GOLDEN / f"{g}.synthetic.json"
    if not path.is_file():  # first run writes the golden file (commit it)
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(now, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    gold = json.loads(path.read_text(encoding="utf-8"))
    assert set(gold) == set(now)
    for k, v in gold.items():
        for f in ("radius_m", "registration_offset_m"):
            if v[f] is None:
                assert now[k][f] is None, (k, f)
            else:
                assert now[k][f] == pytest.approx(v[f], abs=0.001), (k, f)


def test_skirt_is_deterministic_and_carry_moves_nothing(cache: Path, tmp_path: Path) -> None:
    s = run(ASSEMBLIES / "aldrich_skirt.assembly.json", tmp_path / "skirt", synthetic=True, render=False, cache=cache)
    glob = {g["id"]: g for g in s["global"]}
    assert glob["determinism"]["ok"] and glob["carry.moves"]["ok"]
    assert s["pass"]


# -- the verifier catches what it must ---------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def skirt_case(cache: Path, tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    doc = load_assembly(ASSEMBLIES / "aldrich_skirt.assembly.json")
    inp = build_inputs("aldrich_skirt", True, cache)
    derived = tmp_path_factory.mktemp("skirt_derived")
    res = _solve(AlignmentEngine(), inp, spec_of(doc), derived, cache / "refined", "arrange")
    avatar = read_avatar(inp.avatar)
    points = {p["name"]: p for p in json.loads((inp.avatar / "arrangement_points.json").read_text("utf-8"))["points"]}
    vols = {v["name"]: v for v in json.loads((inp.avatar / "bounding_volumes.json").read_text("utf-8"))["volumes"]}
    return {"doc": doc, "inp": inp, "res": res, "derived": derived, "avatar": avatar, "points": points, "vols": vols}


def _verify(
    c: dict[str, Any], mutate: Any = None, res: Any = None, doc: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    doc = doc or c["doc"]
    res = res or c["res"]
    inp: Inputs = c["inp"]
    rv = run_view("t", res, spec_of(doc), bundles_of(inp, c["derived"]))
    rv.pieces = [copy.copy(p) for p in rv.pieces]
    if mutate:
        mutate(rv)
    body = V.Surf(c["avatar"].vertices, c["avatar"].faces)
    V.check_size(rv)
    near = V.check_clearance(rv, body, c["points"], doc.get("expect", {}))
    V.check_orientation(rv, c["points"], near, doc.get("expect", {}), c["vols"])
    V.check_registration(rv, body_levels(c["avatar"]), doc.get("expect", {}))
    V.check_pairs(rv, c["points"], doc.get("expect", {}))
    return rv.rows


def _fails(rows: list[dict[str, Any]], cid: str) -> list[dict[str, Any]]:
    return [r for r in rows if r["id"] == cid and r["status"] == "fail"]


def test_verifier_passes_the_untouched_skirt(skirt_case: dict[str, Any]) -> None:
    assert [r for r in _verify(skirt_case) if r["status"] == "fail"] == []


def test_size_audit_catches_an_injected_scale(skirt_case: dict[str, Any]) -> None:
    def scale(rv: V.RunView) -> None:
        p = rv.pieces[0]
        c = p.world.mean(axis=0)
        p.world = c + 1.01 * (p.world - c)

    rows = _verify(skirt_case, scale)
    assert _fails(rows, "size.singular_values") and _fails(rows, "size.area") and _fails(rows, "size.perimeter")


def test_verifier_catches_a_piece_below_the_floor(skirt_case: dict[str, Any]) -> None:
    def drop(rv: V.RunView) -> None:
        p = rv.pieces[0]
        p.world = p.world - np.array([0.0, float(p.world[:, 1].min()) + 0.05, 0.0])

    assert _fails(_verify(skirt_case, drop), "clearance.floor")


def test_verifier_catches_a_rotated_piece(skirt_case: dict[str, Any]) -> None:
    def turn(rv: V.RunView) -> None:  # 180 degrees about the horizontal axis through its centre: upside down
        p = rv.pieces[0]
        c = p.world.mean(axis=0)
        p.world = (p.world - c) * np.array([-1.0, -1.0, 1.0]) + c

    assert _fails(_verify(skirt_case, turn), "orientation.top_up")


def test_verifier_catches_a_swapped_pair(skirt_case: dict[str, Any]) -> None:
    def swap(rv: V.RunView) -> None:  # each half reflected onto the other's side (the left half on the right)
        for p in rv.pieces:
            if p.name in ("A - Skirt Front", "Skirt_Front_R"):
                p.world = p.world * np.array([-1.0, 1.0, 1.0])

    # a swapped half is the mirror image of what belongs there: its pattern face turns into the body
    assert len(_fails(_verify(skirt_case, swap), "orientation.face_out")) == 2


def test_verifier_catches_a_stacked_pair(skirt_case: dict[str, Any]) -> None:
    def stack(rv: V.RunView) -> None:  # the clone lies on top of its source
        a = next(p for p in rv.pieces if p.name == "A - Skirt Front")
        b = next(p for p in rv.pieces if p.name == "Skirt_Front_R")
        b.world = b.world * np.array([-1.0, 1.0, 1.0]) + np.array([0.0, 0.0, 0.01])
        assert a.world[:, 0].mean() * b.world[:, 0].mean() > 0

    rows = _verify(skirt_case, stack)
    assert _fails(rows, "clearance.pieces") and _fails(rows, "pairs.sides")


def test_verifier_catches_a_skirt_centred_on_its_point(skirt_case: dict[str, Any]) -> None:
    """The bug of the report: without level registration (schema 4) the skirt is centred on the waist point."""
    doc = copy.deepcopy(skirt_case["doc"])
    for pl in doc["placements"]:
        pl.pop("anchor", None)
    spec = spec_of(doc)
    spec["schema_version"] = 4
    res = _solve(
        AlignmentEngine(),
        skirt_case["inp"],
        spec,
        skirt_case["derived"],
        skirt_case["inp"].pieces.parent / "refined",
        "arrange",
    )
    assert len(_fails(_verify(skirt_case, res=res, doc=doc), "registration")) == 4

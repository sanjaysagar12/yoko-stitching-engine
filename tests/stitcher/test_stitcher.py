"""Stitcher: chain resolver, correspondences, orientation, untouched geometry, determinism, the committed stitchings
(synthetic avatar) and one mutated file per failure kind. The assembly runs come from the stitcher's run cache
(build/_stitch_runs, made on first use)."""

from __future__ import annotations

import ast
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from stitcher.chains import build_piece, resolve
from stitcher.errors import StitchingError
from stitcher.load import ROOT, ensure_run, load_run, read_stitching
from stitcher.result import stitch
from stitcher.seam import compute, crossings
from stitcher.verify import check, verdict

STITCHINGS = ROOT / "assert" / "stitchings"
GARMENTS = ["aldrich_skirt", "male_shirt", "trousers"]


# -- hand-made pieces ------------------------------------------------------------------------------------------------
def piece(name: str, outline: list[list[float]], points: dict[str, int], z: float = 0.0, flip: bool = False) -> Any:
    """A placed piece lying flat at height ``z`` (world = flat x, y, z); ``points`` name outline vertices."""
    o = np.asarray(outline, dtype=float)
    seg = np.linalg.norm(np.roll(o, -1, axis=0) - o, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)[:-1]])
    named = [{"name": n, "label": n, "on_outline": True, "outline_s_m": float(cum[i]), "xy_m": o[i].tolist()}
             for n, i in points.items()]  # fmt: skip
    named.append({"name": "IN", "label": "IN", "on_outline": False, "xy_m": o.mean(axis=0).tolist()})
    world = np.column_stack([o[:, 0], o[:, 1], np.full(len(o), z)])
    if flip:
        world[:, 0] *= -1
    wn = [{"name": q["name"], "label": q["name"], "on_outline": q["on_outline"],
           "world_m": [world[i][0], world[i][1], z] if q["on_outline"] else [*q["xy_m"], z]}
          for q, i in zip(named, [*points.values(), 0], strict=True)]  # fmt: skip
    bundle = {"name": name, "outline_m": o.tolist(), "named_points": named}
    return build_piece({"piece": name, "world_outline_m": world.tolist(), "world_edges": [], "world_named_points": wn},
                       bundle)  # fmt: skip


SQUARE = [[0, 0], [0.1, 0], [0.1, 0.001], [0.1, 0.1], [0, 0.1]]  # a 1 mm edge between B and B2
POINTS = {"A": 0, "B": 1, "B2": 2, "C": 3, "D": 4}


def test_chain_walks_tiny_edges_and_wraps_round() -> None:
    p = piece("sq", SQUARE, POINTS)
    c = resolve(p, ["A", "B", "C"])
    assert c.names == ["A", "B", "B2", "C"] and c.length == pytest.approx(0.2)
    w = resolve(p, ["C", "D", "A", "B"])  # through the outline's start
    assert w.names == ["C", "D", "A", "B"] and w.length == pytest.approx(0.3, abs=1e-3)
    back = resolve(p, ["B", "A", "D"])
    assert back.direction == -1 and back.length == pytest.approx(0.2)
    assert resolve(p, ["A", "B"]).length == pytest.approx(0.1)  # one edge: the exact way wins


def test_chain_errors_name_what_is_valid() -> None:
    p = piece("sq", SQUARE, POINTS)
    with pytest.raises(StitchingError, match="ambiguous"):
        resolve(p, ["A", "C"])
    with pytest.raises(StitchingError, match="unknown outline point.*outline points"):
        resolve(p, ["A", "X"])
    with pytest.raises(StitchingError, match="not on the outline"):
        resolve(p, ["A", "IN"])
    with pytest.raises(StitchingError, match="not contiguous"):
        resolve(p, ["A", "C", "B", "D"])


def _two_strips(lb: float) -> dict[str, Any]:
    a = piece("a", [[0, 0], [0.4, 0], [0.4, 0.1], [0, 0.1]], {"A0": 0, "A1": 1, "A2": 2, "A3": 3})
    b = piece("b", [[0, 0.15], [lb, 0.15], [lb, 0.3], [0, 0.3]], {"B0": 0, "B1": 1, "B2": 2, "B3": 3})
    return {"a": a, "b": b}


def _seam(ease: dict[str, Any] | None = None, reverse: bool = False, kind: str = "plain") -> dict[str, Any]:
    s: dict[str, Any] = {"id": "s", "kind": kind, "a": {"piece": "a", "chain": ["A3", "A2"]},
                         "b": {"piece": "b", "chain": ["B0", "B1"], "reverse": reverse}}  # fmt: skip
    if ease:
        s["ease"] = ease
    return s


def test_correspondences_are_monotonic_and_end_at_the_chain_ends() -> None:
    c = compute(_seam(), _two_strips(0.4), {"samples_per_cm": 2})
    sm = c.record["samples"]
    assert sm["count"] == 81 and np.all(np.diff(sm["s"]) > 0)
    assert np.allclose(c.pa[0], [0, 0.1, 0]) and np.allclose(c.pa[-1], [0.4, 0.1, 0])
    assert np.allclose(c.pb[0], [0, 0.15, 0]) and np.allclose(c.pb[-1], [0.4, 0.15, 0])
    assert np.allclose(np.diff(sm["a_arc_m"]), np.diff(sm["b_arc_m"]))  # ratio 1: equal arc steps
    assert c.record["orientation"]["crossing_count"] == 0 and c.record["orientation"]["sewing_rule_ok"]


@pytest.mark.parametrize("mode,lb", [("distribute", 0.412), ("gather", 0.6)])
def test_ease_is_spread_by_normalised_arc_length(mode: str, lb: float) -> None:
    c = compute(_seam({"mode": mode, "ratio": 1.6}), _two_strips(lb), {"samples_per_cm": 2})
    sm = c.record["samples"]
    assert np.allclose(np.asarray(sm["b_arc_m"]) / lb, np.asarray(sm["a_arc_m"]) / 0.4, atol=1e-5)  # µm rounding
    assert c.record["ease"]["gathered_side"] == "b" and c.record["ease"]["applied_ratio"] == pytest.approx(lb / 0.4)


def test_a_reversed_seam_is_twisted_and_the_hint_says_reverse() -> None:
    c = compute(_seam(reverse=True), _two_strips(0.4), {"samples_per_cm": 2})
    o = c.record["orientation"]
    assert o["crossing_count"] > 0 and o["reversed_crossing_count"] == 0 and not o["ok"]
    assert not o["sewing_rule_ok"] and "reverse: true" in o["hint"]
    assert crossings(c.pa, c.pb[::-1], 5) == 0


def test_fold_seams_keep_both_pieces_on_one_side() -> None:
    turned = compute(_seam(kind="fold", reverse=True), _two_strips(0.4), {}).record["orientation"]
    assert turned["sewing_rule_ok"] is True
    flat = compute(_seam(kind="fold"), _two_strips(0.4), {}).record["orientation"]
    assert flat["sewing_rule_ok"] is False  # a flat join declared as fold breaks the rule


def test_trim_cuts_a_chain_between_named_points() -> None:
    p = _two_strips(0.4)
    s = _seam()
    s["a"]["trim_start_m"] = 0.1
    s["b"]["chain"] = ["B0", "B1"]
    s["b"]["trim_end_m"] = 0.1
    c = compute(s, p, {})
    assert c.record["length_a_cm"] == pytest.approx(30.0) and np.allclose(c.pa[0], [0.1, 0.1, 0])
    assert np.allclose(c.pb[-1], [0.3, 0.15, 0])


# -- the committed stitchings ------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def runs() -> dict[str, Path]:
    return {g: ensure_run(ROOT / "assert" / "assemblies" / f"{g}.assembly.json", True) for g in GARMENTS}


def _stitch(g: str, run_dir: Path, path: Path | None = None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path = path or STITCHINGS / f"{g}.stitching.json"
    st = read_stitching(path)
    asm = ROOT / st["assembly"]
    res = stitch(st, path, asm, load_run(run_dir))
    return res, check(res, st, path, asm, run_dir)


@pytest.mark.parametrize("g", GARMENTS)
def test_committed_stitching_passes(runs: dict[str, Path], g: str) -> None:
    res, rows = _stitch(g, runs[g])
    fails = [f"{r['id']} {r['seam']} {r['hint']}" for r in rows if r["status"] == "fail"]
    assert verdict(rows), fails
    s = res["summary"]
    assert s["errors"] == 0 and s["crossings"] == 0 and s["sewing_rule_failures"] == 0 and s["unstitched_edges"] == 0
    assert set(res["required_seams"]) <= {x["id"] for x in res["seams"]}


def test_geometry_untouched_and_deterministic(runs: dict[str, Path], tmp_path: Path) -> None:
    before = (runs["aldrich_skirt"] / "result.json").read_bytes()
    a, _ = _stitch("aldrich_skirt", runs["aldrich_skirt"])
    b, _ = _stitch("aldrich_skirt", runs["aldrich_skirt"])
    assert json.dumps(a, indent=1) == json.dumps(b, indent=1)
    assert a["untouched"] and (runs["aldrich_skirt"] / "result.json").read_bytes() == before


def _mutated(tmp_path: Path, fn: Any) -> Path:
    doc = json.loads((STITCHINGS / "aldrich_skirt.stitching.json").read_text(encoding="utf-8"))
    fn(doc)
    p = tmp_path / "aldrich_skirt.stitching.json"
    p.write_text(json.dumps(doc), encoding="utf-8")
    return p


def _drop_side(d: dict[str, Any]) -> None:
    d["seams"] = [s for s in d["seams"] if s["id"] != "side_L"]


def _short_chain(d: dict[str, Any]) -> None:
    d["seams"][0]["b"]["chain"] = ["A10", "A10b", "A7"]


def _unknown_point(d: dict[str, Any]) -> None:
    d["seams"][0]["a"]["chain"][0] = "Z99"


def _small_gap(d: dict[str, Any]) -> None:
    d["defaults"]["max_gap_m"] = 0.05


def _two_claims(d: dict[str, Any]) -> None:
    d["seams"].append({**copy.deepcopy(d["seams"][0]), "id": "side_R_again"})


@pytest.mark.parametrize(
    "mutate,row",
    [(_drop_side, "required"), (_short_chain, "seam.ratio"), (_unknown_point, "seam.resolve"),
     (_small_gap, "seam.gap"), (_two_claims, "edges.claimed_twice")],
)  # fmt: skip
def test_mutated_stitching_fails(runs: dict[str, Path], tmp_path: Path, mutate: Any, row: str) -> None:
    _, rows = _stitch("aldrich_skirt", runs["aldrich_skirt"], _mutated(tmp_path, mutate))
    assert not verdict(rows)
    assert any(r["id"] == row and r["status"] == "fail" for r in rows), [r for r in rows if r["status"] == "fail"]


def test_a_changed_alignment_result_is_caught(runs: dict[str, Path], tmp_path: Path) -> None:
    import shutil

    src = runs["aldrich_skirt"]
    dst = tmp_path / "run"
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("engine"))
    res, _ = _stitch("aldrich_skirt", dst)
    r = json.loads((dst / "result.json").read_text(encoding="utf-8"))
    r["placements"][0]["wrapped_vertices"][0][0] += 1e-6  # one vertex moved by a micrometre
    (dst / "result.json").write_text(json.dumps(r), encoding="utf-8")
    st_path = STITCHINGS / "aldrich_skirt.stitching.json"
    st = read_stitching(st_path)
    rows = check(res, st, st_path, ROOT / st["assembly"], dst)
    assert any(r["id"] == "geometry.untouched" and r["status"] == "fail" for r in rows)


def test_stitcher_imports_no_engine() -> None:
    allowed = {"assembler.render", "assembler.verify"}
    bad = []
    for py in (ROOT / "src" / "stitcher").rglob("*.py"):
        for node in ast.walk(ast.parse(py.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                mods = [node.module]
            else:
                continue
            bad += [f"{py.name}: {m}" for m in mods
                    if m.split(".")[0] in {"sm2d_engine", "avatar_engine", "alignment_engine", "web"}
                    or (m.split(".")[0] == "assembler" and m not in allowed)]  # fmt: skip
    assert not bad, bad

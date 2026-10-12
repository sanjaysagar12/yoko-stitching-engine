"""Size-exact, collision-free placement (align and arrange) on the synthetic body and, when present, SMPL-X.

Everything is measured here again, independently of the engine's own checks: sizes on the placed meshes against the
flat pattern, distances from 3 mm surface samples with a sampling margin that makes the bound rigorous (if the closest
points lie inside faces or edges, the closest samples are at most ``sqrt(D^2 + (h_a + h_b)^2)`` apart, ``h`` the
largest distance from a mesh point to its nearest sample).
"""

from __future__ import annotations

import copy
import itertools
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy.spatial import cKDTree

from alignment_engine import AlignmentEngine
from alignment_engine.errors import SizeChangedError
from alignment_engine.loaders import read_avatar, read_pieces
from alignment_engine.verify import AvatarSurface, surface_samples

GAP = 0.03
CLEAR = 0.03
TOL_PCT = 0.05
CF = {"type": "edge", "from_point": "A33", "to_point": "A32"}
DERIVED = [
    {"id": "FrontPanel_L", "source": "FrontPanel", "op": "mirror", "axis": CF},
    {"id": "BackPanel_L", "source": "BackPanel", "op": "mirror", "axis": {"type": "x", "value_m": 0.0}},
    {"id": "ShortSleeve_L", "source": "ShortSleeve", "op": "mirror", "axis": {"type": "bbox_center"}},
    {"id": "FullSleeve_L", "source": "FullSleeve", "op": "mirror", "axis": {"type": "bbox_center"}},
    {"id": "Cuff_L", "source": "Cuff", "op": "mirror", "axis": {"type": "bbox_center"}},
    {"id": "CuffInterface_L", "source": "CuffInterface", "op": "mirror", "axis": {"type": "bbox_center"}},
]
# 23 pieces: front pair, back pair, yoke, both sleeves twice, cuffs and interfacings, collars, plackets, pockets
SHIRT = [
    ("FrontPanel", "Chest_Front", "A32"),
    ("FrontPanel_L", "Chest_Front", "A32"),
    ("BackPanel", "Chest_Back", "A24"),
    ("BackPanel_L", "Chest_Back", "A24"),
    ("Yoke", "Chest_Back", None),
    ("ShortSleeve", "UpperArm_Outer_R", None),
    ("ShortSleeve_L", "UpperArm_Outer_L", None),
    ("FullSleeve", "Forearm_Outer_R", None),
    ("FullSleeve_L", "Forearm_Outer_L", None),
    ("Cuff", "Forearm_Outer_R", None),
    ("Cuff_L", "Forearm_Outer_L", None),
    ("CuffInterface", "Forearm_Outer_R", None),
    ("CuffInterface_L", "Forearm_Outer_L", None),
    ("CollarBase", "Neck_Front", None),
    ("CollarBaseInterface", "Neck_Front", None),
    ("CollarTop", "Neck_Back", None),
    ("CollarTopInterface", "Neck_Back", None),
    ("PlacketTop", "Chest_Front", None),
    ("PlacketUnder", "Waist_Front", None),
    ("Pocket", "Chest_FrontSide_L", None),
    ("PocketFlap", "Chest_FrontSide_L", None),
    ("PocketRound", "Chest_FrontSide_R", None),
    ("PocketFlapRound", "Chest_FrontSide_R", None),
]


def placements(wrap: str = "roll", names: list[str] | None = None) -> list[dict[str, Any]]:
    out = []
    for name, point, ref in SHIRT:
        if names is not None and name not in names:
            continue
        d: dict[str, Any] = {"piece": name, "point": point, "wrap": wrap}
        if ref:
            d["reference"] = ref
        out.append(d)
    return out


def spec(pl: list[dict[str, Any]], **opts: Any) -> str:
    doc: dict[str, Any] = {"schema_version": 4, "placements": pl, "derived_pieces": DERIVED}
    if opts:
        doc["arrange_options"] = opts
    return json.dumps(doc)


@pytest.fixture(scope="module")
def ctx(
    avatar: dict[str, Any], shirt_pieces: Path, refine_cache: Path, tmp_path_factory: pytest.TempPathFactory
) -> dict[str, Any]:
    d = tmp_path_factory.mktemp(f"layout_{avatar['kind']}")
    av = read_avatar(avatar["dir"])
    return {
        "avatar": avatar["dir"],
        "kind": avatar["kind"],
        "pieces": shirt_pieces,
        "derived": d / "derived",
        "cache": refine_cache,
        "engine": AlignmentEngine(),
        "surface": AvatarSurface(av.vertices, av.faces),
        "tmp": d,
    }


def run(ctx: dict[str, Any], method: str, text: str, engine: AlignmentEngine | None = None, **kw: Any) -> Any:
    eng = engine or ctx["engine"]
    return getattr(eng, method)(
        ctx["avatar"], ctx["pieces"], text, derived_dir=ctx["derived"], cache_dir=ctx["cache"], **kw
    )


@pytest.fixture(scope="module")
def arranged(ctx: dict[str, Any]) -> Any:
    return run(ctx, "arrange", spec(placements()))


@pytest.fixture(scope="module")
def aligned(ctx: dict[str, Any]) -> Any:
    return run(ctx, "align", spec(placements()))


# -- independent measurements ------------------------------------------------------------------------------------
def _bundles(ctx: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out = {k: v.bundle for k, v in read_pieces(ctx["pieces"]).items()}
    out.update({k: v.bundle for k, v in read_pieces(ctx["derived"]).items()})
    return out


def size_changes(p: Any, bundle: dict[str, Any]) -> dict[str, float]:
    """Per cent changes of a placed piece measured on its mesh and through its transform (not the engine's check)."""
    assert p.flat is not None and len(p.flat) == len(p.vertices)
    f = p.faces
    e = np.unique(np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1), axis=0)
    fl = np.linalg.norm(p.flat[e[:, 0]] - p.flat[e[:, 1]], axis=1)
    wl = np.linalg.norm(p.vertices[e[:, 0]] - p.vertices[e[:, 1]], axis=1)
    t2, t3 = p.flat[f], p.vertices[f]
    u, w = t2[:, 1] - t2[:, 0], t2[:, 2] - t2[:, 0]
    a2 = 0.5 * np.abs(u[:, 0] * w[:, 1] - u[:, 1] * w[:, 0]).sum()
    a3 = 0.5 * np.linalg.norm(np.cross(t3[:, 1] - t3[:, 0], t3[:, 2] - t3[:, 0]), axis=1).sum()
    out = np.asarray(bundle["outline_m"], dtype=np.float64)
    ring = np.vstack([out, out[:1]])
    dense = np.vstack(
        [
            np.linspace(a, b, max(int(np.ceil(np.linalg.norm(b - a) / 0.002)), 1) + 1)[:-1]
            for a, b in zip(ring[:-1], ring[1:], strict=True)
        ]
        + [ring[-1:]]
    )
    per_flat = float(np.linalg.norm(np.diff(ring, axis=0), axis=1).sum())
    per_3d = float(np.linalg.norm(np.diff(p.transform(dense), axis=0), axis=1).sum())
    named = np.array([q["xy_m"] for q in bundle["named_points"]], dtype=np.float64)
    worst_named = 0.0
    for i, j in itertools.combinations(range(len(named)), 2):
        d = float(np.linalg.norm(named[i] - named[j]))
        if d < 0.005:
            continue
        seg = np.linspace(named[i], named[j], int(np.ceil(d / 0.002)) + 1)
        worst_named = max(
            worst_named, abs(float(np.linalg.norm(np.diff(p.transform(seg), axis=0), axis=1).sum()) / d - 1.0)
        )
    return {
        "edge": float(np.abs(wl / fl - 1.0).max() * 100),
        "perimeter": abs(per_3d / per_flat - 1.0) * 100,
        "area": abs(a3 / a2 - 1.0) * 100,
        "named": worst_named * 100,
        "max_edge_m": float(fl.max()),
    }


_S: dict[int, tuple[Any, Any, Any, float]] = {}


def samples(p: Any) -> tuple[Any, Any, float]:
    """Surface samples (vertices plus edge points 3 mm apart), their KD-tree and the cover radius ``h``.

    Cached per placed piece; the entry keeps the piece alive so that its ``id`` cannot be reused by another one."""
    if id(p) not in _S:
        pts = surface_samples(p.vertices, p.faces, 0.003)
        t = p.vertices[p.faces]
        longest = float(np.linalg.norm(t - np.roll(t, -1, axis=1), axis=2).max())
        _S[id(p)] = (p, pts, cKDTree(pts), longest / np.sqrt(3.0))
    _, pts, tree, h = _S[id(p)]
    return pts, tree, h


def pair_lower_bound(a: Any, b: Any, upper: float = 0.08) -> float:
    """A rigorous lower bound of the distance between two placed meshes (inf if farther than ``upper``)."""
    pa, ta, ha = samples(a)
    pb, tb, hb = samples(b)
    lo = np.maximum(pa.min(axis=0), pb.min(axis=0))
    hi = np.minimum(pa.max(axis=0), pb.max(axis=0))
    if np.any(lo - hi > upper):
        return np.inf
    d = float(tb.query(pa, distance_upper_bound=upper)[0].min())
    return float(np.sqrt(max(d * d - (ha + hb) ** 2, 0.0))) if np.isfinite(d) else np.inf


def body_lower_bound(surface: AvatarSurface, p: Any) -> float:
    pts, _, h = samples(p)
    sd = surface.signed_distance(pts)
    d = float(sd.min())
    return float(np.sqrt(max(d * d - h * h, 0.0))) if d > 0 else d


def assert_layout(ctx: dict[str, Any], placed: list[Any], gap: float = GAP, clear: float = CLEAR) -> None:
    for p in placed:
        assert body_lower_bound(ctx["surface"], p) >= clear - 0.001, p.name
    for a, b in itertools.combinations(placed, 2):
        assert pair_lower_bound(a, b) >= gap - 0.001, (a.name, b.name)


# 1 ------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("wrap", ["flat", "roll", "around"])
def test_1_every_placed_piece_is_an_exact_copy(ctx: dict[str, Any], wrap: str) -> None:
    names = sorted({n for n, _, _ in SHIRT} | {"PocketRound"})
    res = run(ctx, "align", spec(placements(wrap, names)))
    bundles = _bundles(ctx)
    assert len(res.placed) == len(names)
    for p, d in zip(res.placed, res.doc["placements"], strict=True):
        m = size_changes(p, bundles[p.name])
        assert max(m["edge"], m["perimeter"], m["area"], m["named"]) <= TOL_PCT, (wrap, p.name, m)
        assert m["max_edge_m"] <= 0.005 + 1e-9, (p.name, m["max_edge_m"])  # refined to 5 mm before rolling
        sc = d["size_check"]
        assert (
            max(
                sc["max_edge_change_pct"],
                sc["perimeter_change_pct"],
                sc["area_change_pct"],
                sc["named_point_change_pct"],
            )
            <= TOL_PCT
        )
        assert d["changes_piece_size"] is False
    assert res.doc["validity"]["size_ok"] is True


def test_1_arranged_pieces_are_exact_copies(ctx: dict[str, Any], arranged: Any) -> None:
    bundles = _bundles(ctx)
    for p in arranged.placed:
        m = size_changes(p, bundles[p.name])
        assert max(m["edge"], m["perimeter"], m["area"], m["named"]) <= TOL_PCT, (p.name, m)


# 2 ------------------------------------------------------------------------------------------------------------
def test_2_full_shirt_arranged_without_any_collision(ctx: dict[str, Any], arranged: Any) -> None:
    assert len(arranged.placed) == 23
    v = arranged.doc["validity"]
    assert v["valid"] is True and v["violations"] == [], v["violations"]
    assert_layout(ctx, arranged.placed)
    assert v["min_distance_between_pieces_m"] >= GAP - 0.001 and v["min_distance_to_avatar_m"] >= CLEAR - 0.001


def test_2_full_shirt_aligned_without_any_collision(ctx: dict[str, Any], aligned: Any) -> None:
    assert aligned.doc["validity"]["valid"] is True, aligned.doc["validity"]["violations"]
    assert_layout(ctx, aligned.placed)


# 3 ------------------------------------------------------------------------------------------------------------
ORDER = [
    "FrontPanel",
    "BackPanel",
    "Yoke",  # partly over the top of the back panel
    "Pocket",  # nested over the front panel
    "PocketFlap",  # a second piece on an occupied point
    "CollarBase",
    "CollarBaseInterface",  # identical shape, same point
    "ShortSleeve",
    "Cuff",
]


def test_3_interactive_order_keeps_earlier_pieces_and_the_invariants(ctx: dict[str, Any]) -> None:
    eng = AlignmentEngine()
    done: list[dict[str, Any]] = []
    previous: dict[str, Any] = {}
    for name in ORDER:
        new = placements("roll", [name])[0]
        res = run(ctx, "align", spec([*done, new]), engine=eng)
        placed = {p.name: p for p in res.placed}
        assert res.doc["validity"]["valid"] is True, (name, res.doc["validity"]["violations"])
        assert_layout(ctx, res.placed)
        docs = {d["piece"]: d for d in res.doc["placements"]}
        for n, verts in previous.items():  # earlier pieces did not move, unless the answer says so (a ring grew)
            if not docs[n]["arrange"]["solved_now"]:
                assert np.abs(placed[n].vertices - verts).max() < 1e-9, (name, n)
        done = [{**pl, "solved": docs[pl["piece"]]["solved"]} for pl in [*done, new]]
        previous = {n: p.vertices.copy() for n, p in placed.items()}
    # the second piece on Chest_FrontSide_L was moved clear of the first one, not left on top of it
    assert pair_lower_bound(placed["Pocket"], placed["PocketFlap"]) >= GAP - 0.001


def test_3_changing_one_piece_moves_only_that_piece(ctx: dict[str, Any], aligned: Any) -> None:
    pl = placements()
    for p, d in zip(pl, aligned.doc["placements"], strict=True):
        p["solved"] = d["solved"]
    changed = copy.deepcopy(pl)
    for p in changed:
        if p["piece"] == "Pocket":
            p["rotation_deg"] = 15.0
            del p["solved"]
    res = run(ctx, "align", spec(changed))
    before = {p.name: p for p in aligned.placed}
    for p in res.placed:
        if p.name != "Pocket":
            assert np.abs(p.vertices - before[p.name].vertices).max() < 1e-9, p.name
    assert res.doc["validity"]["valid"] is True
    assert_layout(ctx, res.placed)


# 4 ------------------------------------------------------------------------------------------------------------
def _cf_gap(res: Any, a: str, b: str) -> float:
    pa, pb = (next(p for p in res.placed if p.name == n) for n in (a, b))
    return pair_lower_bound(pa, pb)


def test_4_centre_line_pairs_keep_the_gap_and_side_pairs_mirror(
    ctx: dict[str, Any], arranged: Any, aligned: Any
) -> None:
    for res in (arranged, aligned):
        for a, b in (("FrontPanel", "FrontPanel_L"), ("BackPanel", "BackPanel_L")):
            pa, pb = (next(p for p in res.placed if p.name == n) for n in (a, b))
            d = float(cKDTree(samples(pb)[0]).query(samples(pa)[0])[0].min())
            assert d == pytest.approx(GAP, abs=0.001), (a, b, d)
    if ctx["kind"] != "synthetic":
        return
    by = {p.name: p for p in arranged.placed}
    bundles = _bundles(ctx)
    flip = np.array([-1.0, 1.0, 1.0])
    for a, b in (("FrontPanel", "FrontPanel_L"), ("ShortSleeve", "ShortSleeve_L"), ("FullSleeve", "FullSleeve_L")):
        # the clone keeps the point names: its named points land on the mirror images of the source's
        na = {q["name"]: q["xy_m"] for q in bundles[a]["named_points"]}
        nb = {q["name"]: q["xy_m"] for q in bundles[b]["named_points"]}
        names = sorted(set(na) & set(nb))
        wa = by[a].transform(np.array([na[n] for n in names]))
        wb = by[b].transform(np.array([nb[n] for n in names])) * flip
        assert np.abs(wa - wb).max() < 0.002, (a, b, float(np.abs(wa - wb).max()))


# 5 ------------------------------------------------------------------------------------------------------------
def test_5_infeasible_limits_are_reported_never_hidden(ctx: dict[str, Any]) -> None:
    res = run(
        ctx,
        "arrange",
        spec(placements("roll", ["FrontPanel", "FrontPanel_L", "BackPanel", "BackPanel_L"]), max_standoff=0.01),
    )
    v = res.doc["validity"]
    assert v["valid"] is False and v["violations"]
    assert all(
        x["pieces"] and x["limit"] and (x["required_m"] > 0 or x["kind"] == "PAIR_STACKED") for x in v["violations"]
    )
    assert any("max_standoff" in x["limit"] for x in v["violations"])
    assert any(w.startswith("ArrangementInfeasible") for w in res.warnings)
    for d in res.doc["placements"]:  # nothing was clamped silently: the limit is on every piece that hit it
        assert d["arrange"]["standoff_m"] <= 0.01 + 1e-9
        assert {"max_standoff", "max_shift"} & set(d["arrange"]["limits"]), d["arrange"]["limits"]


# 6 ------------------------------------------------------------------------------------------------------------
def test_6_a_stretching_wrap_is_a_hard_failure(ctx: dict[str, Any]) -> None:
    pl = [{"piece": "BackPanel", "point": "Chest_Back", "wrap": "body_slice"}]
    with pytest.raises(SizeChangedError):
        run(ctx, "align", spec(pl))
    res = run(ctx, "align", spec(pl, allow_distortion=True))
    d = res.doc["placements"][0]
    assert d["changes_piece_size"] is True and res.doc["validity"]["size_ok"] is False
    assert res.doc["validity"]["valid"] is False
    assert any(v["kind"] == "SIZE_CHANGED" for v in res.doc["validity"]["violations"])


# 7 ------------------------------------------------------------------------------------------------------------
def test_7_deterministic_and_idempotent(ctx: dict[str, Any], arranged: Any) -> None:
    names = ["FrontPanel", "FrontPanel_L", "BackPanel", "BackPanel_L", "Pocket", "ShortSleeve", "ShortSleeve_L"]
    text = spec(placements("roll", names))
    out_a, out_b = ctx["tmp"] / "a", ctx["tmp"] / "b"
    r1 = run(ctx, "arrange", text, out_dir=out_a)
    run(ctx, "arrange", text, engine=AlignmentEngine(), out_dir=out_b)
    for f in sorted(q.name for q in out_a.iterdir()):
        if f == "alignment_result.json":
            da, db = (json.loads((d / f).read_text()) for d in (out_a, out_b))
            da["arrangement"].pop("timing_s"), db["arrangement"].pop("timing_s")
            assert da == db
        else:
            assert (out_a / f).read_bytes() == (out_b / f).read_bytes(), f
    again = [{**p, "solved": d["solved"]} for p, d in zip(placements("roll", names), r1.doc["placements"], strict=True)]
    r3 = run(ctx, "arrange", spec(again))
    for p1, p3 in zip(r1.placed, r3.placed, strict=True):
        assert np.abs(p1.vertices - p3.vertices).max() < 1e-4, p1.name


# 9 ------------------------------------------------------------------------------------------------------------
def test_9_performance_on_cpu(ctx: dict[str, Any], aligned: Any, arranged: Any) -> None:
    eng = ctx["engine"]  # warm: the refined meshes and the checks of unchanged pieces are cached
    pl = [{**p, "solved": d["solved"]} for p, d in zip(placements(), aligned.doc["placements"], strict=True)]
    for name in ("Pocket", "ShortSleeve", "Yoke"):
        changed = copy.deepcopy(pl)
        for p in changed:
            if p["piece"] == name:
                p["rotation_deg"] = 5.0
                del p["solved"]
        t = time.perf_counter()
        res = run(ctx, "align", spec(changed), engine=eng)
        assert time.perf_counter() - t < 2.0, (name, time.perf_counter() - t)
        assert res.doc["validity"]["valid"] is True
    t = time.perf_counter()
    run(ctx, "arrange", spec(placements()), engine=eng)
    assert time.perf_counter() - t < 15.0

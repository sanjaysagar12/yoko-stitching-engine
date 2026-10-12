"""Report of a solved layout: per piece numbers, rings (true ease), seam candidates, validity. Metres.

Distances come from the layout check on the placed meshes (``validity``), not from the solver, so a violation the
solver missed shows up here.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from .arrange import Item, Layout
from .errors import RingNotFoundError, UnknownNamedPointError
from .stitch import edge_between
from .widths import row_width

F64 = NDArray[np.float64]
EDGE_SAMPLES = 21


def _r(x: Any, nd: int = 6) -> Any:
    if isinstance(x, np.ndarray):
        return np.round(x.astype(float), nd).tolist()
    return None if x is None else round(float(x), nd)


def layers(lay: Layout) -> dict[int, int]:
    """Radial level of every piece: roll pieces count gaps above the innermost piece of their ring; others 0."""
    out: dict[int, int] = {}
    gap = lay.o.gap_layer if lay.o.gap_layer is not None else lay.o.gap_piece
    for it in lay.items:
        out[it.index] = 0
        if it.mode == "roll" and it.piece is not None:
            base = min(j.piece.theta[0] for j in lay.items if j.mode == "roll" and j.ring == it.ring and j.piece)
            out[it.index] = int(max(0, round((it.piece.theta[0] - base) / max(gap, 1e-6))))
    return out


def build_report(
    lay: Layout, placed: list[Any], timing_s: float, cache: dict[bytes, dict[str, Any]] | None = None
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], list[str]]:
    """(per placement extras, the ``arrangement`` block, the ``validity`` block, warnings).

    ``cache`` (by the content key of each placed piece) keeps the world outline, edges and named points of pieces that
    did not change since an earlier call."""
    cache = cache if cache is not None else {}
    assert lay.check is not None
    o = lay.o
    lv = layers(lay)
    violations = lay.violations()
    warns: list[str] = []
    for v in violations:
        names = " and ".join(v["pieces"])
        warns.append(
            f"{v['kind']}: {names} {'are' if len(v['pieces']) > 1 else 'is'} {v['value_m'] * 1000:.1f} mm "
            f"{'apart' if len(v['pieces']) > 1 else 'from the body'} (required {v['required_m'] * 1000:.0f} mm; "
            f"limit: {v['limit']})"
        )
    bad_pieces: dict[str, list[str]] = {}
    for v in violations:
        for n in v["pieces"]:
            bad_pieces.setdefault(n, []).append(v["kind"])
    extras: list[dict[str, Any]] = []
    for it in lay.items:
        body_d = lay.check.body[it.index][0]
        others = {
            lay.items[b if a == it.index else a].name: _r(d)
            for (a, b), (d, _, _) in lay.check.pairs.items()
            if it.index in (a, b)
        }
        required = {  # gap_layer applies to pairs that face each other, and only if it was set lower
            lay.items[b if a == it.index else a].name: lay.required(facing)
            for (a, b), (_, _, facing) in lay.check.pairs.items()
            if it.index in (a, b)
        }
        theta = it.piece.theta if it.piece is not None else None
        surf = it.piece.surface if it.piece is not None else None
        arr: dict[str, Any] = {
            "ring": it.ring or "",
            "mode": it.mode,
            "layer": lv[it.index],
            "locked": it.locked,
            "solved_now": bool(it.solve),
            "radius_m": _r(theta[0]) if theta is not None and surf is not None and surf.kind == "cylinder" else None,
            "standoff_m": _r(surf.standoff(theta)) if theta is not None and surf is not None else None,
            "shift_m": [_r(theta[1]), _r(theta[2])] if theta is not None else [0.0, 0.0],
            "min_distance_to_avatar_m": _r(body_d),
            "min_distance_to_pieces_m": others,
            "required_gap_to_pieces_m": required,
            "flags": sorted(set(bad_pieces.get(it.name, []))),
            "limits": sorted(it.piece.limits_hit) if it.piece is not None else [],
        }
        extras.append(
            {
                "arrange": arr,
                "solved": {"radius_m": _r(theta[0], 9), "shift_m": [_r(theta[1], 9), _r(theta[2], 9)]}
                if theta is not None
                else None,
            }
        )
    rings = [_ring(lay, ring) for ring in sorted({it.ring for it in lay.items if it.mode == "roll"})]
    edge_data = _edge_data(lay, placed, cache)
    seams = _seam_candidates(lay, lv, edge_data, o.gap_piece * 3.0)
    for k, it in enumerate(lay.items):
        p = placed[k]
        slot = cache.get(p.key) if p.key is not None else None
        if slot is not None and "world" in slot:
            extras[k].update(slot["world"])
            continue
        o2 = np.asarray(it.info.bundle["outline_m"], dtype=np.float64)
        named = it.info.bundle["named_points"]
        w = p.transform(np.array([q["xy_m"] for q in named], dtype=np.float64)) if named else np.zeros((0, 3))
        world = {
            "world_outline_m": _r(p.transform(o2)),
            "world_edges": [
                {
                    "id": e["edge"],
                    "from": e["from"],
                    "to": e["to"],
                    "length_m": _r(e["length_m"]),
                    "polyline_m": _r(e["polyline"]),
                }
                for e in (edge_data[k][2] if k in edge_data else [])
            ],
            "world_named_points": [
                {"name": q["name"], "label": q["label"], "on_outline": q["on_outline"], "world_m": _r(w[i])}
                for i, q in enumerate(named)
            ],
        }
        if slot is not None:
            slot["world"] = world
        extras[k].update(world)
    size_bad = [p.name for p in placed if p.changes_piece_size]
    for n in size_bad:
        violations.append(
            {
                "kind": "SIZE_CHANGED",
                "pieces": [n],
                "value_m": 0.0,
                "required_m": 0.0,
                "at_m": None,
                "limit": "allow_distortion",
            }
        )
    clearance_ok = not any(v["kind"] == "CLEARANCE_BODY" for v in violations)
    gaps_ok = not any(v["kind"] == "GAP_PIECES" for v in violations)
    min_body = min((d for d, _ in lay.check.body.values()), default=None)
    min_pair = min((d for d, _, _ in lay.check.pairs.values()), default=None)
    validity = {
        "valid": clearance_ok and gaps_ok and not size_bad,
        "size_ok": not size_bad,
        "clearance_ok": clearance_ok,
        "gaps_ok": gaps_ok,
        "max_size_change_pct": _r(max((max(p.size_check[k] for k in _PCT) for p in placed), default=0.0), 4),
        "min_distance_to_avatar_m": _r(min_body),
        "min_distance_between_pieces_m": _r(min_pair),
        "required": {
            "clearance_body_m": o.clearance_body,
            "gap_piece_m": o.gap_piece,
            "gap_layer_m": o.as_dict()["gap_layer"],
        },
        "violations": violations,
    }
    if not validity["valid"]:
        warns.insert(0, f"ArrangementInfeasible: {len(violations)} violation(s); the layout is shown but not valid")
    arrangement = {
        "options": o.as_dict(),
        "rings": rings,
        "seam_candidates": seams,
        "violations": [v for v in violations if v["kind"] != "SIZE_CHANGED"],
        "notes": lay.notes,
        "iterations": lay.iterations,
        "timing_s": round(timing_s, 3),
        "convention": "metres, y up, +z front, +x the wearer's left; standoff is display space, never ease",
    }
    return extras, arrangement, validity, warns


_PCT = ("max_edge_change_pct", "perimeter_change_pct", "area_change_pct", "named_point_change_pct")


def _ring(lay: Layout, ring: str) -> dict[str, Any]:
    """Garment loop against skin loop at the binding row (at the positions the user gave: no option changes it)."""
    from .roll import ring_axis

    its = [it for it in lay.items if it.mode == "roll" and it.ring == ring and it.piece is not None]
    base = [it for it in its if it.layer == 0] or its
    c, a = ring_axis(lay.avatar, ring)
    rows: dict[int, float] = {}
    for it in base:
        assert it.piece is not None
        b = it.outline
        t0 = float(it.piece.theta0[2])
        for y in np.arange(float(b[:, 1].min()), float(b[:, 1].max()) + 0.005, 0.005):
            w = row_width(b, float(y))
            if w > 1e-4:
                key = int(round((it.axial0 + y + t0) / 0.005))
                rows[key] = rows.get(key, 0.0) + w
    reg = lay.shells().regions.get(ring)
    radii = [float(it.piece.theta[0]) for it in its if it.piece is not None]
    skins = [it.piece.surface.r_skin for it in its if it.piece is not None and it.piece.surface is not None]
    used = min(r - s for r, s in zip(radii, skins, strict=True)) if radii else 0.0
    required = max(
        (it.needed_radius - it.piece.surface.r_skin for it in its if it.piece and it.piece.surface), default=used
    )
    exceeded = any("max_standoff" in it.piece.limits_hit for it in its if it.piece is not None)
    if not rows or reg is None:
        garment, body, p_star = 0.0, 0.0, None
    else:
        keys = sorted(rows)[:: max(1, len(rows) // 80)]
        sign = 1.0 if a[1] >= 0 else -1.0
        e_y = its[0].piece.surface.e_y if its[0].piece and its[0].piece.surface else a * sign

        def p_of(k: int) -> float:
            return float(reg.coords(c + k * 0.005 * e_y)[0][0])

        tight: list[tuple[float, int]] = []
        for k in keys:
            try:  # rows outside the body region (a collar above the neck) have no section
                tight.append((rows[k] - reg.hull_at(p_of(k), 0.04)[0].perimeter, k))
            except RingNotFoundError:
                continue
        if tight:
            kb = max(tight)[1]
            p_star = p_of(kb)
            garment, body = float(rows[kb]), float(reg.skin_loop(p_star))
        else:
            garment, body, p_star = 0.0, 0.0, None
    g_r, b_r = round(garment, 6), round(body, 6)
    return {
        "ring": ring,
        "pieces": [it.name for it in base],
        "body_loop_m": b_r,
        "garment_loop_m": g_r,
        "true_ease_m": round(g_r - b_r, 6),
        "standoff_used_m": _r(used),
        "standoff_required_m": _r(max(required, used)),
        "radius_m": _r(lay.ring_radius.get(ring, min(radii, default=0.0))),
        "exceeded": bool(exceeded),
        "binding_row_m": _r(p_star),
        "violating_rows": [],
        "note": "true ease = garment loop - skin loop at the binding row; the standoff is display space, not ease",
    }


def _edge_data(
    lay: Layout, placed: list[Any], cache: dict[bytes, dict[str, Any]]
) -> dict[int, tuple[F64, NDArray[np.int64], list[dict[str, Any]]]]:
    """Per piece: all edge samples in world space (stacked), the edge index of every sample and the edge records."""
    data: dict[int, tuple[F64, NDArray[np.int64], list[dict[str, Any]]]] = {}
    for it in lay.items:
        slot = cache.get(placed[it.index].key) if placed[it.index].key is not None else None
        if slot is not None and "edges" in slot:
            if slot["edges"] is not None:
                data[it.index] = slot["edges"]
            continue
        b = it.info.bundle
        polys: list[F64] = []
        meta: list[dict[str, Any]] = []
        for e in b.get("edges", []):
            try:
                flat, length = edge_between(b, e["from"], e["to"], EDGE_SAMPLES)
            except UnknownNamedPointError:
                continue
            polys.append(placed[it.index].transform(flat))
            meta.append({"edge": f"{e['from']}-{e['to']}", "from": e["from"], "to": e["to"], "length_m": length})
        entry = None
        if polys:
            eid = np.repeat(np.arange(len(polys)), EDGE_SAMPLES)
            entry = (np.vstack(polys), eid, [{**m, "polyline": pl} for m, pl in zip(meta, polys, strict=True)])
            data[it.index] = entry
        if slot is not None:
            slot["edges"] = entry
    return data


def _seam_candidates(
    lay: Layout, lv: dict[int, int], data: dict[int, tuple[F64, NDArray[np.int64], list[dict[str, Any]]]], thr: float
) -> list[dict[str, Any]]:
    """Edge pairs of pieces on the same layer closer than ``thr`` and running side by side for >= 2 samples."""
    trees = {k: cKDTree(v[0]) for k, v in data.items()}
    out: list[dict[str, Any]] = []
    items: list[Item] = lay.items
    for a in items:
        for b in items:
            if a.index >= b.index or lv[a.index] != lv[b.index] or a.index not in data or b.index not in data:
                continue
            pa, ea, ma = data[a.index]
            pb, eb, mb = data[b.index]
            d1, j1 = trees[b.index].query(pa)
            d2, j2 = trees[a.index].query(pb)
            keys_a = np.flatnonzero(d1 < thr)
            keys_b = np.flatnonzero(d2 < thr)
            ka = ea[keys_a] * len(mb) + eb[j1[keys_a]]
            kb = ea[j2[keys_b]] * len(mb) + eb[keys_b]
            cnt_a = dict(zip(*np.unique(ka, return_counts=True), strict=True))
            cnt_b = dict(zip(*np.unique(kb, return_counts=True), strict=True))
            for key in sorted(set(cnt_a) | set(cnt_b)):
                ca, cb = int(cnt_a.get(key, 0)), int(cnt_b.get(key, 0))
                if min(ca, cb) < 2:
                    continue
                i_a, i_b = divmod(int(key), len(mb))
                dm = float(min(d1[keys_a][ka == key].min(initial=9.0), d2[keys_b][kb == key].min(initial=9.0)))
                ia, ib = ma[i_a], mb[i_b]
                out.append(
                    {
                        "a": {
                            "piece": a.name,
                            "edge": ia["edge"],
                            "from": ia["from"],
                            "to": ia["to"],
                            "length_m": _r(ia["length_m"]),
                            "polyline_m": _r(ia["polyline"]),
                        },
                        "b": {
                            "piece": b.name,
                            "edge": ib["edge"],
                            "from": ib["from"],
                            "to": ib["to"],
                            "length_m": _r(ib["length_m"]),
                            "polyline_m": _r(ib["polyline"]),
                        },
                        "distance_m": _r(dm),
                        "length_ratio": _r(ia["length_m"] / max(ib["length_m"], 1e-12), 4),
                        "overlap": _r(min(ca, cb) / EDGE_SAMPLES, 3),
                    }
                )
    out.sort(key=lambda c: (c["distance_m"], c["a"]["piece"], c["b"]["piece"], c["a"]["edge"], c["b"]["edge"]))
    return out

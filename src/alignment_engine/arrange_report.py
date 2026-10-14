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
from .validity import TOLERANCE, PairView, arc, concentric_warnings, pair_violations
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
    pair_viol, pair_warns = pair_violations(_pair_views(lay))
    violations = lay.violations() + pair_viol
    flagged = {n for v in violations for n in v["pieces"]}
    fits = {r: lay.fit_radius([j for j in lay.items if j.mode == "roll" and j.ring == r]) for r in lay.ring_names()}
    checks = {it.index: _placement_check(lay, it, lv, it.name in flagged, fits) for it in lay.items}
    violations += [v for c in checks.values() for v in c["violations"]]
    violations = _pair_no_room(lay, violations)
    concentric = _concentric(lay)
    warns: list[str] = [*pair_warns, *_arm_zone_warnings(lay)]
    for v in violations:
        names = " and ".join(v["pieces"])
        if "hint" in v:
            warns.append(f"{v['kind']}: {v['hint']}")
            continue
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
            "layer_reason": (it.layer_reason or "solver") if lv[it.index] else "",
            "pair_with": it.partner,
            "solved_now": bool(it.solve or it.regrown),
            "radius_m": _r(theta[0]) if theta is not None and surf is not None and surf.kind == "cylinder" else None,
            "standoff_m": _r(surf.standoff(theta)) if theta is not None and surf is not None else None,
            "shift_m": [_r(theta[1]), _r(theta[2])] if theta is not None else [0.0, 0.0],
            "min_distance_to_avatar_m": _r(body_d),
            "min_distance_to_pieces_m": others,
            "required_gap_to_pieces_m": required,
            "flags": sorted(set(bad_pieces.get(it.name, []))),
            "limits": checks[it.index]["limits"],
            "reasons": checks[it.index]["reasons"],
            "point_distance_m": checks[it.index]["point_distance_m"],
            "slide_m": checks[it.index]["slide_m"],
        }
        if it.anchor is not None:
            arr["anchor"] = it.anchor
        arr["applied_rotation_deg"] = _r(it.rotation)
        if it.orient_note:
            arr["orient"] = it.orient_note
        if any(it.tilt):
            arr["tilt_deg"] = {"pitch_deg": _r(it.tilt[0]), "roll_deg": _r(it.tilt[1])}
        off = lay.registration_offset(it)
        if it.registration is not None and off is not None:
            reg = it.registration
            arr["registration"] = {
                "level": reg.get("level") or "point",
                "registered": reg["registered"],
                "level_y_m": _r(reg["level_y_m"]),
                "offset_m": _r(off),
                "shift_m": _r(reg["shift_m"]),
            }
        extras.append(
            {
                "arrange": arr,
                "solved": {
                    "radius_m": _r(theta[0], 9),
                    "shift_m": [_r(theta[1], 9), _r(theta[2], 9)],
                    **({"needed_radius_m": _r(it.needed_radius, 9)} if it.needed_radius else {}),
                }
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
    pairs_ok = not any(v["kind"].startswith("PAIR_") for v in violations)
    placement_ok = not any(v["kind"] in PLACEMENT_KINDS for v in violations)
    min_body = min((d for d, _ in lay.check.body.values()), default=None)
    min_pair = min((d for d, _, _ in lay.check.pairs.values()), default=None)
    validity = {
        "valid": clearance_ok and gaps_ok and pairs_ok and placement_ok and not size_bad,
        "size_ok": not size_bad,
        "clearance_ok": clearance_ok,
        "gaps_ok": gaps_ok,
        "pairs_ok": pairs_ok,
        "placement_ok": placement_ok,
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
    warns += concentric
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


def _arc(lay: Layout, it: Item) -> tuple[float, float] | None:
    """Arc of a rolled piece about its ring axis (None for flat pieces)."""
    from .roll import ring_axis

    if it.piece is None or it.piece.surface is None or it.piece.surface.kind != "cylinder":
        return None
    if it.mode == "roll":
        c, a = ring_axis(lay.avatar, it.ring)
    else:
        c, a = it.piece.surface.origin, it.piece.surface.e_y
    return arc(it.piece.world, c, a)


def _pair_views(lay: Layout) -> list[tuple[PairView, PairView, bool]]:
    """Both halves of every pair, and whether they share one ring (then they must not stack)."""
    out = []
    for a, b, same in lay.pairs:
        if a.piece is not None and b.piece is not None:
            one_ring = same or (a.mode == "roll" and a.ring == b.ring)
            out.append(
                (
                    PairView(a.name, a.piece.theta, _arc(lay, a), a.locked),
                    PairView(b.name, b.piece.theta, _arc(lay, b), b.locked),
                    one_ring,
                )
            )
    return out


def _pair_no_room(lay: Layout, violations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """A pair whose halves cannot slide far enough apart within ``max_shift`` is one ``PAIR_NO_ROOM`` violation (with
    what to change) instead of the gap, stacking and slide violations it causes."""
    for a, b, _ in lay.pairs:
        room = a.piece.pair_room if a.piece is not None else None
        if room is None:
            continue
        need, avail = room
        names = {a.name, b.name}
        violations = [
            v
            for v in violations
            if not (
                (v["kind"] in ("GAP_PIECES", "PAIR_STACKED") and set(v["pieces"]) == names)
                or (v["kind"] == "NOT_AT_POINT" and v["pieces"][0] in names)
            )
        ]
        violations.append(
            {
                "kind": "PAIR_NO_ROOM",
                "pieces": [a.name, b.name],
                "value_m": round(need, 6),
                "required_m": round(avail, 6),
                "at_m": None,
                "limit": "max_shift",
                "hint": (
                    f"{a.name} and {b.name} overlap at the centre line: each half needs an outward slide of "
                    f"{need * 100:.1f} cm from {a.point.name} / {b.point.name} to be {lay.o.gap_piece * 100:.0f} cm "
                    f"apart, max_shift allows {avail * 100:.1f} cm. Choose points farther apart, a reference nearer "
                    "the centre-line edge, or a narrower piece (or raise max_shift)"
                ),
            }
        )
    return violations


def _concentric(lay: Layout) -> list[str]:
    """NOT_CONCENTRIC warnings per ring (pieces with a user layer or a lock have their reason and are left out)."""
    gl = lay.o.gap_layer if lay.o.gap_layer is not None else lay.o.gap_piece
    out: list[str] = []
    for ring in sorted({it.ring for it in lay.items if it.mode == "roll"}):
        rows = []
        for it in lay.items:
            if it.mode != "roll" or it.ring != ring or it.piece is None or it.locked or it.user_layer is not None:
                continue
            a = _arc(lay, it)
            if a is not None:
                y = it.piece.world[:, 1]
                rows.append((it.name, float(it.piece.theta[0]), a, (float(y.min()), float(y.max()))))
        out += concentric_warnings(ring, rows, gl)
    return out


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


MAX_SLIDE = 0.15  # a solver slide farther than this from the anchor leaves the piece off its point (NOT_AT_POINT)
REGISTRATION_TOLERANCE = 0.015  # a registered point this far from its level height is not registered
ORIENTATION_TOLERANCE_DEG = 5.0
PLACEMENT_KINDS = ("STANDOFF_EXCEEDED", "NOT_AT_POINT", "FLOOR", "OFF_REGION", "REGISTRATION", "ORIENTATION")


def _placement_check(
    lay: Layout, it: Item, lv: dict[int, int], flagged: bool, fits: dict[str, float]
) -> dict[str, Any]:
    """Does the piece sit at its point, inside the limits? Limits, the explanation of its radius and slide, and the
    ``STANDOFF_EXCEEDED`` / ``NOT_AT_POINT`` violations (with a hint) when it does not. Locked pieces are the user's."""
    out: dict[str, Any] = {"limits": [], "reasons": {}, "point_distance_m": None, "slide_m": 0.0, "violations": []}
    p = it.piece
    if p is None or p.surface is None:
        return out
    o, surf, th = lay.o, p.surface, p.theta
    limits = set(p.limits_hit)
    if surf.kind == "cylinder" and th[0] >= p.r_max - 1e-6:  # at the limit, or wider than it allows
        limits.add("max_standoff")
    if not flagged:  # the solver stopped short of its margin, but the measured layout holds: no limit mattered
        limits.discard("iterations")
    pt = it.point
    surface_point = pt.position - pt.offset_m * pt.normal / np.linalg.norm(pt.normal)
    out["point_distance_m"] = _r(float(np.linalg.norm(p.world - surface_point, axis=1).min()))
    slide = float(np.linalg.norm(th[1:] - p.theta0[1:]))
    out["slide_m"] = _r(slide)
    standoff = surf.standoff(th)
    gl = o.gap_layer if o.gap_layer is not None else o.gap_piece
    if surf.kind == "cylinder":
        if not it.needed_radius and not it.solve:  # carried from an earlier answer: its body clearance radius again
            it.needed_radius = lay.clear_radius(it)
        fit = fits.get(it.ring, 0.0) if it.mode == "roll" else 0.0
        base = {"piece width": p.r_min, "ring fit": fit, "body clearance": it.needed_radius}
        why, r_need = max(base.items(), key=lambda kv: kv[1])
        r_need += lv.get(it.index, 0) * gl
        if it.mode == "roll" and lv.get(it.index, 0):  # a layer lies over its ring's innermost piece, wherever that is
            ring_base = min(j.piece.theta[0] for j in lay.items if j.mode == "roll" and j.ring == it.ring and j.piece)
            if ring_base + lv[it.index] * gl > r_need:
                why, r_need = f"layer over the ring at {ring_base * 100:.1f} cm", ring_base + lv[it.index] * gl
        explained = r_need - surf.r_skin
        if lv.get(it.index, 0):
            why += f", layer {lv[it.index]} ({it.layer_reason or 'solver'})"
        if th[0] > r_need + 0.002:
            why = f"solver: {(th[0] - r_need) * 100:.1f} cm out to clear the body or a piece (needs {why})"
        out["reasons"]["radius"] = f"{th[0] * 100:.1f} cm: {why}"
    else:
        explained = float(p.theta0[0])
        out["reasons"]["radius"] = f"offset {th[0] * 100:.1f} cm from the point"
    if slide > 0.001:
        out["reasons"]["shift"] = f"solver slid it {slide * 100:.1f} cm from its anchor to clear the body or a piece"
    elif it.anchor is not None and abs(it.anchor["shift_m"]) > 1e-6:
        out["reasons"]["shift"] = it.anchor["reason"]
    if it.locked:
        out["limits"] = sorted(limits)
        return out
    # what the ring needs, plus room for the solver to keep the clearance and one gap to a neighbour
    allowed = max(o.max_standoff_from_point, explained + o.clearance_body + o.gap_piece)
    if "max_standoff" in limits or standoff > allowed + 1e-4:
        req = o.max_standoff if "max_standoff" in limits else allowed
        out["violations"].append(
            _violation("STANDOFF_EXCEEDED", it, standoff, req, _hint(lay, it, standoff, req), limits)
        )
    if "max_shift" in limits or slide > min(MAX_SLIDE, o.max_shift) + 1e-4:
        req = min(MAX_SLIDE, o.max_shift)
        hint = (
            f"{it.name} had to slide {slide * 100:.0f} cm from {pt.name} (at most {req * 100:.0f} cm) to clear the "
            "body or other pieces: pick a free point or another reference, or move the piece that is in the way"
        )
        out["violations"].append(_violation("NOT_AT_POINT", it, slide, req, hint, limits))
    floor = float(p.world[:, 1].min())
    if floor < o.clearance_body - TOLERANCE:
        hint = (
            f"{it.name} reaches {floor * 100:.1f} cm above the floor, below the clearance "
            f"({o.clearance_body * 100:.0f} cm): register it on a body level (anchor {{level, piece_point}}) or move "
            "it up (shift_m)"
        )
        out["violations"].append(_violation("FLOOR", it, floor, o.clearance_body, hint, limits))
    reach = max(allowed, o.max_standoff_from_point) + o.max_standoff_from_point
    dist = float(np.linalg.norm(p.world - surface_point, axis=1).min())
    if it.registration is None and not it.pl.get("over") and dist > reach + 1e-4:
        hint = (
            f"{it.name} lies {dist * 100:.0f} cm from {pt.name} (at most {reach * 100:.0f} cm): it was moved away from "
            "its point (anchor, slide); pick a point on the body region it belongs to or register it on a level"
        )
        out["violations"].append(_violation("OFF_REGION", it, dist, reach, hint, limits))
    off = lay.registration_offset(it)
    if off is not None and abs(off) > REGISTRATION_TOLERANCE:
        reg = it.registration or {}
        hint = (
            f"{reg.get('registered')} of {it.name} is {off * 100:+.1f} cm from the {reg.get('level') or 'point'} level "
            f"(at most {REGISTRATION_TOLERANCE * 100:.1f} cm): the solver had to slide it to clear the body or a piece"
        )
        out["violations"].append(_violation("REGISTRATION", it, abs(off), REGISTRATION_TOLERANCE, hint, limits))
    ang = _orientation_error(it)
    if ang is not None and ang > ORIENTATION_TOLERANCE_DEG:
        hint = f"{it.name} is turned {ang:.0f} degrees from the pattern's up without a rotation, orient or tilt asking"
        out["violations"].append(_violation("ORIENTATION", it, np.radians(ang), np.radians(5.0), hint, limits))
    out["limits"] = sorted(limits)
    return out


def _orientation_error(it: Item) -> float | None:
    """Angle (degrees) between the world image of the pattern's up (after ``orient``) at the reference and the up the
    placement asked for (the point's up, the ring axis); None when the user asked for a turn (``rotation_deg``,
    a quarter-turn ``orient``, ``tilt``) or the piece has no size-exact surface."""
    p = it.piece
    if p is None or p.surface is None or it.axis_up is None:
        return None
    if float(it.pl.get("rotation_deg", 0.0)) or any(it.tilt) or not isinstance(it.pl.get("orient", "auto"), str):
        return None
    rho = np.radians(it.rotation)
    d = np.array([[-np.sin(rho), np.cos(rho)]]) * 1e-3  # the pattern's +y in local coordinates
    w = p.surface.map(np.vstack([np.zeros((1, 2)), d]), p.theta)
    v = w[1] - w[0]
    c = float(v @ it.axis_up) / max(float(np.linalg.norm(v)), 1e-18)
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def _violation(kind: str, it: Item, value: float, required: float, hint: str, limits: set[str]) -> dict[str, Any]:
    limit = ",".join(sorted(limits)) or "none"
    return {
        "kind": kind,
        "pieces": [it.name],
        "value_m": round(float(value), 6),
        "required_m": round(float(required), 6),
        "at_m": None,
        "limit": limit,
        "hint": hint,
    }


def _hint(lay: Layout, it: Item, standoff: float, allowed: float) -> str:
    """What to change when a piece cannot sit close enough to its point."""
    if standoff >= lay.o.max_standoff - 1e-6:
        head = f"{it.name} reached max_standoff ({lay.o.max_standoff * 100:.0f} cm) at {it.point.name}"
    else:
        head = (
            f"{it.name} sits {standoff * 100:.0f} cm from the body at {it.point.name} (at most {allowed * 100:.0f} cm)"
        )
    if it.ring.startswith(("arm_", "leg_")) or it.point.bv.startswith(("UpperArm", "Forearm", "Thigh", "Shin")):
        return (
            f'{head}: it does not fit around the limb there; try "anchor": "root" (or "tip"), another point of '
            "the limb, or a smaller piece"
        )
    down = _arm_overlap(lay, it)
    if down > 0.0:
        side = it.point.name.rsplit("_", 1)[-1][:1]
        alt = f"{it.point.bv}_FrontSide_{side}"
        alt_txt = f" or use {alt}" if alt in lay.avatar.points else ""
        return (
            f"{head}: the T-pose arm lies under it; move it down {down * 100:.0f} cm (shift_m [0, "
            f"{-down:.2f}]){alt_txt}"
        )
    return f"{head}: choose another point or reference, or a smaller piece"


def _arm_overlap(lay: Layout, it: Item) -> float:
    """How far a torso piece reaches above the bottom of a T-pose arm beside it (0 if no arm is under it)."""
    p = it.piece
    if p is None or it.ring != "torso":
        return 0.0
    v = lay.avatar.vertices
    xs = p.world[:, 0]
    best = 0.0
    for side in ("arm_L", "arm_R"):
        reg = lay.shells().regions.get(side)
        if reg is None:
            continue
        arm = v[reg.vertex_ids]
        under = arm[(arm[:, 0] >= xs.min()) & (arm[:, 0] <= xs.max())]
        if len(under):
            best = max(best, float(p.world[:, 1].max() - (under[:, 1].min() - lay.o.clearance_body)))
    return best


def _arm_zone_warnings(lay: Layout) -> list[str]:
    """``ARM_ZONE``: a torso piece reaches into the space a sleeve needs around the arm (the arm axis beyond the
    shoulder joint, within the arm radius plus clearance and gap); sleeves placed there slide outward along the arm."""
    out: list[str] = []
    vols = lay.avatar.volumes
    for it in lay.items:
        if it.ring != "torso" or it.piece is None:
            continue
        for s in ("L", "R"):
            ua, fa = vols.get(f"UpperArm_{s}"), vols.get(f"Forearm_{s}")
            if ua is None or fa is None:
                continue
            d = fa.end - ua.start
            k = (it.piece.world - ua.start) @ d / (d @ d)
            dist = np.linalg.norm(it.piece.world - (ua.start + np.clip(k, 0.0, 1.0)[:, None] * d), axis=1)
            zone = ua.radius_m + lay.o.clearance_body + lay.o.gap_piece
            inside = (k > 0.0) & (dist < zone)
            if inside.any():
                out.append(
                    f"ARM_ZONE: {it.name} reaches {float((zone - dist[inside]).max()) * 100:.1f} cm into the space "
                    f"around the {'left' if s == 'L' else 'right'} arm; a sleeve there slides outward along the arm"
                )
    return out

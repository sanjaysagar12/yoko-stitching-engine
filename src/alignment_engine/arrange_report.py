"""Report of an arrangement (Part D): distances, size preservation, ring numbers, seam candidates. Metres.

All numbers are measured on the placed meshes, independently of the solver, so a violation the solver missed shows up.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from .arrange import RANGE_MARGIN, STEEP_PERIMETER_SLOPE, Arranger, Item
from .errors import UnknownNamedPointError
from .stitch import edge_between
from .verify import self_overlaps, surface_samples

F64 = NDArray[np.float64]
TOLERANCE = 0.001  # a distance this far below its requirement is reported as a violation
EDGE_SAMPLES = 21


def _r(x: Any, nd: int = 6) -> Any:
    if isinstance(x, np.ndarray):
        return np.round(x.astype(float), nd).tolist()
    return None if x is None else round(float(x), nd)


def _area(v: F64, f: NDArray[np.int64]) -> float:
    t = v[f]
    return float(0.5 * np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1).sum())


def _edges(f: NDArray[np.int64]) -> NDArray[np.int64]:
    return np.unique(np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1), axis=0)


def _points(ar: Arranger, it: Item, placed: Any) -> F64:
    """Vertices plus the outline every 5 mm (free pieces) or surface samples (locked pieces)."""
    if it.locked:
        return surface_samples(placed.vertices, placed.faces, 0.006)
    return np.vstack([placed.vertices, ar.boundary_world(it)])


def build_report(
    ar: Arranger, placed: list[Any], timing_s: float
) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
    """(per placement dicts with ``arrange``, ``world_outline_m``, ``world_named_points``; the ``arrangement``
    block; warnings)."""
    o = ar.o
    warns: list[str] = []
    violations: list[dict[str, Any]] = []
    pts = {it.index: _points(ar, it, placed[it.index]) for it in ar.items}
    trees = {k: cKDTree(v) for k, v in pts.items()}

    # distances between pieces and to the body
    pair: dict[tuple[int, int], float] = {}
    near: dict[tuple[int, int], tuple[F64, F64]] = {}
    for a in ar.items:
        for b in ar.items:
            if a.index >= b.index:
                continue
            d1, j1 = trees[b.index].query(pts[a.index])
            k = int(np.argmin(d1))
            pair[(a.index, b.index)] = float(d1[k])
            near[(a.index, b.index)] = (pts[a.index][k], pts[b.index][j1[k]])
    out_pieces: list[dict[str, Any]] = []
    for it in ar.items:
        p = placed[it.index]
        sd = ar.body.signed_distance(pts[it.index])
        k = int(np.argmin(sd))
        dmin = float(sd[k])
        others = {ar.items[b if a == it.index else a].name: _r(d) for (a, b), d in pair.items() if it.index in (a, b)}
        flags: list[str] = []
        if dmin < o.clearance_body - TOLERANCE:
            flags.append("CLEARANCE_BODY")
            warns.append(
                f"CLEARANCE_BODY: {it.name} comes within {dmin * 1000:.1f} mm of the body (clearance_body "
                f"{o.clearance_body * 1000:.0f} mm)"
            )
            violations.append(
                {
                    "kind": "CLEARANCE_BODY",
                    "pieces": [it.name],
                    "value_m": _r(dmin),
                    "required_m": o.clearance_body,
                    "at_m": _r(pts[it.index][k]),
                }
            )
        arr: dict[str, Any] = {
            "ring": it.region,
            "layer": it.layer,
            "locked": it.locked,
            "standoff_m": None,
            "shift_m": _r(it.delta) if not it.locked else [0.0, 0.0],
            "min_distance_to_avatar_m": _r(dmin),
            "min_distance_to_pieces_m": others,
            "flags": flags,
            "area_ratio": None,
            "edge_change_p50_pct": None,
            "edge_change_p90_pct": None,
            "max_edge_change_pct": None,
            "max_edge_change_body_pct": None,
            "max_edge_change_shoulder_pct": None,
            "self_intersections": int(self_overlaps(p.vertices, p.faces)),
            "vertices_outside_shell": 0,
        }
        if not it.locked:
            _size_numbers(ar, it, p, arr)
            arr["standoff_m"] = _r(float(ar.hrows(it)[int(ar.surf[it.region].row_of(np.array([it.p0]))[0])]))
        out_pieces.append({"arrange": arr})
    for (ka, kb), d in sorted(pair.items()):
        ia, ib = ar.items[ka], ar.items[kb]
        need = o.gap_piece if ia.layer == ib.layer else (o.gap_layer if abs(ia.layer - ib.layer) == 1 else 0.0)
        if d < need - TOLERANCE:
            for other in (ia, ib):
                out_pieces[other.index]["arrange"]["flags"].append("GAP_PIECES")
            warns.append(
                f"GAP_PIECES: {ia.name} and {ib.name} are {d * 1000:.1f} mm apart (required {need * 1000:.0f} mm)"
            )
            qa, qb = near[(ka, kb)]
            violations.append(
                {
                    "kind": "GAP_PIECES",
                    "pieces": [ia.name, ib.name],
                    "value_m": _r(d),
                    "required_m": need,
                    "at_m": _r(0.5 * (qa + qb)),
                }
            )

    rings = []
    for region in sorted(ar.surf):
        its = ar.free(region, 0)
        if not its:
            continue
        rings.append(_ring(ar, region, its, placed, warns, violations))
    edge_data = _edge_data(ar, placed)
    seams = _seam_candidates(ar, edge_data, o.gap_piece * 3.0)
    for k, it in enumerate(ar.items):
        p = placed[k]
        outline = it.refined.bundle["outline_m"] if it.refined is not None else it.info.bundle["outline_m"]
        o2 = np.asarray(outline, dtype=np.float64)
        out_pieces[k]["world_outline_m"] = _r(p.transform(o2))
        out_pieces[k]["world_edges"] = [
            {
                "id": e["edge"],
                "from": e["from"],
                "to": e["to"],
                "length_m": _r(e["length_m"]),
                "polyline_m": _r(e["polyline"]),
            }
            for e in (edge_data[k][2] if k in edge_data else [])
        ]
        named = it.info.bundle["named_points"]
        if named:
            w = p.transform(np.array([q["xy_m"] for q in named], dtype=np.float64))
            out_pieces[k]["world_named_points"] = [
                {"name": q["name"], "label": q["label"], "on_outline": q["on_outline"], "world_m": _r(w[i])}
                for i, q in enumerate(named)
            ]
        else:
            out_pieces[k]["world_named_points"] = []
    arrangement = {
        "options": o.as_dict(),
        "rings": rings,
        "seam_candidates": seams,
        "violations": violations,
        "notes": ar.notes,
        "timing_s": round(timing_s, 3),
        "convention": "metres, y up, +z front, +x the wearer's left; standoff is display space, never ease",
    }
    return out_pieces, arrangement, warns


def _size_numbers(ar: Arranger, it: Item, p: Any, arr: dict[str, Any]) -> None:
    r = it.refined
    assert r is not None
    flat = r.vertices[:, :2]
    area_flat = _area(r.vertices, r.faces)
    arr["area_ratio"] = _r(_area(p.vertices, p.faces) / area_flat, 5)
    e = _edges(r.faces)
    fl = np.linalg.norm(flat[e[:, 0]] - flat[e[:, 1]], axis=1)
    wl = np.linalg.norm(p.vertices[e[:, 0]] - p.vertices[e[:, 1]], axis=1)
    pct = np.abs(wl / np.maximum(fl, 1e-12) - 1.0) * 100.0
    surf = ar.surf[it.region]
    reg = surf.region
    pv = ar.p_of(it, it.uv[:, 1])
    outside = (pv < reg.p_range[0] - RANGE_MARGIN) | (pv > reg.p_range[1] + RANGE_MARGIN)
    per = surf.stack.perim
    step = 10
    slope = np.zeros(len(per))
    slope[step:-step] = np.abs(per[2 * step :] - per[: -2 * step]) / (2 * step * 0.001)
    steep = (slope[surf.row_of(pv)] > STEEP_PERIMETER_SLOPE) | outside
    es = steep[e[:, 0]] | steep[e[:, 1]]
    arr["edge_change_p50_pct"] = _r(float(np.percentile(pct, 50)), 3)
    arr["edge_change_p90_pct"] = _r(float(np.percentile(pct[~es], 90)) if (~es).any() else 0.0, 3)
    arr["max_edge_change_pct"] = _r(float(pct.max()), 3)
    arr["max_edge_change_body_pct"] = _r(float(pct[~es].max()) if (~es).any() else 0.0, 3)
    arr["max_edge_change_shoulder_pct"] = _r(float(pct[es].max()), 3) if es.any() else None
    arr["vertices_outside_shell"] = int(outside.sum())
    if outside.any():
        ar.warnings.append(
            f"SHELL_RANGE: {int(outside.sum())} vertices of {it.name} lie beyond the end of the {it.region} shell; the "
            "nearest cross-section is used there"
        )


def _ring(
    ar: Arranger, region: str, its: list[Item], placed: list[Any], warns: list[str], violations: list[dict[str, Any]]
) -> dict[str, Any]:
    o = ar.o
    st = ar.rings[region]
    reg = ar.shells.regions[region]
    ps, total, count = ar.row_sums(region, initial=True)  # at the positions the user gave: no option changes the ease
    # the binding row is where the garment is widest compared with the convex section of the body (a fixed 4 cm window,
    # so that no option changes it); there the standoff has to be largest
    cand = np.flatnonzero(count > 0)
    cand = cand[:: max(1, len(cand) // 80)]
    tight = np.array([total[i] - reg.hull_at(float(ps[i]), 0.04)[0].perimeter for i in cand])
    idx = int(cand[int(np.argmax(tight))])
    p_star = float(ps[idx])
    garment = float(total[idx])
    body = float(reg.skin_loop(p_star))
    g_r, b_r = round(garment, 6), round(body, 6)
    used = float(st.used[0])
    rows: list[dict[str, float]] = []
    if st.exceeded:
        bad = np.flatnonzero(np.isfinite(st.proxy_need) & (st.proxy_need > used + 0.0005))
        rows = [
            {"row_m": round(float(st.proxy_rows_m[i]), 3), "needed_standoff_m": round(float(st.proxy_need[i]), 4)}
            for i in bad[:: max(1, len(bad) // 20)]
        ]
        warns.append(
            f"STANDOFF_EXCEEDED: ring {region} needs a standoff of {st.required * 100:.1f} cm but max_standoff is "
            f"{o.max_standoff * 100:.1f} cm; shown at {used * 100:.1f} cm, pieces may overlap each other or the body"
        )
        violations.append(
            {
                "kind": "STANDOFF_EXCEEDED",
                "pieces": [i.name for i in its],
                "value_m": _r(used),
                "required_m": _r(st.required),
                "at_m": None,
            }
        )
    return {
        "ring": region,
        "pieces": [i.name for i in its],
        "body_loop_m": b_r,
        "garment_loop_m": g_r,
        "true_ease_m": round(g_r - b_r, 6),
        "standoff_used_m": _r(used),
        "standoff_required_m": _r(st.required),
        "exceeded": bool(st.exceeded),
        "binding_row_m": _r(p_star),
        "violating_rows": rows,
        "note": "true ease = garment loop - skin loop at the binding row; the standoff is display space, not ease",
    }


def _edge_data(ar: Arranger, placed: list[Any]) -> dict[int, tuple[F64, NDArray[np.int64], list[dict[str, Any]]]]:
    """Per piece: all edge samples in world space (stacked), the edge index of every sample and the edge records."""
    data: dict[int, tuple[F64, NDArray[np.int64], list[dict[str, Any]]]] = {}
    for it in ar.items:
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
        if not polys:
            continue
        eid = np.repeat(np.arange(len(polys)), EDGE_SAMPLES)
        data[it.index] = (np.vstack(polys), eid, [{**m, "polyline": pl} for m, pl in zip(meta, polys, strict=True)])
    return data


def _seam_candidates(
    ar: Arranger, data: dict[int, tuple[F64, NDArray[np.int64], list[dict[str, Any]]]], thr: float
) -> list[dict[str, Any]]:
    trees = {k: cKDTree(v[0]) for k, v in data.items()}
    out: list[dict[str, Any]] = []
    for a in ar.items:
        for b in ar.items:
            if a.index >= b.index or a.layer != b.layer or a.index not in data or b.index not in data:
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
                if min(ca, cb) < 2:  # the two edges must run side by side for at least two samples
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

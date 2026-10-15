"""Independent check of a stitching result (``python -m stitcher check``), in the style of ``assembler/verify.py``.

It shares no geometry code with the engine: chains are re-walked over the alignment result's ``world_edges`` graph
(not the bundle's arc order), lengths are summed from the alignment engine's own edge lengths, positions between
named points are interpolated along the ``world_edges`` samples, claims and coverage are kept per edge, crossings are
recounted with a separate implementation and attach seams are re-projected with ``assembler.verify.Surf``. Rows
``{id, seam, status: pass|fail|warn|info, value, limit, hint}``.

Thresholds (all new, only here; nothing of the assembler is relaxed):

* ``ENDPOINT_M`` 1 mm: a seam end must sit on its chain's end (the engine interpolates on exact samples).
* ``WORLD_LENGTH_PCT`` 2 %: placements are isometric; the world polyline (chords between exact samples at most L/20
  apart) can only be shorter by its chordal error, so a bigger difference is a wrong chain or a wrong mapping.
* ``LENGTH_CM`` 0.01 cm: the flat length must equal the alignment engine's edge lengths (both are flat arc lengths).
* ``DISTRIBUTE_MAX_PCT`` 5 %: ``distribute`` is ease (sleeve cap into armhole); more must be declared ``gather``.
* ``GATHER_WARN`` 1.5: a gather larger than that is reported.
* ``OVERLAP_M`` 1 mm: two claims sharing more than that of one edge stitch it twice.
* ``UNCOVERED_M`` 5 mm: an edge with more than that neither sewn nor ignored is an orphan (warning).
* ``ONTO_M`` 1 mm: an attach sample must be the host surface's closest point to its side-a point.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .load import ROOT, geometry_hash, load_run, read_json, sha256, validate

F64 = NDArray[np.float64]
ENDPOINT_M = 1e-3
WORLD_LENGTH_PCT = 2.0
LENGTH_CM = 0.01
DISTRIBUTE_MAX_PCT = 5.0
GATHER_WARN = 1.5
OVERLAP_M = 1e-3
UNCOVERED_M = 5e-3
ONTO_M = 1e-3
COINCIDENT_M = 1e-6
SIZE_ROWS = "size."

Key = tuple[str, frozenset[str]]


def _row(rows: list[dict[str, Any]], id_: str, status: str, value: Any = "", limit: Any = "", hint: str = "",
         seam: str = "") -> None:  # fmt: skip
    rows.append({"id": id_, "seam": seam, "status": status, "value": value, "limit": limit, "hint": hint})


def verdict(rows: list[dict[str, Any]]) -> bool:
    return not any(r["status"] == "fail" for r in rows)


class Graph:
    """Outline of one placed piece as the alignment result gives it: world_edges between named points (points at
    the same world position count as one node), each with its length, start node and sample polyline."""

    def __init__(self, placement: dict[str, Any]) -> None:
        self.named = {q["name"]: q for q in placement["world_named_points"]}
        self.node: dict[str, str] = {}
        seen: list[tuple[str, F64]] = []
        for q in placement["world_named_points"]:
            if not q["on_outline"]:
                continue
            w = np.asarray(q["world_m"], dtype=np.float64)
            hit = next((n for n, p in seen if float(np.linalg.norm(p - w)) < COINCIDENT_M), None)
            self.node[q["name"]] = hit or q["name"]
            if hit is None:
                seen.append((q["name"], w))
        self.edges: dict[frozenset[str], tuple[str, float, str, F64]] = {}
        for e in placement["world_edges"]:
            a, b = self.node.get(e["from"], e["from"]), self.node.get(e["to"], e["to"])
            if a != b:
                poly = np.asarray(e["polyline_m"], dtype=np.float64)
                self.edges[frozenset((a, b))] = (e["id"], float(e["length_m"]), a, poly)

    winding = 1

    def walk(self, names: list[str]) -> tuple[list[tuple[frozenset[str], bool]], str]:
        """(edge, walked from its start node) of a chain of named points, or an error text."""
        nodes: list[str] = []
        for n in names:
            if n not in self.named:
                return [], f"unknown point {n}"
            if not self.named[n]["on_outline"]:
                return [], f"{n} is not on the outline"
            c = self.node[n]
            if not nodes or nodes[-1] != c:
                nodes.append(c)
        out = []
        for a, b in zip(nodes[:-1], nodes[1:], strict=True):
            k = frozenset((a, b))
            if k not in self.edges:
                return [], f"{a} -> {b} is not an outline edge"
            out.append((k, self.edges[k][2] == a))
        return out, ""

    def point_at(self, walk: list[tuple[frozenset[str], bool]], t: float) -> F64:
        """World point at flat distance ``t`` along a walk (linear between the edges' arc-length samples)."""
        arcs, pts, c = [], [], 0.0
        for k, fwd in walk:
            _, ln, _, poly = self.edges[k]
            p = poly if fwd else poly[::-1]
            arcs.append(c + np.linspace(0.0, ln, len(p)))
            pts.append(p)
            c += ln
        a, p = np.concatenate(arcs), np.vstack(pts)
        return np.array([np.interp(t, a, p[:, d]) for d in range(3)])


def _crossings(pa: F64, pb: F64, window: int, min_across: float = 1e-3) -> int:
    """Twisted pairs (i, j), |i - j| <= window: segments pa_i-pb_i and pa_j-pb_j intersect in the plane of side a's
    step and the mean across direction (written independently of the engine's ``seam.crossings``)."""
    n = len(pa)
    count = 0
    across = pb - pa
    alen = np.linalg.norm(across, axis=1)
    for i in range(n):
        for j in range(i + 1, min(n, i + window + 1)):
            if alen[i] <= min_across or alen[j] <= min_across:
                continue
            step = pa[j] - pa[i]
            sl = float(np.linalg.norm(step))
            if sl <= 1e-9:
                continue
            u = step / sl
            m = 0.5 * (across[i] + across[j])
            w = m - float(m @ u) * u
            wl = float(np.linalg.norm(w))
            if wl <= 1e-6:
                continue
            w = w / wl
            p2 = np.array([float(across[i] @ u), float(across[i] @ w)])
            q1 = np.array([sl, 0.0])
            q2 = q1 + np.array([float(across[j] @ u), float(across[j] @ w)])
            d1, d2 = p2, q2 - q1
            den = d1[0] * d2[1] - d1[1] * d2[0]
            if abs(den) < 1e-15:
                continue
            t = (q1[0] * d2[1] - q1[1] * d2[0]) / den
            v = (q1[0] * d1[1] - q1[1] * d1[0]) / den
            if 1e-9 < t < 1 - 1e-9 and 1e-9 < v < 1 - 1e-9:
                count += 1
    return count


def _ease_limit(seam: dict[str, Any], defaults: dict[str, Any]) -> tuple[str, float, str]:
    e = dict(seam.get("ease") or {})
    mode = e.get("mode") or {"ease": "distribute", "gather": "gather"}.get(seam["kind"], "none")
    if mode == "gather":
        return mode, float(e.get("ratio", 0.0)), "long/short <= declared ratio"
    if mode == "distribute":
        return mode, 1.0 + DISTRIBUTE_MAX_PCT / 100.0, f"<= {DISTRIBUTE_MAX_PCT} %"
    tol = float(seam.get("tolerance_pct", defaults.get("max_length_diff_pct", 3.0)))
    return mode, 1.0 + tol / 100.0, f"<= {tol} %"


def _claim(claims: dict[Key, list[tuple[str, float, float]]], graph: Graph, piece: str,
           walk: list[tuple[frozenset[str], bool]], t0: float, t1: float, who: str) -> None:  # fmt: skip
    """Record that ``who`` sews [t0, t1] (flat distance along the walk), per edge in the edge's own direction."""
    c = 0.0
    for k, fwd in walk:
        ln = graph.edges[k][1]
        lo, hi = max(c, t0) - c, min(c + ln, t1) - c
        if hi > lo:
            claims.setdefault((piece, k), []).append((who, lo, hi) if fwd else (who, ln - hi, ln - lo))
        c += ln


def check(result: dict[str, Any], stitching: dict[str, Any], stitching_path: Path, assembly_path: Path,
          run_folder: Path | None = None) -> list[dict[str, Any]]:  # fmt: skip
    """All rows for ``result``; ``run_folder`` defaults to the one the result names."""
    rows: list[dict[str, Any]] = []
    try:
        validate("stitching_result", result)
        _row(rows, "input.schema", "pass")
    except ValueError as e:
        _row(rows, "input.schema", "fail", hint=str(e)[:400])
        return rows
    for key, path in (("stitching_sha256", stitching_path), ("assembly_sha256", assembly_path)):
        ok = result["inputs"][key] == sha256(Path(path).read_bytes())
        _row(rows, "input.stale", "pass" if ok else "fail", key, "", "" if ok else f"{path} changed since the run")
    run = load_run(run_folder or ROOT / result["inputs"]["run"])
    if sha256((run.folder / "result.json").read_bytes()) != result["inputs"]["alignment_result_sha256"]:
        _row(rows, "input.stale", "fail", "alignment_result_sha256", hint="the alignment result changed since the run")
    defaults = {"samples_per_cm": 2, "max_length_diff_pct": 3.0, "max_gap_m": 0.30, **stitching.get("defaults", {})}
    place = run.placements
    graphs = {n: Graph(p) for n, p in place.items()}
    for n, g in graphs.items():
        o = np.asarray(run.bundles[n]["outline_m"], dtype=np.float64)
        g.winding = 1 if float(np.sum(o[:, 0] * np.roll(o[:, 1], -1) - np.roll(o[:, 0], -1) * o[:, 1])) > 0 else -1
    for err in result["errors"]:
        _row(rows, "seam.resolve", "fail", seam=err["seam"], hint=err["error"])
    declared = {s["id"]: s for s in stitching["seams"]}
    got = {s["id"]: s for s in result["seams"]}
    claims: dict[Key, list[tuple[str, float, float]]] = {}
    for sid, s in got.items():
        spec = declared.get(sid)
        if spec is None:
            _row(rows, "seam.declared", "fail", sid, seam=sid, hint="in the result but not in the stitching file")
            continue
        _check_seam(rows, s, spec, graphs, place, defaults, claims, run.bundles)
    # one edge stitched twice (by two seams, or by both sides of one)
    for (piece, k), cl in sorted(claims.items(), key=lambda kv: (kv[0][0], sorted(kv[0][1]))):
        for i in range(len(cl)):
            for j in range(i + 1, len(cl)):
                ov = min(cl[i][2], cl[j][2]) - max(cl[i][1], cl[j][1])
                if ov > OVERLAP_M:
                    same = cl[i][0].split(".")[0] == cl[j][0].split(".")[0]
                    _row(rows, "seam.self_overlap" if same else "edges.claimed_twice", "fail",
                         f"{piece} {graphs[piece].edges[k][0]}", f"{OVERLAP_M * 100:.1f} cm",
                         f"{ov * 100:.1f} cm claimed by {cl[i][0]} and {cl[j][0]}",
                         seam=cl[i][0].split(".")[0])  # fmt: skip
    # required seams
    asm = read_json(assembly_path)
    req = {s["id"] for s in stitching["seams"] if s.get("required", True)}
    req |= set((stitching.get("expect") or {}).get("required_seams", []))
    req |= set((asm.get("expect") or {}).get("required_seams", []))
    for sid in sorted(req):
        present = sid in got
        _row(rows, "required", "pass" if present else "fail", sid, seam=sid,
             hint="" if present else "required seam missing from the result")  # fmt: skip
    # geometry untouched, real size
    now = geometry_hash(run.folder)
    same = now == result["pieces_untouched_hash"] and bool(result["untouched"])
    _row(rows, "geometry.untouched", "pass" if same else "fail", now["pieces"][:16],
         result["pieces_untouched_hash"]["pieces"][:16],
         "" if same else "piece or avatar geometry changed")  # fmt: skip
    size_rows = [r for r in (run.verify or {}).get("runs", {}).get("primary", {}).get("rows", [])
                 if r["id"].startswith(SIZE_ROWS)]  # fmt: skip
    bad = [f"{r['id']} {r['piece']}" for r in size_rows if r["status"] == "fail"]
    _row(rows, "geometry.scale", "pass" if size_rows and not bad else "fail", len(size_rows), "all size.* pass",
         "; ".join(bad[:5]) if bad else ("" if size_rows else "no size audit in verify.json"))  # fmt: skip
    for n, p in place.items():
        m = np.asarray(p["matrix"], dtype=np.float64).reshape(4, 4)[:3, :3]
        dev = float(np.abs(m.T @ m - np.eye(3)).max())
        if p.get("changes_piece_size") or dev > 1e-6:
            _row(rows, "geometry.scale", "fail", round(dev, 9), 1e-6, f"{n}: the placement is not rigid")
    # orphan edges (independent of the engine's list)
    ign: dict[Key, list[tuple[float, float]]] = {}
    for ig in stitching.get("ignore_edges", []):
        gi = graphs.get(ig["piece"])
        if gi is None:
            _row(rows, "ignore_edges", "fail", ig["piece"], hint="unknown piece")
            continue
        g = gi
        if "edge" in ig:
            hits = [k for k, e in g.edges.items() if e[0] == ig["edge"]]
            if not hits:
                _row(rows, "ignore_edges", "fail", f"{ig['piece']} {ig['edge']}", hint="no such edge")
            for k in hits:
                ign.setdefault((ig["piece"], k), []).append((0.0, g.edges[k][1]))
        else:
            walk, err = g.walk(ig["chain"])
            if err:
                _row(rows, "ignore_edges", "fail", ig["piece"], hint=err)
            for k, _ in walk:
                ign.setdefault((ig["piece"], k), []).append((0.0, g.edges[k][1]))
    orphans = []
    for n, g in sorted(graphs.items()):
        for k, (eid, ln, _, _) in sorted(g.edges.items(), key=lambda kv: kv[1][0]):
            if ln <= 1e-6:
                continue
            x = (np.arange(max(1, int(ln / 0.001))) + 0.5) * ln / max(1, int(ln / 0.001))
            hit = np.zeros(len(x), dtype=bool)
            for lo, hi in [(c[1], c[2]) for c in claims.get((n, k), [])] + ign.get((n, k), []):
                hit |= (x >= lo - 1e-9) & (x <= hi + 1e-9)
            free = ln * float((~hit).mean())
            if free > UNCOVERED_M:
                orphans.append((n, eid, free))
    for n, eid, free in orphans:
        _row(rows, "edges.orphan", "warn", f"{n} {eid}", "", f"{free * 100:.1f} cm neither sewn nor in ignore_edges")
    honest = len(orphans) == len(result["unstitched_edges"])
    _row(rows, "engine.honesty", "pass" if honest else "fail", len(result["unstitched_edges"]), len(orphans),
         "" if honest else "the engine's unstitched_edges differ from the independent count")  # fmt: skip
    return rows


def _check_seam(rows: list[dict[str, Any]], s: dict[str, Any], spec: dict[str, Any], graphs: dict[str, Graph],
                place: dict[str, Any], defaults: dict[str, Any],
                claims: dict[Key, list[tuple[str, float, float]]],
                bundles: dict[str, dict[str, Any]]) -> None:  # fmt: skip
    sid = s["id"]
    interiors: dict[str, list[tuple[str, list[tuple[frozenset[str], bool]]]]] = {}
    flat: dict[str, float] = {}
    ends: dict[str, tuple[F64, F64]] = {}
    onto = spec["b"].get("onto")
    for side in ("a", "b"):
        if side == "b" and onto is not None:
            continue
        raw = spec[side]["parts"] if "parts" in spec[side] else [spec[side]]
        want = []
        for w in raw:
            tr = (float(w.get("trim_start_m", 0.0)), float(w.get("trim_end_m", 0.0)))
            ch = w.get("chain") or w["line"]
            want.append((ch[::-1], tr[::-1]) if w.get("reverse") else (ch, tr))
        if "parts" in spec[side] and spec[side].get("reverse"):
            want = [(c[::-1], tr[::-1]) for c, tr in want[::-1]]
        parts = s[side]["parts"]
        if len(want) != len(parts):
            _row(rows, "seam.chain", "fail", f"{side}: {len(parts)} parts", len(want), seam=sid)
            return
        total = 0.0
        start = end = np.zeros(3)
        for k, ((listed, tr), p) in enumerate(zip(want, parts, strict=True)):
            if p["piece"] not in graphs:
                _row(rows, "seam.piece", "fail", p["piece"], seam=sid, hint=f"placed: {sorted(graphs)}")
                return
            g = graphs[p["piece"]]
            names = p["chain"]
            if p["direction"] == "line":  # interior line: flat length from the bundle, ends at the named points
                if names != listed:
                    _row(rows, "seam.chain", "fail", f"{side}{k}: {names}", listed, seam=sid)
                xy = {q["name"]: np.asarray(q["xy_m"]) for q in bundles[p["piece"]]["named_points"]}
                ln = float(sum(np.linalg.norm(xy[x] - xy[y]) for x, y in zip(names[:-1], names[1:], strict=True)))
                total += ln
                if abs(ln * 100 - p["length_cm"]) > LENGTH_CM + 1e-4 * p["length_cm"]:
                    _row(rows, "seam.length", "fail", p["length_cm"], round(ln * 100, 4), seam=sid)
                interiors.setdefault(side, []).append((p["piece"], []))
                if k == 0:
                    start = np.asarray(g.named[names[0]]["world_m"])
                if k == len(parts) - 1:
                    end = np.asarray(g.named[names[-1]]["world_m"])
                continue
            it = iter(names)
            if names[0] != listed[0] or names[-1] != listed[-1] or not all(any(x == y for y in it) for x in listed):
                _row(rows, "seam.chain", "fail", f"{side}{k}: {names}", listed, seam=sid,
                     hint="the resolved chain does not run through the declared points in order")  # fmt: skip
            walk, err = g.walk(names)
            if err:
                _row(rows, "seam.chain", "fail", f"{side}{k} {p['piece']}", "", err, seam=sid)
                return
            full = sum(g.edges[e][1] for e, _ in walk)
            ln = full - tr[0] - tr[1]
            total += ln
            if abs(ln * 100 - p["length_cm"]) > LENGTH_CM + 1e-4 * p["length_cm"]:
                _row(rows, "seam.length", "fail", p["length_cm"], round(ln * 100, 4), seam=sid,
                     hint=f"{side} part {k}: flat length differs from the alignment engine's edges")  # fmt: skip
            wl = p["world_length_cm"]
            if p["length_cm"] > 0.5 and abs(wl - p["length_cm"]) / p["length_cm"] * 100 > WORLD_LENGTH_PCT:
                _row(
                    rows,
                    "seam.world_length",
                    "fail",
                    wl,
                    p["length_cm"],
                    seam=sid,
                    hint=f"{side} part {k}: world polyline length off the flat length by > {WORLD_LENGTH_PCT} %",
                )
            _claim(claims, g, p["piece"], walk, tr[0], full - tr[1], f"{sid}.{side}")
            interiors.setdefault(side, []).append((p["piece"], walk))
            if k == 0:
                start = g.point_at(walk, tr[0])
            if k == len(parts) - 1:
                end = g.point_at(walk, full - tr[1])
        flat[side] = total
        ends[side] = (start, end)
    for side in ("a", "b"):
        poly = np.asarray(s[side]["polyline_m"], dtype=np.float64)
        bad_poly = (not np.isfinite(poly).all()) or bool(
            len(poly) > 1 and (np.linalg.norm(np.diff(poly, axis=0), axis=1) < 1e-9).any()
        )
        _row(rows, "seam.polyline", "fail" if bad_poly else "pass", side, seam=sid,
             hint="NaN or repeated points" if bad_poly else "")  # fmt: skip
    sm = s["samples"]
    sv = np.asarray(sm["s"])
    pa, pb = np.asarray(sm["a_xyz_m"], dtype=np.float64), np.asarray(sm["b_xyz_m"], dtype=np.float64)
    la = flat["a"]
    if onto is not None:
        lb = float(np.linalg.norm(np.diff(pb, axis=0), axis=1).sum())
        _check_onto(rows, sid, onto, place, pa, pb, sm)
        _row(rows, "seam.ratio", "info", round(la / lb, 4), "attach: informational", seam=sid)
    else:
        lb = flat["b"]
        ratio = la / lb if lb > 0 else math.inf
        big = max(ratio, 1 / ratio) if ratio > 0 else math.inf
        mode, limit, txt = _ease_limit(spec, defaults)
        ok = big <= limit + 1e-9
        _row(
            rows,
            "seam.ratio",
            "pass" if ok else "fail",
            round(ratio, 4),
            f"{mode} {txt} ({limit:.3f})",
            seam=sid,
            hint="" if ok else f"a {la * 100:.1f} cm, b {lb * 100:.1f} cm: fix the chains, or declare the ease",
        )
        if mode == "gather" and big > GATHER_WARN:
            _row(rows, "seam.gather", "warn", round(big, 3), GATHER_WARN, "large gather", seam=sid)
        if abs(ratio - s["ratio"]) > 1e-4:
            _row(rows, "engine.honesty", "fail", s["ratio"], round(ratio, 6), "ratio differs", seam=sid)
    longest = la if onto is not None else max(la, lb)
    n_want = max(2, int(math.ceil(float(defaults["samples_per_cm"]) * longest * 100.0 - 1e-6)) + 1)
    mono = bool(len(sv) >= 2 and sv[0] == 0.0 and sv[-1] == 1.0 and (np.diff(sv) > 0).all())
    for side in ("a", "b") if onto is None else ("a",):
        part = np.asarray(sm[f"{side}_part"])
        arc = np.asarray(sm[f"{side}_arc_m"])
        lens = [p["length_cm"] / 100 for p in s[side]["parts"]]
        off = np.concatenate([[0.0], np.cumsum(lens)])[part]
        mono &= bool((np.diff(part) >= 0).all() and (np.diff(off + arc) >= -1e-9).all())
    ok = mono and abs(len(sv) - n_want) <= 1 and len(pa) == len(pb) == len(sv)
    _row(rows, "seam.samples", "pass" if ok else "fail", len(sv), n_want, "" if ok else "not monotonic or wrong count",
         seam=sid)  # fmt: skip
    finite = bool(np.isfinite(pa).all() and np.isfinite(pb).all())
    d_end = 0.0
    for side, p0, p1 in (("a", pa[0], pa[-1]), ("b", pb[0], pb[-1])):
        if side == "b" and onto is not None:
            continue
        parts = s[side]["parts"]
        for q, target, k, which in ((p0, ends[side][0], 0, 0), (p1, ends[side][1], len(parts) - 1, 1)):
            if parts[k]["direction"] == "line":  # snapped onto the surface: the named point's own world position
                d_end = max(d_end, float(np.linalg.norm(q - target)))
            elif parts[k]["trim_m"][which] > 0:  # between named points: on the world outline
                edges = place[parts[k]["piece"]]["world_edges"]
                d_end = max(d_end, min(_to_polyline(q, np.asarray(e["polyline_m"])) for e in edges))
            else:
                d_end = max(d_end, float(np.linalg.norm(q - target)))
    ok = finite and d_end <= ENDPOINT_M
    _row(rows, "seam.endpoints", "pass" if ok else "fail", round(d_end, 6), ENDPOINT_M, seam=sid,
         hint="" if ok else "correspondence ends are not the chain ends")  # fmt: skip
    gap = np.linalg.norm(pb - pa, axis=1)
    far = float(gap.max()) > float(defaults["max_gap_m"])
    if far and not spec.get("allow_far"):
        _row(
            rows,
            "seam.gap",
            "fail",
            round(float(gap.max()), 4),
            defaults["max_gap_m"],
            seam=sid,
            hint="the pieces are far apart: check the pieces / chains, or set allow_far (the simulation closes it)",
        )
    else:
        _row(rows, "seam.gap", "info" if far else "pass", round(float(gap.max()), 4), defaults["max_gap_m"], seam=sid)
    lines = any(not w for parts in interiors.values() for _, w in parts)
    if onto is None and not lines:
        sides = {k: {_interior(graphs, p) for p in parts} for k, parts in interiors.items()}
        fold = spec["kind"] == "fold"
        expected = sides["b"] if fold else {-x for x in sides["b"]}
        ok = len(sides["a"]) == 1 and len(sides["b"]) == 1 and sides["a"] == expected
        _row(rows, "seam.sewing_rule", "pass" if ok else "fail", f"a {sorted(sides['a'])} b {sorted(sides['b'])}",
             "same side (fold)" if fold else "opposite sides",
             "" if ok else "the pieces lie on the wrong sides of the seam: set reverse: true on one side",
             seam=sid)  # fmt: skip
    w = max(1, len(pa) // 4)
    c = _crossings(pa, pb, w)
    if c and spec.get("allow_crossings"):
        _row(rows, "seam.orientation", "info", c, 0, f"allow_crossings: {spec['allow_crossings']}"[:200], seam=sid)
    elif c:
        rc = _crossings(pa, pb[::-1], w)
        hint = f"twisted; reversed pairing has {rc} crossings" + (" -> set reverse: true on one side" if rc < c else "")
        _row(rows, "seam.orientation", "fail", c, 0, hint, seam=sid)
    else:
        _row(rows, "seam.orientation", "pass", 0, 0, seam=sid)
    if c != s["orientation"]["crossing_count"]:
        _row(rows, "engine.honesty", "fail", s["orientation"]["crossing_count"], c, "crossing counts differ", seam=sid)
    tb = s.get("through_body", {}).get("segments", 0)
    if tb:
        _row(rows, "seam.through_body", "warn", tb, 0, f"up to {s['through_body']['max_depth_m'] * 100:.1f} cm deep",
             seam=sid)  # fmt: skip


def _interior(graphs: dict[str, Graph], part: tuple[str, list[tuple[frozenset[str], bool]]]) -> int:
    """+1 / -1: the piece lies left / right of a walked chain in its flat frame (0: the walk runs both ways).

    The walk direction comes from the alignment result's edge order (world_edges run along the outline), the winding
    from the shoelace area of the bundle's flat outline (``Graph.winding``)."""
    piece, walk = part
    fw = {f for _, f in walk}
    if len(fw) != 1:
        return 0
    return (1 if fw.pop() else -1) * graphs[piece].winding


def _to_polyline(p: F64, poly: F64) -> float:
    """Distance of ``p`` to an open polyline (an edge's arc-length samples)."""
    a, b = poly[:-1], poly[1:]
    d = b - a
    t = np.clip(((p - a) * d).sum(axis=1) / np.maximum((d * d).sum(axis=1), 1e-30), 0.0, 1.0)
    return float(np.linalg.norm(a + t[:, None] * d - p, axis=1).min())


def _check_onto(rows: list[dict[str, Any]], sid: str, host: str, place: dict[str, Any], pa: F64, pb: F64,
                sm: dict[str, Any]) -> None:  # fmt: skip
    """Attach: every b sample is the host surface's closest point to its a sample, and its triangle and barycentric
    weights rebuild it."""
    from assembler.verify import Surf

    if host not in place:
        _row(rows, "seam.piece", "fail", host, seam=sid, hint="unknown host piece")
        return
    v = np.asarray(place[host]["wrapped_vertices"], dtype=np.float64)
    f = np.asarray(place[host]["wrapped_faces"], dtype=np.int64)
    d_b, _, _ = Surf(v, f).closest(pa, near=1.0)
    off = float(np.abs(np.linalg.norm(pb - pa, axis=1) - d_b).max())
    tri, bary = np.asarray(sm["b_tri"]), np.asarray(sm["b_bary"])
    rebuilt = (v[f[tri]] * bary[:, :, None]).sum(axis=1)
    off = max(off, float(np.linalg.norm(rebuilt - pb, axis=1).max()))
    ok = off <= ONTO_M
    _row(rows, "seam.onto", "pass" if ok else "fail", round(off, 6), ONTO_M, seam=sid,
         hint="" if ok else "attach samples are not the host surface's closest points")  # fmt: skip


def markdown(result: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    """``stitch_verify.md``: verdict, the seam table, failures, warnings."""
    ok = verdict(rows)
    out = [f"# stitching verify: {result['garment']}", "", f"**overall: {'PASS' if ok else 'FAIL'}**", ""]
    sm = result["summary"]
    out += [
        f"seams {sm['seams']}/{sm['seams_declared']}, errors {sm['errors']}, crossings {sm['crossings']}, "
        f"unstitched edges {sm['unstitched_edges']} ({sm['unstitched_cm']} cm), ignored {sm['ignored_edges']}, "
        f"assembly verify pass {sm['assembly_pass']}, geometry untouched {result['untouched']}",
        "",
        "| seam | kind | ease | a cm | b cm | ratio | gap min/mean/max cm | crossings | status |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for s in result["seams"]:
        st = "FAIL" if any(r["status"] == "fail" and r["seam"] == s["id"] for r in rows) else "pass"
        g = s["world_gap_m"]
        e = s["ease"]["mode"] + (f" {s['ease']['declared_ratio']}" if s["ease"]["mode"] == "gather" else "")
        out.append(
            f"| {s['id']} | {s['kind']} | {e} | {s['length_a_cm']:.2f} | {s['length_b_cm']:.2f} | {s['ratio']:.4f} | "
            f"{g['min'] * 100:.1f}/{g['mean'] * 100:.1f}/{g['max'] * 100:.1f} | {s['orientation']['crossing_count']} "
            f"| {st} |"
        )
    fails = [r for r in rows if r["status"] == "fail"]
    out += ["", f"## failures ({len(fails)})", ""]
    out += [f"- {r['id']} {r['seam']}: {r['value']} (limit {r['limit']}) {r['hint']}" for r in fails] or ["- none"]
    warns = [r for r in rows if r["status"] == "warn"]
    out += ["", f"## warnings ({len(warns)})", ""]
    out += [f"- {r['id']} {r['seam']}: {r['value']} {r['hint']}" for r in warns] or ["- none"]
    out += ["", "## ignored edges", ""]
    out += [f"- {u['piece']} {u['edge']} ({u['length_cm']} cm): {u['reason']}" for u in result["ignored_edges"]] or [
        "- none"
    ]
    return "\n".join(out) + "\n"

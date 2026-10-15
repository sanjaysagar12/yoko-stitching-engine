"""One seam: both sides resolved to chains, true lengths, ease, point-to-point correspondences, gap, orientation.

A side is one chain ``{piece, chain, reverse}`` or several sewn one after the other ``{parts: [...], reverse}`` (an
armhole made of front, yoke and back). The engine sews ``a[0] -> a[-1]`` to ``b[0] -> b[-1]``; ``reverse`` flips a
side (it is never flipped silently: the orientation check reports whether the other pairing would be better).

Correspondences: ``N = samples_per_cm * max(length_a, length_b) + 1`` pairs at normalised arc length ``s_i`` (0..1,
both ends included) on each side's true (flat) arc length, so a length difference (ease, gathers) is spread evenly
along the whole seam.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .chains import F64, Line, Path, Piece, line, resolve
from .errors import StitchingError

DISTRIBUTE_MAX_PCT = 5.0  # `distribute`: at most 5 % (sleeve cap ease into an armhole is 0-5 %)
GATHER_WARN = 1.5  # a gather over 1.5 is reported (a large gather)
MIN_ACROSS_M = 1e-3  # pairs closer than 1 mm are coincident: no crossing can be judged from them
EASE_OF_KIND = {"ease": "distribute", "gather": "gather"}


def ease_of(seam: dict[str, Any]) -> dict[str, Any]:
    """The seam's ease ``{mode, ratio}`` (the kind's default when not given)."""
    e = dict(seam.get("ease") or {})
    e.setdefault("mode", EASE_OF_KIND.get(seam["kind"], "none"))
    e.setdefault("ratio", 1.0)
    return e


@dataclass
class Side:
    parts: list[Path | Line]

    @property
    def length(self) -> float:
        return float(sum(p.length for p in self.parts))

    @property
    def offsets(self) -> F64:
        return np.concatenate([[0.0], np.cumsum([p.length for p in self.parts])])

    def at(self, t: F64) -> tuple[F64, F64, NDArray[np.int64], F64]:
        """World points, flat points, part index and distance along that part at distances ``t`` along the side."""
        off = self.offsets
        k = np.clip(np.searchsorted(off, t, side="right") - 1, 0, len(self.parts) - 1)
        local = np.clip(t - off[k], 0.0, None)
        w = np.zeros((len(t), 3))
        f = np.zeros((len(t), 2))
        for i, p in enumerate(self.parts):
            m = k == i
            if m.any():
                w[m], f[m] = p.at(np.minimum(local[m], p.length))
        return w, f, k, local

    def dense(self) -> tuple[F64, list[F64]]:
        """World polyline of every knot of every part (the drawn sewing line) and each part's own polyline."""
        per_part = [p.at(p.knots_t())[0] for p in self.parts]
        allp = np.vstack(per_part)
        keep = np.r_[True, np.linalg.norm(np.diff(allp, axis=0), axis=1) > 1e-9]
        return allp[keep], per_part


def side_of(
    spec: dict[str, Any], pieces: dict[str, Piece], mesh: Callable[[str], tuple[F64, NDArray[np.int64]]] | None = None
) -> Side:
    raw = spec["parts"] if "parts" in spec else [spec]
    parts: list[Path | Line] = []
    for r in raw:
        if r["piece"] not in pieces:
            raise StitchingError(f"unknown piece {r['piece']!r}; placed pieces: {sorted(pieces)}")
        if "line" in r:
            snap = None
            if mesh is not None:
                v, f = mesh(r["piece"])
                snap = lambda w, v=v, f=f: locate(v, f, w)[0]  # noqa: E731
            ln = line(pieces[r["piece"]], r["line"], snap)
            parts.append(ln.reversed() if r.get("reverse") else ln)
            continue
        p = resolve(pieces[r["piece"]], r["chain"])
        if r.get("trim_start_m") or r.get("trim_end_m"):
            p = p.trimmed(float(r.get("trim_start_m", 0.0)), float(r.get("trim_end_m", 0.0)))
        parts.append(p.reversed() if r.get("reverse") else p)
    if "parts" in spec and spec.get("reverse"):
        parts = [p.reversed() for p in parts[::-1]]
    return Side(parts)


def crossings(pa: F64, pb: F64, window: int) -> int:
    """Pairs of sewing segments (``pa_i -> pb_i``) that cross, among pairs at most ``window`` samples apart.

    Each pair (i, j) is judged in its own plane: e1 along side a (``pa_j - pa_i``), e2 the mean across direction
    without its e1 part. The two segments cross when they intersect properly in that plane (a twisted quad). Pairs
    whose across vectors are shorter than 1 mm (coincident edges) are skipped."""
    n = len(pa)
    v = pb - pa
    vlen = np.linalg.norm(v, axis=1)
    total = 0
    for k in range(1, min(window, n - 1) + 1):
        i = np.arange(n - k)
        j = i + k
        e1 = pa[j] - pa[i]
        L = np.linalg.norm(e1, axis=1)
        good = (L > 1e-9) & (vlen[i] > MIN_ACROSS_M) & (vlen[j] > MIN_ACROSS_M)
        e1 = e1 / np.maximum(L, 1e-18)[:, None]
        across = 0.5 * (v[i] + v[j])
        e2 = across - (across * e1).sum(axis=1)[:, None] * e1
        e2n = np.linalg.norm(e2, axis=1)
        good &= e2n > 1e-6
        e2 = e2 / np.maximum(e2n, 1e-18)[:, None]
        # 2D: A_i = (0, 0), B_i, A_j = (L, 0), B_j
        bix, biy = (v[i] * e1).sum(axis=1), (v[i] * e2).sum(axis=1)
        bjx, bjy = L + (v[j] * e1).sum(axis=1), (v[j] * e2).sum(axis=1)

        def orient(px: F64, py: F64, qx: F64, qy: F64, rx: F64, ry: F64) -> F64:
            return np.asarray((qx - px) * (ry - py) - (qy - py) * (rx - px))

        z = np.zeros_like(L)
        o1 = orient(z, z, bix, biy, L, z)
        o2 = orient(z, z, bix, biy, bjx, bjy)
        o3 = orient(L, z, bjx, bjy, z, z)
        o4 = orient(L, z, bjx, bjy, bix, biy)
        tol = 1e-12
        cross = (o1 * o2 < -tol) & (o3 * o4 < -tol)
        total += int((cross & good).sum())
    return total


def window_of(n: int) -> int:
    return max(1, n // 4)


@dataclass
class Computed:
    """The numbers of one seam (``record`` is what goes into the result file)."""

    record: dict[str, Any]
    a: Side
    b: Side | None  # None: attached onto a host surface
    pa: F64
    pb: F64


def _side_record(side: Side, uv: F64, part: NDArray[np.int64], local: F64) -> dict[str, Any]:
    dense, per_part = side.dense()
    return {
        "piece": side.parts[0].piece.name,
        "edge_ids": [e for p in side.parts for e in p.edge_ids],
        "polyline_m": _r(dense),
        "parts": [
            {
                "piece": p.piece.name,
                "chain": p.names,
                "direction": "line" if isinstance(p, Line) else ("outline" if p.direction > 0 else "against_outline"),
                "trim_m": [round(p.trim[0], 6), round(p.trim[1], 6)],
                "edge_ids": p.edge_ids,
                "length_cm": round(p.length * 100.0, 4),
                "world_length_cm": round(_poly_len(pp) * 100.0, 4),
                "polyline_m": _r(pp),
            }
            for p, pp in zip(side.parts, per_part, strict=True)
        ],
    }


def _onto_record(host: str, pb: F64) -> dict[str, Any]:
    keep = np.r_[True, np.linalg.norm(np.diff(pb, axis=0), axis=1) > 1e-9]
    poly = pb[keep]
    part = {
        "piece": host, "chain": [], "direction": "onto", "trim_m": [0.0, 0.0], "edge_ids": [],
        "length_cm": round(_poly_len(poly) * 100.0, 4), "world_length_cm": round(_poly_len(poly) * 100.0, 4),
        "polyline_m": _r(poly),
    }  # fmt: skip
    return {"piece": host, "edge_ids": [], "polyline_m": _r(poly), "parts": [part]}


def _poly_len(p: F64) -> float:
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()) if len(p) > 1 else 0.0


def _r(x: Any, nd: int = 6) -> Any:
    return np.round(np.asarray(x, dtype=np.float64), nd).tolist()


def locate(v: F64, f: NDArray[np.int64], p: F64, k: int = 32) -> tuple[F64, NDArray[np.int64], F64]:
    """Closest point of a triangle mesh to each of ``p``: (point, triangle index into ``f``, barycentric)."""
    from scipy.spatial import cKDTree

    from assembler.verify import _closest_on_tris

    tri = v[f]
    _, cand = cKDTree(tri.mean(axis=1)).query(p, k=min(k, len(f)))
    cand = np.asarray(cand).reshape(len(p), -1)
    rows = np.repeat(np.arange(len(p)), cand.shape[1])
    t = tri[cand.ravel()]
    q = _closest_on_tris(p[rows], t[:, 0], t[:, 1], t[:, 2])
    d = np.linalg.norm(q - p[rows], axis=1).reshape(len(p), -1)
    best = d.argmin(axis=1)
    idx = cand[np.arange(len(p)), best]
    q = q.reshape(len(p), -1, 3)[np.arange(len(p)), best]
    a, b, c = tri[idx, 0], tri[idx, 1], tri[idx, 2]
    v0, v1, v2 = b - a, c - a, q - a
    d00, d01, d11 = (v0 * v0).sum(1), (v0 * v1).sum(1), (v1 * v1).sum(1)
    d20, d21 = (v2 * v0).sum(1), (v2 * v1).sum(1)
    den = np.maximum(d00 * d11 - d01 * d01, 1e-30)
    w1 = (d11 * d20 - d01 * d21) / den
    w2 = (d00 * d21 - d01 * d20) / den
    return q, idx, np.column_stack([1 - w1 - w2, w1, w2])


def compute(
    seam: dict[str, Any],
    pieces: dict[str, Piece],
    defaults: dict[str, Any],
    mesh: Callable[[str], tuple[F64, NDArray[np.int64]]] | None = None,
) -> Computed:
    """One seam. ``mesh(name)`` gives a placed piece's world mesh (for ``attach`` seams: side b ``{onto: host}`` is
    the closest point of the host's surface to every sample of side a, with its triangle and barycentric weights)."""
    a = side_of(seam["a"], pieces, mesh)
    onto = seam["b"].get("onto")
    la = a.length
    spc = float(defaults.get("samples_per_cm", 2))
    if onto is not None:
        if onto not in pieces or mesh is None:
            raise StitchingError(f"seam {seam['id']}: unknown host piece {onto!r}; placed pieces: {sorted(pieces)}")
        n = max(2, int(math.ceil(spc * la * 100.0)) + 1)
        s = np.linspace(0.0, 1.0, n)
        pa, ua, ka, ta = a.at(s * la)
        pb, tri, bary = locate(*mesh(onto), pa)
        tb = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pb, axis=0), axis=1))])
        lb = float(tb[-1])
        b = None
        ub, kb = np.full((n, 2), np.nan), np.zeros(n, dtype=np.int64)
    else:
        b = side_of(seam["b"], pieces, mesh)
        lb = b.length
    if la <= 0 or lb <= 0:
        raise StitchingError(f"seam {seam['id']}: a side has zero length (a {la * 100:.2f} cm, b {lb * 100:.2f} cm)")
    if b is not None:
        n = max(2, int(math.ceil(spc * max(la, lb) * 100.0)) + 1)
        s = np.linspace(0.0, 1.0, n)
        pa, ua, ka, ta = a.at(s * la)
        pb, ub, kb, tb = b.at(s * lb)
    gap = np.linalg.norm(pb - pa, axis=1)
    w = window_of(n)
    cross = crossings(pa, pb, w)
    rev_cross = crossings(pa, pb[::-1], w)
    dist, rev_dist = float(gap.sum()), float(np.linalg.norm(pb[::-1] - pa, axis=1).sum())
    e = ease_of(seam)
    ratio = la / lb
    big = max(ratio, 1.0 / ratio)
    free = bool(seam.get("allow_crossings"))  # the layout crosses for a stated reason: crossings reported only
    # sewing rule: a join keeps one surface, so the two pieces lie on opposite sides of the seam line
    rule = None
    lines = any(isinstance(p, Line) for p in a.parts + (b.parts if b is not None else []))
    if b is not None and not lines:
        ia = sorted({p.interior for p in a.parts})
        ib = sorted({p.interior for p in b.parts})
        # fold (a facing turned in, a bag sewn round its edge, a placket folded over): both on the same side
        rule = len(ia) == 1 and len(ib) == 1 and ia[0] == (ib[0] if seam["kind"] == "fold" else -ib[0])
    hint = ""
    if rule is False:
        hint = "probably reverse: true (the pieces would lie on the wrong sides of the seam: the sewing rule)"
    elif not free and (rev_cross < cross or (rev_cross == cross and rev_dist < 0.8 * dist)):
        hint = (
            f"probably reverse: true (reversed pairing: {rev_cross} crossings, {rev_dist / n * 100:.1f} cm mean gap; "
            f"as given: {cross}, {dist / n * 100:.1f} cm)"
        )
    tol = float(seam.get("tolerance_pct", defaults.get("max_length_diff_pct", 3.0)))
    rec: dict[str, Any] = {
        "id": seam["id"],
        "label": seam.get("label", ""),
        "kind": seam["kind"],
        "required": bool(seam.get("required", True)),
        "allow_far": bool(seam.get("allow_far", False)),
        "a": _side_record(a, ua, ka, ta),
        "b": _side_record(b, ub, kb, tb) if b is not None else _onto_record(onto, pb),
        "length_a_cm": round(la * 100.0, 4),
        "length_b_cm": round(lb * 100.0, 4),
        "world_length_a_cm": round(_poly_len(a.dense()[0]) * 100.0, 4),
        "world_length_b_cm": round((_poly_len(b.dense()[0]) if b is not None else lb) * 100.0, 4),
        "ratio": round(ratio, 6),
        "ease": {
            "mode": e["mode"],
            "declared_ratio": float(e["ratio"]),
            "tolerance_pct": tol,
            "applied_ratio": round(big, 6),
            "gathered_side": "none" if abs(ratio - 1.0) < 1e-9 else ("a" if la > lb else "b"),
            "excess_cm": round(abs(la - lb) * 100.0, 4),
        },
        "samples": {
            "count": n,
            "s": _r(s),
            "a_xyz_m": _r(pa),
            "b_xyz_m": _r(pb),
            "a_uv_m": _r(ua),
            "b_uv_m": None if b is None else _r(ub),
            "a_part": ka.tolist(),
            "b_part": kb.tolist(),
            "a_arc_m": _r(ta),
            "b_arc_m": _r(tb),
        },
        "world_gap_m": {
            "min": round(float(gap.min()), 6),
            "mean": round(float(gap.mean()), 6),
            "max": round(float(gap.max()), 6),
        },
        "orientation": {
            "ok": rule is not False and (cross == 0 or free),
            "sewing_rule_ok": rule,
            "interior": {
                "a": [p.interior for p in a.parts],
                "b": [p.interior for p in b.parts] if b is not None else [],
            },
            "crossings_judged": not free,
            "allow_crossings": seam.get("allow_crossings", ""),
            "crossing_count": cross,
            "reversed_crossing_count": rev_cross,
            "window": w,
            "mean_gap_cm": round(dist / n * 100.0, 3),
            "reversed_mean_gap_cm": round(rev_dist / n * 100.0, 3),
            "hint": hint,
        },
        "warnings": [],
    }
    if b is None:
        rec["samples"]["b_tri"] = tri.tolist()
        rec["samples"]["b_bary"] = _r(bary)
        rec["ease"]["mode"] = "none"
        rec["ease"]["note"] = "attach: side b is the host surface under side a; the ratio is informational"
    if e["mode"] == "gather" and big > GATHER_WARN:
        rec["warnings"].append(f"large gather {big:.2f} (> {GATHER_WARN})")
    if hint:
        rec["warnings"].append(hint)
    return Computed(rec, a, b, pa, pb)

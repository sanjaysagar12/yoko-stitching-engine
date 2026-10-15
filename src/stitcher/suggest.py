"""``python -m stitcher suggest``: seam candidates for a hand-written stitching file. A suggestion only.

Sources, strongest first:

1. mirror / clone twins (the alignment result's ``derived_pieces[].suggested_stitch``): ``join_mirror`` along the
   mirror edge;
2. chain matching: every contiguous run of up to ``MAX_EDGES`` outline edges of one piece against every run of
   another (or the same) piece; kept when the flat lengths agree within ``RATIO_TOL`` and scored by the length ratio,
   the world distance of the two runs (ends paired the nearer way) and shared Seamly point labels at the ends;
3. the alignment engine's ``seam_candidates`` (edges running side by side), as a hint column.

A greedy pass takes the best-scored candidates whose edges are still free. The output names its confidence (0..1);
edges no candidate covers are listed. Gathered seams (waistbands, cuffs) and composite sides (armholes) are not
guessed: they are left for the hand-written file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .chains import Piece, resolve
from .errors import StitchingError
from .load import ROOT, Run, read_json, rel
from .result import pieces_of

MAX_EDGES = 6
RATIO_TOL = 0.03
MIN_LEN_M = 0.03
MAX_GAP_M = 0.35


def _runs(p: Piece) -> list[tuple[list[str], float]]:
    """Contiguous runs of outline edges (as named-point lists in outline order) with their flat length."""
    e = [x for x in p.edges() if x[3] > 1e-6]
    out = []
    n = len(e)
    for i in range(n):
        names, length = [e[i][1]], 0.0
        for k in range(min(MAX_EDGES, n - 1)):
            _, _fr, to, ln = e[(i + k) % n]
            names.append(to)
            length += ln
            if length >= MIN_LEN_M:
                out.append((list(names), length))
    return out


def _shorter(p: Piece, chain: list[str]) -> Any:
    """The chain between two points, the shorter way round (a mirror edge may pass several named points)."""
    i, j = p.cycle.index(chain[0]), p.cycle.index(chain[1])
    n = len(p.cycle)
    fwd = [p.cycle[(i + k) % n] for k in range((j - i) % n + 1)]
    back = [p.cycle[(i - k) % n] for k in range((i - j) % n + 1)]
    walks = sorted((resolve(p, w) for w in (fwd, back) if len(w) >= 2), key=lambda x: x.length)
    if not walks:
        raise StitchingError(f"{p.name}: no chain {chain}")
    return walks[0]


def _ends(p: Piece, names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    w = p.at(np.array([p.s_of[names[0]], p.s_of[names[-1]]]))[0]
    return w[0], w[1]


def _label(n: str) -> str:
    return n.rstrip("abcdefghijklmnopqrstuvwxyz")


def suggest(run: Run, assembly: Path) -> dict[str, Any]:
    pieces = pieces_of(run)
    asm = read_json(assembly)
    cands: list[dict[str, Any]] = []
    # 1. mirror / clone twins
    for d in run.result.get("derived_pieces", []):
        ss = d.get("suggested_stitch")
        if not ss or ss["a"]["piece"] not in pieces or ss["b"]["piece"] not in pieces:
            continue
        try:
            chain = [ss["a"]["from_point"], ss["a"]["to_point"]]
            pa, pb = _shorter(pieces[ss["a"]["piece"]], chain), _shorter(pieces[ss["b"]["piece"]], chain)
        except StitchingError:
            continue
        cands.append(
            {
                "kind": "join_mirror",
                "a": {"piece": ss["a"]["piece"], "chain": pa.names},
                "b": {"piece": ss["b"]["piece"], "chain": pb.names},
                "length_a_cm": round(pa.length * 100, 2),
                "length_b_cm": round(pb.length * 100, 2),
                "ratio": round(pa.length / pb.length, 4),
                "confidence": 1.0,
                "why": f"{d['op']} of {d['source']} across {chain}",
                "edges": [(ss["a"]["piece"], e) for e in pa.edge_ids] + [(ss["b"]["piece"], e) for e in pb.edge_ids],
            }
        )
    # 2. chain matching (twins are only joined along their mirror edge, above)
    twins = {frozenset((d["id"], d["source"])) for d in run.result.get("derived_pieces", [])}
    runs = {n: _runs(p) for n, p in sorted(pieces.items())}
    names = sorted(pieces)
    for ia, na in enumerate(names):
        for nb in names[ia:]:
            if frozenset((na, nb)) in twins:
                continue
            for ca, la in runs[na]:
                a0, a1 = _ends(pieces[na], ca)
                for cb, lb in runs[nb]:
                    r = la / lb
                    if abs(r - 1.0) > RATIO_TOL:
                        continue
                    if na == nb and len({ca[0], ca[-1]} & {cb[0], cb[-1]}) != 1:
                        continue  # a dart's legs share exactly their apex
                    if na == nb and {frozenset(e) for e in zip(ca[:-1], ca[1:], strict=True)} & {
                        frozenset(e) for e in zip(cb[:-1], cb[1:], strict=True)
                    }:
                        continue  # a piece sewn to itself only over different edges (darts, underarm seams)
                    b0, b1 = _ends(pieces[nb], cb)
                    fwd = 0.5 * (np.linalg.norm(a0 - b0) + np.linalg.norm(a1 - b1))
                    rev = 0.5 * (np.linalg.norm(a0 - b1) + np.linalg.norm(a1 - b0))
                    gap = float(min(fwd, rev))
                    if gap > MAX_GAP_M:
                        continue
                    shared = len({_label(ca[0]), _label(ca[-1])} & {_label(cb[0]), _label(cb[-1])})
                    conf = (1.0 - abs(r - 1.0) / RATIO_TOL) * 0.4 + (1.0 - gap / MAX_GAP_M) * 0.45 + 0.075 * shared
                    chain_b = cb if fwd <= rev else cb[::-1]
                    cands.append(
                        {
                            "kind": "dart" if na == nb else "plain",
                            "a": {"piece": na, "chain": ca},
                            "b": {"piece": nb, "chain": chain_b},
                            "length_a_cm": round(la * 100, 2),
                            "length_b_cm": round(lb * 100, 2),
                            "ratio": round(r, 4),
                            "confidence": round(float(conf), 3),
                            "why": f"lengths {la * 100:.1f}/{lb * 100:.1f} cm, ends {gap * 100:.1f} cm apart"
                            + (f", {shared} shared end label(s)" if shared else ""),
                            "edges": [(na, pieces[na].edge_id(x, y)) for x, y in zip(ca[:-1], ca[1:], strict=True)]
                            + [(nb, pieces[nb].edge_id(x, y)) for x, y in zip(cb[:-1], cb[1:], strict=True)],
                        }
                    )
    cands.sort(key=lambda c: (-c["confidence"], -c["length_a_cm"], c["a"]["piece"], c["b"]["piece"]))
    taken: set[tuple[str, str]] = set()
    chosen = []
    for c in cands:
        if any(e in taken for e in c["edges"]):
            continue
        taken |= set(c["edges"])
        chosen.append(c)
    hints = [
        {
            "a": f"{c['a']['piece']} {c['a']['edge']}",
            "b": f"{c['b']['piece']} {c['b']['edge']}",
            "distance_cm": round(c["distance_m"] * 100, 1),
            "length_ratio": c["length_ratio"],
        }
        for c in run.result["arrangement"].get("seam_candidates", [])
    ]
    seams = []
    for i, c in enumerate(chosen):
        seams.append(
            {
                "id": f"s{i + 1:02d}_{c['a']['piece']}_{c['b']['piece']}".replace(" ", "_")[:60],
                "label": f"SUGGESTED (confidence {c['confidence']}): {c['why']}",
                "kind": c["kind"],
                "a": c["a"],
                "b": c["b"],
            }
        )
    table: dict[str, list[dict[str, Any]]] = {}
    lines = []
    for n in sorted(pieces):
        rows = [{"edge": e, "from": f, "to": t, "length_cm": round(ln * 100, 2)} for e, f, t, ln in pieces[n].edges()]
        table[n] = rows
        lines.append(f"{n}: " + " ".join(f"{r['edge']}:{r['length_cm']}" for r in rows))
    unmatched = [
        f"{n} {r['edge']} ({r['length_cm']} cm)"
        for n, rows in table.items()
        for r in rows
        if r["length_cm"] > 0 and (n, r["edge"]) not in taken
    ]
    stitching = {
        "schema_version": "1.0",
        "garment": asm["garment"],
        "assembly": rel(assembly),
        "notes": "SUGGESTED by `python -m stitcher suggest`: review every seam before use",
        "defaults": {"samples_per_cm": 2, "max_length_diff_pct": 3, "max_gap_m": 0.3},
        "seams": seams,
    }
    return {
        "suggested_from": rel(run.folder),
        "stitching": stitching,
        "candidates": [{k: v for k, v in c.items() if k != "edges"} for c in chosen],
        "alignment_seam_candidates": hints,
        "unmatched_edges": unmatched,
        "edge_table": table,
        "table_text": "\n".join(lines),
    }


def write_suggestion(doc: dict[str, Any], out: Path | None, stitching_path: Path | None) -> Path:
    """``<garment>.suggested.json`` in ``out`` (default build/stitching); never overwrites the hand-edited file."""
    garment = doc["stitching"]["garment"]
    folder = out or ROOT / "build" / "stitching"
    path = Path(folder) / f"{garment}.suggested.json"
    if stitching_path is not None and path.resolve() == Path(stitching_path).resolve():
        raise StitchingError(f"refusing to overwrite the hand-edited {stitching_path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    body = {k: v for k, v in doc.items() if k != "table_text"}
    path.write_text(json.dumps(body, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    return path

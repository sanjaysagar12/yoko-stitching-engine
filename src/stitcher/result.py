"""Stitch a garment: every seam of the stitching JSON on the placed pieces of an assembly run ->
``stitching.result.json`` (contracts/stitching_result.schema.json), the input of the later simulation phase.

Nothing is moved: the run folder is only read, and the geometry hash of the pieces and the avatar is taken before and
after (``untouched``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from assembler.verify import Surf

from .chains import Piece, build_piece, resolve
from .errors import StitchingError
from .load import Run, geometry_hash, rel, sha256
from .seam import Computed, compute, ease_of

SCHEMA_VERSION = "1.0"
OVERLAP_M = 1e-3  # two seams sharing more than 1 mm of one outline claim the same edge segment
UNCOVERED_M = 0.005  # an edge with at most 5 mm not sewn (rounding at chain ends) counts as sewn
BODY_DEPTH_M = 0.01  # a sewing segment more than 1 cm inside the body passes through it (reported, not failed)


def pieces_of(run: Run) -> dict[str, Piece]:
    return {name: build_piece(p, run.bundles[name]) for name, p in run.placements.items()}


def through_body(c: Computed, body: Surf) -> dict[str, Any]:
    """How many sewing segments run more than ``BODY_DEPTH_M`` inside the avatar (exact signed distance of the
    segment's quarter points, ``assembler.verify.Surf``)."""
    t = np.array([0.25, 0.5, 0.75])
    q = (c.pa[:, None, :] + t[None, :, None] * (c.pb - c.pa)[:, None, :]).reshape(-1, 3)
    d, _ = body.signed(q)
    inside = (d < -BODY_DEPTH_M).reshape(len(c.pa), 3).any(axis=1)
    return {"segments": int(inside.sum()), "max_depth_m": round(float(max(-d.min(), 0.0)), 4)}


def _intervals(c: Computed) -> list[tuple[str, float, float, str]]:
    return [
        (p.piece.name, *iv, side)
        for side, s in (("a", c.a), ("b", c.b))
        if s
        for p in s.parts
        if (iv := p.interval()) is not None
    ]


def _overlap(a: tuple[float, float], b: tuple[float, float], per: float) -> float:
    """Length shared by two arc intervals (start, length) on a cycle of perimeter ``per``."""
    best = 0.0
    for shift in (-per, 0.0, per):
        lo = max(a[0], b[0] + shift)
        hi = min(a[0] + a[1], b[0] + shift + b[1])
        best += max(0.0, hi - lo)
    return best


def stitch(
    stitching: dict[str, Any],
    stitching_path: Path,
    assembly_path: Path,
    run: Run,
) -> dict[str, Any]:
    """The stitching result of ``stitching`` on ``run``."""
    before = geometry_hash(run.folder)
    defaults = {"samples_per_cm": 2, "max_length_diff_pct": 3.0, "max_gap_m": 0.30, **stitching.get("defaults", {})}
    pieces = pieces_of(run)
    errors: list[dict[str, str]] = []
    done: list[Computed] = []
    place = run.placements

    def mesh(name: str) -> tuple[Any, Any]:
        p = place[name]
        return np.asarray(p["wrapped_vertices"], dtype=np.float64), np.asarray(p["wrapped_faces"], dtype=np.int64)

    for s in stitching["seams"]:
        try:
            done.append(compute(s, pieces, defaults, mesh))
        except StitchingError as e:
            errors.append({"seam": s["id"], "error": str(e)})
    body = Surf(*run.avatar())
    for c in done:
        c.record["through_body"] = tb = through_body(c, body)
        if tb["segments"]:
            c.record["warnings"].append(
                f"{tb['segments']} sewing segment(s) pass through the body "
                f"(up to {tb['max_depth_m'] * 100:.1f} cm deep)"
            )
    # claimed outline: overlaps and coverage
    claims: dict[str, list[tuple[float, float, str]]] = {}
    for c in done:
        for name, start, length, side in _intervals(c):
            claims.setdefault(name, []).append((start, length, f"{c.record['id']}.{side}"))
    overlaps = []
    for name, cl in sorted(claims.items()):
        per = pieces[name].per
        for i in range(len(cl)):
            for j in range(i + 1, len(cl)):
                ov = _overlap(cl[i][:2], cl[j][:2], per)
                if ov > OVERLAP_M:
                    overlaps.append({"piece": name, "claims": [cl[i][2], cl[j][2]], "overlap_cm": round(ov * 100, 3)})
    ignored: list[tuple[str, float, float, str]] = []
    for ig in stitching.get("ignore_edges", []):
        try:
            p = pieces.get(ig["piece"])
            if p is None:
                raise StitchingError(f"ignore_edges: unknown piece {ig['piece']!r}; placed pieces: {sorted(pieces)}")
            if "chain" in ig:
                path = resolve(p, ig["chain"])
            else:
                hits = [e for e in p.edges() if e[0] == ig["edge"]]
                if not hits:
                    raise StitchingError(
                        f"ignore_edges: {p.name} has no edge {ig['edge']!r}; edges: {[e[0] for e in p.edges()]}"
                    )
                path = resolve(p, [hits[0][1], hits[0][2]])
            ignored.append((p.name, *path.interval(), ig.get("reason", "")))
        except StitchingError as e:
            errors.append({"seam": "ignore_edges", "error": str(e)})

    def covered(p: Piece, start: float, ln: float, ivs: list[tuple[float, float]]) -> NDArray[np.bool_]:
        """Which 1 mm steps of the outline stretch [start, start + ln] lie in the intervals."""
        x = start + (np.arange(max(1, int(ln / 0.001))) + 0.5) * ln / max(1, int(ln / 0.001))
        hit = np.zeros(len(x), dtype=bool)
        for a, la in ivs:
            hit |= np.mod(x - a, p.per) <= la + 1e-9
        return hit

    unstitched: list[dict[str, Any]] = []
    ignored_out: list[dict[str, Any]] = []
    zero: list[dict[str, Any]] = []
    for name in sorted(pieces):
        p = pieces[name]
        seam_iv = [(a, ln) for a, ln, _ in claims.get(name, [])]
        ign_iv = [(a, ln, r) for n, a, ln, r in ignored if n == name]
        for eid, fr, to, ln in p.edges():
            row = {"piece": name, "edge": eid, "from": fr, "to": to, "length_cm": round(ln * 100.0, 3)}
            if ln < 1e-6:
                zero.append(row)
                continue
            sewn = covered(p, p.s_of[fr], ln, seam_iv)
            free = ln * float((~sewn).mean())
            if free <= UNCOVERED_M:
                continue
            ign = [r for a, la, r in ign_iv if (covered(p, p.s_of[fr], ln, [(a, la)]) & ~sewn).any()]
            rest = ~sewn & ~covered(p, p.s_of[fr], ln, [(a, la) for a, la, _ in ign_iv])
            if ign and ln * float(rest.mean()) <= UNCOVERED_M:
                ignored_out.append({**row, "reason": ign[0]})
            else:
                unstitched.append({**row, "uncovered_cm": round(ln * float(rest.mean()) * 100.0, 3)})
    after = geometry_hash(run.folder)
    seams = [c.record for c in done]
    verdict = run.verify or {}
    summary = {
        "seams": len(seams),
        "seams_declared": len(stitching["seams"]),
        "errors": len(errors),
        "crossings": int(
            sum(s["orientation"]["crossing_count"] for s in seams if s["orientation"]["crossings_judged"])
        ),
        "crossings_not_judged": int(
            sum(s["orientation"]["crossing_count"] for s in seams if not s["orientation"]["crossings_judged"])
        ),
        "sewing_rule_failures": sum(1 for s in seams if s["orientation"]["sewing_rule_ok"] is False),
        "unstitched_edges": len(unstitched),
        "unstitched_cm": round(sum(u["length_cm"] for u in unstitched), 2),
        "ignored_edges": len(ignored_out),
        "overlaps": len(overlaps),
        "max_gap_m": round(max((s["world_gap_m"]["max"] for s in seams), default=0.0), 6),
        "assembly_pass": bool(verdict.get("pass", False)),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "garment": stitching["garment"],
        "units": "m",
        "convention": (
            "world: metres, y up, +z front, +x the wearer's left (the alignment result's frame); uv: the piece "
            "bundle's outline_m frame (the <piece>_m.obj frame); lengths *_cm are flat (true) piece lengths"
        ),
        "inputs": {
            "stitching": rel(stitching_path),
            "stitching_sha256": sha256(Path(stitching_path).read_bytes()),
            "assembly": rel(assembly_path),
            "assembly_sha256": sha256(Path(assembly_path).read_bytes()),
            "run": rel(run.folder),
            "alignment_result_sha256": sha256((run.folder / "result.json").read_bytes()),
            "synthetic": bool((run.verify or {}).get("inputs", {}).get("synthetic", False)),
        },
        "defaults": defaults,
        "seams": seams,
        "errors": errors,
        "overlaps": overlaps,
        "unstitched_edges": unstitched,
        "ignored_edges": ignored_out,
        "zero_length_edges": zero,
        "pieces_untouched_hash": after,
        "untouched": before == after,
        "required_seams": sorted(
            {s["id"] for s in stitching["seams"] if s.get("required", True)}
            | set((stitching.get("expect") or {}).get("required_seams", []))
        ),
        "ease_modes": {s["id"]: ease_of(s)["mode"] for s in stitching["seams"]},
        "summary": summary,
    }


def write(doc: dict[str, Any], out: Path) -> Path:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    p = out / "stitching.result.json"
    p.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    return p

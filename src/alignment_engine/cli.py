"""Command line: ``alignment-engine run --avatar DIR --pieces DIR --spec spec.json --out DIR``."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .contracts import ContractError
from .engine import AlignmentEngine, AlignmentResult
from .errors import AlignmentError


def format_summary(r: AlignmentResult) -> str:
    """One line per placement plus stitch and ring reports."""
    lines = [f"{'piece':22s}{'point':22s}{'mode':8s}{'w cm':>8s}{'h cm':>8s}  warnings"]
    for p in r.doc["placements"]:
        mode = "around" if p["wrapped_vertices"] is not None else "flat"
        lines.append(
            f"{p['piece']:22s}{p['point']:22s}{mode:8s}{p['bbox_cm']['width']:8.2f}{p['bbox_cm']['height']:8.2f}  "
            + "; ".join(p["warnings"])
        )
    for s in r.doc["stitch_reports"]:
        lines.append(
            f"stitch {s['a']['piece']} {s['a']['from_point']}-{s['a']['to_point']} vs {s['b']['piece']} "
            f"{s['b']['from_point']}-{s['b']['to_point']}: {s['a_length_cm']:.2f} / {s['b_length_cm']:.2f} cm, "
            f"diff {s['diff_cm']:+.2f} cm ({s['diff_pct']:+.1f} %)"
        )
    for g in r.doc["ring_reports"]:
        lines.append(
            f"ring {g['name']}: pieces {g['sum_width_cm']:.1f} cm vs body {g['avatar_cm']:.1f} cm, "
            f"ease {g['ease_cm']:+.1f} cm"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Entry point. Exit codes: 0 ok, 2 fatal."""
    ap = argparse.ArgumentParser(prog="alignment-engine", description="place flat pattern pieces on avatar points")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="solve a placement spec")
    r.add_argument("--avatar", required=True, help="folder written by avatar-engine")
    r.add_argument("--pieces", required=True, help="folder written by sm2d-engine (OBJ + .piece.json)")
    r.add_argument("--spec", required=True, help="placement_spec JSON")
    r.add_argument("--out", required=True, help="output folder")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING)
    try:
        text = Path(args.spec).read_text(encoding="utf-8-sig")
        res = AlignmentEngine().align(Path(args.avatar), Path(args.pieces), text, Path(args.out))
    except (AlignmentError, ContractError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(format_summary(res))
    print(f"wrote {len(res.paths)} files to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

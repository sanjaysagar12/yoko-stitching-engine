"""Command line: ``avatar-engine build body.json``. Reads the file, hands its text to the engine, prints a summary."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from .contracts import ContractError
from .engine import AvatarEngine, AvatarResult
from .errors import AvatarEngineError

ENV_MODEL_DIR = "AVATAR_ENGINE_MODEL_DIR"
ENV_BODY_DIR = "AVATAR_ENGINE_BODY_DIR"
_REPO_ROOT = Path(__file__).resolve().parents[2]


def format_summary(r: AvatarResult) -> str:
    """Measurement table: target / fit (before ring correction) / final, in cm."""
    head = f"{'measurement':22s}{'target':>9s}{'before':>9s}{'err':>8s}{'after':>9s}{'err':>8s}"
    lines = [
        f"avatar {r.name}: {r.bundle['model_type']} {r.bundle['model_file']}, "
        f"{r.report['num_betas_used']} betas (requested {r.report['num_betas_requested']})",
        head,
    ]
    for k, row in r.report["measurements"].items():
        if row["target_cm"] is None:
            continue
        lines.append(
            f"{k:22s}{row['target_cm']:9.2f}{row['pre_correction_cm']:9.2f}{row['pre_error_cm']:+8.3f}"
            f"{row['post_correction_cm']:9.2f}{row['post_error_cm']:+8.3f}"
        )
    lines += [f"warning: {w}" for w in r.warnings]
    lines.append(f"{len(r.volumes)} bounding volumes, {len(r.points)} arrangement points")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Entry point. Exit codes: 0 ok, 2 fatal."""
    ap = argparse.ArgumentParser(prog="avatar-engine", description="SMPL-family avatar from body measurements")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="fit an avatar and write mesh, bounding volumes and arrangement points")
    b.add_argument("body", help="body measurement JSON (file name resolved against --body-dir, or a path)")
    b.add_argument("--model-dir", help=f"folder with SMPL / SMPL-X files (env {ENV_MODEL_DIR}; default models/smplx)")
    b.add_argument("--body-dir", help=f"folder of body JSON files (env {ENV_BODY_DIR}; default assert/body)")
    b.add_argument("--out", help="output folder (default outputs/avatar/<name>)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s"
    )

    body_dir = Path(args.body_dir or os.environ.get(ENV_BODY_DIR) or _REPO_ROOT / "assert" / "body")
    path = Path(args.body)
    if not path.is_absolute() and not path.is_file():
        path = body_dir / args.body
    if path.suffix != ".json":
        path = path.with_name(path.name + ".json")
    model_dir = Path(args.model_dir or os.environ.get(ENV_MODEL_DIR) or _REPO_ROOT / "models" / "smplx")
    try:
        text = path.read_text(encoding="utf-8-sig")
    except OSError as e:
        print(f"error: cannot read {path}: {e}", file=sys.stderr)
        return 2
    out = Path(args.out) if args.out else _REPO_ROOT / "outputs" / "avatar" / path.stem
    try:
        result = AvatarEngine().build(text, model_dir, out)
    except (AvatarEngineError, ContractError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(format_summary(result))
    print(f"wrote {len(result.paths)} files to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Command line: ``sm2d-engine run|inspect job.json``. Reads the input files, hands their text to the Engine."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .engine import Engine, EngineOptions, EngineRequest
from .errors import EngineError
from .reporting import EngineResult

ENV_SM2D = "SM2D_ENGINE_SM2D_DIR"
ENV_SMIS = "SM2D_ENGINE_SMIS_DIR"
_REPO_ROOT = Path(__file__).resolve().parents[2]


class JobSpec(BaseModel):
    """job.json. Unknown keys are an error. ``sm2d``/``smis`` are file names (or absolute paths)."""

    model_config = ConfigDict(extra="forbid")

    sm2d: str
    smis: str
    output_dir: str = "out"
    pieces: list[str] | None = None
    format: Literal["obj", "ply"] = "obj"
    combined_layout: bool = True
    layout_gap_cm: float = Field(5.0, ge=0)
    mirror_fold_pieces: bool = False


def _default_dir(env: str, sub: str) -> Path:
    if os.environ.get(env):
        return Path(os.environ[env])
    for base in (Path.cwd(), _REPO_ROOT):  # "assert" is the repo's (sic) input folder
        if (base / "assert" / sub).is_dir():
            return base / "assert" / sub
    return Path.cwd() / "assert" / sub


def _resolve(name: str, directory: Path) -> Path:
    p = Path(name)
    return p if p.is_absolute() else directory / p


def load_job(path: Path) -> JobSpec:
    """Read and validate a job file."""
    return JobSpec.model_validate_json(path.read_text(encoding="utf-8-sig"))


def build_request(job: JobSpec, sm2d_dir: Path, smis_dir: Path, out: str | None) -> EngineRequest:
    """Read the input files named by ``job`` and wrap their contents in an EngineRequest."""
    sm2d, smis = _resolve(job.sm2d, sm2d_dir), _resolve(job.smis, smis_dir)
    opts = EngineOptions(
        output_dir=out or job.output_dir,
        pieces=job.pieces,
        format=job.format,
        combined_layout=job.combined_layout,
        layout_gap_cm=job.layout_gap_cm,
        mirror_fold_pieces=job.mirror_fold_pieces,
    )
    return EngineRequest(
        sm2d_xml=sm2d.read_text(encoding="utf-8-sig"),
        smis_xml=smis.read_text(encoding="utf-8-sig"),
        options=opts,
        sm2d_name=sm2d.name,
        smis_name=smis.name,
    )


def _f(v: float | None, spec: str) -> str:
    return "-" if v is None else format(v, spec)


def summary(result: EngineResult) -> str:
    """Plain-text table of the per-piece outcome."""
    rows = [("piece", "status", "w cm", "h cm", "area cm2", "tris", "xings", "note")]
    for p in result.pieces:
        note = p.reason or "; ".join(p.gaps)
        rows.append(
            (
                p.name,
                p.status,
                _f(p.width_cm, ".2f"),
                _f(p.height_cm, ".2f"),
                _f(p.area_cm2, ".1f"),
                "-" if p.triangles is None else str(p.triangles),
                "-" if p.self_intersections is None else str(p.self_intersections),
                note,
            )
        )
    w = [max(len(r[i]) for r in rows) for i in range(7)]

    def fmt(r: tuple[str, ...]) -> str:
        cells = [c.ljust(w[i]) if i in (0, 1) else c.rjust(w[i]) for i, c in enumerate(r[:7])]
        return ("  ".join(cells) + "  " + r[7]).rstrip()

    lines = [fmt(r) for r in rows]
    ok = len(result.pieces) - len(result.failed)
    lines.append(f"{ok}/{len(result.pieces)} pieces ok")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Entry point. Exit code 0 = ok, 1 = some pieces failed, 2 = fatal error."""
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("job", type=Path, help="job.json")
    common.add_argument("--sm2d-dir", type=Path, help=f"folder for .sm2d names (env {ENV_SM2D})")
    common.add_argument("--smis-dir", type=Path, help=f"folder for .smis names (env {ENV_SMIS})")
    common.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    ap = argparse.ArgumentParser(prog="sm2d-engine", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", parents=[common], help="write meshes + report.json")
    run.add_argument("--out", help="output directory (overrides job output_dir)")
    sub.add_parser("inspect", parents=[common], help="list pieces and sizes, write nothing")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING, format="%(levelname)s %(message)s")

    try:
        job = load_job(args.job)
        request = build_request(
            job,
            args.sm2d_dir or _default_dir(ENV_SM2D, "sm2d"),
            args.smis_dir or _default_dir(ENV_SMIS, "smis"),
            getattr(args, "out", None),
        )
        engine = Engine()
        result = engine.run(request) if args.cmd == "run" else engine.inspect(request)
    except (OSError, ValidationError, json.JSONDecodeError, EngineError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(summary(result))
    if args.cmd == "run":
        print(f"wrote {len(result.output_paths)} files to {request.options.output_dir}")
    return 1 if result.failed else 0


if __name__ == "__main__":
    sys.exit(main())

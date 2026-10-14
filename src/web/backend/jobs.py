"""Web job files: ``assert/jobs/<name>.json`` = which pattern, measurements and body to combine."""

from __future__ import annotations

import json
import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict, ValidationError

from .config import AppConfig

_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")


class JobFile(BaseModel):
    """Job definition. Unknown keys are an error. File names are resolved against the configured folders."""

    model_config = ConfigDict(extra="forbid")

    name: str
    sm2d: str
    smis: str
    body: str
    pieces: list[str] | None = None


class JobError(Exception):
    """A job file is missing, malformed, or points at files that do not exist."""

    def __init__(self, code: str, message: str, problems: list[dict[str, str]] | None = None) -> None:
        self.code = code
        self.problems = problems
        super().__init__(message)


def list_jobs(cfg: AppConfig) -> list[JobFile]:
    """All valid jobs (malformed files are skipped), sorted by name."""
    out: list[JobFile] = []
    for p in sorted(cfg.jobs_dir.glob("*.json")) if cfg.jobs_dir.is_dir() else []:
        try:
            out.append(JobFile.model_validate(json.loads(p.read_text(encoding="utf-8-sig"))))
        except (ValidationError, json.JSONDecodeError):
            continue
    return out


def load_job(cfg: AppConfig, name: str) -> JobFile:
    """Job by name (raises JobError ``job_not_found`` / ``job_invalid``)."""
    if not _NAME.match(name) or not (cfg.jobs_dir / f"{name}.json").is_file():
        raise JobError("job_not_found", f"no job named {name!r}")
    try:
        return JobFile.model_validate(json.loads((cfg.jobs_dir / f"{name}.json").read_text(encoding="utf-8-sig")))
    except (ValidationError, json.JSONDecodeError) as e:
        raise JobError("job_invalid", f"job {name!r}: {e}") from None


def _resolve(folder: Path, name: str, what: str) -> Path:
    p = Path(name)
    p = p if p.is_absolute() else folder / name
    if not p.is_file():
        raise JobError("job_file_missing", f"{what} file not found: {p}")
    return p


def resolve_inputs(cfg: AppConfig, job: JobFile) -> tuple[Path, Path, Path]:
    """(sm2d path, smis path, body path) of a job."""
    return (
        _resolve(cfg.sm2d_dir, job.sm2d, "sm2d"),
        _resolve(cfg.smis_dir, job.smis, "smis"),
        _resolve(cfg.body_dir, job.body, "body"),
    )

"""Where the backend finds its inputs and writes its outputs (all overridable for tests)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class AppConfig:
    """Folders used by the app. File names in a job are resolved against ``sm2d_dir``, ``smis_dir``, ``body_dir``."""

    repo_root: Path
    sm2d_dir: Path
    smis_dir: Path
    body_dir: Path
    jobs_dir: Path
    models_dir: Path
    outputs_dir: Path
    frontend_dist: Path

    @classmethod
    def default(cls, repo_root: Path | None = None) -> AppConfig:
        """Repository layout (``assert/sm2d`` ...), with environment overrides (``SM2D_WEB_<NAME>``)."""
        root = Path(repo_root) if repo_root else _REPO_ROOT

        def pick(env: str, default: Path) -> Path:
            return Path(os.environ.get(f"SM2D_WEB_{env}", str(default)))

        return cls(
            repo_root=root,
            sm2d_dir=pick("SM2D_DIR", root / "assert" / "sm2d"),
            smis_dir=pick("SMIS_DIR", root / "assert" / "smis"),
            body_dir=pick("BODY_DIR", root / "assert" / "body"),
            jobs_dir=pick("JOBS_DIR", root / "assert" / "jobs"),
            models_dir=pick("MODELS_DIR", root / "models" / "smplx"),
            outputs_dir=pick("OUTPUTS_DIR", root / "outputs"),
            frontend_dist=pick("FRONTEND_DIST", root / "src" / "web" / "frontend" / "dist"),
        )

"""A male_shirt build with the synthetic avatar, written where the backend's build cache looks for it.

``BuildManager.start`` reuses any build folder whose ``build.json`` exists for the input hash, so the API and browser
tests can run the real backend without the SMPL-X files: pieces from ``sm2d_engine``, the avatar from
``tests/synthetic_avatar.py`` (through the real avatar_engine volume and point code), the same ``build.json`` layout.
"""

from __future__ import annotations

import json
from pathlib import Path

from sm2d_engine.engine import Engine, EngineOptions, EngineRequest
from tests.synthetic_avatar import build_avatar
from web.backend.builds import BuildManager, input_hash
from web.backend.config import AppConfig
from web.backend.jobs import load_job, resolve_inputs


def prebuild(outputs_dir: Path, job_name: str = "male_shirt") -> str:
    """Write the build of ``job_name`` into ``outputs_dir`` (the app's ``SM2D_WEB_OUTPUTS_DIR``); returns its id."""
    cfg = AppConfig.default()
    job = load_job(cfg, job_name)
    sm2d_p, smis_p, body_p = resolve_inputs(cfg, job)
    digest = input_hash(sm2d_p.read_bytes(), smis_p.read_bytes(), body_p.read_bytes(), job.pieces)
    bid = f"{job.name}-{digest[:12]}"
    out = Path(outputs_dir) / job.name / digest[:16]
    if (out / "build.json").is_file():
        return bid
    res = Engine().run(
        EngineRequest(
            sm2d_p.read_text(encoding="utf-8-sig"),
            smis_p.read_text(encoding="utf-8-sig"),
            EngineOptions(output_dir=out / "pieces", pieces=job.pieces, combined_layout=True),
            sm2d_p.name,
            smis_p.name,
        )
    )
    avatar_dir = build_avatar(out / "avatar")
    bundle = json.loads((avatar_dir / "avatar_bundle.json").read_text(encoding="utf-8"))

    def read(name: str) -> object:
        return json.loads((avatar_dir / name).read_text(encoding="utf-8"))

    result = {
        "id": bid,
        "job": job.name,
        "sm2d": sm2d_p.name,
        "smis": smis_p.name,
        "body": body_p.name,
        "pieces": [BuildManager._piece_doc(p, out / "pieces", out) for p in res.pieces],
        "layout_file": next((f"pieces/{Path(x).name}" for x in res.output_paths if x.endswith("_layout_m.obj")), None),
        "avatar": {
            "bundle": bundle,
            "mesh": f"avatar/{bundle['mesh']}",
            "measurements": read(bundle["files"]["measurements_report"]),
            "points": read(bundle["files"]["arrangement_points"])["points"],  # type: ignore[index]
            "volumes": read(bundle["files"]["bounding_volumes"])["volumes"],  # type: ignore[index]
        },
        "warnings": res.warnings,
    }
    (out / "build.json").write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    return bid

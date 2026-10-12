"""Avatar engine facade: body measurements JSON text -> avatar mesh, bounding volumes, arrangement points.

The caller reads files and passes their contents; the engine writes its outputs to ``out_dir``. All geometry is in
metres (y up, feet on y = 0, +z front, +x the wearer's left); measurements enter in the unit of the JSON file.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__, contracts
from .bodymodel import BodyModel
from .errors import MeasurementDefinitionError
from .fitting import FitResult, ShapeFitter
from .measurements import KNOWN, MeasurementDef, RingDef, build_definitions
from .modelfiles import load_model, locate
from .points import ArrangementPoint, PointBuilder, load_specs
from .ringcorrect import RingCorrector
from .volumes import BoundingVolume, VolumeBuilder
from .writer import CONVENTION, points_doc, volumes_doc, write_all

log = logging.getLogger(__name__)
_TO_M = {"cm": 0.01, "mm": 0.001, "m": 1.0}
DEFAULT_NUM_BETAS = 30

ModelFactory = Callable[[Path, str, str], BodyModel]


def default_model_factory(model_dir: Path, gender: str, model_type: str) -> BodyModel:
    """Locate (smpl | smplx | auto) and load the model files under ``model_dir``."""
    return load_model(locate(model_dir, gender, model_type))


@dataclass
class AvatarResult:
    """What ``AvatarEngine.build`` returns. Paths are the written files; documents are what they contain."""

    name: str
    bundle: dict[str, Any]
    report: dict[str, Any]
    paths: dict[str, Path]
    vertices: Any
    faces: Any
    volumes: list[BoundingVolume]
    points: list[ArrangementPoint]
    warnings: list[str] = field(default_factory=list)


class AvatarEngine:
    """Builds avatars. Collaborators are injectable so tests can swap the body model."""

    def __init__(
        self,
        model_factory: ModelFactory = default_model_factory,
        fitter: ShapeFitter | None = None,
        corrector: RingCorrector | None = None,
        points_config: Path | None = None,
    ) -> None:
        self.model_factory = model_factory
        self.fitter = fitter or ShapeFitter()
        self.corrector = corrector or RingCorrector()
        self.specs = load_specs(points_config)

    def build(self, body_json_text: str, model_dir: Path, out_dir: Path, write: bool = True) -> AvatarResult:
        """Fit the avatar to the measurements in ``body_json_text`` and write the outputs to ``out_dir``."""
        body = contracts.loads("body_measurements", body_json_text)
        scale = _TO_M[body["units"]]
        targets = {k: float(x) * scale for k, x in body["measurements"].items()}
        opts = body.get("options", {})
        n_betas = int(opts.get("num_betas", DEFAULT_NUM_BETAS))
        exact = bool(opts.get("exact_circumferences", True))
        warns: list[str] = []

        model = self.model_factory(Path(model_dir), body["gender"], opts.get("model_type", "auto"))
        defs = self._definitions(model, list(targets), warns)
        fit = self.fitter.fit(model, defs, targets, n_betas)
        warns += fit.warnings
        v, j = model.rest(fit.betas)
        pre = {k: d.evaluate(v, j) for k, d in defs.items()}

        scales: dict[str, float] = {}
        if exact:
            cr = self.corrector.correct(model, v, j, defs, targets)
            v, scales = cr.vertices, cr.scales
            warns += cr.warnings
        post = {k: d.evaluate(v, j) for k, d in defs.items()}

        volumes = VolumeBuilder(model).build(v, j)
        points = PointBuilder(self.specs).build(volumes, v, model.faces)
        for p in points:
            if p.spec.offset_m < 0:
                warns.append(f"point {p.spec.name}: negative offset")

        bundle = self._bundle(body, model, fit, targets, post, v)
        report = self._report(defs, targets, pre, post, scales, fit, exact, warns)
        vdoc, pdoc = volumes_doc(volumes), points_doc(points)
        header = [
            f"avatar: {body['name']}",
            f"units: metres (1 OBJ unit = 1 m); {CONVENTION}",
            f"model: {model.model_type} {model.source_file}; betas used: {fit.num_betas_used}",
        ]
        paths: dict[str, Path] = {}
        if write:
            paths = write_all(Path(out_dir), bundle, vdoc, pdoc, report, v, model.faces, header)
        return AvatarResult(body["name"], bundle, report, paths, v, model.faces, volumes, points, warns)

    @staticmethod
    def _definitions(model: BodyModel, targeted: list[str], warns: list[str]) -> dict[str, MeasurementDef]:
        try:
            return build_definitions(model, None)
        except MeasurementDefinitionError as e:
            log.warning("not all measurements can be defined on this model: %s", e)
            warns.append(f"only the requested measurements are defined: {e}")
            return build_definitions(model, targeted)

    @staticmethod
    def _bundle(
        body: dict[str, Any],
        model: BodyModel,
        fit: FitResult,
        targets: dict[str, float],
        post: dict[str, float],
        v: Any,
    ) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "units": "m",
            "convention": CONVENTION,
            "name": body["name"],
            "model_type": model.model_type,
            "model_file": model.source_file,
            "gender": body["gender"],
            "betas": [round(float(b), 9) for b in fit.betas],
            "num_betas_requested": fit.num_betas_requested,
            "num_betas_used": fit.num_betas_used,
            "target_cm": {k: round(t * 100, 4) for k, t in targets.items()},
            "achieved_cm": {k: round(x * 100, 4) for k, x in post.items()},
            "mesh": "avatar.obj",
            "files": {
                "bounding_volumes": "bounding_volumes.json",
                "arrangement_points": "arrangement_points.json",
                "measurements_report": "measurements_report.json",
            },
            "bbox_m": {
                "min": [round(float(x), 6) for x in v.min(axis=0)],
                "max": [round(float(x), 6) for x in v.max(axis=0)],
            },
            "engine_version": __version__,
        }

    @staticmethod
    def _report(
        defs: dict[str, MeasurementDef],
        targets: dict[str, float],
        pre: dict[str, float],
        post: dict[str, float],
        scales: dict[str, float],
        fit: FitResult,
        exact: bool,
        warns: list[str],
    ) -> dict[str, Any]:
        rows: dict[str, Any] = {}
        for k in KNOWN:
            if k not in defs:
                continue
            row: dict[str, Any] = {
                "kind": "circumference" if isinstance(defs[k], RingDef) else "length",
                "definition": defs[k].description,
                "target_cm": None if k not in targets else round(targets[k] * 100, 4),
                "pre_correction_cm": round(pre[k] * 100, 4),
                "post_correction_cm": round(post[k] * 100, 4),
                "pre_error_cm": None if k not in targets else round((pre[k] - targets[k]) * 100, 4),
                "post_error_cm": None if k not in targets else round((post[k] - targets[k]) * 100, 4),
            }
            if k in scales:
                row["ring_radial_scale"] = round(scales[k], 6)
            rows[k] = row
        return {
            "schema_version": 1,
            "units": "cm",
            "exact_circumferences": exact,
            "num_betas_requested": fit.num_betas_requested,
            "num_betas_used": fit.num_betas_used,
            "max_abs_beta": round(float(np.abs(fit.betas).max()) if len(fit.betas) else 0.0, 4),
            "measurements": rows,
            "warnings": warns,
        }

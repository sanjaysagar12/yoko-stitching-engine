"""Shape fit: betas that reproduce the target measurements (scipy least squares, CPU, deterministic)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares

from .bodymodel import BodyModel
from .errors import FitDidNotConvergeError
from .measurements import MeasurementDef, RingDef

F64 = NDArray[np.float64]

# residual normalisation: one unit of residual = this many metres of measurement error
TOL_HEIGHT_M = 0.0005
TOL_OTHER_M = 0.003
# Circumferences are mostly left to the ring corrector: it scales the surface isotropically around the body axis, so
# changing a circumference moves the surface predictably (about dC / 2 pi), instead of the shape components putting the
# extra girth wherever the training bodies carry it. The fit only has to get close (within a few cm).
TOL_RING_M = 0.05


@dataclass
class FitResult:
    """Outcome of the shape fit. ``betas`` has ``num_betas_used`` entries (zero start, bounded)."""

    betas: F64
    num_betas_requested: int
    num_betas_used: int  # min(requested, components the model file provides)
    nfev: int
    cost: float
    values_m: dict[str, float]  # measurement values at the fitted betas (metres)
    warnings: list[str]


class ShapeFitter:
    """Bounded, L2-regularised least squares over the shape coefficients, started at zero.

    ``bound`` is the +- limit on every beta, ``l2`` the weight of the ridge penalty (a beta of 1 costs as much as
    ``l2`` residual
    units, one unit being 0.5 mm of height, 3 mm of a length or 5 cm of a circumference).
    """

    def __init__(
        self, bound: float = 5.0, l2: float = 0.3, height_tol_m: float = 0.001, ring_tol_m: float = TOL_RING_M
    ) -> None:
        self.bound = bound
        self.ring_tol_m = ring_tol_m
        self.l2 = l2
        self.height_tol_m = height_tol_m

    def _tol(self, name: str, d: MeasurementDef) -> float:
        if name == "height":
            return TOL_HEIGHT_M
        return self.ring_tol_m if isinstance(d, RingDef) else TOL_OTHER_M

    def fit(
        self,
        model: BodyModel,
        defs: dict[str, MeasurementDef],
        targets_m: dict[str, float],
        num_betas: int,
    ) -> FitResult:
        """Fit ``targets_m`` (metres, subset of ``defs``). No targets gives the mean shape."""
        n = min(num_betas, model.num_betas)
        warns: list[str] = []
        if n < num_betas:
            warns.append(f"model provides {model.num_betas} shape components; using {n} of the {num_betas} requested")
        names = [k for k in targets_m if k in defs]
        if not names:
            v, j = model.rest(np.zeros(n))
            return FitResult(np.zeros(n), num_betas, n, 0, 0.0, {k: d.evaluate(v, j) for k, d in defs.items()}, warns)
        tgt = np.array([targets_m[k] for k in names])
        tol = np.array([self._tol(k, defs[k]) for k in names])

        def resid(b: F64) -> F64:
            v, j = model.rest(b)
            vals = np.array([defs[k].evaluate(v, j) for k in names])
            return np.concatenate([(vals - tgt) / tol, self.l2 * b])

        res = least_squares(
            resid,
            np.zeros(n),
            bounds=(-self.bound, self.bound),
            diff_step=1e-4,
            max_nfev=300,
            x_scale=1.0,
        )
        v, j = model.rest(res.x)
        values = {k: d.evaluate(v, j) for k, d in defs.items()}
        if "height" in targets_m and "height" in defs:
            err = abs(values["height"] - targets_m["height"])
            if err > self.height_tol_m:
                raise FitDidNotConvergeError(
                    "height", values["height"] * 100, targets_m["height"] * 100, self.height_tol_m * 100
                )
        at_bound = int(np.sum(np.abs(res.x) >= self.bound - 1e-6))
        if at_bound:
            warns.append(f"{at_bound} shape coefficient(s) reached the bound +-{self.bound:g}")
        return FitResult(res.x, num_betas, n, int(res.nfev), float(res.cost), values, warns)

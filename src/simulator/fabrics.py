"""Fabrics in measured units, and their conversion to the cloth solver's material parameters.

Each preset carries what a fabric test sheet gives: areal density (g/m^2), bending rigidity B (KES-FB2, N m, the
moment per unit width per unit curvature), the initial tensile modulus of the membrane E (N/m, per unit width), the
Poisson ratio, the largest stretch the weave allows and the friction against skin and cloth. The values are typical
published figures for each class of fabric, not a lab measurement of a particular cloth.

ZOZO's Contact Solver normalises its stiffnesses by density (``frontend/_param_.py``, ``crates/ppf-cts-core/src/
datamodel/params.rs``): the Baraff-Witkin membrane energy of a triangle is its mass times ``young-mod`` times the
strain term, so ``young-mod = E / rho``; the Discrete Shells hinge stiffness is ``BEND_SCALE * bend * rho`` with
``BEND_SCALE = 1.28e-5`` (``kernels/energy/model/shell_bend_stiffness.kernel.cpp``, calibrated by its authors on
Cusick drape tests), so ``bend = B / (BEND_SCALE * rho)``. Both conversions are in ``solver_params``.

Interfaced pieces (collar, stand, cuffs: the pattern has a ``<piece>Interface`` piece for them) are not simulated as
a second layer: their bending rigidity is multiplied by ``interfaced_bend_factor`` (fused shirting is 4 to 8 times
stiffer in bending than the shell fabric alone).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .errors import SimulationError

BEND_SCALE = 1.28e-5  # the solver's shell bending scale (see the module docstring)
GF_CM = 9.80665e-5  # 1 gf cm^2/cm (KES bending rigidity unit) in N m


@dataclass(frozen=True)
class Fabric:
    """A fabric in measured units (see the module docstring)."""

    name: str
    label: str
    areal_density_g_m2: float
    bending_rigidity_n_m: float
    tensile_modulus_n_m: float
    poisson: float
    strain_limit: float
    friction: float
    interfaced_bend_factor: float
    note: str

    @property
    def density_kg_m2(self) -> float:
        return self.areal_density_g_m2 / 1000.0

    def solver_params(self) -> dict[str, Any]:
        """The garment's material parameters for the solver (``young-mod`` and ``bend`` density normalised)."""
        rho = self.density_kg_m2
        return {
            "model": "baraff-witkin",
            "density": rho,
            "young-mod": self.tensile_modulus_n_m / rho,
            "poiss-rat": self.poisson,
            "bend": self.bending_rigidity_n_m / (BEND_SCALE * rho),
            "strain-limit": self.strain_limit,
            "friction": self.friction,
        }

    def doc(self) -> dict[str, Any]:
        return {**asdict(self), "solver": self.solver_params()}


FABRICS: dict[str, Fabric] = {
    f.name: f
    for f in (
        Fabric(
            "cotton_shirting",
            "Cotton shirting (poplin, 115 g/m2)",
            115.0,
            0.06 * GF_CM,
            400.0,
            0.3,
            0.04,
            0.45,
            6.0,
            "plain-woven cotton poplin, the usual men's dress shirt cloth",
        ),
        Fabric(
            "cotton_oxford",
            "Cotton oxford (140 g/m2)",
            140.0,
            0.12 * GF_CM,
            500.0,
            0.3,
            0.04,
            0.5,
            5.0,
            "basket-woven cotton, a casual shirt cloth: heavier and stiffer than poplin",
        ),
        Fabric(
            "linen",
            "Linen (170 g/m2)",
            170.0,
            0.25 * GF_CM,
            700.0,
            0.3,
            0.03,
            0.5,
            4.0,
            "plain-woven linen: stiff, hardly stretches",
        ),
        Fabric(
            "silk",
            "Silk crepe de chine (70 g/m2)",
            70.0,
            0.015 * GF_CM,
            250.0,
            0.3,
            0.05,
            0.3,
            8.0,
            "light, very soft drape",
        ),
        Fabric(
            "denim",
            "Denim (400 g/m2)",
            400.0,
            1.2 * GF_CM,
            1500.0,
            0.3,
            0.03,
            0.6,
            2.0,
            "twill-woven cotton, trousers and jackets",
        ),
        Fabric(
            "jersey",
            "Cotton jersey knit (180 g/m2)",
            180.0,
            0.02 * GF_CM,
            60.0,
            0.35,
            0.25,
            0.45,
            6.0,
            "single knit: soft and stretchy",
        ),
    )  # fmt: skip
}
DEFAULT_FABRIC = "cotton_shirting"


def fabric(name: str | None) -> Fabric:
    """The preset named ``name`` (the default when None)."""
    key = name or DEFAULT_FABRIC
    if key not in FABRICS:
        raise SimulationError(f"unknown fabric {key!r}; fabrics: {sorted(FABRICS)}")
    return FABRICS[key]


# simulation mesh sizes (target triangle edge, metres): CLO3D's "particle distance" is 20 mm by default, 5-10 mm for
# a fitting. The display mesh (the placed pieces, 5 mm) is carried by the simulated one (see scene.py).
QUALITIES: dict[str, float] = {"draft": 0.025, "standard": 0.018, "fine": 0.012}
DEFAULT_QUALITY = "standard"


def mesh_size(quality: str | None) -> float:
    key = quality or DEFAULT_QUALITY
    if key not in QUALITIES:
        raise SimulationError(f"unknown quality {key!r}; qualities: {sorted(QUALITIES)}")
    return QUALITIES[key]

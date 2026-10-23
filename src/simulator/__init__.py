"""Simulation: a stitching result (``stitching.result.json``) + its assembly run -> the garment sewn and draped on the
avatar by ZOZO's Contact Solver (docs/simulation.md).

The simulator reads files only (the stitching result, the ``assembler run`` folder, the piece bundles, the avatar
OBJ) and imports no engine; like the stitcher, its foreign imports are ``assembler.render`` (images) and
``assembler.verify`` (exact distances), plus the stitcher's file loaders. The solver runs in its own Python in a
subprocess (``ppf_runner.py``).
"""

from .errors import SimulationError, SolverFailed, SolverNotFound
from .fabrics import FABRICS, QUALITIES
from .simulate import Options, finish, folder_key, simulate
from .solver import solver_info

__all__ = [
    "FABRICS",
    "QUALITIES",
    "Options",
    "SimulationError",
    "SolverFailed",
    "SolverNotFound",
    "finish",
    "folder_key",
    "simulate",
    "solver_info",
]

"""Errors of the simulator (every one names what is valid or what to do)."""


class SimulationError(ValueError):
    """The stitching result, the run or the request cannot be simulated."""


class SolverNotFound(SimulationError):
    """ZOZO's Contact Solver is not installed where it is looked for (``PPF_DIR``)."""


class SolverFailed(SimulationError):
    """The solver stopped with an error (the message carries the tail of its log)."""

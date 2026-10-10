"""The only place where length units are converted. Canonical evaluation/report unit: cm."""

from __future__ import annotations

from .errors import UnitError

_CM_PER_UNIT = {"mm": 0.1, "cm": 1.0, "m": 100.0, "inch": 2.54}


def _factor(unit: str) -> float:
    try:
        return _CM_PER_UNIT[unit.strip().lower()]
    except KeyError:
        raise UnitError(unit) from None


def to_cm(value: float, unit: str) -> float:
    """Convert ``value`` given in ``unit`` to centimetres."""
    return value * _factor(unit)


def from_cm(value_cm: float, unit: str) -> float:
    """Convert centimetres to ``unit``."""
    return value_cm / _factor(unit)


def cm_to_m(value_cm: float) -> float:
    """Convert centimetres to metres (mesh build time only)."""
    return value_cm / 100.0

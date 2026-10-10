"""Body measurements, canonical unit cm. Lookup is case-sensitive."""

from __future__ import annotations

from collections.abc import Iterator

from ..errors import MissingMeasurementError


class Measurements:
    """Mapping name -> value in **cm** (converted from the .smis unit by the reader)."""

    def __init__(self, values_cm: dict[str, float], source_unit: str = "cm") -> None:
        self._v = dict(values_cm)
        self.source_unit = source_unit

    def get(self, name: str) -> float:
        """Value in cm; raises MissingMeasurementError if absent."""
        try:
            return self._v[name]
        except KeyError:
            raise MissingMeasurementError(name) from None

    def __contains__(self, name: object) -> bool:
        return name in self._v

    def __iter__(self) -> Iterator[str]:
        return iter(self._v)

    def __len__(self) -> int:
        return len(self._v)

    def as_dict(self) -> dict[str, float]:
        """Copy of the name -> cm mapping."""
        return dict(self._v)

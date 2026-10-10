import pytest

from sm2d_engine import units
from sm2d_engine.errors import UnitError


def test_roundtrip():
    assert units.to_cm(100, "mm") == pytest.approx(10)
    assert units.from_cm(10, "mm") == pytest.approx(100)
    assert units.to_cm(1, "inch") == pytest.approx(2.54)
    assert units.cm_to_m(60) == pytest.approx(0.6)


def test_unknown():
    with pytest.raises(UnitError):
        units.to_cm(1, "furlong")

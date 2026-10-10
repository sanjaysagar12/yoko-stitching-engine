import math

import pytest

from sm2d_engine.errors import FormulaError, MissingMeasurementError
from sm2d_engine.eval.expression import evaluate


def ev(s, **vars_):
    def look(name):
        if name in vars_:
            return vars_[name]
        raise MissingMeasurementError(name)

    return evaluate(s, look)


def test_arith_and_power():
    assert ev("2+3*4") == 14
    assert ev("2^3") == 8
    assert ev("(1+2)*3-4/2") == 7
    assert ev("-3+5") == 2


def test_ternary_nested():
    assert ev("1>2?10:20") == 20
    assert ev("a>2 ? (a>5 ? 1 : 2) : 3", a=4) == 2
    assert ev("a>2 ? a>5 ? 1 : 2 : 3", a=9) == 1


def test_trig_dialect():
    assert ev("atan(1)") == pytest.approx(math.pi / 4)
    assert ev("sinD(30)") == pytest.approx(0.5)
    assert ev("atanD(1)") == pytest.approx(45)
    assert ev("asinD(0.5)") == pytest.approx(30)


def test_functions_and_vars():
    assert ev("max(1,2,3)+min(4,5)+abs(-2)+sqrt(9)+sign(-4)") == 3 + 4 + 2 + 3 - 1
    assert ev("rint(2.5)") == 3
    assert ev("#CM*5 + Line_A_B", **{"#CM": 2.0, "Line_A_B": 1.0}) == 11
    assert ev("Пс_1 * 2", **{"Пс_1": 4.0}) == 8


def test_logic():
    assert ev("1<2 && 3>2") == 1
    assert ev("1>2 || 3<2") == 0


def test_errors():
    with pytest.raises(FormulaError):
        ev("2+*3")
    with pytest.raises(FormulaError):
        ev("foo(1)")
    with pytest.raises(MissingMeasurementError):
        ev("nope+1")
    with pytest.raises(FormulaError):
        ev("1/0")

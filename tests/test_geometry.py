import math

import numpy as np
import pytest

from sm2d_engine.geometry.bezier import Curve
from sm2d_engine.geometry.primitives import circle_line, line_line, move, qt_angle


def test_qt_angle_and_move():
    assert qt_angle((0, 0), (1, 0)) == pytest.approx(0)
    assert qt_angle((0, 0), (0, -1)) == pytest.approx(90)  # y-down: up on screen is +90
    assert qt_angle((0, 0), (0, 1)) == pytest.approx(270)
    x, y = move((1, 1), 2, 90)
    assert (x, y) == pytest.approx((1, -1))


def test_line_line_and_circle():
    assert line_line((0, 0), (1, 1), (0, 2), (2, 0)) == pytest.approx((1, 1))
    assert line_line((0, 0), (1, 0), (0, 1), (1, 1)) is None
    ts = circle_line((0, 0), (1, 0), (5, 0), 3)
    assert ts == pytest.approx((2, 8))


def _quarter_circle() -> Curve:
    k = 4 / 3 * (math.sqrt(2) - 1)
    seg = np.array([[1, 0], [1, k], [k, 1], [0, 1]], dtype=float) * 50
    return Curve([seg])


def test_arc_length_accuracy():
    c = _quarter_circle()
    # Bezier quarter circle approximation: length within 0.03% of true; check against dense reference
    t = np.linspace(0, 1, 2_000_001)
    s = c.segments[0]
    u = 1 - t
    x = u**3 * s[0, 0] + 3 * u * u * t * s[1, 0] + 3 * u * t * t * s[2, 0] + t**3 * s[3, 0]
    y = u**3 * s[0, 1] + 3 * u * u * t * s[1, 1] + 3 * u * t * t * s[2, 1] + t**3 * s[3, 1]
    ref = float(np.hypot(np.diff(x), np.diff(y)).sum())
    assert abs(c.length() - ref) < 1e-4  # cm -> 0.001 mm


def test_flatten_deviation_and_extrema():
    c = _quarter_circle()
    pts, _ = c.flatten(0.01)  # 0.1 mm in cm
    # all vertices on the curve's dense polyline within tolerance, and extrema included
    assert pts[:, 0].max() == pytest.approx(50, abs=1e-9)
    assert pts[:, 1].max() == pytest.approx(50, abs=1e-9)
    d = np.hypot(pts[:, 0], pts[:, 1])
    assert np.all(np.abs(d - 50) < 0.03)  # Bezier-vs-circle radial error bound
    # chord midpoints stay within 0.01 cm of the curve
    mids = (pts[:-1] + pts[1:]) / 2
    dist = np.array([c.project((float(m[0]), float(m[1])))[1] for m in mids])
    assert dist.max() < 0.0105


def test_point_at_length_and_project():
    c = _quarter_circle()
    total = c.length()
    p = c.point_at_length(total / 2)
    assert p[0] == pytest.approx(p[1], abs=1e-6)
    u, dist = c.project(p)
    assert dist < 1e-6
    assert c.length_at_u(u) == pytest.approx(total / 2, abs=1e-5)

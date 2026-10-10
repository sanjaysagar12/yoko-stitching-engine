"""Trousers sanity tests: geometry is checked for consistency (no reference sizes exist for this garment)."""

from __future__ import annotations

import math
from typing import Any

import pytest

from sm2d_engine.geometry.primitives import qt_angle
from sm2d_engine.pieces.assembler import assemble

from .conftest import SMIS, inspect, make_scope, read, set_measurement

T = ("trousers.sm2d", "trousers.smis")

# Pieces whose outline crosses itself in the sample pattern itself (see README "known limits"):
# the centre-front waist curve (control point A28) overshoots the centre-front line, and the pocket mouth
# has an implicit straight edge that crosses its neighbour.
CROSSING = {"Front_no_pocket", "Front_with_pocket", "Front_Pleats", "PocketMouth"}
# PocketMouth: the path goes A50 -> curve whose first point is Pocket2, 2.116 cm away. Seamly joins them with a
# straight edge, so the engine reports it as a gap note rather than hiding it.
IMPLICIT_EDGE = {"PocketMouth": 2.1161}


@pytest.fixture(scope="module")
def reports() -> dict[str, Any]:
    return inspect(*T)


def test_all_eleven_pieces_ok(reports: dict[str, Any]) -> None:
    assert len(reports) == 11
    assert {n for n, p in reports.items() if p.status != "ok"} == set()


def test_zero_gaps_except_documented(reports: dict[str, Any]) -> None:
    for n, p in reports.items():
        if n in IMPLICIT_EDGE:
            assert len(p.gaps) == 1 and f"{IMPLICIT_EDGE[n]:.4f} cm" in p.gaps[0], p.gaps
        else:
            assert p.gaps == [], n


def test_mesh_area_equals_outline_area(reports: dict[str, Any]) -> None:
    for n, p in reports.items():
        if n in CROSSING:
            assert p.area_diff_pct < 0.5, n  # earcut on a self-crossing outline: tiny sliver differs
        else:
            assert p.area_diff_pct < 0.001, n
    assert {n for n, p in reports.items() if p.self_intersections} == CROSSING
    assert all(p.self_intersections <= 1 for p in reports.values())


def test_size_regression(reports: dict[str, Any]) -> None:
    exp = {"WaistBand1": (52.0, 4.0), "Waistband2": (56.0, 4.0), "Detail": (4.0, 25.956), "Back": (39.696, 111.664)}
    for n, (w, h) in exp.items():
        assert reports[n].width_cm == pytest.approx(w, abs=0.01), n
        assert reports[n].height_cm == pytest.approx(h, abs=0.01), n


@pytest.mark.parametrize("meas", ["hip_circ", "waist_circ"])
def test_width_responds_to_measurement(meas: str, reports: dict[str, Any]) -> None:
    base = read(SMIS / T[1])
    up = inspect(*T, smis_xml=set_measurement(base, meas, 4.0))
    down = inspect(*T, smis_xml=set_measurement(base, meas, -4.0))
    changed = 0
    for n, p in reports.items():
        du, dd = up[n].width_cm - p.width_cm, down[n].width_cm - p.width_cm
        if abs(du) > 1e-6 or abs(dd) > 1e-6:
            changed += 1
            assert du * dd < 0, f"{n}: not monotone ({du}, {dd})"
            assert du == pytest.approx(-dd, abs=0.1), n  # symmetric response
    assert changed >= 5
    if meas == "hip_circ":  # panel widths follow hip/4 (+rotation effects)
        for n in ("Front_with_pocket", "Front_Pleats", "Back"):
            assert 0.9 < up[n].width_cm - reports[n].width_cm < 1.6, n
    else:  # waistband = waist/2 exactly
        for n in ("WaistBand1", "Waistband2"):
            assert up[n].width_cm - reports[n].width_cm == pytest.approx(2.0, abs=1e-6), n


def _rot(c: tuple[float, float], p: tuple[float, float], deg: float) -> tuple[float, float]:
    """Independent y-down rotation by ``deg`` counter-clockwise on screen."""
    t = math.radians(deg)
    dx, dy = p[0] - c[0], p[1] - c[1]
    return (c[0] + dx * math.cos(t) + dy * math.sin(t), c[1] - dx * math.sin(t) + dy * math.cos(t))


def test_rotated_geometry_lands_where_source_says() -> None:
    sc = make_scope(*T)
    for op_id, src, dst in (("392", "148", "393"), ("143", "19", "144"), ("404", "397", "405")):
        op = sc.model.tools[op_id]
        c, angle = sc.point(op.attrs["center"]), sc.eval(op.attrs["angle"])
        s, d = sc.point(src), sc.point(dst)
        exp = _rot(c, s, angle)
        assert d == pytest.approx(exp, abs=1e-9), op_id
        assert math.dist(c, d) == pytest.approx(math.dist(c, s))
        assert (qt_angle(c, d) - qt_angle(c, s)) % 360 == pytest.approx(angle % 360, abs=1e-7)
    # a rotated curve: control points rotate too
    op = sc.model.tools["143"]
    c, angle = sc.point(op.attrs["center"]), sc.eval(op.attrs["angle"])
    for a, b in zip(sc.curve("44").segments[0], sc.curve("145").segments[0], strict=True):
        assert tuple(b) == pytest.approx(_rot(c, (a[0], a[1]), angle), abs=1e-9)
    # the pattern is evaluated in the same unit chain: a3 operates on a2 output (angle sums, not source)
    assert sc.point("405") != sc.point("397")


def test_back_piece_outline_contains_rotated_point() -> None:
    sc = make_scope(*T)
    back = next(p for p in sc.model.pieces if p.name == "Back")
    outline = assemble(back, sc)
    target = sc.point("144")  # cm pattern: pattern unit == cm
    assert min(math.dist(tuple(v), target) for v in outline.points_cm) < 1e-6

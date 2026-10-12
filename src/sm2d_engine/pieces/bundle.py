"""Piece bundle: outline, named points, edges, grainline in the piece frame (metres, x right, y up, z = 0)."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .. import units
from ..eval.scope import VariableScope
from ..model.geometry_types import Polygon

log = logging.getLogger(__name__)
_ON_OUTLINE_TOL_CM = 0.02  # twice the 0.1 mm flattening chord deviation, so points on curves count as on the outline
F64 = NDArray[np.float64]


def _r(v: float) -> float:
    return round(float(v), 9)


def _named_points(scope: VariableScope) -> list[tuple[str, str, tuple[float, float]]]:
    """(name, label, xy in pattern unit) for every evaluable point, in document order, first name wins."""
    m = scope.model
    cands: list[tuple[str, str, str]] = []  # (id, name, label)
    for id_, t in m.tools.items():
        if t.kind == "point" and t.name:
            cands.append((id_, t.name, t.attrs.get("label", t.name)))
    for dest, (op_id, src) in m.op_dest.items():  # operation copies are named <source name><suffix>
        s, op = m.tools.get(src), m.tools[op_id]
        if s is not None and s.kind == "point" and s.name:
            name = s.name + op.attrs.get("suffix", "")
            cands.append((dest, name, name))
    out: list[tuple[str, str, tuple[float, float]]] = []
    seen: set[str] = set()
    for id_, name, label in cands:
        if name in seen:
            continue
        try:
            p = scope.point(id_)
        except Exception:  # noqa: BLE001 - a point that cannot be evaluated is simply not offered
            log.debug("named point %s (%s) not evaluable", name, id_)
            continue
        seen.add(name)
        out.append((name, label, p))
    return out


def _inside(poly: F64, q: F64) -> bool:
    """Even-odd point-in-polygon."""
    x, y = q
    a, b = poly, np.roll(poly, -1, axis=0)
    cond = (a[:, 1] > y) != (b[:, 1] > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        xs = a[:, 0] + (y - a[:, 1]) * (b[:, 0] - a[:, 0]) / (b[:, 1] - a[:, 1])
    return bool(np.count_nonzero(cond & (x < xs)) % 2 == 1)


def _project(poly: F64, cum: F64, q: F64) -> tuple[float, float]:
    """Closest point on the closed polygon to ``q`` -> (arc position s from vertex 0, distance). cm."""
    a, e = poly, np.roll(poly, -1, axis=0) - poly
    ll = e[:, 0] ** 2 + e[:, 1] ** 2
    w = q - a
    with np.errstate(divide="ignore", invalid="ignore"):
        k = np.where(ll > 0, (w[:, 0] * e[:, 0] + w[:, 1] * e[:, 1]) / ll, 0.0)
    k = np.clip(k, 0.0, 1.0)
    d = np.hypot(w[:, 0] - k * e[:, 0], w[:, 1] - k * e[:, 1])
    i = int(np.argmin(d))
    return float(cum[i] + k[i] * np.sqrt(ll[i])), float(d[i])


def build_piece_bundle(
    name: str,
    obj_name: str,
    poly_cm: Polygon,
    outline_cm_ydown: Polygon,
    grainline: dict[str, str],
    scope: VariableScope,
) -> dict[str, Any]:
    """Assemble the ``piece_bundle`` document.

    ``poly_cm`` is the final mesh boundary (y up, bbox bottom-left at the origin, cm); ``outline_cm_ydown`` is the
    outline before that placement, so pattern coordinates can be mapped into the same frame. Output is metres.
    """
    x0, y1 = float(outline_cm_ydown[:, 0].min()), float(outline_cm_ydown[:, 1].max())

    def to_piece_cm(p: tuple[float, float]) -> F64:
        return np.array([units.to_cm(p[0], scope.unit) - x0, y1 - units.to_cm(p[1], scope.unit)])

    seg = np.roll(poly_cm, -1, axis=0) - poly_cm
    seg_len = np.hypot(seg[:, 0], seg[:, 1])
    cum = np.concatenate([[0.0], np.cumsum(seg_len)[:-1]])
    perimeter = float(seg_len.sum())
    w, h = float(poly_cm[:, 0].max()), float(poly_cm[:, 1].max())
    lo, hi = np.array([0.0, 0.0]), np.array([w, h])

    pts: list[dict[str, Any]] = []
    on_list: list[tuple[float, str]] = []
    for pname, label, p in _named_points(scope):
        q = to_piece_cm(p)
        if np.any(q < lo - _ON_OUTLINE_TOL_CM) or np.any(q > hi + _ON_OUTLINE_TOL_CM):
            continue
        s, d = _project(poly_cm, cum, q)
        on = d <= _ON_OUTLINE_TOL_CM
        if not on and not _inside(poly_cm, q):
            continue
        pts.append(
            {
                "name": pname,
                "label": label,
                "xy_m": [_r(units.cm_to_m(q[0])), _r(units.cm_to_m(q[1]))],
                "on_outline": on,
                "outline_s_m": _r(units.cm_to_m(s)) if on else None,
            }
        )
        if on:
            on_list.append((s, pname))

    edges: list[dict[str, Any]] = []
    on_list.sort(key=lambda t: t[0])
    for i, (s, a) in enumerate(on_list):
        s2, b = on_list[(i + 1) % len(on_list)]
        length = (s2 - s) if i + 1 < len(on_list) else (on_list[0][0] + perimeter - s)
        edges.append({"from": a, "to": b, "length_m": _r(units.cm_to_m(length))})

    return {
        "schema_version": 3,
        "name": name,
        "units": "m",
        "obj": obj_name,
        "bbox_m": {"min": [0.0, 0.0], "max": [_r(units.cm_to_m(w)), _r(units.cm_to_m(h))]},
        "outline_m": [[_r(units.cm_to_m(x)), _r(units.cm_to_m(y))] for x, y in poly_cm],
        "named_points": pts,
        "edges": edges,
        "fold_edge": None,  # never guessed: no sample pattern defines foldPosition
        "grainline": _grainline(grainline, scope, to_piece_cm),
        "mirror_of": None,
        "axis": None,
        "origin_shift_m": None,
        # the piece origin in the drafting frame (y up): pieces drafted on shared base lines keep their heights
        "pattern_origin_m": [_r(units.cm_to_m(x0)), _r(units.cm_to_m(-y1))],
        "default_reference": {
            "type": "bbox_centre",
            "xy_m": [_r(units.cm_to_m(w / 2)), _r(units.cm_to_m(h / 2))],
        },
    }


def _grainline(attrs: dict[str, str], scope: VariableScope, to_piece_cm: Any) -> dict[str, Any] | None:
    """Grainline as ``{anchor_xy_m, rotation_deg, length_m}`` or None when absent/invisible/unresolvable.

    ``rotation_deg`` is Seamly's value (counter-clockwise from +x in the y-up piece frame; 90 = vertical).
    """
    if not attrs or attrs.get("visible", "true").lower() not in ("1", "true"):
        return None
    if not attrs.get("length") or not attrs.get("rotation") or not attrs.get("centerAnchor"):
        return None
    try:
        anchor = scope.model.tools[attrs["centerAnchor"]]
        pid = anchor.attrs["idObject"] if anchor.type == "anchor" else anchor.id
        q = to_piece_cm(scope.point(pid))
        length_cm = units.to_cm(scope.eval(attrs["length"]), scope.unit)
        rotation = scope.eval(attrs["rotation"])
    except Exception:  # noqa: BLE001 - unresolvable grainline is reported as null, never guessed
        return None
    return {
        "anchor_xy_m": [_r(units.cm_to_m(q[0])), _r(units.cm_to_m(q[1]))],
        "rotation_deg": _r(rotation),
        "length_m": _r(units.cm_to_m(length_cm)),
    }

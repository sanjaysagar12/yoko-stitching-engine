"""Stitch report: compare two outline edges between named points and sample them for drawing. No simulation."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .errors import UnknownNamedPointError, UnknownPieceError

F64 = NDArray[np.float64]


def _outline(piece: dict[str, Any]) -> tuple[F64, F64, float]:
    p = np.asarray(piece["outline_m"], dtype=np.float64)
    seg = np.linalg.norm(np.roll(p, -1, axis=0) - p, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)[:-1]])
    return p, cum, float(seg.sum())


def _point_at(p: F64, cum: F64, per: float, s: F64) -> F64:
    s = np.mod(s, per)
    i = np.clip(np.searchsorted(cum, s, side="right") - 1, 0, len(p) - 1)
    q = np.roll(p, -1, axis=0)
    seg = np.linalg.norm(q[i] - p[i], axis=1)
    k = np.where(seg > 0, (s - cum[i]) / np.where(seg > 0, seg, 1.0), 0.0)
    return np.asarray(p[i] + k[:, None] * (q[i] - p[i]), dtype=np.float64)


def _s_of(piece: dict[str, Any], name: str) -> float:
    for q in piece["named_points"]:
        if q["name"] == name and q["on_outline"]:
            return float(q["outline_s_m"])
    known = [q["name"] for q in piece["named_points"] if q["on_outline"]]
    raise UnknownNamedPointError(piece["name"], name, known)


def edge_between(piece: dict[str, Any], from_name: str, to_name: str, samples: int = 21) -> tuple[F64, float]:
    """Outline between two named points (the shorter way round), resampled by arc length.

    Returns (points (samples,2) in piece coordinates running from ``from_name`` to ``to_name``; length in metres).
    """
    p, cum, per = _outline(piece)
    s0, s1 = _s_of(piece, from_name), _s_of(piece, to_name)
    fwd = (s1 - s0) % per
    if fwd <= per - fwd:
        length, ss = fwd, s0 + np.linspace(0.0, fwd, samples)
    else:
        length, ss = per - fwd, s0 - np.linspace(0.0, per - fwd, samples)
    return _point_at(p, cum, per, ss), float(length)


def stitch_report(
    stitch: dict[str, Any],
    pieces: dict[str, dict[str, Any]],
    placed: dict[str, Callable[[F64], F64]],
    samples: int = 21,
) -> dict[str, Any]:
    """Report for one stitch ``{a: {piece, from_point, to_point}, b: {...}}``.

    ``placed`` maps a piece name to a function turning piece-plane points (N,2) into world points (N,3); stitch
    segments (pairs of world points, one per sample) are only produced if both pieces are placed. Lengths in cm,
    ``diff_cm = a - b`` and ``diff_pct`` relative to ``b``.
    """
    sides = {}
    for key in ("a", "b"):
        e = stitch[key]
        if e["piece"] not in pieces:
            raise UnknownPieceError(e["piece"], list(pieces))
        sides[key] = edge_between(pieces[e["piece"]], e["from_point"], e["to_point"], samples)
    la, lb = sides["a"][1], sides["b"][1]
    out: dict[str, Any] = {
        "a": stitch["a"],
        "b": stitch["b"],
        "a_length_cm": la * 100.0,
        "b_length_cm": lb * 100.0,
        "diff_cm": (la - lb) * 100.0,
        "diff_pct": 100.0 * (la - lb) / lb if lb > 0 else 0.0,
        "segments": None,
        "warnings": [],
    }
    pa, pb = stitch["a"]["piece"], stitch["b"]["piece"]
    if pa in placed and pb in placed:
        wa, wb = placed[pa](sides["a"][0]), placed[pb](sides["b"][0])
        if np.linalg.norm(wa - wb, axis=1).sum() > np.linalg.norm(wa - wb[::-1], axis=1).sum():
            wb = wb[::-1]  # pair the samples so that the stitch lines do not cross
        out["segments"] = [
            [[round(float(x), 6) for x in p], [round(float(x), 6) for x in q]] for p, q in zip(wa, wb, strict=True)
        ]
    else:
        out["warnings"].append("stitch segments need both pieces to be placed")
    return out

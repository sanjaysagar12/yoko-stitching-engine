"""Distance helpers used by the arrangement verification."""

from __future__ import annotations

import numpy as np
import pytest

from alignment_engine.verify import AvatarSurface, point_triangle_distance, self_overlaps, surface_samples


def brute(p: np.ndarray, tri: np.ndarray) -> float:
    rng = np.random.default_rng(0)
    w = rng.dirichlet([1, 1, 1], size=40000)
    pts = w @ tri
    edge = [tri[i] + (tri[(i + 1) % 3] - tri[i]) * t for i in range(3) for t in np.linspace(0, 1, 2000)[:, None]]
    return float(np.linalg.norm(np.vstack([pts, edge]) - p, axis=1).min())


def test_point_triangle_distance_matches_brute_force_in_all_regions() -> None:
    rng = np.random.default_rng(3)
    tri = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.2, 0.9, 0.3]])
    pts = rng.uniform(-1.0, 2.0, size=(60, 3))
    d = point_triangle_distance(pts, np.broadcast_to(tri, (len(pts), 3, 3)).copy())
    for p, di in zip(pts, d, strict=True):
        assert di == pytest.approx(brute(p, tri), abs=2e-3), p
    flat = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    assert point_triangle_distance(np.array([[0.2, 0.2, 0.5]]), flat[None])[0] == pytest.approx(0.5)  # above the face
    assert point_triangle_distance(np.array([[-1.0, 0.0, 0.0]]), flat[None])[0] == pytest.approx(1.0)  # beside a vertex


def test_avatar_surface_distance_on_a_sphere() -> None:
    from scipy.spatial import ConvexHull

    rng = np.random.default_rng(1)
    v = rng.normal(size=(1500, 3))
    v = 0.1 * v / np.linalg.norm(v, axis=1, keepdims=True)
    f = ConvexHull(v).simplices
    s = AvatarSurface(v, f, spacing=0.004)
    q = rng.normal(size=(200, 3))
    q = q / np.linalg.norm(q, axis=1, keepdims=True) * rng.uniform(0.11, 0.2, size=(200, 1))
    d = s.distance(q)
    assert np.abs(d - (np.linalg.norm(q, axis=1) - 0.1)).max() < 2.5e-3  # a polyhedron inside the sphere


def test_surface_samples_and_self_overlaps() -> None:
    v = np.array([[0, 0, 0], [0.02, 0, 0], [0, 0.02, 0.0], [0.02, 0.02, 0]], dtype=float)
    f = np.array([[0, 1, 2], [1, 3, 2]])
    s = surface_samples(v, f, spacing=0.003)
    assert len(s) > 4 + 5 * 5 and np.abs(s[:, 2]).max() == 0
    assert self_overlaps(v, f) == 0
    # a duplicated, non-adjacent copy of the sheet is a self overlap
    v2 = np.vstack([v, v + [0, 0, 0.0005]])
    f2 = np.vstack([f, f + 4])
    assert self_overlaps(v2, f2) == 2

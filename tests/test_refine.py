"""refine_piece: a conforming interior-refined triangulation of a piece outline (alignment_engine Part A)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from alignment_engine.loaders import read_pieces
from alignment_engine.refine import RefineCache, refine_piece

MAX_EDGE = 0.01


def poly_area(p: np.ndarray) -> float:
    return float(0.5 * np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1]))


def tri_stats(v: np.ndarray, f: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(areas, min angle per triangle in degrees, shortest edge per triangle)."""
    a, b, c = v[f[:, 0], :2], v[f[:, 1], :2], v[f[:, 2], :2]
    la, lb, lc = np.linalg.norm(b - c, axis=1), np.linalg.norm(a - c, axis=1), np.linalg.norm(a - b, axis=1)
    area = 0.5 * ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1]))

    def ang(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:  # angle opposite x
        return np.degrees(np.arccos(np.clip((y * y + z * z - x * x) / (2 * y * z), -1, 1)))

    m = np.minimum(np.minimum(ang(la, lb, lc), ang(lb, la, lc)), ang(lc, la, lb))
    return area, m, np.minimum(np.minimum(la, lb), lc)


@pytest.fixture(scope="module")
def refined(scene: dict[str, Path]) -> dict[str, tuple[dict, object]]:
    out = {}
    for name, info in read_pieces(scene["pieces"]).items():
        out[name] = (info.bundle, refine_piece(info.bundle, MAX_EDGE, info.mesh()))
    return out


def test_1_outline_vertices_kept_exactly_and_area_preserved(refined: dict) -> None:
    assert len(refined) == 17
    for name, (b, r) in refined.items():
        outline = np.array(b["outline_m"])
        bnd = r.vertices[r.boundary, :2]
        # every original outline vertex is a boundary vertex with identical coordinates, in the original order
        idx = r.original_index
        assert len(idx) == len(outline) and np.array_equal(r.vertices[idx, :2], outline), name
        assert np.all(
            np.diff(np.where(np.isin(r.boundary, idx), np.arange(len(r.boundary)), -1)[np.isin(r.boundary, idx)]) > 0
        )
        # the boundary only gains vertices on the original edges (collinear): its area is unchanged
        if not r.fallback:
            assert poly_area(bnd) == pytest.approx(poly_area(outline), abs=1e-9), name
        area, _, _ = tri_stats(r.vertices, r.faces)
        assert area.sum() == pytest.approx(poly_area(outline), abs=1e-9), name
        assert r.bundle["named_points"] == b["named_points"] and r.bundle["bbox_m"] == b["bbox_m"]


def test_1_edges_short_and_triangles_well_shaped(refined: dict) -> None:
    worst = 180.0
    for name, (_, r) in refined.items():
        f = r.faces
        e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
        assert np.linalg.norm(r.vertices[e[:, 0]] - r.vertices[e[:, 1]], axis=1).max() <= MAX_EDGE * 1.05, name
        _, ang, short = tri_stats(r.vertices, f)
        # triangles next to dense outline vertices (curves are flattened to about 1 mm) cannot be better
        # than the outline
        if r.fallback:  # the midpoint subdivision of a self-touching outline has slivers by construction
            continue
        regular = short >= 0.25 * MAX_EDGE
        # Delaunay maximises the minimum angle of a given point set; the few triangles below 20 degrees touch outline
        # runs whose vertices are 5 mm apart or a very sharp outline corner (BackPanel has a 6 degree spike)
        assert ang[regular].min() >= 15.0, (name, ang[regular].min())
        assert np.mean(ang[regular] < 20.0) < 1e-3, (name, np.mean(ang[regular] < 20.0))
        worst = min(worst, float(ang[regular].min()))
        assert len(r.vertices) < 12 * 1e4 * np.sum(np.abs(poly_area(np.array(refined[name][0]["outline_m"])))) + 200, (
            name
        )
    assert worst >= 15.0


def test_1_conforming_no_t_junctions_and_normals_up(refined: dict) -> None:
    for name, (_, r) in refined.items():
        f = r.faces
        e = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
        uniq, counts = np.unique(e, axis=0, return_counts=True)
        assert counts.max() <= 2, name  # every edge is shared by at most two triangles
        boundary_edges = {
            tuple(sorted((int(a), int(b)))) for a, b in zip(r.boundary, np.roll(r.boundary, -1), strict=True)
        }
        single = {tuple(map(int, x)) for x in uniq[counts == 1]}
        if not r.fallback:  # the self-touching FullSleeve outline has no single boundary loop
            assert single == boundary_edges, name  # the free edges are exactly the outline: no T-junctions, no holes
        t = r.vertices[f]
        nz = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])[:, 2]
        assert (nz > 0).all() and np.all(r.vertices[:, 2] == 0.0), name
        # all vertices are used
        if not r.fallback:
            assert len(np.unique(f)) == len(r.vertices), name


def test_the_refined_mesh_is_cached_by_content(scene: dict[str, Path], tmp_path: Path) -> None:
    info = read_pieces(scene["pieces"])["Cuff"]
    cache = RefineCache(tmp_path)
    a = cache.get(info.bundle, MAX_EDGE)
    assert list(tmp_path.glob("*.npz")) and cache.hits == 0
    b = RefineCache(tmp_path).get(info.bundle, MAX_EDGE)  # a new cache object reads the file
    assert np.array_equal(a.vertices, b.vertices) and np.array_equal(a.faces, b.faces)
    assert cache.get(info.bundle, MAX_EDGE) is a and cache.hits == 1
    c = cache.get(info.bundle, 0.02)  # another max edge is another entry
    assert len(c.vertices) < len(a.vertices) and len(list(tmp_path.glob("*.npz"))) == 2


def test_a_concave_outline_is_conforming() -> None:
    # an L shape with a thin neck: Delaunay alone would cut the notch
    outline = [(0, 0), (0.3, 0), (0.3, 0.02), (0.02, 0.02), (0.02, 0.3), (0, 0.3)]
    b = {"outline_m": [list(p) for p in outline], "named_points": [], "bbox_m": {"min": [0, 0], "max": [0.3, 0.3]}}
    r = refine_piece(b, 0.01)
    area, _, _ = tri_stats(r.vertices, r.faces)
    assert area.sum() == pytest.approx(0.3 * 0.02 + 0.28 * 0.02, abs=1e-12) and (area > 0).all()


def test_degenerate_input_is_rejected() -> None:
    from alignment_engine.errors import BundleError

    with pytest.raises(BundleError):
        refine_piece(
            {"outline_m": [[0, 0], [1, 0]], "named_points": [], "bbox_m": {"min": [0, 0], "max": [1, 0]}}, 0.01
        )


def test_a_self_touching_outline_falls_back_to_midpoint_subdivision(refined: dict) -> None:
    fb = [n for n, (_, r) in refined.items() if r.fallback]
    assert fb == ["FullSleeve"], fb  # the known placket slit; every other shirt piece is refined properly
    _, r = refined["FullSleeve"]
    assert np.array_equal(r.vertices[r.original_index, :2], np.array(refined["FullSleeve"][0]["outline_m"])[:, :2])

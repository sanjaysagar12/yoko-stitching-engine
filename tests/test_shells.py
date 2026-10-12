"""Body shells: convex hull cross-sections offset outward (alignment_engine Part B)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.spatial import cKDTree

from alignment_engine.errors import RingNotFoundError
from alignment_engine.loaders import read_avatar
from alignment_engine.shells import Hull, Shells


def square(w: float = 0.2, h: float = 0.1) -> Hull:
    return Hull(np.array([[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]]))


def point_polygon_distance(p: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Distance of points (K,2) to a convex polygon (outside points only: distance to the boundary)."""
    a, b = poly, np.roll(poly, -1, axis=0)
    e = b - a
    t = np.clip(np.einsum("kij,ij->ki", p[:, None, :] - a[None], e) / np.einsum("ij,ij->i", e, e), 0.0, 1.0)
    d = np.linalg.norm(p[:, None, :] - (a[None] + t[..., None] * e[None]), axis=2)
    return np.asarray(d.min(axis=1))


def test_hull_offset_length_is_perimeter_plus_two_pi_h() -> None:
    hull = square()
    assert hull.perimeter == pytest.approx(0.6)
    for h in (0.0, 0.01, 0.03, 0.15):
        assert hull.length(h) == pytest.approx(0.6 + 2 * np.pi * h, abs=1e-12)
        a = np.linspace(0, hull.length(h), 20001)[:-1]
        pos, _ = hull.point_at(a, h)
        # the points are at exactly distance h from the hull, and walking them gives the Steiner length
        assert np.abs(point_polygon_distance(pos, hull.pts) - h).max() < 1e-9
        ring = np.vstack([pos, pos[:1]])
        assert np.linalg.norm(np.diff(ring, axis=0), axis=1).sum() == pytest.approx(hull.length(h), rel=1e-4)


def test_normals_and_angle_to_arc_are_consistent() -> None:
    hull = Hull(np.array([[0.0, 0.0], [0.2, 0.02], [0.25, 0.1], [0.1, 0.16], [-0.02, 0.08]]))
    for h in (0.0, 0.03):
        for phi in np.radians(np.arange(5, 360, 17.0)):
            arc = hull.arc_of_angle(phi, h)
            pos, ang = hull.point_at(np.array([arc]), h)
            if h > 0:  # at h = 0 the normal of a corner point is undefined
                assert np.cos(ang[0] - phi) > 1 - 1e-9  # the normal at that arc has the requested direction
            # the point is the support point of the hull in that direction, pushed out by h
            support = hull.pts[np.argmax(hull.pts @ np.array([np.cos(phi), np.sin(phi)]))]
            assert np.linalg.norm(pos[0] - (support + h * np.array([np.cos(phi), np.sin(phi)]))) < 1e-9


def test_arc_distance_between_two_directions_grows_with_h() -> None:
    hull = square()
    d0 = hull.arc_of_angle(np.pi / 2, 0.0) - hull.arc_of_angle(0.0, 0.0)
    d1 = hull.arc_of_angle(np.pi / 2, 0.05) - hull.arc_of_angle(0.0, 0.05)
    assert d1 - d0 == pytest.approx(0.05 * np.pi / 2, abs=1e-12)  # h times the angle between the two normals


@pytest.fixture(scope="module")
def shells(scene: dict[str, Path]) -> tuple[Shells, np.ndarray, np.ndarray]:
    av = read_avatar(scene["avatar"])
    return Shells.from_avatar(av), av.vertices, av.faces


def test_regions_and_rings(shells: tuple[Shells, np.ndarray, np.ndarray]) -> None:
    sh, v, _ = shells
    assert set(sh.regions) == {"torso", "neck", "arm_L", "arm_R", "leg_L", "leg_R"}
    assert sh.ring_of_bv("Chest") == "torso" and sh.ring_of_bv("Hip") == "torso" and sh.ring_of_bv("Neck") == "neck"
    assert sh.ring_of_bv("UpperArm_L") == "arm_L" and sh.ring_of_bv("Forearm_R") == "arm_R"
    assert sh.ring_of_bv("Thigh_R") == "leg_R" and sh.ring_of_bv("Shin_L") == "leg_L"
    with pytest.raises(RingNotFoundError, match="Head"):
        sh.ring_of_bv("Head")
    t, a = sh.regions["torso"], sh.regions["arm_L"]
    assert np.allclose(t.axis, [0, 1, 0]) and a.axis[0] > 0.9  # the left arm points to +x in the T-pose
    assert v[t.vertex_ids, 0].min() < -0.1 and v[t.vertex_ids, 0].max() > 0.1
    assert len(set(t.vertex_ids) & set(a.vertex_ids)) == 0  # the torso and an arm share no vertex
    assert v[a.vertex_ids, 0].min() > 0.15  # the left arm is on the +x side, beyond the shoulder cap


def test_shell_points_keep_the_clearance_from_their_own_region(shells: tuple[Shells, np.ndarray, np.ndarray]) -> None:
    """The point of Part B: a point on the hull offset by h >= clearance is at least `clearance` from the body
    part it wraps.

    (Other body parts, for example the T-pose arms next to the torso, are handled by the clearance check of
    ``arrange``.)
    """
    sh, v, f = shells
    rng = np.random.default_rng(1)
    for name, reg in sh.regions.items():
        inreg = np.zeros(len(v), dtype=bool)
        inreg[reg.vertex_ids] = True
        tri = v[f[inreg[f].all(axis=1)]]
        samples = np.vstack(
            [v[reg.vertex_ids]]
            + [np.einsum("ij,ijk->ik", rng.dirichlet([1, 1, 1], size=len(tri)), tri) for _ in range(4)]
        )
        tree = cKDTree(samples)
        for h in (0.03, 0.06):
            worst = 1.0
            lo, hi = reg.p_range
            for p in np.linspace(lo + 0.03, hi - 0.03, 25):
                hull, ok = reg.hull_at(p, window=0.04)  # the clearance (3 cm) plus about one triangle edge
                assert ok
                arcs = np.linspace(0, hull.length(h), 90, endpoint=False)
                pos, _ = hull.point_at(arcs, h)
                worst = min(worst, float(tree.query(reg.to_world(p, pos))[0].min()))
            # vertices inside the window are at least h away, those outside at least the window (3 cm) axially
            assert worst >= 0.03 - 0.0015, (name, h, worst)  # samples are about 2-3 mm apart


def test_skin_loop_is_shorter_than_the_hull_and_matches_the_chest(
    shells: tuple[Shells, np.ndarray, np.ndarray],
) -> None:
    sh, _, _ = shells
    t = sh.regions["torso"]
    for p in np.linspace(*t.p_range, 9)[1:-1]:
        hull, _ = t.hull_at(p, window=0.03)
        assert t.skin_loop(p) <= hull.perimeter * 1.001
    # the chest ring of male_173 is 100 cm at the level of the chest ring; the loop at that height is close to it
    loops = [t.skin_loop(p) for p in np.linspace(1.15, 1.3, 16)]
    assert 0.90 < min(loops) / 1.0 and max(loops) / 1.0 < 1.15


def test_hull_cache_and_range_flag(shells: tuple[Shells, np.ndarray, np.ndarray]) -> None:
    sh, _, _ = shells
    t = sh.regions["torso"]
    a, ok_a = t.hull_at(1.2, window=0.03)
    b, ok_b = t.hull_at(1.2004, window=0.03)  # the same 1 mm row
    assert a is b and ok_a and ok_b
    _, ok_hi = t.hull_at(t.p_range[1] + 0.2, window=0.03)
    assert not ok_hi  # above the region: the nearest section is used and the row is flagged


def test_arc_of_ray_lands_on_the_ray_and_grows_with_h() -> None:
    hull = Hull(np.array([[0.0, 0.0], [0.2, 0.02], [0.25, 0.1], [0.1, 0.16], [-0.02, 0.08]]))
    c = hull.centroid
    for h in (0.0, 0.03, 0.12):
        for psi in np.radians(np.arange(3, 360, 23.0)):
            arc = hull.arc_of_ray(c, psi, h)
            pos, _ = hull.point_at(np.array([arc]), h)
            r = pos[0] - c
            assert (
                abs(r[0] * np.sin(psi) - r[1] * np.cos(psi)) < 1e-9 and r @ np.array([np.cos(psi), np.sin(psi)]) > 0
            ), (h, psi)
            # and it is on the offset curve: exactly h from the hull
            assert abs(point_polygon_distance(pos, hull.pts)[0] - h) < 1e-9
    a0 = hull.arc_of_ray(c, 1.0, 0.0), hull.arc_of_ray(c, 2.0, 0.0)
    a1 = hull.arc_of_ray(c, 1.0, 0.05), hull.arc_of_ray(c, 2.0, 0.05)
    assert (a1[1] - a1[0]) > (a0[1] - a0[0])  # the arc between two rays grows with the standoff


def test_hull_stack_matches_the_scalar_hulls() -> None:
    from alignment_engine.stack import HullStack

    hulls = [
        Hull(np.array([[0.0, 0.0], [0.2, 0.02], [0.25, 0.1], [0.1, 0.16], [-0.02, 0.08]])),
        square(0.3, 0.2),
        Hull(np.array([[0.0, 0.0], [0.1, -0.02], [0.2, 0.0], [0.22, 0.1], [0.12, 0.2], [0.0, 0.15]])),
    ]
    st = HullStack(hulls)
    rng = np.random.default_rng(3)
    for h in (0.0, 0.02, 0.1):
        rows = rng.integers(0, 3, 200)
        hh = np.full(200, h)
        arcs = rng.uniform(0, 1.6, 200)
        pos = st.point_at(rows, arcs, hh)
        psi = rng.uniform(0, 2 * np.pi, 200)
        ray = st.arc_of_ray(rows, psi, hh)
        for i in range(200):
            hl = hulls[rows[i]]
            assert np.allclose(pos[i], hl.point_at(np.array([arcs[i]]), h)[0][0], atol=1e-9), (h, i)
            L = hl.length(h)
            if h > 0 or True:
                a = hl.arc_of_ray(hl.centroid, psi[i], h)
                assert abs((ray[i] - a + L / 2) % L - L / 2) < 1e-7, (h, i, ray[i], a)

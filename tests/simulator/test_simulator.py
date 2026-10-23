"""Simulator: flat coordinates of a placed mesh, the simulation mesh (slits, spikes, quality), arc location, the
isometric start, the fabric conversion, the scene of the committed shirt (stitch ends and display meshes on the
pieces), the import rules, and a short solver run (skipped without ZOZO's Contact Solver). The shirt's assembly run and
stitching result come from the stitcher's caches (build/_stitch_runs, made on first use)."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy.spatial import Delaunay

from simulator.errors import SimulationError
from simulator.fabrics import BEND_SCALE, FABRICS, fabric, mesh_size
from simulator.geometry import (
    flat_coords,
    isometric_fit,
    locate2d,
    locate_arc,
    min_angle_deg,
    outline_arcs,
    sim_mesh,
)
from simulator.simulate import Options
from simulator.solver import solver_info
from stitcher.load import ROOT, ensure_run, load_run, read_stitching
from stitcher.result import stitch


def _area(p: Any) -> float:
    return float(abs(0.5 * np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1])))


def _grid_mesh(w: float, hgt: float, n: int) -> tuple[Any, Any, Any]:
    """A flat rectangle: vertices (boundary first is not required), triangles, the outline (counter-clockwise)."""
    xs, ys = np.meshgrid(np.linspace(0, w, n + 1), np.linspace(0, hgt, n + 1))
    v = np.column_stack([xs.ravel(), ys.ravel()])
    f = Delaunay(v).simplices.astype(np.int64)
    outline = np.array([[0, 0], [w, 0], [w, hgt], [0, hgt]], dtype=float)
    return v, f, outline


def _roll(v: Any, r: float) -> Any:
    """Wrap a flat mesh around a vertical cylinder of radius r (isometric: x becomes arc length)."""
    t = v[:, 0] / r
    return np.column_stack([r * np.sin(t), v[:, 1], r * np.cos(t)])


# -- geometry ----------------------------------------------------------------------------------------------------------
def test_flat_coords_recover_a_rolled_piece() -> None:
    v, f, outline = _grid_mesh(0.3, 0.2, 30)
    world = _roll(v, 0.12)
    uv, loop, arcs = flat_coords(world, f, outline, _roll(outline, 0.12))
    assert np.abs(uv - v).max() < 2e-6  # the flat mesh, back to the micrometre
    _, per = outline_arcs(outline)
    assert len(loop) == 120 and np.all(np.diff(arcs) > 0) and arcs.max() < per


def test_flat_coords_walk_a_pinch_vertex_once_per_pass() -> None:
    # two squares touching at one corner: the outline passes that vertex twice (as the shirt sleeve's slit does)
    v = 0.1 * np.array([[0, 0], [1, 0], [1, 1], [0, 1], [2, 1], [2, 2], [1, 2]], dtype=float)
    f = np.array([[0, 1, 2], [0, 2, 3], [2, 4, 5], [2, 5, 6]])
    order = [0, 1, 2, 4, 5, 6, 2, 3]
    world = np.column_stack([v[:, 0], np.zeros(len(v)), v[:, 1]])
    uv, loop, arcs = flat_coords(world, f, v[order], world[order])
    assert loop.tolist() == order and np.abs(uv - v).max() < 1e-12
    assert np.all(np.diff(arcs) > 0)
    with pytest.raises(SimulationError, match="off the placed mesh"):
        flat_coords(world, f, v[order], world[order] + 0.01)


def test_sim_mesh_quality_size_and_boundary_arcs() -> None:
    t = np.linspace(0, np.pi, 40)
    outline = np.vstack(
        [[[0.0, 0.0], [0.4, 0.0]], np.column_stack([0.2 + 0.2 * np.cos(t), 0.3 + 0.2 * np.sin(t)])[::1]]
    )
    v, f, arcs, opened = sim_mesh(outline, 0.02)
    assert opened == 0 and min_angle_deg(v, f) >= 20.0
    u, w = v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]
    tri_area = np.abs(u[:, 0] * w[:, 1] - u[:, 1] * w[:, 0]).sum() / 2
    assert tri_area == pytest.approx(_area(outline), rel=5e-3)
    edges = np.linalg.norm(v[f[:, [1, 2, 0]]] - v[f], axis=2)
    assert edges.max() < 0.02 * 1.6
    _, per = outline_arcs(outline)
    lo, hi, frac = locate_arc(arcs, per, np.array([0.0, 0.2, per - 1e-6]))
    p = v[lo] + frac[:, None] * (v[hi] - v[lo])
    assert np.linalg.norm(p[1] - [0.2, 0.0]) < 1e-9  # arc 0.2 m along the bottom edge


def test_a_slit_drafted_0_1_mm_wide_is_opened_and_a_needle_tip_cut() -> None:
    slit = np.array([[0, 0], [0.1495, 0], [0.15, 0.15], [0.1505, 0], [0.3, 0], [0.3, 0.3], [0, 0.3]], dtype=float)
    v, f, arcs, opened = sim_mesh(slit, 0.02)
    assert opened > 0 and min_angle_deg(v, f) >= 12.0
    b = v[: len(arcs)]
    left, right = (
        b[(b[:, 1] > 0.03) & (b[:, 1] < 0.12) & (b[:, 0] < 0.15) & (b[:, 0] > 0.14)],
        b[(b[:, 1] > 0.03) & (b[:, 1] < 0.12) & (b[:, 0] > 0.15) & (b[:, 0] < 0.16)],
    )
    assert len(left) and len(right) and right[:, 0].min() - left[:, 0].max() >= 0.0039
    needle = np.array([[0.003, 0.007], [0.0, 0.0074], [0.008, 0.0062], [0.2, 0.0], [0.2, 0.04], [0.005, 0.027]])
    v2, f2, _, _ = sim_mesh(needle, 0.015)
    assert min_angle_deg(v2, f2) >= 12.0


def test_locate2d_and_isometric_fit() -> None:
    v, f, _ = _grid_mesh(0.2, 0.1, 10)
    tri, bary = locate2d(v, f, np.array([[0.05, 0.05], [0.25, 0.05]]))
    p = np.einsum("nk,nkd->nd", bary, v[f[tri]])
    assert np.linalg.norm(p[0] - [0.05, 0.05]) < 1e-12 and np.linalg.norm(p[1] - [0.2, 0.05]) < 1e-9
    world = _roll(v, 0.1)
    noisy = world * np.array([1.03, 1.0, 1.03])  # 3 % too wide
    fixed = isometric_fit(noisy, v, f)
    e = np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    err = np.abs(
        np.linalg.norm(fixed[e[:, 0]] - fixed[e[:, 1]], axis=1) / np.linalg.norm(v[e[:, 0]] - v[e[:, 1]], axis=1) - 1
    )
    assert err.max() < 0.004  # 3 % -> under 0.4 % (the start of a real piece is off by 0.05 % at most)


# -- fabrics and options ----------------------------------------------------------------------------------------------
def test_fabrics_convert_measured_units() -> None:
    f = fabric("cotton_shirting")
    p = f.solver_params()
    assert p["density"] == pytest.approx(0.115)
    assert p["young-mod"] * p["density"] == pytest.approx(f.tensile_modulus_n_m)
    assert BEND_SCALE * p["bend"] * p["density"] == pytest.approx(f.bending_rigidity_n_m)
    assert all(0 < x.strain_limit < 0.3 for x in FABRICS.values())
    assert (
        FABRICS["denim"].bending_rigidity_n_m
        > FABRICS["cotton_shirting"].bending_rigidity_n_m
        > FABRICS["silk"].bending_rigidity_n_m
    )
    assert mesh_size("fine") < mesh_size("standard") < mesh_size("draft")
    with pytest.raises(SimulationError, match="unknown fabric"):
        fabric("tweed")


def test_options_are_validated() -> None:
    assert Options.of(None) == Options()
    with pytest.raises(SimulationError, match="unknown simulation options"):
        Options.of({"speed": 2})
    with pytest.raises(SimulationError, match="frames"):
        Options.of({"frames": 3})


# -- the committed shirt -----------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def shirt(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Any, dict[str, Any]]:
    st_path = ROOT / "assert" / "stitchings" / "male_shirt.stitching.json"
    st = read_stitching(st_path)
    asm = ROOT / st["assembly"]
    run = load_run(ensure_run(asm, True))
    res = stitch(st, st_path, asm, run)
    p = tmp_path_factory.mktemp("sim") / "stitching.result.json"
    p.write_text(json.dumps(res), encoding="utf-8")
    return p, run, res


def test_shirt_scene_puts_stitches_and_display_meshes_on_the_pieces(shirt: tuple[Path, Any, dict[str, Any]]) -> None:
    from simulator.scene import build

    _, run, res = shirt
    sc = build(res, run, mesh_size("standard"), fabric(None), cache=ROOT / "build" / "_sim_flat")
    v = sc.vertices
    assert len(sc.pieces) == 20 and 4000 < len(v) < 9000
    # every piece starts at its pattern shape, where the alignment engine placed it
    assert max(p.flat_error_mm for p in sc.pieces) < 0.1
    pa = np.einsum("kj,kjd->kd", sc.stitch_w[:, :3], v[sc.stitch_ind[:, :3]])
    pb = np.einsum("kj,kjd->kd", sc.stitch_w[:, 3:], v[sc.stitch_ind[:, 3:]])
    for k, s in enumerate(res["seams"]):
        m = sc.stitch_seam == k
        apart = np.linalg.norm(np.subtract(s["samples"]["a_xyz_m"], s["samples"]["b_xyz_m"]), axis=1) > 0.002
        assert apart.sum() <= m.sum() <= s["samples"]["count"]  # only pairs that already meet are left out
        assert (np.linalg.norm(pa[m] - pb[m], axis=1) > 1e-4).all()  # the solver's spring needs a direction
        for side, p in (("a", pa), ("b", pb)):
            near = np.min(
                np.linalg.norm(p[m][:, None] - np.asarray(s["samples"][f"{side}_xyz_m"])[None], axis=2), axis=1
            )
            assert near.max() < 0.005, (s["id"], side, near.max())  # within the 4 mm a slit is opened by, plus chords
    for p in sc.pieces:
        placed = np.asarray(run.placements[p.name]["wrapped_vertices"])
        shown = np.einsum("nk,nkd->nd", p.display_bary, p.world[p.faces[p.display_tri]])
        err = np.linalg.norm(shown - placed, axis=1)
        assert np.quantile(err, 0.99) < 0.002 and err.max() < 0.006, (p.name, err.max())
    assert {p.name for p in sc.pieces if p.interfaced} == {
        "CollarBase", "CollarBase_L", "CollarTop", "CollarTop_L", "Cuff", "Cuff_L"
    }  # fmt: skip


def test_simulator_imports_no_engine() -> None:
    allowed = {"assembler.render", "assembler.verify"}
    bad = []
    for py in (ROOT / "src" / "simulator").rglob("*.py"):
        for node in ast.walk(ast.parse(py.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                mods = [node.module]
            else:
                continue
            bad += [f"{py.name}: {m}" for m in mods
                    if m.split(".")[0] in {"sm2d_engine", "avatar_engine", "alignment_engine", "web"}
                    or (m.split(".")[0] == "assembler" and m not in allowed)]  # fmt: skip
    runner = (ROOT / "src" / "simulator" / "ppf_runner.py").read_text(encoding="utf-8")
    assert "from stitcher" not in runner and "from ." not in runner  # it runs in the solver's own Python
    assert not bad, bad


# -- the solver --------------------------------------------------------------------------------------------------------
needs_solver = pytest.mark.skipif(not solver_info()["available"], reason="ZOZO's Contact Solver is not installed")


@needs_solver
def test_short_simulation_sews_the_shirt(shirt: tuple[Path, Any, dict[str, Any]], tmp_path: Path) -> None:
    from simulator.simulate import simulate

    res_path, run, _ = shirt
    events: list[dict[str, Any]] = []
    doc = simulate(res_path, run, tmp_path, Options(quality="draft", frames=12, sew_time=0.2), events.append,
                   render=False)  # fmt: skip
    assert doc["run"]["finished"], doc["log"][-10:]
    first, last = doc["closure"][0]["gap_max_m"], doc["closure"][-1]["gap_max_m"]
    assert first > 0.4 and last < 0.03  # 49 cm apart at the start, sewn within 12 frames
    assert any(e.get("event") == "progress" for e in events)
    draped = np.load(tmp_path / "draped.npz")
    for name in ("FrontPanel", "FullSleeve_L", "Pocket"):
        assert draped[name].shape == (len(run.placements[name]["wrapped_vertices"]), 3)
    assert (tmp_path / "draped_m.obj").is_file()
    rows = {(r["id"], r.get("seam")): r for r in doc["checks"]}
    assert rows[("body.penetration", None)]["status"] == "pass"

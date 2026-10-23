"""The simulation scene: a stitching result and its assembly run -> one cloth mesh, its stitches, the body.

* Pieces: every piece a seam names. Its placed mesh (the alignment result's ``wrapped_vertices`` / ``wrapped_faces``,
  5 mm triangles: the display mesh) gets its flat coordinates (``geometry.flat_coords``); the piece is meshed again
  in the flat for the solver (``geometry.sim_mesh``, edges of ``h``) and every simulation vertex is put at the world
  point of its flat position on the placed mesh. The cloth therefore starts exactly where the alignment engine put
  the piece, at its pattern size.
* Display: every vertex of the placed mesh is tied to the simulation mesh (triangle and barycentric weights in the
  flat; outline vertices by their outline arc), so the simulated garment is shown and exported on the placed meshes,
  vertex for vertex: the viewer swaps vertex positions, faces and named points stay.
* Stitches: every correspondence of the stitching result (``samples``) becomes one row joining a point of side a to
  a point of side b, each a triangle of the simulation mesh with barycentric weights (the solver's six-column stitch).
  Outline points are found by their outline arc (``chain``, ``direction``, ``trim_m`` and ``a_arc_m``), never by
  position; dart legs (``line``) by their flat point; attach seams by the host triangle the stitcher found.
* Body: the avatar OBJ of the run, a fixed collider.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from stitcher.load import Run

from .errors import SimulationError
from .fabrics import Fabric
from .geometry import (
    F64,
    I64,
    edge_lengths,
    flat_coords,
    isometric_fit,
    locate2d,
    locate_arc,
    outline_arcs,
    sim_mesh,
)

H_MIN_M = 0.006  # the simulation mesh of a narrow piece (a placket) is finer, never below this
MEET_M = 1e-4  # the two ends of a stitch pair closer than this at the start are one point (already sewn)
FOLD_FROM = 0.35  # a dart's fold line starts this far from the apex towards the outline
STRIPS = 4  # a piece is at least this many triangle edges across (2 * area / perimeter)


@dataclass
class PieceScene:
    """One piece in the simulation: its simulation mesh (flat and world) and its display mesh tied to it."""

    name: str
    first: int  # first vertex in the garment mesh
    uv: F64  # (n,2) flat
    world: F64  # (n,3) initial position
    faces: I64  # (m,3) local indices
    arcs: F64  # outline arc of the first len(arcs) vertices (the boundary loop)
    per: float
    s_of: dict[str, float]
    h: float
    opened: int
    interfaced: bool
    display_tri: I64  # (N,) triangle of ``faces`` of every placed-mesh vertex
    display_bary: F64  # (N,3)
    flat_error_mm: float
    boundary_tri: dict[tuple[int, int], int] = field(default_factory=dict)

    @property
    def count(self) -> int:
        return len(self.uv)

    def on_arc(self, s: F64) -> tuple[I64, F64]:
        """Global triangle vertex indices (k,3) and weights (k,3) of outline points at arcs ``s``."""
        lo, hi, t = locate_arc(self.arcs, self.per, s)
        tri = np.array([self.boundary_tri[(min(a, b), max(a, b))] for a, b in zip(lo, hi, strict=True)])
        f = self.faces[tri]
        w = np.zeros((len(s), 3))
        for k in range(3):
            w[:, k] = np.where(f[:, k] == lo, 1.0 - t, 0.0) + np.where(f[:, k] == hi, t, 0.0)
        return f + self.first, w

    def at_uv(self, uv: F64) -> tuple[I64, F64]:
        """Global triangle vertex indices and weights of flat points (inside the piece)."""
        tri, bary = locate2d(self.uv, self.faces, uv)
        return self.faces[tri] + self.first, bary


@dataclass
class Scene:
    """Everything the solver needs (``arrays``) and what the result is mapped back with (``pieces``)."""

    pieces: list[PieceScene]
    stitch_ind: I64  # (K,6)
    stitch_w: F64  # (K,6)
    stitch_seam: I64  # (K,) index into ``seams``
    seams: list[dict[str, Any]]
    body_v: F64
    body_f: I64
    report: list[str]

    @property
    def vertices(self) -> F64:
        return np.vstack([p.world for p in self.pieces])

    @property
    def faces(self) -> I64:
        return np.vstack([p.faces + p.first for p in self.pieces])

    @property
    def uv(self) -> F64:
        return np.vstack([p.uv for p in self.pieces])

    @property
    def stiff(self) -> F64:
        return np.concatenate([np.full(p.count, 1.0 if p.interfaced else 0.0) for p in self.pieces])


def _hash(*arrays: Any) -> str:
    h = hashlib.sha1()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:20]


def placed_flat(name: str, run: Run, cache: Path | None) -> tuple[F64, I64, F64, I64, F64]:
    """(world vertices, faces, flat coordinates, boundary loop, loop arcs) of a placed piece (cached by content)."""
    p, b = run.placements[name], run.bundles[name]
    if p.get("wrapped_vertices") is None:
        raise SimulationError(f"{name}: the alignment result has no placed mesh (wrapped_vertices)")
    v = np.asarray(p["wrapped_vertices"], dtype=np.float64)
    f = np.asarray(p["wrapped_faces"], dtype=np.int64)
    flat = np.asarray(b["outline_m"], dtype=np.float64)
    ow = np.asarray(p["world_outline_m"], dtype=np.float64)
    path = cache / f"flat-{_hash(v, f, flat, ow)}.npz" if cache else None
    if path is not None and path.is_file():
        z = np.load(path)
        return v, f, z["uv"], z["loop"], z["arcs"]
    uv, loop, arcs = flat_coords(v, f, flat, ow)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, uv=uv, loop=loop, arcs=arcs)
    return v, f, uv, loop, arcs


def interfaced(name: str, run: Run) -> bool:
    """A piece is interfaced when the pattern has a ``<piece>Interface`` piece for it (or for its source)."""
    b = run.bundles[name]
    src = b.get("mirror_of") or b.get("clone_of") or name
    return f"{src}Interface" in run.bundles or f"{name}Interface" in run.bundles


def piece_h(outline: F64, h: float) -> float:
    """The simulation mesh size of a piece: ``h``, finer for a narrow piece (``STRIPS`` edges across)."""
    area = abs(0.5 * np.sum(outline[:, 0] * np.roll(outline[:, 1], -1) - np.roll(outline[:, 0], -1) * outline[:, 1]))
    _, per = outline_arcs(outline)
    width = 2.0 * area / max(per, 1e-9)
    return float(np.clip(min(h, width * 2.0 / STRIPS), H_MIN_M, h))


Flat = tuple[F64, I64, F64, I64, F64]  # placed_flat of a piece


def inner_lines(result: dict[str, Any], name: str, bundle: dict[str, Any]) -> list[F64]:
    """The lines inside piece ``name`` the simulation mesh must follow: every ``line`` part of a seam (a dart leg drawn
    on the piece) and, for two legs meeting at an apex, the dart's fold line (apex -> the middle of the two other ends,
    on to the outline, from ``FOLD_FROM`` of the way out), so the wedge between the legs can fold into a flap when they
    are sewn, as a real dart does."""
    from shapely.geometry import LineString, Point, Polygon

    xy = {q["name"]: np.asarray(q["xy_m"], dtype=np.float64) for q in bundle["named_points"]}
    out: list[F64] = []
    for s in result["seams"]:
        legs = []
        for side in ("a", "b"):
            for part in s[side]["parts"]:
                if part["piece"] == name and part["direction"] == "line" and all(n in xy for n in part["chain"]):
                    legs.append(np.array([xy[n] for n in part["chain"]]))
        out += legs
        if len(legs) == 2 and np.linalg.norm(legs[0][-1] - legs[1][-1]) < 1e-6:  # a dart: both legs end at the apex
            apex, mid = legs[0][-1], 0.5 * (legs[0][0] + legs[1][0])
            d = mid - apex
            ray = LineString([apex, apex + d * 10.0])
            hit = ray.intersection(Polygon(bundle["outline_m"]).exterior)
            pts = [np.array(g.coords[0]) for g in getattr(hit, "geoms", [hit]) if not g.is_empty]
            end = min(pts, key=lambda p: float(np.linalg.norm(p - apex)), default=mid)
            if Point(end).distance(Point(apex)) > 1e-4:  # not at the apex itself: its 3-degree triangles lock the cloth
                out.append(np.array([apex + FOLD_FROM * (end - apex), end]))
    return out


def build_piece(name: str, run: Run, h: float, first: int, flat: Flat, lines: list[F64] | None = None) -> PieceScene:
    b = run.bundles[name]
    v, f, uv, loop, larc = flat
    outline = np.asarray(b["outline_m"], dtype=np.float64)
    hp = piece_h(outline, h)
    suv, sf, arcs, opened = sim_mesh(outline, hp, lines)
    # world of every simulation vertex: its flat point on the placed mesh, then every triangle at its pattern shape
    tri, bary = locate2d(uv, f, suv)
    sw = isometric_fit(np.einsum("nk,nkd->nd", bary, v[f[tri]]), suv, sf)
    # faces wound like the placed mesh (its normals point away from the body)
    t = uv[f]
    sa = (t[:, 1, 0] - t[:, 0, 0]) * (t[:, 2, 1] - t[:, 0, 1]) - (t[:, 1, 1] - t[:, 0, 1]) * (t[:, 2, 0] - t[:, 0, 0])
    if np.sum(sa < 0) > np.sum(sa > 0):
        sf = sf[:, [0, 2, 1]]
    _, per = outline_arcs(outline)
    nb = len(arcs)
    btri: dict[tuple[int, int], int] = {}
    for k, tr in enumerate(sf.tolist()):
        for i, j in ((tr[0], tr[1]), (tr[1], tr[2]), (tr[2], tr[0])):
            if i < nb and j < nb and ((j - i) % nb in (1, nb - 1)):
                btri[(min(i, j), max(i, j))] = k
    ps = PieceScene(
        name, first, suv, sw, sf, arcs, per,
        {q["name"]: float(q["outline_s_m"]) % per for q in b["named_points"] if q["on_outline"]},
        hp, opened, False, np.zeros(0, dtype=np.int64), np.zeros((0, 3)), 0.0, btri,
    )  # fmt: skip
    # display mesh tied to the simulation mesh: outline vertices by arc, the others by flat position
    dt, db = locate2d(suv, sf, uv)
    seen: set[int] = set()
    for vi, s in zip(loop.tolist(), larc.tolist(), strict=True):
        if vi in seen:  # a pinch vertex: its first pass
            continue
        seen.add(vi)
        lo, hi, tt = locate_arc(arcs, per, np.array([s]))
        k = btri[(min(int(lo[0]), int(hi[0])), max(int(lo[0]), int(hi[0])))]
        dt[vi] = k
        db[vi] = [(1.0 - tt[0]) if x == lo[0] else (tt[0] if x == hi[0] else 0.0) for x in sf[k]]
    ps.display_tri, ps.display_bary = dt, db
    # how far the start shape is from the pattern (rest lengths are taken from it)
    e, lw = edge_lengths(sw, sf)
    lf = np.linalg.norm(suv[e[:, 0]] - suv[e[:, 1]], axis=1)
    ps.flat_error_mm = float(np.abs(lw - lf).max() * 1000.0)
    return ps


def seam_pieces(result: dict[str, Any]) -> list[str]:
    """Every piece a seam of the stitching result names, in first-seen order."""
    out: list[str] = []
    for s in result["seams"]:
        for side in ("a", "b"):
            for part in s[side]["parts"]:
                if part["piece"] not in out:
                    out.append(part["piece"])
    return out


def _side_points(
    seam: dict[str, Any], side: str, pieces: dict[str, PieceScene], flats: dict[str, Flat]
) -> tuple[I64, F64]:
    """(K,3) global vertex indices and (K,3) weights of every correspondence of one side of a seam."""
    smp = seam["samples"]
    n = smp["count"]
    parts = seam[side]["parts"]
    ind = np.zeros((n, 3), dtype=np.int64)
    w = np.zeros((n, 3))
    if parts[0]["direction"] == "onto":  # attach: the host triangle the stitcher found under every sample of side a
        host = parts[0]["piece"]
        _, hf, uv, _, _ = flats[host]
        tri = np.asarray(smp["b_tri"], dtype=np.int64)
        bary = np.asarray(smp["b_bary"], dtype=np.float64)
        return pieces[host].at_uv(np.einsum("nk,nkd->nd", bary, uv[hf[tri]]))
    kk = np.asarray(smp[f"{side}_part"], dtype=np.int64)
    arc = np.asarray(smp[f"{side}_arc_m"], dtype=np.float64)
    uvs = smp[f"{side}_uv_m"]
    for k, part in enumerate(parts):
        m = kk == k
        if not m.any():
            continue
        ps = pieces[part["piece"]]
        if part["direction"] == "line":
            ind[m], w[m] = ps.at_uv(np.asarray(uvs, dtype=np.float64)[m])
            continue
        d = 1.0 if part["direction"] == "outline" else -1.0
        s = ps.s_of[part["chain"][0]] + d * (float(part["trim_m"][0]) + arc[m])
        ind[m], w[m] = ps.on_arc(s)
    return ind, w


def build(result: dict[str, Any], run: Run, h: float, fab: Fabric, cache: Path | None = None) -> Scene:
    """The scene of a stitching result (see the module docstring)."""
    report: list[str] = []
    names = seam_pieces(result)
    missing = [n for n in names if n not in run.placements]
    if missing:
        raise SimulationError(f"pieces of the stitching result are not placed in the run: {missing}")
    unsewn = sorted(set(run.placements) - set(names))
    if unsewn:
        report.append(f"[INFO] placed but in no seam (left out of the simulation): {unsewn}")
    pieces: dict[str, PieceScene] = {}
    flats: dict[str, Flat] = {}
    first = 0
    for n in names:
        flats[n] = placed_flat(n, run, cache)
        ps = build_piece(n, run, h, first, flats[n], inner_lines(result, n, run.bundles[n]))
        ps.interfaced = interfaced(n, run)
        pieces[n] = ps
        first += ps.count
    v = np.vstack([p.world for p in pieces.values()])
    rows_i, rows_w, rows_s, seams = [], [], [], []
    dropped = 0
    for k, s in enumerate(result["seams"]):
        ia, wa = _side_points(s, "a", pieces, flats)
        ib, wb = _side_points(s, "b", pieces, flats)
        # a pair whose ends already meet (the two legs of a dart at its apex) is sewn: the solver's stitch spring has
        # no direction there and would give a NaN force
        gap = np.linalg.norm(np.einsum("kj,kjd->kd", wa, v[ia]) - np.einsum("kj,kjd->kd", wb, v[ib]), axis=1)
        keep = gap > MEET_M
        dropped += int((~keep).sum())
        rows_i.append(np.hstack([ia, ib])[keep])
        rows_w.append(np.hstack([wa, wb])[keep])
        rows_s.append(np.full(int(keep.sum()), k))
        seams.append({"id": s["id"], "kind": s["kind"], "pairs": int(keep.sum()), "gap_max_m": s["world_gap_m"]["max"]})
    bv, bf = run.avatar()
    sc = Scene(
        list(pieces.values()), np.vstack(rows_i), np.vstack(rows_w), np.concatenate(rows_s), seams,
        np.asarray(bv, dtype=np.float64), np.asarray(bf, dtype=np.int64), report,
    )  # fmt: skip
    worst = max(sc.pieces, key=lambda p: p.flat_error_mm)
    report.append(
        f"[INFO] simulation mesh: {len(sc.vertices)} vertices, {len(sc.faces)} triangles, {len(sc.stitch_ind)} stitch "
        f"rows; start shape within {worst.flat_error_mm:.2f} mm of the pattern (largest: {worst.name})"
    )
    if dropped:
        report.append(f"[INFO] {dropped} stitch pair(s) already meet at the start (dart apexes): left out")
    opened = [f"{p.name} ({p.opened})" for p in sc.pieces if p.opened]
    if opened:
        report.append(f"[INFO] slits narrower than 4 mm opened to 4 mm in the simulation mesh: {', '.join(opened)}")
    stiff = [p.name for p in sc.pieces if p.interfaced]
    if stiff:
        report.append(f"[INFO] interfaced (stiffer in bending x{fab.interfaced_bend_factor:g}): {', '.join(stiff)}")
    return sc


def save(sc: Scene, folder: Path) -> Path:
    """``scene.npz`` (the solver's input) and ``scene.json`` (what the arrays mean) in ``folder``."""
    folder.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, NDArray[Any]] = {
        "V": sc.vertices, "F": sc.faces, "UV": sc.uv, "STIFF": sc.stiff,
        "SI": sc.stitch_ind, "SW": sc.stitch_w, "SEAM": sc.stitch_seam, "BV": sc.body_v, "BF": sc.body_f,
    }  # fmt: skip
    for p in sc.pieces:
        arrays[f"dtri:{p.name}"] = p.display_tri
        arrays[f"dbary:{p.name}"] = p.display_bary
    np.savez_compressed(folder / "scene.npz", **arrays)  # type: ignore[arg-type]
    doc = {
        "pieces": [
            {
                "name": p.name, "first": p.first, "count": p.count, "faces": len(p.faces), "h_m": round(p.h, 4),
                "interfaced": p.interfaced, "opened": p.opened, "display_vertices": len(p.display_tri),
                "start_error_mm": round(p.flat_error_mm, 3),
            }
            for p in sc.pieces
        ],
        "seams": sc.seams,
        "report": sc.report,
    }  # fmt: skip
    (folder / "scene.json").write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8", newline="\n")
    return folder / "scene.npz"

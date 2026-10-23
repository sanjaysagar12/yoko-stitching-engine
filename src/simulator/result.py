"""The simulation result: the solver's frames mapped back onto the placed pieces, the checks, the result file.

``simulation.result.json`` (contracts/simulation_result.schema.json) lists the inputs (stitching result, run,
avatar, with their hashes), the fabric in measured and in solver units, the run (frames, timing, seam closure per
frame) and the checks; ``draped.npz`` holds per piece the draped position of every vertex of its placed mesh (the
alignment result's ``wrapped_vertices`` order, metres, world frame), and ``frames.npz`` the simulation mesh of the
kept frames. The embedding that carries the placed meshes (``scene.npz``: ``dtri:<piece>``, ``dbary:<piece>``) is
what the viewer uses to animate them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .geometry import F64, I64, edge_lengths

SCHEMA_VERSION = "1.0"
GAP_FAIL_M = 0.01  # a stitch still open by more than 1 cm at the end: the seam did not close
GAP_WARN_M = 0.005
INSIDE_M = 0.002  # a simulated vertex more than 2 mm inside the body (the solver's contact keeps it outside)
STRETCH_TOL = 0.005  # stretch allowed above the fabric's strain limit (the limit is enforced per triangle)
TIGHT_MARGIN = 0.003  # stretch within this of the strain limit: the cloth cannot give any more
WINDING_INSIDE = 0.8  # inside: winding number about 1; near an opening of the body mesh it is fractional


class SceneArrays:
    """``scene.npz`` + ``scene.json`` of a simulation folder."""

    def __init__(self, folder: Path) -> None:
        self.folder = Path(folder)
        self.z = np.load(self.folder / "scene.npz")
        self.doc: dict[str, Any] = json.loads((self.folder / "scene.json").read_text(encoding="utf-8"))
        self.V: F64 = self.z["V"]
        self.F: I64 = self.z["F"]
        self.UV: F64 = self.z["UV"]
        self.SI: I64 = self.z["SI"]
        self.SW: F64 = self.z["SW"]
        self.SEAM: I64 = self.z["SEAM"]
        self.body = (self.z["BV"], self.z["BF"])
        off = np.cumsum([0] + [p["faces"] for p in self.doc["pieces"]])
        self.face_first = {p["name"]: int(o) for p, o in zip(self.doc["pieces"], off[:-1], strict=True)}

    def embedding(self, name: str) -> tuple[I64, F64]:
        """(N,3) simulation vertex indices and (N,3) weights of every vertex of the piece's placed mesh."""
        tri = self.z[f"dtri:{name}"] + self.face_first[name]
        return self.F[tri], self.z[f"dbary:{name}"]

    def display(self, x: F64) -> dict[str, F64]:
        """Every placed mesh at the simulation positions ``x`` (vertex for vertex)."""
        out = {}
        for p in self.doc["pieces"]:
            idx, w = self.embedding(p["name"])
            out[p["name"]] = np.einsum("nk,nkd->nd", w, x[idx])
        return out

    def stitch_gaps(self, x: F64) -> F64:
        pa = np.einsum("kj,kjd->kd", self.SW[:, :3], x[self.SI[:, :3]])
        pb = np.einsum("kj,kjd->kd", self.SW[:, 3:], x[self.SI[:, 3:]])
        return np.asarray(np.linalg.norm(pa - pb, axis=1), dtype=np.float64)


def winding(p: F64, v: F64, f: I64, chunk: int = 2_000_000) -> F64:
    """Generalised winding number of the mesh ``v, f`` at points ``p`` (Jacobson et al. 2013): about +-1 inside, 0
    outside, also for a body made of overlapping closed parts (the synthetic avatar), where the side of the nearest
    triangle is not the inside."""
    a0, b0, c0 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    out = np.zeros(len(p))
    step = max(1, chunk // max(len(f), 1))
    for s in range(0, len(p), step):
        q = p[s : s + step, None, :]
        a, b, c = a0[None] - q, b0[None] - q, c0[None] - q
        la, lb, lc = np.linalg.norm(a, axis=2), np.linalg.norm(b, axis=2), np.linalg.norm(c, axis=2)
        det = np.einsum("pti,pti->pt", a, np.cross(b, c))
        den = la * lb * lc + np.einsum("pti,pti->pt", a, b) * lc + np.einsum("pti,pti->pt", b, c) * la
        den += np.einsum("pti,pti->pt", c, a) * lb
        out[s : s + step] = 2.0 * np.arctan2(det, den).sum(axis=1) / (4.0 * np.pi)
    return out


def _row(rows: list[dict[str, Any]], rid: str, status: str, value: Any, hint: str = "", **extra: Any) -> None:
    rows.append({"id": rid, "status": status, "value": value, "hint": hint, **extra})


def check(sa: SceneArrays, x: F64, done: dict[str, Any], strain_limit: float) -> list[dict[str, Any]]:
    """The checks of the final frame ``x`` (simulation mesh, metres)."""
    from assembler.verify import Surf

    rows: list[dict[str, Any]] = []
    fin = bool(done.get("finished"))
    _row(rows, "solver.finished", "pass" if fin else "fail", fin,
         "" if fin else "the solver stopped before the last frame (see the log)")  # fmt: skip
    ok_num = bool(np.isfinite(x).all())
    _row(rows, "result.finite", "pass" if ok_num else "fail", ok_num)
    if not ok_num:
        return rows
    e, now = edge_lengths(x, sa.F)
    rest = np.linalg.norm(sa.UV[e[:, 0]] - sa.UV[e[:, 1]], axis=1)  # the pattern (flat) lengths
    strain = now / np.maximum(rest, 1e-12) - 1.0
    smax = float(strain.max())
    tight = smax >= strain_limit - TIGHT_MARGIN  # the cloth is as stretched as the fabric allows
    g = sa.stitch_gaps(x)
    for k, s in enumerate(sa.doc["seams"]):
        m = sa.SEAM == k
        gm = float(g[m].max()) if m.any() else 0.0
        st = "fail" if gm > GAP_FAIL_M else ("warn" if gm > GAP_WARN_M else "pass")
        hint = "" if st == "pass" else "the seam is still open (cm)"
        if st != "pass" and tight:
            hint += "; the fabric is at its strain limit: the garment is too small for this body to close"
        _row(rows, "seam.closed", st, round(gm * 100, 2), hint, seam=s["id"], start_gap_cm=round(s["gap_max_m"] * 100, 1))
    bv, bf = sa.body
    dist, _, _ = Surf(bv, bf).closest(x)
    near = np.flatnonzero(dist < 0.05)  # only points near the skin can be inside it
    w = np.zeros(len(x))
    if len(near):
        w[near] = winding(x[near], bv, bf)
    inside_m = (np.abs(w) > WINDING_INSIDE) & (dist > INSIDE_M)
    inside = int(inside_m.sum())
    _row(rows, "body.penetration", "pass" if inside == 0 else "fail", inside,
         f"vertices more than {INSIDE_M * 1000:g} mm inside the body (winding number); closest to the skin "
         f"{dist.min() * 1000:.1f} mm, median {np.median(dist) * 1000:.1f} mm")  # fmt: skip
    st = "pass" if smax <= strain_limit + STRETCH_TOL else "fail"
    _row(rows, "fabric.stretch", st, round(smax * 100, 2),
         f"edges vs the pattern: max {smax * 100:+.2f} %, 99 % {np.quantile(strain, 0.99) * 100:+.2f} %, median "
         f"{np.median(strain) * 100:+.2f} % (strain limit {strain_limit * 100:g} %)")  # fmt: skip
    _row(rows, "fabric.compression", "info", round(float(strain.min()) * 100, 1),
         f"shortest edge vs the pattern {strain.min() * 100:+.1f} %, 1 % {np.quantile(strain, 0.01) * 100:+.1f} % "
         "(gathers and folds compress the cloth)")  # fmt: skip
    lo = float(x[:, 1].min())
    _row(rows, "garment.above_floor", "pass" if lo > 0 else "fail", round(lo, 4), "lowest garment point (m)")
    return rows


def verdict(rows: list[dict[str, Any]]) -> bool:
    return not any(r["status"] == "fail" for r in rows)


def summary(sa: SceneArrays, x: F64, rows: list[dict[str, Any]], done: dict[str, Any]) -> dict[str, Any]:
    g = sa.stitch_gaps(x) if np.isfinite(x).all() else np.array([np.inf])
    stretch = next((r for r in rows if r["id"] == "fabric.stretch"), None)
    pen = next((r for r in rows if r["id"] == "body.penetration"), None)
    return {
        "pass": verdict(rows),
        "finished": bool(done.get("finished")),
        "frames": int(done.get("frames") or 0),
        "wall_s": done.get("wall_s"),
        "seam_gap_max_cm": round(float(g.max()) * 100, 2),
        "seam_gap_mean_cm": round(float(g.mean()) * 100, 3),
        "seams_open": sum(1 for r in rows if r["id"] == "seam.closed" and r["status"] == "fail"),
        "stretch_max_pct": stretch["value"] if stretch else None,
        "inside_body": pen["value"] if pen else None,
        "sim_vertices": int(len(sa.V)),
        "sim_triangles": int(len(sa.F)),
        "fails": [f"{r['id']} {r.get('seam', '')}: {r['hint'] or r['value']}".strip() for r in rows
                  if r["status"] == "fail"],
    }  # fmt: skip


def write_draped(sa: SceneArrays, x: F64, folder: Path) -> Path:
    disp = sa.display(x)
    path = Path(folder) / "draped.npz"
    np.savez_compressed(path, **{k: v.astype(np.float32) for k, v in disp.items()})  # type: ignore[arg-type]
    return path


def write_obj(path: Path, meshes: list[tuple[str, F64, NDArray[np.int64]]]) -> Path:
    """One OBJ with a group per mesh (metres, y up)."""
    lines = ["# simulated garment + avatar, metres, y up, +z front, +x the wearer's left"]
    base = 1
    for name, v, f in meshes:
        lines.append(f"g {name}")
        lines += [f"v {a:.6f} {b:.6f} {c:.6f}" for a, b, c in v]
        lines += [f"f {i + base} {j + base} {k + base}" for i, j, k in f]
        base += len(v)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return Path(path)

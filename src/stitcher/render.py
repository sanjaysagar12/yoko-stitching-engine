"""Stitch images on the CPU renderer of ``assembler.render`` (numpy z-buffer + Pillow, deterministic pixels).

* ``stitch_<view>.png``: avatar and pieces (pale) with every seam drawn on both pieces in its colour, thin lines
  between every few correspondences, the seam id; parts hidden behind a surface are drawn thin and pale.
* ``stitch_seam_<id>.png``: the seam close up (seen from outside), plus the flat pieces with the chains, a dot at
  s = 0 and a square at s = 1 on both sides, and the numbers (lengths, ratio, ease, gap, crossings, verdict).
* ``stitch_unstitched.png``: every piece flat, each outline edge in the colour of its seam, unstitched edges thick red,
  ignored edges grey.
* ``stitch_sheet.png``: the views and the flat map.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

import assembler.render as R

from .chains import F64, Piece

SEAM_COLOURS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b", "#e377c2", "#17becf", "#bcbd22",
    "#3b4cc0", "#d35400", "#16a085", "#8e44ad", "#c0392b", "#2c3e50", "#7f8c00", "#b8860b",
]  # fmt: skip
RED = (214, 39, 40)
IGNORED = (150, 150, 150)
VIEWS = [
    R.View("front", R._dir(0)),
    R.View("back", R._dir(180)),
    R.View("left", R._dir(90)),
    R.View("right", R._dir(-90)),
    R.View("top", (0.0, 1.0, 0.0), (0.0, 0.0, -1.0)),
    R.View("three_quarter", R._dir(35, 15)),
]


def seam_colour(i: int) -> tuple[int, int, int]:
    return R._hex(SEAM_COLOURS[i % len(SEAM_COLOURS)])


def _pale(c: tuple[int, int, int], k: float = 0.65) -> tuple[int, int, int]:
    return tuple(int(x + (255 - x) * k) for x in c)  # type: ignore[return-value]


def _scene(run: Any) -> R.Scene:
    v, f = run.avatar()
    av = R.Mesh("avatar", v, f, R.AVATAR)
    meshes = []
    for i, p in enumerate(run.result["placements"]):
        pv = np.asarray(p["wrapped_vertices"], dtype=np.float64)
        pf = np.asarray(p["wrapped_faces"], dtype=np.int64)
        meshes.append(R.Mesh(p["piece"], pv, pf, _pale(R.piece_colour(i)), label=False))
    return R.Scene(av, meshes)


class _Proj:
    def __init__(self, view: R.View, fr: R.Frame, depth: F64) -> None:
        self.r, self.u, self.b = view.basis()
        self.fr, self.depth = fr, depth
        self.sx = (fr.width - 1) / (fr.hi[0] - fr.lo[0])
        self.sy = (fr.height - 1) / (fr.hi[1] - fr.lo[1])

    def xy(self, p: F64) -> F64:
        return np.column_stack([(p @ self.r - self.fr.lo[0]) * self.sx, (self.fr.hi[1] - p @ self.u) * self.sy])

    def visible(self, p: F64, eps: float = 0.012) -> F64:
        q = self.xy(p)
        x = np.clip(np.round(q[:, 0]).astype(int), 0, self.fr.width - 1)
        y = np.clip(np.round(q[:, 1]).astype(int), 0, self.fr.height - 1)
        d = self.depth[y, x]
        return np.asarray((p @ self.b >= d - eps) | ~np.isfinite(d))


def _line(d: ImageDraw.ImageDraw, pj: _Proj, p: F64, col: tuple[int, int, int], width: int) -> None:
    """A world polyline: visible parts in ``col`` at ``width``, hidden parts thin and pale."""
    if len(p) < 2:
        return
    # densify so visibility is judged every ~5 mm
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    k = np.maximum(1, np.ceil(seg / 0.005)).astype(int)
    pts = [p[0][None]]
    for i in range(len(p) - 1):
        t = np.linspace(0, 1, k[i] + 1)[1:, None]
        pts.append(p[i] + t * (p[i + 1] - p[i]))
    q = np.vstack(pts)
    vis = pj.visible(q)
    xy = pj.xy(q)
    for i in range(len(q) - 1):
        a, b = tuple(xy[i]), tuple(xy[i + 1])
        if vis[i] and vis[i + 1]:
            d.line([a, b], fill=col, width=width)
        else:
            d.line([a, b], fill=_pale(col, 0.55), width=1)


def _draw_seams(
    d: ImageDraw.ImageDraw, pj: _Proj, seams: list[dict[str, Any]], colours: dict[str, tuple[int, int, int]]
) -> None:
    labels: list[tuple[float, float]] = []
    f = R.font(13)
    for s in seams:
        col = colours[s["id"]]
        pa = np.asarray(s["samples"]["a_xyz_m"])
        pb = np.asarray(s["samples"]["b_xyz_m"])
        step = max(1, len(pa) // 14)
        for i in range(0, len(pa), step):
            _line(d, pj, np.vstack([pa[i], pb[i]]), col, 1)
        for side in ("a", "b"):
            _line(d, pj, np.asarray(s[side]["polyline_m"]), col, 3)
        for p in (pa[0], pb[0]):  # s = 0
            x, y = pj.xy(p[None])[0]
            d.ellipse([x - 4, y - 4, x + 4, y + 4], fill=col, outline=R.INK)
    for s in seams:  # labels on top
        pa = np.asarray(s["samples"]["a_xyz_m"])
        vis = pj.visible(pa)
        m = pa[vis] if vis.any() else pa
        x, y = pj.xy(m[len(m) // 2][None])[0]
        while any(abs(x - a) < 50 and abs(y - b) < 14 for a, b in labels):
            y += 15
        labels.append((x, y))
        tw = d.textlength(s["id"], font=f)
        d.rectangle([x - tw / 2 - 3, y - 9, x + tw / 2 + 3, y + 8], fill=(255, 255, 255), outline=colours[s["id"]])
        d.text((x - tw / 2, y - 8), s["id"], fill=R.INK, font=f)


def view_image(
    scene: R.Scene,
    view: R.View,
    seams: list[dict[str, Any]],
    colours: dict[str, tuple[int, int, int]],
    title: str,
    frame: R.Frame | None = None,
    height: int = R.HEIGHT_PX,
) -> Image.Image:
    meshes = scene.all()
    fr = frame or R.frame_for(meshes, view, height)
    rgb, _, depth = R.rasterise(meshes, view, fr.lo, fr.hi, fr.width, fr.height)
    img = Image.fromarray(rgb)
    _draw_seams(ImageDraw.Draw(img), _Proj(view, fr, depth), seams, colours)
    legend = [(s["id"], colours[s["id"]]) for s in seams]
    return R._decorate(img, fr, view, title, R._axis_names(view), legend)


# -- flat drawings ---------------------------------------------------------------------------------------------------
def _flat_panel(
    piece: Piece,
    marks: list[tuple[F64, tuple[int, int, int], int]],
    size: int,
    edges: list[tuple[str, str, str, float, tuple[int, int, int], int, str]] | None = None,
    dots: list[tuple[F64, tuple[int, int, int], str]] | None = None,
) -> Image.Image:
    """One flat piece (outline grey), polylines ``marks`` (uv, colour, width), optional coloured edges with labels and
    named points, and ``dots`` (uv, colour, 'o' circle / 's' square)."""
    out = np.vstack([piece.knot_f, piece.knot_f[:1]])
    lo, hi = out.min(axis=0), out.max(axis=0)
    span = max(float((hi - lo).max()), 0.05)
    sc = (size - 60) / span
    h = int((hi[1] - lo[1]) * sc) + 70
    img = Image.new("RGB", (size, h), R.SURFACE)
    d = ImageDraw.Draw(img)

    def tp(p: F64) -> list[tuple[float, float]]:
        p = np.atleast_2d(p)
        return [(30 + (x - lo[0]) * sc, 40 + (hi[1] - y) * sc) for x, y in p]

    d.polygon(tp(out), fill=(238, 238, 234), outline=(120, 120, 116))
    if edges:
        f = R.font(10)
        for _eid, fr, _to, ln, col, width, lab in edges:
            s = piece.s_of[fr] + np.linspace(0.0, ln, max(2, int(ln / 0.003) + 1))
            d.line(tp(piece.at(s)[1]), fill=col, width=width)
            if lab:
                x, y = tp(piece.at(np.array([piece.s_of[fr] + ln / 2]))[1])[0]
                d.text((x + 3, y - 6), lab, fill=col, font=f)
        for n in piece.cycle:
            x, y = tp(piece.at(np.array([piece.s_of[n]]))[1])[0]
            d.ellipse([x - 2, y - 2, x + 2, y + 2], fill=R.INK)
            d.text((x + 3, y + 1), n, fill=R.INK2, font=f)
    for uv, col, width in marks:
        d.line(tp(uv), fill=col, width=width)
    for uv, col, shape in dots or []:
        x, y = tp(uv)[0]
        box = [x - 6, y - 6, x + 6, y + 6]
        if shape == "o":
            d.ellipse(box, fill=col, outline=R.INK)
        else:
            d.rectangle(box, fill=col, outline=R.INK)
    d.text((8, 8), piece.name, fill=R.INK, font=R.font(15))
    return img


def _row(images: list[Image.Image], pad: int = 10) -> Image.Image:
    w = sum(i.width for i in images) + pad * (len(images) + 1)
    h = max(i.height for i in images) + 2 * pad
    out = Image.new("RGB", (w, h), R.SURFACE)
    x = pad
    for i in images:
        out.paste(i, (x, pad))
        x += i.width + pad
    return out


def _grid(images: list[Image.Image], per_row: int, title: str, pad: int = 10) -> Image.Image:
    rows = [_row(images[i : i + per_row], pad) for i in range(0, len(images), per_row)]
    w = max(r.width for r in rows)
    out = Image.new("RGB", (w, 46 + sum(r.height for r in rows)), R.SURFACE)
    ImageDraw.Draw(out).text((10, 12), title, fill=R.INK, font=R.font(20))
    y = 46
    for r in rows:
        out.paste(r, (0, y))
        y += r.height
    return out


def unstitched_image(
    pieces: dict[str, Piece], result: dict[str, Any], colours: dict[str, tuple[int, int, int]], title: str
) -> Image.Image:
    """Every piece flat: outline edges in their seam's colour (labelled), unstitched red, ignored grey."""
    by_edge: dict[tuple[str, str], tuple[str, tuple[int, int, int]]] = {}
    for s in result["seams"]:
        for side in ("a", "b"):
            for part in s[side]["parts"]:
                for e in part["edge_ids"]:
                    by_edge[(part["piece"], e)] = (s["id"], colours[s["id"]])
    uns = {(u["piece"], u["edge"]) for u in result["unstitched_edges"]}
    ign = {(u["piece"], u["edge"]) for u in result["ignored_edges"]}
    panels = []
    for name in sorted(pieces):
        p = pieces[name]
        edges = []
        for eid, fr, to, ln in p.edges():
            if (name, eid) in by_edge:
                sid, col = by_edge[(name, eid)]
                edges.append((eid, fr, to, ln, col, 4, sid))
            elif (name, eid) in uns:
                edges.append((eid, fr, to, ln, RED, 6, eid))
            elif (name, eid) in ign:
                edges.append((eid, fr, to, ln, IGNORED, 2, ""))
            else:
                edges.append((eid, fr, to, ln, (60, 60, 60), 1, ""))
        panels.append(_flat_panel(p, [], 420, edges))
    n = result["summary"]["unstitched_edges"]
    return _grid(panels, 5, f"{title}: {n} unstitched edge(s) in red, ignored grey, seams in their colour")


def seam_image(
    scene: R.Scene,
    pieces: dict[str, Piece],
    s: dict[str, Any],
    col: tuple[int, int, int],
    rows: list[dict[str, Any]],
    title: str,
    avatar_v: F64,
) -> Image.Image:
    pa = np.asarray(s["samples"]["a_xyz_m"])
    pb = np.asarray(s["samples"]["b_xyz_m"])
    pts = np.vstack([pa, pb])
    c = pts.mean(axis=0)
    mid = 0.5 * (pa + pb)
    from scipy.spatial import cKDTree

    _, i = cKDTree(avatar_v).query(mid)
    n = (mid - avatar_v[i]).mean(axis=0)
    if np.linalg.norm(n) < 1e-3:
        n = c - avatar_v.mean(axis=0)
        n[1] = 0.0
    if np.linalg.norm(n) < 1e-6:
        n = np.array([0.0, 0.0, 1.0])
    n = n / np.linalg.norm(n)
    up = np.array([0.0, 1.0, 0.0]) if abs(n[1]) < 0.9 else np.array([0.0, 0.0, -1.0])
    view = R.View("close", tuple(float(x) for x in n), tuple(float(x) for x in up))  # type: ignore[arg-type]
    lo3, hi3 = pts.min(axis=0) - 0.12, pts.max(axis=0) + 0.12

    def near(m: R.Mesh) -> R.Mesh:
        cc = m.vertices[m.faces].mean(axis=1)
        keep = np.all((cc >= lo3) & (cc <= hi3), axis=1)
        return R.Mesh(m.name, m.vertices, m.faces[keep], m.colour, m.vcolour, m.label)

    sub = R.Scene(near(scene.avatar), [near(m) for m in scene.pieces])
    probe = R.Mesh("seam", pts, np.zeros((0, 3), dtype=np.int64), col)
    fr = R.frame_for([probe], view, 760, margin=0.07)
    img3 = view_image(sub, view, [s], {s["id"]: col}, f"{s['id']} close-up", frame=fr)
    flats = []
    for side, key in (("a", "a_uv_m"), ("b", "b_uv_m")):
        if s["samples"][key] is None:
            continue  # attached onto a host surface: no flat chain
        uv = np.asarray(s["samples"][key])
        part = np.asarray(s["samples"][f"{side}_part"])
        for k, prt in enumerate(s[side]["parts"]):
            m = part == k
            if not m.any():
                continue
            dots = []
            if k == 0:
                dots.append((uv[0], col, "o"))
            if k == len(s[side]["parts"]) - 1:
                dots.append((uv[-1], col, "s"))
            flats.append(_flat_panel(pieces[prt["piece"]], [(uv[m], col, 5)], 300, None, dots))
            ImageDraw.Draw(flats[-1]).text(
                (8, 24),
                f"side {side}{f' part {k + 1}' if len(s[side]['parts']) > 1 else ''}",
                fill=R.INK2,
                font=R.font(12),
            )
    e = s["ease"]
    o = s["orientation"]
    g = s["world_gap_m"]
    status = "PASS" if not any(r["status"] == "fail" for r in rows) else "FAIL"
    lines = [
        f"{s['id']}  [{status}]  {s['label']}",
        f"kind {s['kind']}, ease {e['mode']}"
        + (f" (declared {e['declared_ratio']:.2f})" if e["mode"] == "gather" else "")
        + f"; a {s['length_a_cm']:.2f} cm, b {s['length_b_cm']:.2f} cm, ratio a/b {s['ratio']:.4f}",
        f"gap min/mean/max {g['min'] * 100:.1f}/{g['mean'] * 100:.1f}/{g['max'] * 100:.1f} cm; crossings "
        f"{o['crossing_count']} (reversed {o['reversed_crossing_count']}); {s['samples']['count']} pairs; "
        "dot = s 0, square = s 1",
        *[f"{r['status'].upper()}: {r['id']} {r['value']} (limit {r['limit']}) {r['hint']}"[:150]
          for r in rows if r["status"] != "pass"],
        *[f"warn: {w}"[:150] for w in s["warnings"]],
    ]  # fmt: skip
    flat_col = _grid(flats, 2, "flat (piece space)") if flats else Image.new("RGB", (10, 10), R.SURFACE)
    body = _row([img3, flat_col])
    cap_h = 22 * len(lines) + 16
    out = Image.new("RGB", (body.width, body.height + cap_h), R.SURFACE)
    out.paste(body, (0, cap_h))
    dd = ImageDraw.Draw(out)
    for k, ln in enumerate(lines):
        fill = RED if ln.startswith("FAIL") or (k == 0 and status == "FAIL") else R.INK
        dd.text((12, 8 + 22 * k), ln, fill=fill, font=R.font(16 if k == 0 else 14))
    return out


def write_images(
    out: Path,
    run: Any,
    pieces: dict[str, Piece],
    result: dict[str, Any],
    rows: list[dict[str, Any]],
    title: str,
    fast: bool = False,
) -> dict[str, Path]:
    """Every stitch image into ``out/images`` (``fast``: lower resolution)."""
    img = Path(out) / "images"
    colours = {s["id"]: seam_colour(i) for i, s in enumerate(result["seams"])}
    scene = _scene(run)
    h = 700 if fast else R.HEIGHT_PX
    paths: dict[str, Path] = {}
    shown: list[tuple[str, Image.Image]] = []
    for v in VIEWS:
        im = view_image(scene, v, result["seams"], colours, f"{title} - stitches {v.name}", height=h)
        paths[v.name] = R.save(im, img / f"stitch_{v.name}.png")
        shown.append((v.name, im))
    un = unstitched_image(pieces, result, colours, f"{title} - outline coverage")
    paths["unstitched"] = R.save(un, img / "stitch_unstitched.png")
    shown.append(("unstitched", un))
    av = run.avatar()[0]
    for s in result["seams"]:
        srows = [r for r in rows if r.get("seam") == s["id"]]
        im = seam_image(scene, pieces, s, colours[s["id"]], srows, f"{title} - {s['id']}", av)
        safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in s["id"])
        paths[f"seam:{s['id']}"] = R.save(im, img / f"stitch_seam_{safe}.png")
    sheet = R.contact_sheet(shown, f"{title} - stitches")
    paths["sheet"] = R.save(sheet, img / "stitch_sheet.png")
    return paths

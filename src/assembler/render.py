"""Headless CPU renderer: numpy z-buffer rasteriser + Pillow. No GL, no GPU, no browser. Metres, y up.

Orthographic views (so the rulers are true metres), flat Lambert shading with a head light, piece outlines from the
id buffer, labels, a legend, rulers. Everything is deterministic: same meshes, same pixels.

``Scene`` holds the avatar and the placed pieces; ``render_view`` draws one view; ``cross_sections`` plots horizontal
slices; ``contact_sheet`` tiles images. ``write_all`` writes the full image set of one assembly iteration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont

F64 = NDArray[np.float64]
I64 = NDArray[np.int64]

# categorical slots (validated reference palette, light mode); more than 8 pieces get a darker second round, and
# every piece is labelled by name, so identity never rests on colour alone
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE = (252, 252, 251)
INK = (11, 11, 11)
INK2 = (82, 81, 78)
GRID = (226, 225, 220)
AVATAR = (196, 194, 188)
CRITICAL, WARNING, GOOD = (208, 59, 59), (250, 178, 25), (12, 163, 12)
HEIGHT_PX = 1100


def _hex(h: str) -> tuple[int, int, int]:
    return int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)


def piece_colour(i: int) -> tuple[int, int, int]:
    """Colour of the i-th piece (fixed order)."""
    c = np.array(_hex(PALETTE[i % len(PALETTE)]), dtype=float)
    if (i // len(PALETTE)) % 2:
        c = c * 0.62
    return int(c[0]), int(c[1]), int(c[2])


def font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    """Pillow's bundled font (identical on every machine)."""
    return ImageFont.load_default(size=size)


@dataclass
class Mesh:
    """A triangle mesh to draw. ``vcolour`` (N,3 0..255) overrides ``colour`` per vertex (heat map)."""

    name: str
    vertices: F64
    faces: I64
    colour: tuple[int, int, int]
    vcolour: NDArray[np.uint8] | None = None
    label: bool = True


@dataclass
class Scene:
    """Avatar + pieces (world metres)."""

    avatar: Mesh
    pieces: list[Mesh] = field(default_factory=list)
    points: dict[str, F64] = field(default_factory=dict)  # optional markers (arrangement points)

    def all(self) -> list[Mesh]:
        return [self.avatar, *self.pieces]


@dataclass(frozen=True)
class View:
    """Orthographic camera: looking along ``-back`` (``back`` points from the scene to the camera), ``up`` on screen."""

    name: str
    back: tuple[float, float, float]
    up: tuple[float, float, float] = (0.0, 1.0, 0.0)

    def basis(self) -> tuple[F64, F64, F64]:
        b = np.asarray(self.back, dtype=float)
        b = b / np.linalg.norm(b)
        u = np.asarray(self.up, dtype=float)
        u = u - (u @ b) * b
        u = u / np.linalg.norm(u)
        r = np.cross(u, b)
        return r, u, b


def _dir(az: float, el: float = 0.0) -> tuple[float, float, float]:
    """Camera direction from azimuth (0 = front +z, 90 = the wearer's left +x) and elevation, degrees."""
    a, e = np.radians(az), np.radians(el)
    return float(np.sin(a) * np.cos(e)), float(np.sin(e)), float(np.cos(a) * np.cos(e))


VIEWS = [
    View("front", _dir(0)),
    View("back", _dir(180)),
    View("left", _dir(90)),
    View("right", _dir(-90)),
    View("top", (0.0, 1.0, 0.0), (0.0, 0.0, -1.0)),
    View("front_left", _dir(45, 10)),
    View("front_right", _dir(-45, 10)),
    View("back_left", _dir(135, 10)),
    View("back_right", _dir(-135, 10)),
]


def rasterise(
    meshes: list[Mesh], view: View, lo: F64, hi: F64, width: int, height: int
) -> tuple[NDArray[np.uint8], NDArray[np.int32], F64]:
    """(rgb (H,W,3), owner (H,W): mesh index or -1, depth (H,W)) of the meshes inside the screen box ``lo``..``hi``
    (screen x, y in metres). Triangles are rasterised all at once (candidate pixels of every triangle's bounding box,
    barycentric test, nearest depth per pixel)."""
    r, u, b = view.basis()
    sx = (width - 1) / (hi[0] - lo[0])
    sy = (height - 1) / (hi[1] - lo[1])
    light = b  # head light
    tris, cols, owner = [], [], []
    for k, m in enumerate(meshes):
        if not len(m.faces):
            continue
        v = m.vertices
        p = np.column_stack([(v @ r - lo[0]) * sx, (hi[1] - v @ u) * sy, v @ b])
        t = p[m.faces]
        w = v[m.faces]
        n = np.cross(w[:, 1] - w[:, 0], w[:, 2] - w[:, 0])
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-18)
        shade = 0.38 + 0.62 * np.abs(n @ light)
        base = (
            m.vcolour[m.faces].astype(float).mean(axis=1)
            if m.vcolour is not None
            else np.tile(np.asarray(m.colour, dtype=float), (len(t), 1))
        )
        tris.append(t)
        cols.append(np.clip(base * shade[:, None], 0, 255))
        owner.append(np.full(len(t), k, dtype=np.int32))
    rgb = np.empty((height, width, 3), dtype=np.uint8)
    rgb[:] = SURFACE
    own = np.full((height, width), -1, dtype=np.int32)
    depth = np.full((height, width), -np.inf)
    if not tris:
        return rgb, own, depth
    t_all, c_all, o_all = np.vstack(tris), np.vstack(cols), np.concatenate(owner)
    x0 = np.clip(np.floor(t_all[:, :, 0].min(axis=1)), 0, width - 1).astype(np.int64)
    x1 = np.clip(np.ceil(t_all[:, :, 0].max(axis=1)), 0, width - 1).astype(np.int64)
    y0 = np.clip(np.floor(t_all[:, :, 1].min(axis=1)), 0, height - 1).astype(np.int64)
    y1 = np.clip(np.ceil(t_all[:, :, 1].max(axis=1)), 0, height - 1).astype(np.int64)
    keep = (t_all[:, :, 0].max(axis=1) >= 0) & (t_all[:, :, 0].min(axis=1) <= width - 1)
    keep &= (t_all[:, :, 1].max(axis=1) >= 0) & (t_all[:, :, 1].min(axis=1) <= height - 1)
    idx = np.flatnonzero(keep)
    bw, bh = (x1 - x0 + 1)[idx], (y1 - y0 + 1)[idx]
    cnt = bw * bh
    flat_pix = np.full(height * width, -np.inf)
    best_tri = np.full(height * width, -1, dtype=np.int64)
    chunk = 4_000_000
    start = 0
    cum = np.cumsum(cnt)
    while start < len(idx):
        end = int(np.searchsorted(cum, (cum[start - 1] if start else 0) + chunk, side="right"))
        end = max(end, start + 1)
        sel = idx[start:end]
        c = cnt[start:end]
        tri_id = np.repeat(sel, c)
        off = np.arange(int(c.sum())) - np.repeat(np.cumsum(c) - c, c)
        wsel = np.repeat(bw[start:end], c)
        px = x0[tri_id] + off % wsel
        py = y0[tri_id] + off // wsel
        t = t_all[tri_id]
        qx, qy = px + 0.5 - 0.5, py + 0.5 - 0.5  # pixel centres sit on integer coordinates
        ax, ay, bx, by, cx, cy = t[:, 0, 0], t[:, 0, 1], t[:, 1, 0], t[:, 1, 1], t[:, 2, 0], t[:, 2, 1]
        den = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        ok = np.abs(den) > 1e-12
        den = np.where(ok, den, 1.0)
        l1 = ((by - cy) * (qx - cx) + (cx - bx) * (qy - cy)) / den
        l2 = ((cy - ay) * (qx - cx) + (ax - cx) * (qy - cy)) / den
        l3 = 1.0 - l1 - l2
        eps = -1e-3
        inside = ok & (l1 >= eps) & (l2 >= eps) & (l3 >= eps)
        z = l1 * t[:, 0, 2] + l2 * t[:, 1, 2] + l3 * t[:, 2, 2]
        pix = (py * width + px)[inside]
        z, tri_id = z[inside], tri_id[inside]
        order = np.lexsort((tri_id, -z, pix))  # nearest first; ties by triangle id (deterministic)
        pix, z, tri_id = pix[order], z[order], tri_id[order]
        first = np.r_[True, pix[1:] != pix[:-1]]
        pix, z, tri_id = pix[first], z[first], tri_id[first]
        better = z > flat_pix[pix]
        flat_pix[pix[better]] = z[better]
        best_tri[pix[better]] = tri_id[better]
        start = end
    hit = best_tri >= 0
    rgb.reshape(-1, 3)[hit] = c_all[best_tri[hit]].astype(np.uint8)
    own.reshape(-1)[hit] = o_all[best_tri[hit]]
    depth.reshape(-1)[:] = flat_pix
    # outlines: where the owner changes, or the depth jumps (a piece folding behind itself)
    edge = np.zeros_like(own, dtype=bool)
    for dy, dx in ((0, 1), (1, 0)):
        a = own[: height - dy, : width - dx]
        bb = own[dy:, dx:]
        diff = a != bb
        edge[: height - dy, : width - dx] |= diff
        edge[dy:, dx:] |= diff
    edge &= own >= 0
    rgb[edge & (own > 0)] = (rgb[edge & (own > 0)] * 0.45).astype(np.uint8)
    rgb[edge & (own == 0)] = (120, 118, 112)
    return rgb, own, depth


@dataclass
class Frame:
    """Screen window of a view (metres) and the image size."""

    lo: F64
    hi: F64
    width: int
    height: int


def frame_for(meshes: list[Mesh], view: View, height: int = HEIGHT_PX, margin: float = 0.06) -> Frame:
    r, u, _ = view.basis()
    pts = np.vstack([m.vertices for m in meshes if len(m.vertices)])
    sx, sy = pts @ r, pts @ u
    lo = np.array([sx.min() - margin, sy.min() - margin])
    hi = np.array([sx.max() + margin, sy.max() + margin])
    if view.name not in ("top", "close") and lo[1] > -margin:  # whole-body views always show the floor
        lo[1] = -margin
    width = int(round(height * (hi[0] - lo[0]) / (hi[1] - lo[1])))
    return Frame(lo, hi, max(width, 200), height)


PAD_L, PAD_B, PAD_T, PAD_R = 70, 50, 44, 20


def _decorate(
    img: Image.Image,
    fr: Frame,
    view: View,
    title: str,
    axis_names: tuple[str, str],
    legend: list[tuple[str, tuple[int, int, int]]],
) -> Image.Image:
    """Rulers in metres, title, legend below."""
    w, h = img.size
    leg_rows = (len(legend) + 3) // 4
    canvas = Image.new("RGB", (w + PAD_L + PAD_R, h + PAD_T + PAD_B + 22 * leg_rows + 8), SURFACE)
    canvas.paste(img, (PAD_L, PAD_T))
    d = ImageDraw.Draw(canvas)
    f, fb = font(14), font(18)
    d.text((PAD_L, 10), title, fill=INK, font=fb)
    sx = (fr.width - 1) / (fr.hi[0] - fr.lo[0])
    sy = (fr.height - 1) / (fr.hi[1] - fr.lo[1])
    step = 0.1 if (fr.hi[1] - fr.lo[1]) > 0.5 else 0.05
    for v in np.arange(np.ceil(fr.lo[1] / step) * step, fr.hi[1] + 1e-9, step):
        y = PAD_T + (fr.hi[1] - v) * sy
        major = abs(v / (2 * step) - round(v / (2 * step))) < 1e-6
        d.line([(PAD_L - (10 if major else 5), y), (PAD_L, y)], fill=INK2)
        if major:
            d.text((4, y - 8), f"{v:.2f}", fill=INK2, font=f)
    for v in np.arange(np.ceil(fr.lo[0] / step) * step, fr.hi[0] + 1e-9, step):
        x = PAD_L + (v - fr.lo[0]) * sx
        major = abs(v / (2 * step) - round(v / (2 * step))) < 1e-6
        d.line([(x, PAD_T + h), (x, PAD_T + h + (10 if major else 5))], fill=INK2)
        if major:
            d.text((x - 14, PAD_T + h + 12), f"{v:.1f}", fill=INK2, font=f)
    d.text((PAD_L + w - 160, PAD_T + h + 28), f"{axis_names[0]} (m) ->", fill=INK2, font=f)
    d.text((4, PAD_T - 22), f"{axis_names[1]} (m)", fill=INK2, font=f)
    for i, (name, col) in enumerate(legend):
        x = PAD_L + (i % 4) * ((w - 10) // 4)
        y = PAD_T + h + PAD_B + 22 * (i // 4)
        d.rectangle([x, y + 3, x + 12, y + 15], fill=col)
        d.text((x + 18, y), name[:34], fill=INK, font=f)
    return canvas


def _axis_names(view: View) -> tuple[str, str]:
    r, u, _ = view.basis()
    names = ["x", "y", "z"]

    def nm(v: F64) -> str:
        k = int(np.argmax(np.abs(v)))
        if abs(v[k]) < 0.95:
            return "oblique"
        return ("+" if v[k] > 0 else "-") + names[k]

    return nm(r), nm(u)


def render_view(
    scene: Scene,
    view: View,
    title: str,
    height: int = HEIGHT_PX,
    frame: Frame | None = None,
    pieces: list[Mesh] | None = None,
    labels: bool = True,
) -> Image.Image:
    """One decorated view of the avatar and the pieces."""
    meshes = [scene.avatar, *(scene.pieces if pieces is None else pieces)]
    fr = frame or frame_for(meshes, view, height)
    rgb, own, _ = rasterise(meshes, view, fr.lo, fr.hi, fr.width, fr.height)
    img = Image.fromarray(rgb)
    d = ImageDraw.Draw(img)
    r, u, _ = view.basis()
    sx = (fr.width - 1) / (fr.hi[0] - fr.lo[0])
    sy = (fr.height - 1) / (fr.hi[1] - fr.lo[1])
    # floor line
    if view.name != "top":
        yf = (fr.hi[1] - 0.0) * sy
        d.line([(0, yf), (fr.width, yf)], fill=INK2, width=1)
    for name, p in scene.points.items():
        x, y = (p @ r - fr.lo[0]) * sx, (fr.hi[1] - p @ u) * sy
        d.ellipse([x - 4, y - 4, x + 4, y + 4], outline=INK, width=2)
        d.text((x + 6, y - 7), name, fill=INK, font=font(12))
    if labels:
        f = font(13)
        placed: list[tuple[float, float]] = []
        for k, m in enumerate(meshes[1:], start=1):
            if not m.label:
                continue
            ys, xs = np.nonzero(own == k)
            if len(xs) < 30:
                continue
            cx, cy = float(np.median(xs)), float(np.median(ys))
            while any(abs(cx - a) < 60 and abs(cy - b) < 15 for a, b in placed):
                cy += 16
            placed.append((cx, cy))
            tw = d.textlength(m.name, font=f)
            d.rectangle([cx - tw / 2 - 3, cy - 9, cx + tw / 2 + 3, cy + 9], fill=(255, 255, 255))
            d.text((cx - tw / 2, cy - 8), m.name, fill=INK, font=f)
    legend = [(m.name, m.colour) for m in meshes[1:] if m.vcolour is None]
    return _decorate(img, fr, view, title, _axis_names(view), legend)


def heat_colours(dist: F64, clearance: float) -> NDArray[np.uint8]:
    """Red below the clearance, amber within 1 cm above it, green beyond."""
    out = np.empty((len(dist), 3), dtype=np.uint8)
    out[:] = GOOD
    out[dist < clearance + 0.01] = WARNING
    out[dist < clearance] = CRITICAL
    return out


# -- horizontal sections ---------------------------------------------------------------------------------------------
def slice_segments(v: F64, f: I64, y: float) -> F64:
    """Segments (S,2,2) in (x, z) where the plane at height ``y`` cuts the mesh."""
    t = v[f]
    s = t[:, :, 1] - y
    sign = s > 0
    cut = sign.any(axis=1) & ~sign.all(axis=1)
    t, s = t[cut], s[cut]
    out = []
    for a, b in ((0, 1), (1, 2), (2, 0)):
        cross = (s[:, a] > 0) != (s[:, b] > 0)
        w = np.where(cross, s[:, a] / np.where(cross, s[:, a] - s[:, b], 1.0), np.nan)
        p = t[:, a] + w[:, None] * (t[:, b] - t[:, a])
        out.append(p[:, [0, 2]])
    pts = np.stack(out, axis=1)  # (T,3,2) with exactly two finite rows
    good = np.isfinite(pts[:, :, 0])
    segs = np.array([pts[i][good[i]][:2] for i in range(len(pts)) if good[i].sum() >= 2])
    return segs.reshape(-1, 2, 2)


def cross_sections(scene: Scene, levels: dict[str, float], title: str, size: int = 520) -> Image.Image:
    """One top-down panel (x right, +z = front up) per level: avatar slice in grey, piece slices in their colours."""
    panels = []
    meshes = scene.pieces
    for name, y in levels.items():
        img = Image.new("RGB", (size, size + 60), SURFACE)
        d = ImageDraw.Draw(img)
        half = 0.45
        sc = (size - 40) / (2 * half)

        def tp(p: F64, half: float = half, sc: float = sc) -> tuple[float, float]:
            return 20 + (p[0] + half) * sc, 40 + (half - p[1]) * sc

        for g in np.arange(-0.4, 0.41, 0.1):
            a, b = tp(np.array([g, -half])), tp(np.array([g, half]))
            d.line([a, b], fill=GRID)
            a, b = tp(np.array([-half, g])), tp(np.array([half, g]))
            d.line([a, b], fill=GRID)
        d.line([tp(np.array([0.0, -half])), tp(np.array([0.0, half]))], fill=(200, 198, 192))
        for seg in slice_segments(scene.avatar.vertices, scene.avatar.faces, y):
            d.line([tp(seg[0]), tp(seg[1])], fill=(110, 108, 104), width=2)
        n_hit = 0
        for m in meshes:
            segs = slice_segments(m.vertices, m.faces, y)
            if len(segs):
                n_hit += 1
            for seg in segs:
                d.line([tp(seg[0]), tp(seg[1])], fill=m.colour, width=3)
        f = font(15)
        d.text((10, 8), f"{name}  y = {y:.3f} m   ({n_hit} pieces)", fill=INK, font=f)
        d.text((size - 70, size + 30), "+x ->", fill=INK2, font=font(12))
        d.text((24, 44), "front (+z) ^", fill=INK2, font=font(12))
        d.text((24, size + 30), "grid 0.1 m", fill=INK2, font=font(12))
        panels.append(img)
    cols = min(3, len(panels))
    rows = (len(panels) + cols - 1) // cols
    leg_rows = (len(meshes) + 3) // 4
    sheet = Image.new("RGB", (cols * size, rows * (size + 60) + 50 + 22 * leg_rows), SURFACE)
    for i, p in enumerate(panels):
        sheet.paste(p, ((i % cols) * size, 40 + (i // cols) * (size + 60)))
    d = ImageDraw.Draw(sheet)
    d.text((10, 8), title, fill=INK, font=font(18))
    for i, m in enumerate(meshes):
        x = 10 + (i % 4) * (cols * size // 4)
        y = 45 + rows * (size + 60) + 22 * (i // 4)
        d.rectangle([x, y + 3, x + 12, y + 15], fill=m.colour)
        d.text((x + 18, y), m.name[:34], fill=INK, font=font(13))
    return sheet


def contact_sheet(images: list[tuple[str, Image.Image]], title: str, cell_h: int = 560) -> Image.Image:
    """Every image scaled to one height, four per row."""
    cells = []
    for name, im in images:
        w = int(round(im.width * cell_h / im.height))
        cells.append((name, im.resize((w, cell_h), Image.Resampling.LANCZOS)))
    per_row = 4
    rows = [cells[i : i + per_row] for i in range(0, len(cells), per_row)]
    width = max(sum(c[1].width for c in r) + 10 * (len(r) + 1) for r in rows)
    sheet = Image.new("RGB", (width, 50 + len(rows) * (cell_h + 10)), SURFACE)
    d = ImageDraw.Draw(sheet)
    d.text((10, 12), title, fill=INK, font=font(22))
    for ri, r in enumerate(rows):
        x = 10
        for _, im in r:
            sheet.paste(im, (x, 50 + ri * (cell_h + 10)))
            x += im.width + 10
    return sheet


def close_up(scene: Scene, piece: Mesh, normal: F64, title: str, reach: float = 0.15) -> Image.Image:
    """The piece seen face-on (from its outward normal) with the body and pieces within ``reach`` of it."""
    n = normal / np.linalg.norm(normal)
    up = np.array([0.0, 1.0, 0.0]) if abs(n[1]) < 0.9 else np.array([0.0, 0.0, -1.0])
    view = View("close", (float(n[0]), float(n[1]), float(n[2])), (float(up[0]), float(up[1]), float(up[2])))
    lo3, hi3 = piece.vertices.min(axis=0) - reach, piece.vertices.max(axis=0) + reach

    def near(m: Mesh) -> Mesh:
        c = m.vertices[m.faces].mean(axis=1)
        keep = np.all((c >= lo3) & (c <= hi3), axis=1)
        return Mesh(m.name, m.vertices, m.faces[keep], m.colour, m.vcolour, m.label)

    others = [near(m) if m is not piece else m for m in scene.pieces]
    sub = Scene(near(scene.avatar), others)
    fr = frame_for([piece], view, 900, margin=0.08)
    return render_view(sub, view, title, frame=fr)


def save(img: Image.Image, path: Path) -> Path:
    """PNG without timestamps or other metadata (deterministic bytes)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format="PNG", optimize=False)
    return path

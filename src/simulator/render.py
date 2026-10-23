"""Images of a simulation (``assembler.render``, the CPU renderer the stitcher uses): the draped garment on the
avatar from the front, back, both sides and the front three-quarters, a stretch heat map, the placed layout before
the simulation for comparison, and a contact sheet."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

import assembler.render as R

from .geometry import F64, I64

VIEWS = ["front", "back", "left", "right", "front_left", "front_right"]


def _scene(
    avatar: tuple[F64, I64], meshes: dict[str, tuple[F64, I64]], vcolour: dict[str, Any] | None = None
) -> R.Scene:
    pieces = [
        R.Mesh(name, v, f, R.piece_colour(i), (vcolour or {}).get(name), label=False)
        for i, (name, (v, f)) in enumerate(meshes.items())
    ]
    return R.Scene(R.Mesh("avatar", avatar[0], avatar[1], R.AVATAR, label=False), pieces)


def stretch_colours(strain: F64, limit: float) -> Any:
    """Blue (no stretch) -> amber (half the limit) -> red (at the limit) per vertex."""
    t = np.clip(strain / max(limit, 1e-9), 0.0, 1.0)
    lo, mid, hi = np.array([42, 120, 214]), np.array([250, 178, 25]), np.array([208, 59, 59])
    c = np.where((t < 0.5)[:, None], lo + (mid - lo) * (2 * t)[:, None], mid + (hi - mid) * (2 * t - 1)[:, None])
    return c.astype(np.uint8)


def write_images(
    out: Path,
    avatar: tuple[F64, I64],
    draped: dict[str, tuple[F64, I64]],
    placed: dict[str, tuple[F64, I64]],
    strain: dict[str, F64],
    limit: float,
    title: str,
    height: int = 900,
) -> dict[str, Path]:
    """``images/sim_<view>.png``, ``sim_stretch_<front|back>.png``, ``sim_placed_front.png``, ``sim_sheet.png``."""
    folder = Path(out) / "images"
    folder.mkdir(parents=True, exist_ok=True)
    views = {v.name: v for v in R.VIEWS}
    sc = _scene(avatar, draped)
    paths: dict[str, Path] = {}
    sheet: list[tuple[str, Any]] = []
    for name in VIEWS:
        img = R.render_view(sc, views[name], f"{title}: simulated, {name}", height=height, labels=False)
        paths[name] = R.save(img, folder / f"sim_{name}.png")
        sheet.append((name, img))
    heat = _scene(avatar, draped, {k: stretch_colours(s, limit) for k, s in strain.items()})
    for name in ("front", "back"):
        img = R.render_view(heat, views[name], f"{title}: stretch vs pattern (blue 0, red {limit * 100:g} %)",
                            height=height, labels=False)  # fmt: skip
        paths[f"stretch_{name}"] = R.save(img, folder / f"sim_stretch_{name}.png")
        sheet.append((f"stretch {name}", img))
    before = _scene(avatar, placed)
    img = R.render_view(before, views["front"], f"{title}: placed (before)", height=height, labels=False)
    paths["placed_front"] = R.save(img, folder / "sim_placed_front.png")
    sheet.insert(0, ("placed", img))
    paths["sheet"] = R.save(R.contact_sheet(sheet, f"{title}: simulation"), folder / "sim_sheet.png")
    return paths

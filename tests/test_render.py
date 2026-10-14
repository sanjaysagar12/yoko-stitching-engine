"""Headless renderer: every view renders, is not empty, is deterministic and fast; sections show body and pieces."""

from __future__ import annotations

import hashlib
import io
import time
from pathlib import Path

import numpy as np
import pytest

from alignment_engine.loaders import read_avatar
from assembler import render as R


@pytest.fixture(scope="module")
def scene(tmp_path_factory: pytest.TempPathFactory) -> R.Scene:
    from tests.synthetic_avatar import build_avatar

    a = read_avatar(build_avatar(tmp_path_factory.mktemp("render") / "avatar"))
    # a 40 x 50 cm panel rolled half way round the torso in front, and a flat 20 cm square at the left side
    th = np.linspace(-0.9, 0.9, 30)
    ys = np.linspace(0.85, 1.35, 20)
    t, y = np.meshgrid(th, ys)
    v = np.column_stack([0.25 * np.sin(t).ravel(), y.ravel(), 0.25 * np.cos(t).ravel()])
    f = []
    for i in range(19):
        for j in range(29):
            k = i * 30 + j
            f += [(k, k + 1, k + 31), (k, k + 31, k + 30)]
    sq = np.array([[0.3, 1.0, -0.1], [0.3, 1.0, 0.1], [0.3, 1.2, 0.1], [0.3, 1.2, -0.1]])
    return R.Scene(
        R.Mesh("avatar", a.vertices, a.faces, R.AVATAR),
        [
            R.Mesh("Front", v, np.array(f), R.piece_colour(0)),
            R.Mesh("Side", sq, np.array([[0, 1, 2], [0, 2, 3]]), R.piece_colour(1)),
        ],
    )


def _png(im: R.Image.Image) -> bytes:
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


@pytest.mark.parametrize("view", R.VIEWS, ids=[v.name for v in R.VIEWS])
def test_every_view_renders_fast_and_deterministically(scene: R.Scene, view: R.View) -> None:
    t0 = time.perf_counter()
    a = R.render_view(scene, view, view.name)
    assert time.perf_counter() - t0 < 20.0
    assert a.height >= 1000
    arr = np.asarray(a)
    assert (np.abs(arr.astype(int) - np.array(R.SURFACE)).sum(axis=2) > 30).mean() > 0.02  # not empty
    b = R.render_view(scene, view, view.name)
    assert hashlib.sha256(_png(a)).hexdigest() == hashlib.sha256(_png(b)).hexdigest()


def test_front_view_shows_the_piece_colour(scene: R.Scene) -> None:
    arr = np.asarray(R.render_view(scene, R.VIEWS[0], "front")).astype(int)
    target = np.array(R.piece_colour(0))
    # flat shading darkens the colour: look for pixels of the same hue ratio
    close = np.abs(arr[:, :, :3] - target).sum(axis=2) < 120
    assert close.sum() > 2000


def test_cross_section_contains_body_and_pieces(scene: R.Scene) -> None:
    im = np.asarray(R.cross_sections(scene, {"Waist": 1.1, "Knee": 0.48}, "sections")).astype(int)
    panel = im[40:600, :520]  # the first panel (y = 1.1 m): avatar slice in grey and the rolled piece
    grey = (np.abs(panel - np.array([110, 108, 104])).sum(axis=2) < 10).sum()
    piece = (np.abs(panel - np.array(R.piece_colour(0))).sum(axis=2) < 10).sum()
    assert grey > 200 and piece > 200
    segs = R.slice_segments(scene.pieces[0].vertices, scene.pieces[0].faces, 1.1)
    assert len(segs) > 10 and np.allclose(np.hypot(segs[:, :, 0], segs[:, :, 1]), 0.25, atol=0.005)


def test_heat_colours_follow_the_clearance() -> None:
    c = R.heat_colours(np.array([0.01, 0.035, 0.08]), 0.03)
    assert [tuple(x) for x in c] == [R.CRITICAL, R.WARNING, R.GOOD]


def test_contact_sheet_and_save(scene: R.Scene, tmp_path: Path) -> None:
    ims = [(v.name, R.render_view(scene, v, v.name, height=400)) for v in R.VIEWS[:3]]
    sheet = R.contact_sheet(ims, "sheet", cell_h=200)
    p = R.save(sheet, tmp_path / "s.png")
    assert p.stat().st_size > 1000 and sheet.width > 3 * 100

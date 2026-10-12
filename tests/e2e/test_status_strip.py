"""Browser test of the status strip under the 3D view (headless Chromium via Playwright).

Runs the real backend on a male_shirt build with the synthetic avatar (no SMPL-X files needed). Needs the built frontend
(``npm run build`` in src/web/frontend) and ``python -m playwright install chromium``.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
DIST = ROOT / "src" / "web" / "frontend" / "dist" / "index.html"
SHOTS = ROOT / "outputs" / "e2e"
pytestmark = pytest.mark.skipif(not DIST.is_file(), reason="frontend not built (npm run build in src/web/frontend)")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture(scope="module")
def server(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    from tests.web_build import prebuild

    out = tmp_path_factory.mktemp("outputs")
    prebuild(out)
    port = free_port()
    env = {**os.environ, "SM2D_WEB_OUTPUTS_DIR": str(out)}
    proc = subprocess.Popen(
        [sys.executable, "-m", "web.backend", "--port", str(port)],
        cwd=ROOT,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    url = f"http://127.0.0.1:{port}"
    try:
        for _ in range(150):
            try:
                urllib.request.urlopen(url + "/api/jobs", timeout=1).read()
                break
            except OSError:
                time.sleep(0.2)
        else:
            raise AssertionError("backend did not start")
        yield url
    finally:
        proc.terminate()
        proc.wait(timeout=10)


@pytest.fixture(scope="module")
def page(server: str) -> Iterator[Any]:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        try:
            browser = p.chromium.launch(
                args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"]
            )
        except Exception as e:  # noqa: BLE001
            pytest.skip(f"Chromium not available: {e}")
        pg = browser.new_page(viewport={"width": 1440, "height": 900})
        pg.on("pageerror", lambda e: pytest.fail(f"page error: {e}"))
        pg.goto(server)
        yield pg
        browser.close()


def place(page: Any, piece: str, point: str) -> dict[str, Any]:
    """Select a piece, click an arrangement point; returns the /api/align answer."""
    page.click(f"[data-testid=piece-{piece}]")
    pos = page.evaluate(f"window.__fit.pointClient('{point}')")
    assert pos is not None, point
    with page.expect_response(lambda r: r.url.endswith("/api/align"), timeout=60_000) as info:
        page.mouse.click(pos["x"], pos["y"])
    assert info.value.status == 200
    page.wait_for_timeout(300)
    return info.value.json()  # type: ignore[no-any-return]


def cm(m: float) -> str:
    return f"{m * 100:.1f} cm"


def test_status_strip_is_green_with_the_engine_numbers(page: Any, server: str) -> None:
    page.wait_for_function("document.querySelectorAll('[data-testid=job-select] option').length >= 4")
    page.select_option("[data-testid=job-select]", "male_shirt")
    page.wait_for_function("window.__fit && window.__fit.ready()", timeout=120_000)
    place(page, "FrontPanel", "Chest_Front")
    assert page.input_value("[data-testid=ctl-wrap]") == "roll"  # the default on the torso: size exact
    place(page, "BackPanel", "Chest_Back")
    doc = place(page, "Yoke", "Chest_Back")  # a second piece on an occupied point
    v = doc["validity"]
    assert v["valid"] is True, v["violations"]
    assert [p["piece"] for p in doc["placements"]] == ["FrontPanel", "BackPanel", "Yoke"]
    # earlier pieces were sent back with their solved place and did not move; the yoke is clear of the back panel
    assert all(p["solved"] for p in doc["placements"])
    assert doc["placements"][2]["arrange"]["min_distance_to_pieces_m"]["BackPanel"] >= 0.03 - 0.0005
    page.wait_for_function("window.__fit.status().visible && window.__fit.status().valid === true")
    st = page.evaluate("window.__fit.status()")
    assert "ok" in st["className"].split() and "bad" not in st["className"].split()
    text = page.inner_text("[data-testid=status-strip]")
    assert f"Size unchanged (max {v['max_size_change_pct']:.2f} %)" in text
    assert f"Body clearance {cm(v['min_distance_to_avatar_m'])}" in text
    assert f"Smallest gap between pieces {cm(v['min_distance_between_pieces_m'])}" in text
    color = page.eval_on_selector("[data-testid=status-strip]", "e => getComputedStyle(e).backgroundColor")
    assert color == "rgb(240, 253, 244)"  # green
    assert page.evaluate("window.__fit.state().selectedPiece") == "Yoke"  # the piece just placed stays selected
    page.wait_for_selector("[data-testid=size-check]")
    assert "Size unchanged" in page.inner_text("[data-testid=size-check]")
    assert "display space, not ease" in page.inner_text("[data-testid=arrange-piece]")
    SHOTS.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=str(SHOTS / "status_strip_green.png"))


def test_status_strip_is_red_and_lists_the_pair_when_not_valid(page: Any) -> None:
    def forge(route: Any) -> None:
        resp = route.fetch()
        doc = resp.json()
        doc["validity"]["valid"] = False
        doc["validity"]["gaps_ok"] = False
        doc["validity"]["violations"] = [
            {
                "kind": "GAP_PIECES",
                "pieces": ["BackPanel", "Yoke"],
                "value_m": 0.012,
                "required_m": 0.03,
                "at_m": None,
                "limit": "max_shift",
            }
        ]
        route.fulfill(response=resp, body=json.dumps(doc), headers={**resp.headers, "content-type": "application/json"})

    page.route("**/api/align", forge)
    try:
        with page.expect_response(lambda r: r.url.endswith("/api/align"), timeout=60_000):
            page.fill("[data-testid=ctl-rotation]", "5")
            page.press("[data-testid=ctl-rotation]", "Enter")
        page.wait_for_function("window.__fit.status().valid === false")
        st = page.evaluate("window.__fit.status()")
        assert "bad" in st["className"].split() and "ok" not in st["className"].split()
        item = page.inner_text("[data-testid=status-violation-0]")
        assert "BackPanel / Yoke" in item and "gap too small" in item and "1.2 cm" in item
        color = page.eval_on_selector("[data-testid=status-strip]", "e => getComputedStyle(e).backgroundColor")
        assert color == "rgb(254, 242, 242)"  # red, never green while not valid
        page.click("[data-testid=status-violation-0]")  # frames the camera on the pair
        page.screenshot(path=str(SHOTS / "status_strip_red.png"))
    finally:
        page.unroute("**/api/align")

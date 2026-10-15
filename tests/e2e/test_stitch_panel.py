"""Browser test of the stitching panel (headless Chromium via Playwright): import the skirt assembly, import its
stitching file, stitch, the seam list with every seam passing, click a seam (highlight and zoom), export the JSON.

Real backend, synthetic avatar. Needs the built frontend (``npm run build`` in src/web/frontend) and Chromium.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
import urllib.request
from collections.abc import Iterator
from pathlib import Path

import pytest

from tests.e2e.test_web_e2e import DIST, SHOTS, free_port

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(not DIST.is_file(), reason="frontend not built (npm run build in src/web/frontend)")


@pytest.fixture(scope="module")
def server(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    from tests.web_build import prebuild

    out = tmp_path_factory.mktemp("outputs")
    prebuild(out, "aldrich_skirt")
    port = free_port()
    env = {**os.environ, "SM2D_WEB_OUTPUTS_DIR": str(out)}
    proc = subprocess.Popen([sys.executable, "-m", "web.backend", "--port", str(port)], cwd=ROOT, env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)  # fmt: skip
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


def test_stitch_panel(server: str, tmp_path: Path) -> None:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        try:
            browser = p.chromium.launch(
                args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"]
            )
        except Exception as e:  # noqa: BLE001
            pytest.skip(f"Chromium not available: {e}")
        page = browser.new_page(viewport={"width": 1440, "height": 900}, accept_downloads=True)
        errors: list[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(server)
        page.wait_for_function("document.querySelectorAll('[data-testid=job-select] option').length >= 4")
        page.set_input_files(
            "[data-testid=import-assembly-input]", str(ROOT / "assert/assemblies/aldrich_skirt.assembly.json")
        )
        page.wait_for_function(
            "document.querySelector('[data-testid=assembly-summary]') && window.__fit.status().valid !== null"
            " && window.__fit.ready()",
            timeout=600_000,
        )
        assert "aldrich_skirt" in page.inner_text("[data-testid=assembly-summary]")
        page.set_input_files(
            "[data-testid=import-stitching-input]", str(ROOT / "assert/stitchings/aldrich_skirt.stitching.json")
        )
        page.wait_for_function("!document.querySelector('[data-testid=run-stitching]').disabled", timeout=30_000)
        page.click("[data-testid=run-stitching]")
        page.wait_for_selector("[data-testid=seam-list]", timeout=180_000)
        status = page.inner_text("[data-testid=stitch-status]")
        assert status.startswith("PASS"), status
        assert page.locator("[data-testid=seam-list] li").count() == 10
        assert page.locator("[data-testid=seam-list] li.fail").count() == 0
        page.click("[data-testid=seam-side_R]")
        page.wait_for_selector("[data-testid=seam-side_R].selected")
        SHOTS.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(SHOTS / "stitch_panel.png"))
        with page.expect_download() as dl:
            page.click("[data-testid=export-stitching]")
        assert dl.value.suggested_filename == "aldrich_skirt.stitching.json"
        browser.close()
        assert not errors, errors

"""Browser test of the simulation panel (headless Chromium via Playwright): import the skirt assembly and its stitching,
stitch, Simulate (draft mesh, 12 frames), the live view, the verdict, the animation slider, Show placed / Show draped
and the OBJ download.

Real backend, synthetic avatar, the cloth solver on the GPU. Needs the built frontend, Chromium and ZOZO's Contact
Solver (skipped without any of them).
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

from simulator.solver import solver_info
from tests.e2e.test_web_e2e import DIST, SHOTS, free_port

ROOT = Path(__file__).resolve().parents[2]
pytestmark = [
    pytest.mark.skipif(not DIST.is_file(), reason="frontend not built (npm run build in src/web/frontend)"),
    pytest.mark.skipif(not solver_info()["available"], reason="ZOZO's Contact Solver is not installed"),
]


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
        proc.wait(timeout=30)


def test_sim_panel(server: str) -> None:
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
        assert page.locator("[data-testid=run-simulation]").is_disabled()  # nothing stitched yet
        page.set_input_files(
            "[data-testid=import-assembly-input]", str(ROOT / "assert/assemblies/aldrich_skirt.assembly.json")
        )
        page.wait_for_function(
            "document.querySelector('[data-testid=assembly-summary]') && window.__fit.status().valid !== null"
            " && window.__fit.ready()",
            timeout=600_000,
        )
        page.set_input_files(
            "[data-testid=import-stitching-input]", str(ROOT / "assert/stitchings/aldrich_skirt.stitching.json")
        )
        page.wait_for_function("!document.querySelector('[data-testid=run-stitching]').disabled", timeout=30_000)
        page.click("[data-testid=run-stitching]")
        page.wait_for_selector("[data-testid=seam-list]", timeout=180_000)
        page.wait_for_function("!document.querySelector('[data-testid=run-simulation]').disabled", timeout=30_000)
        page.select_option("[data-testid=sim-quality]", "draft")
        page.fill("[data-testid=sim-frames]", "12")
        page.dispatch_event("[data-testid=sim-frames]", "change")
        page.click("[data-testid=run-simulation]")
        page.wait_for_function(
            "(() => { const s = document.querySelector('[data-testid=sim-status]');"
            " return s && /^(PASS|FAIL|failed)/.test(s.textContent || ''); })()",
            timeout=1_800_000,
        )
        status = page.inner_text("[data-testid=sim-status]")
        # the size 12 skirt cannot close round the synthetic (male) test body: a finished run with its verdict
        assert status.startswith(("PASS", "FAIL")) and "0 vertices inside the body" in status, status
        assert page.locator("[data-testid=sim-frame]").count() == 1  # the kept frames can be played back
        assert page.inner_text("[data-testid=sim-toggle]") == "Show placed"  # the draped garment is on show
        SHOTS.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(SHOTS / "sim_panel.png"))
        page.click("[data-testid=sim-toggle]")
        assert page.inner_text("[data-testid=sim-toggle]") == "Show draped"
        with page.expect_download() as dl:
            page.click("[data-testid=sim-obj]")
        assert dl.value.suggested_filename == "aldrich_skirt_draped_m.obj"
        browser.close()
        assert not errors, errors

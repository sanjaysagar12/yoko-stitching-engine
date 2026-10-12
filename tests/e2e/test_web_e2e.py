"""Browser end-to-end tests (headless Chromium via Playwright).

Needs the built frontend (``npm run build`` in src/web/frontend) and ``python -m playwright install chromium``.
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
pytestmark = [
    pytest.mark.skipif(not DIST.is_file(), reason="frontend not built (npm run build in src/web/frontend)"),
    pytest.mark.skipif(not (ROOT / "models/smplx/SMPLX_MALE.npz").is_file(), reason="models/smplx not present"),
]
SHOTS = ROOT / "outputs" / "e2e"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture(scope="module")
def server(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    port = free_port()
    env = {**os.environ, "SM2D_WEB_OUTPUTS_DIR": str(tmp_path_factory.mktemp("outputs"))}
    proc = subprocess.Popen(
        [sys.executable, "-m", "web.backend", "--port", str(port)],
        cwd=ROOT,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    url = f"http://127.0.0.1:{port}"
    try:
        for _ in range(100):
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


def fresh(page: Any, server: str) -> None:
    """Reload the app and build male_shirt (cached after the first time)."""
    page.goto(server)
    page.wait_for_function("document.querySelectorAll('[data-testid=job-select] option').length >= 4")
    page.select_option("[data-testid=job-select]", "male_shirt")
    page.wait_for_function("window.__fit && window.__fit.ready()", timeout=120_000)


def click_point(page: Any, name: str) -> dict[str, Any]:
    """Click an arrangement point and return the /api/align answer that follows."""
    pos = page.evaluate(f"window.__fit.pointClient('{name}')")
    assert pos is not None
    with page.expect_response(lambda r: r.url.endswith("/api/align")) as info:
        page.mouse.click(pos["x"], pos["y"])
    assert info.value.status == 200
    return info.value.json()  # type: ignore[no-any-return]


def build_result(server: str) -> dict[str, Any]:
    req = urllib.request.Request(f"{server}/api/jobs/male_shirt/build", method="POST")
    started = json.loads(urllib.request.urlopen(req).read())
    return json.loads(urllib.request.urlopen(f"{server}/api/builds/{started['build_id']}").read())["result"]  # type: ignore[no-any-return]


def test_dropdown_lists_the_three_jobs(page: Any) -> None:
    page.wait_for_function("document.querySelectorAll('[data-testid=job-select] option').length >= 4")
    options = page.eval_on_selector_all(
        "[data-testid=job-select] option", "els => els.map(e => e.value).filter(Boolean)"
    )
    assert options == ["aldrich_skirt", "male_shirt", "trousers"]


def test_select_piece_then_click_chest_front(page: Any, server: str) -> None:
    fresh(page, server)
    assert page.inner_text("[data-testid=status]").startswith("ready: 17 pieces, 80 points")
    assert "+x = wearer's left" in page.inner_text("[data-testid=convention]")
    assert "chest_circ" in page.inner_text("[data-testid=measurements]")

    page.click("[data-testid=piece-FrontPanel]")
    assert page.evaluate("window.__fit.state().selectedPiece") == "FrontPanel"
    pt = next(p for p in build_result(server)["avatar"]["points"] if p["name"] == "Chest_Front")
    placement = click_point(page, "Chest_Front")["placements"][0]
    assert placement["piece"] == "FrontPanel" and placement["point"] == "Chest_Front"
    # a torso point defaults to rolling the piece (size exact); the reference sits on the ray from the torso axis
    # through the point, at the roll radius (display space), and the answer says the layout is valid
    assert page.input_value("[data-testid=ctl-wrap]") == "roll"
    assert placement["wrapped_mesh"]["vertices"] and placement["size_check"]["max_edge_change_pct"] <= 0.05
    ref = placement["reference_position_m"]
    assert abs(ref[0] - pt["position_m"][0]) < 0.002 and ref[2] >= pt["position_m"][2] - 0.001
    assert placement["arrange"]["min_distance_to_avatar_m"] >= 0.029
    assert placement["bbox_cm"] == {"width": pytest.approx(37.25, abs=0.01), "height": pytest.approx(74.40, abs=0.01)}
    page.wait_for_selector("[data-testid=scale-check-placed]")
    text = page.inner_text("[data-testid=scale-check-placed]")
    assert "37.25 x 74.40 cm" in text and "unchanged" in text
    assert page.inner_text("[data-testid=placed-on]") == "placed on Chest_Front"
    assert "extends" in page.inner_text("[data-testid=extends]")


def test_wrap_control_calls_the_backend_and_keeps_the_size(page: Any) -> None:
    with page.expect_response(lambda r: r.url.endswith("/api/align")) as info:
        page.select_option("[data-testid=ctl-wrap]", "around")
    r = info.value.json()["placements"][0]
    assert r["wrapped_mesh"]["vertices"] and r["bbox_cm"]["width"] == pytest.approx(37.25, abs=0.01)
    with page.expect_response(lambda r: r.url.endswith("/api/align")) as info:
        page.fill("[data-testid=ctl-rotation]", "10")
        page.dispatch_event("[data-testid=ctl-rotation]", "change")
    assert info.value.status == 200


def test_clone_and_place_a_centre_line_pair(page: Any, server: str) -> None:
    fresh(page, server)
    page.click("[data-testid=piece-FrontPanel]")
    # the inspector shows the outline with clickable named points: the CF point becomes the reference
    page.click("[data-testid=insp-pt-A33]")
    assert page.input_value("[data-testid=ctl-reference]") == "A33"
    page.click("[data-testid=clone-open]")
    page.check("[data-testid=axis-edge]")
    page.select_option("[data-testid=clone-from]", "A33")
    page.select_option("[data-testid=clone-to]", "A32")
    assert page.input_value("[data-testid=clone-name]") == "FrontPanel_L"
    assert page.is_checked("[data-testid=clone-pair-now]")
    assert page.input_value("[data-testid=clone-pair-mode]") == "same_anchor"
    with page.expect_response(lambda r: r.url.endswith("/api/clone")):
        page.click("[data-testid=clone-create]")
    page.wait_for_selector("[data-testid=pending-pair]")
    assert page.evaluate("window.__fit.state().derived") == ["FrontPanel_L"]
    assert "mirror of FrontPanel" in page.inner_text("[data-testid=piece-FrontPanel_L]")

    doc = click_point(page, "Chest_Front")  # the pair is placed on the same anchor
    placements = {p["piece"]: p for p in doc["placements"]}
    assert set(placements) == {"FrontPanel", "FrontPanel_L"}
    a, b = placements["FrontPanel"], placements["FrontPanel_L"]
    assert a["point"] == b["point"] == "Chest_Front"
    ra, rb = (
        a["reference_position_m"],
        b["reference_position_m"],
    )  # the same anchor, half a gap off the centre line each
    assert abs(ra[0] + rb[0]) < 1e-5 and abs(ra[1] - rb[1]) < 1e-5 and abs(ra[2] - rb[2]) < 1e-5
    assert {a["extends"], b["extends"]} == {"left", "right"}  # opposite sides of the centre line
    size = {"width": pytest.approx(37.25, abs=0.01), "height": pytest.approx(74.40, abs=0.01)}
    assert a["bbox_cm"] == size and b["bbox_cm"] == size
    # the CF edges are the gap apart all along (room to draw the seam), never on top of each other
    (stitch,) = doc["stitch_reports"]
    assert stitch["diff_cm"] == pytest.approx(0.0, abs=1e-6) and stitch["segments"]
    gaps = [sum((u - v) ** 2 for u, v in zip(p, q, strict=True)) ** 0.5 for p, q in stitch["segments"]]
    assert min(gaps) == pytest.approx(0.03, abs=0.001) and doc["validity"]["valid"] is True
    assert doc["derived_pieces"][0]["id"] == "FrontPanel_L"
    assert not any("SIDE_" in w for w in a["warnings"] + b["warnings"])

    page.wait_for_selector("[data-testid=scale-check-pair]")
    assert "74.5 cm" in page.inner_text("[data-testid=scale-check-pair]")  # 37.25 + 37.25
    page.click("[data-testid=piece-FrontPanel_L]")
    assert "unchanged" in page.inner_text("[data-testid=scale-check-placed]")
    page.wait_for_timeout(900)
    SHOTS.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=str(SHOTS / "centre_line_pair.png"))


def test_a_side_pair_for_sleeves(page: Any, server: str) -> None:
    fresh(page, server)
    page.click("[data-testid=piece-ShortSleeve]")
    page.click("[data-testid=clone-open]")
    page.check("[data-testid=axis-bbox]")
    assert page.input_value("[data-testid=clone-pair-mode]") == "mirror_anchor"  # a flip in place is a side pair
    assert page.input_value("[data-testid=clone-name]") == "ShortSleeve_L"
    with page.expect_response(lambda r: r.url.endswith("/api/clone")):
        page.click("[data-testid=clone-create]")
    page.wait_for_selector("[data-testid=pending-pair]")
    doc = click_point(page, "UpperArm_Outer_R")
    placements = {p["piece"]: p for p in doc["placements"]}
    assert placements["ShortSleeve"]["point"] == "UpperArm_Outer_R"
    assert placements["ShortSleeve_L"]["point"] == "UpperArm_Outer_L"
    a, b = placements["ShortSleeve"]["reference_position_m"], placements["ShortSleeve_L"]["reference_position_m"]
    assert abs(a[0] + b[0]) < 0.005 and abs(a[1] - b[1]) < 0.005 and abs(a[2] - b[2]) < 0.005
    assert page.input_value("[data-testid=ctl-wrap]") == "roll"  # limb points roll around the arm (size exact)


def test_clone_makes_plain_copies_that_are_never_a_pair(page: Any, server: str) -> None:
    fresh(page, server)
    page.click("[data-testid=piece-ShortSleeve]")
    click_point(page, "UpperArm_Outer_R")
    page.fill("[data-testid=ctl-rotation]", "15")
    with page.expect_response(lambda r: r.url.endswith("/api/align")):
        page.dispatch_event("[data-testid=ctl-rotation]", "change")
    page.click("[data-testid=copy-open]")
    page.wait_for_selector("[data-testid=copy-dialog]")
    assert page.input_value("[data-testid=copy-name]") == "ShortSleeve_copy"  # never _L / _R for a copy
    page.fill("[data-testid=copy-count]", "3")
    assert page.input_value("[data-testid=copy-place]") == "opposite"  # the source is on a side point
    with (
        page.expect_request(lambda r: r.url.endswith("/api/align")) as req,
        page.expect_response(lambda r: r.url.endswith("/api/align")) as resp,
    ):
        page.click("[data-testid=copy-create]")
    copies = ["ShortSleeve_copy", "ShortSleeve_copy2", "ShortSleeve_copy3"]
    assert page.evaluate("window.__fit.state().derived") == copies
    sent = req.value.post_data_json
    assert all(d["op"] == "copy" and "axis" not in d for d in sent["derived_pieces"])
    doc = resp.value.json()
    placements = {p["piece"]: p for p in doc["placements"]}
    a, b = placements["ShortSleeve"], placements["ShortSleeve_copy"]
    asked = {p["piece"]: p for p in sent["placements"]}
    assert b["point"] == "UpperArm_Outer_L" and asked["ShortSleeve_copy"]["rotation_deg"] == 15  # same rotation
    assert a["arrange"]["pair_with"] == b["arrange"]["pair_with"] == ""
    assert doc["validity"]["valid"] and not any(v["kind"].startswith("PAIR_") for v in doc["validity"]["violations"])
    assert "copy of ShortSleeve" in page.inner_text("[data-testid=piece-ShortSleeve_copy]")
    # a copy has no pair rows (and neither has its source, which has no mirror clone)
    assert page.evaluate("window.__fit.state().selectedPiece") == "ShortSleeve_copy"  # the new copy is selected
    page.wait_for_selector("[data-testid=copy-of]")
    page.wait_for_selector("[data-testid=copy-handedness]")  # a sleeve is not symmetric: wrong way round there
    assert page.query_selector("[data-testid^=pair-same-]") is None
    page.click("[data-testid=piece-ShortSleeve]")
    assert page.query_selector("[data-testid^=pair-same-]") is None
    page.click("[data-testid=piece-ShortSleeve_copy2]")
    page.click("[data-testid=derived-delete]")
    assert page.evaluate("window.__fit.state().derived") == ["ShortSleeve_copy", "ShortSleeve_copy3"]


def test_back_points_are_not_pickable_in_the_front_view_until_xray(page: Any, server: str) -> None:
    fresh(page, server)
    page.click("[data-testid=view-front]")
    page.wait_for_timeout(700)  # the occlusion map follows the camera
    back = page.evaluate("window.__fit.pointInfo('Chest_Back')")
    front = page.evaluate("window.__fit.pointInfo('Chest_Front')")
    assert back["hidden"] and not back["pickable"] and front["pickable"]
    page.mouse.click(back["x"], back["y"])
    assert (
        page.evaluate("window.__fit.state().pendingPoint") != "Chest_Back"
    )  # the click did not reach the hidden point
    page.check("[data-testid=toggle-xray]")
    assert page.evaluate("window.__fit.pointInfo('Chest_Back').pickable")
    page.mouse.click(back["x"], back["y"])
    assert page.evaluate("window.__fit.state().pendingPoint") == "Chest_Back"
    # from behind it is the front point that is hidden
    page.click("[data-testid=view-back]")
    page.wait_for_timeout(700)
    page.uncheck("[data-testid=toggle-xray]")
    assert page.evaluate("window.__fit.pointInfo('Chest_Front').hidden")
    # the point filter hides non-matching points
    page.fill("[data-testid=point-filter]", "Thigh")
    assert page.evaluate("window.__fit.pointInfo('Chest_Back').filtered")
    assert not page.evaluate("window.__fit.pointInfo('Thigh_Front_L').filtered")


def clone_pair(page: Any, piece: str, axis: str, mode: str, point: str, setup: Any = None) -> None:
    """Clone a piece through the dialog and place the pair on a point (the pair is placed by the backend)."""
    page.click(f"[data-testid=piece-{piece}]")
    if setup:
        setup()
    page.click("[data-testid=clone-open]")
    page.check(f"[data-testid=axis-{axis}]")
    if axis == "x":
        page.select_option("[data-testid=clone-xpoint]", "A24")
    elif axis == "edge":
        page.select_option("[data-testid=clone-from]", "A33")
        page.select_option("[data-testid=clone-to]", "A32")
    page.select_option("[data-testid=clone-pair-mode]", mode)
    with page.expect_response(lambda r: r.url.endswith("/api/clone")):
        page.click("[data-testid=clone-create]")
    page.wait_for_selector("[data-testid=pending-pair]")
    page.check("[data-testid=toggle-xray]")  # back points are hidden behind the body in the front view
    click_point(page, point)


def test_auto_arrange_shows_the_pieces_apart_with_true_ease(page: Any, server: str) -> None:
    fresh(page, server)
    clone_pair(page, "FrontPanel", "edge", "same_anchor", "Neck_Front", lambda: page.click("[data-testid=insp-pt-A32]"))
    clone_pair(page, "BackPanel", "x", "same_anchor", "Chest_Back", lambda: page.click("[data-testid=insp-pt-A24]"))
    clone_pair(page, "ShortSleeve", "bbox", "mirror_anchor", "UpperArm_Outer_R")
    for p in ("FrontPanel", "FrontPanel_L"):  # Neck_Front belongs to the neck: these two wrap the torso
        page.click(f"[data-testid=piece-{p}]")
        page.select_option("[data-testid=ctl-ring]", "torso")
    clearance = 0.03
    SHOTS.mkdir(parents=True, exist_ok=True)
    for view in ("front", "left"):  # the same two views before and after
        page.click(f"[data-testid=view-{view}]")
        page.wait_for_timeout(900)
        page.screenshot(path=str(SHOTS / f"before_{view}.png"))
    with page.expect_response(lambda r: r.url.endswith("/api/arrange"), timeout=240_000) as info:
        page.click("[data-testid=arrange-run]")
    assert info.value.status == 200
    doc = info.value.json()
    page.wait_for_selector("[data-testid=arrange-rings]")
    arr = doc["arrangement"]
    assert not any(w.startswith(("CLEARANCE_BODY", "GAP_PIECES")) for w in doc["warnings"]), doc["warnings"]
    assert arr["violations"] == [] and doc["validity"]["valid"] is True
    gap = arr["options"]["gap_piece"]
    for p in doc["placements"]:  # every pair, every layer: no exemption any more
        a = p["arrange"]
        assert a["min_distance_to_avatar_m"] >= clearance - 0.001, p["piece"]
        assert all(d >= gap - 0.001 for d in a["min_distance_to_pieces_m"].values()), p["piece"]
    page.wait_for_function("window.__fit.status().valid === true")
    for ring in arr["rings"]:  # one row of the true-ease table per ring, with the numbers of the report
        assert page.inner_text(f"[data-testid=ring-{ring['ring']}]")
        shown = float(page.inner_text(f"[data-testid=ease-{ring['ring']}]"))
        assert shown == pytest.approx(ring["true_ease_m"] * 100, abs=0.06)
    assert page.inner_text("[data-testid=arrange-honesty]").count("never ease") == 1
    for view, name in (("front", "front"), ("left", "side")):
        page.click(f"[data-testid=view-{view}]")
        page.wait_for_timeout(900)
        page.screenshot(path=str(SHOTS / f"arranged_{name}.png"))
    # click an outline edge: its id and length appear in the selection panel
    edges = page.evaluate("window.__fit.arrangedEdges()")
    page.click("[data-testid=view-front]")
    page.wait_for_timeout(900)
    picked = None
    for piece in edges:
        for eid in piece["ids"]:
            pos = page.evaluate(f"window.__fit.edgeClient('{piece['piece']}', '{eid}')")
            if pos is None:
                continue
            page.mouse.click(pos["x"], pos["y"])
            sel = page.evaluate("window.__fit.selection()")
            if sel and sel["kind"] == "edge":
                picked = sel
                break
        if picked:
            break
    assert picked, "no edge could be picked"
    text = page.inner_text("[data-testid=selection]")
    assert picked["id"] in text and f"{picked['length_m'] * 100:.2f}" in text
    with page.expect_response(lambda r: r.url.endswith("/api/align"), timeout=240_000):
        page.click("[data-testid=arrange-reset]")
    page.wait_for_function("!window.__fit.arrangeState().arranged")
    # a copy of a copy, then deleting the first copy asks first and takes its dependant along
    page.click("[data-testid=piece-ShortSleeve_copy]")
    with page.expect_response(lambda r: r.url.endswith("/api/clone")):
        page.click("[data-testid=copy-open]")
        page.select_option("[data-testid=copy-place]", "none")
        page.click("[data-testid=copy-create]")
    assert page.evaluate("window.__fit.state().derived")[-1] == "ShortSleeve_copy_copy"
    seen: list[str] = []
    page.once("dialog", lambda d: (seen.append(d.message), d.accept()))
    page.click("[data-testid=piece-ShortSleeve_copy]")
    page.click("[data-testid=derived-delete]")
    assert "ShortSleeve_copy_copy" in seen[0]
    assert page.evaluate("window.__fit.state().derived") == ["ShortSleeve_copy3"]


def test_symmetric_copy_on_the_other_side_has_no_handedness_hint(page: Any, server: str) -> None:
    fresh(page, server)
    page.click("[data-testid=piece-Cuff]")
    click_point(page, "Forearm_Outer_R")
    page.click("[data-testid=copy-Cuff]")  # the piece list button
    page.wait_for_selector("[data-testid=copy-dialog]")
    with page.expect_response(lambda r: r.url.endswith("/api/align")) as resp:
        page.click("[data-testid=copy-create]")
    placements = {p["piece"]: p for p in resp.value.json()["placements"]}
    assert placements["Cuff_copy"]["point"] == "Forearm_Outer_L" and placements["Cuff_copy"]["mirrored"] is False
    page.wait_for_selector("[data-testid=copy-of]")
    page.wait_for_timeout(300)
    assert page.query_selector("[data-testid=copy-handedness]") is None

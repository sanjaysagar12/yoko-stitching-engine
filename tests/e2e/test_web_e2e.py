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


# -- assembly import / export, image export ----------------------------------------------------------------
MINI = {
    "schema_version": 1,
    "garment": "mini_shirt",
    "job": "male_shirt",
    "notes": "two front halves and a pocket",
    "mode": "arrange",
    "derived_pieces": [
        {
            "id": "FrontPanel_L",
            "source": "FrontPanel",
            "op": "mirror",
            "axis": {"type": "edge", "from_point": "A33", "to_point": "A32"},
        }
    ],
    "placements": [
        {"piece": "FrontPanel", "point": "Chest_Front", "reference": "A32", "wrap": "roll"},
        {"piece": "FrontPanel_L", "point": "Chest_Front", "reference": "A32", "wrap": "roll"},
        {"piece": "Pocket", "point": "Chest_FrontSide_L", "wrap": "roll"},
    ],
    "expect": {"required_pieces": ["FrontPanel", "FrontPanel_L", "Pocket"]},
}


def mini_file(tmp_path: Path, doc: Any = None, name: str = "mini_shirt.assembly.json") -> str:
    f = tmp_path / name
    f.write_text(json.dumps(MINI if doc is None else doc), encoding="utf-8")
    return str(f)


def import_file(page: Any, path: str) -> None:
    page.set_input_files("[data-testid=import-assembly-input]", path)


def wait_assembled(page: Any) -> str:
    page.wait_for_function(
        "document.querySelector('[data-testid=assembly-summary]') && window.__fit.status().valid !== null"
        " && window.__fit.ready()",
        timeout=180_000,
    )
    return str(page.inner_text("[data-testid=assembly-summary]"))


def test_import_assembly_builds_assembles_and_reports(page: Any, server: str, tmp_path: Path) -> None:
    page.goto(server)  # nothing built yet: the job comes from the file
    page.wait_for_function("document.querySelectorAll('[data-testid=job-select] option').length >= 4")
    assert page.is_disabled("[data-testid=export-assembly]") and page.is_disabled("[data-testid=export-images]")
    import_file(page, mini_file(tmp_path))
    text = wait_assembled(page)
    assert text == "Assembly: mini_shirt (male_shirt, 3 pieces, 1 derived) - valid"
    st = page.evaluate("window.__fit.state()")
    assert [p["piece"] for p in st["placements"]] == ["FrontPanel", "FrontPanel_L", "Pocket"]
    assert st["derived"] == ["FrontPanel_L"] and all(p.get("solved") for p in st["placements"])
    assert page.evaluate("window.__fit.arrangeState().arranged") is True
    assert page.input_value("[data-testid=job-select]") == "male_shirt"
    assert page.is_enabled("[data-testid=export-assembly]") and page.is_enabled("[data-testid=export-images]")
    page.click("[data-testid=assembly-banner] summary")
    assert "two front halves" in page.inner_text("[data-testid=assembly-notes]")
    page.click("[data-testid=assembly-banner] details:nth-of-type(2) summary")
    assert "required pieces (3)" in page.inner_text("[data-testid=assembly-expect]")
    assert page.evaluate("window.__fit.status().valid") is True


def test_export_assembly_downloads_a_valid_file(page: Any, server: str, tmp_path: Path) -> None:
    from assembler.run import load_assembly

    fresh(page, server)
    import_file(page, mini_file(tmp_path))
    wait_assembled(page)
    page.click("[data-testid=export-assembly]")
    assert page.input_value("[data-testid=export-garment]") == "mini_shirt"
    assert "two front halves" in page.input_value("[data-testid=export-notes]")
    page.fill("[data-testid=export-garment]", "my_shirt")
    with page.expect_download() as dl:
        page.click("[data-testid=export-assembly-go]")
    assert dl.value.suggested_filename == "my_shirt.assembly.json"
    out = tmp_path / "exported.assembly.json"
    dl.value.save_as(out)
    text = out.read_bytes().decode("utf-8")
    assert "\r" not in text and text.endswith("\n")
    doc = load_assembly(out)  # python -m assembler check
    assert doc["garment"] == "my_shirt" and doc["job"] == "male_shirt" and doc["expect"] == MINI["expect"]
    assert [p["piece"] for p in doc["placements"]] == ["FrontPanel", "FrontPanel_L", "Pocket"]
    assert all("solved" in p for p in doc["placements"]) and doc["derived_pieces"][0]["id"] == "FrontPanel_L"
    # the exported file imports again
    page.on("dialog", lambda d: d.accept())
    import_file(page, str(out))
    page.wait_for_function("document.querySelector('[data-testid=assembly-summary]').textContent.includes('my_shirt')")
    assert wait_assembled(page).startswith("Assembly: my_shirt")


def test_export_images_downloads_a_zip(page: Any, server: str, tmp_path: Path) -> None:
    import io
    import zipfile

    fresh(page, server)
    import_file(page, mini_file(tmp_path))
    wait_assembled(page)
    page.click("[data-testid=export-images]")
    page.uncheck("[data-testid=images-closeups]")
    page.check("[data-testid=images-viewport]")
    with page.expect_download(timeout=600_000) as dl:
        page.click("[data-testid=export-images-go]")
    assert dl.value.suggested_filename == "mini_shirt_images.zip"
    out = tmp_path / "images.zip"
    dl.value.save_as(out)
    names = set(zipfile.ZipFile(io.BytesIO(out.read_bytes())).namelist())
    assert len(names) >= 12 and {"sheet.png", "sections.png", "front.png", "assembly.json"} <= names
    assert {"viewport/front.png", "viewport/top.png"} <= names


def test_bad_file_keeps_the_layout_and_revert_goes_back(page: Any, server: str, tmp_path: Path) -> None:
    page.on("dialog", lambda d: d.accept())  # "Replace the current layout?"
    fresh(page, server)
    page.click("[data-testid=piece-FrontPanel]")
    click_point(page, "Chest_Front")
    before = page.evaluate("window.__fit.state().placements")
    assert [p["piece"] for p in before] == ["FrontPanel"]
    import_file(page, mini_file(tmp_path))
    wait_assembled(page)
    assert len(page.evaluate("window.__fit.state().placements")) == 3
    # a bad file: the error lists every problem and nothing changes
    bad = {
        **MINI,
        "placements": [{**MINI["placements"][0], "piece": "Ghost"}, {**MINI["placements"][2], "point": "Nope"}],
    }
    import_file(page, mini_file(tmp_path, bad, "bad.assembly.json"))
    page.wait_for_selector("[data-testid=error-problems]")
    problems = page.inner_text("[data-testid=error-problems]")
    assert "Ghost" in problems and "Nope" in problems
    names = [p["piece"] for p in page.evaluate("window.__fit.state().placements")]
    assert names == ["FrontPanel", "FrontPanel_L", "Pocket"]
    page.click("[data-testid=import-skip-missing]")  # leaves Ghost out, but the unknown point stays an error
    page.wait_for_function(
        "(t => t.includes('Nope') && !t.includes('Ghost'))"
        "(document.querySelector('[data-testid=error-problems]')?.textContent ?? '')"
    )
    # not JSON at all
    junk = tmp_path / "junk.json"
    junk.write_text("{oops", encoding="utf-8")
    import_file(page, str(junk))
    page.wait_for_function("document.querySelector('[data-testid=error-panel]').textContent.includes('not valid JSON')")
    assert len(page.evaluate("window.__fit.state().placements")) == 3
    # revert: the FrontPanel alone again
    page.click("[data-testid=assembly-revert]")
    page.wait_for_function("window.__fit.state().placements.length === 1 && window.__fit.ready()")
    assert page.evaluate("window.__fit.state().placements")[0]["piece"] == "FrontPanel"
    assert page.evaluate("window.__fit.state().derived") == []
    # re-assemble from the file brings the mini shirt back
    import_file(page, mini_file(tmp_path))
    wait_assembled(page)
    page.click("[data-testid=assembly-reassemble]")
    page.wait_for_function("window.__fit.ready() && window.__fit.status().valid !== null")
    assert len(page.evaluate("window.__fit.state().placements")) == 3

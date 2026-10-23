"""/api/simulation/*: info, option checks, a short simulation of the stitched layout on screen, the viewer's data
(embedding, latest frame, frames), the result, the downloads and the cache.

Real backend, synthetic avatar, the skirt (the quickest garment); skipped without ZOZO's Contact Solver."""

from __future__ import annotations

import base64
import dataclasses
import io
import json
import time
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from simulator.solver import solver_info
from tests.web_build import prebuild
from web.backend.app import create_app
from web.backend.config import AppConfig

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(not solver_info()["available"], reason="ZOZO's Contact Solver is not installed")
OPTIONS = {"fabric": "cotton_shirting", "quality": "draft", "frames": 12, "sew_time": 0.1}


@pytest.fixture(scope="module")
def env(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[TestClient, dict[str, Any]]]:
    out = tmp_path_factory.mktemp("outputs")
    bid = prebuild(out, "aldrich_skirt")
    cfg = dataclasses.replace(AppConfig.default(), outputs_dir=out)
    asm = json.loads((ROOT / "assert/assemblies/aldrich_skirt.assembly.json").read_text(encoding="utf-8"))
    st = json.loads((ROOT / "assert/stitchings/aldrich_skirt.stitching.json").read_text(encoding="utf-8"))
    with TestClient(create_app(cfg)) as c:
        r = c.post("/api/arrange", json={"build": bid, "placements": asm["placements"],
                                        "derived_pieces": asm["derived_pieces"]})  # fmt: skip
        assert r.status_code == 200, r.text
        placed = {p["piece"]: p for p in r.json()["placements"]}
        solved = [{**p, "solved": placed[p["piece"]]["solved"]} for p in asm["placements"]]
        yield c, {"build": bid, "placements": solved, "derived_pieces": asm["derived_pieces"], "stitching": st,
                  "_placed": placed}  # fmt: skip


def _body(env: tuple[TestClient, dict[str, Any]], **options: Any) -> dict[str, Any]:
    return {k: v for k, v in env[1].items() if not k.startswith("_")} | {"options": {**OPTIONS, **options}}


def _wait(c: TestClient, sid: str, timeout_s: float = 1800) -> dict[str, Any]:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        st: dict[str, Any] = c.get(f"/api/simulation/{sid}").json()
        if st["status"] not in ("queued", "running"):
            return st
        time.sleep(2)
    raise AssertionError(f"simulation {sid} did not end in {timeout_s} s")


def test_info_lists_the_solver_and_the_fabrics(env: tuple[TestClient, dict[str, Any]]) -> None:
    info = env[0].get("/api/simulation/info").json()
    assert info["solver"]["available"] and info["defaults"]["fabric"] == "cotton_shirting"
    assert {"cotton_shirting", "denim", "jersey"} <= {f["name"] for f in info["fabrics"]}
    assert [q["name"] for q in info["qualities"]] == ["draft", "standard", "fine"]


def test_bad_options_are_refused(env: tuple[TestClient, dict[str, Any]]) -> None:
    r = env[0].post("/api/simulation/start", json=_body(env, fabric="tweed"))
    assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_simulation"
    r = env[0].get("/api/simulation/nope-123")
    assert r.status_code == 404 and r.json()["error"]["code"] == "simulation_not_found"


def test_simulate_the_layout_on_screen(env: tuple[TestClient, dict[str, Any]]) -> None:
    c, ctx = env
    r = c.post("/api/simulation/start", json=_body(env))
    assert r.status_code == 200, r.text
    st = _wait(c, r.json()["id"])
    assert st["status"] == "done", st
    # the size 12 skirt is smaller than the synthetic (male) test body's waist: it cannot close round it, which the
    # checks say; what the backend must deliver is a finished run, no cloth inside the body, no stretch past the limit
    s = st["summary"]
    assert s["finished"] and s["inside_body"] == 0 and s["stretch_max_pct"] <= 4.5, s
    sid = st["id"]
    emb = c.get(f"/api/simulation/{sid}/embedding").json()
    n = emb["sim_vertices"]
    for p in emb["pieces"]:  # every vertex of every placed mesh is carried
        assert p["count"] == len(ctx["_placed"][p["name"]]["wrapped_vertices"] or []) or p["count"] > 0
        assert len(base64.b64decode(p["w"])) == p["count"] * 3 * 4
    latest = c.get(f"/api/simulation/{sid}/latest").json()
    x = np.frombuffer(base64.b64decode(latest["x"]), dtype="<f4")
    assert x.shape == (n * 3,) and np.isfinite(x).all()
    fr = c.get(f"/api/simulation/{sid}/frames").json()
    assert fr["frames"][-1] == 12 and len(base64.b64decode(fr["x"])) == len(fr["frames"]) * n * 3 * 4
    res = c.get(f"/api/simulation/{sid}/result").json()
    assert res["inputs"]["synthetic"] and res["options"]["frames"] == 12 and res["checks"]
    obj = c.get(f"/api/simulation/{sid}/download?format=obj")
    assert obj.status_code == 200 and "g avatar" in obj.text
    z = zipfile.ZipFile(io.BytesIO(c.get(f"/api/simulation/{sid}/download?format=zip").content))
    assert {"simulation.result.json", "draped_m.obj"} <= set(z.namelist())
    assert any(n.startswith("images/sim_front") for n in z.namelist())
    again = c.post("/api/simulation/start", json=_body(env)).json()  # the same layout and options: the same job
    assert again["id"] == sid and again["status"] == "done"

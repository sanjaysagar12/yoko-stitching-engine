"""Every piece at every arrangement point it can go on (synthetic body, no SMPL-X): properly aligned or a clear
violation, never a silent valid result far from its point.

Pieces: the shirt, the trousers and the Aldrich skirt (``assert/jobs``, built with sm2d_engine). Points by body class:
torso panels on chest / waist / hip / neck points, sleeves and cuffs on arm points, trousers on leg, hip and waist
points, skirts on waist and hip points, collars on neck points, small pieces (pockets, plackets, bands) anywhere. Each
piece alone, ``roll`` and ``flat``. By default one representative point per side and height; ``SWEEP=full`` runs
every point. Invariants (``docs/alignment_limbs.md``):

* I4 sizes exact; I8 no limit hit on a valid result; I2 clearance on a valid result, and its nearest part within
  ``max_standoff_from_point`` of the body; I3 the piece's up (local +y) along the point's up; a rolled limb piece
  wraps its limb (radius <= 15 cm);
* an invalid result names what failed, with a hint for STANDOFF_EXCEEDED / NOT_AT_POINT;
* I5 a piece on an ``_R`` point and its mirror clone on the ``_L`` point are mirror images (1 mm);
* I9 the same call twice gives the same answer.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from alignment_engine import AlignmentEngine
from alignment_engine.loaders import read_avatar
from alignment_engine.verify import AvatarSurface, surface_samples
from tests.test_layout import CLEAR, body_lower_bound

ROOT = Path(__file__).resolve().parents[1]
FULL = os.environ.get("SWEEP", "") == "full"
FLIP = np.array([-1.0, 1.0, 1.0])
MAX_FROM_POINT = 0.12
SMALL = 0.26  # pieces no larger than this (both sides) go anywhere

TORSO = ["Chest_Front", "Chest_Right", "Waist_BackSide_R", "Hip_Front"]
NECK = ["Neck_Front"]
ARM = ["UpperArm_Outer_R", "Forearm_Inner_R"]
LEG = ["Thigh_Front_R", "Shin_Back_R"]
WAIST_HIP = ["Waist_Front", "Hip_FrontSide_R"]
ANYWHERE = ["Chest_FrontSide_L", "UpperArm_Outer_R"]
MIRRORED = ["UpperArm_Outer_R", "Forearm_Front_R", "Thigh_Front_R", "Chest_FrontSide_R", "Hip_BackSide_R"]


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory, shirt_pieces: Path, refine_cache: Path) -> dict[str, Any]:
    from sm2d_engine.engine import Engine, EngineOptions, EngineRequest
    from tests.synthetic_avatar import build_avatar

    d = tmp_path_factory.mktemp("sweep")
    av = build_avatar(d / "avatar")
    folders = {"male_shirt": shirt_pieces}
    for job in ("trousers", "aldrich_skirt"):
        j = json.loads((ROOT / f"assert/jobs/{job}.json").read_text())
        sm2d = (ROOT / "assert/sm2d" / j["sm2d"]).read_text(encoding="utf-8-sig")
        smis = (ROOT / "assert/smis" / j["smis"]).read_text(encoding="utf-8-sig")
        Engine().run(
            EngineRequest(sm2d, smis, EngineOptions(output_dir=d / job, pieces=j["pieces"]), j["sm2d"], j["smis"])
        )
        folders[job] = d / job
    a = read_avatar(av)
    return {
        "avatar": av,
        "av": a,
        "folders": folders,
        "cache": refine_cache,
        "tmp": d,
        "surface": AvatarSurface(a.vertices, a.faces),
        "engine": AlignmentEngine(),
    }


def pieces_of(folder: Path) -> dict[str, dict[str, Any]]:
    return {b["name"]: b for b in (json.loads(p.read_text()) for p in sorted(folder.glob("*.piece.json")))}


def points_for(job: str, name: str, bundle: dict[str, Any], all_points: list[str]) -> list[str]:
    """The points a piece can sensibly go on (the full list, or the representative subset)."""
    w = bundle["bbox_m"]["max"][0] - bundle["bbox_m"]["min"][0]
    h = bundle["bbox_m"]["max"][1] - bundle["bbox_m"]["min"][1]

    def pick(prefixes: tuple[str, ...], subset: list[str]) -> list[str]:
        return [p for p in all_points if p.startswith(prefixes)] if FULL else subset

    if name.startswith("Collar"):
        return pick(("Neck_",), NECK)
    if name.endswith("Sleeve") or name.startswith("Cuff"):
        return pick(("UpperArm_", "Forearm_"), ARM)
    if job == "aldrich_skirt" or name.startswith("WaistBand") or name.startswith("Waistband"):
        return pick(("Waist_", "Hip_"), WAIST_HIP)
    if job == "trousers" and h > 0.8:
        return pick(("Thigh_", "Shin_", "Waist_", "Hip_"), LEG + WAIST_HIP)
    if max(w, h) <= SMALL:
        return pick(("Chest_", "Waist_", "Hip_", "UpperArm_", "Forearm_", "Thigh_", "Shin_"), ANYWHERE)
    return pick(("Chest_", "Waist_", "Hip_", "Neck_"), TORSO + NECK)


# piece names per job are only known after sm2d_engine has run: the sweep is one test per (job, piece, mode)
JOB_PIECES = {
    "male_shirt": [
        "FrontPanel", "BackPanel", "Yoke", "ShortSleeve", "FullSleeve", "Cuff", "CuffInterface", "CollarBase",
        "CollarBaseInterface", "CollarTop", "CollarTopInterface", "PlacketTop", "PlacketUnder", "Pocket",
        "PocketFlap", "PocketRound", "PocketFlapRound",
    ],
    "trousers": [
        "Back", "Front_no_pocket", "Front_Pleats", "Front_with_pocket", "Detail", "PocketBack", "PocketFront",
        "PocketMouth", "PocketPanel", "WaistBand1", "Waistband2",
    ],
    "aldrich_skirt": ["A - Skirt Back", "A - Skirt Front"],
}  # fmt: skip
# same shape as another piece of the list: only in the full sweep
TWINS = {"CuffInterface", "CollarBaseInterface", "CollarTopInterface", "PocketRound", "PocketFlapRound",
         "Front_no_pocket", "Front_with_pocket", "PocketMouth"}  # fmt: skip
CASES = [
    (job, piece, mode)
    for job, ps in JOB_PIECES.items()
    for piece in ps
    if FULL or piece not in TWINS
    for mode in ("roll", "flat")
]


def align(
    world: dict[str, Any], job: str, pls: list[dict[str, Any]], derived: list[dict[str, Any]] | None = None
) -> Any:
    text = json.dumps({"schema_version": 4, "placements": pls, "derived_pieces": derived or []})
    return world["engine"].align(
        world["avatar"],
        world["folders"][job],
        text,
        derived_dir=world["tmp"] / f"derived_{job}",
        cache_dir=world["cache"],
    )


def check(world: dict[str, Any], res: Any, where: str) -> None:
    """The invariants of one single-piece answer."""
    v = res.doc["validity"]
    d = res.doc["placements"][0]
    a = d["arrange"]
    p = res.placed[0]
    assert v["size_ok"] and d["changes_piece_size"] is False, where  # I4
    if not v["valid"]:  # a clear violation, never silent
        assert v["violations"] and res.warnings[0].startswith("ArrangementInfeasible"), where
        for x in v["violations"]:
            if x["kind"] in ("STANDOFF_EXCEEDED", "NOT_AT_POINT"):
                assert x.get("hint"), (where, x)
        return
    assert a["limits"] == [] and a["flags"] == [], (where, a["limits"], a["flags"])  # I8
    assert body_lower_bound(world["surface"], p) >= CLEAR - 0.001, where  # I2
    nearest = float(world["surface"].distance(surface_samples(p.vertices, p.faces, 0.01)).min())
    assert nearest <= MAX_FROM_POINT, (where, nearest)  # I1: never floating away from the body
    point = world["av"].points[d["point"]]
    ref = np.array([0.0, 0.0])
    up_world = p.transform(np.array([[0.0, 0.01]]) + ref) - p.transform(ref[None, :])
    assert float(up_world[0] @ point.up) > 0.0, (where, "upside down")  # I3
    if a["ring"].startswith(("arm_", "leg_")):
        assert a["radius_m"] <= 0.15, (where, a["radius_m"])  # wraps the limb, not the torso


@pytest.mark.parametrize(("job", "piece", "mode"), CASES, ids=[f"{j}:{p}:{m}" for j, p, m in CASES])
def test_every_point(world: dict[str, Any], job: str, piece: str, mode: str) -> None:
    bundles = pieces_of(world["folders"][job])
    assert piece in bundles, (piece, sorted(bundles))
    for point in points_for(job, piece, bundles[piece], list(world["av"].points)):
        res = align(world, job, [{"piece": piece, "point": point, "wrap": mode}])
        check(world, res, f"{piece}@{point}:{mode}")


MIRROR_CASES = [("male_shirt", "FullSleeve"), ("male_shirt", "Pocket"), ("trousers", "Front_Pleats"),
                ("aldrich_skirt", "A - Skirt Front")]  # fmt: skip


@pytest.mark.parametrize(("job", "piece"), MIRROR_CASES, ids=[p for _, p in MIRROR_CASES])
def test_left_and_right_are_mirror_images(world: dict[str, Any], job: str, piece: str) -> None:
    clone = piece.replace(" ", "_").replace("-", "_") + "__M"
    derived = [{"id": clone, "source": piece, "op": "mirror", "axis": {"type": "bbox_center"}}]
    for right in MIRRORED if FULL else MIRRORED[:3]:
        left = right[:-2] + "_L"
        a = align(world, job, [{"piece": piece, "point": right, "wrap": "roll"}], derived)
        b = align(world, job, [{"piece": clone, "point": left, "wrap": "roll"}], derived)
        assert a.doc["validity"]["valid"] == b.doc["validity"]["valid"], (piece, right)
        if not a.doc["validity"]["valid"]:
            continue
        # exact when the solver had nothing to do; its iterations break ties by vertex order, which a mirrored mesh
        # does not keep, so a piece the solver slid far may differ by millimetres (docs/alignment_limbs.md)
        tol = 0.001 if a.doc["placements"][0]["arrange"]["slide_m"] < 0.005 else 0.005
        wa = {n["name"]: np.array(n["world_m"]) for n in a.doc["placements"][0]["world_named_points"]}
        wb = {n["name"]: np.array(n["world_m"]) * FLIP for n in b.doc["placements"][0]["world_named_points"]}
        assert max(float(np.abs(wa[k] - wb[k]).max()) for k in wa) < tol, (piece, right)


def test_deterministic(world: dict[str, Any]) -> None:
    pls = [{"piece": "FullSleeve", "point": "Forearm_Outer_R", "wrap": "roll"}]
    first = align(world, "male_shirt", pls).doc
    again = AlignmentEngine().align(
        world["avatar"], world["folders"]["male_shirt"], json.dumps({"schema_version": 4, "placements": pls}),
        derived_dir=world["tmp"] / "derived_det", cache_dir=world["cache"],
    ).doc  # fmt: skip
    assert first["placements"][0]["solved"] == again["placements"][0]["solved"]
    assert first["validity"] == again["validity"]


def test_every_point_lies_on_its_volume(world: dict[str, Any]) -> None:
    """R8: points are on the surface of their bounding volume (the synthetic legs once resolved 3 to 11 m away)."""
    av = world["av"]
    for name, p in av.points.items():
        v = av.volumes[p.bv]
        d = v.end - v.start
        k = float(np.clip((p.position - v.start) @ d / (d @ d), 0.0, 1.0))
        foot = v.start + k * d
        r = float(np.linalg.norm(p.position - foot))
        assert r <= 0.4 and r <= 3.0 * v.radius_m + p.offset_m, (name, r, v.radius_m)
        assert float(p.normal @ (p.position - foot)) > 0, name
        sd = float(world["surface"].distance(p.position[None, :])[0])
        assert sd == pytest.approx(p.offset_m, abs=0.002), (name, sd)

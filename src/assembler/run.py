"""``assembler run``: assembly JSON -> cached inputs -> engine runs -> result, meshes, verification, images.

Runs (all on the same avatar and pieces):

* ``primary``: ``arrange`` (or ``align`` for ``mode: incremental``) of the whole spec;
* ``replay``: the web app's way, one placement at a time with every earlier ``solved`` block carried, in the listed
  order; ``replay_reversed`` the same in reverse order;
* ``carry``: ``align`` of the primary result with all ``solved`` blocks: nothing may move by more than 1 mm;
* ``repeat``: the primary run again in a fresh engine: the result must be identical (determinism).
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from alignment_engine import AlignmentEngine
from alignment_engine.contracts import validate
from alignment_engine.engine import AlignmentResult
from alignment_engine.levels import body_levels
from alignment_engine.loaders import read_avatar

from . import render as R
from . import verify as V

ROOT = Path(__file__).resolve().parents[2]
CARRY_TOL = 0.001
SECTION_LEVELS = ("Chest", "Waist", "Hip", "Crotch", "Knee")


class AssemblyError(ValueError):
    """The assembly file is invalid or names something that does not exist."""


# -- inputs (cached by content) ---------------------------------------------------------------------------------------
def _src_hash(pkg: str) -> str:
    h = hashlib.sha256()
    for p in sorted((ROOT / "src" / pkg).rglob("*.py")):
        h.update(p.relative_to(ROOT).as_posix().encode())
        h.update(p.read_bytes())
    for p in sorted((ROOT / "src" / pkg).rglob("*.yaml")):
        h.update(p.read_bytes())
    return h.hexdigest()


def _key(*parts: bytes | str) -> str:
    h = hashlib.sha256()
    for part in parts:
        b = part.encode() if isinstance(part, str) else part
        h.update(len(b).to_bytes(8, "big"))
        h.update(b)
    return h.hexdigest()[:16]


@dataclass
class Inputs:
    """Built inputs of a job: piece folder, avatar folder, the body JSON (None: synthetic) and the avatar mesh hash."""

    pieces: Path
    avatar: Path
    body: dict[str, Any] | None
    avatar_hash: str
    cached: dict[str, bool]


def build_inputs(job_name: str, synthetic: bool, cache: Path) -> Inputs:
    """Pieces (sm2d_engine) and avatar (avatar_engine, or the synthetic test body) of a job, cached by content."""
    job_file = ROOT / "assert" / "jobs" / f"{job_name}.json"
    if not job_file.is_file():
        raise AssemblyError(f"no job {job_name!r} ({job_file})")
    job = json.loads(job_file.read_text(encoding="utf-8-sig"))
    sm2d = (ROOT / "assert" / "sm2d" / job["sm2d"]).read_bytes()
    smis = (ROOT / "assert" / "smis" / job["smis"]).read_bytes()
    body_b = (ROOT / "assert" / "body" / job["body"]).read_bytes()
    cached = {}
    pk = _key(sm2d, smis, json.dumps(job.get("pieces")), _src_hash("sm2d_engine"))
    pdir = cache / f"pieces-{job_name}-{pk}"
    cached["pieces"] = (pdir / "done").is_file()
    if not cached["pieces"]:
        from sm2d_engine import Engine, EngineOptions, EngineRequest

        Engine().run(
            EngineRequest(
                sm2d.decode("utf-8-sig"),
                smis.decode("utf-8-sig"),
                EngineOptions(output_dir=pdir, pieces=job.get("pieces")),
                job["sm2d"],
                job["smis"],
            )
        )
        (pdir / "done").write_text("ok\n", encoding="utf-8")
    if synthetic:
        ak = _key("synthetic", _src_hash("avatar_engine"), (ROOT / "tests" / "synthetic_avatar.py").read_bytes())
        adir = cache / f"avatar-synthetic-{ak}"
    else:
        ak = _key(
            body_b,
            _src_hash("avatar_engine"),
            (ROOT / "models" / "smplx" / "version.txt").read_bytes()
            if (ROOT / "models" / "smplx" / "version.txt").is_file()
            else b"",
        )
        adir = cache / f"avatar-{Path(job['body']).stem}-{ak}"
    cached["avatar"] = (adir / "avatar_hash.txt").is_file()
    if not cached["avatar"]:
        if synthetic:
            import importlib
            import sys

            if str(ROOT) not in sys.path:  # the CI body lives with the tests
                sys.path.insert(0, str(ROOT))
            importlib.import_module("tests.synthetic_avatar").build_avatar(adir)
        else:
            from avatar_engine import AvatarEngine

            AvatarEngine().build(body_b.decode("utf-8-sig"), ROOT / "models" / "smplx", adir)
        (adir / "avatar_hash.txt").write_text(V.mesh_hash(read_avatar(adir).vertices) + "\n", encoding="utf-8")
    body = None if synthetic else json.loads(body_b.decode("utf-8-sig"))
    return Inputs(pdir, adir, body, (adir / "avatar_hash.txt").read_text(encoding="utf-8").strip(), cached)


# -- the spec ----------------------------------------------------------------------------------------------------------
def check_doc(doc: dict[str, Any]) -> None:
    """Validate an assembly document (assembly schema; its placement spec against placement_spec; real size kept)."""
    try:
        validate("assembly", doc)
        validate("placement_spec", spec_of(doc))
    except ValueError as e:
        raise AssemblyError(str(e)) from None
    if (doc.get("arrange_options") or {}).get("allow_distortion"):
        raise AssemblyError("arrange_options.allow_distortion must stay false: pieces keep their real size")
    if any(pl.get("wrap") == "body_slice" for pl in doc["placements"]):
        raise AssemblyError("wrap body_slice changes piece sizes and is not allowed in an assembly")


def load_assembly(path: Path) -> dict[str, Any]:
    """Read and validate an assembly file (assembly schema; its placement spec against placement_spec)."""
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as e:
        raise AssemblyError(f"{path}: invalid JSON: {e}") from None
    check_doc(doc)
    out: dict[str, Any] = doc
    return out


def spec_of(doc: dict[str, Any], placements: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """The placement spec (schema_version 5) of an assembly."""
    spec: dict[str, Any] = {
        "schema_version": 5,
        "placements": placements if placements is not None else doc["placements"],
    }
    for k in ("derived_pieces", "arrange_options", "stitches", "rings"):
        if doc.get(k):
            spec[k] = doc[k]
    return spec


# -- engine runs -------------------------------------------------------------------------------------------------------
@dataclass
class Run:
    name: str
    result: AlignmentResult
    spec: dict[str, Any]
    seconds: float


def _solve(
    eng: AlignmentEngine,
    inp: Inputs,
    spec: dict[str, Any],
    derived: Path,
    refined: Path,
    mode: str,
    out_dir: Path | None = None,
) -> AlignmentResult:
    f = eng.arrange if mode == "arrange" else eng.align
    return f(inp.avatar, inp.pieces, json.dumps(spec), out_dir=out_dir, derived_dir=derived, cache_dir=refined)


def replay(doc: dict[str, Any], inp: Inputs, order: list[dict[str, Any]], derived: Path, refined: Path) -> Run:
    """One placement at a time, every earlier ``solved`` block carried (as the web app does)."""
    t0 = time.perf_counter()
    eng = AlignmentEngine()
    done: list[dict[str, Any]] = []
    res = None
    for pl in order:
        fresh = {k: v for k, v in pl.items() if k != "solved"}
        spec = spec_of(doc, [*done, fresh])
        res = _solve(eng, inp, spec, derived, refined, "align")
        done = [
            {**p, "solved": d["solved"]} if d.get("solved") else dict(p)
            for p, d in zip([*done, fresh], res.doc["placements"], strict=True)
        ]
    assert res is not None
    return Run("", res, spec_of(doc, done), time.perf_counter() - t0)


def _fingerprint(res: AlignmentResult) -> str:
    d = dict(res.doc)
    d["arrangement"] = {**d["arrangement"], "timing_s": 0.0}
    h = hashlib.sha256(json.dumps(d, sort_keys=True).encode())
    for p in res.placed:
        h.update(np.ascontiguousarray(p.vertices).tobytes())
    return h.hexdigest()[:16]


# -- verification --------------------------------------------------------------------------------------------------
def run_view(name: str, res: AlignmentResult, spec: dict[str, Any], bundles: dict[str, dict[str, Any]]) -> V.RunView:
    pieces = []
    for p, d, pl in zip(res.placed, res.doc["placements"], spec["placements"], strict=True):
        assert p.flat is not None
        pieces.append(V.PieceView(p.name, p.point, p.vertices, p.flat, p.faces, bundles[p.name], pl, d))
    return V.RunView(name, pieces, res.doc, res.doc["arrangement"]["options"])


def bundles_of(inp: Inputs, derived: Path) -> dict[str, dict[str, Any]]:
    out = {}
    for folder in (inp.pieces, derived):
        for f in sorted(Path(folder).glob("*.piece.json")):
            b = json.loads(f.read_text(encoding="utf-8"))
            out[b["name"]] = b
    return out


def verify_run(
    rv: V.RunView,
    body: V.Surf,
    points: dict[str, Any],
    levels: dict[str, float],
    avatar: Any,
    inp: Inputs,
    expect: dict[str, Any],
) -> dict[str, Any]:
    V.check_avatar(
        rv.rows, avatar.vertices, inp.avatar_hash, float(inp.body["measurements"]["height"]) if inp.body else None
    )
    V.check_size(rv)
    near = V.check_clearance(rv, body, points, expect)
    vols = {v["name"]: v for v in json.loads((inp.avatar / "bounding_volumes.json").read_text("utf-8"))["volumes"]}
    V.check_orientation(rv, points, near, expect, vols)
    fit = V.check_registration(rv, levels, expect)
    V.check_pairs(rv, points, expect)
    ease = V.check_semantics(rv, levels, avatar.vertices, avatar.faces, expect)
    V.check_honesty(rv)
    return {"near": near, "fit": fit, "ease": ease}


# -- outputs -----------------------------------------------------------------------------------------------------------
def _glb(path: Path, avatar: Any, placed: list[Any]) -> None:
    import trimesh

    scene = trimesh.Scene()
    scene.add_geometry(trimesh.Trimesh(avatar.vertices, avatar.faces, process=False), geom_name="avatar")
    for i, p in enumerate(placed):
        m = trimesh.Trimesh(p.vertices, p.faces, process=False)
        m.visual.face_colors = [*R.piece_colour(i), 255]  # type: ignore[union-attr]
        scene.add_geometry(m, geom_name=p.name)
    path.write_bytes(scene.export(file_type="glb"))  # type: ignore[no-untyped-call]


def images(
    out: Path,
    garment: str,
    it_name: str,
    res: AlignmentResult,
    avatar: Any,
    levels: dict[str, float],
    near: dict[str, Any] | None,
    options: dict[str, Any],
    expect: dict[str, Any],
    views: list[str] | None,
    points: dict[str, Any],
    closeups: bool = True,
    heat: bool = True,
    progress: Callable[[str, float], None] | None = None,
    failures: list[str] | None = None,
) -> dict[str, Path]:
    """Render every view, the heat maps, sections, close-ups and the contact sheet into ``out``.

    ``near`` (closest body points per piece) orients the close-ups; None computes it. ``progress(name, fraction)`` is
    called before each image (it may raise to cancel). With ``failures`` a failing image is listed there instead of
    stopping the others.
    """
    img = out / "images"
    av = R.Mesh("avatar", avatar.vertices, avatar.faces, R.AVATAR)
    pieces = [R.Mesh(p.name, p.vertices, p.faces, R.piece_colour(i)) for i, p in enumerate(res.placed)]
    scene = R.Scene(av, pieces)
    title = f"{garment} {it_name}"
    paths: dict[str, Path] = {}
    shown: list[tuple[str, Any]] = []
    todo = len(R.VIEWS) + 3 + (2 if heat else 0) + (len(pieces) if closeups else 0)
    count = 0

    def step(name: str, fn: Callable[[], Any]) -> Any:
        nonlocal count
        if progress:
            progress(name, count / todo)
        count += 1
        if failures is None:
            return fn()
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - one broken image must not lose the others
            failures.append(f"{name}: {type(e).__name__}: {e}")
            return None

    for v in R.VIEWS:
        if views and v.name not in views:
            continue
        im = step(v.name, lambda v=v: R.render_view(scene, v, f"{title} - {v.name}"))  # type: ignore[misc]
        if im is not None:
            paths[v.name] = R.save(im, img / f"{v.name}.png")
            shown.append((v.name, im))
    # with arrangement points used by the placements
    used = {p.point for p in res.placed}
    pts_scene = R.Scene(av, pieces, {n: np.asarray(points[n]["position_m"]) for n in sorted(used) if n in points})
    im = step(
        "points", lambda: R.render_view(pts_scene, R.VIEWS[0], f"{title} - front, arrangement points", labels=False)
    )
    if im is not None:
        paths["points"] = R.save(im, img / "points_front.png")
    body = V.Surf(avatar.vertices, avatar.faces)
    if heat:  # clearance heat map
        clear = float(options.get("clearance_body", 0.03))
        hm = []
        for m in pieces:
            d, _ = body.signed(m.vertices)
            hm.append(R.Mesh(m.name, m.vertices, m.faces, m.colour, R.heat_colours(d, clear)))
        hs = R.Scene(av, hm)
        for vname in ("front", "back"):
            v = next(x for x in R.VIEWS if x.name == vname)
            t = f"{title} - clearance {vname} (red < {clear * 100:.0f} cm, amber < +1 cm, green ok)"
            im = step(f"heat_{vname}", lambda v=v, t=t: R.render_view(hs, v, t))  # type: ignore[misc]
            if im is not None:
                paths[f"heat_{vname}"] = R.save(im, img / f"heat_{vname}.png")
                shown.append((f"heat_{vname}", im))
    lv = {k: levels[k] for k in (expect.get("sections") or SECTION_LEVELS) if k in levels}
    xs = step("sections", lambda: R.cross_sections(scene, lv, f"{title} - horizontal sections (top view, +z front up)"))
    if xs is not None:
        paths["sections"] = R.save(xs, img / "sections.png")
        shown.append(("sections", xs))
    if closeups:
        for m, p in zip(pieces, res.placed, strict=True):

            def one(m: R.Mesh = m, p: Any = p) -> Image.Image:
                q = near.get(p.name) if near is not None else body.closest(p.vertices)[1]
                n = (p.vertices - q).mean(axis=0) if q is not None else np.array([0.0, 0.0, 1.0])
                return R.close_up(scene, m, n, f"{title} - close-up {p.name} on {p.point}")

            im = step(f"closeup {p.name}", one)
            if im is not None:
                safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in p.name)
                paths[f"close:{p.name}"] = R.save(im, img / "closeups" / f"{safe}.png")
    if shown:
        sheet = step("sheet", lambda: R.contact_sheet(shown, f"{title} - all views"))
        if sheet is not None:
            paths["sheet"] = R.save(sheet, out / "sheet.png")
    return paths


def render_result(
    out: Path,
    doc: dict[str, Any],
    inp: Inputs,
    res: AlignmentResult,
    near: dict[str, Any] | None = None,
    all_views: bool = False,
    **kw: Any,
) -> dict[str, Path]:
    """``images`` for a solved assembly on a built ``Inputs`` (shared by ``assembler render`` and the web export)."""
    avatar = read_avatar(inp.avatar)
    points = {p["name"]: p for p in json.loads((inp.avatar / "arrangement_points.json").read_text("utf-8"))["points"]}
    return images(
        out,
        doc["garment"],
        out.name,
        res,
        avatar,
        body_levels(avatar),
        near,
        res.doc["arrangement"]["options"],
        doc.get("expect", {}),
        None if all_views else doc.get("views"),
        points,
        **kw,
    )


def _md(garment: str, it_name: str, runs: dict[str, dict[str, Any]], extra: dict[str, Any]) -> str:
    lines = [f"# verify: {garment} {it_name}", ""]
    overall = all(V.verdict(r["rows"]) for r in runs.values()) and all(x["ok"] for x in extra["global"])
    lines += [f"**overall: {'PASS' if overall else 'FAIL'}**", ""]
    lines += ["| run | engine valid | independent | fails | checks |", "|---|---|---|---|---|"]
    for name, r in runs.items():
        fails = [x for x in r["rows"] if x["status"] == "fail"]
        lines.append(f"| {name} | {r['engine_valid']} | {V.verdict(r['rows'])} | {len(fails)} | {len(r['rows'])} |")
    lines += ["", "| global check | ok | value |", "|---|---|---|"]
    for g in extra["global"]:
        lines.append(f"| {g['id']} | {g['ok']} | {g['value']} |")
    for name, r in runs.items():
        fails = [x for x in r["rows"] if x["status"] == "fail"]
        if fails:
            lines += [
                "",
                f"## failures in {name}",
                "",
                "| id | piece | value | limit | hint |",
                "|---|---|---|---|---|",
            ]
            for x in fails[:40]:
                lines.append(f"| {x['id']} | {x['piece']} | {x['value']} | {x['limit']} | {x['hint']} |")
    prim = runs["primary"]
    lines += ["", "## worst offenders (primary)", "", "| id | piece | value | limit |", "|---|---|---|---|"]
    for cid in ("clearance.body", "clearance.pieces", "clearance.floor"):
        rows = sorted([x for x in prim["rows"] if x["id"] == cid], key=lambda x: x["value"])[:4]
        for x in rows:
            lines.append(f"| {cid} | {x['piece']} | {x['value']} | {x['limit']} |")
    lines += [
        "",
        "## registration and level offsets (primary)",
        "",
        "| piece | line | level | y | body y | offset |",
        "|---|---|---|---|---|---|",
    ]
    for f in prim["fit"]:
        flag = " (check)" if abs(f["offset_m"]) > V.LEVEL_CHECK else ""
        lines.append(
            f"| {f['piece']} | {f['line']} | {f['level']} | {f['y_m']} | {f['body_y_m']} | {f['offset_m']}{flag} |"
        )
    for x in prim["rows"]:
        if x["id"] == "registration":
            lines.append(f"| {x['piece']} | registration | | | | {x['value']} ({x['status']}) {x['hint']} |")
    lines += [
        "",
        "## ease (primary, informational)",
        "",
        "| ring | level | y | garment | body | ease | coverage |",
        "|---|---|---|---|---|---|---|",
    ]
    for e in prim["ease"]:
        lines.append(
            f"| {e['ring']} | {e['level']} | {e['y_m']} | {e['garment_width_m']} | {e['body_girth_m']} | "
            f"{e['ease_m']} | {e['coverage']} |"
        )
    lines += ["", "## engine warnings (primary)", ""]
    lines += [f"- {w}" for w in extra["warnings"][:40]] or ["- none"]
    return "\n".join(lines) + "\n"


def run(
    assembly: Path,
    out: Path,
    synthetic: bool = False,
    fast: bool = False,
    render: bool = True,
    cache: Path | None = None,
) -> dict[str, Any]:
    """Run an assembly. Returns the summary (also written as ``verify.json``)."""
    t0 = time.perf_counter()
    doc = load_assembly(assembly)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "assembly.json").write_text(
        json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    cache = cache or ROOT / "build" / "_cache"
    inp = build_inputs(doc["job"], synthetic, cache)
    derived, refined = out / "derived", cache / "refined"
    mode = doc.get("mode", "arrange")
    expect = doc.get("expect", {})
    spec = spec_of(doc)
    eng = AlignmentEngine()
    t = time.perf_counter()
    prim = _solve(eng, inp, spec, derived, refined, mode, out_dir=out / "engine")
    runs: dict[str, Run] = {"primary": Run("primary", prim, spec, time.perf_counter() - t)}
    glob: list[dict[str, Any]] = []
    if not fast:
        for name, order in (("replay", list(doc["placements"])), ("replay_reversed", list(doc["placements"])[::-1])):
            r = replay(doc, inp, order, derived, refined)
            r.name = name
            runs[name] = r
            (out / f"{name}_result.json").write_text(
                json.dumps({**r.result.doc, "arrangement": {**r.result.doc["arrangement"], "timing_s": 0.0}}, indent=1)
                + "\n",
                encoding="utf-8",
                newline="\n",
            )
        t = time.perf_counter()
        carried = [
            {**pl, "solved": d["solved"]} if d.get("solved") else pl
            for pl, d in zip(doc["placements"], prim.doc["placements"], strict=True)
        ]
        cres = _solve(AlignmentEngine(), inp, spec_of(doc, carried), derived, refined, "align")
        moved = max(
            (float(np.abs(a.vertices - b.vertices).max()) if a.vertices.shape == b.vertices.shape else 9.0)
            for a, b in zip(prim.placed, cres.placed, strict=True)
        )
        glob.append({"id": "carry.moves", "ok": moved <= CARRY_TOL, "value": round(moved, 6), "limit": CARRY_TOL})
        rep = _solve(AlignmentEngine(), inp, spec, derived, refined, mode)
        same = _fingerprint(rep) == _fingerprint(prim)
        glob.append({"id": "determinism", "ok": same, "value": _fingerprint(prim), "limit": _fingerprint(rep)})
    (out / "result.json").write_text(
        json.dumps({**prim.doc, "arrangement": {**prim.doc["arrangement"], "timing_s": 0.0}}, indent=1) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    avatar = read_avatar(inp.avatar)
    points = {p["name"]: p for p in json.loads((inp.avatar / "arrangement_points.json").read_text("utf-8"))["points"]}
    levels = body_levels(avatar)
    body = V.Surf(avatar.vertices, avatar.faces)
    bundles = bundles_of(inp, derived)
    reports: dict[str, dict[str, Any]] = {}
    near_primary: dict[str, Any] = {}
    for name, r in runs.items():
        rv = run_view(name, r.result, r.spec, bundles)
        extra = verify_run(rv, body, points, levels, avatar, inp, expect)
        if name == "primary":
            near_primary = extra["near"]
        reports[name] = {
            "rows": rv.rows,
            "engine_valid": bool(r.result.doc["validity"]["valid"]),
            "fit": extra["fit"],
            "ease": extra["ease"],
            "seconds": round(r.seconds, 1),
        }
    _glb(out / "scene.glb", avatar, prim.placed)
    it_name = out.name
    summary = {
        "garment": doc["garment"],
        "iteration": it_name,
        "pass": all(V.verdict(r["rows"]) for r in reports.values()) and all(g["ok"] for g in glob),
        "levels_m": {k: round(v, 4) for k, v in levels.items()},
        "inputs": {"pieces": str(inp.pieces), "avatar": str(inp.avatar), "cached": inp.cached, "synthetic": synthetic},
        "global": glob,
        "runs": {k: {kk: vv for kk, vv in v.items()} for k, v in reports.items()},
    }
    (out / "verify.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8", newline="\n")
    (out / "verify.md").write_text(
        _md(doc["garment"], it_name, reports, {"global": glob, "warnings": prim.warnings}),
        encoding="utf-8",
        newline="\n",
    )
    if render:
        images(
            out,
            doc["garment"],
            it_name,
            prim,
            avatar,
            levels,
            near_primary,
            prim.doc["arrangement"]["options"],
            expect,
            doc.get("views"),
            points,
        )
    summary["seconds"] = round(time.perf_counter() - t0, 1)
    return summary


def render_assembly(assembly: Path, out: Path, synthetic: bool = False, cache: Path | None = None) -> dict[str, Path]:
    """``assembler render``: solve an assembly (its mode) and write only the images, all views, to ``out``."""
    doc = load_assembly(assembly)
    out = Path(out)
    cache = cache or ROOT / "build" / "_cache"
    inp = build_inputs(doc["job"], synthetic, cache)
    res = _solve(AlignmentEngine(), inp, spec_of(doc), out / "derived", cache / "refined", doc.get("mode", "arrange"))
    return render_result(out, doc, inp, res, all_views=True)

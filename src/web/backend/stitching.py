"""Stitching of the layout on screen: import / check a stitching JSON, run the stitcher, export its result and images.

The request carries the placements with their ``solved`` blocks: the server aligns them incrementally (nothing moves),
writes a run folder the stitcher reads (``assembler.webio.stitch_run_folder``) and runs ``stitcher`` on it. Runs are
cached in the build by a hash of (layout, stitching file).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from alignment_engine import AlignmentEngine
from assembler import webio
from assembler.run import spec_of
from stitcher.errors import StitchingError
from stitcher.load import load_run, read_stitching, validate
from stitcher.result import pieces_of, stitch, write
from stitcher.verify import check, markdown, verdict

from .builds import Build
from .jobs import JobError

MAX_BYTES = 2_000_000
PAIRS = 16  # correspondence lines sent per seam (the viewer draws every one)


def parse(raw: bytes | str) -> dict[str, Any]:
    """A stitching document from the request body (every schema problem listed)."""
    if len(raw) > MAX_BYTES:
        raise JobError("invalid_stitching", "file is larger than 2 MB")
    try:
        doc: dict[str, Any] = json.loads(raw)
        validate("stitching", doc)
    except (json.JSONDecodeError, StitchingError) as e:
        raise JobError("invalid_stitching", str(e)) from None
    ids = [s["id"] for s in doc["seams"]]
    dup = sorted({i for i in ids if ids.count(i) > 1})
    if dup:
        raise JobError("invalid_stitching", f"duplicate seam ids: {dup}")
    return doc


def _folder(b: Build, body: dict[str, Any], stitching: dict[str, Any]) -> Path:
    key = json.dumps({k: body.get(k) for k in ("placements", "derived_pieces", "arrange_options")}, sort_keys=True)
    h = hashlib.sha256((b.id + key + json.dumps(stitching, sort_keys=True)).encode()).hexdigest()[:16]
    return b.dir / "stitching" / h


def run(b: Build, body: dict[str, Any]) -> tuple[Path, dict[str, Any], list[dict[str, Any]], Any]:
    """(folder, stitching result, verification rows, the run) of the layout in ``body`` with ``body['stitching']``."""
    st_doc = parse(json.dumps(body.get("stitching")))
    folder = _folder(b, body, st_doc)
    spec = spec_of(body)
    res = AlignmentEngine().align(
        b.dir / "avatar", b.dir / "pieces", json.dumps(spec), derived_dir=b.dir / "derived", cache_dir=b.dir / "refined"
    )
    placements = [
        {**pl, "solved": d["solved"]} if d.get("solved") else pl
        for pl, d in zip(spec["placements"], res.doc["placements"], strict=True)
    ]
    asm = webio.export_doc(
        job=b.job,
        placements=placements,
        derived_pieces=spec.get("derived_pieces"),
        arrange_options=spec.get("arrange_options"),
        stitches=None,
        rings=spec.get("rings"),
        garment=st_doc["garment"],
        notes="",
        mode="incremental",
        include_solved=True,
    )
    folder.mkdir(parents=True, exist_ok=True)
    asm_path = folder / "assembly.json"
    asm_path.write_text(webio.dumps(asm), encoding="utf-8", newline="\n")
    st_path = folder / f"{st_doc['garment']}.stitching.json"
    st_path.write_text(json.dumps({**st_doc, "assembly": asm_path.name}, indent=2) + "\n", encoding="utf-8")
    run_dir = webio.stitch_run_folder(folder / "run", res, spec, b.dir / "pieces", b.dir / "avatar", b.dir / "derived")
    st = read_stitching(st_path)
    r = load_run(run_dir)
    try:
        result = stitch(st, st_path, asm_path, r)
    except StitchingError as e:
        raise JobError("invalid_stitching", str(e)) from None
    write(result, folder)
    rows = check(result, st, st_path, asm_path, run_dir)
    (folder / "stitch_verify.md").write_text(markdown(result, rows), encoding="utf-8", newline="\n")
    return folder, result, rows, r


def answer(result: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    """What the viewer draws and lists: per seam its lines, a few correspondences and its verdict."""
    seams = []
    for s in result["seams"]:
        mine = [r for r in rows if r.get("seam") == s["id"] and r["status"] in ("fail", "warn")]
        n = s["samples"]["count"]
        idx = sorted({round(i * (n - 1) / (PAIRS - 1)) for i in range(PAIRS)})
        seams.append(
            {
                "id": s["id"],
                "label": s["label"],
                "kind": s["kind"],
                "ease": s["ease"]["mode"],
                "declared_ratio": s["ease"]["declared_ratio"],
                "length_a_cm": s["length_a_cm"],
                "length_b_cm": s["length_b_cm"],
                "ratio": s["ratio"],
                "gap_m": s["world_gap_m"],
                "crossings": s["orientation"]["crossing_count"],
                "status": "fail" if any(r["status"] == "fail" for r in mine) else "pass",
                "problems": [f"{r['status']}: {r['id']} {r['hint'] or r['value']}" for r in mine],
                "a": s["a"]["polyline_m"],
                "b": s["b"]["polyline_m"],
                "pairs": [[s["samples"]["a_xyz_m"][i], s["samples"]["b_xyz_m"][i]] for i in idx],
            }
        )
    return {
        "pass": verdict(rows),
        "summary": result["summary"],
        "seams": seams,
        "errors": result["errors"],
        "unstitched_edges": result["unstitched_edges"],
        "failures": [f"{r['id']} {r['seam']}: {r['hint'] or r['value']}" for r in rows if r["status"] == "fail"],
    }


def export_zip(b: Build, body: dict[str, Any]) -> Path:
    """Zip of the stitching file, ``stitching.result.json``, ``stitch_verify.md`` and every stitch image."""
    from stitcher.render import write_images

    folder, result, rows, r = run(b, body)
    zp = folder / "stitching.zip"
    if not zp.is_file():
        paths = write_images(folder, r, pieces_of(r), result, rows, result["garment"])
        files: dict[str, Path | bytes] = {
            "stitching.result.json": folder / "stitching.result.json",
            "stitch_verify.md": folder / "stitch_verify.md",
            f"{result['garment']}.stitching.json": json.dumps(body["stitching"], indent=2).encode("utf-8"),
        }
        for p in paths.values():
            files[f"images/{p.name}"] = p
        webio.write_zip(zp, files)
    return zp

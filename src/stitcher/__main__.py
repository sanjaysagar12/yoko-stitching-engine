"""``python -m stitcher suggest|run|check|render --stitching <json> [--assembly A] [--run DIR] [--synthetic] [--out DIR]
[--fast] [--no-render]``

* ``run``: stitch (the assembly run is reused from ``--run``, from the cache, or made now in a subprocess), verify,
  write ``stitching.result.json``, ``stitch_verify.json|md`` and the images; exit 1 when the verification fails.
* ``check``: re-verify an existing result independently (``--result`` or ``--out``/stitching.result.json).
* ``render``: the images of an existing result.
* ``suggest``: seam candidates and the edge table into ``<garment>.suggested.json`` (never over a hand-edited file).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .errors import StitchingError
from .load import ROOT, Run, assembly_path, ensure_run, load_run, read_json, read_stitching


def _out(a: argparse.Namespace, garment: str) -> Path:
    return Path(a.out) if a.out else ROOT / "build" / "stitching" / f"{garment}-{'syn' if a.synthetic else 'real'}"


def _run_folder(a: argparse.Namespace, asm: Path) -> Path:
    if a.run:
        return Path(a.run)
    return ensure_run(asm, a.synthetic, log=lambda m: print(m, file=sys.stderr))


def verify_and_write(
    out: Path, result: dict[str, Any], stitching: dict[str, Any], st_path: Path, asm: Path, run_folder: Path
) -> list[dict[str, Any]]:
    from .verify import check, markdown, verdict

    rows = check(result, stitching, st_path, asm, run_folder)
    (out / "stitch_verify.json").write_text(
        json.dumps({"pass": verdict(rows), "rows": rows}, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    (out / "stitch_verify.md").write_text(markdown(result, rows), encoding="utf-8", newline="\n")
    return rows


def images(out: Path, run: Run, result: dict[str, Any], rows: list[dict[str, Any]], fast: bool) -> None:
    from .render import write_images
    from .result import pieces_of

    title = f"{result['garment']} ({'synthetic' if result['inputs']['synthetic'] else 'SMPL-X'})"
    write_images(out, run, pieces_of(run), result, rows, title, fast)


def run_cmd(a: argparse.Namespace) -> int:
    from .result import stitch, write
    from .verify import verdict

    st_path = Path(a.stitching)
    st = read_stitching(st_path)
    asm = assembly_path(st, st_path, a.assembly)
    run = load_run(_run_folder(a, asm))
    out = _out(a, st["garment"])
    result = stitch(st, st_path, asm, run)
    write(result, out)
    rows = verify_and_write(out, result, st, st_path, asm, run.folder)
    if not a.no_render:
        images(out, run, result, rows, a.fast)
    ok = verdict(rows)
    fails = [f"{r['id']} {r['seam']}: {r['hint'] or r['value']}" for r in rows if r["status"] == "fail"]
    print(json.dumps({"pass": ok, "summary": result["summary"], "fails": fails[:30]}, indent=1))
    print(f"wrote {out / 'stitching.result.json'}, {out / 'stitch_verify.md'}")
    return 0 if ok else 1


def check_cmd(a: argparse.Namespace) -> int:
    from .verify import check, verdict

    st_path = Path(a.stitching)
    st = read_stitching(st_path)
    asm = assembly_path(st, st_path, a.assembly)
    res_path = Path(a.result) if a.result else _out(a, st["garment"]) / "stitching.result.json"
    result = read_json(res_path)
    rows = check(result, st, st_path, asm, Path(a.run) if a.run else None)
    fails = [f"{r['id']} {r['seam']}: {r['hint'] or r['value']}" for r in rows if r["status"] == "fail"]
    warns = sum(r["status"] == "warn" for r in rows)
    print(json.dumps({"pass": verdict(rows), "fails": fails, "warnings": warns}, indent=1))
    return 0 if verdict(rows) else 1


def render_cmd(a: argparse.Namespace) -> int:
    from .verify import check

    st_path = Path(a.stitching)
    st = read_stitching(st_path)
    asm = assembly_path(st, st_path, a.assembly)
    out = _out(a, st["garment"])
    result = read_json(Path(a.result) if a.result else out / "stitching.result.json")
    run = load_run(Path(a.run) if a.run else ROOT / result["inputs"]["run"])
    images(out, run, result, check(result, st, st_path, asm, run.folder), a.fast)
    print(f"wrote {out / 'images'}")
    return 0


def suggest_cmd(a: argparse.Namespace) -> int:
    from .suggest import suggest, write_suggestion

    if a.stitching:
        st_path = Path(a.stitching)
        st = read_json(st_path)
        asm = assembly_path(st, st_path, a.assembly)
    elif a.assembly:
        asm = Path(a.assembly)
        st_path = None
    else:
        raise StitchingError("suggest needs --stitching or --assembly")
    run = load_run(_run_folder(a, asm))
    doc = suggest(run, asm)
    path = write_suggestion(doc, Path(a.out) if a.out else None, st_path)
    print(doc["table_text"])
    print(f"wrote {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="stitcher", description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("cmd", choices=["suggest", "run", "check", "render"])
    ap.add_argument("--stitching", type=Path, help="assert/stitchings/<garment>.stitching.json")
    ap.add_argument("--assembly", type=Path, help="the assembly JSON (default: the stitching file's)")
    ap.add_argument("--run", type=Path, help="an `assembler run` output folder to stitch (default: cached / made now)")
    ap.add_argument("--result", type=Path, help="check/render: the stitching.result.json")
    ap.add_argument("--synthetic", action="store_true", help="the synthetic test body instead of SMPL-X")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--fast", action="store_true", help="smaller images")
    ap.add_argument("--no-render", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd != "suggest" and not a.stitching:
        ap.error(f"{a.cmd} needs --stitching")
    try:
        return {"run": run_cmd, "check": check_cmd, "render": render_cmd, "suggest": suggest_cmd}[a.cmd](a)
    except StitchingError as e:
        print(f"stitching error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())

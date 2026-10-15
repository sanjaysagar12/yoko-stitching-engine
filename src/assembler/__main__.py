"""``python -m assembler run <assembly.json> --out build/<garment>/iter_NN [--synthetic] [--fast] [--no-render]``;
``python -m assembler stitch <assembly.json> --stitching <stitching.json> --out DIR [--synthetic] [--fast]``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="assembler", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run an assembly and write result, meshes, verify.md and images")
    r.add_argument("assembly", type=Path)
    r.add_argument("--out", type=Path, required=True)
    g = r.add_mutually_exclusive_group()
    g.add_argument("--synthetic", action="store_true", help="the synthetic test body instead of SMPL-X")
    g.add_argument("--real-avatar", action="store_true", help="SMPL-X from models/smplx (default)")
    r.add_argument("--renderer", choices=["soft", "web"], default="soft")
    r.add_argument("--fast", action="store_true", help="primary run only (no replays, carry or determinism checks)")
    r.add_argument("--no-render", action="store_true")
    d = sub.add_parser("render", help="solve an assembly and write only the images (all views, sheet, sections)")
    d.add_argument("assembly", type=Path)
    d.add_argument("--out", type=Path, required=True)
    d.add_argument("--synthetic", action="store_true", help="the synthetic test body instead of SMPL-X")
    c = sub.add_parser("check", help="validate an assembly file only")
    c.add_argument("assembly", type=Path)
    st = sub.add_parser("stitch", help="run an assembly, then stitch it (stitcher run on that run's output)")
    st.add_argument("assembly", type=Path)
    st.add_argument("--stitching", type=Path, required=True)
    st.add_argument("--out", type=Path, required=True)
    st.add_argument("--synthetic", action="store_true", help="the synthetic test body instead of SMPL-X")
    st.add_argument("--fast", action="store_true", help="primary run only, smaller stitch images")
    st.add_argument("--no-render", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "stitch":
        return stitch(a)
    from .run import AssemblyError, load_assembly, render_assembly, run

    try:
        if a.cmd == "check":
            load_assembly(a.assembly)
            print("ok")
            return 0
        if a.cmd == "render":
            print(json.dumps(sorted(str(p) for p in render_assembly(a.assembly, a.out, a.synthetic).values())))
            return 0
        if a.renderer == "web":
            print("--renderer web needs the web app and Playwright; the soft renderer is used", file=sys.stderr)
        s = run(a.assembly, a.out, synthetic=a.synthetic, fast=a.fast, render=not a.no_render)
    except AssemblyError as e:
        print(f"assembly error: {e}", file=sys.stderr)
        return 2
    fails = {k: sum(x["status"] == "fail" for x in v["rows"]) for k, v in s["runs"].items()}
    print(json.dumps({"pass": s["pass"], "fails": fails, "global": s["global"], "seconds": s["seconds"]}))
    print(f"wrote {a.out}/verify.md, {a.out}/sheet.png")
    return 0 if s["pass"] else 1


def stitch(a: argparse.Namespace) -> int:
    """Assembly (``run``, no images) into ``<out>/assembly``, then ``stitcher run`` on it into ``<out>``."""
    from stitcher.__main__ import main as stitcher_main

    from .run import AssemblyError, run

    try:
        s = run(a.assembly, a.out / "assembly", synthetic=a.synthetic, fast=a.fast, render=False)
    except AssemblyError as e:
        print(f"assembly error: {e}", file=sys.stderr)
        return 2
    print(json.dumps({"assembly_pass": s["pass"], "seconds": s["seconds"]}))
    args = ["run", "--stitching", str(a.stitching), "--assembly", str(a.assembly), "--run", str(a.out / "assembly")]
    args += ["--out", str(a.out)] + (["--fast"] if a.fast else []) + (["--no-render"] if a.no_render else [])
    code = stitcher_main(args)
    return code if s["pass"] else max(code, 1)


if __name__ == "__main__":
    sys.exit(main())

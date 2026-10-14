"""Run the three final assemblies on the real SMPL-X avatars (needs models/smplx; not run in CI).

    python scripts/run_assembly_loop_real.py [--out build/final] [--fast] [garment ...]

Writes build/final/<garment>/ (result, meshes, verify.md, images, sheet.png) and copies verify.md and sheet.png to
docs/assemblies/<garment>/ when --publish is given. Exit code 0 only when every garment passes.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from assembler.run import run  # noqa: E402

GARMENTS = ("male_shirt", "aldrich_skirt", "trousers")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("garments", nargs="*", default=list(GARMENTS))
    ap.add_argument("--out", type=Path, default=ROOT / "build" / "final")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument(
        "--publish", action="store_true", help="copy verify.md, verify.json and sheet.png to docs/assemblies"
    )
    a = ap.parse_args()
    if not (ROOT / "models" / "smplx" / "SMPLX_MALE.npz").is_file():
        print("models/smplx not present: the real avatars cannot be built", file=sys.stderr)
        return 2
    ok = True
    for g in a.garments:
        s = run(ROOT / "assert" / "assemblies" / f"{g}.assembly.json", a.out / g, fast=a.fast)
        fails = {k: sum(r["status"] == "fail" for r in v["rows"]) for k, v in s["runs"].items()}
        print(json.dumps({"garment": g, "pass": s["pass"], "fails": fails, "global": s["global"], "s": s["seconds"]}))
        ok &= bool(s["pass"])
        if a.publish:
            dst = ROOT / "docs" / "assemblies" / g
            dst.mkdir(parents=True, exist_ok=True)
            for f in ("verify.md", "verify.json", "sheet.png"):
                shutil.copy2(a.out / g / f, dst / f)
            for f in ("front.png", "left.png", "top.png", "sections.png", "heat_front.png"):
                shutil.copy2(a.out / g / "images" / f, dst / f)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

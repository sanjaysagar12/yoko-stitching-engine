"""``python -m simulator run|finish|fabrics|solver --stitching <json> [--synthetic] [--result R] [--run DIR] [--out DIR]
[--fabric F] [--quality Q] [--frames N] [--sew-time S] [--no-render]``

* ``run``: stitch first when there is no ``--result`` (``python -m stitcher run`` in a subprocess, reusing its cached
  assembly run), then simulate: write ``simulation.result.json``, ``draped.npz``, ``draped_m.obj`` and the images;
  exit 1 when a check fails.
* ``finish``: redo the checks, outputs and images of a simulation folder from its solver frames.
* ``fabrics``: the fabric presets in measured and in solver units. ``solver``: where the solver is.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from stitcher.load import ROOT, load_run, read_json, read_stitching

from .errors import SimulationError
from .fabrics import FABRICS, QUALITIES
from .simulate import Options, finish, folder_key, simulate
from .solver import solver_info


def _stitched(a: argparse.Namespace) -> tuple[Path, Path]:
    """(stitching result, its run folder): ``--result`` or the stitcher's default output, made now if missing."""
    st = read_stitching(Path(a.stitching))
    default = ROOT / "build" / "stitching" / f"{st['garment']}-{_tag(a)}" / "stitching.result.json"
    res = Path(a.result) if a.result else default
    if not res.is_file():
        cmd = [sys.executable, "-m", "stitcher", "run", "--stitching", str(a.stitching), "--no-render"]
        if a.synthetic:
            cmd.append("--synthetic")
        if a.run:
            cmd += ["--run", str(a.run)]
        print(f"stitching first ({' '.join(cmd[1:])})", file=sys.stderr)
        p = subprocess.run(cmd, cwd=ROOT)
        if p.returncode not in (0, 1) or not res.is_file():
            raise SimulationError(f"the stitcher did not write {res} (exit {p.returncode})")
    doc = read_json(res)
    run = Path(a.run) if a.run else ROOT / doc["inputs"]["run"]
    return res, run


def _tag(a: argparse.Namespace) -> str:
    return "syn" if a.synthetic else "real"


def _options(a: argparse.Namespace) -> Options:
    d: dict[str, Any] = {"fabric": a.fabric, "quality": a.quality}
    if a.frames:
        d["frames"] = a.frames
    if a.sew_time:
        d["sew_time"] = a.sew_time
    return Options.of(d)


def _print(ev: dict[str, Any], t0: float) -> None:
    if ev.get("event") == "progress":
        print(f"  frame {ev['frame']}/{ev['frames']}  max seam gap {ev['gap_max_m'] * 100:.1f} cm  "
              f"({ev['wall_s']:.0f} s)", file=sys.stderr, flush=True)  # fmt: skip
    elif ev.get("event") == "stage":
        print(f"[{time.time() - t0:5.0f} s] {ev['stage']}", file=sys.stderr, flush=True)


def run_cmd(a: argparse.Namespace) -> int:
    res, run_dir = _stitched(a)
    o = _options(a)
    st = read_stitching(Path(a.stitching))
    out = Path(a.out) if a.out else ROOT / "build" / "simulation" / f"{st['garment']}-{_tag(a)}-{folder_key(res, o)}"
    t0 = time.time()
    doc = simulate(res, load_run(run_dir), out, o, lambda e: _print(e, t0), render=not a.no_render)
    print(json.dumps({"pass": doc["pass"], "summary": doc["summary"]}, indent=1))
    print(f"wrote {out / 'simulation.result.json'}")
    return 0 if doc["pass"] else 1


def finish_cmd(a: argparse.Namespace) -> int:
    if not a.out:
        raise SimulationError("finish needs --out (the simulation folder)")
    out = Path(a.out)
    doc = read_json(out / "simulation.result.json") if (out / "simulation.result.json").is_file() else None
    res, run_dir = _stitched(a)
    o = Options.of(doc["options"]) if doc else _options(a)
    params = read_json(out / "solver" / "params.json")
    done = {"finished": True, "frames": params["frames"], "closure": (doc or {}).get("closure", [])}
    done.update((doc or {}).get("run", {}))
    t0 = time.time()
    doc = finish(out, res, load_run(run_dir), o, done, lambda e: _print(e, t0), render=not a.no_render)
    print(json.dumps({"pass": doc["pass"], "summary": doc["summary"]}, indent=1))
    return 0 if doc["pass"] else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="simulator", description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("cmd", choices=["run", "finish", "fabrics", "solver"])
    ap.add_argument("--stitching", type=Path, help="assert/stitchings/<garment>.stitching.json")
    ap.add_argument("--result", type=Path, help="the stitching.result.json (default: the stitcher's output)")
    ap.add_argument("--run", type=Path, help="the assembly run folder (default: the stitching result's)")
    ap.add_argument("--synthetic", action="store_true", help="the synthetic test body instead of SMPL-X")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--fabric", default="cotton_shirting", choices=sorted(FABRICS))
    ap.add_argument("--quality", default="standard", choices=sorted(QUALITIES))
    ap.add_argument("--frames", type=int)
    ap.add_argument("--sew-time", type=float)
    ap.add_argument("--no-render", action="store_true")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "fabrics":
            print(json.dumps({k: f.doc() for k, f in FABRICS.items()}, indent=1))
            return 0
        if a.cmd == "solver":
            info = solver_info()
            print(json.dumps(info, indent=1))
            return 0 if info["available"] else 1
        if not a.stitching:
            ap.error(f"{a.cmd} needs --stitching")
        return run_cmd(a) if a.cmd == "run" else finish_cmd(a)
    except SimulationError as e:
        print(f"simulation error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())

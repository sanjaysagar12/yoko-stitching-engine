"""Inputs of the stitcher, read from files only: the stitching JSON, the assembly JSON and an ``assembler run`` output
folder (``result.json``, ``verify.json``, ``derived/``), the piece bundles and the avatar OBJ it names.

``ensure_run`` gets that run folder: a cached one (keyed by the assembly, the job inputs and the engines' sources), or
a fresh ``python -m assembler run --fast --no-render`` in a subprocess (the stitcher imports no engine).
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .errors import StitchingError

ROOT = Path(__file__).resolve().parents[2]
CONTRACTS = ROOT / "contracts"
RUNS = ROOT / "build" / "_stitch_runs"


@cache
def _schema(name: str) -> dict[str, Any]:
    out: dict[str, Any] = json.loads((CONTRACTS / f"{name}.schema.json").read_text(encoding="utf-8"))
    return out


def validate(name: str, doc: Any) -> None:
    """Validate ``doc`` against ``contracts/<name>.schema.json``; every problem is listed."""
    import jsonschema

    v = jsonschema.Draft202012Validator(_schema(name))
    errs = sorted(v.iter_errors(doc), key=lambda e: list(e.absolute_path))
    if errs:
        raise StitchingError(
            f"{name}: " + "; ".join(f"/{'/'.join(map(str, e.absolute_path))}: {e.message}" for e in errs[:20])
        )


def read_json(path: Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        raise StitchingError(f"{path}: file not found") from None
    except json.JSONDecodeError as e:
        raise StitchingError(f"{path}: invalid JSON: {e}") from None


def read_stitching(path: Path) -> dict[str, Any]:
    doc: dict[str, Any] = read_json(path)
    validate("stitching", doc)
    ids = [s["id"] for s in doc["seams"]]
    dup = sorted({i for i in ids if ids.count(i) > 1})
    if dup:
        raise StitchingError(f"duplicate seam ids: {dup}")
    return doc


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def rel(p: Path) -> str:
    """``p`` relative to the repository (posix), or absolute when outside it."""
    p = Path(p).resolve()
    try:
        return p.relative_to(ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def assembly_path(stitching: dict[str, Any], stitching_path: Path, override: Path | None = None) -> Path:
    """The assembly the stitching file refers to: a path (repository- or file-relative) or ``sha256:<hex>`` of a file
    in ``assert/assemblies``. ``override`` (``--assembly``) wins but must have the same garment."""
    if override is not None:
        return Path(override)
    ref = stitching["assembly"]
    if ref.startswith("sha256:"):
        for f in sorted((ROOT / "assert" / "assemblies").glob("*.json")):
            if sha256(f.read_bytes()) == ref[7:]:
                return f
        raise StitchingError(f"no assembly in assert/assemblies has {ref}")
    for base in (ROOT, Path(stitching_path).parent):
        if (base / ref).is_file():
            return Path(base / ref)
    raise StitchingError(f"assembly {ref!r} not found (relative to the repository or {Path(stitching_path).parent})")


def _src_hash(*pkgs: str) -> bytes:
    """The code a run depends on: the engines and the assembler's run and verification (not its CLI or renderer)."""
    h = hashlib.sha256()
    files = [p for pkg in pkgs for p in (ROOT / "src" / pkg).rglob("*.py")]
    files += [ROOT / "src" / "assembler" / f for f in ("run.py", "verify.py")]
    for p in sorted(files):
        h.update(p.relative_to(ROOT).as_posix().encode())
        h.update(p.read_bytes())
    return h.digest()


def run_key(assembly: Path, synthetic: bool) -> str:
    """Content key of an assembly run: the assembly, its job's inputs, the body source and the engines' code."""
    doc = read_json(assembly)
    h = hashlib.sha256(Path(assembly).read_bytes())
    job_file = ROOT / "assert" / "jobs" / f"{doc.get('job')}.json"
    if job_file.is_file():
        job = read_json(job_file)
        h.update(job_file.read_bytes())
        for sub, key in (("sm2d", "sm2d"), ("smis", "smis"), ("body", "body")):
            f = ROOT / "assert" / sub / job.get(key, "")
            if f.is_file():
                h.update(f.read_bytes())
    h.update(b"synthetic" if synthetic else b"real")
    ver = ROOT / "models" / "smplx" / "version.txt"
    if not synthetic and ver.is_file():
        h.update(ver.read_bytes())
    h.update(_src_hash("sm2d_engine", "avatar_engine", "alignment_engine"))
    if synthetic:
        h.update((ROOT / "tests" / "synthetic_avatar.py").read_bytes())
    return h.hexdigest()[:16]


def ensure_run(assembly: Path, synthetic: bool, runs: Path | None = None, log: Any = None) -> Path:
    """The ``assembler run`` folder of ``assembly``: cached by content, else run now in a subprocess (``--fast
    --no-render``). The assembler's exit status 1 (its verification failed) still leaves a usable result; the
    stitcher's own verifier reads that verdict (``verify.json``)."""
    doc = read_json(assembly)
    out = (
        runs or RUNS
    ) / f"{doc.get('garment', 'garment')}-{'syn' if synthetic else 'real'}-{run_key(assembly, synthetic)}"
    if (out / "result.json").is_file() and (out / "verify.json").is_file():
        return out
    cmd = [sys.executable, "-m", "assembler", "run", str(assembly), "--out", str(out), "--fast", "--no-render"]
    if synthetic:
        cmd.append("--synthetic")
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(ROOT / "src"), os.environ.get("PYTHONPATH", "")])}
    if log:
        log(f"running the assembly ({' '.join(cmd[1:])})")
    p = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    if p.returncode not in (0, 1) or not (out / "result.json").is_file():
        raise StitchingError(f"assembler run failed ({p.returncode}): {p.stderr.strip()[-2000:]}")
    return out


def read_obj(path: Path) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Vertices and triangles of an OBJ (``v`` and ``f`` lines; polygons fanned)."""
    verts: list[list[float]] = []
    faces: list[list[int]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith("v "):
            verts.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("f "):
            idx = [int(t.split("/")[0]) - 1 for t in line.split()[1:]]
            faces += [[idx[0], idx[k], idx[k + 1]] for k in range(1, len(idx) - 1)]
    return np.asarray(verts, dtype=np.float64), np.asarray(faces, dtype=np.int64).reshape(-1, 3)


@dataclass
class Run:
    """An ``assembler run`` folder: the alignment result, the piece bundles (job pieces + derived), the avatar."""

    folder: Path
    result: dict[str, Any]
    bundles: dict[str, dict[str, Any]]
    avatar_dir: Path
    verify: dict[str, Any] | None
    _avatar: tuple[NDArray[np.float64], NDArray[np.int64]] | None = field(default=None, repr=False)

    @property
    def placements(self) -> dict[str, dict[str, Any]]:
        return {p["piece"]: p for p in self.result["placements"]}

    def avatar(self) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
        if self._avatar is None:
            bundle = read_json(self.avatar_dir / "avatar_bundle.json")
            self._avatar = read_obj(self.avatar_dir / bundle["mesh"])
        return self._avatar


def load_run(folder: Path) -> Run:
    """Read an ``assembler run`` output folder (``result.json``, ``verify.json``, ``derived/``)."""
    folder = Path(folder)
    result = read_json(folder / "result.json")
    verify = read_json(folder / "verify.json") if (folder / "verify.json").is_file() else None
    if verify is None:
        raise StitchingError(f"{folder}: no verify.json (needs the output folder of `python -m assembler run`)")
    pieces_dir, avatar_dir = Path(verify["inputs"]["pieces"]), Path(verify["inputs"]["avatar"])
    derived = Path(verify["inputs"].get("derived") or folder / "derived")  # the web app's run names its build's
    bundles: dict[str, dict[str, Any]] = {}
    for d in (pieces_dir, derived):
        for f in sorted(d.glob("*.piece.json")):
            b = read_json(f)
            bundles[b["name"]] = b
    missing = [p["piece"] for p in result["placements"] if p["piece"] not in bundles]
    if missing:
        raise StitchingError(f"{folder}: no piece bundle for {missing}")
    return Run(folder, result, bundles, avatar_dir, verify)


def geometry_hash(run_folder: Path) -> dict[str, str]:
    """sha256 of the placed piece geometry (result.json placements: matrix, wrapped mesh, world outline; the placed
    OBJs) and of the avatar mesh file, read from disk."""
    folder = Path(run_folder)
    res = read_json(folder / "result.json")
    h = hashlib.sha256()
    for p in sorted(res["placements"], key=lambda p: p["piece"]):
        geo = {k: p.get(k) for k in ("piece", "matrix", "mirrored", "wrapped_vertices", "wrapped_faces")}
        geo["world_outline_m"] = p.get("world_outline_m")
        h.update(json.dumps(geo, sort_keys=True).encode())
    for f in sorted((folder / "engine").glob("*_placed_m.obj")):
        h.update(f.name.encode())
        h.update(f.read_bytes())
    verify = read_json(folder / "verify.json")
    adir = Path(verify["inputs"]["avatar"])
    mesh = read_json(adir / "avatar_bundle.json")["mesh"]
    return {"pieces": h.hexdigest(), "avatar": sha256((adir / mesh).read_bytes())}

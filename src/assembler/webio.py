"""Assembly file <-> web layout, shared by the web backend (the frontend only applies what this module returns).

``parse`` + ``check_build`` turn an uploaded assembly (or an older web ``placements.json``) into a normalised payload
and list *every* problem with a specific code; ``export_doc`` builds a valid assembly from a web layout. Nothing here
touches the avatar or the pieces: sizes are never changed, ``allow_distortion`` and ``body_slice`` are refused.
"""

from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from alignment_engine.contracts import contracts_dir

from .run import AssemblyError, check_doc, spec_of

MAX_BYTES = 2_000_000
SAFE = re.compile(r"^[A-Za-z0-9_.-]+$")
CODES = ("invalid_assembly", "unknown_job", "unknown_piece", "unknown_point", "unsupported_option")
# the ranges of the sliders in the web UI (metres); a value outside is applied as it is, with a warning
UI_RANGES = {"clearance_body": (0, 0.08), "gap_piece": (0, 0.10), "gap_layer": (0, 0.05), "max_standoff": (0.03, 0.40)}
TOP_ORDER = (
    "schema_version",
    "garment",
    "job",
    "notes",
    "mode",
    "derived_pieces",
    "placements",
    "arrange_options",
    "stitches",
    "rings",
    "expect",
    "views",
)
PLACEMENT_ORDER = (
    "piece",
    "point",
    "reference",
    "wrap",
    "anchor",
    "orient",
    "rotation_deg",
    "tilt",
    "offset_m",
    "flip_facing",
    "mirror",
    "ring",
    "layer",
    "locked",
    "shift_m",
    "over",
    "solved",
)


class AssemblyProblems(AssemblyError):
    """Every problem found, each ``(code, message)``; ``str()`` lists them all."""

    def __init__(self, problems: list[tuple[str, str]]) -> None:
        self.problems = problems
        super().__init__("; ".join(f"[{c}] {m}" for c, m in problems))

    @property
    def code(self) -> str:
        return self.problems[0][0]


def _schema_problems(name: str, doc: Any, skip: tuple[str, ...] = ()) -> list[str]:
    schema = json.loads((contracts_dir() / f"{name}.schema.json").read_text(encoding="utf-8"))
    errs = sorted(Draft202012Validator(schema).iter_errors(doc), key=lambda e: list(e.absolute_path))
    return [
        f"{name}: {'/'.join(str(p) for p in e.absolute_path) or '<root>'}: {e.message}"
        for e in errs
        if not (e.absolute_path and str(e.absolute_path[0]) in skip)
    ][:20]


def parse(raw: bytes | str | dict[str, Any], job: str | None = None) -> tuple[dict[str, Any], list[str]]:
    """Read an uploaded assembly. Returns the document and warnings; raises ``AssemblyProblems`` (all problems).

    Accepts the older web shape ``{placements, derived_pieces}`` (``job`` names its job) and upgrades it.
    """
    warnings: list[str] = []
    if isinstance(raw, dict):
        doc: Any = raw
    else:
        if len(raw) > MAX_BYTES:
            raise AssemblyProblems([("invalid_assembly", f"file is larger than {MAX_BYTES // 1_000_000} MB")])
        text = raw.decode("utf-8-sig") if isinstance(raw, bytes) else raw.lstrip("﻿")
        try:
            doc = json.loads(text)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise AssemblyProblems([("invalid_assembly", f"not valid JSON: {e}")]) from None
    if not isinstance(doc, dict):
        raise AssemblyProblems([("invalid_assembly", "the file must be a JSON object")])
    if "placements" in doc and "garment" not in doc and "job" not in doc:  # a web placements.json
        if not job:
            raise AssemblyProblems([("unknown_job", "a placements.json has no job: name it with ?job=")])
        keep = ("derived_pieces", "placements", "arrange_options", "stitches", "rings")
        doc = {"schema_version": 1, "garment": job, "job": job, **{k: doc[k] for k in keep if k in doc}}
        warnings.append("upgraded from an older placements.json (no expect, views or notes)")
    problems: list[tuple[str, str]] = []
    jb = doc.get("job")
    if not isinstance(jb, str) or not SAFE.match(jb):
        problems.append(("unknown_job", f"job {jb!r} is not a job name (letters, digits, _ . - only; no paths)"))
    problems += [("invalid_assembly", m) for m in _schema_problems("assembly", doc, skip=("job",) if problems else ())]
    if isinstance(doc.get("placements"), list):
        problems += [("invalid_assembly", m) for m in _schema_problems("placement_spec", spec_of(doc))]
    if (doc.get("arrange_options") or {}).get("allow_distortion"):
        problems.append(
            ("unsupported_option", "arrange_options.allow_distortion must stay false: pieces keep their real size")
        )
    for pl in doc.get("placements") or []:
        if isinstance(pl, dict) and pl.get("wrap") == "body_slice":
            problems.append(("unsupported_option", f"{pl.get('piece')}: wrap body_slice changes piece sizes"))
    names = [pl.get("piece") for pl in doc.get("placements") or [] if isinstance(pl, dict)]
    for n in sorted({n for n in names if names.count(n) > 1}, key=str):
        problems.append(("invalid_assembly", f"piece {n!r} is placed more than once"))
    if problems:
        raise AssemblyProblems(problems)
    return doc, warnings


def check_build(
    doc: dict[str, Any], pieces: set[str], points: set[str], skip_missing: bool = False
) -> tuple[dict[str, Any], list[str]]:
    """Check an assembly against the built job; returns the (possibly reduced) document and warnings.

    With ``skip_missing`` placements of pieces the job does not have are dropped (listed as warnings).
    """
    problems: list[tuple[str, str]] = []
    warnings: list[str] = []
    derived = list(doc.get("derived_pieces") or [])
    ids = [d["id"] for d in derived]
    for d in derived:
        if d["id"] in pieces or ids.count(d["id"]) > 1:
            problems.append(("invalid_assembly", f"derived piece id {d['id']!r} is already taken"))
    known = set(pieces)
    dropped: set[str] = set()
    # a derived piece needs its source; copies and mirrors of derived pieces resolve in any order
    changed = True
    live = {d["id"]: d for d in derived}
    while changed:
        changed = False
        for i, d in list(live.items()):
            if d["source"] not in known | set(live) or d["source"] in dropped:
                if skip_missing:
                    dropped.add(i)
                    del live[i]
                    changed = True
                    warnings.append(f"skipped derived piece {i}: source {d['source']!r} is not in the job")
                else:
                    problems.append(("unknown_piece", f"derived piece {i}: unknown source piece {d['source']!r}"))
                    del live[i]
                    changed = True
    known |= set(live)
    keep = []
    for pl in doc["placements"]:
        if pl["piece"] not in known:
            if skip_missing:
                warnings.append(f"skipped placement of {pl['piece']!r}: no such piece in the job")
                continue
            problems.append(("unknown_piece", f"placement of {pl['piece']!r}: no such piece in the job"))
        if pl["point"] not in points:
            problems.append(("unknown_point", f"{pl['piece']}: unknown arrangement point {pl['point']!r}"))
        keep.append(pl)
    gone = {pl["piece"] for pl in doc["placements"]} - {pl["piece"] for pl in keep}
    stitches = []
    for st in doc.get("stitches") or []:
        ends = {st.get("a", {}).get("piece"), st.get("b", {}).get("piece")} if isinstance(st, dict) else set()
        if ends & (gone | dropped):
            warnings.append(f"skipped a stitch on {sorted(ends & (gone | dropped))}")
        elif ends - known - {None}:
            problems.append(("unknown_piece", f"stitch names unknown piece(s) {sorted(ends - known)}"))
        else:
            stitches.append(st)
    if problems:
        raise AssemblyProblems(problems)
    if not keep:
        raise AssemblyProblems([("unknown_piece", "no placed piece is in the job")])
    out = {**doc, "placements": keep, "derived_pieces": list(live.values())}
    if "stitches" in doc:
        out["stitches"] = stitches
    return out, warnings


def normalise(doc: dict[str, Any], warnings: list[str]) -> dict[str, Any]:
    """The payload the frontend applies. ``run`` says how to assemble: ``arrange`` (joint solve), ``incremental`` (one
    placement at a time) or ``align`` (every placement carries ``solved``: nothing moves)."""
    mode = doc.get("mode", "arrange")
    opts = dict(doc.get("arrange_options") or {})
    warns = list(warnings)
    for k, (lo, hi) in UI_RANGES.items():
        if isinstance(opts.get(k), int | float) and not lo <= opts[k] <= hi:
            warns.append(f"arrange_options.{k} = {opts[k]} is outside the slider range {lo}..{hi} m; applied as it is")
    solved = all(pl.get("solved") for pl in doc["placements"] if not pl.get("locked"))
    return {
        "garment": doc["garment"],
        "job": doc["job"],
        "notes": doc.get("notes", ""),
        "mode": mode,
        "run": "align" if solved and mode == "arrange" else mode,
        "derived_pieces": doc.get("derived_pieces") or [],
        "placements": doc["placements"],
        "arrange_options": opts,
        "stitches": doc.get("stitches") or [],
        "rings": doc.get("rings") or [],
        "expect": doc.get("expect") or {},
        "views": doc.get("views") or [],
        "warnings": warns,
    }


def _ordered(d: dict[str, Any], order: tuple[str, ...]) -> dict[str, Any]:
    return {**{k: d[k] for k in order if k in d}, **{k: d[k] for k in sorted(d) if k not in order}}


def export_doc(
    *,
    job: str,
    placements: list[dict[str, Any]],
    derived_pieces: list[dict[str, Any]] | None = None,
    arrange_options: dict[str, Any] | None = None,
    stitches: list[dict[str, Any]] | None = None,
    rings: list[dict[str, Any]] | None = None,
    garment: str | None = None,
    notes: str = "",
    mode: str = "arrange",
    include_solved: bool = True,
    expect: dict[str, Any] | None = None,
    views: list[str] | None = None,
) -> dict[str, Any]:
    """A valid assembly (schema_version 1) from a web layout; raises ``AssemblyError`` when it would not be valid."""
    if mode not in ("arrange", "incremental"):
        raise AssemblyError(f"mode must be arrange or incremental, not {mode!r}")
    pls = []
    for pl in placements:
        d = {k: v for k, v in pl.items() if v is not None or k == "offset_m"}
        if not include_solved:
            d.pop("solved", None)
        pls.append(_ordered(d, PLACEMENT_ORDER))
    doc: dict[str, Any] = {"schema_version": 1, "garment": garment or job, "job": job}
    if notes:
        doc["notes"] = notes
    doc["mode"] = mode
    if derived_pieces:
        doc["derived_pieces"] = [_ordered(d, ("id", "source", "op", "axis")) for d in derived_pieces]
    doc["placements"] = pls
    if arrange_options:
        doc["arrange_options"] = dict(arrange_options)
    if stitches:
        doc["stitches"] = stitches
    if rings:
        doc["rings"] = rings
    doc["expect"] = expect if expect else {"required_pieces": [p["piece"] for p in pls]}
    if views:
        doc["views"] = views
    doc = _ordered(doc, TOP_ORDER)
    check_doc(doc)
    return doc


def dumps(doc: dict[str, Any]) -> str:
    """The file text: 2-space indent, LF, trailing newline."""
    return json.dumps(doc, indent=2, ensure_ascii=False) + "\n"


def write_zip(path: Path, files: dict[str, Path | bytes]) -> Path:
    """A deterministic zip (sorted names, fixed times): the same files give the same bytes."""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for name in sorted(files):
            src = files[name]
            info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            z.writestr(info, src if isinstance(src, bytes) else src.read_bytes())
    return path


def stitch_run_folder(
    out: Path, res: Any, spec: dict[str, Any], pieces: Path, avatar: Path, derived: Path, synthetic: bool = False
) -> Path:
    """A folder the stitcher reads like an ``assembler run`` output, for a layout solved elsewhere (the web app):
    ``result.json`` and ``verify.json`` with the inputs and the size audit (every ``size.*`` row of the primary run)."""
    from . import verify as V
    from .run import Inputs, bundles_of, run_view

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    doc = {**res.doc, "arrangement": {**res.doc["arrangement"], "timing_s": 0.0}}
    (out / "result.json").write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8", newline="\n")
    rv = run_view("primary", res, spec, bundles_of(Inputs(Path(pieces), Path(avatar), None, "", {}), Path(derived)))
    V.check_size(rv)
    verify = {
        "pass": V.verdict(rv.rows),
        "inputs": {"pieces": str(pieces), "avatar": str(avatar), "derived": str(derived), "synthetic": synthetic},
        "runs": {"primary": {"rows": rv.rows}},
    }
    (out / "verify.json").write_text(json.dumps(verify, indent=1) + "\n", encoding="utf-8", newline="\n")
    return out

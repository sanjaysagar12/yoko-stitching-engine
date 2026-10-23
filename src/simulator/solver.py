"""ZOZO's Contact Solver (``ppf-contact-solver``, Apache 2.0): where it is, and running ``ppf_runner.py`` with it.

The solver ships its own Python with its ``frontend`` package (Windows: ``<dir>/python/python.exe``); the runner is
started with that Python in a subprocess, with the environment of the solver's ``headless.bat``. Found at
``PPF_DIR`` (environment), else the newest ``ppf-contact-solver-*`` folder in ``tools/`` of this repository or next
to it. Tested release: 2026-10-13-12-32 (Windows x64, CUDA).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from stitcher.load import ROOT

from .errors import SolverFailed, SolverNotFound

RUNNER = Path(__file__).with_name("ppf_runner.py")
TESTED_RELEASE = "2026-10-13-12-32"


def _python(d: Path) -> Path:
    return d / "python" / ("python.exe" if sys.platform == "win32" else "bin/python3")


def find_solver() -> Path:
    """The solver folder (``PPF_DIR`` or the newest ``ppf-contact-solver-*`` in ``tools/`` or next to the repo)."""
    env = os.environ.get("PPF_DIR")
    if env:
        d = Path(env)
        if _python(d).is_file() and (d / "frontend").is_dir():
            return d
        raise SolverNotFound(f"PPF_DIR={env} is not a ZOZO's Contact Solver folder (no python/ and frontend/ in it)")
    cands = sorted(
        (d for base in (ROOT / "tools", ROOT.parent) if base.is_dir() for d in base.glob("ppf-contact-solver*")),
        key=lambda d: d.name,
        reverse=True,
    )
    for d in cands:
        if _python(d).is_file() and (d / "frontend").is_dir():
            return d
    raise SolverNotFound(
        "ZOZO's Contact Solver was not found: download ppf-contact-solver-<date>-win64.zip from "
        "https://github.com/st-tech/ppf-contact-solver/releases (tested: "
        f"{TESTED_RELEASE}), unzip it next to this repository or into tools/, or set PPF_DIR to its folder"
    )


def solver_info() -> dict[str, Any]:
    """``{available, path, release, message}`` (never raises)."""
    try:
        d = find_solver()
    except SolverNotFound as e:
        return {"available": False, "path": None, "release": None, "message": str(e)}
    rel = d.name.removeprefix("ppf-contact-solver-").removesuffix("-win64")
    return {"available": True, "path": str(d), "release": rel, "message": ""}


def _env(d: Path) -> dict[str, str]:
    env = dict(os.environ)
    sep = os.pathsep
    extra = [d / "python", d / "python" / "Scripts", d / "bin", d / "mingit" / "cmd"]
    env["PATH"] = sep.join([*(str(p) for p in extra), env.get("PATH", "")])
    env["PYTHONPATH"] = str(d)
    cert = d / "python" / "Lib" / "site-packages" / "certifi" / "cacert.pem"
    if cert.is_file():
        env["SSL_CERT_FILE"] = env["REQUESTS_CA_BUNDLE"] = str(cert)
    env["PYTHONUNBUFFERED"] = "1"
    return env


class SolverRun:
    """One run of ``ppf_runner.py`` in a subprocess; ``events`` are its JSON lines, ``cancel()`` stops it."""

    def __init__(self, scene: Path, params: dict[str, Any], out: Path, on_event: Callable[[dict[str, Any]], None]):
        self.dir = find_solver()
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "cancel").unlink(missing_ok=True)
        params_path = self.out / "params.json"
        params_path.write_text(json.dumps(params, indent=1) + "\n", encoding="utf-8")
        self.on_event = on_event
        self.done: dict[str, Any] | None = None
        self._err = open(self.out / "runner.stderr.txt", "w", encoding="utf-8")  # noqa: SIM115 - closed in wait()
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        self.proc = subprocess.Popen(
            [str(_python(self.dir)), str(RUNNER), str(scene), str(params_path), str(self.out)],
            cwd=self.out, env=_env(self.dir), stdout=subprocess.PIPE, stderr=self._err, text=True,
            encoding="utf-8", errors="replace", creationflags=flags,
        )  # fmt: skip
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()

    def _read(self) -> None:
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            line = line.strip()
            if not line.startswith("{"):
                continue  # the solver's own progress output
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev.get("event") == "done":
                self.done = ev
            self.on_event(ev)

    def cancel(self) -> None:
        """Ask the runner to stop the solver (it terminates the solver process); killed if it does not stop."""
        (self.out / "cancel").write_text("cancel\n", encoding="utf-8")
        try:
            self.proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            self.proc.kill()

    def wait(self) -> dict[str, Any]:
        """Wait for the end; the ``done`` event. Raises ``SolverFailed`` when the runner died without one."""
        code = self.proc.wait()
        self._reader.join(timeout=30)
        self._err.close()
        if self.done is None:
            tail = (self.out / "runner.stderr.txt").read_text(encoding="utf-8", errors="replace")[-3000:]
            raise SolverFailed(f"the solver runner stopped (exit {code}) without a result:\n{tail}")
        return self.done

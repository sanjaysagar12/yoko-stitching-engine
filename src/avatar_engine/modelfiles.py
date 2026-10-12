"""Locate and load SMPL / SMPL-X model files. Nothing is ever downloaded.

* SMPL-X: ``SMPLX_<GENDER>.npz`` (the ``.pkl`` twins and the zip are ignored).
* SMPL: ``SMPL_<GENDER>.npz|pkl`` or the original ``basicmodel_{f,m}_lbs_*.pkl`` / ``basicModel_neutral_*.pkl``.
  A ``.pkl`` is converted once to ``.npz`` in a cache folder with a restricted unpickler that stubs ``chumpy``,
  so ``chumpy`` is not a dependency.
* ``.npz`` files are opened with ``allow_pickle=False`` and only the arrays that are needed are read.
"""

from __future__ import annotations

import fnmatch
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .bodymodel import SMPL_JOINTS, SMPLX_JOINTS, SmplLikeModel
from .errors import ModelFilesNotFoundError, ModelFormatError

REQUIRED_KEYS = ("v_template", "shapedirs", "J_regressor", "weights", "f")
_SMPL_PKL_NAMES = {"male": "basicmodel_m_lbs", "female": "basicmodel_f_lbs", "neutral": "basicmodel_neutral_lbs"}


@dataclass(frozen=True)
class ModelFiles:
    """A located model file. ``warnings`` carries e.g. a gender fallback."""

    path: Path
    model_type: str
    gender: str
    warnings: tuple[str, ...] = ()


def _find(model_dir: Path, model_type: str, gender: str) -> tuple[Path | None, list[str]]:
    """First matching file under ``model_dir`` (recursive, case-insensitive) and the patterns searched."""
    g = gender.upper()
    if model_type == "smplx":
        patterns = [f"SMPLX_{g}.npz"]
    else:
        patterns = [f"SMPL_{g}.npz", f"SMPL_{g}.pkl", f"{_SMPL_PKL_NAMES[gender]}*.pkl"]
    files = sorted(p for p in model_dir.rglob("*") if p.is_file()) if model_dir.is_dir() else []
    for pat in patterns:
        for p in files:
            if fnmatch.fnmatch(p.name.lower(), pat.lower()):
                return p, patterns
    return None, patterns


def locate(model_dir: Path, gender: str, model_type: str = "auto") -> ModelFiles:
    """Find the model file for ``gender`` (falling back to neutral) and ``model_type`` (smpl | smplx | auto).

    Raises ModelFilesNotFoundError listing everything searched.
    """
    model_dir = Path(model_dir)
    types = ["smplx", "smpl"] if model_type == "auto" else [model_type]
    searched: list[str] = []
    for t in types:
        for g in dict.fromkeys([gender, "neutral"]):
            p, pats = _find(model_dir, t, g)
            searched += [f"{model_dir}/**/{x}" for x in pats]
            if p is not None:
                warn = () if g == gender else (f"no {t} file for gender {gender!r}; using {g!r}",)
                return ModelFiles(p, t, g, warn)
    raise ModelFilesNotFoundError(model_dir, gender, model_type, searched)


# -- SMPL pickle conversion (no chumpy) ----------------------------------------------------------------------------
class _ChStub:
    """Stand-in for chumpy objects: keeps the pickled state, whose ``x`` entry is the ndarray."""

    def __init__(self, *_: Any, **__: Any) -> None:
        self.state: Any = None

    def __setstate__(self, state: Any) -> None:
        self.state = state
        if isinstance(state, dict):
            self.__dict__.update(state)


_ALLOWED_PREFIXES = ("numpy", "scipy.sparse", "collections", "copyreg", "builtins", "_codecs")


class _SafeUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if module.split(".")[0] == "chumpy":
            return _ChStub
        if module.startswith(_ALLOWED_PREFIXES) and not (module == "builtins" and name in ("eval", "exec", "open")):
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"refusing to unpickle {module}.{name}")


def _to_array(obj: Any) -> NDArray[np.float64]:
    if hasattr(obj, "toarray"):
        return np.asarray(obj.toarray(), dtype=np.float64)
    if isinstance(obj, _ChStub):
        return np.asarray(obj.__dict__["x"], dtype=np.float64)
    return np.asarray(obj, dtype=np.float64)


def convert_smpl_pkl(pkl: Path, npz: Path) -> Path:
    """Convert an original SMPL ``.pkl`` to a plain ``.npz`` (restricted unpickler, no chumpy needed)."""
    with open(pkl, "rb") as fh:
        data = _SafeUnpickler(fh, encoding="latin1").load()
    try:
        arrays = {
            "v_template": _to_array(data["v_template"]),
            "shapedirs": _to_array(data["shapedirs"]),
            "J_regressor": _to_array(data["J_regressor"]),
            "weights": _to_array(data["weights"]),
            "f": np.asarray(data["f"], dtype=np.int64),
        }
    except KeyError as e:
        raise ModelFormatError(f"{pkl}: missing key {e}") from None
    npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(npz, **arrays)  # type: ignore[arg-type]
    return npz


def load_model(files: ModelFiles, cache_dir: Path | None = None, symmetrize: bool = True) -> SmplLikeModel:
    """Load ``files`` into a ``SmplLikeModel``. SMPL ``.pkl`` files go through the ``.npz`` cache."""
    path = files.path
    if path.suffix.lower() == ".pkl":
        if files.model_type != "smpl":
            raise ModelFormatError(f"{path}: .pkl is only supported for SMPL; use the SMPL-X .npz")
        cache = Path(cache_dir) if cache_dir else path.parent / ".avatar_cache"
        npz = cache / (path.stem + ".npz")
        if not npz.is_file():
            convert_smpl_pkl(path, npz)
        path = npz
    try:
        z = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as e:
        raise ModelFormatError(f"{path}: cannot read as npz without pickle: {e}") from None
    missing = [k for k in REQUIRED_KEYS if k not in z.files]
    if missing:
        raise ModelFormatError(f"{path}: missing arrays {missing}")
    names = SMPLX_JOINTS if files.model_type == "smplx" else SMPL_JOINTS
    return SmplLikeModel(
        files.model_type,
        files.gender,
        files.path.name,
        z["v_template"],
        z["shapedirs"],
        z["J_regressor"],
        z["weights"],
        z["f"],
        list(names),
        symmetrize=symmetrize,
    )

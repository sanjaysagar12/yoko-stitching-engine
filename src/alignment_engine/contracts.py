"""Schema validation for the files exchanged between components (JSON Schema in ``contracts/``).

This file is deliberately identical in sm2d_engine, avatar_engine and alignment_engine (the engines may not
import each other); ``tests/test_contracts.py`` fails if the copies drift apart.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

ENV_VAR = "SM2D_CONTRACTS_DIR"


class ContractError(ValueError):
    """A document does not satisfy its schema, or the schema cannot be found."""


def contracts_dir() -> Path:
    """Folder holding ``*.schema.json``: ``$SM2D_CONTRACTS_DIR`` or the nearest ``contracts/`` above this file."""
    env = os.environ.get(ENV_VAR)
    if env:
        return Path(env)
    for parent in Path(__file__).resolve().parents:
        cand = parent / "contracts"
        if cand.is_dir():
            return cand
    raise ContractError(f"contracts/ folder not found above {Path(__file__).resolve()} (set {ENV_VAR})")


def validate(schema_name: str, doc: Any) -> None:
    """Validate ``doc`` against ``contracts/<schema_name>.schema.json``; raise ContractError listing all problems."""
    path = contracts_dir() / f"{schema_name}.schema.json"
    if not path.is_file():
        raise ContractError(f"schema {schema_name!r} not found at {path}")
    schema = json.loads(path.read_text(encoding="utf-8"))
    errors = sorted(Draft202012Validator(schema).iter_errors(doc), key=lambda e: list(e.absolute_path))
    if errors:
        lines = [f"{'/'.join(str(p) for p in e.absolute_path) or '<root>'}: {e.message}" for e in errors[:10]]
        raise ContractError(f"{schema_name}: " + "; ".join(lines))


def loads(schema_name: str, text: str) -> Any:
    """Parse JSON text and validate it."""
    try:
        doc = json.loads(text)
    except json.JSONDecodeError as e:
        raise ContractError(f"{schema_name}: invalid JSON: {e}") from None
    validate(schema_name, doc)
    return doc


def read_json(path: Path, schema_name: str) -> Any:
    """Read a UTF-8 JSON file and validate it."""
    return loads(schema_name, Path(path).read_text(encoding="utf-8-sig"))


def write_json(path: Path, doc: Any, schema_name: str) -> Path:
    """Validate ``doc`` then write it (2-space indent, LF, trailing newline). Returns ``path``."""
    validate(schema_name, doc)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    return path

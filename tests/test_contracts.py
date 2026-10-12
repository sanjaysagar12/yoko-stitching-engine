"""Schemas in contracts/ and the per-engine loaders (which must stay identical)."""

from __future__ import annotations

import copy
import importlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "tests" / "contract_samples"
LOADERS = ["sm2d_engine.contracts", "avatar_engine.contracts", "alignment_engine.contracts"]
SCHEMAS = sorted(p.name.removesuffix(".schema.json") for p in (ROOT / "contracts").glob("*.schema.json"))


def test_every_schema_has_a_sample() -> None:
    assert len(SCHEMAS) == 7
    assert SCHEMAS == sorted(p.stem for p in SAMPLES.glob("*.json"))


@pytest.mark.parametrize("module", LOADERS)
@pytest.mark.parametrize("schema", SCHEMAS)
def test_all_loaders_accept_the_same_samples(module: str, schema: str) -> None:
    m = importlib.import_module(module)
    doc = json.loads((SAMPLES / f"{schema}.json").read_text())
    m.validate(schema, doc)
    assert m.read_json(SAMPLES / f"{schema}.json", schema) == doc


@pytest.mark.parametrize("module", LOADERS)
def test_all_loaders_reject_the_same_bad_documents(module: str) -> None:
    m = importlib.import_module(module)
    bad = json.loads((SAMPLES / "body_measurements.json").read_text())
    bad["measurements"]["bogus"] = 1
    with pytest.raises(m.ContractError, match="bogus"):
        m.validate("body_measurements", bad)
    bad = json.loads((SAMPLES / "arrangement_points.json").read_text())
    bad["units"] = "cm"
    with pytest.raises(m.ContractError):
        m.validate("arrangement_points", bad)
    with pytest.raises(m.ContractError, match="not found"):
        m.validate("nope", {})


@pytest.mark.parametrize("module", LOADERS)
def test_write_validates_before_writing(module: str, tmp_path: Path) -> None:
    m = importlib.import_module(module)
    good = json.loads((SAMPLES / "placement_spec.json").read_text())
    p = m.write_json(tmp_path / "a" / "spec.json", good, "placement_spec")
    assert json.loads(p.read_text()) == good
    bad = copy.deepcopy(good)
    bad["placements"][0]["wrap"] = "sideways"
    with pytest.raises(m.ContractError):
        m.write_json(tmp_path / "b.json", bad, "placement_spec")
    assert not (tmp_path / "b.json").exists()


def test_loader_sources_are_identical() -> None:
    texts = {m: Path(importlib.import_module(m).__file__ or "").read_bytes() for m in LOADERS}
    assert len(set(texts.values())) == 1, "contracts.py copies have drifted apart"


def test_env_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    m = importlib.import_module("sm2d_engine.contracts")
    monkeypatch.setenv(m.ENV_VAR, str(tmp_path))
    with pytest.raises(m.ContractError, match="not found"):
        m.validate("placement_spec", {})

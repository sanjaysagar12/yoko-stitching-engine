"""Parse an individual measurement file (.smis) into ``Measurements`` (cm)."""

from __future__ import annotations

import xml.etree.ElementTree as ET

from .. import units
from ..errors import PatternError
from ..model.measurements import Measurements


def parse_smis(xml: str) -> Measurements:
    """Read .smis text; values are converted from the file's ``<unit>`` to cm."""
    try:
        root = ET.fromstring(xml)
    except ET.ParseError as e:
        raise PatternError(f"smis: invalid XML: {e}") from None
    unit = (root.findtext("unit") or "").strip()
    if not unit:
        raise PatternError("smis: missing <unit>")
    body = root.find("body-measurements")
    out: dict[str, float] = {}
    for m in body if body is not None else []:
        name, raw = m.get("name"), (m.get("value") or "").strip()
        if not name:
            continue
        try:
            out[name] = units.to_cm(float(raw), unit)
        except ValueError:
            raise PatternError(f"smis: measurement {name!r} has non-numeric value {raw!r}") from None
    return Measurements(out, source_unit=unit)

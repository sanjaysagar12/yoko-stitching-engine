"""Parse a Seamly2D pattern (.sm2d) into a ``PatternModel``. Pure parsing, no maths."""

from __future__ import annotations

import xml.etree.ElementTree as ET

from ..errors import PatternError
from ..model.pattern import IncrementDef, NodeRef, PatternModel, PieceDef, ToolDef

_KIND = {"point": "point", "spline": "curve", "operation": "operation", "arc": "other", "elArc": "other"}


def _truthy(v: str | None) -> bool:
    return (v or "").lower() in ("1", "true")


def parse_sm2d(xml: str) -> PatternModel:
    """Read .sm2d text. Draft blocks are flattened into one id table."""
    try:
        root = ET.fromstring(xml)
    except ET.ParseError as e:
        raise PatternError(f"sm2d: invalid XML: {e}") from None
    unit = (root.findtext("unit") or "").strip()
    if not unit:
        raise PatternError("sm2d: missing <unit>")
    model = PatternModel(unit=unit)
    for inc in root.iter("increment"):
        name = inc.get("name")
        if name:
            model.increments[name] = IncrementDef(name, inc.get("formula") or "")
    for el in root.iter():
        id_ = el.get("id")
        kind = _KIND.get(el.tag)
        if not id_ or kind is None:
            continue
        type_ = el.get("type") or ""
        if type_.startswith("modeling"):
            model.modeling[id_] = el.get("idObject") or ""
            continue
        sources: tuple[str, ...] = ()
        destinations: tuple[str, ...] = ()
        if kind == "operation":
            sources = tuple(i.get("idObject") or "" for i in el.findall("source/item"))
            destinations = tuple(i.get("idObject") or "" for i in el.findall("destination/item"))
            for d, s in zip(destinations, sources, strict=False):
                model.op_dest[d] = (id_, s)
        model.tools[id_] = ToolDef(
            id=id_,
            kind=kind,
            type=type_,
            attrs=dict(el.attrib),
            path_points=tuple(dict(p.attrib) for p in el.findall("pathPoint")),
            sources=sources,
            destinations=destinations,
        )
    for pe in root.iter("piece"):
        nodes_el = pe.find("nodes")
        nodes = tuple(
            NodeRef(
                id_object=n.get("idObject") or "",
                type=n.get("type") or "",
                reverse=_truthy(n.get("reverse")),
                excluded=_truthy(n.get("excluded")),
            )
            for n in (nodes_el if nodes_el is not None else [])
        )
        data = pe.find("data")
        model.pieces.append(
            PieceDef(
                name=pe.get("name") or "",
                nodes=nodes,
                attrs=dict(pe.attrib),
                data=dict(data.attrib) if data is not None else {},
            )
        )
    return model

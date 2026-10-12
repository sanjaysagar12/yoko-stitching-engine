"""Pure-data pattern model produced by ``io.sm2d_reader``. Values are unparsed strings in the pattern unit."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class IncrementDef:
    """An increment: ``name`` (with leading ``#``) and its formula (pattern unit)."""

    name: str
    formula: str


@dataclass(frozen=True)
class ToolDef:
    """A construction object (point / curve / operation) with its raw XML attributes."""

    id: str
    kind: str  # "point" | "curve" | "operation" | "other"
    type: str
    attrs: dict[str, str]
    path_points: tuple[dict[str, str], ...] = ()  # <pathPoint> children of path splines
    sources: tuple[str, ...] = ()  # operation <source> ids
    destinations: tuple[str, ...] = ()  # operation <destination> ids

    @property
    def name(self) -> str:
        """The point name (empty for objects without one)."""
        return self.attrs.get("name", "")


@dataclass(frozen=True)
class NodeRef:
    """A piece main-path node."""

    id_object: str
    type: str
    reverse: bool = False
    excluded: bool = False


@dataclass(frozen=True)
class PieceDef:
    """A pattern piece: ordered main-path nodes plus raw attributes."""

    name: str
    nodes: tuple[NodeRef, ...]
    attrs: dict[str, str] = field(default_factory=dict)
    data: dict[str, str] = field(default_factory=dict)
    grainline: dict[str, str] = field(default_factory=dict)


@dataclass
class PatternModel:
    """Everything parsed from a .sm2d. ``unit`` is the pattern unit all formulas/coordinates are in."""

    unit: str
    increments: dict[str, IncrementDef] = field(default_factory=dict)
    tools: dict[str, ToolDef] = field(default_factory=dict)
    modeling: dict[str, str] = field(default_factory=dict)  # in-use copy id -> source id
    op_dest: dict[str, tuple[str, str]] = field(default_factory=dict)  # destination id -> (operation id, source id)
    pieces: list[PieceDef] = field(default_factory=list)

    def resolve(self, id_: str) -> str:
        """Map an in-use (modeling) id to the calculation object id."""
        return self.modeling.get(id_, id_)

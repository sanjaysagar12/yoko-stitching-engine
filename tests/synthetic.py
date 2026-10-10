"""Tiny hand-written .sm2d / .smis documents for tests (no files on disk)."""

from __future__ import annotations


def smis(hip: float = 100.0, unit: str = "cm") -> str:
    return (
        f'<smis><unit>{unit}</unit><body-measurements><m name="hip_circ" value="{hip!r}"/></body-measurements></smis>'
    )


def sm2d(unit: str, body: str, pieces: str, increments: str = "") -> str:
    return (
        f"<pattern><unit>{unit}</unit><increments>{increments}</increments>"
        f'<draftBlock name="b"><calculation>{body}</calculation></draftBlock>'
        f"<pieces>{pieces}</pieces></pattern>"
    )


def piece(name: str, ids: list[str], types: dict[str, str] | None = None) -> str:
    types = types or {}
    nodes = "".join(f'<node idObject="{i}" type="{types.get(i, "NodePoint")}"/>' for i in ids)
    return f'<piece name="{name}"><data quantity="1" onFold="false"/><nodes>{nodes}</nodes></piece>'


def single(id_: str, x: float, y: float) -> str:
    return f'<point id="{id_}" name="P{id_}" type="single" x="{x!r}" y="{y!r}"/>'


def end_line(id_: str, base: str, length: str, angle: str) -> str:
    return f'<point id="{id_}" name="P{id_}" type="endLine" basePoint="{base}" length="{length}" angle="{angle}"/>'


def line_piece(unit: str, length: str) -> str:
    """Two points A -> B, B = A + ``length`` at angle 0."""
    return sm2d(unit, single("1", 0, 0) + end_line("2", "1", length, "0"), piece("Line", ["1", "2"]))


def rect_with_curve(k: float, unit: str) -> str:
    """Piece using hip_circ, an increment and a spline with formula handles.

    ``k`` scales the literals (1 for cm, 10 for mm) so both variants describe the same garment.
    """
    body = (
        single("1", 5 * k, 5 * k)
        + end_line("2", "1", "#W", "0")
        + end_line("3", "2", f"{12 * k}", "270")
        + end_line("4", "1", f"{12 * k}", "270")
        + f'<spline id="9" type="simpleInteractive" point1="2" point4="3" angle1="300" '
        f'length1="{2 * k}" angle2="240" length2="hip_circ/20*0.6"/>'
    )
    p = piece("RectCurve", ["1", "2", "9", "3", "4"], {"9": "NodeSpline"})
    inc = f'<increment name="#W" formula="hip_circ/4+{2 * k}"/>'
    return sm2d(unit, body, p, inc)

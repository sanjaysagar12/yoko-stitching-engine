"""Specific errors of the alignment engine. Messages name the offending piece / point / measurement."""

from __future__ import annotations


class AlignmentError(Exception):
    """Base class for all alignment engine errors."""


class UnknownPieceError(AlignmentError):
    """A placement, stitch or ring refers to a piece that has no piece bundle."""

    def __init__(self, name: str, known: list[str]) -> None:
        self.name = name
        super().__init__(f"unknown piece {name!r}; available: {', '.join(sorted(known)[:20])}")


class UnknownPointError(AlignmentError):
    """A placement refers to an arrangement point the avatar does not have."""

    def __init__(self, name: str, known: list[str]) -> None:
        self.name = name
        super().__init__(f"unknown arrangement point {name!r}; known: {', '.join(sorted(known)[:12])} ...")


class UnknownNamedPointError(AlignmentError):
    """A reference / stitch point name that is not a (usable) named point of the piece."""

    def __init__(self, piece: str, name: str, known: list[str]) -> None:
        self.piece = piece
        self.name = name
        super().__init__(f"piece {piece!r} has no usable named point {name!r}; available: {', '.join(known[:15])} ...")


class UnknownMeasurementError(AlignmentError):
    """A ring check refers to an avatar measurement that is not in the avatar bundle."""

    def __init__(self, name: str, known: list[str]) -> None:
        self.name = name
        super().__init__(f"avatar has no measurement {name!r}; available: {', '.join(known)}")


class BundleError(AlignmentError):
    """An avatar or piece folder is incomplete or inconsistent."""


class WrapWidthExceededWarning(UserWarning):
    """A piece is wider than the circumference it is wrapped around (it is clamped, not scaled)."""


class DerivedOfDerivedError(AlignmentError):
    """A mirror clone whose source is itself a derived piece (not allowed: mirror the base piece; copies may)."""

    def __init__(self, derived_id: str, source: str) -> None:
        self.derived_id = derived_id
        super().__init__(
            f"derived piece {derived_id!r}: source {source!r} is itself derived; mirror the base piece instead "
            "(only a copy may be made of a derived piece)"
        )


class DerivedCycleError(AlignmentError):
    """Derived pieces whose sources form a cycle (``A`` copies ``B`` copies ``A``)."""

    def __init__(self, chain: list[str]) -> None:
        self.chain = chain
        super().__init__(f"derived pieces form a cycle: {' -> '.join(chain)}")


class NotAMirrorCloneError(AlignmentError):
    """A pair asked for a plain copy (``op: "copy"``): only a mirror clone makes a pair."""

    def __init__(self, derived_id: str) -> None:
        self.derived_id = derived_id
        super().__init__(f"{derived_id!r} is a copy, not a mirror clone; place it like any other piece")


class DuplicatePieceError(AlignmentError):
    """A derived piece id that is already used by a piece or by another derived piece."""

    def __init__(self, piece_id: str) -> None:
        self.piece_id = piece_id
        super().__init__(f"piece id {piece_id!r} is already in use")


class AxisError(AlignmentError):
    """A mirror axis that cannot be used (two coincident points)."""


class ReferenceNotOnAxisError(AlignmentError):
    """A centre-line pair needs a reference point on the mirror axis."""

    def __init__(self, reference: str, distance_m: float) -> None:
        self.reference = reference
        self.distance_m = distance_m
        super().__init__(
            f"reference {reference!r} is {distance_m * 1000:.1f} mm from the mirror axis; a centre-line pair needs a "
            "named point on the axis (for example a point on the centre-front edge)"
        )


class NoMirroredPointError(AlignmentError):
    """An arrangement point whose left/right counterpart does not exist."""

    def __init__(self, point: str, wanted: str) -> None:
        self.point = point
        super().__init__(f"arrangement point {point!r} has no mirrored counterpart {wanted!r}")


class RingNotFoundError(AlignmentError):
    """No body shell (ring) for a bounding volume, or a ring name that does not exist."""


class StandoffExceeded(UserWarning):
    """The standoff needed to keep the pieces apart is larger than ``max_standoff`` (the arrangement uses the limit)."""


class ClearanceBodyViolation(UserWarning):
    """A piece is closer to the avatar than the required body clearance."""


class GapPiecesViolation(UserWarning):
    """Two pieces are closer than the required gap."""


class SizeChangedError(AlignmentError):
    """A placed piece is not an exact copy of its flat pattern piece (only possible with a distorting wrap)."""

    def __init__(self, piece: str, check: dict[str, float], tolerance_pct: float) -> None:
        self.piece = piece
        self.check = check
        nums = ", ".join(f"{k} {v:.3f}" for k, v in check.items() if k.endswith("_pct"))
        super().__init__(
            f"piece {piece!r} would change size ({nums}; the limit is {tolerance_pct} %); the placement is not returned"
        )


class ArrangementInfeasible(UserWarning):
    """The clearance and gap constraints cannot all be met within ``max_standoff`` and ``max_shift``; the best layout
    is returned with ``validity.valid = false`` and the violations."""

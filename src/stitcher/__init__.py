"""Stitching: a stitching JSON + the alignment result of an assembly -> sewing lines with point-to-point
correspondences on the placed pieces (docs/stitching.md). No simulation, nothing is moved.

The stitcher reads the alignment engine's output through files only (``result.json`` of an ``assembler run``, the
piece bundles, the avatar OBJ); it imports no engine. The only foreign import is ``assembler.render`` (the CPU
renderer) for the images.
"""

from .errors import StitchingError

__all__ = ["StitchingError"]

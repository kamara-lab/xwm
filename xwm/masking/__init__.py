"""Mask samplers: the choice of *what* a predictive world model predicts.

All samplers share one signature -- ``sampler(key) -> MaskBatch`` -- and return
static shapes, so a sampled mask can be fed straight into a jitted step.
"""

from .base import MaskBatch, block_shapes, boolean_mask, complement, gather, subsample
from .block import MultiBlockMask2d
from .random import RandomMask, random_split
from .temporal import TemporalSplit, frame_indices
from .tube import TubeMask3d, long_range_tubes, short_range_tubes

__all__ = [
    "MaskBatch",
    "MultiBlockMask2d",
    "RandomMask",
    "TemporalSplit",
    "TubeMask3d",
    "block_shapes",
    "boolean_mask",
    "complement",
    "frame_indices",
    "gather",
    "long_range_tubes",
    "random_split",
    "short_range_tubes",
    "subsample",
]

"""Multi-block masking for images (I-JEPA).

I-JEPA's key design choice is *what* to predict: several large, roughly square
blocks, from a context that excludes them. Predicting big contiguous regions
forces semantic rather than textural features -- you cannot interpolate a whole
block from its neighbours' pixels.

References
----------
Assran et al., *Self-Supervised Learning from Images with a Joint-Embedding
Predictive Architecture* (I-JEPA), CVPR 2023. arXiv:2301.08243 -- multi-block
masking, and the finding that large, roughly square target blocks are what force
semantic rather than textural features.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from ..core.module import Module
from ..core.types import PRNGKey
from .base import (
    MaskBatch,
    expected_context_fraction,
    sample_context,
    snap_area,
    to_numpy_rng,
)


def sample_block(
    rng: np.random.Generator, grid: tuple[int, int], shape: tuple[int, int]
) -> np.ndarray:
    """Flat indices of an ``(h, w)`` block placed uniformly at random in ``grid``."""
    gh, gw = grid
    h, w = min(shape[0], gh), min(shape[1], gw)
    top = rng.integers(0, gh - h + 1)
    left = rng.integers(0, gw - w + 1)
    rows = np.arange(top, top + h)[:, None]
    cols = np.arange(left, left + w)[None, :]
    return (rows * gw + cols).reshape(-1)


class MultiBlockMask2d(Module):
    """I-JEPA style multi-block sampler over a 2-D patch grid.

    Args:
        grid: ``(gh, gw)`` patch grid, e.g. ``PatchEmbed2d.grid``.
        n_targets: number of target blocks to predict (I-JEPA uses 4).
        target_scale: fraction of the grid covered by each target block.
        aspect_range: allowed width/height ratios for target blocks.
        context_scale: fraction of *all* tokens kept as context. ``None``
            derives a value the sampler can reliably satisfy given the expected
            overlap between target blocks.
        max_tries: resampling budget when a draw leaves too little context.

    Every target block holds exactly :attr:`target_size` tokens -- the closest
    usable size to ``round(target_scale * n_tokens)``. The aspect ratio is
    redrawn each call from the divisor pairs of that area, so block geometry
    varies while every returned array's shape stays static, which is what lets
    the training step be jitted once.
    """

    grid: tuple[int, int] = eqx.field(static=True)
    n_targets: int = eqx.field(static=True)
    target_size: int = eqx.field(static=True)
    context_size: int = eqx.field(static=True)
    shapes: tuple[tuple[int, int], ...] = eqx.field(static=True)
    max_tries: int = eqx.field(static=True)

    def __init__(
        self,
        grid: tuple[int, int],
        *,
        n_targets: int = 4,
        target_scale: float = 0.15,
        aspect_range: tuple[float, float] = (0.75, 1.5),
        context_scale: float | None = None,
        max_tries: int = 100,
    ):
        gh, gw = grid
        n_tokens = gh * gw
        requested = max(1, int(round(target_scale * n_tokens)))
        target_size, shapes = snap_area(requested, aspect_range, (gh, gw))
        if context_scale is None:
            context_scale = expected_context_fraction(target_size / n_tokens, n_targets)
        context_size = max(1, int(round(context_scale * n_tokens)))
        if context_size >= n_tokens:
            raise ValueError("context_scale must leave at least one token masked")
        self.grid = (gh, gw)
        self.n_targets = n_targets
        self.target_size = target_size
        self.context_size = context_size
        self.shapes = tuple(shapes)
        self.max_tries = max_tries

    @property
    def n_tokens(self) -> int:
        return self.grid[0] * self.grid[1]

    def _draw_targets(self, rng: np.random.Generator) -> list[np.ndarray]:
        return [
            sample_block(rng, self.grid, self.shapes[rng.integers(0, len(self.shapes))])
            for _ in range(self.n_targets)
        ]

    def __call__(self, key: PRNGKey) -> MaskBatch:
        rng = to_numpy_rng(key)
        targets, context = sample_context(
            rng, self._draw_targets, self.n_tokens, self.context_size, self.max_tries
        )
        return MaskBatch(
            context=jnp.asarray(context, jnp.int32),
            targets=jnp.asarray(targets, jnp.int32),
        )

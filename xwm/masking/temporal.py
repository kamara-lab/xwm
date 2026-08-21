"""Temporal context/target splits for action-conditioned prediction.

Action-conditioned world models (V-JEPA 2-AC and friends) do not mask random
regions: they see a prefix of the clip and must predict *whole future frames*
given the actions taken. The split is therefore along time, and each target
block is one future frame's worth of tokens.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from ..core.module import Module
from ..core.types import PRNGKey
from .base import MaskBatch


def frame_indices(grid: tuple[int, int, int], t: int) -> np.ndarray:
    """Flat token indices belonging to temporal position ``t``."""
    _, gh, gw = grid
    return np.arange(t * gh * gw, (t + 1) * gh * gw)


class TemporalSplit(Module):
    """Context is the first ``n_context_frames``; targets are the frames after.

    Args:
        grid: ``(gt, gh, gw)`` token grid.
        n_context_frames: number of leading temporal positions kept visible.
        horizon: how many future frames to predict; ``None`` uses all remaining.

    Deterministic -- the ``key`` argument exists only so every sampler in xwm
    shares one call signature.
    """

    grid: tuple[int, int, int] = eqx.field(static=True)
    n_context_frames: int = eqx.field(static=True)
    horizon: int = eqx.field(static=True)

    def __init__(
        self,
        grid: tuple[int, int, int],
        *,
        n_context_frames: int = 1,
        horizon: int | None = None,
    ):
        gt = grid[0]
        max_horizon = gt - n_context_frames
        if max_horizon < 1:
            raise ValueError(f"grid has {gt} temporal positions, nothing left to predict")
        self.grid = tuple(grid)
        self.n_context_frames = n_context_frames
        self.horizon = max_horizon if horizon is None else min(horizon, max_horizon)

    def __call__(self, key: PRNGKey | None = None) -> MaskBatch:
        context = np.concatenate(
            [frame_indices(self.grid, t) for t in range(self.n_context_frames)]
        )
        targets = np.stack(
            [
                frame_indices(self.grid, self.n_context_frames + h)
                for h in range(self.horizon)
            ]
        )
        return MaskBatch(
            context=jnp.asarray(context, jnp.int32),
            targets=jnp.asarray(targets, jnp.int32),
        )

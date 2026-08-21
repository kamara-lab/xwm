"""Uniform random masking.

The simplest context/target split, and the only sampler in xwm that is pure
JAX: it needs no combinatorial search, so it can live inside ``jit`` and use a
fresh key every step.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr

from ..core.module import Module
from ..core.types import Array, PRNGKey
from .base import MaskBatch


def random_split(key: PRNGKey, n_tokens: int, n_context: int) -> tuple[Array, Array]:
    """Partition ``n_tokens`` into ``n_context`` visible and the rest masked.

    Uses the argsort-of-noise trick, so both outputs have static shapes and the
    whole thing is jittable and differentiable-through-free.
    """
    order = jnp.argsort(jr.uniform(key, (n_tokens,)))
    return jnp.sort(order[:n_context]), jnp.sort(order[n_context:])


class RandomMask(Module):
    """Sample a uniformly random context set; predict everything else.

    Args:
        n_tokens: size of the token grid.
        mask_ratio: fraction of tokens to predict.
        n_targets: split the masked tokens into this many equal target blocks.
    """

    n_tokens: int = eqx.field(static=True)
    mask_ratio: float = eqx.field(static=True)
    n_targets: int = eqx.field(static=True)

    def __init__(self, n_tokens: int, mask_ratio: float = 0.75, n_targets: int = 1):
        n_masked = int(round(mask_ratio * n_tokens))
        if not 0 < n_masked < n_tokens:
            raise ValueError(f"mask_ratio={mask_ratio} leaves nothing to predict or to see")
        if n_masked % n_targets:
            n_masked -= n_masked % n_targets  # keep target blocks equal-sized
        self.n_tokens = n_tokens
        self.mask_ratio = n_masked / n_tokens
        self.n_targets = n_targets

    @property
    def n_masked(self) -> int:
        return int(round(self.mask_ratio * self.n_tokens))

    def __call__(self, key: PRNGKey) -> MaskBatch:
        ctx, masked = random_split(key, self.n_tokens, self.n_tokens - self.n_masked)
        return MaskBatch(context=ctx, targets=masked.reshape(self.n_targets, -1))

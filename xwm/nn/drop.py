"""Stochastic-depth (drop-path) regularisation."""

from __future__ import annotations

import equinox as eqx
import jax.random as jr

from ..core.module import Module
from ..core.types import Array, PRNGKey


class DropPath(Module):
    """Drop a residual branch entirely, with probability ``p``.

    Applied per-sample; since xwm modules are unbatched this is a single
    Bernoulli draw that zeroes (or rescales) the whole branch.
    """

    p: float = eqx.field(static=True)
    inference: bool

    def __init__(self, p: float = 0.0, *, inference: bool = False):
        self.p = p
        self.inference = inference

    def __call__(self, x: Array, *, key: PRNGKey | None = None) -> Array:
        if self.p <= 0.0 or self.inference or key is None:
            return x
        keep = 1.0 - self.p
        drop = jr.bernoulli(key, keep).astype(x.dtype)
        return x * drop / keep


def linear_droppath_schedule(depth: int, max_rate: float) -> list[float]:
    """Linearly increasing drop-path rates across depth (the ViT default)."""
    if depth == 1:
        return [max_rate]
    return [max_rate * i / (depth - 1) for i in range(depth)]

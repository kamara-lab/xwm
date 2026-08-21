"""Encoder for low-dimensional state observations.

Robot proprioception -- joint angles, velocities, end-effector pose -- is a short
vector, not an image. Running a ViT over it would be absurd, so this is an MLP
that emits a single token, keeping the ``(N, D)`` contract the rest of the
library expects (with ``N = 1``) so a family does not care which encoder it got.

Useful on its own for state-based control, and as the second half of a
multimodal encoder when combined with an image encoder's tokens.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.random as jr

from ..core.module import Module
from ..core.random import resolve_key
from ..core.types import Array, PRNGKey
from ..nn.mlp import Mlp
from ..nn.norm import LayerNorm

__all__ = ["StateEncoder"]


class StateEncoder(Module):
    """Embed a state vector to ``(1, D)`` tokens.

    Args:
        state_dim: length of the observation vector.
        embed_dim: token width.
        hidden_dim: MLP width; defaults to ``4 * embed_dim``.
        depth: number of MLP blocks.
        normalize: LayerNorm the input, which matters when the components have
            wildly different units (radians next to metres next to velocities).
    """

    layers: list[Mlp]
    input_norm: LayerNorm | None
    out_norm: LayerNorm
    state_dim: int = eqx.field(static=True)
    embed_dim: int = eqx.field(static=True)

    def __init__(
        self,
        state_dim: int,
        embed_dim: int = 256,
        *,
        key: PRNGKey | None = None,
        hidden_dim: int | None = None,
        depth: int = 2,
        normalize: bool = True,
    ):
        key = resolve_key(key)
        keys = jr.split(key, depth)
        hidden_dim = hidden_dim or 4 * embed_dim
        dims = [state_dim] + [embed_dim] * depth
        self.layers = [
            Mlp(dims[i], hidden_dim, dims[i + 1], key=keys[i]) for i in range(depth)
        ]
        self.input_norm = LayerNorm(state_dim) if normalize else None
        self.out_norm = LayerNorm(embed_dim)
        self.state_dim = state_dim
        self.embed_dim = embed_dim

    @property
    def n_tokens(self) -> int:
        return 1

    @property
    def grid(self) -> tuple[int, ...]:
        return (1,)

    def __call__(self, x: Array, *, key: PRNGKey | None = None) -> Array:
        """``x``: ``(state_dim,)``. Returns ``(1, embed_dim)``."""
        keys = [None] * len(self.layers) if key is None else list(jr.split(key, len(self.layers)))
        h = self.input_norm(x) if self.input_norm is not None else x
        for layer, k in zip(self.layers, keys, strict=True):
            h = layer(h, key=k)
            h = jax.nn.gelu(h)
        return self.out_norm(h)[None, :]

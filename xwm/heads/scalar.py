"""An MLP trunk with a categorical scalar output -- rewards and values."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from ..core.module import Module
from ..core.random import resolve_key
from ..core.types import Array, PRNGKey
from ..nn.norm import LayerNorm
from .categorical import CategoricalScalar

__all__ = ["ScalarHead"]


class ScalarHead(Module):
    """Predict a scalar from a latent (and optionally an action).

    Emits *logits over bins* rather than a number; call :meth:`value` to decode
    or :meth:`loss` to train. See :mod:`xwm.heads.categorical` for why that
    beats a squared error here.

    Args:
        latent_dim: width of the input latent.
        action_dim: width of an action to condition on; ``0`` for a
            state-only head (a value function) rather than a state-action one
            (a reward model or a Q-function).
        scalar: the bin grid. Defaults to TD-MPC2's 101 bins with symlog.
    """

    layers: list[eqx.nn.Linear]
    norms: list[LayerNorm]
    out: eqx.nn.Linear
    scalar: CategoricalScalar
    latent_dim: int = eqx.field(static=True)
    action_dim: int = eqx.field(static=True)

    def __init__(
        self,
        latent_dim: int,
        *,
        action_dim: int = 0,
        key: PRNGKey | None = None,
        hidden_dim: int = 512,
        depth: int = 2,
        scalar: CategoricalScalar | None = None,
    ):
        key = resolve_key(key)
        scalar = scalar or CategoricalScalar()
        keys = jr.split(key, depth + 1)
        dims = [latent_dim + action_dim] + [hidden_dim] * depth
        self.layers = [eqx.nn.Linear(dims[i], hidden_dim, key=keys[i]) for i in range(depth)]
        self.norms = [LayerNorm(hidden_dim) for _ in range(depth)]
        self.out = eqx.nn.Linear(hidden_dim, scalar.n_bins, key=keys[-1])
        self.scalar = scalar
        self.latent_dim = latent_dim
        self.action_dim = action_dim

    def logits(self, z: Array, action: Array | None = None) -> Array:
        """``(n_bins,)`` logits."""
        if self.action_dim:
            if action is None:
                raise ValueError("this head is action-conditioned; pass an action")
            h = jnp.concatenate([z, jnp.atleast_1d(action)])
        else:
            h = z
        for layer, norm in zip(self.layers, self.norms, strict=True):
            h = jax.nn.mish(norm(layer(h)))
        return self.out(h)

    def value(self, z: Array, action: Array | None = None) -> Array:
        """Decoded scalar."""
        return self.scalar.decode(self.logits(z, action))

    def loss(self, z: Array, target: Array, action: Array | None = None) -> Array:
        """Cross-entropy against the two-hot encoding of ``target``."""
        return self.scalar.loss(self.logits(z, action), target)

    def __call__(self, z: Array, action: Array | None = None) -> Array:
        return self.value(z, action)

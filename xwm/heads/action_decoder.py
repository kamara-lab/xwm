"""Recover the action from the change it caused in the latent.

An inverse dynamics head, and the one loss term in :mod:`xwm.objectives` that
prevents collapse without any distributional assumption at all. If ``a_t`` can
be read back from ``z_{t+1} - z_t``, then adjacent embeddings cannot be equal
and different actions must move the latent differently -- which is precisely the
property a planner needs and a prediction loss alone does not guarantee.

The *displacement* is the input, not the pair of endpoints. Decoding from
``concat(z_t, z_{t+1})`` lets the head solve the task from either endpoint's
absolute position -- a shortcut that leaves the transition geometry
unconstrained, which is the failure the difference form removes.

References
----------
*Delta-JEPA: Learning Action-Sensitive World Models via Latent Difference
Decoding*, 2026. arXiv:2606.31232 -- the latent difference action decoder, and
the ablation showing displacement beats endpoint concatenation.

Sobal et al., *Learning from Reward-Free Offline Data: A Case for Planning with
Latent Dynamics Models*, NeurIPS 2025. arXiv:2502.14819 -- an inverse-dynamics
term as one of PLDM's regularizers.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp

from ..core.module import Module
from ..core.random import resolve_key
from ..core.types import Array, PRNGKey
from ..nn.mlp import Mlp
from ..nn.transformer import mean_pool

__all__ = ["LatentDifferenceActionDecoder"]


class LatentDifferenceActionDecoder(Module):
    """``z_{t+1} - z_t -> a_t``.

    Args:
        latent_dim: encoder width.
        action_dim: what to predict.
        hidden_dim: MLP width; defaults to four times ``latent_dim``.

    The paper uses a small transformer with learned action queries over the
    token grid. An MLP over the pooled displacement is the default here because
    the latent this decodes is usually a single pooled vector per frame, where
    queries have nothing to attend over; pass a token grid and it pools first.
    """

    mlp: Mlp
    latent_dim: int = eqx.field(static=True)
    action_dim: int = eqx.field(static=True)

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        *,
        key: PRNGKey | None = None,
        hidden_dim: int | None = None,
    ):
        self.mlp = Mlp(
            latent_dim, hidden_dim or 4 * latent_dim, action_dim, key=resolve_key(key)
        )
        self.latent_dim = latent_dim
        self.action_dim = action_dim

    def __call__(self, delta: Array, *, key: PRNGKey | None = None) -> Array:
        """``delta``: ``(N, D)`` token displacements or a flat ``(D,)``. Returns ``(A,)``."""
        pooled = mean_pool(delta) if delta.ndim == 2 else delta
        return self.mlp(pooled, key=key)

    def loss(self, delta: Array, action: Array, *, key: PRNGKey | None = None) -> Array:
        """Mean squared error between the decoded action and the one taken."""
        return jnp.mean(jnp.square(self(delta, key=key) - action))

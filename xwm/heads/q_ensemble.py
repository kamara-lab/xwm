"""An ensemble of Q-functions, with a pessimistic aggregate.

Model-based value learning overestimates: the planner searches for actions that
look good, and anything the critic is wrongly optimistic about is exactly what
it will find. An ensemble whose members disagree gives a cheap uncertainty
signal, and taking the *minimum over a random subset* turns that disagreement
into pessimism -- optimistic outliers get suppressed rather than exploited.

Sampling a subset rather than always taking the min over all members keeps the
pessimism from becoming crushing as the ensemble grows. TD-MPC2 uses five
members and a subset of two.

References
----------
Chen et al., *Randomized Ensembled Double Q-Learning* (REDQ), ICLR 2021.
arXiv:2101.05982 -- the minimum over a random subset of an ensemble.

Fujimoto, van Hoof & Meger, *Addressing Function Approximation Error in
Actor-Critic Methods* (TD3), ICML 2018. arXiv:1802.09477 -- clipped double-Q,
the two-member case.

Hansen, Su & Wang, *TD-MPC2*, ICLR 2024. arXiv:2310.16828 -- five members with a
subset of two, the defaults used here.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr

from ..core.module import Module
from ..core.random import resolve_key
from ..core.types import Array, PRNGKey
from .categorical import CategoricalScalar
from .scalar import ScalarHead

__all__ = ["QEnsemble"]


class QEnsemble(Module):
    """``n_members`` action-conditioned scalar heads over a shared latent.

    Args:
        n_members: ensemble size.
        subset_size: how many members the pessimistic aggregate draws from.
    """

    members: list[ScalarHead]
    subset_size: int = eqx.field(static=True)

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        *,
        key: PRNGKey | None = None,
        n_members: int = 5,
        subset_size: int = 2,
        hidden_dim: int = 512,
        depth: int = 2,
        scalar: CategoricalScalar | None = None,
    ):
        key = resolve_key(key)
        if not 1 <= subset_size <= n_members:
            raise ValueError(f"subset_size {subset_size} must be in [1, {n_members}]")
        keys = jr.split(key, n_members)
        self.members = [
            ScalarHead(
                latent_dim,
                action_dim=action_dim,
                key=k,
                hidden_dim=hidden_dim,
                depth=depth,
                scalar=scalar,
            )
            for k in keys
        ]
        self.subset_size = subset_size

    @property
    def n_members(self) -> int:
        return len(self.members)

    def logits(self, z: Array, action: Array) -> Array:
        """``(n_members, n_bins)``."""
        return jnp.stack([member.logits(z, action) for member in self.members])

    def values(self, z: Array, action: Array) -> Array:
        """``(n_members,)`` decoded Q-values."""
        return jnp.stack([member.value(z, action) for member in self.members])

    def pessimistic(self, z: Array, action: Array, *, key: PRNGKey | None = None) -> Array:
        """Minimum over a random subset of members -- the aggregate to plan with.

        ``key`` is required during training (the subset must be resampled every
        step); pass ``None`` to use the min over *all* members, which is the
        deterministic choice for evaluation.
        """
        values = self.values(z, action)
        if key is None:
            return jnp.min(values)
        chosen = jr.choice(key, self.n_members, (self.subset_size,), replace=False)
        return jnp.min(values[chosen])

    def loss(self, z: Array, action: Array, target: Array) -> Array:
        """Mean cross-entropy across members against a shared scalar target."""
        return jnp.mean(
            jnp.stack([member.loss(z, target, action) for member in self.members])
        )

    def __call__(self, z: Array, action: Array, *, key: PRNGKey | None = None) -> Array:
        return self.pessimistic(z, action, key=key)

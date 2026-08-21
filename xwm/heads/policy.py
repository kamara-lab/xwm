"""A squashed-Gaussian policy head.

Two jobs in a model-based setting, and the second is the one people forget:

1. Act without planning, when there is no time to search.
2. **Seed the planner.** MPPI over a 7-DoF arm searches a space no sampler
   covers by chance; starting from a policy's proposals is what makes a few
   hundred samples enough. TD-MPC2 mixes policy-prior samples into every
   planning iteration for exactly this reason.

Actions are ``tanh``-squashed into ``[-1, 1]``, and the log-probability carries
the change-of-variables correction for that squashing -- omitting it is a
classic silent bug that biases the entropy term.

References
----------
Haarnoja et al., *Soft Actor-Critic: Off-Policy Maximum Entropy Deep
Reinforcement Learning with a Stochastic Actor*, ICML 2018. arXiv:1801.01290 --
the squashed-Gaussian policy and the tanh change-of-variables correction in the
log-probability (their Appendix C).

Hansen, Su & Wang, *TD-MPC2*, ICLR 2024. arXiv:2310.16828 -- using the policy as
a prior that seeds a sampling planner.
"""

from __future__ import annotations

from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from ..core.module import Module
from ..core.random import resolve_key
from ..core.types import Array, PRNGKey
from ..nn.norm import LayerNorm

__all__ = ["GaussianPolicy", "PolicyOutput"]

#: Clamp on the predicted log-std. Without it the policy can collapse to a
#: delta (log_std -> -inf) and stop exploring, or explode and emit noise.
LOG_STD_RANGE = (-5.0, 2.0)


class PolicyOutput(NamedTuple):
    """A sampled action with the statistics needed for a SAC-style update."""

    action: Array
    log_prob: Array
    mean: Array
    log_std: Array


class GaussianPolicy(Module):
    """``z -> tanh(Normal(mu(z), sigma(z)))``, bounded in ``[-1, 1]``."""

    layers: list[eqx.nn.Linear]
    norms: list[LayerNorm]
    out: eqx.nn.Linear
    latent_dim: int = eqx.field(static=True)
    action_dim: int = eqx.field(static=True)

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        *,
        key: PRNGKey | None = None,
        hidden_dim: int = 512,
        depth: int = 2,
    ):
        key = resolve_key(key)
        keys = jr.split(key, depth + 1)
        dims = [latent_dim] + [hidden_dim] * depth
        self.layers = [eqx.nn.Linear(dims[i], hidden_dim, key=keys[i]) for i in range(depth)]
        self.norms = [LayerNorm(hidden_dim) for _ in range(depth)]
        self.out = eqx.nn.Linear(hidden_dim, 2 * action_dim, key=keys[-1])
        self.latent_dim = latent_dim
        self.action_dim = action_dim

    def distribution(self, z: Array) -> tuple[Array, Array]:
        """``(mean, log_std)`` before squashing."""
        h = z
        for layer, norm in zip(self.layers, self.norms, strict=True):
            h = jax.nn.mish(norm(layer(h)))
        mean, log_std = jnp.split(self.out(h), 2)
        low, high = LOG_STD_RANGE
        # Squash into range rather than hard-clipping, so the gradient survives.
        log_std = low + 0.5 * (high - low) * (jnp.tanh(log_std) + 1.0)
        return mean, log_std

    def act(self, z: Array, *, key: PRNGKey | None = None) -> Array:
        """Sample an action, or return the deterministic mean when ``key`` is None."""
        mean, log_std = self.distribution(z)
        if key is None:
            return jnp.tanh(mean)
        return jnp.tanh(mean + jnp.exp(log_std) * jr.normal(key, mean.shape))

    def sample(self, z: Array, key: PRNGKey) -> PolicyOutput:
        """Sample with the log-probability of the squashed action."""
        mean, log_std = self.distribution(z)
        std = jnp.exp(log_std)
        noise = jr.normal(key, mean.shape)
        pre_tanh = mean + std * noise
        action = jnp.tanh(pre_tanh)
        # Gaussian log-prob, then the tanh change-of-variables correction.
        log_prob = jnp.sum(
            -0.5 * noise**2 - log_std - 0.5 * jnp.log(2.0 * jnp.pi)
        )
        log_prob -= jnp.sum(jnp.log(jnp.clip(1.0 - action**2, 1e-6, None)))
        return PolicyOutput(action=action, log_prob=log_prob, mean=mean, log_std=log_std)

    def __call__(self, z: Array, *, key: PRNGKey | None = None) -> Array:
        return self.act(z, key=key)

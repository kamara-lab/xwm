"""Sampling-based planners: CEM and MPPI.

Both solve the same problem -- pick an action sequence whose imagined rollout
has the lowest cost -- and both do it without gradients, which is what you want
when the learned dynamics are only locally accurate and their gradients are
noisy. They differ only in how they turn the scored samples into the next
proposal distribution: CEM keeps the best few and refits, MPPI takes a
softmax-weighted average of all of them.

The whole search is one ``jit``-able function: candidate rollouts are ``vmap``ed
and the refinement iterations are a ``lax.fori_loop``, so a plan is a single
device call rather than a Python loop over thousands of small ones.

References
----------
Williams, Aldrich & Theodorou, *Model Predictive Path Integral Control using
Covariance Variable Importance Sampling*, 2015. arXiv:1509.01149 -- the
softmax-weighted update in :class:`MPPI`.

Rubinstein & Kroese, *The Cross-Entropy Method*, Springer 2004 -- the elite
refit in :class:`CEM`.

Hansen, Su & Wang, *TD-MPC2*, ICLR 2024. arXiv:2310.16828 -- the variant that
seeds the sampling distribution with a learned policy prior.
"""

from __future__ import annotations

from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from ..core.module import Module
from ..core.rollout import rollout_cost
from ..core.types import Array, LatentDynamics, PRNGKey
from .cost import CostFn


class Plan(NamedTuple):
    """The result of a planning call.

    Attributes:
        actions: ``(H, A)`` chosen action sequence.
        cost: its predicted cost under the world model.
        mean: ``(H, A)`` final proposal mean -- pass it back as ``init_mean``
            next step to warm-start, which is most of what makes receding-horizon
            control cheap.
        std: ``(H, A)`` final proposal spread.
    """

    actions: Array
    cost: Array
    mean: Array
    std: Array


def _evaluate(
    dynamics: LatentDynamics,
    z0: Array,
    candidates: Array,
    cost_fn: CostFn,
    cost_on: str = "next",
) -> Array:
    """Cost of every candidate action sequence: ``(S, H, A) -> (S,)``."""
    return jax.vmap(
        lambda a: rollout_cost(dynamics, z0, a, cost_fn, cost_on=cost_on)
    )(candidates)


class CEM(Module):
    """Cross-entropy method: iteratively refit a Gaussian to the elite samples.

    Args:
        horizon: planning horizon in steps.
        action_dim: action dimensionality.
        n_samples: candidates per iteration.
        n_elites: how many best candidates define the next proposal.
        n_iters: refinement iterations.
        low, high: action bounds; candidates are clipped into them.
        init_std: initial per-dimension spread.
        min_std: floor on the spread, so the search cannot collapse to a point
            and stop exploring.
        momentum: smoothing of the proposal across iterations, in ``[0, 1)``.
    """

    horizon: int = eqx.field(static=True)
    action_dim: int = eqx.field(static=True)
    n_samples: int = eqx.field(static=True)
    n_elites: int = eqx.field(static=True)
    n_iters: int = eqx.field(static=True)
    low: Array
    high: Array
    init_std: float = eqx.field(static=True)
    min_std: float = eqx.field(static=True)
    momentum: float = eqx.field(static=True)
    cost_on: str = eqx.field(static=True)

    def __init__(
        self,
        horizon: int,
        action_dim: int,
        *,
        n_samples: int = 512,
        n_elites: int = 64,
        n_iters: int = 6,
        low: float | Array = -1.0,
        high: float | Array = 1.0,
        init_std: float = 0.5,
        min_std: float = 0.05,
        momentum: float = 0.1,
        cost_on: str = "next",
    ):
        if n_elites > n_samples:
            raise ValueError(f"n_elites={n_elites} exceeds n_samples={n_samples}")
        self.horizon = horizon
        self.action_dim = action_dim
        self.n_samples = n_samples
        self.n_elites = n_elites
        self.n_iters = n_iters
        self.low = jnp.broadcast_to(jnp.asarray(low, jnp.float32), (action_dim,))
        self.high = jnp.broadcast_to(jnp.asarray(high, jnp.float32), (action_dim,))
        self.init_std = init_std
        self.min_std = min_std
        self.momentum = momentum
        self.cost_on = cost_on

    def plan(
        self,
        key: PRNGKey,
        dynamics: LatentDynamics,
        z0: Array,
        cost_fn: CostFn,
        *,
        init_mean: Array | None = None,
    ) -> Plan:
        """Search for a low-cost action sequence from latent state ``z0``."""
        shape = (self.horizon, self.action_dim)
        mean = jnp.zeros(shape) if init_mean is None else jnp.asarray(init_mean)
        std = jnp.full(shape, self.init_std)

        def iteration(i, carry):
            mean, std = carry
            noise = jr.normal(jr.fold_in(key, i), (self.n_samples, *shape))
            candidates = jnp.clip(mean + std * noise, self.low, self.high)
            costs = _evaluate(dynamics, z0, candidates, cost_fn, self.cost_on)
            elites = candidates[jnp.argsort(costs)[: self.n_elites]]
            new_mean = jnp.mean(elites, axis=0)
            new_std = jnp.maximum(jnp.std(elites, axis=0), self.min_std)
            m = self.momentum
            return m * mean + (1 - m) * new_mean, m * std + (1 - m) * new_std

        mean, std = jax.lax.fori_loop(0, self.n_iters, iteration, (mean, std))
        actions = jnp.clip(mean, self.low, self.high)
        cost = rollout_cost(dynamics, z0, actions, cost_fn, cost_on=self.cost_on)
        return Plan(actions=actions, cost=cost, mean=mean, std=std)


class MPPI(Module):
    """Model-predictive path integral: softmax-weighted average of all samples.

    Unlike CEM's hard elite cut, every candidate contributes in proportion to
    ``exp(-cost / temperature)``. The soft weighting makes the update smoother
    across control steps, which matters when the planner is in a feedback loop.

    Args:
        temperature: lower values approach CEM's greedy behaviour; higher values
            average more broadly. Costs are shifted by their minimum before
            exponentiating, so the scale is relative and numerically safe.
        noise_std: proposal spread, held fixed rather than refit.
    """

    horizon: int = eqx.field(static=True)
    action_dim: int = eqx.field(static=True)
    n_samples: int = eqx.field(static=True)
    n_iters: int = eqx.field(static=True)
    low: Array
    high: Array
    temperature: float = eqx.field(static=True)
    noise_std: float = eqx.field(static=True)
    cost_on: str = eqx.field(static=True)

    def __init__(
        self,
        horizon: int,
        action_dim: int,
        *,
        n_samples: int = 512,
        n_iters: int = 4,
        low: float | Array = -1.0,
        high: float | Array = 1.0,
        temperature: float = 1.0,
        noise_std: float = 0.5,
        cost_on: str = "next",
    ):
        self.horizon = horizon
        self.action_dim = action_dim
        self.n_samples = n_samples
        self.n_iters = n_iters
        self.low = jnp.broadcast_to(jnp.asarray(low, jnp.float32), (action_dim,))
        self.high = jnp.broadcast_to(jnp.asarray(high, jnp.float32), (action_dim,))
        self.temperature = temperature
        self.noise_std = noise_std
        self.cost_on = cost_on

    def plan(
        self,
        key: PRNGKey,
        dynamics: LatentDynamics,
        z0: Array,
        cost_fn: CostFn,
        *,
        init_mean: Array | None = None,
    ) -> Plan:
        shape = (self.horizon, self.action_dim)
        mean = jnp.zeros(shape) if init_mean is None else jnp.asarray(init_mean)

        def iteration(i, mean):
            noise = jr.normal(jr.fold_in(key, i), (self.n_samples, *shape))
            candidates = jnp.clip(mean + self.noise_std * noise, self.low, self.high)
            costs = _evaluate(dynamics, z0, candidates, cost_fn, self.cost_on)
            weights = jax.nn.softmax(-(costs - jnp.min(costs)) / self.temperature)
            return jnp.einsum("s,sha->ha", weights, candidates)

        mean = jax.lax.fori_loop(0, self.n_iters, iteration, mean)
        actions = jnp.clip(mean, self.low, self.high)
        cost = rollout_cost(dynamics, z0, actions, cost_fn, cost_on=self.cost_on)
        return Plan(
            actions=actions,
            cost=cost,
            mean=mean,
            std=jnp.full(shape, self.noise_std),
        )

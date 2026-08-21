"""Gradient-based planning.

The world model is differentiable, so the action sequence can be optimised
directly by backpropagating the rollout cost through it. When it works this is
dramatically more sample-efficient than sampling -- one gradient step uses the
model's full Jacobian, where a CEM iteration uses only a ranking.

It is not the default for a reason. Gradients through a long rollout of a
*learned* model are often ill-conditioned, and the optimiser will happily find
adversarial action sequences that exploit inaccuracies rather than solve the
task. Prefer :class:`~xwm.planning.CEM` unless you have checked that the
gradients are trustworthy over your horizon, or use this to polish a
sampling-based solution.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from ..core.module import Module
from ..core.rollout import rollout_cost
from ..core.types import Array, LatentDynamics, PRNGKey
from .cost import CostFn
from .sampling import Plan


class GradientPlanner(Module):
    """Optimise an action sequence by gradient descent on the rollout cost.

    Args:
        horizon: planning horizon.
        action_dim: action dimensionality.
        n_steps: optimisation steps.
        learning_rate: Adam step size on the actions.
        low, high: bounds, enforced by projection after each step.
    """

    horizon: int = eqx.field(static=True)
    action_dim: int = eqx.field(static=True)
    n_steps: int = eqx.field(static=True)
    learning_rate: float = eqx.field(static=True)
    low: Array
    high: Array
    cost_on: str = eqx.field(static=True)

    def __init__(
        self,
        horizon: int,
        action_dim: int,
        *,
        n_steps: int = 100,
        learning_rate: float = 0.05,
        low: float | Array = -1.0,
        high: float | Array = 1.0,
        cost_on: str = "next",
    ):
        self.horizon = horizon
        self.action_dim = action_dim
        self.n_steps = n_steps
        self.learning_rate = learning_rate
        self.low = jnp.broadcast_to(jnp.asarray(low, jnp.float32), (action_dim,))
        self.high = jnp.broadcast_to(jnp.asarray(high, jnp.float32), (action_dim,))
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
        actions = jnp.zeros(shape) if init_mean is None else jnp.asarray(init_mean)
        actions = jnp.clip(actions, self.low, self.high)
        optimizer = optax.adam(self.learning_rate)
        opt_state = optimizer.init(actions)

        objective = jax.value_and_grad(
            lambda a: rollout_cost(dynamics, z0, a, cost_fn, cost_on=self.cost_on)
        )

        def step(_, carry):
            actions, opt_state = carry
            _, grads = objective(actions)
            updates, opt_state = optimizer.update(grads, opt_state)
            actions = jnp.clip(optax.apply_updates(actions, updates), self.low, self.high)
            return actions, opt_state

        actions, _ = jax.lax.fori_loop(0, self.n_steps, step, (actions, opt_state))
        return Plan(
            actions=actions,
            cost=rollout_cost(dynamics, z0, actions, cost_fn, cost_on=self.cost_on),
            mean=actions,
            std=jnp.zeros(shape),
        )

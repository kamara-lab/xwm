"""Cost functions defined in latent space.

A planner needs to score imagined trajectories, and a predictive world model
gives it nowhere to do that but the latent space. That turns out to be an
advantage: a goal specified as an *image* becomes a goal specified as an
*embedding*, and distance in a representation that discards uncontrollable
detail is a far better objective than pixel distance -- which would happily
trade the task for a matching background.

Every cost is a callable ``(z, a, t) -> scalar`` matching
:func:`xwm.core.rollout_cost`.

References
----------
Hansen, Su & Wang, *TD-MPC2*, ICLR 2024. arXiv:2310.16828 -- :func:`return_cost`,
the discounted-reward-plus-terminal-value planning objective.

Assran et al., *V-JEPA 2*, 2025 -- planning to a goal *image* by comparing
embeddings, which is what :func:`goal_cost` implements.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import jax.numpy as jnp

from ..core.types import Array

CostFn = Callable[[Array, Array, Array], Array]
Distance = Literal["l1", "l2", "cosine"]


def latent_distance(z: Array, z_goal: Array, kind: Distance = "l2") -> Array:
    """Scalar distance between two latent states of identical shape."""
    if kind == "l1":
        return jnp.mean(jnp.abs(z - z_goal))
    if kind == "l2":
        return jnp.mean(jnp.square(z - z_goal))
    if kind == "cosine":
        a = z.reshape(-1) / (jnp.linalg.norm(z.reshape(-1)) + 1e-8)
        b = z_goal.reshape(-1) / (jnp.linalg.norm(z_goal.reshape(-1)) + 1e-8)
        return 1.0 - jnp.sum(a * b)
    raise ValueError(f"unknown distance {kind!r}")


def goal_cost(
    z_goal: Array,
    *,
    kind: Distance = "l2",
    horizon: int | None = None,
    terminal_only: bool = False,
    action_penalty: float = 0.0,
    discount: float = 1.0,
) -> CostFn:
    """Drive the latent state toward ``z_goal``.

    Args:
        z_goal: target latent, same shape as the rollout states.
        kind: distance to use.
        horizon: needed only when ``terminal_only`` is set, to know which step
            is terminal.
        terminal_only: score just the final state. Charging every step instead
            (the default) rewards *reaching* the goal early and staying there,
            which is usually what you want and is much better conditioned.
        action_penalty: weight on ``mean(a ** 2)``, discouraging thrash.
        discount: per-step multiplier; ``< 1`` prefers reaching the goal sooner.
    """
    if terminal_only and horizon is None:
        raise ValueError("terminal_only requires horizon")

    def cost(z: Array, a: Array, t: Array) -> Array:
        distance = latent_distance(z, z_goal, kind)
        if terminal_only:
            distance = jnp.where(t == horizon - 1, distance, 0.0)
        else:
            distance = distance * discount**t
        return distance + action_penalty * jnp.mean(jnp.square(a))

    return cost


def reward_cost(
    reward_fn: Callable[[Array], Array],
    *,
    action_penalty: float = 0.0,
    discount: float = 1.0,
) -> CostFn:
    """Turn a learned latent reward (higher is better) into a cost to minimise."""

    def cost(z: Array, a: Array, t: Array) -> Array:
        return -discount**t * reward_fn(z) + action_penalty * jnp.mean(jnp.square(a))

    return cost


def return_cost(
    reward_fn: Callable[[Array, Array], Array],
    value_fn: Callable[[Array], Array] | None = None,
    *,
    horizon: int,
    discount: float = 0.99,
) -> CostFn:
    """Negated discounted return -- the objective a value-based agent plans on.

    ``-(sum_t gamma^t r(z_t, a_t) + gamma^H V(z_H))``.

    The terminal value is what makes this different from a goal cost, and it is
    the whole reason TD-MPC2 can plan with a horizon of three: the value head
    summarises everything beyond the horizon, so the planner does not have to
    simulate it. Without that term a short-horizon planner is myopic by
    construction -- it cannot prefer a move whose payoff arrives on step four.

    Use with ``cost_on="current"`` (the default for planners built by
    :func:`xwm.families.tdmpc2.planner`): a reward head is trained as
    ``r(z_t, a_t)``, so it must be evaluated at the latent the action was taken
    from.

    Args:
        reward_fn: ``(z, a) -> r``, typically a learned reward head.
        value_fn: ``z -> V``, applied once at the final step. ``None`` drops the
            bootstrap, making the objective purely myopic.
        horizon: planning horizon, needed to know which step is terminal.
        discount: RL discount.
    """

    def cost(z: Array, a: Array, t: Array) -> Array:
        total = -(discount**t) * reward_fn(z, a)
        if value_fn is not None:
            # The terminal latent is the one *after* the last action, so it is
            # not visible here; bootstrap from the last pre-transition latent,
            # discounted one extra step. Exact enough for ranking candidates,
            # and it avoids a second dynamics call per sample.
            terminal = -(discount ** (t + 1)) * value_fn(z)
            total = total + jnp.where(t == horizon - 1, terminal, 0.0)
        return total

    return cost


def sum_costs(*costs: CostFn, weights: tuple[float, ...] | None = None) -> CostFn:
    """Weighted sum of several cost terms."""
    w = (1.0,) * len(costs) if weights is None else weights
    if len(w) != len(costs):
        raise ValueError(f"got {len(costs)} costs but {len(w)} weights")

    def cost(z: Array, a: Array, t: Array) -> Array:
        return sum(wi * ci(z, a, t) for wi, ci in zip(w, costs, strict=True))

    return cost

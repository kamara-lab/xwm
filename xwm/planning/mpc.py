"""Receding-horizon control (MPC) on top of a planner.

Open-loop plans drift: the world model's error compounds over the horizon, so
the tenth imagined step is not worth acting on. Model-predictive control fixes
this by acting only on the first step and replanning from the new observation --
so the plan is always anchored to reality and only ever one step old.

Warm-starting matters as much as replanning. Shifting the previous solution
forward by one step gives the next search a nearly-correct starting point,
which is what makes a few planner iterations enough per control step.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple, Protocol

import jax.numpy as jnp
import jax.random as jr

from ..core.types import Array, LatentDynamics, PRNGKey
from .cost import CostFn
from .sampling import Plan


class Planner(Protocol):
    """Anything with a ``plan`` method: CEM, MPPI, GradientPlanner."""

    def plan(
        self,
        key: PRNGKey,
        dynamics: LatentDynamics,
        z0: Array,
        cost_fn: CostFn,
        *,
        init_mean: Array | None = ...,
    ) -> Plan: ...


class ControlStep(NamedTuple):
    """One closed-loop control step.

    Attributes:
        action: the action to execute now.
        warm_start: shifted plan to seed the next call.
        plan: the full plan, for logging or diagnostics.
    """

    action: Array
    warm_start: Array
    plan: Plan


def shift_plan(mean: Array) -> Array:
    """Advance a plan by one step, repeating the last action at the tail."""
    return jnp.concatenate([mean[1:], mean[-1:]], axis=0)


def control_step(
    key: PRNGKey,
    planner: Planner,
    dynamics: LatentDynamics,
    z: Array,
    cost_fn: CostFn,
    *,
    warm_start: Array | None = None,
) -> ControlStep:
    """Plan from the current latent state and return the first action."""
    plan = planner.plan(key, dynamics, z, cost_fn, init_mean=warm_start)
    return ControlStep(
        action=plan.actions[0],
        warm_start=shift_plan(plan.mean),
        plan=plan,
    )


def run_mpc(
    key: PRNGKey,
    planner: Planner,
    dynamics: LatentDynamics,
    cost_fn: CostFn,
    *,
    encode: Callable[[Array], Array],
    step_env: Callable[[Array, Array], Array],
    observation: Array,
    n_steps: int,
    on_step: Callable[[int, ControlStep], None] | None = None,
) -> tuple[list[Array], list[Array], list[Array]]:
    """Run a closed loop against a real (or simulated) environment.

    Args:
        encode: observation -> latent state, i.e. the world model's encoder.
        step_env: ``(observation, action) -> next_observation``. Deliberately a
            plain callable: xwm does not own your simulator.
        observation: the starting observation.
        n_steps: control steps to execute.
        on_step: called with ``(t, control_step)`` after each plan, before the
            environment advances. The whole :class:`ControlStep` is passed, so
            a logger sees the plan and not only the action taken from it.

    Returns:
        ``(observations, actions, costs)`` -- the realised trajectory. This is a
        Python loop because the environment step is outside JAX; each planning
        call is still a single jitted device call.
    """
    observations, actions, costs = [observation], [], []
    warm_start = None
    for t in range(n_steps):
        z = encode(observation)
        step = control_step(
            jr.fold_in(key, t), planner, dynamics, z, cost_fn, warm_start=warm_start
        )
        if on_step is not None:
            on_step(t, step)
        observation = step_env(observation, step.action)
        observations.append(observation)
        actions.append(step.action)
        costs.append(step.plan.cost)
        warm_start = step.warm_start
    return observations, actions, costs

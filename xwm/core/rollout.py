"""Latent rollouts.

A predictive world model is useful exactly to the extent that you can *roll it
forward*: given a starting latent and a sequence of actions, produce the
sequence of imagined latents without ever decoding to pixels. Everything here
is written with :func:`jax.lax.scan` so rollouts stay cheap under ``jit`` and
differentiable for gradient-based planning.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import jax.random as jr

from .types import Array, LatentDynamics, PRNGKey


def rollout(
    dynamics: LatentDynamics,
    z0: Array,
    actions: Array,
    *,
    key: PRNGKey | None = None,
) -> Array:
    """Roll ``dynamics`` forward over an action sequence.

    Args:
        dynamics: one-step latent model ``(z, a) -> z'``.
        z0: initial latent, any shape (typically ``(N, D)`` tokens).
        actions: ``(H, A)`` action sequence.
        key: optional RNG for stochastic dynamics.

    Returns:
        ``(H, *z0.shape)`` -- the latent *after* each action. ``z0`` itself is
        not included, so ``out[t]`` is the state reached by ``actions[:t + 1]``.
    """
    horizon = actions.shape[0]
    keys = jr.split(key, horizon) if key is not None else jnp.zeros((horizon, 2), jnp.uint32)

    def step(z, inputs):
        a, k = inputs
        z_next = dynamics(z, a) if key is None else dynamics(z, a, key=k)
        return z_next, z_next

    _, traj = jax.lax.scan(step, z0, (actions, keys))
    return traj


def rollout_cost(
    dynamics: LatentDynamics,
    z0: Array,
    actions: Array,
    cost_fn: Callable[[Array, Array, int], Array],
    *,
    key: PRNGKey | None = None,
    cost_on: str = "next",
) -> Array:
    """Accumulate ``cost_fn(z, a, t)`` along a rollout.

    Fused with the rollout so planners never materialise the whole trajectory,
    which matters when sampling thousands of candidate action sequences.

    Args:
        cost_on: which latent the cost sees.

            * ``"next"`` -- the state the action led to. Right for a goal cost:
              "how close did this action get me?"
            * ``"current"`` -- the state the action was taken *from*. Right for a
              learned reward head, which is trained as ``r(z_t, a_t)``; scoring
              it at ``z_{t+1}`` would evaluate the model off-distribution by one
              step and quietly bias every plan.
    """
    if cost_on not in ("next", "current"):
        raise ValueError(f"cost_on must be 'next' or 'current', got {cost_on!r}")
    horizon = actions.shape[0]
    keys = jr.split(key, horizon) if key is not None else jnp.zeros((horizon, 2), jnp.uint32)
    on_current = cost_on == "current"

    def step(carry, inputs):
        z, total, t = carry
        a, k = inputs
        cost = cost_fn(z, a, t) if on_current else jnp.zeros(())
        z_next = dynamics(z, a) if key is None else dynamics(z, a, key=k)
        if not on_current:
            cost = cost_fn(z_next, a, t)
        return (z_next, total + cost, t + 1), None

    (_, total, _), _ = jax.lax.scan(step, (z0, jnp.zeros(()), 0), (actions, keys))
    return total


def teacher_forced_rollout(
    dynamics: LatentDynamics,
    latents: Array,
    actions: Array,
    *,
    key: PRNGKey | None = None,
) -> Array:
    """One-step predictions from *ground-truth* latents (teacher forcing).

    Args:
        latents: ``(T, ...)`` encoded observations.
        actions: ``(T - 1, A)`` actions, ``actions[t]`` joining ``t`` to ``t+1``.

    Returns:
        ``(T - 1, ...)`` predictions of ``latents[1:]``.
    """
    n = actions.shape[0]
    keys = jr.split(key, n) if key is not None else jnp.zeros((n, 2), jnp.uint32)

    def one(z, a, k):
        return dynamics(z, a) if key is None else dynamics(z, a, key=k)

    return jax.vmap(one)(latents[:n], actions, keys)

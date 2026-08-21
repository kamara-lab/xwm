"""Exponential-moving-average targets.

Teacher/student JEPAs (I-JEPA, V-JEPA, V-JEPA 2) build their prediction
targets with an encoder whose weights are an EMA of the student's. The EMA is
*not* an optimizer state: it is a plain tree operation applied after each step,
and gradients never flow through it.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from .types import Array, PyTree


def ema_init(model: PyTree) -> PyTree:
    """Create the initial target as a detached copy of ``model``."""
    return jax.tree_util.tree_map(
        lambda x: jax.lax.stop_gradient(x) if eqx.is_inexact_array(x) else x, model
    )


def ema_update(target: PyTree, model: PyTree, momentum: Array | float) -> PyTree:
    """``target <- m * target + (1 - m) * model`` over inexact arrays.

    Non-array leaves (ints, bools, static config) are taken from ``model`` so
    the target never drifts out of structural sync with the student.
    """
    m = jnp.asarray(momentum)

    def step(t, s):
        if eqx.is_inexact_array(t) and eqx.is_inexact_array(s):
            return jax.lax.stop_gradient(m * t + (1.0 - m) * s)
        return s

    return jax.tree_util.tree_map(step, target, model)


def stop_gradient(tree: PyTree) -> PyTree:
    """Detach every array leaf of ``tree``."""
    return jax.tree_util.tree_map(
        lambda x: jax.lax.stop_gradient(x) if eqx.is_inexact_array(x) else x, tree
    )

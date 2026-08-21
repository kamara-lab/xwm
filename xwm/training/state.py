"""Training state."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp

from ..core.module import WorldModel
from ..core.types import Array, PyTree


class TrainState(eqx.Module):
    """Everything needed to resume training.

    Attributes:
        model: the world model.
        target: EMA teacher, or ``None`` for models that don't use one.
        opt_state: optimizer state, covering only trainable parameters.
        step: steps completed.
    """

    model: WorldModel
    target: WorldModel | None
    opt_state: PyTree
    step: Array

    def __init__(
        self,
        model: WorldModel,
        target: WorldModel | None,
        opt_state: PyTree,
        step: Array | int = 0,
    ):
        self.model = model
        self.target = target
        self.opt_state = opt_state
        self.step = jnp.asarray(step, jnp.int32)

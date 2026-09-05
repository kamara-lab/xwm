"""Core abstractions: types, base modules, EMA targets, latent rollouts."""

from .ema import ema_init, ema_update, stop_gradient
from .module import Module, WorldModel, batched_apply, vmap_apply
from .random import (
    DEFAULT_SEED,
    KeySource,
    default_key,
    key_source,
    resolve_key,
    seed,
    set_seed,
    split,
)
from .rollout import rollout, rollout_cost, teacher_forced_rollout
from .types import (
    Array,
    Batch,
    Encoder,
    Frame,
    LatentDynamics,
    Metrics,
    Objective,
    Plannable,
    Predictor,
    PRNGKey,
    PyTree,
)

__all__ = [
    "DEFAULT_SEED",
    "Array",
    "Batch",
    "Encoder",
    "Frame",
    "KeySource",
    "LatentDynamics",
    "Metrics",
    "Module",
    "Objective",
    "PRNGKey",
    "Plannable",
    "Predictor",
    "PyTree",
    "WorldModel",
    "batched_apply",
    "default_key",
    "ema_init",
    "key_source",
    "ema_update",
    "resolve_key",
    "rollout",
    "rollout_cost",
    "seed",
    "set_seed",
    "split",
    "stop_gradient",
    "teacher_forced_rollout",
    "vmap_apply",
]

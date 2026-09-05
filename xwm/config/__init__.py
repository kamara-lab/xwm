"""Experiment configuration: frozen dataclasses, TOML files, dotted overrides.

>>> import xwm
>>> cfg = xwm.config.load("configs/pusht/smoke.toml", overrides=["train.steps=500"])
>>> cfg.train.steps
500

There is no Hydra here and no interpolation language. A config is a tree of
dataclasses with defaults, a file says only what it changes, and an override
that names a field the schema does not have is an error rather than a silently
ignored line.
"""

from __future__ import annotations

from .loader import apply_overrides, coerce, config_hash, from_dict, load, save, to_dict
from .schema import EvalConfig, ExperimentConfig, ModelConfig, TaskConfig, TrainConfig

__all__ = [
    "EvalConfig",
    "ExperimentConfig",
    "ModelConfig",
    "TaskConfig",
    "TrainConfig",
    "apply_overrides",
    "coerce",
    "config_hash",
    "from_dict",
    "load",
    "save",
    "to_dict",
]

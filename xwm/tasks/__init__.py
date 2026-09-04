"""Benchmark tasks: a dataset, an environment, and a metric, bound together.

A *task* is what makes two world models comparable. It fixes the things a
result depends on and that a paper usually mentions only in passing: how many
raw environment steps one model action covers, what resolution the encoder
sees, how far apart a start and its goal are, how long the planner gets, and
what counts as arriving. :class:`~xwm.tasks.TaskSpec` is where those are
written down, once, so training and evaluation cannot disagree about them.

>>> import xwm
>>> xwm.tasks.available()
['maze/synthetic', 'pusht/synthetic']
>>> task = xwm.tasks.create("pusht/synthetic")
>>> env = task.make_env()

The registry follows the shape :data:`xwm.families.registry.REGISTRY` and
:data:`xwm.datasets.DATASETS` already use: slash-namespaced names,
:func:`available`, :func:`describe` (which touches nothing heavy) and
:func:`create` (which may).

References
----------
Zhou et al., *DINO-WM: World Models on Pre-trained Visual Features enable
Zero-shot Planning*, 2024. arXiv:2411.04983 -- the goal-reaching evaluation
these tasks are shaped for.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..envs.protocol import Env
from .data import (
    cached_episodes,
    denormalize_action,
    frameskip_episode,
    normalize_action,
    preprocess_episode,
    split_episodes,
    unblock,
)
from .metrics import METRICS, sliced_metric
from .spec import TaskSpec
from .synthetic import MAZE_SYNTHETIC, PUSH_SYNTHETIC, generated_episodes, maze_env, push_env
from .task import Task

__all__ = [
    "METRICS",
    "TASKS",
    "Task",
    "TaskSpec",
    "available",
    "cached_episodes",
    "create",
    "denormalize_action",
    "describe",
    "frameskip_episode",
    "normalize_action",
    "preprocess_episode",
    "sliced_metric",
    "split_episodes",
    "unblock",
]

#: name -> what the task is. Every entry is a plain dataclass, so listing and
#: describing tasks costs no imports beyond this module.
TASKS: dict[str, TaskSpec] = {
    PUSH_SYNTHETIC.name: PUSH_SYNTHETIC,
    MAZE_SYNTHETIC.name: MAZE_SYNTHETIC,
}

#: env key -> builder. Lazily imported inside each builder, so a task that needs
#: a simulator costs nothing until someone calls :meth:`Task.make_env`.
ENV_BUILDERS: dict[str, Callable[..., Env]] = {
    "pusht/synthetic": push_env,
    "maze/synthetic": maze_env,
}

#: env key -> a generator of episodes, for tasks whose data is produced rather
#: than recorded. A task with a ``dataset`` uses :mod:`xwm.datasets` instead.
EPISODE_SOURCES: dict[str, Callable[[TaskSpec], Any]] = {
    "pusht/synthetic": generated_episodes,
    "maze/synthetic": generated_episodes,
}


def available() -> list[str]:
    """Registered task names."""
    return sorted(TASKS)


def describe(name: str) -> TaskSpec:
    """What a task is, without building anything or downloading anything."""
    if name not in TASKS:
        raise KeyError(f"unknown task {name!r}; available: {available()}")
    return TASKS[name]


def create(name: str, **overrides: Any) -> Task:
    """Build a task by name.

    ``overrides`` replace fields on its :class:`TaskSpec` -- ``image_size=64``,
    ``episodes=4``, ``frameskip=2`` -- which is how a config file or a CLI
    override reaches the task without a second copy of the defaults.
    """
    import dataclasses

    spec = describe(name)
    if overrides:
        known = {f.name for f in dataclasses.fields(spec)}
        unknown = sorted(set(overrides) - known)
        if unknown:
            raise TypeError(f"TaskSpec has no field(s) {unknown}; valid: {sorted(known)}")
        spec = dataclasses.replace(spec, **overrides)
    if spec.env not in ENV_BUILDERS:
        raise KeyError(f"task {name!r} names an unknown environment {spec.env!r}")
    if spec.metric not in METRICS:
        raise KeyError(f"task {name!r} names an unknown metric {spec.metric!r}")
    source = EPISODE_SOURCES.get(spec.env)
    return Task(
        spec,
        env_builder=ENV_BUILDERS[spec.env],
        metric=METRICS[spec.metric],
        episode_source=source,
    )

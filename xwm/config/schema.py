"""An experiment, as data.

Frozen dataclasses, like :class:`xwm.envs.FrankaConfig` and
:class:`xwm.tasks.TaskSpec` -- no Hydra, no OmegaConf, no registry of
interpolations. A config file is read with :mod:`tomllib` from the standard
library and lands in this tree; every field has a default, so a file says only
what it changes.

The one place this is deliberately untyped is :attr:`ModelConfig.kwargs`. Every
constructor in :data:`xwm.families.registry.REGISTRY` is keyword-only and their
signatures genuinely differ, so typing them here would mean maintaining a
seventh copy of seven signatures. Instead the kwargs go straight to
:func:`xwm.families.create`, where Python itself validates them -- a typo is a
``TypeError`` naming the bad argument, at build time, before anything trains.

What a config must **not** contain is anything the task already fixes:
``action_dim``, ``img_size`` and ``in_channels`` are injected from the
:class:`~xwm.tasks.TaskSpec`, so a model cannot be built that disagrees with
the data it is about to be trained on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..bench.protocol import PlanConfig

__all__ = [
    "EvalConfig",
    "ExperimentConfig",
    "LogConfig",
    "ModelConfig",
    "TaskConfig",
    "TrainConfig",
]


@dataclass(frozen=True)
class ModelConfig:
    """Which model, and how it is built.

    Attributes:
        name: a key in :data:`xwm.families.registry.REGISTRY`.
        kwargs: passed to :func:`xwm.families.create`. Shape arguments the task
            determines are injected and must not appear here.
        encoder_checkpoint: a stage-one encoder to build on, for the two-stage
            recipe -- pretrain a representation, freeze it, learn dynamics.
        checkpoint: weights to resume from or evaluate.
    """

    name: str = "jepa/action"
    kwargs: dict[str, Any] = field(default_factory=dict)
    encoder_checkpoint: str | None = None
    checkpoint: str | None = None


@dataclass(frozen=True)
class TaskConfig:
    """Which task, and any deviation from its registered defaults."""

    name: str = "pusht/synthetic"
    overrides: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainConfig:
    """The optimisation. Defaults follow the LeJEPA-family world models."""

    steps: int = 10_000
    batch_size: int = 64
    learning_rate: float = 5e-5
    weight_decay: float = 1e-3
    warmup: float = 0.05
    grad_clip: float = 1.0
    ema_momentum: float = 0.996
    clip_length: int | None = None
    stride: int = 1
    log_every: int = 50
    checkpoint_every: int = 0
    limit_episodes: int | None = None


@dataclass(frozen=True)
class EvalConfig:
    """The measurement.

    ``policy`` defaults to all four because a success rate without its floors
    is unreadable -- and because ``replay`` is the gate that says whether the
    other three mean anything.
    """

    episodes: int | None = None
    budget: int | None = None
    goal_offset: int | None = None
    policy: tuple[str, ...] = ("planner", "noop", "random", "replay")
    plan: PlanConfig = field(default_factory=PlanConfig)
    video: bool = False


@dataclass(frozen=True)
class LogConfig:
    """Where a run reports itself, beyond the files it always writes.

    Excluded from :func:`xwm.config.config_hash`, unlike every other section:
    logging cannot change a result, so two runs that differ only in whether
    they were watched are the same experiment and must land in the same
    directory name.

    Attributes:
        rerun: write Rerun recordings into the run directory -- ``train.rrd``
            from :mod:`xwm.cli.train`, ``eval.rrd`` from :mod:`xwm.cli.evaluate`.
        spawn: also open the viewer as a child process. For a laptop.
        connect: also stream to an already-running viewer at this gRPC address,
            e.g. ``"rerun+http://127.0.0.1:9876/proxy"``. For watching a remote
            run live while it still writes its own file.
        every: log one in every ``every`` training callbacks. The callback
            already fires only on ``train.log_every``, so this thins further.
    """

    rerun: bool = False
    spawn: bool = False
    connect: str = ""
    every: int = 1


@dataclass(frozen=True)
class ExperimentConfig:
    """One run: a model, a task, how to train it, and how to measure it."""

    model: ModelConfig = field(default_factory=ModelConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    log: LogConfig = field(default_factory=LogConfig)
    seed: int = 0
    output_dir: str = "runs"
    name: str = ""

    def run_name(self, digest: str) -> str:
        """Directory name for this run: readable, then unambiguous."""
        stem = self.name or (
            f"{self.task.name.replace('/', '-')}-{self.model.name.replace('/', '-')}-s{self.seed}"
        )
        return f"{stem}-{digest}"

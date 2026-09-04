"""What a benchmark task *is*: a dataset, an environment, and how to score them.

A :class:`TaskSpec` is the single place a task's shape is written down. That
matters most for the parameters that appear in two places at once and are
catastrophic when they disagree:

* ``frameskip`` decides both how training clips are subsampled and how many raw
  actions the planner's each output covers. Written once here, the two cannot
  drift.
* ``image_size`` and ``native_size`` decide how a recorded frame and a rendered
  one are resized. Both go through the same :func:`xwm.datasets.spec.to_frames`
  call, so parity is structural rather than a convention someone has to keep.
* ``goal_offset`` and ``eval_budget`` decide the difficulty of the evaluation.
  Reporting a success rate without them is reporting nothing.

Flat scalars only, like :class:`xwm.envs.FrankaConfig` and
:class:`xwm.datasets.DatasetSpec`.

References
----------
Zhou et al., *DINO-WM*, 2024. arXiv:2411.04983 -- the goal-reaching protocol
these defaults follow.

Chi et al., *Diffusion Policy*, RSS 2023. arXiv:2303.04137 -- Push-T.

Park et al., *OGBench*, ICLR 2025. arXiv:2410.20092 -- the cube tasks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = ["TaskSpec"]


@dataclass(frozen=True)
class TaskSpec:
    """A task: where its data comes from, what simulates it, and what counts as success.

    Attributes:
        name: slash-namespaced, ``"pusht/lerobot"``.
        env: key into :data:`xwm.tasks.ENV_BUILDERS`.
        dataset: an :mod:`xwm.datasets` name, or ``None`` for an environment-only
            task whose data is generated rather than recorded.
        action_dim: width of one **raw** action, before frameskip grouping.
        action_low, action_high: raw-unit bounds, used to normalise to
            ``[-1, 1]``. Declared rather than measured from the data, so a
            checkpoint's action space does not depend on which split trained it.
        observation: which frame fields the model sees.
        native_size: ``(H, W)`` the recordings and the simulator both use.
        image_size: what the model sees; ``None`` keeps ``native_size``.
        frameskip: how many raw steps one model step covers. Actions are
            *grouped* into a ``frameskip * action_dim`` block, not repeated --
            repeating would throw away four fifths of the recorded control
            signal at the Push-T default.
        history: frames of context the model conditions on.
        horizon, receding_horizon: planning horizon and how much of each plan is
            executed before replanning, both in blocked steps.
        goal_offset: how many raw steps separate an episode's start frame from
            its goal frame. The task's difficulty knob.
        eval_budget: raw steps allowed per evaluation episode.
        episodes: evaluation episodes.
        holdout, split_seed: the held-out fraction, split by episode (never by
            clip: clips overlap, so a clip-level split leaks).
        metric: key into :data:`xwm.tasks.METRICS`, giving ``(distance, success)``.
    """

    name: str
    summary: str
    env: str
    env_options: dict[str, Any] = field(default_factory=dict)
    dataset: str | None = None
    dataset_options: dict[str, Any] = field(default_factory=dict)

    action_dim: int = 2
    action_low: tuple[float, ...] | float = -1.0
    action_high: tuple[float, ...] | float = 1.0
    observation: tuple[str, ...] = ("video",)
    native_size: tuple[int, int] = (96, 96)
    image_size: int | tuple[int, int] | None = None
    state_dim: int = 0

    frameskip: int = 1
    history: int = 1
    horizon: int = 5
    receding_horizon: int = 5
    goal_offset: int = 25
    eval_budget: int = 50
    episodes: int = 50
    holdout: float = 0.1
    split_seed: int = 0

    metric: str = "euclidean"
    metric_options: dict[str, Any] = field(default_factory=dict)
    citation: str = ""

    def __post_init__(self):
        if self.frameskip < 1:
            raise ValueError(f"frameskip must be at least 1, got {self.frameskip}")
        if self.history < 1:
            raise ValueError(f"history must be at least 1, got {self.history}")
        if self.receding_horizon > self.horizon:
            raise ValueError(
                f"receding_horizon {self.receding_horizon} exceeds horizon {self.horizon}: "
                "a plan cannot execute more steps than it planned"
            )
        if not 0.0 < self.holdout < 1.0:
            raise ValueError(f"holdout must be in (0, 1), got {self.holdout}")

    @property
    def blocked_action_dim(self) -> int:
        """Width of one model action: ``action_dim * frameskip``.

        The number to check when a model trains cleanly and plans badly.
        """
        return self.action_dim * self.frameskip

    @property
    def resize(self) -> int | tuple[int, int] | None:
        """What to pass :func:`xwm.datasets.spec.to_frames`; ``None`` when native."""
        if self.image_size is None:
            return None
        wanted = (
            (self.image_size, self.image_size)
            if isinstance(self.image_size, int)
            else tuple(self.image_size)
        )
        return None if wanted == tuple(self.native_size) else self.image_size

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """``(low, high)`` as ``(action_dim,)`` arrays, whatever form they were given in."""
        low = np.broadcast_to(np.asarray(self.action_low, np.float32), (self.action_dim,))
        high = np.broadcast_to(np.asarray(self.action_high, np.float32), (self.action_dim,))
        if np.any(high <= low):
            raise ValueError(f"action_high must exceed action_low, got {low} and {high}")
        return np.array(low), np.array(high)

    @property
    def clip_length(self) -> int:
        """Frames per training clip: ``history`` of context plus one prediction."""
        return self.history + 1

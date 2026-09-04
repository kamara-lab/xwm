"""What an evaluation *is*, before any model is involved.

The episodes a model is judged on are drawn here, from the held-out split,
with a seed and no reference to the model. Two consequences, both deliberate:

* Two models are evaluated on **literally the same instances**, so their
  numbers are paired and a difference between them is not a difference in which
  problems they were given.
* The instance list is written into the results file, so a run is reproducible
  from its own output rather than from a remembered command line.

Every start is a state some recorded episode actually visited, and every goal
is a state that same episode reached ``goal_offset`` steps later. The task is
therefore known to be solvable -- by the demonstrator, in exactly the budget
allowed -- which is what the ``replay`` baseline in :mod:`xwm.bench.policies`
measures and what a randomly sampled goal cannot promise.

References
----------
Zhou et al., *DINO-WM*, 2024. arXiv:2411.04983.

Maes et al., *stable-worldmodel*, 2026. arXiv:2605.21800 -- the dataset-driven
evaluation mode and the step-budget convention this follows.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = ["SCHEMA", "Episode", "PlanConfig", "sample_episodes"]

#: Version of the results layout in :mod:`xwm.bench.report`. Bump on a breaking
#: change, so a tool reading an old file can say so instead of misreading it.
SCHEMA = "xwm.bench.goal_reaching/1"


@dataclass(frozen=True)
class PlanConfig:
    """How the planner is run inside the control loop.

    Attributes:
        horizon: planning horizon, in **blocked** steps -- one blocked step is
            ``frameskip`` environment steps.
        receding_horizon: how much of each plan is executed before replanning.
            Equal to ``horizon`` means open-loop within a plan; ``1`` replans
            every step and is the most expensive and most accurate.
        warm_start: seed each search with the previous plan, shifted. Cheap, and
            it is what makes a handful of planner iterations enough.
        planner: ``cem``, ``mppi``, ``gradient`` or ``mcts``.
        objective: ``goal`` compares latents to an encoded goal frame -- every
            model supports it. ``return`` uses a learned reward and value, so it
            needs a family that has them and a task with rewards.
        distance: which latent distance a goal objective uses.
    """

    horizon: int = 5
    receding_horizon: int = 5
    warm_start: bool = True
    planner: str = "cem"
    planner_options: dict[str, Any] = field(default_factory=dict)
    objective: str = "goal"
    distance: str = "l2"
    action_penalty: float = 0.0
    discount: float = 1.0

    def __post_init__(self):
        if self.receding_horizon < 1 or self.receding_horizon > self.horizon:
            raise ValueError(
                f"receding_horizon must be in [1, horizon={self.horizon}], "
                f"got {self.receding_horizon}"
            )


@dataclass(frozen=True)
class Episode:
    """One evaluation instance: where to start, what to reach, and with which seed.

    ``start`` and ``goal`` index the **frameskipped** episode, since that is
    what the model steps through; ``seed`` is per-instance so a stochastic
    environment is reproducible instance by instance rather than only as a whole.
    """

    index: int
    start: int
    goal: int
    seed: int


def sample_episodes(
    episodes: list[dict[str, np.ndarray]],
    *,
    n: int,
    goal_offset: int,
    seed: int = 0,
    frameskip: int = 1,
    distance: Callable[[np.ndarray, np.ndarray], float] | None = None,
    min_distance: float = 0.0,
) -> list[Episode]:
    """Draw ``n`` (episode, start, goal) instances from a held-out split.

    Args:
        episodes: the held-out episodes, already frameskipped.
        n: how many instances. Drawn with replacement only if the split cannot
            supply that many distinct ones, which is reported rather than hidden.
        goal_offset: how far ahead the goal sits, in **raw** environment steps.
        frameskip: raw steps per recorded frame, to convert that offset.
        distance: ground-truth distance between two recorded states, used with
            ``min_distance`` to reject instances where nothing moved.
        min_distance: discard candidate instances whose goal is closer to their
            start than this. Set it to the success threshold and an instance
            that ``noop`` would solve cannot be drawn.

            This matters more than it sounds. A recording in which the
            demonstrator idles -- or, in a contact task, never touches the
            object -- contributes windows whose start and goal are identical.
            Sample enough of them and a planner that does nothing scores well,
            and the evaluation is measuring the sampler.

    Raises:
        ValueError: if no episode is long enough to contain one instance, or if
            the filter leaves nothing. A silently empty or silently trivial
            evaluation is the failure mode worth ruling out.
    """
    if n < 1:
        raise ValueError(f"n must be positive, got {n}")
    offset = max(1, goal_offset // max(frameskip, 1))
    candidates = [
        (i, start)
        for i, episode in enumerate(episodes)
        for start in range(_length(episode) - offset)
    ]
    reachable = len(candidates)
    if distance is not None and min_distance > 0:
        if any("state" not in episodes[i] for i, _ in candidates):
            raise ValueError(
                "filtering trivial instances needs a recorded 'state' per frame, and "
                "these episodes have none. Either record one, or set the task's "
                "metric_options threshold to 0 to draw instances unfiltered -- at the "
                "cost of drawing some the `noop` baseline already solves."
            )
        candidates = [
            (i, start)
            for i, start in candidates
            if distance(episodes[i]["state"][start], episodes[i]["state"][start + offset])
            >= min_distance
        ]
        if not candidates:
            raise ValueError(
                f"none of {reachable} candidate instances move further than "
                f"{min_distance}: every start is already at its goal, so the "
                "evaluation would measure nothing. Increase goal_offset, or "
                "collect data in which the task is actually performed."
            )
    if not candidates:
        longest = max((_length(e) for e in episodes), default=0)
        raise ValueError(
            f"no episode is longer than the goal offset: the longest has {longest} frames "
            f"and the goal sits {offset} frames ahead (goal_offset={goal_offset} raw steps "
            f"at frameskip {frameskip}). Shorten goal_offset or record longer episodes."
        )
    rng = np.random.default_rng(seed)
    picked = rng.choice(len(candidates), size=n, replace=n > len(candidates))
    return [
        Episode(
            index=candidates[c][0],
            start=candidates[c][1],
            goal=candidates[c][1] + offset,
            seed=int(seed * 1000 + k),
        )
        for k, c in enumerate(picked)
    ]


def _length(episode: dict[str, np.ndarray]) -> int:
    for field_name in ("video", "image", "state", "position"):
        if field_name in episode:
            return int(episode[field_name].shape[0])
    raise ValueError(f"episode has no frame field; got {sorted(episode)}")

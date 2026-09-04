"""What counts as having reached the goal.

Every metric is a pair: a distance in the environment's own units, and a
predicate. Both take two *task states* -- never observations, never latents --
because a success criterion computed from the thing the model is optimising is
not a measurement of anything.

The distinction that matters is **which part of the state**. In a pushing task
the goal is a puck position and the pusher's own position is irrelevant: a
planner that parks its pusher exactly where the demonstrator's was, with the
puck untouched, has done nothing. So the default metrics compare a named slice
rather than the whole vector, and the slice is part of the task definition.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

__all__ = ["METRICS", "sliced_metric"]

Metric = tuple[Callable[..., float], Callable[..., bool]]


def sliced_metric(start: int, stop: int) -> Metric:
    """Euclidean distance over ``state[start:stop]``, with a threshold predicate."""

    def distance(state: np.ndarray, goal: np.ndarray, *, threshold: float = 0.0) -> float:
        del threshold  # part of the predicate, not the distance
        a = np.asarray(state, np.float64)[start:stop]
        b = np.asarray(goal, np.float64)[start:stop]
        return float(np.linalg.norm(a - b))

    def success(state: np.ndarray, goal: np.ndarray, *, threshold: float) -> bool:
        return distance(state, goal) < threshold

    return distance, success


def _euclidean() -> Metric:
    def distance(state: np.ndarray, goal: np.ndarray, *, threshold: float = 0.0) -> float:
        del threshold
        return float(np.linalg.norm(np.asarray(state, np.float64) - np.asarray(goal, np.float64)))

    def success(state: np.ndarray, goal: np.ndarray, *, threshold: float) -> bool:
        return distance(state, goal) < threshold

    return distance, success


#: name -> (distance, success). Slices index the *environment* state, which for
#: the JAX worlds is their NamedTuple flattened in field order.
METRICS: dict[str, Metric] = {
    "euclidean": _euclidean(),
    # PushState(pusher, puck, goal): the puck is the thing being moved.
    "puck": sliced_metric(2, 4),
    # MazeState(pos, vel, goal): the agent's position, not its velocity.
    "agent": sliced_metric(0, 2),
}

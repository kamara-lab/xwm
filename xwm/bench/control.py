"""One evaluation episode: reset to a recorded state, plan, act, measure.

Deliberately not built on :func:`xwm.planning.run_mpc`. That function is the
short, readable demonstration of closed-loop control and it is used by the
examples and the planning guide; it executes the first action of each plan,
takes a plain ``step_env`` callable, and knows nothing about frameskip, history
windows, receding horizons or success criteria. Bending it into both shapes
would leave a function with four mutually exclusive argument sets. This module
instead composes the same primitives -- :func:`xwm.planning.control_step`'s
warm-start logic via :mod:`xwm.bench.policies` -- into the loop the protocol
needs, and leaves ``run_mpc`` alone.

Two things the loop does that are easy to get wrong:

**It counts raw environment steps, not model steps.** One planned action covers
``frameskip`` of them, and the budget, the distance trace and the
steps-to-success are all in raw steps, because that is the unit a task's
difficulty is stated in.

**It does not stop on success.** Reaching the goal is latched as ``solved_at``
and the episode keeps running to the budget, so ``final_distance`` tells the
truth about a planner that arrives and then wanders off. An environment's own
``terminated`` flag is likewise ignored: it answers a different question (Push-T
fires it on coverage of a *fixed* target) and letting it end the episode would
silently change the metric.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

import jax.numpy as jnp
import jax.random as jr
import numpy as np

from ..core.types import PRNGKey
from .protocol import Episode

__all__ = ["EpisodeResult", "StepInfo", "Context", "run_episode"]


@dataclass
class StepInfo:
    """One raw environment step, as handed to ``run_episode``'s ``on_step``.

    Everything the loop knows at that instant, so an observer need not
    reconstruct any of it. ``action`` is the *planned* action, which covers
    ``frameskip`` raw steps, and ``raw`` is the one actually executed now.
    """

    step: int
    action: np.ndarray
    raw: np.ndarray
    state: np.ndarray
    goal_state: np.ndarray
    distance: float
    success: bool
    frame: np.ndarray | None = None
    plan: object | None = None


@dataclass
class EpisodeResult:
    """What one instance produced. Distances are in the environment's own units."""

    success: bool
    steps: int
    initial_distance: float
    final_distance: float
    best_distance: float
    solved_at: int | None = None
    plan_cost: float | None = None
    distances: list[float] = field(default_factory=list)
    frames: list[np.ndarray] = field(default_factory=list)

    @property
    def distance_closed(self) -> float:
        """Fraction of the starting gap that was closed. ``1`` is arrival.

        Negative when the planner ended further away than it began, which is a
        real and common outcome and is why this is reported rather than success
        alone.
        """
        if self.initial_distance <= 0:
            return 1.0 if self.success else 0.0
        return 1.0 - self.final_distance / self.initial_distance


class Context:
    """The frames and actions seen so far, as the model's ``initial_state`` wants them.

    A model conditioning on ``history`` frames needs ``history`` of them before
    it can act, and at the start of an episode there is one. The missing context
    is filled by repeating the oldest real frame rather than by zeros: a zero
    frame is not an observation of anything, and an encoder trained on real ones
    has no reason to map it anywhere sensible.
    """

    def __init__(self, history: int, action_width: int):
        self.history = history
        self._frames: deque = deque(maxlen=history)
        self._actions: deque = deque(maxlen=max(history - 1, 1))
        self._width = action_width

    def start(self, frame: jnp.ndarray) -> None:
        self._frames.clear()
        self._actions.clear()
        for _ in range(self.history):
            self._frames.append(frame)
        for _ in range(self.history - 1):
            self._actions.append(jnp.zeros((self._width,)))

    def observe(self, frame: jnp.ndarray) -> None:
        self._frames.append(frame)

    def act(self, action: np.ndarray) -> None:
        if self.history > 1:
            self._actions.append(jnp.asarray(action))

    def frames(self) -> jnp.ndarray:
        return jnp.stack(list(self._frames))

    def actions(self) -> jnp.ndarray | None:
        if self.history == 1:
            return None
        return jnp.stack(list(self._actions))


def run_episode(
    task,
    env,
    episode: Episode,
    *,
    policy,
    plan,
    key: PRNGKey,
    budget: int,
    record: bool = False,
    on_step: Callable[[StepInfo], None] | None = None,
) -> EpisodeResult:
    """Run one instance to the budget and measure it.

    Args:
        task: supplies preprocessing, action scaling and the success criterion.
        env: reset to the recorded start state; stepped in raw units.
        episode: which recording, which start frame, which goal frame.
        policy: anything from :mod:`xwm.bench.policies`.
        plan: the :class:`~xwm.bench.protocol.PlanConfig` in force.
        budget: raw environment steps allowed.
        record: keep every rendered frame, for a GIF. Off by default -- it is
            the only part of this loop whose cost grows with the episode.
        on_step: called with a :class:`StepInfo` after every raw environment
            step. Rendering is expensive, so ``StepInfo.frame`` is filled only
            when ``record`` is on; a logger that wants frames should ask for
            both. Never called for the initial state -- an observer that wants
            it has the environment.
    """
    recording = task.episodes(split="val")[episode.index]
    goal_state = task.env_state(recording["state"][episode.goal], env)

    env.reset(seed=episode.seed, state=task.env_state(recording["state"][episode.start], env))
    if hasattr(policy, "set_recording"):
        policy.set_recording(recording["action"][episode.start :])
    policy.reset()

    context = Context(task.spec.history, task.spec.blocked_action_dim)
    context.start(task.observation(env.observe()))

    initial = task.distance(env.state(), goal_state)
    result = EpisodeResult(
        success=False,
        steps=0,
        initial_distance=initial,
        final_distance=initial,
        best_distance=initial,
        distances=[initial],
    )
    if record:
        result.frames.append(env.render())

    steps = 0
    while steps < budget:
        action = policy(context, jr.fold_in(key, steps))
        context.act(action)
        # One planned action is `frameskip` raw actions, executed in order.
        for raw in task.unblock(np.asarray(action, np.float32)):
            env.step(task.denormalize_action(raw))
            steps += 1
            distance = task.distance(env.state(), goal_state)
            result.distances.append(distance)
            result.best_distance = min(result.best_distance, distance)
            if task.success(env.state(), goal_state) and result.solved_at is None:
                result.solved_at = steps
                result.success = True
            frame = env.render() if record else None
            if frame is not None:
                result.frames.append(frame)
            if on_step is not None:
                on_step(
                    StepInfo(
                        step=steps,
                        action=action,
                        raw=raw,
                        state=env.state(),
                        goal_state=goal_state,
                        distance=distance,
                        success=result.success,
                        frame=frame,
                        plan=getattr(policy, "last_plan", None),
                    )
                )
            if steps >= budget:
                break
        context.observe(task.observation(env.observe()))

    result.steps = steps
    result.final_distance = result.distances[-1]
    result.plan_cost = getattr(policy, "last_cost", None)
    return result

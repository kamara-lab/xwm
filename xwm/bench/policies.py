"""What acts during an evaluation episode: the model, and the baselines that frame it.

A success rate on its own is unreadable. Three baselines run through the same
loop, on the same instances, with the same seeds:

``noop``
    Do nothing. Measures how much of the gap the start state already closes --
    a task where the goal is nearly where you begin will flatter every planner.

``random``
    Uniform actions. The floor a search has to beat to be doing any searching.

``replay``
    Execute the demonstrator's own recorded actions from the start frame. This
    one is not a baseline but a **gate**. It should reach the goal essentially
    always, because the goal is by construction where those actions led. When
    it does not, the environment is not being reset to the recorded state
    faithfully -- an unrecorded velocity, a contact that has to be re-settled,
    a state vector that does not determine the simulator -- and *no* model
    number from that task means anything until that is fixed. Run it first.

Every policy has the same shape: given the frames and actions so far, return
one blocked action in ``[-1, 1]``. The planner's version is the only one that
looks at the model.
"""

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
import jax.random as jr
import numpy as np

from ..planning.mpc import shift_plan

__all__ = ["POLICIES", "make_policy"]


class _Policy:
    """Base: stateless unless a subclass says otherwise."""

    needs_model = False

    def reset(self) -> None:
        """Called once per episode, before the first action."""

    def __call__(self, context, key) -> np.ndarray:
        raise NotImplementedError


class NoopPolicy(_Policy):
    """Zero action -- the middle of the normalised range, i.e. no displacement."""

    def __init__(self, task, **_):
        self.width = task.spec.blocked_action_dim

    def __call__(self, context, key) -> np.ndarray:
        return np.zeros(self.width, np.float32)


class RandomPolicy(_Policy):
    """Uniform in the normalised action box."""

    def __init__(self, task, **_):
        self.width = task.spec.blocked_action_dim

    def __call__(self, context, key) -> np.ndarray:
        return np.asarray(jr.uniform(key, (self.width,), minval=-1.0, maxval=1.0), np.float32)


class ReplayPolicy(_Policy):
    """The demonstrator's recorded actions, from the start frame onward.

    Runs out of recorded actions at the end of the episode and then holds still,
    which is the honest thing to do: the recording ended there.
    """

    def __init__(self, task, **_):
        self.width = task.spec.blocked_action_dim
        self._actions: np.ndarray | None = None
        self._cursor = 0

    def set_recording(self, actions: np.ndarray) -> None:
        self._actions = np.asarray(actions, np.float32)

    def reset(self) -> None:
        self._cursor = 0

    def __call__(self, context, key) -> np.ndarray:
        if self._actions is None or self._cursor >= len(self._actions):
            return np.zeros(self.width, np.float32)
        action = self._actions[self._cursor]
        self._cursor += 1
        return action


class PlannerPolicy(_Policy):
    """Receding-horizon planning through the model's own dynamics.

    Holds the warm start between calls, which is the only state here. The
    latent, the cost and the dynamics all come from the model, so this class
    knows nothing about the model's internals beyond
    :class:`xwm.core.types.Plannable`.
    """

    needs_model = True

    def __init__(self, task, *, model, planner, cost_fn, plan, **_):
        self.task = task
        self.model = model
        self.planner = planner
        self.cost_fn = cost_fn
        self.plan = plan
        self.dynamics = model.dynamics_fn()
        self._warm: jnp.ndarray | None = None
        self._queue: list[np.ndarray] = []
        self.last_cost: float | None = None
        #: The most recent :class:`~xwm.planning.Plan`, kept for diagnostics.
        #: The executed action is one column of it; the proposal mean and its
        #: spread are what say whether the search converged.
        self.last_plan = None

    def reset(self) -> None:
        self._warm = None
        self._queue = []
        self.last_plan = None

    def __call__(self, context, key) -> np.ndarray:
        if not self._queue:
            z = self.model.initial_state(context.frames(), context.actions())
            plan = self.planner.plan(key, self.dynamics, z, self.cost_fn, init_mean=self._warm)
            self.last_cost = float(plan.cost)
            self.last_plan = plan
            if self.plan.warm_start:
                self._warm = shift_plan(plan.mean)
            take = min(self.plan.receding_horizon, self.plan.horizon)
            self._queue = [np.asarray(a, np.float32) for a in plan.actions[:take]]
        return self._queue.pop(0)


#: name -> class. ``planner`` is the model under test; the rest are the floor.
POLICIES: dict[str, type[_Policy]] = {
    "planner": PlannerPolicy,
    "noop": NoopPolicy,
    "random": RandomPolicy,
    "replay": ReplayPolicy,
}


def make_policy(name: str, task, **kwargs) -> _Policy:
    """Build a policy by name, or say which names exist."""
    if name not in POLICIES:
        raise KeyError(f"unknown policy {name!r}; available: {sorted(POLICIES)}")
    return POLICIES[name](task, **kwargs)


#: What a policy is, from the control loop's side.
Policy = Callable[..., np.ndarray]

"""What :mod:`xwm.bench` requires of a simulator.

A structural :class:`~typing.Protocol`, like everything in
:mod:`xwm.core.types`: nothing subclasses it, and a class satisfies it by
having the right methods. That matters here more than elsewhere, because the
things being unified are a Gymnasium environment written by someone else, a
pure-JAX world in :mod:`xwm.data`, and a Newton scene -- three codebases with
no common ancestor and no prospect of one.

Two decisions are worth stating, because they are what the evaluation protocol
rests on.

**An environment emits the same field vocabulary a dataset does.** A
:data:`~xwm.core.types.Frame` is a dict keyed by
:data:`xwm.datasets.spec.FRAME_FIELDS` -- ``image``, ``state``, ``position`` --
so the function that turns a recorded frame into a model input is the *same
function*, called with the same arguments, that turns a live observation into
one. Preprocessing parity between training and evaluation is then a structural
property rather than a discipline. The classic version of this bug is an
encoder trained on 96x96 recordings and evaluated on natively-rendered 224x224
frames: sharper, differently anti-aliased, and out of distribution in a way no
error message mentions.

**Resetting to a recorded state is part of the protocol.**
:meth:`Env.reset` takes a ``state=``, because the headline evaluation starts
each episode from a state some recorded episode actually visited and asks the
planner to reach one it reached later. That guarantees the task is solvable,
which random goals do not. The invariant the conformance test checks is

    ``env.reset(state=env.state())`` reproduces ``env.state()``

and it is not free: a simulator carries velocities, contacts and solver state
that a state vector may not name. Where the round trip is inexact, the
``replay`` baseline in :mod:`xwm.bench.policies` is what measures the damage --
run it before reading any model's number.

References
----------
Zhou et al., *DINO-WM*, 2024. arXiv:2411.04983 -- goal-reaching from recorded
start and goal frames, the protocol these types serve.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from ..core.types import Frame

__all__ = ["Env", "GoalEnv"]


@runtime_checkable
class Env(Protocol):
    """A steppable simulator. NumPy in, NumPy out, outside the ``jit`` boundary.

    Attributes:
        action_dim: width of one *raw* action, before any frameskip grouping.
        action_low, action_high: ``(action_dim,)`` bounds in the simulator's own
            units. :class:`xwm.tasks.Task` normalises to ``[-1, 1]`` from these
            declared bounds rather than from dataset statistics, so a checkpoint
            means the same thing whatever data it was trained on.
        native_size: ``(height, width)`` the simulator renders observations at.
            Match the recording; resize once, in one place.
        state_dim: width of :meth:`state`.
    """

    action_dim: int
    action_low: np.ndarray
    action_high: np.ndarray
    native_size: tuple[int, int]
    state_dim: int

    def reset(self, *, seed: int | None = ..., state: np.ndarray | None = ...) -> Frame:
        """Start an episode, optionally at a given :meth:`state`."""
        ...

    def step(self, action: np.ndarray) -> Frame:
        """Advance one step under a raw-unit action."""
        ...

    def observe(self) -> Frame:
        """The current observation, without stepping."""
        ...

    def state(self) -> np.ndarray:
        """``(state_dim,)`` canonical task state -- what :meth:`reset` accepts back."""
        ...

    def render(self, size: int | None = ...) -> np.ndarray:
        """``(3, H, W)`` float32 in ``[0, 1]``, for figures rather than for training."""
        ...

    def close(self) -> None:
        """Release the simulator. Idempotent."""
        ...


@runtime_checkable
class GoalEnv(Env, Protocol):
    """An environment that owns its goals rather than taking them from a dataset.

    The dataset-driven protocol supplies goals itself, so most environments do
    not need this. OGBench does: its observation vector cannot be inverted to a
    simulator state, so an episode's start and goal have to come from the
    environment's own task list instead of from a recorded frame.
    """

    def set_goal(self, state: np.ndarray) -> None:
        """Aim at a goal state, for environments that render or score one."""
        ...

    def goal_frame(self) -> Frame:
        """The goal as an observation, for a model that encodes goals from pixels."""
        ...

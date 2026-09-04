"""The pure-JAX worlds of :mod:`xwm.data`, behind the :class:`~xwm.envs.Env` protocol.

:class:`xwm.data.PushWorld` and :class:`~xwm.data.MazeWorld` are Equinox
modules: jittable, vmappable, and written to be rolled out in bulk inside a
single device call. That is the right shape for generating training data and
the wrong shape for the evaluation loop, which steps one environment at a time,
interleaved with a planner, on the host. This adapter is the bridge, and it is
what lets the whole benchmark -- protocol, config, CLI, metrics -- be exercised
in CI with no simulator installed and no dataset downloaded.

**The state round trip is exact, and generically so.** These worlds keep their
state in a :class:`~typing.NamedTuple` of flat arrays, so flattening it gives a
vector and the recorded shapes give it back. Nothing here knows what a puck is.
Note that this state is the *simulator's*, not
:meth:`~xwm.data.PushWorld.state_vector`, which is a feature vector for an
encoder and is deliberately redundant (it repeats the goal-to-puck offset). A
state you can rebuild the world from and a state you can hand a model are
different objects, and conflating them is how a reset silently loses a velocity.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from ..core.types import Frame

__all__ = ["JaxWorldEnv"]


class JaxWorldEnv:
    """Step one :mod:`xwm.data` world at a time, in NumPy.

    Args:
        world: a :class:`xwm.data.PushWorld`, :class:`~xwm.data.MazeWorld` or
            :class:`~xwm.data.SpriteWorld`.
        seed: default seed for :meth:`reset`.

    The world's ``step`` and ``render`` are jitted once on construction, so the
    per-step cost is a device call rather than a trace.
    """

    def __init__(self, world: Any, seed: int = 0):
        self.world = world
        self.seed = seed
        self.action_dim = int(world.action_dim)
        # These worlds take actions in [-1, 1] and clip internally.
        self.action_low = -np.ones(self.action_dim, np.float32)
        self.action_high = np.ones(self.action_dim, np.float32)
        self.native_size = (int(world.size), int(world.size))
        self._step = jax.jit(world.step)
        self._render = jax.jit(world.render)
        self._state = world.reset(jr.PRNGKey(seed))
        # Recorded once, from a real state: the shapes flattening needs to be
        # reversible, and the field order a NamedTuple already fixes.
        leaves = jax.tree_util.tree_leaves(self._state)
        self._shapes = [leaf.shape for leaf in leaves]
        self._sizes = [int(np.prod(shape)) for shape in self._shapes]
        self.state_dim = sum(self._sizes)
        self._has_features = hasattr(world, "state_vector")

    # -- state ----------------------------------------------------------------
    def state(self) -> np.ndarray:
        """The simulator state, flattened. What :meth:`reset` accepts back."""
        leaves = jax.tree_util.tree_leaves(self._state)
        return np.concatenate([np.asarray(x, np.float32).reshape(-1) for x in leaves])

    def _unflatten(self, flat: np.ndarray) -> Any:
        flat = np.asarray(flat, np.float32).reshape(-1)
        if flat.size != self.state_dim:
            raise ValueError(f"expected a state of width {self.state_dim}, got {flat.size}")
        leaves, offset = [], 0
        for shape, size in zip(self._shapes, self._sizes, strict=True):
            leaves.append(jnp.asarray(flat[offset : offset + size].reshape(shape)))
            offset += size
        treedef = jax.tree_util.tree_structure(self._state)
        return jax.tree_util.tree_unflatten(treedef, leaves)

    # -- the Env protocol -----------------------------------------------------
    def reset(self, *, seed: int | None = None, state: np.ndarray | None = None) -> Frame:
        """Start an episode, either from ``seed`` or at an exact ``state``."""
        if state is not None:
            self._state = self._unflatten(state)
        else:
            self._state = self.world.reset(jr.PRNGKey(self.seed if seed is None else seed))
        return self.observe()

    def step(self, action: np.ndarray) -> Frame:
        self._state = self._step(self._state, jnp.asarray(action, jnp.float32))
        return self.observe()

    def observe(self) -> Frame:
        """Pixels plus the model-facing feature vector, in the dataset's vocabulary.

        ``state`` is omitted for a world that defines no ``state_vector`` --
        :class:`xwm.data.SpriteWorld` has no goal and so no task state. Such a
        world is still steppable and renderable; it just cannot back a
        goal-reaching task.
        """
        frame: Frame = {
            # uint8 like a recorded frame: the resize happens once, downstream,
            # through the same to_frames() call the dataset path uses.
            "image": np.asarray(self._render(self._state) * 255.0, np.uint8),
        }
        if self._has_features:
            frame["state"] = np.asarray(self.world.state_vector(self._state), np.float32)
        return frame

    def render(self, size: int | None = None) -> np.ndarray:
        """``(3, H, W)`` float32 in ``[0, 1]``. ``size`` is accepted and ignored.

        These worlds rasterise at a fixed resolution set on the world itself, so
        there is no supersampled path like :meth:`xwm.envs.FrankaEnv.render`'s.
        """
        del size
        return np.asarray(self._render(self._state), np.float32)

    def close(self) -> None:
        """Nothing to release."""

    # -- task quantities ------------------------------------------------------
    def goal_distance(self) -> float:
        """Ground-truth distance to the goal, for evaluation rather than training."""
        return float(self.world.goal_distance(self._state))

    @property
    def has_task_state(self) -> bool:
        """Whether this world exposes a ``state_vector`` and so can back a task."""
        return self._has_features

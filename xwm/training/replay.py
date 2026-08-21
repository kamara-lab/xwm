"""A trajectory replay buffer for the reward-driven families.

JEPA trains on a shuffled pile of clips, so :func:`xwm.data.iter_batches` is
enough. TD-MPC2 and MuZero cannot use that: their losses unroll a model for
``horizon`` steps, so a sample is a *contiguous slice* of one episode, aligned
across observations, actions and rewards. Sampling independent transitions would
make the unroll meaningless.

Storage is a flat ring of steps plus an episode id per step, and a slice is
rejected if it straddles two episodes -- otherwise the model would be trained to
predict across a reset, which is the one transition its dynamics can never get
right.

References
----------
Lin, *Self-Improving Reactive Agents Based on Reinforcement Learning, Planning
and Teaching*, Machine Learning 1992 -- experience replay.

Schrittwieser et al., *MuZero*, Nature 2020. arXiv:1911.08265 -- storing search
statistics alongside transitions, which the ``extra`` fields support.
"""

from __future__ import annotations

import numpy as np

from ..core.types import Batch

__all__ = ["ReplayBuffer"]


class ReplayBuffer:
    """Fixed-capacity ring buffer over trajectory steps.

    Args:
        capacity: number of steps to retain.
        observation_shape: shape of a single observation.
        action_shape: shape of a single action (``()`` for a discrete index).
        extra: additional per-step fields to store, as ``name -> shape``. Use it
            for MuZero's search targets (``value_target``, ``policy_target``).
        seed: RNG seed for sampling.

    NumPy rather than JAX: this is host-side mutable storage with random writes,
    which is exactly what JAX arrays are bad at. Batches are handed over as
    NumPy and converted at the jit boundary.
    """

    def __init__(
        self,
        capacity: int,
        observation_shape: tuple[int, ...],
        action_shape: tuple[int, ...] = (),
        *,
        extra: dict[str, tuple[int, ...]] | None = None,
        seed: int = 0,
    ):
        if capacity < 2:
            raise ValueError(f"capacity must be at least 2, got {capacity}")
        self.capacity = capacity
        self._observation = np.zeros((capacity, *observation_shape), np.float32)
        self._action = np.zeros((capacity, *action_shape), np.float32)
        self._reward = np.zeros((capacity,), np.float32)
        self._episode = np.full((capacity,), -1, np.int64)
        self._extra = {
            name: np.zeros((capacity, *shape), np.float32)
            for name, shape in (extra or {}).items()
        }
        self._cursor = 0
        self._size = 0
        self._episode_counter = 0
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self._size

    @property
    def episodes(self) -> int:
        return self._episode_counter

    def add_episode(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        **extra: np.ndarray,
    ) -> None:
        """Append one episode.

        Args:
            observations: ``(T + 1, ...)`` -- one more than the actions, since the
                final observation is the state the last action led to.
            actions: ``(T, ...)``.
            rewards: ``(T,)``.
            extra: any fields declared in ``extra`` at construction, ``(T + 1, ...)``
                or ``(T, ...)``.
        """
        observations = np.asarray(observations, np.float32)
        actions = np.asarray(actions, np.float32)
        rewards = np.asarray(rewards, np.float32)
        steps = actions.shape[0]
        if observations.shape[0] != steps + 1:
            raise ValueError(
                f"expected {steps + 1} observations for {steps} actions, "
                f"got {observations.shape[0]}"
            )
        if rewards.shape[0] != steps:
            raise ValueError(f"expected {steps} rewards, got {rewards.shape[0]}")

        episode = self._episode_counter
        self._episode_counter += 1
        for t in range(steps):
            i = self._cursor
            self._observation[i] = observations[t]
            self._action[i] = actions[t]
            self._reward[i] = rewards[t]
            self._episode[i] = episode
            for name, buffer in self._extra.items():
                if name not in extra:
                    raise KeyError(f"missing extra field {name!r}")
                buffer[i] = np.asarray(extra[name], np.float32)[t]
            self._cursor = (self._cursor + 1) % self.capacity
            self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int, horizon: int) -> Batch:
        """Sample ``batch_size`` slices of ``horizon`` steps.

        Returns a dict with ``observation`` ``(B, horizon + 1, ...)``, ``action``
        and ``reward`` ``(B, horizon, ...)``, plus any extra fields at
        ``(B, horizon + 1, ...)``.

        Raises:
            ValueError: if no slice of that length fits inside a single episode.
        """
        if horizon < 1:
            raise ValueError(f"horizon must be positive, got {horizon}")
        starts = self._valid_starts(horizon)
        if starts.size == 0:
            raise ValueError(
                f"no episode in the buffer holds {horizon + 1} consecutive steps; "
                "collect longer episodes or lower the horizon"
            )
        chosen = self._rng.choice(starts, size=batch_size, replace=True)
        offsets = np.arange(horizon + 1)
        index = (chosen[:, None] + offsets[None, :]) % self.capacity

        batch: Batch = {
            "observation": self._observation[index],
            "action": self._action[index[:, :horizon]],
            "reward": self._reward[index[:, :horizon]],
        }
        for name, buffer in self._extra.items():
            batch[name] = buffer[index]
        return batch

    def _valid_starts(self, horizon: int) -> np.ndarray:
        """Start offsets whose next ``horizon + 1`` steps stay in one episode.

        A slice crossing an episode boundary would train the dynamics to predict
        through a reset, so those starts are excluded rather than clipped.
        """
        if self._size < horizon + 1:
            return np.empty((0,), np.int64)
        offsets = np.arange(horizon + 1)
        # Only consider windows fully inside the written region.
        candidates = np.arange(self._size - horizon)
        if self._size == self.capacity:
            candidates = np.arange(self.capacity)
        window = (candidates[:, None] + offsets[None, :]) % self.capacity
        episodes = self._episode[window]
        same = np.all(episodes == episodes[:, :1], axis=1) & np.all(episodes >= 0, axis=1)
        return candidates[same]

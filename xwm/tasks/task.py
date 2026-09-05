"""A task, bound: an environment to step, batches to train on, a metric to score with."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

import jax.numpy as jnp
import numpy as np

from ..core.types import Batch, Frame, PRNGKey
from ..data.batching import iter_batches
from ..datasets.spec import clips, to_frames
from ..envs.protocol import Env
from .data import cached_episodes, denormalize_action, normalize_action, unblock
from .spec import TaskSpec

__all__ = ["Task"]


class Task:
    """Everything a model needs to be trained and evaluated on one task.

    Built by :func:`xwm.tasks.create`, not directly. Holds no model and no
    planner: a :class:`Task` is what both of those are pointed *at*, so the same
    instance serves training, evaluation and the baselines.
    """

    def __init__(
        self,
        spec: TaskSpec,
        *,
        env_builder: Callable[..., Env],
        metric: tuple[Callable[..., float], Callable[..., bool]],
        episode_source: Callable[[TaskSpec], Iterator[dict[str, np.ndarray]]] | None = None,
    ):
        self.spec = spec
        self._env_builder = env_builder
        self._distance, self._success = metric
        self._episode_source = episode_source
        self._episodes: dict[str, list[dict[str, np.ndarray]]] = {}

    def __repr__(self) -> str:
        spec = self.spec
        return f"Task({spec.name!r}, frameskip={spec.frameskip}, history={spec.history})"

    # -- environment ----------------------------------------------------------
    def make_env(self, **overrides: Any) -> Env:
        """Construct the simulator. Heavy imports happen here, not at import time."""
        return self._env_builder(**{**self.spec.env_options, **overrides})

    # -- observations ---------------------------------------------------------
    def preprocess_frame(self, frame: Frame) -> dict[str, jnp.ndarray]:
        """One observation -> model input, by the same route a recorded frame takes.

        This is the function that keeps training and evaluation in the same
        distribution: it is called with a dataset frame during training and with
        a live environment frame during evaluation, with identical arguments.
        """
        out: dict[str, jnp.ndarray] = {}
        if "image" in frame:
            resized = to_frames(np.asarray(frame["image"])[None], resize=self.spec.resize)
            out["video"] = jnp.asarray(resized[0])
        if "state" in frame:
            out["state"] = jnp.asarray(np.asarray(frame["state"], np.float32))
        return out

    def observation(self, frame: Frame) -> jnp.ndarray:
        """The single array the model encodes, picked by ``spec.observation``."""
        prepared = self.preprocess_frame(frame)
        wanted = "video" if "video" in self.spec.observation else "state"
        if wanted not in prepared:
            raise KeyError(
                f"task {self.spec.name!r} observes {wanted!r} but the frame has "
                f"{sorted(prepared)}; check the environment's observe()"
            )
        return prepared[wanted]

    # -- actions --------------------------------------------------------------
    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return self.spec.bounds

    def normalize_action(self, action: np.ndarray) -> np.ndarray:
        low, high = self.bounds
        return normalize_action(action, low, high)

    def denormalize_action(self, action: np.ndarray) -> np.ndarray:
        low, high = self.bounds
        return denormalize_action(action, low, high)

    def unblock(self, blocked: np.ndarray) -> np.ndarray:
        """One planned action -> the ``frameskip`` raw actions it stands for."""
        return unblock(blocked, self.spec.action_dim)

    def env_state(self, recorded: np.ndarray, env: Env) -> np.ndarray:
        """The simulator state inside a recorded state vector.

        A recording stores what a *model* should see, which is usually the
        simulator's own state followed by derived conveniences --
        :meth:`xwm.data.PushWorld.state_vector` appends the goal-to-puck offset
        because making an encoder learn a subtraction is a waste. The simulator
        needs only the leading part, and taking a prefix is exact for every task
        registered here.

        A task whose recording does not embed the simulator state this way
        overrides this. The conformance test to keep is that
        ``env.reset(state=task.env_state(recorded, env))`` reproduces that
        state, and that the ``replay`` baseline then reaches the goal.
        """
        recorded = np.asarray(recorded, np.float32).reshape(-1)
        if recorded.size < env.state_dim:
            raise ValueError(
                f"task {self.spec.name!r} records a {recorded.size}-wide state but "
                f"{type(env).__name__} needs {env.state_dim} to reset"
            )
        return recorded[: env.state_dim]

    # -- data -----------------------------------------------------------------
    def episodes(
        self, *, split: str = "val", limit: int | None = None, refresh: bool = False
    ) -> list[dict[str, np.ndarray]]:
        """Preprocessed episodes for a split, cached in memory and on disk."""
        if split not in ("train", "val"):
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")
        if split not in self._episodes or refresh:
            source = None if self._episode_source is None else self._episode_source(self.spec)
            self._episodes[split] = cached_episodes(
                self.spec, split, source=source, limit=limit, refresh=refresh
            )
        return self._episodes[split]

    def batches(
        self,
        *,
        batch_size: int,
        key: PRNGKey,
        split: str = "train",
        length: int | None = None,
        stride: int = 1,
        limit: int | None = None,
        observation_key: str = "video",
    ) -> Iterator[Batch]:
        """Endless shuffled clips, in the layout the model's ``loss`` expects.

        ``{observation_key: (B, T, ...) float32, "action": (B, T - 1, k * A)}``,
        with ``T = length`` defaulting to ``history + 1``.

        Args:
            observation_key: what to call the observation field. The JEPA
                families read ``"video"``; TD-MPC2 and MuZero read
                ``"observation"``. Which array fills it is the task's business
                (``spec.observation``), which name it goes under is the model's,
                and :func:`xwm.cli.build.batch_keys` reads the latter off the
                model rather than guessing from its class.
        """
        length = self.spec.clip_length if length is None else length
        cut = clips(self.episodes(split=split, limit=limit), length=length, stride=stride)
        dropped = int(cut.pop("dropped", np.zeros(1))[0]) if "dropped" in cut else 0
        if dropped:
            total = len(self.episodes(split=split))
            if dropped > total // 2:
                raise ValueError(
                    f"{dropped} of {total + dropped} episodes are shorter than {length} frames "
                    f"at frameskip {self.spec.frameskip}; the clips would come from a minority "
                    "of the corpus. Lower history/horizon or the frameskip."
                )
        cut.pop("episode", None)
        # float32 in [0, 1] at the jit boundary; uint8 up to here.
        if "video" in cut:
            cut["video"] = np.asarray(cut["video"], np.float32) / 255.0
        # The task decides *which* array the model observes; the model decides
        # what the field is called.
        source = "video" if "video" in self.spec.observation else "state"
        if source not in cut:
            raise KeyError(
                f"task {self.spec.name!r} observes {source!r} but its episodes have "
                f"{sorted(k for k in cut if k != 'dropped')}"
            )
        if observation_key != source:
            cut[observation_key] = cut.pop(source)
        # Nothing else travels: a state-observation model does not want a batch
        # of 32x32 frames crossing the jit boundary on every step.
        cut = {
            k: v for k, v in cut.items() if k in (observation_key, "action", "reward", "success")
        }
        # epochs=None, i.e. endless. The default is a single pass, which would
        # silently cap a run at one epoch: Trainer.fit stops when either the
        # step budget or the iterator runs out, and a 2000-step run that ends
        # after 41 steps reports success just as loudly as one that does not.
        return iter_batches(cut, batch_size, key=key, shuffle=True, epochs=None)

    # -- scoring --------------------------------------------------------------
    def distance(self, state: np.ndarray, goal: np.ndarray) -> float:
        """Ground-truth distance between two task states. Never seen by the model."""
        return float(self._distance(state, goal, **self.spec.metric_options))

    def success(self, state: np.ndarray, goal: np.ndarray) -> bool:
        """Whether ``state`` counts as having reached ``goal``."""
        return bool(self._success(state, goal, **self.spec.metric_options))

"""Offline RL datasets: OGBench over plain HTTP, and Minari.

These are the datasets a state-based world model should be measured on. xwm has
had an MLP state encoder and two reward-driven families since the beginning and
nothing to point them at except a Franka arm in a simulator, which is one task.
OGBench is 85 datasets across eight environments, and it is a static file server
of ``.npz`` -- no client library, no hub, no new dependency.

**One thing to get right.** OGBench distinguishes ``terminals`` from ``masks``,
and they are not the same flag. ``terminals`` marks the end of a *trajectory*;
``masks`` marks *task completion*, and is what belongs in a Bellman backup. Treat
them as interchangeable and the value function learns to bootstrap across a
reset, which is silent -- the loss still falls. This reader segments episodes on
``terminals`` and passes ``masks`` through untouched when present.

References
----------
Park, Frans, Eysenbach & Levine, *OGBench: Benchmarking Offline
Goal-Conditioned RL*, ICLR 2025. arXiv:2410.20092.

Younis et al., *Minari: A dataset API for offline RL*, 2024 -- the successor to
D4RL, and the reason this module reads its format too.

Fu et al., *D4RL*, 2020. arXiv:2004.07219 -- where the flat
``observations``/``actions``/``terminals`` layout comes from.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np

from .cache import cache_dir, http_fetch, require
from .spec import clips, episode_slices, to_frames

__all__ = ["OGBENCH_URL", "iter_episodes", "load", "load_minari"]

#: OGBench's static file server. Datasets are ``<name>.npz`` with a ``<name>-val.npz``
#: beside them.
OGBENCH_URL = "https://rail.eecs.berkeley.edu/datasets/ogbench"


def _download(name: str, *, split: str = "train", progress: bool = True) -> Path:
    """Fetch one OGBench ``.npz``, cached under ``ogbench/``."""
    stem = name if split == "train" else f"{name}-val"
    return http_fetch(
        f"{OGBENCH_URL}/{stem}.npz",
        cache_dir("ogbench") / f"{stem}.npz",
        progress=progress,
    )


def iter_episodes(
    name: str,
    *,
    split: str = "train",
    observation: str = "both",
    limit: int | None = None,
    resize: int | None = None,
    dtype: str = "float32",
    progress: bool = True,
    path: str | Path | None = None,
) -> Iterator[dict[str, np.ndarray]]:
    """Stream one OGBench dataset as per-episode dicts.

    Args:
        name: an OGBench dataset name, e.g. ``antmaze-large-navigate-v0``.
            Names starting with ``visual-`` carry ``(64, 64, 3)`` frames instead
            of a state vector.
        split: ``"train"`` or ``"val"``.
        observation: ``"video"``, ``"state"`` or ``"both"``. Each OGBench file
            holds one or the other, never both, so this is an assertion rather
            than a selection: asking for the modality the file does not have is
            an error, and ``"both"`` accepts whichever it is. It exists so the
            registry can pass the same argument to every reader.
        limit: stop after this many episodes. The full datasets are millions of
            transitions, so this is the parameter that decides whether a call
            takes a second or fills memory.
        resize, dtype: passed to :func:`~xwm.datasets.spec.to_frames`, for the
            visual variants.
        path: read this file instead of downloading. For tests, and for a
            dataset you generated yourself with OGBench's scripts.

    Yields dicts with ``state`` or ``video`` (``T`` entries) and ``action``
    (``T - 1``), plus ``mask`` where the dataset provides one.

    OGBench datasets carry **no reward**: the reward is a property of the task
    you evaluate against, not of the data, which is the whole premise of a
    goal-conditioned benchmark. :func:`xwm.datasets.to_replay_buffer` therefore
    wants an explicit ``reward=`` for these.
    """
    if observation not in ("video", "state", "both"):
        raise ValueError(f"observation must be 'video', 'state' or 'both', got {observation!r}")
    source = Path(path) if path is not None else _download(name, split=split, progress=progress)
    with np.load(source, allow_pickle=False) as raw:
        arrays = {key: raw[key] for key in raw.files}

    observations = arrays.get("observations")
    actions = arrays.get("actions")
    if observations is None or actions is None:
        raise ValueError(
            f"{source} has {sorted(arrays)}; expected at least 'observations' and 'actions'"
        )
    # `valids` is the compact variant's inverse of `terminals`: 0 on the step
    # whose next observation is not in the file.
    if "terminals" in arrays:
        terminals = arrays["terminals"]
    elif "valids" in arrays:
        terminals = 1.0 - np.asarray(arrays["valids"])
    else:
        raise ValueError(f"{source} has no 'terminals' or 'valids'; cannot find episode ends")
    next_observations = arrays.get("next_observations")
    masks = arrays.get("masks")
    visual = observations.ndim == 4
    if observation != "both" and observation != ("video" if visual else "state"):
        held = "video" if visual else "state"
        raise ValueError(
            f"{source} holds {held} observations, not {observation}; OGBench ships "
            "state and pixel variants as separate datasets (the pixel ones are "
            "prefixed 'visual-')"
        )

    for count, (start, stop) in enumerate(episode_slices(terminals)):
        if limit is not None and count >= limit:
            return
        frames = observations[start:stop]
        if next_observations is not None:
            # The recorded final state, so T = steps + 1 and every action has a
            # consequence in the episode. Without it the last step has to go.
            frames = np.concatenate([frames, next_observations[stop - 1 : stop]])
            episode_actions = actions[start:stop]
            episode_masks = None if masks is None else masks[start:stop]
        else:
            episode_actions = actions[start : stop - 1]
            episode_masks = None if masks is None else masks[start : stop - 1]
        if episode_actions.shape[0] < 1:
            continue

        episode: dict[str, np.ndarray] = {"action": np.asarray(episode_actions, np.float32)}
        if visual:
            episode["video"] = to_frames(frames, resize=resize, dtype=dtype)
        else:
            episode["state"] = np.asarray(frames, np.float32)
        if episode_masks is not None:
            episode["mask"] = np.asarray(episode_masks, np.float32)
        yield episode


def load(
    name: str,
    *,
    length: int = 16,
    stride: int | None = None,
    limit: int | None = None,
    max_clips: int | None = None,
    **kwargs,
) -> dict[str, np.ndarray]:
    """Load an OGBench dataset as fixed-length clips.

    See :func:`iter_episodes` for the arguments and
    :func:`~xwm.datasets.spec.clips` for the windowing. Everything is held in
    memory, so pass ``limit`` unless you know the dataset is small.
    """
    episodes = iter_episodes(name, limit=limit, **kwargs)
    return clips(episodes, length=length, stride=stride, max_clips=max_clips)


def _flatten_observations(observations) -> np.ndarray:
    """Minari observations as one ``(T, S)`` float32 array.

    Goal-conditioned datasets -- every ``D4RL/pointmaze/*`` and
    ``D4RL/antmaze/*`` -- use a **dict** observation space, stored as a dict of
    arrays rather than one array. Sorted-key concatenation turns that into a
    state vector, which for pointmaze means
    ``achieved_goal | desired_goal | observation``: the goal is part of the
    state, which is what a goal-conditioned value function needs and what
    :class:`xwm.encoders.StateEncoder` can take. Nested dicts flatten the same
    way, one level at a time.

    The key order is alphabetical, so it is stable but arbitrary: a state
    dimension read off one dataset does not transfer to another.
    """
    if isinstance(observations, dict):
        # Recursive, because the space can nest: D4RL/kitchen's `achieved_goal`
        # is itself a Dict of per-object boxes.
        parts = [_flatten_observations(observations[key]) for key in sorted(observations)]
        return np.concatenate(parts, axis=-1)
    value = np.asarray(observations, np.float32)
    return value.reshape(value.shape[0], -1)


def load_minari(
    name: str,
    *,
    length: int = 16,
    stride: int | None = None,
    limit: int | None = None,
    max_clips: int | None = None,
    observation: str = "state",
    download: bool = True,
) -> dict[str, np.ndarray]:
    """Load a Minari dataset as fixed-length clips.

    Minari already stores per-episode observations of length ``T + 1`` against
    ``T`` actions and rewards, which is exactly this library's convention, so
    this reader is close to a transcription. The one thing it does is flatten
    dict observation spaces -- see :func:`_flatten_observations`.

    Needs ``minari`` itself: ``pip install minari``. It is not in the ``data``
    extra because it brings Gymnasium and a specific set of environment
    dependencies along with it, and most users of this function already have the
    version they want pinned.
    """
    if observation not in ("state", "both"):
        raise ValueError(
            f"Minari datasets are state-based; observation must be 'state', got "
            f"{observation!r}"
        )
    (minari,) = require("minari", extra="")
    dataset = minari.load_dataset(name, download=download)

    def episodes() -> Iterator[dict[str, np.ndarray]]:
        for count, episode in enumerate(dataset.iterate_episodes()):
            if limit is not None and count >= limit:
                return
            observations = _flatten_observations(episode.observations)
            actions = np.asarray(episode.actions, np.float32)
            rewards = np.asarray(episode.rewards, np.float32)
            # Minari stores the reset observation, so T = actions + 1 already.
            # A dataset that does not gets its last action dropped instead.
            if observations.shape[0] == actions.shape[0]:
                actions, rewards = actions[:-1], rewards[:-1]
            yield {
                "state": observations,
                "action": actions,
                "reward": rewards,
            }

    return clips(episodes(), length=length, stride=stride, max_clips=max_clips)

"""From recorded episodes to training batches, in the order that order matters.

The pipeline is::

    stream -> frameskip -> preprocess -> split -> cache -> clip -> batch

and the order is not arbitrary. Frameskipping *after* clipping would let a clip
straddle a subsample boundary; splitting *after* clipping would put overlapping
windows of one episode on both sides of the train/held-out line, which leaks and
flatters every number downstream. :func:`xwm.datasets.spec.check_episode` is
re-asserted after the frameskip step so a misalignment fails immediately rather
than becoming a dynamics model that has quietly learned a blurred identity.

Episodes are cached, clips are not. At ``stride=1`` a four-frame window
duplicates every frame four times, so caching clips would multiply an already
large corpus by the window length; cutting them in memory costs a slice.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from ..datasets.spec import FRAME_FIELDS, TRANSITION_FIELDS, check_episode, to_frames
from ..tools.cache import cache_dir
from .spec import TaskSpec

__all__ = [
    "cached_episodes",
    "frameskip_episode",
    "normalize_action",
    "denormalize_action",
    "preprocess_episode",
    "split_episodes",
    "unblock",
]


def frameskip_episode(episode: dict[str, np.ndarray], k: int) -> dict[str, np.ndarray]:
    """Subsample frames by ``k`` and **group** the actions between them.

    One model step then covers ``k`` raw steps, and carries all ``k`` actions
    concatenated -- ``(T' - 1, k * A)``. The alternative, repeating a single
    action ``k`` times, discards the rest of the recorded control signal and
    leaves the model unable to express what the demonstrator actually did.

    Per-transition fields other than ``action`` are summed over the group when
    they are rewards and taken from the last raw step otherwise, because that is
    the step whose consequence the next frame shows.
    """
    if k < 1:
        raise ValueError(f"frameskip must be at least 1, got {k}")
    n_frames = check_episode(episode)
    if k == 1:
        return dict(episode)

    # Frames at 0, k, 2k, ...  There are ``n_frames - 1`` actions, so the number
    # of *complete* groups is ``(n_frames - 1) // k``; a partial tail group is
    # dropped because the frame it would land on was never recorded.
    kept = (n_frames - 1) // k + 1
    if kept < 2:
        raise ValueError(
            f"an episode of {n_frames} frames yields {kept} frames at frameskip {k}; "
            "at least 2 are needed for one transition"
        )
    out: dict[str, np.ndarray] = {}
    for name, value in episode.items():
        if name in FRAME_FIELDS:
            out[name] = value[: (kept - 1) * k + 1 : k]
        elif name == "action":
            grouped = value[: (kept - 1) * k]
            out[name] = grouped.reshape(kept - 1, -1)
        elif name in TRANSITION_FIELDS:
            block = value[: (kept - 1) * k].reshape(kept - 1, k, *value.shape[1:])
            out[name] = block.sum(axis=1) if name == "reward" else block[:, -1]
        else:
            out[name] = value
    check_episode(out)
    return out


def unblock(blocked: np.ndarray, action_dim: int) -> np.ndarray:
    """``(k * A,)`` -> ``(k, A)``: the raw actions inside one model action."""
    blocked = np.asarray(blocked).reshape(-1)
    if blocked.size % action_dim:
        raise ValueError(f"a {blocked.size}-wide action does not divide into {action_dim}")
    return blocked.reshape(-1, action_dim)


def normalize_action(action: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """Raw units -> ``[-1, 1]``, from declared bounds.

    Declared, not measured: normalising by dataset statistics makes a
    checkpoint's action space depend on the split that trained it, so the same
    weights mean different things on different data.
    """
    action = np.asarray(action, np.float32)
    # The trailing axis may hold several grouped actions, so split it into
    # (..., k, A) before broadcasting the per-dimension bounds against it.
    grouped = action.reshape(*action.shape[:-1], -1, low.size)
    return (2.0 * (grouped - low) / (high - low) - 1.0).reshape(action.shape)


def denormalize_action(action: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """``[-1, 1]`` -> raw units. The inverse of :func:`normalize_action`."""
    action = np.asarray(action, np.float32)
    grouped = action.reshape(*action.shape[:-1], -1, low.size)
    return ((grouped + 1.0) / 2.0 * (high - low) + low).reshape(action.shape)


def preprocess_episode(episode: dict[str, np.ndarray], spec: TaskSpec) -> dict[str, np.ndarray]:
    """Resize frames to the model's resolution and normalise actions to ``[-1, 1]``.

    Frames stay ``uint8``: four times less memory than float32, and the cast
    belongs at the ``jit`` boundary rather than in a cache file.
    """
    out = dict(episode)
    if "video" in out:
        out["video"] = to_frames(out["video"], resize=spec.resize, dtype="uint8")
    low, high = spec.bounds
    if "action" in out:
        out["action"] = normalize_action(out["action"], low, high)
    check_episode(out)
    return out


def split_episodes(n: int, *, holdout: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic train/held-out episode indices.

    By episode, never by clip: overlapping windows of one episode on both sides
    of the split is a leak that shows up as a suspiciously good held-out number.
    """
    if n < 2:
        raise ValueError(f"need at least 2 episodes to split, got {n}")
    order = np.random.default_rng(seed).permutation(n)
    n_val = max(1, int(round(holdout * n)))
    return np.sort(order[n_val:]), np.sort(order[:n_val])


def fingerprint(spec: TaskSpec, split: str, limit: int | None = None) -> str:
    """Everything that changes the cached bytes, hashed.

    Deliberately *not* the whole spec: the planning and evaluation fields
    (``horizon``, ``episodes``, ``eval_budget``) do not affect a single
    preprocessed frame, and including them would invalidate a multi-gigabyte
    cache every time a planner setting moved.

    ``limit`` *is* included, and must be. It changes how many episodes the
    cache holds, so leaving it out lets a truncated debug run poison the cache
    for every full run after it -- silently, since a smaller corpus trains and
    reports exactly like a large one.
    """
    material = {
        "limit": limit,
        "dataset": spec.dataset,
        "dataset_options": spec.dataset_options,
        "image_size": spec.image_size,
        "native_size": list(spec.native_size),
        "frameskip": spec.frameskip,
        "action_dim": spec.action_dim,
        "bounds": [spec.bounds[0].tolist(), spec.bounds[1].tolist()],
        "holdout": spec.holdout,
        "split_seed": spec.split_seed,
        "split": split,
    }
    digest = json.dumps(material, sort_keys=True).encode()
    return hashlib.sha256(digest).hexdigest()[:12]


def cache_path(spec: TaskSpec, split: str, limit: int | None = None) -> Path:
    digest = fingerprint(spec, split, limit)
    return cache_dir("tasks") / f"{spec.name.replace('/', '-')}-{split}-{digest}.npz"


def cached_episodes(
    spec: TaskSpec,
    split: str,
    *,
    source: Iterator[dict[str, np.ndarray]] | None = None,
    limit: int | None = None,
    refresh: bool = False,
) -> list[dict[str, np.ndarray]]:
    """Preprocessed episodes for one split, read from or written to the cache.

    Args:
        source: a stream of raw episodes. ``None`` streams from
            ``spec.dataset``. Passing one is how a task with a bespoke reader
            (Push-T joins two datasets) reuses the rest of this pipeline.
        limit: cap on *raw* episodes read, applied before the split.
        refresh: recompute even if a cache file exists.

    Episodes are stored flattened into one ``.npz`` -- ``ep{i}.{field}`` -- and
    rebuilt on load, because ``np.savez`` has no notion of a ragged list and
    every episode here has a different length.
    """
    path = cache_path(spec, split, limit)
    if path.exists() and not refresh:
        with np.load(path) as handle:
            grouped: dict[int, dict[str, np.ndarray]] = {}
            for key in handle.files:
                index, _, name = key.partition(".")
                grouped.setdefault(int(index[2:]), {})[name] = handle[key]
        return [grouped[i] for i in sorted(grouped)]

    if source is None:
        from ..datasets import stream as dataset_stream

        if spec.dataset is None:
            raise ValueError(f"task {spec.name!r} has no dataset; pass source=")
        source = dataset_stream(spec.dataset, limit=limit, **spec.dataset_options)

    raw = []
    for count, episode in enumerate(source):
        if limit is not None and count >= limit:
            break
        raw.append(preprocess_episode(frameskip_episode(episode, spec.frameskip), spec))

    train_idx, val_idx = split_episodes(len(raw), holdout=spec.holdout, seed=spec.split_seed)
    # Both splits are written, not just the one asked for. Reading a corpus is
    # the expensive half of this pipeline -- decoding 206 mp4 episodes, say --
    # and it has already been done for every episode by the time we get here.
    # Writing only the requested split would make the very next call, for the
    # held-out episodes the evaluator needs, do all of it again.
    written = {}
    for name, indices in (("train", train_idx), ("val", val_idx)):
        chosen = [raw[i] for i in indices]
        np.savez_compressed(
            cache_path(spec, name, limit),
            **{f"ep{i}.{k}": v for i, ep in enumerate(chosen) for k, v in ep.items()},
        )
        written[name] = chosen
    return written[split]

"""The contract a recorded dataset has to satisfy, and the tools to satisfy it.

Every reader in this package emits the same thing, because every model in xwm
consumes the same thing -- the field layout
:func:`xwm.data.sprite_sequences` and :func:`xwm.envs.franka_sequences` already
produce:

===========  ==========================  =============================================
field        shape                       meaning
===========  ==========================  =============================================
``video``    ``(n, T, C, H, W)``         frames, or ``image`` ``(n, C, H, W)`` for stills
``state``    ``(n, T, S)``               proprioception, for a state encoder
``action``   ``(n, T - 1, A)``           ``action[i, t]`` joins frames ``t`` and ``t + 1``
``reward``   ``(n, T - 1)``              the reward of the state each action led to
``task``     ``(n,)`` int32              index into ``task_names``
===========  ==========================  =============================================

The ``T - 1`` is the part worth stating twice. Every recorded format on disk
stores one action per frame, including a final action whose consequence was
never recorded. Keeping it would misalign the entire dataset by one step and the
symptom -- a dynamics model that learns a slightly blurred identity -- looks like
underfitting rather than like a bug. So readers drop it, and
:func:`check_episode` refuses anything that did not.

Episodes in the wild have unequal length, and models take fixed-length clips, so
:func:`clips` cuts each episode into windows. Windows never straddle an episode
boundary, for the same reason :class:`xwm.training.ReplayBuffer` rejects slices
that do: no dynamics model can predict through a reset.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .cache import require

__all__ = [
    "EPISODE_FIELDS",
    "FRAME_FIELDS",
    "TRANSITION_FIELDS",
    "DatasetSpec",
    "check_episode",
    "clips",
    "episode_slices",
    "to_frames",
]

#: One entry per frame: ``T`` of them.
FRAME_FIELDS = ("video", "image", "state", "position")
#: One entry per transition: ``T - 1`` of them.
TRANSITION_FIELDS = ("action", "reward", "success", "done", "terminal", "mask")
#: One entry per episode.
EPISODE_FIELDS = ("task", "episode")


@dataclass(frozen=True)
class DatasetSpec:
    """What a registered dataset is, and what asking for it will cost.

    Flat scalars only, like :class:`xwm.envs.FrankaConfig`. ``size_gb`` and
    ``episodes`` are the fields that matter most in practice: they are what
    :func:`xwm.datasets.describe` exists to show a user *before* a download
    starts rather than after.
    """

    name: str
    reader: str
    source: str
    summary: str
    action_dim: int
    observation: tuple[str, ...] = ("video",)
    episodes: int | None = None
    size_gb: float | None = None
    licence: str = "see source"
    citation: str = ""
    options: dict[str, Any] = field(default_factory=dict)


def check_episode(episode: dict[str, np.ndarray]) -> int:
    """Validate one episode against the contract and return its frame count.

    Raises:
        ValueError: if the per-frame and per-transition fields disagree, which
            almost always means a reader forgot to drop the last action.
    """
    frames = {k: v.shape[0] for k, v in episode.items() if k in FRAME_FIELDS}
    if not frames:
        raise ValueError(
            f"episode has no per-frame field; expected one of {FRAME_FIELDS}, "
            f"got {sorted(episode)}"
        )
    if len(set(frames.values())) != 1:
        raise ValueError(f"per-frame fields disagree on length: {frames}")
    n_frames = next(iter(frames.values()))

    transitions = {k: v.shape[0] for k, v in episode.items() if k in TRANSITION_FIELDS}
    for name, length in transitions.items():
        if length != n_frames - 1:
            raise ValueError(
                f"{name!r} has {length} entries for {n_frames} frames; expected "
                f"{n_frames - 1}. Per-transition fields carry one fewer entry than "
                "frames -- drop the final recorded action, whose result was never "
                "observed."
            )
    return n_frames


def clips(
    episodes: Iterable[dict[str, np.ndarray]],
    *,
    length: int = 16,
    stride: int | None = None,
    max_clips: int | None = None,
) -> dict[str, np.ndarray]:
    """Cut a stream of episodes into fixed-length clips and stack them.

    Args:
        episodes: per-episode dicts, each satisfying :func:`check_episode`.
        length: frames per clip. Episodes shorter than this are skipped, and how
            many were skipped is reported in ``dropped``.
        stride: window step. Defaults to ``length``, i.e. no overlap.
        max_clips: stop after this many clips. Readers take ``limit`` instead,
            which counts *episodes* -- the two are deliberately named
            differently, because "give me 100" means 100 episodes to a person
            asking for data and 100 clips to the function that cuts them.

    Returns the stacked dataset, plus two bookkeeping keys, both ``(n,)``:
    ``episode``, the source episode index of each clip, and ``dropped``, the
    number of episodes too short to yield one, repeated per clip. ``dropped`` is
    returned rather than warned about because a ``length`` that silently discards
    most of a corpus is the single easiest way to draw a conclusion from 4% of a
    dataset.
    """
    if length < 2:
        raise ValueError(f"length must be at least 2, got {length}")
    stride = length if stride is None else stride
    if stride < 1:
        raise ValueError(f"stride must be positive, got {stride}")

    collected: dict[str, list[np.ndarray]] = {}
    dropped = 0
    n_clips = 0
    for index, episode in enumerate(episodes):
        n_frames = check_episode(episode)
        if n_frames < length:
            dropped += 1
            continue
        for start in range(0, n_frames - length + 1, stride):
            for name, value in episode.items():
                if name in FRAME_FIELDS:
                    window = value[start : start + length]
                elif name in TRANSITION_FIELDS:
                    window = value[start : start + length - 1]
                elif name in EPISODE_FIELDS:
                    window = value  # per-episode metadata, copied onto each clip
                else:
                    raise ValueError(
                        f"unknown field {name!r}: it has to be declared in one of "
                        "FRAME_FIELDS, TRANSITION_FIELDS or EPISODE_FIELDS so that "
                        "windowing knows whether to cut it. Falling back to "
                        "'per-episode' would copy it whole into every clip and "
                        "misalign it without failing."
                    )
                collected.setdefault(name, []).append(window)
            collected.setdefault("episode", []).append(np.int32(index))
            n_clips += 1
            if max_clips is not None and n_clips >= max_clips:
                break
        if max_clips is not None and n_clips >= max_clips:
            break

    if not collected:
        raise ValueError(
            f"no episode was at least {length} frames long ({dropped} too short); "
            "lower `length`"
        )
    out = {name: np.stack(values) for name, values in collected.items()}
    # Per-clip rather than a scalar, so every field in the returned dict shares a
    # leading axis and the whole thing can be handed to
    # :func:`xwm.data.iter_batches` unpicked. A scalar here makes that raise on
    # `shape[0]`, which is a poor way to learn about a bookkeeping field.
    out["dropped"] = np.full(out["episode"].shape[0], dropped, np.int32)
    return out


def to_frames(
    frames: np.ndarray,
    *,
    resize: int | tuple[int, int] | None = None,
    dtype: str = "float32",
) -> np.ndarray:
    """Normalise recorded frames to xwm's ``(T, C, H, W)`` convention.

    Accepts ``(T, H, W, C)`` -- what every robot dataset stores, since that is
    what a camera and a video codec produce -- as well as already-transposed
    ``(T, C, H, W)``, and grayscale ``(T, H, W)``.

    Args:
        resize: target ``size`` or ``(height, width)``. Needs Pillow. Floating
            point frames are assumed to be in ``[0, 1]`` and are round-tripped
            through 8-bit for the resample, since that is what PIL resizes;
            anything outside that range is an error rather than a black clip.
        dtype: ``"float32"`` rescales ``uint8`` to ``[0, 1]``, matching
            :class:`xwm.data.SpriteWorld`. ``"uint8"`` keeps the bytes, which is
            four times less memory and worth it for anything large -- convert at
            the ``jit`` boundary instead.
    """
    frames = np.asarray(frames)
    if frames.ndim == 3:  # (T, H, W) grayscale
        frames = frames[..., None]
    if frames.ndim != 4:
        raise ValueError(f"expected 3 or 4 dimensions, got {frames.shape}")
    # Channels-last unless the last axis is implausible as a channel count and
    # the second is plausible -- 1, 3 and 4 are the only real channel counts.
    if frames.shape[-1] not in (1, 3, 4) and frames.shape[1] in (1, 3, 4):
        frames = frames.transpose(0, 2, 3, 1)

    if resize is not None:
        (Image,) = require("PIL.Image", extra="data")
        height, width = (resize, resize) if isinstance(resize, int) else resize
        squeeze = frames.shape[-1] == 1
        # PIL resamples 8-bit channels, so float input has to be scaled up first.
        # Casting [0, 1] floats straight to uint8 truncates every pixel to 0 and
        # returns a black clip, which is a silent and very confusing failure.
        scaled = np.issubdtype(frames.dtype, np.floating)
        if scaled:
            if float(np.nanmax(frames)) > 1.0 + 1e-6:
                raise ValueError(
                    "resize= on floating-point frames assumes the range [0, 1]; "
                    f"got a maximum of {float(np.nanmax(frames)):.3f}. Pass uint8 "
                    "frames, or rescale first."
                )
            source = np.clip(frames * 255.0, 0, 255).astype(np.uint8)
        else:
            source = np.asarray(frames, np.uint8)
        resized = []
        for frame in source:
            array = frame[..., 0] if squeeze else frame
            image = Image.fromarray(array).resize((width, height), Image.BILINEAR)
            out = np.asarray(image)
            resized.append(out[..., None] if squeeze else out)
        frames = np.stack(resized)
        if scaled:
            frames = frames.astype(np.float32) / 255.0

    frames = frames.transpose(0, 3, 1, 2)  # (T, C, H, W)
    if dtype == "uint8":
        return np.ascontiguousarray(frames, dtype=np.uint8)
    if dtype != "float32":
        raise ValueError(f"dtype must be 'float32' or 'uint8', got {dtype!r}")
    if frames.dtype == np.uint8:
        return np.ascontiguousarray(frames, dtype=np.float32) / 255.0
    return np.ascontiguousarray(frames, dtype=np.float32)


def episode_slices(terminals: np.ndarray) -> Iterator[tuple[int, int]]:
    """Yield ``(start, stop)`` for each episode in a flat step array.

    ``terminals[t] == 1`` marks the last step of an episode, the convention
    OGBench, D4RL and Minari all use. A trailing unterminated episode is
    yielded too -- a truncated recording is still data.
    """
    terminals = np.asarray(terminals).reshape(-1)
    ends = np.flatnonzero(terminals > 0)
    start = 0
    for end in ends:
        yield start, int(end) + 1
        start = int(end) + 1
    if start < terminals.shape[0]:
        yield start, terminals.shape[0]

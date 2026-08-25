"""LIBERO and RoboMimic: the flat HDF5 layout the manipulation benchmarks share.

``data/demo_0/obs/<key>``, ``data/demo_0/actions``, and so on. It is the cheapest
format of the four to read -- one dependency, random access, no video codec --
and the two benchmarks stored in it are the ones planning results are usually
reported against, which is why it is here.

Three details that bite:

* **Demo order.** ``h5py`` lists groups in lexicographic order, so ``demo_10``
  comes between ``demo_1`` and ``demo_2``. Sorting numerically matters the moment
  ``limit`` is used: a "first 50 episodes" that is really a lexicographic sample
  is not reproducible against anything.
* **Two spellings of a camera.** LIBERO writes ``agentview_rgb``, RoboMimic
  writes ``agentview_image``, and some LIBERO releases JPEG-compress into
  ``agentview_rgb_jpeg`` with one variable-length byte string per frame. All
  three are handled.
* **The final action.** These files store ``T`` observations against ``T``
  actions, so the last action's result is not in the file. Where a ``next_obs``
  group exists it supplies that frame and every action is kept; otherwise the
  last action is dropped.

References
----------
Liu et al., *LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot
Learning*, NeurIPS 2023. arXiv:2306.03310.

Mandlekar et al., *What Matters in Learning from Offline Human Demonstrations
for Robot Manipulation*, CoRL 2021. arXiv:2108.03298 -- RoboMimic, and the
origin of this file layout.
"""

from __future__ import annotations

import io
import re
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np

from .cache import require
from .spec import clips, to_frames

__all__ = ["iter_episodes", "load"]

_IMAGE_HINT = re.compile(r"(rgb|image|depth)")
_DIGITS = re.compile(r"(\d+)")


def _demo_order(names: Sequence[str]) -> list[str]:
    """Sort ``demo_2`` before ``demo_10``, which lexicographic order does not."""

    def index(name: str) -> tuple[int, str]:
        found = _DIGITS.search(name)
        return (int(found.group(1)) if found else -1, name)

    return sorted(names, key=index)


def _decode(frames) -> np.ndarray:
    """Read a camera dataset, decompressing per-frame JPEG if that is what it is."""
    array = frames[()]
    if array.dtype == np.uint8 and array.ndim >= 3:
        return array
    # Variable-length byte strings: one encoded image per frame.
    (Image,) = require("PIL.Image", extra="data")
    return np.stack([np.asarray(Image.open(io.BytesIO(bytes(item)))) for item in array])


def _pick_cameras(obs, cameras: Sequence[str] | None) -> list[str]:
    if cameras is not None:
        chosen = []
        for name in cameras:
            for candidate in (name, f"{name}_jpeg"):
                if candidate in obs:
                    chosen.append(candidate)
                    break
            else:
                raise KeyError(f"no camera {name!r} in {sorted(obs)}")
        return chosen
    found = [k for k in obs if _IMAGE_HINT.search(k) and "depth" not in k]
    return _demo_order(found)[:1]  # one camera by default; ask for more explicitly


def _pick_state_keys(obs, state_keys: Sequence[str] | None) -> list[str]:
    if state_keys is not None:
        missing = [k for k in state_keys if k not in obs]
        if missing:
            raise KeyError(f"no state key(s) {missing} in {sorted(obs)}")
        return list(state_keys)
    # Everything that is one vector per frame and not a camera.
    return sorted(k for k in obs if obs[k].ndim == 2 and not _IMAGE_HINT.search(k))


def _paths(path) -> Iterator[Path]:
    """Normalise ``path`` to a stream of files.

    Accepts a single file, a directory of ``.hdf5``, or an iterable of files --
    including a *generator*, which is what lets the registry download one shard
    at a time. LIBERO-90 is 90 separate per-task files and has no LeRobot
    conversion, so ``limit=4`` has to mean four episodes and four downloads, not
    ninety.
    """
    if isinstance(path, (str, Path)):
        resolved = Path(path)
        if resolved.is_dir():
            yield from sorted(resolved.glob("*.hdf5")) + sorted(resolved.glob("*.h5"))
        else:
            yield resolved
        return
    for item in path:
        yield Path(item)


def iter_episodes(
    path: str | Path,
    *,
    cameras: Sequence[str] | None = None,
    state_keys: Sequence[str] | None = None,
    observation: str = "both",
    limit: int | None = None,
    resize: int | None = None,
    dtype: str = "float32",
    split: str | None = None,
) -> Iterator[dict[str, np.ndarray]]:
    """Stream a LIBERO or RoboMimic file as per-episode dicts.

    Args:
        path: an ``.hdf5`` file, a directory of them, or an iterable of files.
            Several files are read in the order given and ``limit`` counts
            episodes across all of them, so an iterable that resolves lazily
            only resolves as far as ``limit`` requires.
        cameras: observation keys to use as video. ``None`` picks the first
            camera-looking key. Several are concatenated on the **channel** axis,
            since a patch embedding accepts any channel count and a second video
            field would need every model to know about it.
        state_keys: proprioception keys to concatenate. ``None`` takes every
            per-frame vector that is not an image, which for LIBERO is
            ``ee_ori``, ``ee_pos``, ``ee_states``, ``gripper_states`` and
            ``joint_states``.
        observation: ``"video"``, ``"state"`` or ``"both"``. Video is the
            expensive half; skip it when training a state encoder. ``"both"``
            means "whatever the file has" and yields state alone for a file with
            no cameras, which is what every RoboMimic ``low_dim`` release is;
            ``"video"`` on such a file is an error.
        limit: stop after this many demos, in numeric demo order.
        resize, dtype: passed to :func:`~xwm.datasets.spec.to_frames`.
        split: a key under the file's ``mask`` group, e.g. ``"train"`` or
            ``"valid"``, restricting which demos are read. RoboMimic files ship
            these; LIBERO files generally do not.
    """
    (h5py,) = require("h5py", extra="data")
    if observation not in ("video", "state", "both"):
        raise ValueError(f"observation must be 'video', 'state' or 'both', got {observation!r}")

    seen = 0
    files = _paths(path)
    # `while` before `next`, not `for` with a check inside: a for-loop pulls the
    # next path *before* it can see that `limit` is already met, and for an
    # hf:// directory pulling a path means downloading a file.
    while limit is None or seen < limit:
        try:
            one = next(files)
        except StopIteration:
            return
        for episode in _iter_file(
            h5py,
            one,
            cameras=cameras,
            state_keys=state_keys,
            observation=observation,
            limit=None if limit is None else limit - seen,
            resize=resize,
            dtype=dtype,
            split=split,
        ):
            seen += 1
            yield episode


def _iter_file(
    h5py,
    path: Path,
    *,
    cameras,
    state_keys,
    observation: str,
    limit: int | None,
    resize,
    dtype: str,
    split: str | None,
) -> Iterator[dict[str, np.ndarray]]:
    """One file's demos, in numeric order."""
    with h5py.File(str(path), "r") as handle:
        data = handle["data"] if "data" in handle else handle
        names = _demo_order(list(data))
        if split is not None:
            if "mask" not in handle:
                raise KeyError(f"{path} has no 'mask' group, so split={split!r} cannot be applied")
            allowed = {
                item.decode() if isinstance(item, bytes) else str(item)
                for item in handle["mask"][split][()]
            }
            names = [n for n in names if n in allowed]

        for count, name in enumerate(names):
            if limit is not None and count >= limit:
                return
            demo = data[name]
            obs = demo["obs"]
            following = demo.get("next_obs")

            episode: dict[str, np.ndarray] = {}
            n_frames = None
            chosen = _pick_cameras(obs, cameras) if observation != "state" else []
            if not chosen and observation == "video":
                raise KeyError(
                    f"{path} has no camera-looking observation in {sorted(obs)}. "
                    "RoboMimic distributes low-dim files with no images at all -- "
                    "its image variants are rendered locally from the recorded "
                    "simulator states."
                )
            if chosen:
                # "both" means "whatever this file has", so a file with no camera
                # yields state alone rather than failing.
                channels = []
                for camera in chosen:
                    frames = _decode(obs[camera])
                    if following is not None and camera in following:
                        frames = np.concatenate([frames, _decode(following[camera])[-1:]])
                    channels.append(to_frames(frames, resize=resize, dtype=dtype))
                video = np.concatenate(channels, axis=1) if len(channels) > 1 else channels[0]
                episode["video"] = video
                n_frames = video.shape[0]
            if observation in ("state", "both"):
                keys = _pick_state_keys(obs, state_keys)
                vectors = [np.asarray(obs[k][()], np.float32) for k in keys]
                if following is not None and all(k in following for k in keys):
                    vectors = [
                        np.concatenate([v, np.asarray(following[k][()], np.float32)[-1:]])
                        for v, k in zip(vectors, keys, strict=True)
                    ]
                state = np.concatenate([v.reshape(v.shape[0], -1) for v in vectors], axis=-1)
                episode["state"] = state
                n_frames = state.shape[0] if n_frames is None else n_frames

            actions = np.asarray(demo["actions"][()], np.float32)
            # Keep every action only if a frame for its result exists.
            episode["action"] = actions if actions.shape[0] == n_frames - 1 else actions[:-1]
            for source, target in (("rewards", "reward"), ("dones", "done")):
                if source in demo:
                    values = np.asarray(demo[source][()], np.float32).reshape(-1)
                    episode[target] = values[: episode["action"].shape[0]]
            yield episode


def load(
    path: str | Path,
    *,
    length: int = 16,
    stride: int | None = None,
    limit: int | None = None,
    max_clips: int | None = None,
    **kwargs,
) -> dict[str, np.ndarray]:
    """Load a LIBERO or RoboMimic file as fixed-length clips.

    See :func:`iter_episodes` for the arguments. LIBERO demos are 100-300 frames
    of 128x128 RGB, so a whole suite at ``dtype="float32"`` is tens of
    gigabytes; ``resize``, ``limit`` and ``dtype="uint8"`` are the three knobs
    that matter.
    """
    episodes = iter_episodes(path, limit=limit, **kwargs)
    return clips(episodes, length=length, stride=stride, max_clips=max_clips)

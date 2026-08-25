"""Open X-Embodiment in its native RLDS form, via ``tensorflow_datasets``.

The canonical OXE release is TFRecord in RLDS layout, hosted on a public GCS
bucket. This module reads it. It is deliberately the smallest reader in the
package, for two reasons.

First, the dependency. ``tensorflow_datasets`` brings TensorFlow, whose CUDA
pins fight JAX's on any GPU machine, so it is **not** in the ``data`` extra --
this module asks for it by name at call time. Second, most of OXE has been
mirrored into the LeRobot layout, which needs no TensorFlow at all and is what
:mod:`xwm.datasets.lerobot` reads; the registry points ``oxe/*`` at those
mirrors. Use this when you need a dataset that was never mirrored, or when you
want the bytes the RT-X papers were trained on rather than a conversion of them.

RLDS is nested: a dataset is episodes, an episode has ``steps``, and a step has
``observation``, ``action``, ``reward``, ``is_terminal``. Two shapes of
irregularity have to be flattened out:

* ``action`` is sometimes a tensor and sometimes a dict -- RT-1 splits it into
  ``world_vector``, ``rotation_delta`` and ``gripper_closedness_action``. Dict
  actions are concatenated in sorted key order, which is stable but arbitrary,
  so an action dimension read off one dataset does not transfer to another.
* Camera keys differ per dataset (``image``, ``rgb_static``, ``hand_image``).
  With ``cameras=None`` the first image-shaped observation is used.

References
----------
Open X-Embodiment Collaboration, *Open X-Embodiment: Robotic Learning Datasets
and RT-X Models*, ICRA 2024. arXiv:2310.08864.

Ramos et al., *RLDS: an Ecosystem to Generate, Share and Use Datasets in
Reinforcement Learning*, 2021. arXiv:2111.02767.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np

from .cache import require
from .spec import clips, to_frames

__all__ = ["OXE_BUCKET", "iter_episodes", "load"]

#: Where the Open X-Embodiment release lives.
OXE_BUCKET = "gs://gresearch/robotics"


def _flatten(value) -> np.ndarray:
    """A step's action or state as one 1-D float32 vector."""
    if isinstance(value, dict):
        parts = [np.asarray(value[key], np.float32).reshape(-1) for key in sorted(value)]
        return np.concatenate(parts)
    return np.asarray(value, np.float32).reshape(-1)


def _is_image(array) -> bool:
    array = np.asarray(array)
    return array.ndim == 3 and array.shape[-1] in (1, 3, 4)


def iter_episodes(
    name: str,
    *,
    version: str = "0.1.0",
    split: str = "train",
    data_dir: str = OXE_BUCKET,
    cameras: Sequence[str] | None = None,
    observation: str = "video",
    state_key: str = "state",
    limit: int | None = None,
    resize: int | None = None,
    dtype: str = "float32",
) -> Iterator[dict[str, np.ndarray]]:
    """Stream an RLDS dataset as per-episode dicts.

    Args:
        name: the TFDS dataset name, e.g. ``"fractal20220817_data"`` (RT-1) or
            ``"bridge"``.
        version: TFDS version directory. Most of OXE is ``0.1.0``.
        split: usually ``"train"``; OXE datasets rarely ship a test split.
        data_dir: the bucket or local directory holding ``name/version``.
        cameras: observation keys to use as video. ``None`` takes the first
            image-shaped one.
        observation: ``"video"``, ``"state"`` or ``"both"``.
        state_key: which observation key is proprioception, when asked for.
        limit: stop after this many episodes. OXE datasets run to hundreds of
            thousands, and this reader holds one episode in memory at a time
            but returns them all if you let it.
        resize, dtype: passed to :func:`~xwm.datasets.spec.to_frames`.
    """
    (tfds,) = require("tensorflow_datasets", extra="")
    builder = tfds.builder_from_directory(f"{data_dir.rstrip('/')}/{name}/{version}")
    dataset = builder.as_dataset(split=split)

    for count, record in enumerate(dataset):
        if limit is not None and count >= limit:
            return
        steps = [
            {key: value for key, value in step.items()}
            for step in tfds.as_numpy(record["steps"])
        ]
        if len(steps) < 2:
            continue

        observations = [step["observation"] for step in steps]
        episode: dict[str, np.ndarray] = {}
        if observation in ("video", "both"):
            keys = list(cameras) if cameras is not None else [
                key for key, value in observations[0].items() if _is_image(value)
            ][:1]
            if not keys:
                raise KeyError(
                    f"no image-shaped observation in {sorted(observations[0])}; "
                    "pass cameras= explicitly"
                )
            channels = [
                to_frames(
                    np.stack([np.asarray(obs[key]) for obs in observations]),
                    resize=resize,
                    dtype=dtype,
                )
                for key in keys
            ]
            episode["video"] = (
                np.concatenate(channels, axis=1) if len(channels) > 1 else channels[0]
            )
        if observation in ("state", "both") and state_key in observations[0]:
            episode["state"] = np.stack([_flatten(obs[state_key]) for obs in observations])

        # RLDS records one action per step, the last of which acts on the final
        # observation and leads nowhere in the file, so it goes.
        episode["action"] = np.stack([_flatten(step["action"]) for step in steps])[:-1]
        if "reward" in steps[0]:
            episode["reward"] = np.asarray(
                [float(np.asarray(step["reward"])) for step in steps], np.float32
            )[:-1]
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
    """Load an RLDS dataset as fixed-length clips.

    See :func:`iter_episodes`. Pass ``limit``: the default reads the whole split,
    and the whole split can be a terabyte.
    """
    episodes = iter_episodes(name, limit=limit, **kwargs)
    return clips(episodes, length=length, stride=stride, max_clips=max_clips)

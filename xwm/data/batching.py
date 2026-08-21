"""Turning arrays into batch streams.

xwm does not ship a data loader. Models consume plain dicts of arrays, so any
pipeline that produces those works -- these helpers just cover the in-memory
case that tests and examples need.
"""

from __future__ import annotations

from collections.abc import Iterator

import jax.numpy as jnp
import jax.random as jr

from ..core.types import Array, Batch, PRNGKey


def iter_batches(
    data: dict[str, Array],
    batch_size: int,
    *,
    key: PRNGKey | None = None,
    shuffle: bool = True,
    drop_last: bool = True,
    epochs: int | None = 1,
) -> Iterator[Batch]:
    """Iterate mini-batches over a dict of equally-long arrays.

    Args:
        data: field name -> array with a common leading axis.
        epochs: passes over the data; ``None`` repeats forever, which is what
            :meth:`xwm.training.Trainer.fit` wants when driven by ``steps``.
    """
    sizes = {k: v.shape[0] for k, v in data.items()}
    if len(set(sizes.values())) != 1:
        raise ValueError(f"fields disagree on length: {sizes}")
    n = next(iter(sizes.values()))
    if batch_size > n:
        raise ValueError(f"batch_size={batch_size} exceeds dataset size {n}")

    epoch = 0
    while epochs is None or epoch < epochs:
        order = jnp.arange(n)
        if shuffle:
            if key is None:
                raise ValueError("shuffle=True needs a key")
            order = jr.permutation(jr.fold_in(key, epoch), n)
        limit = n - n % batch_size if drop_last else n
        for start in range(0, limit, batch_size):
            idx = order[start : start + batch_size]
            yield {k: v[idx] for k, v in data.items()}
        epoch += 1


def clip_windows(video: Array, window: int, *, stride: int = 1) -> Array:
    """Cut ``(T, ...)`` frames into overlapping clips of ``window`` frames.

    Returns ``(n_windows, window, ...)``. Video encoders take fixed-length
    clips, so this is how a long sequence is fed to one.
    """
    t = video.shape[0]
    if window > t:
        raise ValueError(f"window={window} exceeds sequence length {t}")
    starts = jnp.arange(0, t - window + 1, stride)
    return jnp.stack([video[s : s + window] for s in starts])

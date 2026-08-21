"""Training: one loop for every world model in xwm.

:class:`Trainer` is family-agnostic -- it only needs ``loss``, ``prepare_batch``
and ``trainable``. What differs is where batches come from:
:func:`xwm.data.iter_batches` for a fixed dataset (JEPA), and
:class:`ReplayBuffer` for the reward-driven families, whose losses need
contiguous slices of a single episode.
"""

from .replay import ReplayBuffer
from .schedules import adamw, cosine_warmup, ema_momentum, weight_decay_schedule
from .state import TrainState
from .trainer import Callback, Trainer, global_norm, print_metrics

__all__ = [
    "Callback",
    "ReplayBuffer",
    "TrainState",
    "Trainer",
    "adamw",
    "cosine_warmup",
    "ema_momentum",
    "global_norm",
    "print_metrics",
    "weight_decay_schedule",
]

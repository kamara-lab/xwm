"""Learning-rate, weight-decay and EMA-momentum schedules.

JEPA training is schedule-sensitive in a way supervised training is not, because
two of the schedules here directly control the collapse dynamics: the EMA
momentum sets how fast the teacher can be dragged toward the student, and the
weight decay sets how hard the encoder is pushed toward zero.
"""

from __future__ import annotations

import optax


def cosine_warmup(
    peak: float,
    total_steps: int,
    *,
    warmup_steps: int = 0,
    init: float = 0.0,
    final: float = 0.0,
) -> optax.Schedule:
    """Linear warmup then cosine decay -- the standard ViT recipe."""
    if warmup_steps <= 0:
        return optax.cosine_decay_schedule(peak, total_steps, alpha=final / max(peak, 1e-12))
    return optax.warmup_cosine_decay_schedule(
        init_value=init,
        peak_value=peak,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        end_value=final,
    )


def ema_momentum(
    total_steps: int,
    *,
    base: float = 0.996,
    final: float = 1.0,
) -> optax.Schedule:
    """Teacher momentum, increasing toward ``final`` over training.

    Starting lower lets the teacher track the student while the representation
    is still changing fast; ending near ``1.0`` freezes it into a stable target
    once it is worth imitating. I-JEPA and V-JEPA both ramp it this way.
    """
    return optax.linear_schedule(base, final, total_steps)


def weight_decay_schedule(
    total_steps: int,
    *,
    init: float = 0.04,
    final: float = 0.4,
) -> optax.Schedule:
    """Weight decay increasing over training, as in the DINO/I-JEPA recipes."""
    return optax.linear_schedule(init, final, total_steps)


def adamw(
    learning_rate: float | optax.Schedule,
    *,
    weight_decay: float | optax.Schedule = 0.05,
    b1: float = 0.9,
    b2: float = 0.95,
    grad_clip: float | None = 1.0,
) -> optax.GradientTransformation:
    """AdamW with the defaults xwm uses for JEPA training.

    ``b2 = 0.95`` rather than 0.999: the loss is a moving target (the teacher
    moves, or the regularizer's random projections change every step), so a
    shorter second-moment window tracks it better.
    """
    tx = optax.adamw(learning_rate, b1=b1, b2=b2, weight_decay=weight_decay)
    if grad_clip is not None:
        tx = optax.chain(optax.clip_by_global_norm(grad_clip), tx)
    return tx

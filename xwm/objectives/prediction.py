"""Latent prediction losses -- the primary signal in every JEPA.

The defining property of a *predictive* (as opposed to generative) world model
is that this loss is computed in representation space. Nothing here ever touches
pixels, so the model is free to discard unpredictable detail instead of being
penalised for failing to reproduce it.

References
----------
Assran et al., *I-JEPA*, CVPR 2023. arXiv:2301.08243 -- smooth-L1 on predicted
embeddings.

Bardes et al., *V-JEPA*, 2024. arXiv:2404.08471 -- L1 on LayerNorm-ed targets,
the ``normalize_target`` option.
"""

from __future__ import annotations

from typing import Literal

import jax.numpy as jnp

from ..core.types import Array

LossKind = Literal["l1", "l2", "smooth_l1", "cosine"]


def layer_normalize(x: Array, eps: float = 1e-6) -> Array:
    """Parameter-free LayerNorm over the last axis.

    Applied to prediction *targets* it removes the scale degree of freedom that
    a teacher/student pair can otherwise exploit to shrink the loss without
    improving prediction. V-JEPA normalises targets this way.
    """
    mu = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    return (x - mu) / jnp.sqrt(var + eps)


def prediction_loss(
    pred: Array,
    target: Array,
    *,
    kind: LossKind = "l1",
    beta: float = 1.0,
    normalize_target: bool = False,
) -> Array:
    """Distance between predicted and target embeddings, averaged over everything.

    Args:
        pred: predicted embeddings, any shape ending in ``D``.
        target: same shape as ``pred``. Detach it before calling if it comes
            from a teacher -- this function does not stop gradients for you.
        kind: ``"l1"`` (V-JEPA), ``"l2"``, ``"smooth_l1"`` (I-JEPA), or
            ``"cosine"``.
        beta: transition point of the smooth-L1 / Huber loss.
        normalize_target: LayerNorm the target (and, for symmetry, the
            prediction) before comparing.

    Returns:
        A scalar.
    """
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape} vs target {target.shape}")
    if normalize_target:
        pred, target = layer_normalize(pred), layer_normalize(target)

    if kind == "l1":
        return jnp.mean(jnp.abs(pred - target))
    if kind == "l2":
        return jnp.mean(jnp.square(pred - target))
    if kind == "smooth_l1":
        d = jnp.abs(pred - target)
        return jnp.mean(jnp.where(d < beta, 0.5 * d**2 / beta, d - 0.5 * beta))
    if kind == "cosine":
        p = pred / (jnp.linalg.norm(pred, axis=-1, keepdims=True) + 1e-8)
        t = target / (jnp.linalg.norm(target, axis=-1, keepdims=True) + 1e-8)
        return jnp.mean(1.0 - jnp.sum(p * t, axis=-1))
    raise ValueError(f"unknown loss kind {kind!r}")

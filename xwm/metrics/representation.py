"""Diagnosing a representation without labels.

Self-supervised training gives you a loss that can fall while the
representation gets worse -- a partially collapsed encoder predicts its own
degenerate targets very well. These metrics are the ones to watch instead of
the loss, because they measure how much of the embedding space is actually in
use.

References
----------
Garrido et al., *RankMe: Assessing the Downstream Performance of Pretrained
Self-Supervised Representations by Their Rank*, ICML 2023. arXiv:2210.02885 --
:func:`rankme`, the smooth effective-rank measure.

Jing et al., *Understanding Dimensional Collapse in Contrastive
Self-Supervised Learning*, ICLR 2022. arXiv:2110.09348 -- the partial-collapse
failure mode these diagnostics are designed to catch.
"""

from __future__ import annotations

import jax.numpy as jnp

from ..core.types import Array


def _flatten(z: Array) -> Array:
    return z.reshape(-1, z.shape[-1])


def singular_values(z: Array) -> Array:
    """Singular values of the centred embedding matrix, descending."""
    z = _flatten(z)
    z = z - jnp.mean(z, axis=0, keepdims=True)
    return jnp.linalg.svd(z, compute_uv=False)


def rankme(z: Array, eps: float = 1e-7) -> Array:
    """RankMe: the effective rank as the entropy of the singular-value spectrum.

    ``exp(H(p))`` where ``p`` is the normalised spectrum. Equals ``D`` when every
    direction carries equal energy and ``1`` when all the energy is in one
    direction, and unlike a hard rank it responds smoothly to *partial*
    collapse -- the failure mode that actually shows up in JEPA training.

    Note that the spectrum is taken after centring, so this is blind to a
    constant offset: an encoder emitting ``c + tiny_noise`` still scores a high
    rank. :func:`feature_std` and :func:`mean_cosine_similarity` catch that
    case, which is why :func:`collapse_report` reports all three.
    """
    s = singular_values(z)
    p = s / (jnp.sum(s) + eps)
    return jnp.exp(-jnp.sum(p * jnp.log(p + eps)))


def effective_rank_ratio(z: Array) -> Array:
    """:func:`rankme` divided by the embedding width -- ``1.0`` is ideal."""
    return rankme(z) / z.shape[-1]


def feature_std(z: Array) -> Array:
    """Mean per-dimension standard deviation. Near zero means collapse."""
    return jnp.mean(jnp.std(_flatten(z), axis=0))


def mean_cosine_similarity(z: Array) -> Array:
    """Average pairwise cosine similarity between samples (excluding self).

    Approaching ``1.0`` means every input maps to nearly the same direction --
    collapse, even if the per-dimension variance still looks healthy.
    """
    z = _flatten(z)
    z = z / (jnp.linalg.norm(z, axis=-1, keepdims=True) + 1e-8)
    n = z.shape[0]
    sim = z @ z.T
    off_diagonal = jnp.sum(sim) - jnp.trace(sim)
    return off_diagonal / max(n * (n - 1), 1)


def collapse_report(z: Array) -> dict[str, Array]:
    """All of the above at once, for logging alongside the loss."""
    return {
        "rankme": rankme(z),
        "rank_ratio": effective_rank_ratio(z),
        "feature_std": feature_std(z),
        "mean_cosine": mean_cosine_similarity(z),
    }

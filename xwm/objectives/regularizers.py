"""Variance/covariance regularizers (VICReg) and the InfoNCE contrastive loss.

These are the *alternative* answers to collapse, kept here so a world model can
swap its anti-collapse term and be compared on equal footing:

* :func:`vicreg` constrains the first two moments -- per-dimension variance and
  cross-dimension covariance -- rather than the whole distribution.
* :func:`info_nce` avoids collapse by contrast with negatives, which requires
  large batches and a notion of "negative" that video rarely supplies cleanly.

Compare with :func:`xwm.objectives.sigreg`, which constrains the full
distribution and needs no negatives.

References
----------
Bardes, Ponce & LeCun, *VICReg: Variance-Invariance-Covariance Regularization
for Self-Supervised Learning*, ICLR 2022. arXiv:2105.04906

van den Oord, Li & Vinyals, *Representation Learning with Contrastive Predictive
Coding*, 2018. arXiv:1807.03748 -- InfoNCE.

Zbontar et al., *Barlow Twins: Self-Supervised Learning via Redundancy
Reduction*, ICML 2021. arXiv:2103.03230 -- the closely related
decorrelation objective.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from ..core.types import Array


def variance_loss(z: Array, gamma: float = 1.0, eps: float = 1e-4) -> Array:
    """Hinge each dimension's standard deviation up to ``gamma``.

    ``z`` is ``(..., D)``; leading axes are all treated as samples, matching
    :func:`xwm.objectives.sigreg`, so a token sequence ``(B, N, D)`` contributes
    ``B * N`` samples rather than erroring.
    """
    z = z.reshape(-1, z.shape[-1])
    std = jnp.sqrt(jnp.var(z, axis=0) + eps)
    return jnp.mean(jax.nn.relu(gamma - std))


def covariance_loss(z: Array) -> Array:
    """Penalise off-diagonal covariance, decorrelating the dimensions.

    ``z`` is ``(..., D)``; see :func:`variance_loss` on leading axes.
    """
    z = z.reshape(-1, z.shape[-1])
    n, d = z.shape
    z = z - jnp.mean(z, axis=0, keepdims=True)
    cov = (z.T @ z) / max(n - 1, 1)
    off_diagonal = cov - jnp.diag(jnp.diag(cov))
    return jnp.sum(jnp.square(off_diagonal)) / d


def vicreg(
    z_a: Array,
    z_b: Array,
    *,
    sim_coeff: float = 25.0,
    var_coeff: float = 25.0,
    cov_coeff: float = 1.0,
    gamma: float = 1.0,
) -> tuple[Array, dict[str, Array]]:
    """Variance-Invariance-Covariance regularization on two views.

    Args:
        z_a, z_b: ``(B, D)`` embeddings of two views of the same inputs.

    Returns:
        ``(loss, parts)`` where ``parts`` holds the three terms for logging.
    """
    z_a = z_a.reshape(-1, z_a.shape[-1])
    z_b = z_b.reshape(-1, z_b.shape[-1])
    invariance = jnp.mean(jnp.square(z_a - z_b))
    variance = variance_loss(z_a, gamma) + variance_loss(z_b, gamma)
    covariance = covariance_loss(z_a) + covariance_loss(z_b)
    loss = sim_coeff * invariance + var_coeff * variance + cov_coeff * covariance
    return loss, {
        "invariance": invariance,
        "variance": variance,
        "covariance": covariance,
    }


def info_nce(z_a: Array, z_b: Array, temperature: float = 0.1) -> Array:
    """Symmetric InfoNCE over in-batch negatives.

    Args:
        z_a, z_b: ``(B, D)`` embeddings; row ``i`` of each is a positive pair.
    """
    z_a = z_a / (jnp.linalg.norm(z_a, axis=-1, keepdims=True) + 1e-8)
    z_b = z_b / (jnp.linalg.norm(z_b, axis=-1, keepdims=True) + 1e-8)
    logits = (z_a @ z_b.T) / temperature
    labels = jnp.arange(z_a.shape[0])
    log_p_ab = logits - logsumexp(logits, axis=1, keepdims=True)
    log_p_ba = logits.T - logsumexp(logits.T, axis=1, keepdims=True)
    return -0.5 * (
        jnp.mean(log_p_ab[labels, labels]) + jnp.mean(log_p_ba[labels, labels])
    )

"""Scalar regression as classification: two-hot targets over a fixed bin grid.

TD-MPC2 and MuZero both predict rewards and values this way rather than with a
squared error, and it is not a detail -- it is why they are stable across tasks
whose reward scales differ by orders of magnitude.

The problem with regressing a scalar directly is that the gradient scales with
the error, so a task with rewards in the thousands produces gradients a thousand
times larger than one with rewards in single digits, and a single set of
hyperparameters cannot serve both. Turning the scalar into a distribution over
bins makes the loss a cross-entropy: bounded, scale-free, and it lets the model
represent uncertainty (a bimodal value is expressible; a point estimate is not).

Two-hot encoding places a target between the two bins it falls between, with
linear weights, so the encoding is exact and invertible for any value in range.

For values spanning several orders of magnitude, apply :func:`symlog` first --
MuZero's invertible squashing transform -- and :func:`symexp` to come back.

References
----------
Bellemare, Dabney & Munos, *A Distributional Perspective on Reinforcement
Learning* (C51), ICML 2017. arXiv:1707.06887 -- predicting a return
*distribution* over a fixed support rather than its mean.

Hafner et al., *Mastering Diverse Domains through World Models* (DreamerV3),
2023. arXiv:2301.04104 -- the symlog transform and two-hot encoding as
implemented here.

Schrittwieser et al., *MuZero*, Nature 2020. arXiv:1911.08265 -- categorical
reward and value heads with an invertible value transform.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from ..core.module import Module
from ..core.types import Array

__all__ = [
    "CategoricalScalar",
    "cross_entropy",
    "symexp",
    "symlog",
    "two_hot",
]


def symlog(x: Array) -> Array:
    """``sign(x) * log(|x| + 1)`` -- invertible, squashes large magnitudes."""
    return jnp.sign(x) * jnp.log1p(jnp.abs(x))


def symexp(x: Array) -> Array:
    """Inverse of :func:`symlog`."""
    return jnp.sign(x) * (jnp.expm1(jnp.abs(x)))


class CategoricalScalar(Module):
    """A fixed grid of bins for encoding and decoding scalars.

    Args:
        n_bins: number of bins. More bins means finer resolution and a harder
            classification problem; 101 is TD-MPC2's choice.
        low, high: range covered. Values outside are clipped, so pick a range
            that actually contains your returns.
        transform: apply :func:`symlog` before binning and :func:`symexp` after
            decoding. Use it when values span orders of magnitude.
    """

    n_bins: int = eqx.field(static=True)
    low: float = eqx.field(static=True)
    high: float = eqx.field(static=True)
    transform: bool = eqx.field(static=True)

    def __init__(
        self,
        n_bins: int = 101,
        low: float = -10.0,
        high: float = 10.0,
        *,
        transform: bool = True,
    ):
        if n_bins < 2:
            raise ValueError(f"need at least two bins, got {n_bins}")
        if not high > low:
            raise ValueError(f"high ({high}) must exceed low ({low})")
        self.n_bins = n_bins
        self.low = low
        self.high = high
        self.transform = transform

    @property
    def bins(self) -> Array:
        """``(n_bins,)`` bin centres, in transformed space."""
        return jnp.linspace(self.low, self.high, self.n_bins)

    def encode(self, value: Array) -> Array:
        """Scalar(s) -> two-hot distribution ``(..., n_bins)``."""
        x = symlog(value) if self.transform else value
        return two_hot(x, self.n_bins, self.low, self.high)

    def decode(self, logits: Array) -> Array:
        """Logits ``(..., n_bins)`` -> expected scalar under the softmax."""
        probs = jax.nn.softmax(logits, axis=-1)
        expected = jnp.sum(probs * self.bins, axis=-1)
        return symexp(expected) if self.transform else expected

    def loss(self, logits: Array, target: Array) -> Array:
        """Cross-entropy between predicted logits and the two-hot target."""
        return cross_entropy(logits, self.encode(target))


def two_hot(x: Array, n_bins: int, low: float, high: float) -> Array:
    """Two-hot encode ``x`` over ``n_bins`` uniform bins spanning ``[low, high]``.

    The value's mass is split linearly between its two neighbouring bins, so
    ``sum(bins * two_hot(x)) == clip(x, low, high)`` exactly.
    """
    x = jnp.clip(x, low, high)
    width = (high - low) / (n_bins - 1)
    position = (x - low) / width
    lower = jnp.floor(position).astype(jnp.int32)
    upper = jnp.clip(lower + 1, 0, n_bins - 1)
    lower = jnp.clip(lower, 0, n_bins - 1)
    upper_weight = position - lower
    onehot = lambda idx: jax.nn.one_hot(idx, n_bins)  # noqa: E731
    return (
        onehot(lower) * (1.0 - upper_weight)[..., None]
        + onehot(upper) * upper_weight[..., None]
    )


def cross_entropy(logits: Array, target_probs: Array) -> Array:
    """Mean cross-entropy against a (possibly soft) target distribution."""
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.mean(jnp.sum(target_probs * log_probs, axis=-1))

"""Normalisation layers.

References
----------
Ba, Kiros & Hinton, *Layer Normalization*, 2016. arXiv:1607.06450

Zhang & Sennrich, *Root Mean Square Layer Normalization*, NeurIPS 2019.
arXiv:1910.07467

Hansen, Su & Wang, *TD-MPC2*, ICLR 2024. arXiv:2310.16828 -- :class:`SimNorm`.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from ..core.module import Module
from ..core.types import Array


class LayerNorm(Module):
    """Layer normalisation over the last axis, with optional affine params."""

    weight: Array | None
    bias: Array | None
    eps: float = eqx.field(static=True)

    def __init__(self, dim: int, *, eps: float = 1e-6, affine: bool = True):
        self.weight = jnp.ones((dim,)) if affine else None
        self.bias = jnp.zeros((dim,)) if affine else None
        self.eps = eps

    def __call__(self, x: Array) -> Array:
        mu = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.var(x, axis=-1, keepdims=True)
        out = (x - mu) / jnp.sqrt(var + self.eps)
        if self.weight is not None:
            out = out * self.weight + self.bias
        return out


class RMSNorm(Module):
    """Root-mean-square normalisation (no mean subtraction)."""

    weight: Array | None
    eps: float = eqx.field(static=True)

    def __init__(self, dim: int, *, eps: float = 1e-6, affine: bool = True):
        self.weight = jnp.ones((dim,)) if affine else None
        self.eps = eps

    def __call__(self, x: Array) -> Array:
        out = x / jnp.sqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + self.eps)
        return out * self.weight if self.weight is not None else out


def l2_normalize(x: Array, axis: int = -1, eps: float = 1e-8) -> Array:
    """Project onto the unit sphere along ``axis``."""
    return x / (jnp.linalg.norm(x, axis=axis, keepdims=True) + eps)


class SimNorm(Module):
    """Simplicial normalisation: softmax over fixed-size groups of channels.

    TD-MPC2's latent normalisation, and the detail its stability rests on. The
    latent is split into groups of ``group_size`` channels and each group is
    softmaxed, so the latent becomes a concatenation of points on probability
    simplices.

    Why that helps where LayerNorm does not: the representation is bounded (every
    entry in ``[0, 1]``, every group summing to one), so a long recurrent unroll
    cannot drift to infinity -- but it also cannot collapse to a single point,
    because each group must keep its mass distributed to stay off the simplex
    corners. Bounded without being degenerate is exactly what a model that gets
    unrolled inside a planner needs.
    """

    group_size: int = eqx.field(static=True)

    def __init__(self, group_size: int = 8):
        if group_size < 2:
            raise ValueError(f"group_size must be at least 2, got {group_size}")
        self.group_size = group_size

    def __call__(self, x: Array) -> Array:
        if x.shape[-1] % self.group_size:
            raise ValueError(
                f"last axis {x.shape[-1]} is not divisible by group_size {self.group_size}"
            )
        grouped = x.reshape(*x.shape[:-1], -1, self.group_size)
        return jax.nn.softmax(grouped, axis=-1).reshape(x.shape)

"""Feed-forward blocks.

References
----------
Shazeer, *GLU Variants Improve Transformer*, 2020. arXiv:2002.05202 --
:class:`SwiGLU`.

Misra, *Mish: A Self Regularized Non-Monotonic Activation Function*, BMVC 2020.
arXiv:1908.08681 -- the activation the reward-driven families use.
"""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.random as jr

from ..core.module import Module
from ..core.random import resolve_key
from ..core.types import Array, PRNGKey


class Mlp(Module):
    """Two-layer position-wise MLP, applied to each token independently."""

    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear
    drop: eqx.nn.Dropout
    act: Callable[[Array], Array] = eqx.field(static=True)

    def __init__(
        self,
        dim: int,
        hidden_dim: int | None = None,
        out_dim: int | None = None,
        *,
        key: PRNGKey | None = None,
        act: Callable[[Array], Array] = jax.nn.gelu,
        dropout: float = 0.0,
    ):
        key = resolve_key(key)
        hidden_dim = hidden_dim or 4 * dim
        out_dim = out_dim or dim
        k1, k2 = jr.split(key)
        self.fc1 = eqx.nn.Linear(dim, hidden_dim, key=k1)
        self.fc2 = eqx.nn.Linear(hidden_dim, out_dim, key=k2)
        self.act = act
        self.drop = eqx.nn.Dropout(dropout)

    def __call__(self, x: Array, *, key: PRNGKey | None = None) -> Array:
        """``x``: ``(N, D)`` tokens (or a single ``(D,)`` vector)."""
        k1, k2 = (None, None) if key is None else jr.split(key)
        h = jax.vmap(self.fc1)(x) if x.ndim == 2 else self.fc1(x)
        h = self.drop(self.act(h), key=k1)
        out = jax.vmap(self.fc2)(h) if h.ndim == 2 else self.fc2(h)
        return self.drop(out, key=k2)


class SwiGLU(Module):
    """Gated feed-forward (SwiGLU), a common drop-in for :class:`Mlp`."""

    w_in: eqx.nn.Linear
    w_gate: eqx.nn.Linear
    w_out: eqx.nn.Linear
    drop: eqx.nn.Dropout

    def __init__(
        self,
        dim: int,
        hidden_dim: int | None = None,
        out_dim: int | None = None,
        *,
        key: PRNGKey | None = None,
        dropout: float = 0.0,
    ):
        # 2/3 keeps the parameter count comparable to a 4x dense MLP.
        key = resolve_key(key)
        hidden_dim = hidden_dim or int(8 * dim / 3)
        out_dim = out_dim or dim
        k1, k2, k3 = jr.split(key, 3)
        self.w_in = eqx.nn.Linear(dim, hidden_dim, key=k1)
        self.w_gate = eqx.nn.Linear(dim, hidden_dim, key=k2)
        self.w_out = eqx.nn.Linear(hidden_dim, out_dim, key=k3)
        self.drop = eqx.nn.Dropout(dropout)

    def __call__(self, x: Array, *, key: PRNGKey | None = None) -> Array:
        h = jax.vmap(self.w_in)(x) * jax.nn.silu(jax.vmap(self.w_gate)(x))
        return self.drop(jax.vmap(self.w_out)(h), key=key)

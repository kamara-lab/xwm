"""Self- and cross-attention over unbatched token sequences ``(N, D)``."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from ..core.module import Module
from ..core.random import resolve_key
from ..core.types import Array, PRNGKey
from .embed import AxialRoPE
from .norm import LayerNorm


def _split_heads(x: Array, num_heads: int) -> Array:
    """``(N, D) -> (heads, N, head_dim)``."""
    n, d = x.shape
    return x.reshape(n, num_heads, d // num_heads).transpose(1, 0, 2)


def _merge_heads(x: Array) -> Array:
    """``(heads, N, head_dim) -> (N, D)``."""
    h, n, hd = x.shape
    return x.transpose(1, 0, 2).reshape(n, h * hd)


def attend(q: Array, k: Array, v: Array, mask: Array | None = None) -> Array:
    """Scaled dot-product attention on ``(heads, N, head_dim)`` inputs.

    ``mask`` is a boolean ``(N_q, N_kv)`` array where ``True`` means *attend*.
    Delegates to :func:`jax.nn.dot_product_attention` so the fused backends are
    used where available.
    """
    q_, k_, v_ = (x.transpose(1, 0, 2)[None] for x in (q, k, v))  # (1, N, H, Dh)
    m = None if mask is None else mask[None, None]  # (1, 1, Nq, Nkv)
    out = jax.nn.dot_product_attention(q_, k_, v_, mask=m)
    return out[0].transpose(1, 0, 2)


class Attention(Module):
    """Multi-head attention with optional QK-norm and axial RoPE.

    Set ``rope`` to an :class:`~xwm.nn.embed.AxialRoPE` to use relative
    positions; leave it ``None`` and add an absolute table to the inputs
    instead (what I-JEPA and V-JEPA do).
    """

    qkv: eqx.nn.Linear
    proj: eqx.nn.Linear
    q_norm: LayerNorm | None
    k_norm: LayerNorm | None
    rope: AxialRoPE | None
    drop: eqx.nn.Dropout
    num_heads: int = eqx.field(static=True)

    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        key: PRNGKey | None = None,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        dropout: float = 0.0,
        rope: AxialRoPE | None = None,
    ):
        key = resolve_key(key)
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        k1, k2 = jr.split(key)
        head_dim = dim // num_heads
        self.num_heads = num_heads
        self.qkv = eqx.nn.Linear(dim, 3 * dim, use_bias=qkv_bias, key=k1)
        self.proj = eqx.nn.Linear(dim, dim, key=k2)
        self.q_norm = LayerNorm(head_dim) if qk_norm else None
        self.k_norm = LayerNorm(head_dim) if qk_norm else None
        self.rope = rope
        self.drop = eqx.nn.Dropout(dropout)

    def __call__(
        self,
        x: Array,
        *,
        idx: Array | None = None,
        mask: Array | None = None,
        key: PRNGKey | None = None,
    ) -> Array:
        """``x``: ``(N, D)``. ``idx``: grid positions of each token, for RoPE."""
        qkv = jax.vmap(self.qkv)(x)  # (N, 3D)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        q, k, v = (_split_heads(t, self.num_heads) for t in (q, k, v))
        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        if self.rope is not None:
            q, k = self.rope(q, idx), self.rope(k, idx)
        out = _merge_heads(attend(q, k, v, mask))
        return self.drop(jax.vmap(self.proj)(out), key=key)


class CrossAttention(Module):
    """Attention from a query sequence into a separate key/value sequence.

    Used by the attentive probe (a learned query pools an encoder's tokens) and
    by predictors that keep context tokens read-only.
    """

    to_q: eqx.nn.Linear
    to_kv: eqx.nn.Linear
    proj: eqx.nn.Linear
    q_norm: LayerNorm | None
    k_norm: LayerNorm | None
    drop: eqx.nn.Dropout
    num_heads: int = eqx.field(static=True)

    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        key: PRNGKey | None = None,
        kv_dim: int | None = None,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        dropout: float = 0.0,
    ):
        key = resolve_key(key)
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        kv_dim = kv_dim or dim
        k1, k2, k3 = jr.split(key, 3)
        head_dim = dim // num_heads
        self.num_heads = num_heads
        self.to_q = eqx.nn.Linear(dim, dim, use_bias=qkv_bias, key=k1)
        self.to_kv = eqx.nn.Linear(kv_dim, 2 * dim, use_bias=qkv_bias, key=k2)
        self.proj = eqx.nn.Linear(dim, dim, key=k3)
        self.q_norm = LayerNorm(head_dim) if qk_norm else None
        self.k_norm = LayerNorm(head_dim) if qk_norm else None
        self.drop = eqx.nn.Dropout(dropout)

    def __call__(
        self,
        q_tokens: Array,
        kv_tokens: Array,
        *,
        mask: Array | None = None,
        key: PRNGKey | None = None,
    ) -> Array:
        q = _split_heads(jax.vmap(self.to_q)(q_tokens), self.num_heads)
        kv = jax.vmap(self.to_kv)(kv_tokens)
        k, v = (_split_heads(t, self.num_heads) for t in jnp.split(kv, 2, axis=-1))
        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        out = _merge_heads(attend(q, k, v, mask))
        return self.drop(jax.vmap(self.proj)(out), key=key)


def causal_attention_mask(n: int) -> Array:
    """Boolean ``(n, n)`` mask allowing each position to see itself and the past."""
    return jnp.tril(jnp.ones((n, n), dtype=bool))


def block_causal_attention_mask(n_blocks: int, block_size: int) -> Array:
    """Causal across blocks, fully connected within a block.

    This is the mask an action-conditioned video predictor wants: all tokens of
    a frame see each other, and frames only see the past.
    """
    b = jnp.arange(n_blocks * block_size) // block_size
    return b[:, None] >= b[None, :]

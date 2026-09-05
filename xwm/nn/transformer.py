"""Pre-norm transformer blocks and stacks.

References
----------
Vaswani et al., *Attention Is All You Need*, NeurIPS 2017. arXiv:1706.03762

Dosovitskiy et al., *An Image is Worth 16x16 Words* (ViT), ICLR 2021.
arXiv:2010.11929

Touvron et al., *Going Deeper with Image Transformers* (CaiT), ICCV 2021.
arXiv:2103.17239 -- :class:`LayerScale`.

Huang et al., *Deep Networks with Stochastic Depth*, ECCV 2016.
arXiv:1603.09382 -- the drop-path schedule.

Chen et al., *Training Deep Nets with Sublinear Memory Cost*, 2016.
arXiv:1604.06174 -- the ``remat`` option.
"""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from ..core.module import Module
from ..core.random import resolve_key
from ..core.types import Array, PRNGKey
from .attention import Attention, CrossAttention
from .drop import DropPath, linear_droppath_schedule
from .embed import AxialRoPE
from .mlp import Mlp
from .norm import LayerNorm


class LayerScale(Module):
    """Per-channel learnable residual gain, initialised near zero.

    Stabilises deep ViTs by starting each residual branch as a near-no-op.
    """

    gamma: Array

    def __init__(self, dim: int, init: float = 1e-4):
        self.gamma = jnp.full((dim,), init)

    def __call__(self, x: Array) -> Array:
        return x * self.gamma


class Block(Module):
    """Pre-norm self-attention block: ``x + attn(ln(x))``, ``x + mlp(ln(x))``."""

    norm1: LayerNorm
    attn: Attention
    norm2: LayerNorm
    mlp: Mlp
    ls1: LayerScale | None
    ls2: LayerScale | None
    dp1: DropPath
    dp2: DropPath

    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        key: PRNGKey | None = None,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        dropout: float = 0.0,
        drop_path: float = 0.0,
        layer_scale: float | None = None,
        rope: AxialRoPE | None = None,
        act: Callable[[Array], Array] = jax.nn.gelu,
    ):
        key = resolve_key(key)
        ka, km = jr.split(key)
        self.norm1 = LayerNorm(dim)
        self.attn = Attention(
            dim, num_heads, key=ka, qkv_bias=qkv_bias, qk_norm=qk_norm, dropout=dropout, rope=rope
        )
        self.norm2 = LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), key=km, act=act, dropout=dropout)
        self.ls1 = LayerScale(dim, layer_scale) if layer_scale else None
        self.ls2 = LayerScale(dim, layer_scale) if layer_scale else None
        self.dp1 = DropPath(drop_path)
        self.dp2 = DropPath(drop_path)

    def __call__(
        self,
        x: Array,
        *,
        idx: Array | None = None,
        mask: Array | None = None,
        key: PRNGKey | None = None,
    ) -> Array:
        keys = (None,) * 4 if key is None else jr.split(key, 4)
        h = self.attn(self.norm1(x), idx=idx, mask=mask, key=keys[0])
        if self.ls1 is not None:
            h = self.ls1(h)
        x = x + self.dp1(h, key=keys[1])
        h = self.mlp(self.norm2(x), key=keys[2])
        if self.ls2 is not None:
            h = self.ls2(h)
        return x + self.dp2(h, key=keys[3])


class CrossBlock(Module):
    """Pre-norm cross-attention block, then an MLP."""

    norm_q: LayerNorm
    norm_kv: LayerNorm
    attn: CrossAttention
    norm2: LayerNorm
    mlp: Mlp

    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        key: PRNGKey | None = None,
        kv_dim: int | None = None,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        key = resolve_key(key)
        ka, km = jr.split(key)
        self.norm_q = LayerNorm(dim)
        self.norm_kv = LayerNorm(kv_dim or dim)
        self.attn = CrossAttention(dim, num_heads, key=ka, kv_dim=kv_dim, dropout=dropout)
        self.norm2 = LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), key=km, dropout=dropout)

    def __call__(
        self,
        q_tokens: Array,
        kv_tokens: Array,
        *,
        mask: Array | None = None,
        key: PRNGKey | None = None,
    ) -> Array:
        k1, k2 = (None, None) if key is None else jr.split(key)
        q = q_tokens + self.attn(self.norm_q(q_tokens), self.norm_kv(kv_tokens), mask=mask, key=k1)
        return q + self.mlp(self.norm2(q), key=k2)


class Transformer(Module):
    """A stack of :class:`Block` s with a final norm.

    Args:
        depth: number of blocks.
        drop_path: maximum stochastic-depth rate; rates increase linearly with
            depth, as in the ViT/DeiT recipe.
        act: the MLP nonlinearity. The default is JAX's ``gelu``, which is the
            *tanh approximation*; pass ``partial(jax.nn.gelu,
            approximate=False)`` to match a checkpoint trained under PyTorch's
            exact one (see :class:`xwm.encoders.DINOv2Encoder`).
        remat: wrap each block in :func:`jax.checkpoint`, trading recomputation
            for activation memory. Worth enabling for long video sequences.
    """

    blocks: list[Block]
    norm: LayerNorm | None
    remat: bool = eqx.field(static=True)

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        *,
        key: PRNGKey | None = None,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        dropout: float = 0.0,
        drop_path: float = 0.0,
        layer_scale: float | None = None,
        rope: AxialRoPE | None = None,
        act: Callable[[Array], Array] = jax.nn.gelu,
        final_norm: bool = True,
        remat: bool = False,
    ):
        key = resolve_key(key)
        rates = linear_droppath_schedule(depth, drop_path)
        keys = jr.split(key, depth)
        self.blocks = [
            Block(
                dim,
                num_heads,
                key=keys[i],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                dropout=dropout,
                drop_path=rates[i],
                layer_scale=layer_scale,
                rope=rope,
                act=act,
            )
            for i in range(depth)
        ]
        self.norm = LayerNorm(dim) if final_norm else None
        self.remat = remat

    @property
    def depth(self) -> int:
        return len(self.blocks)

    def __call__(
        self,
        x: Array,
        *,
        idx: Array | None = None,
        mask: Array | None = None,
        key: PRNGKey | None = None,
        return_hidden: bool = False,
    ) -> Array | tuple[Array, list[Array]]:
        keys = [None] * self.depth if key is None else list(jr.split(key, self.depth))
        hidden: list[Array] = []
        for block, k in zip(self.blocks, keys, strict=True):
            call = eqx.filter_checkpoint(block) if self.remat else block
            x = call(x, idx=idx, mask=mask, key=k)
            if return_hidden:
                hidden.append(x)
        if self.norm is not None:
            x = self.norm(x)
        return (x, hidden) if return_hidden else x


class AttentivePooler(Module):
    """Pool a token sequence into ``n_queries`` vectors via cross-attention.

    The standard read-out head for frozen JEPA encoders: a learned query
    attends over the tokens, which recovers substantially more of the
    representation than mean-pooling for the same probe budget.
    """

    query: Array
    blocks: list[CrossBlock]
    norm: LayerNorm

    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        key: PRNGKey | None = None,
        kv_dim: int | None = None,
        n_queries: int = 1,
        depth: int = 1,
        mlp_ratio: float = 4.0,
    ):
        key = resolve_key(key)
        kq, *kb = jr.split(key, depth + 1)
        self.query = 0.02 * jr.normal(kq, (n_queries, dim))
        self.blocks = [
            CrossBlock(dim, num_heads, key=kb[i], kv_dim=kv_dim, mlp_ratio=mlp_ratio)
            for i in range(depth)
        ]
        self.norm = LayerNorm(dim)

    def __call__(self, tokens: Array, *, key: PRNGKey | None = None) -> Array:
        """``tokens``: ``(N, D_kv)``. Returns ``(n_queries, D)``."""
        keys = [None] * len(self.blocks) if key is None else list(jr.split(key, len(self.blocks)))
        q = self.query
        for block, k in zip(self.blocks, keys, strict=True):
            q = block(q, tokens, key=k)
        return self.norm(q)


def mean_pool(tokens: Array) -> Array:
    """Average over the token axis of an ``(N, D)`` sequence."""
    return jnp.mean(tokens, axis=0)

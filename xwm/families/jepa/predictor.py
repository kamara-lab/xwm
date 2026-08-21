"""The JEPA predictor: predict embeddings at positions it has not seen."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from ...core.module import Module
from ...core.random import resolve_key
from ...core.types import Array, PRNGKey
from ...encoders.vision import PosKind, make_pos
from ...nn.embed import AxialRoPE, LearnedPosEmbed, SinCosPosEmbed
from ...nn.transformer import Transformer

__all__ = ["JEPAPredictor"]


class JEPAPredictor(Module):
    """Predict embeddings *at positions it has not seen*, given context tokens.

    This is the module that makes a JEPA predictive rather than merely
    contrastive. Context tokens are projected into a narrower predictor width,
    concatenated with one learned mask token per target position -- each carrying
    the target's positional embedding, and nothing else about its content -- and
    the stack attends over the whole sequence. Reading out the mask-token
    positions gives the prediction.

    Keeping the predictor narrower than the encoder (``pred_dim < embed_dim``)
    is deliberate: a predictor strong enough to model detail removes the
    pressure on the encoder to make its representation predictable.

    Args:
        grid: token grid the positions index into.
        embed_dim: encoder width (both the input and the output width).
        pred_dim: internal predictor width.
        n_mask_tokens: distinct learned mask tokens. Using one per target block
            lets the predictor tell concurrent prediction problems apart.
    """

    embed_in: eqx.nn.Linear
    embed_out: eqx.nn.Linear
    mask_tokens: Array
    pos_embed: SinCosPosEmbed | LearnedPosEmbed | None
    rope: AxialRoPE | None
    blocks: Transformer
    embed_dim: int = eqx.field(static=True)
    pred_dim: int = eqx.field(static=True)
    grid: tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self,
        grid: tuple[int, ...],
        embed_dim: int,
        *,
        pred_dim: int,
        depth: int,
        num_heads: int,
        key: PRNGKey | None = None,
        n_mask_tokens: int = 1,
        pos: PosKind = "sincos",
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        drop_path: float = 0.0,
        layer_scale: float | None = None,
        remat: bool = False,
    ):
        key = resolve_key(key)
        k_in, k_out, k_mask, k_pos, k_blocks = jr.split(key, 5)
        self.embed_in = eqx.nn.Linear(embed_dim, pred_dim, key=k_in)
        self.embed_out = eqx.nn.Linear(pred_dim, embed_dim, key=k_out)
        self.mask_tokens = 0.02 * jr.normal(k_mask, (n_mask_tokens, pred_dim))
        self.pos_embed, self.rope = make_pos(pos, tuple(grid), pred_dim, num_heads, key=k_pos)
        self.blocks = Transformer(
            pred_dim,
            depth,
            num_heads,
            key=k_blocks,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            drop_path=drop_path,
            layer_scale=layer_scale,
            rope=self.rope,
            remat=remat,
        )
        self.embed_dim = embed_dim
        self.pred_dim = pred_dim
        self.grid = tuple(grid)

    def __call__(
        self,
        context: Array,
        context_idx: Array,
        target_idx: Array,
        *,
        mask_index: Array | int = 0,
        key: PRNGKey | None = None,
    ) -> Array:
        """Predict the embeddings at ``target_idx``.

        Args:
            context: ``(K_ctx, embed_dim)`` encoder output for visible tokens.
            context_idx: ``(K_ctx,)`` grid positions of those tokens.
            target_idx: ``(K_tgt,)`` grid positions to predict.
            mask_index: which learned mask token to use.

        Returns:
            ``(K_tgt, embed_dim)``.
        """
        ctx = jax.vmap(self.embed_in)(context)
        tgt = jnp.broadcast_to(
            self.mask_tokens[mask_index], (target_idx.shape[0], self.pred_dim)
        )
        if self.pos_embed is not None:
            ctx = ctx + self.pos_embed(context_idx)
            tgt = tgt + self.pos_embed(target_idx)
        idx = jnp.concatenate([context_idx, target_idx]) if self.rope is not None else None
        h = self.blocks(jnp.concatenate([ctx, tgt], axis=0), idx=idx, key=key)
        return jax.vmap(self.embed_out)(h[context.shape[0] :])

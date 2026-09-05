"""The vision encoder every family shares: tokenise, add position, transform.

Images and video differ only in how pixels become tokens. Once a clip is a flat
``(N, D)`` sequence with an n-dimensional grid behind it, the transformer, the
positional scheme and the masking logic are identical -- so both live here, and
:mod:`xwm.encoders.image` / :mod:`xwm.encoders.video` only supply the tokeniser.

References
----------
Dosovitskiy et al., *An Image is Worth 16x16 Words* (ViT), ICLR 2021.
arXiv:2010.11929

He et al., *Masked Autoencoders Are Scalable Vision Learners* (MAE), CVPR 2022.
arXiv:2111.06377 -- encoding only the visible subset of tokens, which is where
the ``keep`` argument's speedup comes from.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from ..core.module import Module
from ..core.random import resolve_key
from ..core.types import Array, PRNGKey
from ..nn.embed import AxialRoPE, LearnedPosEmbed, SinCosPosEmbed
from ..nn.patch import PatchEmbed2d, PatchEmbed3d
from ..nn.transformer import Transformer

PosKind = Literal["sincos", "learned", "rope", "none"]
PatchEmbed = PatchEmbed2d | PatchEmbed3d

__all__ = ["PatchEmbed", "PosKind", "VisionEncoder", "make_pos"]


def make_pos(
    kind: PosKind,
    grid: tuple[int, ...],
    dim: int,
    num_heads: int,
    *,
    key: PRNGKey | None = None,
) -> tuple[SinCosPosEmbed | LearnedPosEmbed | None, AxialRoPE | None]:
    """Build the additive table and/or the rotary embedding for ``kind``."""
    n_tokens = int(jnp.prod(jnp.asarray(grid)))
    if kind == "sincos":
        return SinCosPosEmbed(grid, dim), None
    if kind == "learned":
        return LearnedPosEmbed(n_tokens, dim, key=key), None
    if kind == "rope":
        return None, AxialRoPE(grid, dim // num_heads)
    if kind == "none":
        return None, None
    raise ValueError(f"unknown positional kind {kind!r}")


class VisionEncoder(Module):
    """Tokenise, add position, run a transformer. Returns ``(N, D)`` tokens.

    The one feature that separates this from a stock ViT is ``keep``: the
    encoder can be run on an arbitrary *subset* of the token grid. A JEPA
    context encoder sees only visible tokens, so masked positions cost nothing
    to compute -- which is where most of the training speedup over pixel
    reconstruction comes from.

    There is deliberately no ``[CLS]`` token. Predictive world models are
    trained with no global objective to attach one to; pool the tokens instead
    (:func:`xwm.nn.mean_pool` or :class:`xwm.nn.AttentivePooler`).
    """

    patch_embed: PatchEmbed
    pos_embed: SinCosPosEmbed | LearnedPosEmbed | None
    rope: AxialRoPE | None
    blocks: Transformer
    embed_dim: int = eqx.field(static=True)
    grid: tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self,
        patch_embed: PatchEmbed,
        *,
        depth: int,
        num_heads: int,
        key: PRNGKey | None = None,
        pos: PosKind = "sincos",
        mlp_ratio: float = 4.0,
        qk_norm: bool = False,
        dropout: float = 0.0,
        drop_path: float = 0.0,
        layer_scale: float | None = None,
        act: Callable[[Array], Array] = jax.nn.gelu,
        remat: bool = False,
    ):
        key = resolve_key(key)
        k_pos, k_blocks = jr.split(key)
        dim = patch_embed.embed_dim
        grid = tuple(patch_embed.grid)
        self.patch_embed = patch_embed
        self.pos_embed, self.rope = make_pos(pos, grid, dim, num_heads, key=k_pos)
        self.blocks = Transformer(
            dim,
            depth,
            num_heads,
            key=k_blocks,
            mlp_ratio=mlp_ratio,
            qk_norm=qk_norm,
            dropout=dropout,
            drop_path=drop_path,
            layer_scale=layer_scale,
            rope=self.rope,
            act=act,
            remat=remat,
        )
        self.embed_dim = dim
        self.grid = grid

    @property
    def n_tokens(self) -> int:
        return self.patch_embed.n_patches

    def __call__(
        self,
        x: Array,
        *,
        keep: Array | None = None,
        key: PRNGKey | None = None,
    ) -> Array:
        """Encode one sample.

        Args:
            x: ``(C, H, W)`` image or ``(T, C, H, W)`` clip.
            keep: ``(K,)`` flat token indices to encode; ``None`` encodes all.
            key: RNG for dropout / drop-path.

        Returns:
            ``(K, D)`` tokens, aligned with ``keep``.
        """
        tokens = self.patch_embed(x)
        if self.pos_embed is not None:
            tokens = tokens + self.pos_embed()
        if keep is not None:
            tokens = tokens[keep]
        return self.blocks(tokens, idx=keep, key=key)

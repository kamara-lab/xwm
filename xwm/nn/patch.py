"""Patch and tubelet embeddings -- the entry point from pixels to tokens.

References
----------
Dosovitskiy et al., *An Image is Worth 16x16 Words* (ViT), ICLR 2021.
arXiv:2010.11929 -- patch embedding.

Arnab et al., *ViViT: A Video Vision Transformer*, ICCV 2021. arXiv:2103.15691 --
tubelet embedding, the space-time tokenisation in :class:`PatchEmbed3d`.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from einops import rearrange

from ..core.module import Module
from ..core.random import resolve_key
from ..core.types import Array, PRNGKey


class PatchEmbed2d(Module):
    """Split an image into non-overlapping patches and linearly embed them.

    Implemented as a reshape plus a matmul rather than a strided convolution:
    identical arithmetic, but it makes the token ordering explicit (row-major
    over ``(H // p, W // p)``), which the masking code depends on.
    """

    weight: Array
    bias: Array | None
    img_size: tuple[int, int] = eqx.field(static=True)
    patch_size: int = eqx.field(static=True)
    in_channels: int = eqx.field(static=True)
    embed_dim: int = eqx.field(static=True)

    def __init__(
        self,
        img_size: int | tuple[int, int],
        patch_size: int,
        in_channels: int,
        embed_dim: int,
        *,
        key: PRNGKey | None = None,
        bias: bool = True,
    ):
        key = resolve_key(key)
        h, w = (img_size, img_size) if isinstance(img_size, int) else img_size
        if h % patch_size or w % patch_size:
            raise ValueError(f"img_size {(h, w)} not divisible by patch_size {patch_size}")
        fan_in = in_channels * patch_size * patch_size
        k1, k2 = jr.split(key)
        self.weight = jr.normal(k1, (fan_in, embed_dim)) * fan_in**-0.5
        self.bias = jnp.zeros((embed_dim,)) if bias else None
        self.img_size = (h, w)
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

    @property
    def grid(self) -> tuple[int, int]:
        h, w = self.img_size
        return (h // self.patch_size, w // self.patch_size)

    @property
    def n_patches(self) -> int:
        gh, gw = self.grid
        return gh * gw

    def __call__(self, x: Array) -> Array:
        """``x``: ``(C, H, W)``. Returns ``(N, D)`` with ``N = gh * gw``."""
        p = self.patch_size
        tokens = rearrange(x, "c (gh p1) (gw p2) -> (gh gw) (c p1 p2)", p1=p, p2=p)
        out = tokens @ self.weight
        return out if self.bias is None else out + self.bias


class PatchEmbed3d(Module):
    """Tubelet embedding: patches spanning ``tubelet_size`` frames.

    Video JEPAs tokenise space *and* time jointly, so a clip of ``T`` frames at
    ``tubelet_size = 2`` yields ``T // 2`` temporal positions. Token order is
    row-major over ``(T // ts, H // p, W // p)``.
    """

    weight: Array
    bias: Array | None
    img_size: tuple[int, int] = eqx.field(static=True)
    num_frames: int = eqx.field(static=True)
    patch_size: int = eqx.field(static=True)
    tubelet_size: int = eqx.field(static=True)
    in_channels: int = eqx.field(static=True)
    embed_dim: int = eqx.field(static=True)

    def __init__(
        self,
        img_size: int | tuple[int, int],
        patch_size: int,
        num_frames: int,
        tubelet_size: int,
        in_channels: int,
        embed_dim: int,
        *,
        key: PRNGKey | None = None,
        bias: bool = True,
    ):
        key = resolve_key(key)
        h, w = (img_size, img_size) if isinstance(img_size, int) else img_size
        if h % patch_size or w % patch_size:
            raise ValueError(f"img_size {(h, w)} not divisible by patch_size {patch_size}")
        if num_frames % tubelet_size:
            raise ValueError(f"num_frames {num_frames} not divisible by tubelet {tubelet_size}")
        fan_in = in_channels * tubelet_size * patch_size * patch_size
        k1, _ = jr.split(key)
        self.weight = jr.normal(k1, (fan_in, embed_dim)) * fan_in**-0.5
        self.bias = jnp.zeros((embed_dim,)) if bias else None
        self.img_size = (h, w)
        self.num_frames = num_frames
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

    @property
    def grid(self) -> tuple[int, int, int]:
        h, w = self.img_size
        return (
            self.num_frames // self.tubelet_size,
            h // self.patch_size,
            w // self.patch_size,
        )

    @property
    def n_patches(self) -> int:
        gt, gh, gw = self.grid
        return gt * gh * gw

    def __call__(self, x: Array) -> Array:
        """``x``: ``(T, C, H, W)``. Returns ``(N, D)``."""
        p, ts = self.patch_size, self.tubelet_size
        tokens = rearrange(
            x,
            "(gt t) c (gh p1) (gw p2) -> (gt gh gw) (c t p1 p2)",
            t=ts,
            p1=p,
            p2=p,
        )
        out = tokens @ self.weight
        return out if self.bias is None else out + self.bias


def unpatchify_2d(tokens: Array, grid: tuple[int, int], patch_size: int, channels: int) -> Array:
    """Fold ``(N, C * p * p)`` patch values back into a ``(C, H, W)`` image.

    Only needed for visualisation and for pixel-space baselines -- predictive
    world models in xwm never reconstruct pixels during training.
    """
    gh, gw = grid
    return rearrange(
        tokens,
        "(gh gw) (c p1 p2) -> c (gh p1) (gw p2)",
        gh=gh,
        gw=gw,
        c=channels,
        p1=patch_size,
        p2=patch_size,
    )

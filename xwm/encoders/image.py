"""Image encoders."""

from __future__ import annotations

from ..core.random import resolve_key
from ..core.types import PRNGKey
from ..nn.patch import PatchEmbed2d
from .presets import preset
from .vision import PosKind, VisionEncoder


class ImageEncoder(VisionEncoder):
    """A ViT over 2-D patches, maskable via ``keep``.

    Args:
        img_size: input resolution, ``int`` or ``(H, W)``.
        patch_size: side length of a square patch.
        in_channels: input channels.
        embed_dim: token width.
        depth: transformer blocks.
        num_heads: attention heads.
        pos: positional scheme -- ``"sincos"``, ``"learned"``, ``"rope"``.
    """

    def __init__(
        self,
        *,
        key: PRNGKey | None = None,
        img_size: int | tuple[int, int] = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        pos: PosKind = "sincos",
        **kwargs,
    ):
        key = resolve_key(key)
        import jax.random as jr

        k_patch, k_rest = jr.split(key)
        patch_embed = PatchEmbed2d(img_size, patch_size, in_channels, embed_dim, key=k_patch)
        super().__init__(
            patch_embed, depth=depth, num_heads=num_heads, key=k_rest, pos=pos, **kwargs
        )


def image_encoder(size: str = "small", **kwargs) -> ImageEncoder:
    """Build an :class:`ImageEncoder` from a size preset.

    >>> enc = image_encoder("base", img_size=224, patch_size=16, key=key)
    """
    return ImageEncoder(**{**preset(size), **kwargs})

"""Video encoders."""

from __future__ import annotations

from ..core.random import resolve_key
from ..core.types import PRNGKey
from ..nn.patch import PatchEmbed3d
from .presets import preset
from .vision import PosKind, VisionEncoder


class VideoEncoder(VisionEncoder):
    """A ViT over space-time tubelets, maskable via ``keep``.

    Tokenising time jointly with space (``tubelet_size > 1``) is what makes the
    token count tractable for clips, and it is what makes tube masking a
    non-trivial prediction problem rather than a frame-copy.

    Args:
        img_size: spatial resolution, ``int`` or ``(H, W)``.
        num_frames: clip length in frames.
        tubelet_size: frames per token; ``num_frames`` must be divisible by it.
    """

    def __init__(
        self,
        *,
        key: PRNGKey | None = None,
        img_size: int | tuple[int, int] = 224,
        patch_size: int = 16,
        num_frames: int = 16,
        tubelet_size: int = 2,
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
        patch_embed = PatchEmbed3d(
            img_size, patch_size, num_frames, tubelet_size, in_channels, embed_dim, key=k_patch
        )
        super().__init__(
            patch_embed, depth=depth, num_heads=num_heads, key=k_rest, pos=pos, **kwargs
        )


def video_encoder(size: str = "small", **kwargs) -> VideoEncoder:
    """Build a :class:`VideoEncoder` from a size preset."""
    return VideoEncoder(**{**preset(size), **kwargs})

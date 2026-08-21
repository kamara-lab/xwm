"""Generic neural building blocks: layers that know nothing about world models.

These are the pieces you compose into encoders and predictors. Every module is
written for a single unbatched sample; vmap for batches.
"""

from .attention import (
    Attention,
    CrossAttention,
    attend,
    block_causal_attention_mask,
    causal_attention_mask,
)
from .drop import DropPath, linear_droppath_schedule
from .embed import (
    AxialRoPE,
    LearnedPosEmbed,
    SinCosPosEmbed,
    grid_coords,
    sincos_pos_embed,
)
from .mlp import Mlp, SwiGLU
from .norm import LayerNorm, RMSNorm, SimNorm, l2_normalize
from .patch import PatchEmbed2d, PatchEmbed3d, unpatchify_2d
from .transformer import (
    AttentivePooler,
    Block,
    CrossBlock,
    LayerScale,
    Transformer,
    mean_pool,
)

__all__ = [
    "Attention",
    "AttentivePooler",
    "AxialRoPE",
    "Block",
    "CrossAttention",
    "CrossBlock",
    "DropPath",
    "LayerNorm",
    "LayerScale",
    "LearnedPosEmbed",
    "Mlp",
    "PatchEmbed2d",
    "PatchEmbed3d",
    "RMSNorm",
    "SimNorm",
    "SinCosPosEmbed",
    "SwiGLU",
    "Transformer",
    "attend",
    "block_causal_attention_mask",
    "causal_attention_mask",
    "grid_coords",
    "l2_normalize",
    "linear_droppath_schedule",
    "mean_pool",
    "sincos_pos_embed",
    "unpatchify_2d",
]

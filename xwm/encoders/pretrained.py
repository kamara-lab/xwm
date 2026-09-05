"""Pretrained vision encoders, converted into `xwm`'s own modules.

Every other encoder in this package is trained from scratch. This one is not:
it loads DINOv2 and returns an ordinary :class:`~xwm.encoders.VisionEncoder`
subclass, so the frozen features that make DINO-WM work are available to any
family without a second framework in the process.

The conversion is a *translation*, not a wrapper. There is no PyTorch here, no
``transformers``, and nothing calls out to another runtime at inference time --
:func:`xwm.tools.read_safetensors` reads the published weights as numpy and they
are placed into Equinox modules. Two things about that are easy to get subtly
wrong and are therefore checked rather than assumed:

* **The patch projection is a convolution upstream and a matmul here.** Both
  flatten ``(C, p, p)`` in the same order, so the transposed reshape is exact;
  the test compares a real forward pass against known-good reference features.
* **Every tensor in the file must be consumed.** A checkpoint that quietly
  leaves half its weights behind still produces plausible-looking features, so
  :func:`dinov2` refuses one it did not fully use.

References
----------
Oquab et al., *DINOv2: Learning Robust Visual Features without Supervision*,
TMLR 2024. arXiv:2304.07193.

Darcet et al., *Vision Transformers Need Registers*, ICLR 2024. arXiv:2309.16588
-- the ``registers=True`` variants.
"""

from __future__ import annotations

from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from ..core.random import resolve_key
from ..core.types import Array, PRNGKey
from ..nn.patch import PatchEmbed2d
from ..tools.cache import require
from ..tools.safetensors import read_safetensors
from .vision import VisionEncoder

__all__ = ["DINOv2Encoder", "dinov2", "DINOV2_MODELS", "IMAGENET_MEAN", "IMAGENET_STD"]

#: The normalisation DINOv2 was trained under. Inputs to `xwm` encoders are
#: ``[0, 1]``, so this is applied inside the encoder rather than asked of the
#: caller -- an un-normalised image produces features that are wrong but not
#: obviously so.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

#: ``size -> (hub repo, embed_dim, depth, num_heads)``. Patch size is 14 for all
#: of them, and the published position table is for a 518-pixel input.
DINOV2_MODELS: dict[str, tuple[str, int, int, int]] = {
    "small": ("facebook/dinov2-small", 384, 12, 6),
    "base": ("facebook/dinov2-base", 768, 12, 12),
    "large": ("facebook/dinov2-large", 1024, 24, 16),
    "giant": ("facebook/dinov2-giant", 1536, 40, 24),
}

#: The same weights with register tokens, which remove the high-norm artefact
#: patches that make raw DINOv2 attention maps noisy.
DINOV2_REGISTER_MODELS: dict[str, str] = {
    "small": "facebook/dinov2-with-registers-small",
    "base": "facebook/dinov2-with-registers-base",
    "large": "facebook/dinov2-with-registers-large",
    "giant": "facebook/dinov2-with-registers-giant",
}

PATCH_SIZE = 14
SOURCE_GRID = 37  # 518 / 14, the grid the published position table is for

#: PyTorch's ``nn.GELU()``. ``jax.nn.gelu`` defaults to the tanh approximation.
_EXACT_GELU = partial(jax.nn.gelu, approximate=False)


class DINOv2Encoder(VisionEncoder):
    """A DINOv2 ViT that returns patch tokens, ``(n_patches, embed_dim)``.

    A :class:`~xwm.encoders.VisionEncoder` with the three things DINOv2 has and
    it does not: a ``[CLS]`` token, optional register tokens, and input
    normalisation. All three are consumed inside :meth:`__call__` and none
    appears in the output -- what leaves is the patch grid, in row-major order,
    which is what a world model conditions on.

    Constructed by :func:`dinov2` rather than directly; the bare constructor
    produces a randomly initialised model of the right shape, which is what the
    tests use to avoid a download.
    """

    cls_token: Array
    cls_pos: Array
    registers: Array | None
    mean: Array
    std: Array

    def __init__(
        self,
        *,
        img_size: int | tuple[int, int] = 224,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        n_registers: int = 0,
        key: PRNGKey | None = None,
        **kwargs,
    ):
        key = resolve_key(key)
        k_body, k_cls, k_reg = jax.random.split(key, 3)
        super().__init__(
            PatchEmbed2d(img_size, PATCH_SIZE, 3, embed_dim, key=k_body),
            depth=depth,
            num_heads=num_heads,
            key=k_body,
            pos="learned",
            layer_scale=1.0,
            # DINOv2 was trained with PyTorch's exact GELU. JAX's default is the
            # tanh approximation, which is close enough to look right and far
            # enough to shift every feature -- 0.99999 cosine against the
            # reference implementation rather than the 1.0 the weights deserve.
            act=_EXACT_GELU,
            **kwargs,
        )
        self.cls_token = 0.02 * jax.random.normal(k_cls, (embed_dim,))
        self.cls_pos = jnp.zeros((embed_dim,))
        self.registers = (
            0.02 * jax.random.normal(k_reg, (n_registers, embed_dim)) if n_registers else None
        )
        self.mean = jnp.asarray(IMAGENET_MEAN, jnp.float32).reshape(3, 1, 1)
        self.std = jnp.asarray(IMAGENET_STD, jnp.float32).reshape(3, 1, 1)

    @property
    def n_registers(self) -> int:
        return 0 if self.registers is None else self.registers.shape[0]

    def __call__(self, x: Array, *, keep: Array | None = None, key: PRNGKey | None = None) -> Array:
        """``x``: ``(3, H, W)`` in ``[0, 1]``. Returns ``(n_patches, embed_dim)``.

        ``keep`` is refused. Masking is what a JEPA does to *learn* a
        representation; this one is already learned, and dropping patches from a
        pretrained grid would change the statistics its attention was fitted to
        rather than save anything worth saving.
        """
        if keep is not None:
            raise ValueError(
                "DINOv2Encoder cannot encode a token subset: it is a pretrained encoder, "
                "not a JEPA context encoder. Encode the whole grid and index the result."
            )
        tokens = self.patch_embed((x - self.mean) / self.std)
        tokens = tokens + self.pos_embed()
        prefix = [self.cls_token[None] + self.cls_pos[None]]
        if self.registers is not None:
            prefix.append(self.registers)
        n_prefix = sum(p.shape[0] for p in prefix)
        out = self.blocks(jnp.concatenate([*prefix, tokens], axis=0), key=key)
        return out[n_prefix:]


#: The cubic convolution coefficient used by PyTorch's ``bicubic`` resampling.
#: JAX's ``jax.image.resize(method="bicubic")`` uses Keys' ``-0.5`` instead, and
#: the two disagree by enough to matter here -- see :func:`_bicubic_weights`.
_CUBIC_A = -0.75


def _bicubic_weights(in_size: int, out_size: int) -> np.ndarray:
    """``(out_size, in_size)`` resampling matrix matching PyTorch's bicubic.

    Not ``jax.image.resize``. Both are called "bicubic" and they are different
    filters: JAX uses Keys' cubic convolution with ``a = -0.5``, PyTorch uses
    ``a = -0.75``. On DINOv2's position table that difference is worth about
    0.007 of cosine similarity in the output features -- small enough to look
    like numerical noise and large enough to be a different encoder from the one
    everyone else is comparing against. Since the upstream models resample their
    position table with PyTorch, so does this.

    Half-pixel centres (PyTorch's ``align_corners=False``): output pixel ``i``
    samples the input at ``(i + 0.5) * scale - 0.5``.
    """
    scale = in_size / out_size
    centres = (np.arange(out_size) + 0.5) * scale - 0.5
    left = np.floor(centres).astype(np.int64)
    t = (centres - left)[:, None]

    a = _CUBIC_A
    # The standard cubic convolution kernel, evaluated at the four taps.
    taps = np.stack(
        [
            ((a * (t + 1) - 5 * a) * (t + 1) + 8 * a) * (t + 1) - 4 * a,
            ((a + 2) * t - (a + 3)) * t * t + 1,
            ((a + 2) * (1 - t) - (a + 3)) * (1 - t) * (1 - t) + 1,
            np.zeros_like(t),
        ],
        axis=-1,
    )[:, 0, :]
    taps[:, 3] = 1.0 - taps[:, 0] - taps[:, 1] - taps[:, 2]

    matrix = np.zeros((out_size, in_size), np.float64)
    for offset in range(4):
        index = np.clip(left - 1 + offset, 0, in_size - 1)
        np.add.at(matrix, (np.arange(out_size), index), taps[:, offset])
    return matrix


def _interpolate_positions(table: np.ndarray, grid: tuple[int, int]) -> np.ndarray:
    """Resample DINOv2's ``37 x 37`` position table onto ``grid``.

    The published table is for a 518-pixel input. Any other resolution needs the
    grid resampled rather than truncated: the embeddings encode *where in the
    image* a patch is, so keeping the first ``gh * gw`` rows would place every
    patch into the top-left corner of the picture.
    """
    dim = table.shape[-1]
    square = table.reshape(SOURCE_GRID, SOURCE_GRID, dim).astype(np.float64)
    if grid == (SOURCE_GRID, SOURCE_GRID):
        return square.reshape(-1, dim).astype(np.float32)
    rows = _bicubic_weights(SOURCE_GRID, grid[0])
    cols = _bicubic_weights(SOURCE_GRID, grid[1])
    resized = np.einsum("hi,wj,ijd->hwd", rows, cols, square)
    return resized.reshape(-1, dim).astype(np.float32)


def _load_into(encoder: DINOv2Encoder, weights: dict[str, np.ndarray], grid) -> DINOv2Encoder:
    """Place every published tensor into the matching Equinox leaf."""
    used: set[str] = set()

    def take(name: str) -> np.ndarray:
        used.add(name)
        return weights[name]

    def as_array(x) -> Array:
        return jnp.asarray(np.asarray(x, np.float32))

    updates: list[tuple] = []

    # Patch projection: a (D, C, p, p) convolution upstream, a (C*p*p, D) matmul
    # here. Both flatten (C, p, p) in the same order, so this is a reshape.
    conv = take("embeddings.patch_embeddings.projection.weight")
    updates.append((lambda m: m.patch_embed.weight, as_array(conv.reshape(conv.shape[0], -1).T)))
    updates.append(
        (
            lambda m: m.patch_embed.bias,
            as_array(take("embeddings.patch_embeddings.projection.bias")),
        )
    )

    positions = take("embeddings.position_embeddings")[0]
    updates.append((lambda m: m.cls_pos, as_array(positions[0])))
    updates.append(
        (lambda m: m.pos_embed.table, as_array(_interpolate_positions(positions[1:], grid)))
    )
    updates.append((lambda m: m.cls_token, as_array(take("embeddings.cls_token")[0, 0])))
    if encoder.registers is not None:
        updates.append(
            (lambda m: m.registers, as_array(take("embeddings.register_tokens")[0]))
        )
    # `mask_token` belongs to DINOv2's own masked training objective and has no
    # role at inference. Marked used so the completeness check stays meaningful.
    used.add("embeddings.mask_token")

    for i in range(len(encoder.blocks.blocks)):
        p = f"encoder.layer.{i}."

        def at(index):
            return lambda m: m.blocks.blocks[index]

        # One fused qkv projection here, three separate ones upstream. xwm splits
        # the output into thirds in q, k, v order, so the rows concatenate the
        # same way.
        qkv_w = np.concatenate(
            [take(p + f"attention.attention.{n}.weight") for n in ("query", "key", "value")], 0
        )
        qkv_b = np.concatenate(
            [take(p + f"attention.attention.{n}.bias") for n in ("query", "key", "value")], 0
        )
        block = at(i)
        updates += [
            (lambda m, b=block: b(m).attn.qkv.weight, as_array(qkv_w)),
            (lambda m, b=block: b(m).attn.qkv.bias, as_array(qkv_b)),
            (
                lambda m, b=block: b(m).attn.proj.weight,
                as_array(take(p + "attention.output.dense.weight")),
            ),
            (
                lambda m, b=block: b(m).attn.proj.bias,
                as_array(take(p + "attention.output.dense.bias")),
            ),
            (lambda m, b=block: b(m).norm1.weight, as_array(take(p + "norm1.weight"))),
            (lambda m, b=block: b(m).norm1.bias, as_array(take(p + "norm1.bias"))),
            (lambda m, b=block: b(m).norm2.weight, as_array(take(p + "norm2.weight"))),
            (lambda m, b=block: b(m).norm2.bias, as_array(take(p + "norm2.bias"))),
            (lambda m, b=block: b(m).mlp.fc1.weight, as_array(take(p + "mlp.fc1.weight"))),
            (lambda m, b=block: b(m).mlp.fc1.bias, as_array(take(p + "mlp.fc1.bias"))),
            (lambda m, b=block: b(m).mlp.fc2.weight, as_array(take(p + "mlp.fc2.weight"))),
            (lambda m, b=block: b(m).mlp.fc2.bias, as_array(take(p + "mlp.fc2.bias"))),
            (lambda m, b=block: b(m).ls1.gamma, as_array(take(p + "layer_scale1.lambda1"))),
            (lambda m, b=block: b(m).ls2.gamma, as_array(take(p + "layer_scale2.lambda1"))),
        ]

    updates += [
        (lambda m: m.blocks.norm.weight, as_array(take("layernorm.weight"))),
        (lambda m: m.blocks.norm.bias, as_array(take("layernorm.bias"))),
    ]

    leftover = sorted(set(weights) - used)
    if leftover:
        raise ValueError(
            f"{len(leftover)} tensor(s) in the checkpoint were not loaded, so this is not "
            f"the architecture it was published for: {leftover[:6]}"
        )
    return eqx.tree_at(
        lambda m: [getter(m) for getter, _ in updates], encoder, [value for _, value in updates]
    )


def dinov2(
    size: str = "small",
    *,
    img_size: int | tuple[int, int] = 224,
    registers: bool = False,
    repo: str | None = None,
    revision: str | None = None,
) -> DINOv2Encoder:
    """Load pretrained DINOv2 as an `xwm` encoder.

    Args:
        size: ``"small"``, ``"base"``, ``"large"`` or ``"giant"``.
        img_size: what this encoder will be shown. Must be a multiple of 14; the
            position table is resampled to the resulting grid.
        registers: load the register-token variant.
        repo, revision: override the Hub location.

    Needs ``pip install xwm[pretrained]`` for :mod:`huggingface_hub`, and the
    network on first use; the weights are cached by the Hub thereafter.

    Example:
        >>> encoder = xwm.encoders.dinov2("small", img_size=224)   # doctest: +SKIP
        >>> encoder(image).shape                                   # doctest: +SKIP
        (256, 384)
    """
    if size not in DINOV2_MODELS:
        raise ValueError(f"unknown size {size!r}; one of {sorted(DINOV2_MODELS)}")
    default_repo, embed_dim, depth, num_heads = DINOV2_MODELS[size]
    if repo is None:
        repo = DINOV2_REGISTER_MODELS[size] if registers else default_repo

    height, width = (img_size, img_size) if isinstance(img_size, int) else img_size
    if height % PATCH_SIZE or width % PATCH_SIZE:
        raise ValueError(
            f"DINOv2 uses {PATCH_SIZE}-pixel patches, so img_size {(height, width)} must be a "
            f"multiple of {PATCH_SIZE}: try {PATCH_SIZE * round(height / PATCH_SIZE)}"
        )

    (hub,) = require("huggingface_hub", extra="pretrained", what=f"loading DINOv2 from {repo}")
    path = hub.hf_hub_download(repo, "model.safetensors", revision=revision)

    encoder = DINOv2Encoder(
        img_size=(height, width),
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        n_registers=4 if registers else 0,
        key=jax.random.PRNGKey(0),
    )
    grid = (height // PATCH_SIZE, width // PATCH_SIZE)
    return _load_into(encoder, read_safetensors(path), grid)

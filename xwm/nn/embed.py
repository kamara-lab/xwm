"""Positional information: fixed sin-cos tables, learned tables, axial RoPE.

JEPA-style models need positional embeddings that can be *indexed by arbitrary
token subsets*: the context encoder sees a masked subset of the grid, and the
predictor is queried at target positions it never encoded. Every helper here
therefore builds a full ``(N, D)`` table over the grid, from which callers
gather rows with a flat index array.

Fixed tables are derived from static shape fields rather than stored as array
leaves. They are recomputed at trace time and constant-folded, which keeps them
out of the parameter tree entirely -- no "is this buffer trainable?" ambiguity,
nothing for the optimizer or the EMA to accidentally update.

References
----------
Vaswani et al., *Attention Is All You Need*, NeurIPS 2017. arXiv:1706.03762 --
the sin-cos positional table.

Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding*,
2021. arXiv:2104.09864 -- :class:`AxialRoPE`.

Heo et al., *Rotary Position Embedding for Vision Transformer*, ECCV 2024.
arXiv:2403.13298 -- the axial extension to 2-D and 3-D grids.
"""

from __future__ import annotations

from functools import lru_cache

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from ..core.module import Module
from ..core.random import resolve_key
from ..core.types import Array, PRNGKey

Grid = tuple[int, ...]


def axis_dims(dim: int, n_axes: int) -> list[int]:
    """Split ``dim`` into ``n_axes`` even chunks summing to ``dim``.

    A factorised positional embedding gives each grid axis its own slice of the
    width, and each slice needs an even size to hold sin/cos pairs. Requiring
    ``dim % (2 * n_axes) == 0`` would rule out ordinary widths -- a 3-D video
    grid would reject ``dim=64`` -- so the remainder is spread over the leading
    axes instead: 64 over three axes becomes ``[22, 22, 20]``.
    """
    if dim % 2:
        raise ValueError(f"positional width must be even, got {dim}")
    if dim < 2 * n_axes:
        raise ValueError(f"positional width {dim} is too small for {n_axes} axes")
    pairs, extra = divmod(dim // 2, n_axes)
    return [2 * (pairs + (1 if i < extra else 0)) for i in range(n_axes)]


def _sincos_1d(dim: int, positions: np.ndarray) -> np.ndarray:
    """Classic transformer sinusoids for one axis. Returns ``(len(pos), dim)``."""
    if dim % 2 != 0:
        raise ValueError(f"sin-cos embedding dim must be even, got {dim}")
    omega = 1.0 / (10000 ** (np.arange(dim // 2, dtype=np.float64) / (dim / 2.0)))
    out = positions.reshape(-1, 1) * omega.reshape(1, -1)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


# The caches below hold NumPy, not JAX, arrays. A cache of JAX arrays built
# inside one ``jit`` trace would hand those tracers to the next trace, which
# JAX rejects as a leak; NumPy values are trace-independent and become ordinary
# constants wherever they are used.
@lru_cache(maxsize=64)
def _sincos_table(grid: Grid, dim: int) -> np.ndarray:
    widths = axis_dims(dim, len(grid))
    coords = np.meshgrid(*[np.arange(s, dtype=np.float64) for s in grid], indexing="ij")
    parts = [_sincos_1d(w, c.reshape(-1)) for w, c in zip(widths, coords, strict=True)]
    return np.concatenate(parts, axis=1).astype(np.float32)


def sincos_pos_embed(grid: Grid, dim: int) -> Array:
    """Fixed sin-cos table for an n-dimensional grid, flattened row-major.

    The width is split across axes by :func:`axis_dims`. For a video grid
    ``(T, H, W)`` this is the factorised 3-D embedding used by V-JEPA.

    Returns ``(prod(grid), dim)``.
    """
    return jnp.asarray(_sincos_table(tuple(grid), dim))


@lru_cache(maxsize=64)
def _coords_table(grid: Grid) -> np.ndarray:
    coords = np.meshgrid(*[np.arange(s) for s in grid], indexing="ij")
    return np.stack([c.reshape(-1) for c in coords], axis=1).astype(np.int32)


def grid_coords(grid: Grid) -> Array:
    """``(prod(grid), len(grid))`` integer coordinates, row-major flattened."""
    return jnp.asarray(_coords_table(tuple(grid)))


@lru_cache(maxsize=64)
def _rope_tables_np(grid: Grid, head_dim: int, base: float) -> tuple[np.ndarray, np.ndarray]:
    widths = axis_dims(head_dim, len(grid))
    coords = _coords_table(grid).astype(np.float64)  # (N, n_axes)
    columns = []
    for axis, width in enumerate(widths):
        half = width // 2
        freqs = base ** (-np.arange(half) / half)
        columns.append(coords[:, axis : axis + 1] * freqs[None, :])
    angles = np.concatenate(columns, axis=1)  # (N, head_dim // 2)
    return np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)


def _rope_tables(grid: Grid, head_dim: int, base: float) -> tuple[Array, Array]:
    cos, sin = _rope_tables_np(grid, head_dim, base)
    return jnp.asarray(cos), jnp.asarray(sin)


class SinCosPosEmbed(Module):
    """A frozen factorised sin-cos table over a grid."""

    grid: Grid = eqx.field(static=True)
    dim: int = eqx.field(static=True)

    def __init__(self, grid: Grid, dim: int):
        self.grid = tuple(grid)
        self.dim = dim
        sincos_pos_embed(self.grid, dim)  # fail fast on bad dims

    @property
    def table(self) -> Array:
        return sincos_pos_embed(self.grid, self.dim)

    def __call__(self, idx: Array | None = None) -> Array:
        table = self.table
        return table if idx is None else table[idx]


class LearnedPosEmbed(Module):
    """A trainable ``(N, D)`` position table."""

    table: Array

    def __init__(
        self,
        n_positions: int,
        dim: int,
        *,
        key: PRNGKey | None = None,
        scale: float = 0.02,
    ):
        self.table = scale * jr.normal(resolve_key(key), (n_positions, dim))

    def __call__(self, idx: Array | None = None) -> Array:
        return self.table if idx is None else self.table[idx]


class AxialRoPE(Module):
    """Axial rotary embeddings for 1-D, 2-D (image) or 3-D (video) grids.

    The head dimension is split across axes by :func:`axis_dims` and each chunk
    is rotated by its axis' coordinate. Unlike additive tables, RoPE acts inside
    attention and encodes *relative* position, which extrapolates better to
    resolutions and clip lengths unseen during training.
    """

    grid: Grid = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    base: float = eqx.field(static=True)

    def __init__(self, grid: Grid, head_dim: int, *, base: float = 100.0):
        self.grid = tuple(grid)
        self.head_dim = head_dim
        self.base = base
        _rope_tables(self.grid, head_dim, base)  # fail fast

    def __call__(self, x: Array, idx: Array | None = None) -> Array:
        """Rotate ``x`` of shape ``(heads, N, head_dim)``.

        ``idx`` selects grid positions when ``N`` is a masked subset of tokens.
        """
        cos, sin = _rope_tables(self.grid, self.head_dim, self.base)
        if idx is not None:
            cos, sin = cos[idx], sin[idx]
        x1, x2 = jnp.split(x, 2, axis=-1)
        cos, sin = cos[None], sin[None]
        return jnp.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)

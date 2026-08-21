"""Tube masking for video (V-JEPA).

Masking video frame-by-frame is nearly free to solve: the same patch in the
adjacent frame gives the answer away. V-JEPA instead masks *tubes* -- a spatial
region extended over the temporal extent of the clip -- so the model must infer
content it cannot copy from any visible frame.

Two regimes are combined in the paper, and both have presets here:

* **short-range**: many small tubes (:func:`short_range_tubes`)
* **long-range**: a few large tubes (:func:`long_range_tubes`)

References
----------
Bardes et al., *Revisiting Feature Prediction for Learning Visual
Representations from Video* (V-JEPA), 2024. arXiv:2404.08471 -- tube masking,
and the short-range/long-range presets reproduced here.

Tong et al., *VideoMAE: Masked Autoencoders are Data-Efficient Learners for
Self-Supervised Video Pre-Training*, NeurIPS 2022. arXiv:2203.12602 -- tube
masking for the reconstruction-based counterpart.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from ..core.module import Module
from ..core.types import PRNGKey
from .base import (
    MaskBatch,
    expected_context_fraction,
    sample_context,
    snap_area,
    to_numpy_rng,
)


def sample_tube(
    rng: np.random.Generator,
    grid: tuple[int, int, int],
    shape: tuple[int, int],
    n_frames: int,
) -> np.ndarray:
    """Flat indices of an ``(h, w)`` spatial block extended over ``n_frames`` steps."""
    gt, gh, gw = grid
    h, w = min(shape[0], gh), min(shape[1], gw)
    top = rng.integers(0, gh - h + 1)
    left = rng.integers(0, gw - w + 1)
    t0 = rng.integers(0, gt - n_frames + 1)
    t = np.arange(t0, t0 + n_frames)[:, None, None]
    rows = np.arange(top, top + h)[None, :, None]
    cols = np.arange(left, left + w)[None, None, :]
    return (t * gh * gw + rows * gw + cols).reshape(-1)


class TubeMask3d(Module):
    """V-JEPA style tube sampler over a ``(gt, gh, gw)`` token grid.

    Args:
        grid: token grid, e.g. ``PatchEmbed3d.grid``.
        n_targets: number of tubes to predict.
        spatial_scale: fraction of the *spatial* grid each tube covers.
        temporal_extent: tube length in temporal tokens; ``None`` spans the
            whole clip (the V-JEPA default).
        aspect_range: allowed width/height ratios for the tube cross-section.
        context_scale: fraction of *all* tokens kept as context. ``None``
            derives a reliably satisfiable value from the expected tube overlap.
        max_tries: resampling budget when a draw leaves too little context.
    """

    grid: tuple[int, int, int] = eqx.field(static=True)
    n_targets: int = eqx.field(static=True)
    temporal_extent: int = eqx.field(static=True)
    spatial_size: int = eqx.field(static=True)
    context_size: int = eqx.field(static=True)
    shapes: tuple[tuple[int, int], ...] = eqx.field(static=True)
    max_tries: int = eqx.field(static=True)

    def __init__(
        self,
        grid: tuple[int, int, int],
        *,
        n_targets: int = 8,
        spatial_scale: float = 0.15,
        temporal_extent: int | None = None,
        aspect_range: tuple[float, float] = (0.75, 1.5),
        context_scale: float | None = None,
        max_tries: int = 100,
    ):
        gt, gh, gw = grid
        n_frames = gt if temporal_extent is None else min(temporal_extent, gt)
        requested = max(1, int(round(spatial_scale * gh * gw)))
        spatial_size, shapes = snap_area(requested, aspect_range, (gh, gw))
        n_tokens = gt * gh * gw
        if context_scale is None:
            # A tube spanning the full clip covers `spatial_size / (gh * gw)` of
            # every frame, so coverage is measured on the spatial grid alone.
            coverage = (spatial_size / (gh * gw)) * (n_frames / gt)
            context_scale = expected_context_fraction(coverage, n_targets)
        context_size = max(1, int(round(context_scale * n_tokens)))
        if context_size >= n_tokens:
            raise ValueError("context_scale must leave at least one token masked")
        self.grid = (gt, gh, gw)
        self.n_targets = n_targets
        self.temporal_extent = n_frames
        self.spatial_size = spatial_size
        self.context_size = context_size
        self.shapes = tuple(shapes)
        self.max_tries = max_tries

    @property
    def n_tokens(self) -> int:
        gt, gh, gw = self.grid
        return gt * gh * gw

    @property
    def target_size(self) -> int:
        """Tokens per tube: cross-section area times temporal extent."""
        return self.spatial_size * self.temporal_extent

    def _draw_targets(self, rng: np.random.Generator) -> list[np.ndarray]:
        return [
            sample_tube(
                rng,
                self.grid,
                self.shapes[rng.integers(0, len(self.shapes))],
                self.temporal_extent,
            )
            for _ in range(self.n_targets)
        ]

    def __call__(self, key: PRNGKey) -> MaskBatch:
        rng = to_numpy_rng(key)
        targets, context = sample_context(
            rng, self._draw_targets, self.n_tokens, self.context_size, self.max_tries
        )
        return MaskBatch(
            context=jnp.asarray(context, jnp.int32),
            targets=jnp.asarray(targets, jnp.int32),
        )


def short_range_tubes(grid: tuple[int, int, int], **kwargs) -> TubeMask3d:
    """V-JEPA short-range preset: 8 tubes covering ~15% of the spatial grid each."""
    kwargs.setdefault("n_targets", 8)
    kwargs.setdefault("spatial_scale", 0.15)
    return TubeMask3d(grid, **kwargs)


def long_range_tubes(grid: tuple[int, int, int], **kwargs) -> TubeMask3d:
    """V-JEPA long-range preset: 2 tubes covering ~40% of the spatial grid each."""
    kwargs.setdefault("n_targets", 2)
    kwargs.setdefault("spatial_scale", 0.4)
    return TubeMask3d(grid, **kwargs)

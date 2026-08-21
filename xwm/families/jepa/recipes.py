"""Recipes: the published JEPA configurations, assembled from the parts.

Each function returns a :class:`~xwm.families.jepa.JEPA`, differing only in the
tokeniser, the mask sampler and the anti-collapse strategy. Nothing here is new
machinery -- that is the point of the family being one class.

References
----------
Assran et al., *Self-Supervised Learning from Images with a Joint-Embedding
Predictive Architecture* (I-JEPA), CVPR 2023. arXiv:2301.08243

Bardes et al., *Revisiting Feature Prediction for Learning Visual
Representations from Video* (V-JEPA), 2024. arXiv:2404.08471

Balestriero & LeCun, *LeJEPA: Provable and Scalable Self-Supervised Learning
Without the Heuristics*, 2025.
"""

from __future__ import annotations

import jax.random as jr

from ...core.random import resolve_key
from ...core.types import PRNGKey
from ...encoders.image import ImageEncoder
from ...encoders.presets import preset
from ...encoders.video import VideoEncoder
from ...masking.block import MultiBlockMask2d
from ...masking.tube import TubeMask3d
from .model import JEPA
from .predictor import JEPAPredictor

__all__ = ["ijepa", "lejepa", "video_lejepa", "vjepa"]


def _build_image(
    *,
    key: PRNGKey | None = None,
    size: str,
    img_size: int | tuple[int, int],
    patch_size: int,
    in_channels: int,
    n_targets: int,
    target_scale: float,
    context_scale: float | None,
    encoder_kwargs: dict | None,
    predictor_kwargs: dict | None,
    **jepa_kwargs,
) -> JEPA:
    k_enc, k_pred = jr.split(resolve_key(key))
    enc_cfg = {**preset(size, "encoder"), **(encoder_kwargs or {})}
    encoder = ImageEncoder(
        key=k_enc,
        img_size=img_size,
        patch_size=patch_size,
        in_channels=in_channels,
        **enc_cfg,
    )
    pred_cfg = {**preset(size, "predictor"), **(predictor_kwargs or {})}
    predictor = JEPAPredictor(
        encoder.grid,
        encoder.embed_dim,
        key=k_pred,
        n_mask_tokens=n_targets,
        **pred_cfg,
    )
    sampler = MultiBlockMask2d(
        encoder.grid,
        n_targets=n_targets,
        target_scale=target_scale,
        context_scale=context_scale,
    )
    return JEPA(encoder, predictor, sampler, input_key="image", **jepa_kwargs)


def ijepa(
    *,
    key: PRNGKey | None = None,
    size: str = "small",
    img_size: int | tuple[int, int] = 224,
    patch_size: int = 16,
    in_channels: int = 3,
    n_targets: int = 4,
    target_scale: float = 0.15,
    context_scale: float | None = None,
    loss_kind: str = "smooth_l1",
    encoder_kwargs: dict | None = None,
    predictor_kwargs: dict | None = None,
) -> JEPA:
    """I-JEPA: multi-block latent prediction with an EMA teacher.

    Train it with :class:`xwm.training.Trainer`, which maintains the teacher.
    """
    return _build_image(
        key=key,
        size=size,
        img_size=img_size,
        patch_size=patch_size,
        in_channels=in_channels,
        n_targets=n_targets,
        target_scale=target_scale,
        context_scale=context_scale,
        encoder_kwargs=encoder_kwargs,
        predictor_kwargs=predictor_kwargs,
        collapse="ema",
        loss_kind=loss_kind,
    )


def lejepa(
    *,
    key: PRNGKey | None = None,
    size: str = "small",
    img_size: int | tuple[int, int] = 224,
    patch_size: int = 16,
    in_channels: int = 3,
    n_targets: int = 4,
    target_scale: float = 0.15,
    context_scale: float | None = None,
    reg_weight: float = 1.0,
    n_proj: int = 256,
    loss_kind: str = "smooth_l1",
    encoder_kwargs: dict | None = None,
    predictor_kwargs: dict | None = None,
) -> JEPA:
    """LeJEPA for images: the same prediction problem, SIGReg instead of a teacher.

    ``reg_weight`` is the only knob that governs the collapse/expressivity
    trade-off, and there is no teacher for the trainer to maintain.
    """
    return _build_image(
        key=key,
        size=size,
        img_size=img_size,
        patch_size=patch_size,
        in_channels=in_channels,
        n_targets=n_targets,
        target_scale=target_scale,
        context_scale=context_scale,
        encoder_kwargs=encoder_kwargs,
        predictor_kwargs=predictor_kwargs,
        collapse="sigreg",
        loss_kind=loss_kind,
        reg_weight=reg_weight,
        n_proj=n_proj,
    )


def _build_video(
    *,
    key: PRNGKey | None = None,
    size: str,
    img_size: int | tuple[int, int],
    patch_size: int,
    num_frames: int,
    tubelet_size: int,
    in_channels: int,
    n_targets: int,
    spatial_scale: float,
    temporal_extent: int | None,
    context_scale: float | None,
    encoder_kwargs: dict | None,
    predictor_kwargs: dict | None,
    **jepa_kwargs,
) -> JEPA:
    k_enc, k_pred = jr.split(resolve_key(key))
    enc_cfg = {**preset(size, "encoder"), **(encoder_kwargs or {})}
    encoder = VideoEncoder(
        key=k_enc,
        img_size=img_size,
        patch_size=patch_size,
        num_frames=num_frames,
        tubelet_size=tubelet_size,
        in_channels=in_channels,
        **enc_cfg,
    )
    pred_cfg = {**preset(size, "predictor"), **(predictor_kwargs or {})}
    predictor = JEPAPredictor(
        encoder.grid,
        encoder.embed_dim,
        key=k_pred,
        n_mask_tokens=n_targets,
        **pred_cfg,
    )
    sampler = TubeMask3d(
        encoder.grid,
        n_targets=n_targets,
        spatial_scale=spatial_scale,
        temporal_extent=temporal_extent,
        context_scale=context_scale,
    )
    return JEPA(encoder, predictor, sampler, input_key="video", **jepa_kwargs)


def vjepa(
    *,
    key: PRNGKey | None = None,
    size: str = "small",
    img_size: int | tuple[int, int] = 224,
    patch_size: int = 16,
    num_frames: int = 16,
    tubelet_size: int = 2,
    in_channels: int = 3,
    n_targets: int = 8,
    spatial_scale: float = 0.15,
    temporal_extent: int | None = None,
    context_scale: float | None = None,
    loss_kind: str = "l1",
    normalize_target: bool = True,
    encoder_kwargs: dict | None = None,
    predictor_kwargs: dict | None = None,
) -> JEPA:
    """V-JEPA: tube-masked latent prediction with an EMA teacher.

    Defaults follow the paper: L1 loss on LayerNorm-ed targets, eight
    short-range tubes spanning the whole clip.
    """
    return _build_video(
        key=key,
        size=size,
        img_size=img_size,
        patch_size=patch_size,
        num_frames=num_frames,
        tubelet_size=tubelet_size,
        in_channels=in_channels,
        n_targets=n_targets,
        spatial_scale=spatial_scale,
        temporal_extent=temporal_extent,
        context_scale=context_scale,
        encoder_kwargs=encoder_kwargs,
        predictor_kwargs=predictor_kwargs,
        collapse="ema",
        loss_kind=loss_kind,
        normalize_target=normalize_target,
    )


def video_lejepa(
    *,
    key: PRNGKey | None = None,
    size: str = "small",
    img_size: int | tuple[int, int] = 224,
    patch_size: int = 16,
    num_frames: int = 16,
    tubelet_size: int = 2,
    in_channels: int = 3,
    n_targets: int = 8,
    spatial_scale: float = 0.15,
    temporal_extent: int | None = None,
    context_scale: float | None = None,
    reg_weight: float = 1.0,
    n_proj: int = 256,
    loss_kind: str = "l1",
    encoder_kwargs: dict | None = None,
    predictor_kwargs: dict | None = None,
) -> JEPA:
    """LeJEPA for video: tube-masked prediction with SIGReg instead of a teacher."""
    return _build_video(
        key=key,
        size=size,
        img_size=img_size,
        patch_size=patch_size,
        num_frames=num_frames,
        tubelet_size=tubelet_size,
        in_channels=in_channels,
        n_targets=n_targets,
        spatial_scale=spatial_scale,
        temporal_extent=temporal_extent,
        context_scale=context_scale,
        encoder_kwargs=encoder_kwargs,
        predictor_kwargs=predictor_kwargs,
        collapse="sigreg",
        loss_kind=loss_kind,
        reg_weight=reg_weight,
        n_proj=n_proj,
    )

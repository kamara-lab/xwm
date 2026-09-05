"""Observation encoders: pixels or state vectors to latent tokens.

Every family in :mod:`xwm.families` takes an encoder from here. Which one you
pick is a question about the observation, not about the algorithm.

One of them is not trained here at all: :func:`dinov2` loads published DINOv2
weights into the same :class:`VisionEncoder` machinery, which is what
:mod:`xwm.families.dinowm` builds its dynamics on.
"""

from .image import ImageEncoder, image_encoder
from .presets import PREDICTOR_PRESETS, VIT_PRESETS, preset
from .pretrained import DINOV2_MODELS, DINOv2Encoder, dinov2
from .state import StateEncoder
from .video import VideoEncoder, video_encoder
from .vision import PatchEmbed, PosKind, VisionEncoder, make_pos

__all__ = [
    "DINOV2_MODELS",
    "DINOv2Encoder",
    "PREDICTOR_PRESETS",
    "VIT_PRESETS",
    "ImageEncoder",
    "PatchEmbed",
    "PosKind",
    "StateEncoder",
    "VideoEncoder",
    "VisionEncoder",
    "dinov2",
    "image_encoder",
    "make_pos",
    "preset",
    "video_encoder",
]

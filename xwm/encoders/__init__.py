"""Observation encoders: pixels or state vectors to latent tokens.

Every family in :mod:`xwm.families` takes an encoder from here. Which one you
pick is a question about the observation, not about the algorithm.
"""

from .image import ImageEncoder, image_encoder
from .presets import PREDICTOR_PRESETS, VIT_PRESETS, preset
from .state import StateEncoder
from .video import VideoEncoder, video_encoder
from .vision import PatchEmbed, PosKind, VisionEncoder, make_pos

__all__ = [
    "PREDICTOR_PRESETS",
    "VIT_PRESETS",
    "ImageEncoder",
    "PatchEmbed",
    "PosKind",
    "StateEncoder",
    "VideoEncoder",
    "VisionEncoder",
    "image_encoder",
    "make_pos",
    "preset",
    "video_encoder",
]

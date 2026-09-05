"""The JEPA family: predict masked latents, don't collapse.

Self-supervised world models that learn a representation by predicting their own
embeddings at positions they were not shown. No reward, no reconstruction.

* :class:`JEPA` -- the model; :func:`ijepa`, :func:`vjepa`, :func:`lejepa` are
  recipes over it.
* :class:`ActionWorldModel` -- the V-JEPA 2-AC stage: freeze a JEPA encoder and
  learn action-conditioned dynamics in its latent space. This is the family's
  bridge to robotics.
* :class:`ARWorldModel` -- the same bridge without the stage boundary: a window
  of frames, a block-causal predictor, and the encoder trained *through* the
  prediction loss. :func:`lewm` and :func:`delta_jepa` are recipes over it, and
  they differ only in what stops the encoder collapsing.
"""

from .action import ActionWorldModel, action_world_model
from .autoregressive import ARWorldModel, Window, delta_jepa, lewm
from .model import JEPA, Collapse
from .predictor import JEPAPredictor
from .recipes import ijepa, lejepa, video_lejepa, vjepa

__all__ = [
    "JEPA",
    "ActionWorldModel",
    "ARWorldModel",
    "Window",
    "Collapse",
    "JEPAPredictor",
    "action_world_model",
    "delta_jepa",
    "ijepa",
    "lewm",
    "lejepa",
    "video_lejepa",
    "vjepa",
]

"""The JEPA family: predict masked latents, don't collapse.

Self-supervised world models that learn a representation by predicting their own
embeddings at positions they were not shown. No reward, no reconstruction.

* :class:`JEPA` -- the model; :func:`ijepa`, :func:`vjepa`, :func:`lejepa` are
  recipes over it.
* :class:`ActionWorldModel` -- the V-JEPA 2-AC stage: freeze a JEPA encoder and
  learn action-conditioned dynamics in its latent space. This is the family's
  bridge to robotics.
"""

from .action import ActionWorldModel, action_world_model
from .model import JEPA, Collapse
from .predictor import JEPAPredictor
from .recipes import ijepa, lejepa, video_lejepa, vjepa

__all__ = [
    "JEPA",
    "ActionWorldModel",
    "Collapse",
    "JEPAPredictor",
    "action_world_model",
    "ijepa",
    "lejepa",
    "video_lejepa",
    "vjepa",
]

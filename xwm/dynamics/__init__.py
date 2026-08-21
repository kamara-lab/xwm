"""Action-conditioned latent dynamics -- the heart of a robotics world model.

``(z, a) -> z'`` in latent space, shared by every family: JEPA's V-JEPA 2-AC
stage, TD-MPC2's consistency loss, and MuZero's recurrent unroll all consume a
model from here. Which one you pick is a compute/expressivity trade, not an
algorithmic commitment:

* :class:`ActionConditionedPredictor` -- a transformer over the token grid.
  Expressive, and the right choice when the latent is a sequence of patch tokens.
* :class:`MLPDynamics` -- a residual MLP on a pooled latent. Far cheaper, and
  what TD-MPC2 and MuZero actually use, because they run it thousands of times
  inside a planner.
"""

from .action_conditioned import ActionConditionedPredictor, Conditioning
from .action_embed import (
    ContinuousActionEmbed,
    DiscreteActionEmbed,
    PoseActionEmbed,
)
from .mlp_dynamics import MLPDynamics

__all__ = [
    "ActionConditionedPredictor",
    "Conditioning",
    "ContinuousActionEmbed",
    "DiscreteActionEmbed",
    "MLPDynamics",
    "PoseActionEmbed",
]

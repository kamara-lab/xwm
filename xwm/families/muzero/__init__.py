"""MuZero: a latent model trained to agree with its own tree search."""

from .model import MuZero, PolicyHead, muzero
from .targets import n_step_value_targets

__all__ = ["MuZero", "PolicyHead", "muzero", "n_step_value_targets"]

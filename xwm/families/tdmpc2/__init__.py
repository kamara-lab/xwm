"""TD-MPC2: latent dynamics trained by reward and temporal-difference value."""

from .model import TDMPC2, planner, tdmpc2

__all__ = ["TDMPC2", "planner", "tdmpc2"]

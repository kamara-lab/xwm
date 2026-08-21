"""Planning: using a learned world model to choose actions.

This is what a *predictive* world model is for. The encoder turns observations
into latents, the action-conditioned predictor imagines what actions would do,
and a planner searches for the action sequence whose imagined outcome is best --
all without ever rendering a pixel.

Which planner depends on the action space and on whether you have a value
function:

* :class:`CEM`, :class:`MPPI` -- continuous actions, sample whole sequences.
  What TD-MPC2 and the JEPA action-conditioned models use.
* :class:`GradientPlanner` -- continuous, differentiates through the rollout.
  Sample-efficient but happy to exploit model error.
* :class:`MCTS` -- discrete actions, grows a tree and spends its budget where
  the model is least certain. What MuZero uses.
"""

from .cost import (
    CostFn,
    Distance,
    goal_cost,
    latent_distance,
    return_cost,
    reward_cost,
    sum_costs,
)
from .gradient import GradientPlanner
from .mcts import MCTS, SearchResult
from .mpc import ControlStep, Planner, control_step, run_mpc, shift_plan
from .sampling import CEM, MPPI, Plan

__all__ = [
    "CEM",
    "MCTS",
    "MPPI",
    "ControlStep",
    "CostFn",
    "Distance",
    "GradientPlanner",
    "Plan",
    "Planner",
    "SearchResult",
    "control_step",
    "goal_cost",
    "latent_distance",
    "return_cost",
    "reward_cost",
    "run_mpc",
    "shift_plan",
    "sum_costs",
]

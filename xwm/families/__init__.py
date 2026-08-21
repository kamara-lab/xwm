"""Model families.

Three ways to learn an action-conditioned world model, all sharing the same
encoders (:mod:`xwm.encoders`), latent dynamics (:mod:`xwm.dynamics`) and
planners (:mod:`xwm.planning`). What separates them is *what signal trains the
latent space*:

======================================  ==========================  =============
family                                  learning signal             planner
======================================  ==========================  =============
:mod:`~xwm.families.jepa`               its own future embeddings    CEM / MPPI
:mod:`~xwm.families.tdmpc2`             reward + TD value           MPPI
:mod:`~xwm.families.muzero`             search-improved targets     MCTS
======================================  ==========================  =============

JEPA needs no reward, so it can pretrain on passive video -- abundant, unlabelled
robot footage. TD-MPC2 and MuZero need reward and therefore interaction, but they
learn a value function, so their planner can look beyond its horizon. They are
complementary: a JEPA encoder is a reasonable initialisation for either.
"""

from . import jepa, muzero, tdmpc2
from .registry import REGISTRY, available, create, families

__all__ = ["REGISTRY", "available", "create", "families", "jepa", "muzero", "tdmpc2"]

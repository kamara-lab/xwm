"""Model families.

Four ways to learn an action-conditioned world model, all sharing the same
encoders (:mod:`xwm.encoders`), latent dynamics (:mod:`xwm.dynamics`) and
planners (:mod:`xwm.planning`). What separates them is *what signal trains the
latent space*:

======================================  ==========================  =============
family                                  learning signal             planner
======================================  ==========================  =============
:mod:`~xwm.families.jepa`               its own future embeddings    CEM / MPPI
:mod:`~xwm.families.dinowm`             its own future embeddings    CEM / MPPI
:mod:`~xwm.families.tdmpc2`             reward + TD value           MPPI
:mod:`~xwm.families.muzero`             search-improved targets     MCTS
======================================  ==========================  =============

JEPA needs no reward, so it can pretrain on passive video -- abundant, unlabelled
robot footage. TD-MPC2 and MuZero need reward and therefore interaction, but they
learn a value function, so their planner can look beyond its horizon. They are
complementary: a JEPA encoder is a reasonable initialisation for either.

:mod:`~xwm.families.dinowm` shares JEPA's signal and differs in where the
representation comes from: it is not learned here at all, but loaded frozen from
DINOv2. That makes it the control for every model that *does* learn one -- if a
general visual encoder trained on no robot data plans just as well, the
representation learning was not what was doing the work.
"""

from . import dinowm, jepa, muzero, tdmpc2
from .registry import REGISTRY, available, create, families

__all__ = [
    "REGISTRY",
    "available",
    "create",
    "dinowm",
    "families",
    "jepa",
    "muzero",
    "tdmpc2",
]

"""Prediction heads: reward, value, policy, Q.

The pieces a *reward-driven* world model needs and a self-supervised one does
not. JEPA gets by with an encoder and a predictor; TD-MPC2 and MuZero also have
to say how good a state is and what to do in it.
"""

from .action_decoder import LatentDifferenceActionDecoder
from .categorical import CategoricalScalar, cross_entropy, symexp, symlog, two_hot
from .policy import GaussianPolicy, PolicyOutput
from .q_ensemble import QEnsemble
from .scalar import ScalarHead

__all__ = [
    "LatentDifferenceActionDecoder",
    "CategoricalScalar",
    "GaussianPolicy",
    "PolicyOutput",
    "QEnsemble",
    "ScalarHead",
    "cross_entropy",
    "symexp",
    "symlog",
    "two_hot",
]

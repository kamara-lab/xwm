"""Training objectives for predictive world models.

Two independent choices make up a JEPA objective:

1. **What to predict, and how to score it** -- :func:`prediction_loss`,
   always computed in latent space.
2. **How to prevent collapse** -- :func:`sigreg` (LeJEPA: constrain the
   embedding distribution), :func:`vicreg` (constrain its first two moments),
   :func:`info_nce` (contrast against negatives), or an EMA teacher
   (architectural, see :mod:`xwm.core.ema`).

Mixing and matching those two axes is what the model classes in
:mod:`xwm.image`, :mod:`xwm.video` and :mod:`xwm.action` expose.
"""

from .prediction import LossKind, layer_normalize, prediction_loss
from .regularizers import (
    covariance_loss,
    info_nce,
    variance_loss,
    vicreg,
)
from .sigreg import (
    Quadrature,
    Statistic,
    cramer_von_mises,
    epps_pulley,
    random_directions,
    sigreg,
)

__all__ = [
    "LossKind",
    "Quadrature",
    "Statistic",
    "covariance_loss",
    "cramer_von_mises",
    "epps_pulley",
    "info_nce",
    "layer_normalize",
    "prediction_loss",
    "random_directions",
    "sigreg",
    "variance_loss",
    "vicreg",
]

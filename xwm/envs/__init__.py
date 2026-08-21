"""Simulated environments with real dynamics.

Distinct from :mod:`xwm.data`, which generates synthetic arrays in pure JAX.
Everything here wraps an external simulator, carries optional heavy
dependencies, and lives outside the ``jit`` boundary.
"""

from .discretize import discrete_action_table
from .newton_franka import (
    HOME_POSE,
    N_ARM_JOINTS,
    FrankaConfig,
    FrankaEnv,
    franka_sequences,
    smooth_actions,
)
from .render import (
    RENDER_BACKENDS,
    HighQualityRenderer,
    look_at_angles,
    supersample,
    which_backends,
)

__all__ = [
    "RENDER_BACKENDS",
    "HighQualityRenderer",
    "look_at_angles",
    "HOME_POSE",
    "N_ARM_JOINTS",
    "FrankaConfig",
    "FrankaEnv",
    "discrete_action_table",
    "franka_sequences",
    "supersample",
    "which_backends",
    "smooth_actions",
]

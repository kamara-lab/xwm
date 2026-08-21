"""xwm -- building blocks for predictive world models, in JAX.

A *world model* answers "what happens if I do this?". An **action-conditioned**
one answers it in a latent space of its own choosing, which is what makes it
usable for robotics: you can search over imagined action sequences without
rendering a single pixel.

Layout
------
The library separates *generic machinery* from *families* from *use*:

============================  ===================================================
:mod:`xwm.core`               types, base modules, EMA targets, rollouts, keys
:mod:`xwm.nn`                 layers -- attention, transformers, SimNorm, patches
:mod:`xwm.encoders`           observation -> latent (image, video, state)
:mod:`xwm.dynamics`           **(z, a) -> z'** -- the heart of a robotics model
:mod:`xwm.heads`              reward, value, policy, Q-ensemble
:mod:`xwm.masking`            what a JEPA predicts: blocks, tubes, time splits
:mod:`xwm.objectives`         losses: latent prediction, SIGReg, TD, categorical
:mod:`xwm.families`           **jepa**, **tdmpc2**, **muzero**
:mod:`xwm.planning`           CEM, MPPI, gradient, MPC, MCTS
:mod:`xwm.training`           one trainer, schedules, replay buffer
:mod:`xwm.envs`               simulated robots (a Franka arm in Newton)
:mod:`xwm.data`               batch streams and a synthetic controllable world
:mod:`xwm.metrics`            probes and collapse diagnostics
:mod:`xwm.plots`              figures, GIFs, JSON/LaTeX tables
:mod:`xwm.tools`              checkpointing and model summaries
============================  ===================================================

The three families differ only in what trains the latent space -- its own future
embeddings (JEPA), reward and TD value (TD-MPC2), or search-improved targets
(MuZero). They share encoders, dynamics and planners.

Conventions
-----------
Modules follow Equinox: immutable PyTrees, written for a **single unbatched
sample** and ``vmap``ed by the caller. Batch-level entry points are the methods
named ``loss``. Images are ``(C, H, W)``, clips ``(T, C, H, W)``, token
sequences ``(N, D)``, flat latents ``(D,)``, actions ``(A,)``.

``key=`` is optional wherever a model is *built* (see :mod:`xwm.core.random`);
it stays required wherever a key is consumed inside ``jit``.

Quick start
-----------
>>> import xwm
>>> xwm.set_seed(0)
>>> model = xwm.families.jepa.lejepa(size="tiny", img_size=64, patch_size=8)
>>> trainer = xwm.training.Trainer(model, xwm.training.adamw(1e-4))
"""

from . import (
    core,
    data,
    dynamics,
    encoders,
    envs,
    families,
    heads,
    masking,
    metrics,
    nn,
    objectives,
    planning,
    plots,
    tools,
    training,
)
from .core import Module, WorldModel, key_source, seed, set_seed
from .families.jepa import JEPA
from .masking import MaskBatch

__version__ = "0.1.0"

__all__ = [
    "JEPA",
    "MaskBatch",
    "Module",
    "WorldModel",
    "__version__",
    "core",
    "data",
    "dynamics",
    "encoders",
    "envs",
    "families",
    "heads",
    "key_source",
    "masking",
    "metrics",
    "nn",
    "objectives",
    "planning",
    "plots",
    "seed",
    "set_seed",
    "tools",
    "training",
]

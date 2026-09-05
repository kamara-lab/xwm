"""Interactive inspection of a run, through the Rerun viewer.

Everything else xwm writes is static: a PNG of a loss curve, a GIF of an
episode, a JSON of the numbers. That is the right format for a paper and the
wrong one for a diagnosis. A latent rollout that diverges from the truth at step
four, a planner whose proposal spread never shrinks, an arm that reaches past
its goal -- all are shapes over time, and a shape over time wants a timeline you
can scrub, not a frame someone chose in advance.

This module logs xwm's own objects, not Rerun's: a
:class:`~xwm.training.Trainer` callback, an
:class:`~xwm.bench.EpisodeResult`, a :class:`~xwm.planning.Plan`, a
:class:`~xwm.envs.FrankaEnv`. Every function takes a :class:`Recording` first
and accepts ``None`` for it, so instrumenting a script adds arguments rather
than branches.

The SDK is an optional dependency (``pip install xwm[rerun]``). Importing this
module without it succeeds; the error arrives only when a helper that needs it
is called, exactly as in :mod:`xwm.plots`.

Timelines
---------
============  ==============================================================
``step``      optimizer step -- training metrics, collapse diagnostics
``episode``   which evaluation instance
``env_step``  raw environment step within an episode
``horizon``   position within one imagined rollout or plan
``frame``     position within a logged clip
============  ==============================================================

Example:
    >>> import xwm                                          # doctest: +SKIP
    >>> rec = xwm.rerun.Recording("xwm", path="run.rrd",
    ...                           blueprint=xwm.rerun.blueprints.training())
    >>> state, history = trainer.fit(
    ...     batches, steps=1000, callbacks=[xwm.rerun.training_callback(rec)])
    >>> rec.close()                       # then: rerun run.rrd
"""

from . import blueprints
from ._backend import available
from .bench import episode_hook, log_episode, log_evaluation
from .franka import franka_hook, log_franka_scene, log_franka_step
from .latent import log_clip, log_latent_trajectory, log_plan
from .recording import Recording
from .training import log_collapse, log_history, log_summary, training_callback

__all__ = [
    "available",
    "blueprints",
    "episode_hook",
    "franka_hook",
    "log_clip",
    "log_collapse",
    "log_episode",
    "log_evaluation",
    "log_franka_scene",
    "log_franka_step",
    "log_history",
    "log_latent_trajectory",
    "log_plan",
    "log_summary",
    "Recording",
    "training_callback",
]

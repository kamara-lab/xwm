"""Benchmark episodes: what the policy saw, what it did, and how far it got.

An evaluation number -- ``success 0.00, gap closed 0.012`` -- says a planner
failed but not how. These loggers put the observation, the goal it is aiming at,
the executed actions and the distance trace on one scrubbable timeline, which is
where the difference between "the model's dynamics are wrong" and "the planner
never left the start state" becomes visible.

Two entry points, because there are two moments to log at. :func:`episode_hook`
observes the loop as it runs, and needs :func:`xwm.bench.run_episode`'s
``on_step``. :func:`log_episode` works from a finished
:class:`~xwm.bench.EpisodeResult`, so a results directory written last week can
be replayed without re-running anything.
"""

from __future__ import annotations

import numpy as np

from ._backend import rerun

__all__ = ["episode_hook", "log_episode", "log_evaluation"]


def _log_step(rec, info, *, prefix: str) -> None:
    rr = rerun()
    rec.set_time("env_step", info.step)
    rec.log(f"{prefix}/distance", rr.Scalars(float(info.distance)))
    rec.log(f"{prefix}/success", rr.Scalars(float(info.success)))
    action = np.asarray(info.raw, np.float32).ravel()
    for i, value in enumerate(action):
        rec.log(f"{prefix}/action/{i}", rr.Scalars(float(value)))
    if info.frame is not None:
        from ..plots.export import frames_to_uint8

        rec.log(f"{prefix}/observation", rr.Image(frames_to_uint8(info.frame[None])[0]))
    if info.plan is not None:
        rec.log(f"{prefix}/plan/cost", rr.Scalars(float(info.plan.cost)))
        spread = np.asarray(info.plan.std, np.float32)
        rec.log(f"{prefix}/plan/spread", rr.Scalars(float(spread.mean())))


def episode_hook(recording, *, prefix: str = "episode"):
    """An ``on_step`` hook for :func:`xwm.bench.goal_reaching.evaluate`.

    Entities are namespaced by policy -- ``<prefix>/<policy>/...`` -- so the
    planner and its three floors overlay on one plot, which is the comparison
    the protocol exists to make. The ``episode`` timeline selects the instance
    and ``env_step`` the step within it.

    Example:
        >>> outcome = xwm.bench.evaluate(              # doctest: +SKIP
        ...     model, task, on_step=xwm.rerun.episode_hook(rec))
    """
    if recording is None:
        return None

    def hook(policy_name: str, index: int, info) -> None:
        recording.set_time("episode", index)
        _log_step(recording, info, prefix=f"{prefix}/{policy_name}")

    return hook


def log_episode(recording, result, *, name: str = "planner", index: int = 0, prefix="episode"):
    """Log a finished :class:`~xwm.bench.EpisodeResult`.

    Logs the distance trace on ``env_step`` and, when the episode was recorded
    with ``record=True``, its frames. ``solved_at`` is logged as a static text
    note rather than a scalar, because it is a property of the episode and not
    of any step in it.
    """
    if recording is None:
        return
    rr = rerun()
    entity = f"{prefix}/{name}"
    recording.set_time("episode", index)
    for step, distance in enumerate(result.distances):
        recording.set_time("env_step", step)
        recording.log(f"{entity}/distance", rr.Scalars(float(distance)))
    if result.frames:
        from ..plots.export import frames_to_uint8

        images = frames_to_uint8(np.stack([np.asarray(f) for f in result.frames]))
        for step, image in enumerate(images):
            recording.set_time("env_step", step)
            recording.log(f"{entity}/observation", rr.Image(image))
    solved = "never" if result.solved_at is None else f"step {result.solved_at}"
    recording.log(
        f"{entity}/outcome",
        rr.TextDocument(
            f"success={result.success}  solved={solved}  steps={result.steps}\n"
            f"initial={result.initial_distance:.4f}  final={result.final_distance:.4f}  "
            f"best={result.best_distance:.4f}  closed={result.distance_closed:.4f}"
        ),
    )


def log_evaluation(recording, outcome, *, prefix: str = "episode") -> None:
    """Log every episode of an :func:`xwm.bench.evaluate` result.

    Reads the private ``_episodes`` entry that ``evaluate`` leaves on each
    policy -- the raw :class:`~xwm.bench.EpisodeResult` objects, which
    :func:`xwm.bench.report.build` strips before writing JSON.
    """
    if recording is None:
        return
    for name, summary in outcome["results"].items():
        for index, result in enumerate(summary.get("_episodes", [])):
            log_episode(recording, result, name=name, index=index, prefix=prefix)

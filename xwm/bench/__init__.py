"""The evaluation protocol: goal reaching under a step budget.

One question, asked the same way of every model: reset the environment to a
state a held-out recording visited, show the model -- as an *observation* -- a
state that recording reached later, and give its planner a fixed budget of
environment steps to get there. Success is a ground-truth predicate on
simulator state that the model never sees.

>>> import xwm
>>> task = xwm.tasks.create("pusht/synthetic")
>>> outcome = xwm.bench.evaluate(model, task, episodes=8)
>>> outcome["results"]["planner"]["success_rate"]

Read ``replay`` before reading anything else. It executes the demonstrator's own
actions from the start state, so it should succeed essentially always; when it
does not, the environment is not being reset faithfully and no model number
from that task means anything yet.
"""

from __future__ import annotations

from .control import Context, EpisodeResult, run_episode
from .goal_reaching import build_cost, build_planner, build_search, evaluate, summarize
from .policies import POLICIES, make_policy
from .protocol import SCHEMA, Episode, PlanConfig, sample_episodes
from .report import build, comparison_table, environment, write

__all__ = [
    "POLICIES",
    "SCHEMA",
    "Context",
    "Episode",
    "EpisodeResult",
    "PlanConfig",
    "build",
    "build_cost",
    "build_planner",
    "build_search",
    "comparison_table",
    "environment",
    "evaluate",
    "make_policy",
    "run_episode",
    "sample_episodes",
    "summarize",
    "write",
]

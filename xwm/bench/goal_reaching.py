"""The headline evaluation: can a model's planner reach a goal it was shown?

The measurement is deliberately narrow. A start state and a goal state are
taken from one held-out recording, the environment is reset to the start, the
goal is handed to the model **as an observation** -- never as coordinates, never
as a reward -- and the planner gets a fixed budget of environment steps. Success
is a ground-truth predicate on simulator state that the model never sees.

That last point is what makes the number comparable across families. A JEPA has
no reward head and a TD-MPC2 has one; scoring both on latent distance to an
encoded goal asks them the same question, and if that handicaps the
reward-driven model, the handicap is visible rather than baked into two
incomparable protocols.

References
----------
Zhou et al., *DINO-WM*, 2024. arXiv:2411.04983.

Maes et al., *stable-worldmodel*, 2026. arXiv:2605.21800.
"""

from __future__ import annotations

import time
from typing import Any

import jax.random as jr
import numpy as np

from ..core.types import PRNGKey
from ..planning.cost import goal_cost
from ..planning.sampling import CEM, MPPI
from .control import run_episode
from .policies import make_policy
from .protocol import Episode, PlanConfig, sample_episodes

__all__ = ["build_cost", "build_planner", "build_search", "evaluate", "summarize"]


def build_planner(task, plan: PlanConfig, model=None):
    """The planner named by a :class:`~xwm.bench.protocol.PlanConfig`.

    Bounds are ``[-1, 1]`` because a task normalises its actions there; the
    denormalisation back to raw units happens in the control loop, once, at the
    environment boundary.
    """
    if model is not None:
        _reject_discrete(model, plan)
    width = task.spec.blocked_action_dim
    options = dict(plan.planner_options)
    if plan.planner == "cem":
        return CEM(plan.horizon, width, **{"n_samples": 256, "n_elites": 32, **options})
    if plan.planner == "mppi":
        return MPPI(plan.horizon, width, **{"n_samples": 256, **options})
    if plan.planner == "gradient":
        from ..planning.gradient import GradientPlanner

        return GradientPlanner(plan.horizon, width, **options)
    raise KeyError(
        f"unknown planner {plan.planner!r}; available: cem, mppi, gradient "
        "(mcts is reached through a discrete-action family, not from here)"
    )


def _reject_discrete(model, plan: PlanConfig) -> None:
    """Refuse a continuous planner over a discrete-action model, with a reason.

    Every planner here samples real-valued action *vectors*; a model that
    consumes an action index or a one-hot over ``n_actions`` will either
    misinterpret them or fail on a shape deep inside its dynamics, and neither
    says what is wrong.
    """
    n_actions = getattr(model, "n_actions", None)
    if n_actions is not None:
        raise TypeError(
            f"{type(model).__name__} takes one of {n_actions} discrete actions, and "
            f"{plan.planner!r} samples continuous vectors. Plan it with "
            "xwm.planning.MCTS through model.search_fns(), mapping indices to actions "
            "with xwm.envs.discrete_action_table."
        )


def build_cost(model, task, goal_frame, plan: PlanConfig):
    """The objective the planner minimises.

    ``goal``: latent distance to the encoded goal observation, through the
    model's own readout, so a latent that carries several frames is compared
    only on its newest one.

    ``return``: the model's learned reward and value. Only meaningful for a
    family that has them.
    """
    if plan.objective == "goal":
        z_goal = model.goal_embedding(task.observation(goal_frame))
        return goal_cost(
            z_goal,
            kind=plan.distance,
            readout=model.readout,
            action_penalty=plan.action_penalty,
            discount=plan.discount,
        )
    if plan.objective == "return":
        raise KeyError(
            "objective='return' is built by the family that owns the reward and value "
            "heads, not here; xwm.bench.build_search dispatches to it."
        )
    raise KeyError(f"unknown objective {plan.objective!r}; available: goal, return")


def build_search(model, task, goal_frame, plan: PlanConfig):
    """``(planner, cost_fn)`` for one instance -- the pair, because they are one choice.

    A goal objective works for any :class:`~xwm.core.types.Plannable`: encode
    the goal observation, compare latents. A *return* objective does not, and
    cannot be assembled from the outside. The reward head was trained as
    ``r(z_t, a_t)``, so it has to be scored at the pre-transition latent
    (``cost_on="current"``, a planner setting) and bootstrapped with the
    critic evaluated at the policy's own action (a model detail). Both live in
    :func:`xwm.families.tdmpc2.planner`, which returns the matched pair -- so
    this dispatches to it rather than reconstructing it and getting the
    convention subtly wrong.
    """
    if plan.objective == "return":
        if not hasattr(model, "reward") or not hasattr(model, "critic"):
            raise TypeError(
                f"objective='return' needs a model with reward and value heads; "
                f"{type(model).__name__} has none. Use objective='goal', which every "
                "model supports -- that is what makes families comparable here."
            )
        from ..families.tdmpc2.model import planner as tdmpc2_planner

        options = dict(plan.planner_options)
        return tdmpc2_planner(model, horizon=plan.horizon, **options)
    return build_planner(task, plan, model), build_cost(model, task, goal_frame, plan)


def summarize(results: list, wall_seconds: float) -> dict[str, Any]:
    """Aggregate per-episode results into the numbers a table shows."""
    solved = [r for r in results if r.success]
    return {
        "success_rate": float(np.mean([r.success for r in results])),
        "final_distance_mean": float(np.mean([r.final_distance for r in results])),
        "final_distance_median": float(np.median([r.final_distance for r in results])),
        "best_distance_mean": float(np.mean([r.best_distance for r in results])),
        "initial_distance_mean": float(np.mean([r.initial_distance for r in results])),
        "distance_closed_mean": float(np.mean([r.distance_closed for r in results])),
        "steps_to_success_mean": (
            float(np.mean([r.solved_at for r in solved])) if solved else None
        ),
        "steps_to_success_n": len(solved),
        "plan_cost_mean": (
            float(np.mean([r.plan_cost for r in results if r.plan_cost is not None]))
            if any(r.plan_cost is not None for r in results)
            else None
        ),
        "wall_seconds": round(wall_seconds, 2),
        "per_episode": [
            {
                "success": bool(r.success),
                "steps": int(r.steps),
                "solved_at": r.solved_at,
                "initial_distance": r.initial_distance,
                "final_distance": r.final_distance,
                "best_distance": r.best_distance,
                "distance_closed": r.distance_closed,
            }
            for r in results
        ],
    }


def evaluate(
    model,
    task,
    *,
    plan: PlanConfig | None = None,
    policies: tuple[str, ...] = ("planner", "noop", "random", "replay"),
    episodes: int | None = None,
    budget: int | None = None,
    goal_offset: int | None = None,
    seed: int = 0,
    key: PRNGKey | None = None,
    instances: list[Episode] | None = None,
    record: bool = False,
    progress: bool = False,
) -> dict[str, Any]:
    """Score a model, and the baselines that make its score readable.

    Args:
        model: anything satisfying :class:`xwm.core.types.Plannable`. Only the
            ``planner`` policy touches it, so the baselines can be run with
            ``model=None``.
        task: a :class:`xwm.tasks.Task`.
        plan: planner and objective settings; defaults follow the task.
        policies: which to run. All see the same instances and the same seeds,
            so the comparison is paired.
        episodes, budget, goal_offset: override the task's defaults.
        instances: run exactly these, instead of sampling. Pass the
            ``instances`` from a previous result to re-run it exactly.
        record: keep rendered frames for the first ``planner`` episode.

    Returns a dict per policy; :func:`xwm.bench.report.write` gives it a schema
    and writes it out.
    """
    spec = task.spec
    plan = plan or PlanConfig(horizon=spec.horizon, receding_horizon=spec.receding_horizon)
    n = spec.episodes if episodes is None else episodes
    budget = spec.eval_budget if budget is None else budget
    offset = spec.goal_offset if goal_offset is None else goal_offset
    key = jr.PRNGKey(seed) if key is None else key

    held_out = task.episodes(split="val")
    if instances is None:
        instances = sample_episodes(
            held_out,
            n=n,
            goal_offset=offset,
            seed=seed,
            frameskip=spec.frameskip,
            distance=task.distance,
            # An instance whose goal is already within the success radius is one
            # `noop` solves; drawing them would flatter every policy equally and
            # measure none of them.
            min_distance=float(spec.metric_options.get("threshold", 0.0)),
        )

    results: dict[str, Any] = {}
    for name in policies:
        if name == "planner" and model is None:
            raise ValueError("policies include 'planner' but no model was given")
        started = time.perf_counter()
        env = task.make_env()
        per_episode = []
        try:
            for i, instance in enumerate(instances):
                if name == "planner":
                    goal_frame = _goal_frame(task, held_out[instance.index], instance)
                    search, cost_fn = build_search(model, task, goal_frame, plan)
                    policy = make_policy(
                        name, task, model=model, planner=search, cost_fn=cost_fn, plan=plan
                    )
                else:
                    policy = make_policy(name, task)
                per_episode.append(
                    run_episode(
                        task,
                        env,
                        instance,
                        policy=policy,
                        plan=plan,
                        key=jr.fold_in(key, i),
                        budget=budget,
                        record=record and name == "planner" and i == 0,
                    )
                )
                if progress:
                    print(f"\r{name}: {i + 1}/{len(instances)}   ", end="", flush=True)
        finally:
            env.close()
        if progress:
            print(f"\r{name}: {len(instances)} episodes      ", flush=True)
        results[name] = summarize(per_episode, time.perf_counter() - started)
        results[name]["_episodes"] = per_episode
    return {"instances": instances, "plan": plan, "budget": budget, "results": results}


def _goal_frame(task, recording, instance: Episode):
    """The goal, as an observation -- the only form the model ever sees it in."""
    frame = {}
    if "video" in recording:
        frame["image"] = recording["video"][instance.goal]
    if "state" in recording:
        frame["state"] = recording["state"][instance.goal]
    return frame

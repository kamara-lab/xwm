"""Writing an evaluation down, in a form another program can read.

One file per evaluation, one schema, everything needed to reproduce it: the
task and its full spec, the model and the hash of the config that built it, the
planner settings, and the exact instance list. A results file that cannot be
re-run from its own contents is a screenshot.

Numbers go through :func:`xwm.plots.jsonable`, so a non-finite value becomes
``null`` rather than bare ``NaN`` -- which strict JSON parsers reject, and which
is how a results file becomes unreadable weeks later.
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path
from typing import Any

import jax

from ..plots.tables import save_json, save_table
from .protocol import SCHEMA

__all__ = ["build", "write", "comparison_table"]

#: The per-policy numbers a table shows, in the order they read best.
_COLUMNS = (
    ("success_rate", "success"),
    ("distance_closed_mean", "gap closed"),
    ("final_distance_mean", "final dist"),
    ("best_distance_mean", "best dist"),
    ("steps_to_success_mean", "steps"),
    ("wall_seconds", "seconds"),
)


def environment() -> dict[str, Any]:
    """What ran it. Cheap to record, and the first thing wanted when two runs disagree."""
    return {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "jax_backend": jax.default_backend(),
        "devices": [str(d) for d in jax.devices()],
    }


def build(
    outcome: dict[str, Any],
    *,
    task,
    model_name: str,
    model_kwargs: dict[str, Any] | None = None,
    config_hash: str | None = None,
    n_params: int | None = None,
    seed: int = 0,
    prediction: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the results payload from what :func:`xwm.bench.evaluate` returned."""
    import dataclasses

    # Imported here rather than at module scope: xwm/__init__ imports this
    # package while its own __version__ is still being defined.
    from .. import __version__

    plan = outcome["plan"]
    results = {
        name: {k: v for k, v in numbers.items() if not k.startswith("_")}
        for name, numbers in outcome["results"].items()
    }
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "xwm_version": __version__,
        "task": task.spec.name,
        "task_spec": dataclasses.asdict(task.spec),
        "model": {
            "name": model_name,
            "kwargs": model_kwargs or {},
            "config_hash": config_hash,
            "n_params": n_params,
        },
        "seed": seed,
        "plan": dataclasses.asdict(plan),
        "budget": outcome["budget"],
        "goal_offset": task.spec.goal_offset,
        "episodes": len(outcome["instances"]),
        "instances": [dataclasses.asdict(i) for i in outcome["instances"]],
        "results": results,
        "environment": environment(),
    }
    if prediction is not None:
        payload["prediction"] = prediction
    return payload


def comparison_table(payload: dict[str, Any]) -> tuple[list[str], list[list[Any]]]:
    """The per-policy comparison, as ``(headers, rows)`` for :func:`xwm.plots.save_table`."""
    headers = ["policy", *(label for _, label in _COLUMNS)]
    # Model first, then the floors, whatever order they were requested in.
    order = ["planner", "replay", "noop", "random"]
    names = sorted(payload["results"], key=lambda n: (order.index(n) if n in order else 99, n))
    rows = [[name, *(payload["results"][name].get(key) for key, _ in _COLUMNS)] for name in names]
    return headers, rows


def write(payload: dict[str, Any], directory: str | Path, *, stem: str = "eval") -> dict[str, Path]:
    """Write ``<stem>.json`` and a formatted ``<stem>_table.{json,tex}``."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    written = {"json": save_json(directory / stem, payload)}
    headers, rows = comparison_table(payload)
    written |= {
        f"table_{ext}": path
        for ext, path in save_table(
            directory / f"{stem}_table",
            headers,
            rows,
            caption=f"Goal reaching on {payload['task']} "
            f"({payload['episodes']} episodes, {payload['budget']}-step budget).",
            label=f"tab:{payload['task'].replace('/', '-')}",
        ).items()
    }
    return written

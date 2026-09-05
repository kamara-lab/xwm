"""``xwm eval``: load a trained run and measure it against the protocol."""

from __future__ import annotations

import json
from pathlib import Path

import jax.random as jr

from ..bench.goal_reaching import evaluate
from ..bench.protocol import Episode, PlanConfig
from ..bench.report import build, write
from ..config.loader import config_hash, from_dict
from ..config.schema import ExperimentConfig
from ..core.random import set_seed
from .build import build_model, build_task

__all__ = ["run", "load_run"]


def load_run(target: str | Path) -> tuple[ExperimentConfig, Path]:
    """Read the resolved config from a run directory.

    Accepts the directory or the ``config.json`` inside it, because both are
    things a person reasonably pastes.
    """
    path = Path(target)
    candidate = path / "config.json" if path.is_dir() else path
    if not candidate.exists():
        raise FileNotFoundError(
            f"no config.json at {candidate}. `xwm eval` takes a run directory written "
            "by `xwm train`; to evaluate a bare checkpoint, point a config at it with "
            "model.checkpoint=<path>."
        )
    return from_dict(json.loads(candidate.read_text())), candidate.parent


def run(
    config: ExperimentConfig,
    directory: Path,
    *,
    checkpoint: str | Path | None = None,
    progress: bool = True,
) -> Path:
    """Evaluate, write ``eval.json`` and the comparison table, return the JSON path."""
    set_seed(config.seed)
    task = build_task(config)

    weights = Path(checkpoint) if checkpoint else directory / "checkpoint" / "model.eqx"
    if weights.exists():
        import dataclasses

        config = dataclasses.replace(
            config,
            model=dataclasses.replace(config.model, checkpoint=str(weights)),
        )
    elif "planner" in config.eval.policy:
        raise FileNotFoundError(
            f"no weights at {weights}; train first, or evaluate only the baselines "
            "with --policy noop --policy random --policy replay"
        )

    model = None
    if "planner" in config.eval.policy:
        model = build_model(config, task, key=jr.PRNGKey(config.seed)).eval_mode()

    evaluation = config.eval
    plan = evaluation.plan or PlanConfig()
    from .train import _recording

    recording = _recording(config, directory, "eval", "episode")
    hook = None
    if recording is not None:
        from ..rerun import episode_hook

        hook = episode_hook(recording)
    outcome = evaluate(
        model,
        task,
        plan=plan,
        policies=tuple(evaluation.policy),
        episodes=evaluation.episodes,
        budget=evaluation.budget,
        goal_offset=evaluation.goal_offset,
        seed=config.seed,
        record=evaluation.video,
        progress=progress,
        on_step=hook,
    )
    if recording is not None:
        recording.close()
    payload = build(
        outcome,
        task=task,
        model_name=config.model.name,
        model_kwargs=config.model.kwargs,
        config_hash=config_hash(config),
        n_params=int(model.n_params) if model is not None else None,
        seed=config.seed,
    )
    written = write(payload, directory)
    if progress:
        _print_table(payload)
    return written["json"]


def _print_table(payload: dict) -> None:
    from ..bench.report import comparison_table

    headers, rows = comparison_table(payload)
    widths = [max(len(str(h)), *(len(_fmt(r[i])) for r in rows)) for i, h in enumerate(headers)]
    line = "  ".join(h.rjust(w) for h, w in zip(headers, widths, strict=True))
    print(line)
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print("  ".join(_fmt(v).rjust(w) for v, w in zip(row, widths, strict=True)))


def _fmt(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def replay_instances(payload: dict) -> list[Episode]:
    """The instance list from a results file, for re-running it exactly."""
    return [Episode(**instance) for instance in payload["instances"]]

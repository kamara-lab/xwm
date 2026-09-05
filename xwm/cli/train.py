"""``xwm train``: build a model from a config, fit it, write a run directory."""

from __future__ import annotations

import json
import time
from pathlib import Path

import jax.random as jr

from ..config.loader import config_hash, save
from ..config.schema import ExperimentConfig
from ..core.random import set_seed
from ..training.schedules import adamw, cosine_warmup
from ..training.trainer import Trainer
from .build import build_model, build_task

__all__ = ["run", "run_directory"]


def run_directory(config: ExperimentConfig) -> Path:
    """Where this run's artefacts go: readable name, then the config digest."""
    return Path(config.output_dir) / config.run_name(config_hash(config))


def run(config: ExperimentConfig, *, source: Path | None = None, progress: bool = True) -> Path:
    """Train, and return the run directory.

    The directory holds everything needed to re-run or evaluate: the resolved
    config, the weights with the sidecar that rebuilds them, the loss history,
    and -- when :mod:`matplotlib` is present -- a training curve.
    """
    set_seed(config.seed)
    key = jr.PRNGKey(config.seed)
    k_model, k_data, k_fit = jr.split(key, 3)

    task = build_task(config)
    model = build_model(config, task, key=k_model)
    _check_trainable_offline(config.model.name, model)

    directory = run_directory(config)
    directory.mkdir(parents=True, exist_ok=True)
    save(config, directory, source=source)

    train = config.train
    schedule = cosine_warmup(
        train.learning_rate,
        train.steps,
        warmup_steps=max(1, int(train.warmup * train.steps)),
    )
    optimizer = adamw(
        schedule,
        weight_decay=train.weight_decay,
        grad_clip=train.grad_clip or None,
    )
    trainer = Trainer(model, optimizer, ema_momentum=train.ema_momentum)

    from .build import batch_keys

    batches = task.batches(
        batch_size=train.batch_size,
        key=k_data,
        split="train",
        length=train.clip_length,
        stride=train.stride,
        limit=train.limit_episodes,
        observation_key=batch_keys(model)["observation"],
    )
    if progress:
        print(
            f"{config.model.name} on {task.spec.name}: "
            f"{trainer.n_trainable:,} trainable of {model.n_params:,} parameters, "
            f"{train.steps} steps"
        )
    started = time.perf_counter()
    callbacks = []
    recording = _recording(config, directory, "train", "training")
    if recording is not None:
        from ..rerun import log_summary, training_callback

        log_summary(recording, model)
        callbacks.append(training_callback(recording, every=config.log.every))
    if progress:

        def report(state, metrics):
            loss = float(metrics.get("loss", 0.0))
            print(f"\rstep {int(state.step):>6} loss {loss:.5f}   ", end="", flush=True)

        callbacks.append(report)

    state, history = trainer.fit(
        batches, key=k_fit, steps=train.steps, log_every=train.log_every, callbacks=callbacks
    )
    elapsed = time.perf_counter() - started
    if progress:
        print(f"\rtrained {train.steps} steps in {elapsed:.1f}s")

    from ..tools.checkpoint import save as save_model

    save_model(
        directory / "checkpoint" / "model.eqx",
        state.model,
        config={"name": config.model.name, "kwargs": config.model.kwargs, "task": task.spec.name},
    )
    (directory / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    _plot(history, directory)
    if recording is not None:
        recording.close()
        if progress:
            print(f"rerun recording at {directory / 'train.rrd'}")
    return directory


def _check_trainable_offline(name: str, model) -> None:
    """Refuse, with a reason, the models a recording cannot train.

    MuZero's targets are produced by its own tree search during interaction --
    a value and a policy per visited state -- so they exist nowhere in a
    recorded episode. It is perfectly evaluable here, being
    :class:`~xwm.core.types.Plannable` like everything else; it just cannot be
    fitted from an offline corpus without a search loop to label it.
    """
    needs = getattr(model, "value_key", None)
    if needs is not None:
        raise ValueError(
            f"{name} is trained on targets from its own search ({needs!r}, "
            f"{getattr(model, 'policy_key', 'policy_target')!r}), which a recorded "
            "dataset does not contain. Train it against an environment with "
            "xwm.training.ReplayBuffer -- see examples/08_muzero_franka.py -- and "
            "use `xwm eval` on the result."
        )


def _recording(config: ExperimentConfig, directory: Path, stem: str, layout: str):
    """A :class:`xwm.rerun.Recording` for this run, or ``None``. Never fatal.

    Same contract as :func:`_plot`: an optional dependency that is absent, or a
    viewer that refuses to start, must not cost a run that is otherwise fine.
    """
    if not config.log.rerun:
        return None
    try:
        from .. import rerun as xwm_rerun

        return xwm_rerun.Recording(
            "xwm",
            path=directory / f"{stem}.rrd",
            spawn=config.log.spawn,
            connect=config.log.connect or None,
            blueprint=getattr(xwm_rerun.blueprints, layout)(),
        )
    except Exception as error:  # pragma: no cover - depends on the environment
        print(f"warning: rerun logging disabled ({type(error).__name__}: {error})")
        return None


def _plot(history, directory: Path) -> None:
    """A loss curve, when the plotting extra is installed. Never fatal."""
    try:
        from ..plots.curves import plot_history
        from ..plots.export import save_figure
    except Exception:  # pragma: no cover - depends on the environment
        return
    try:
        figure = plot_history(history, keys=("loss",))
        save_figure(figure, directory / "training_curve.png")
    except Exception:  # pragma: no cover - a missing backend must not lose a run
        pass

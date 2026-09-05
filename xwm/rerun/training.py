"""Training-side logging: the loss history, and what the loss cannot tell you.

The reason a training run wants more than a loss curve is
``docs/concepts/collapse.md``'s central point: **the loss is not the metric**. A
collapsing encoder drives its prediction loss *down*, because it is predicting
its own degenerate output. :func:`log_collapse` puts the four collapse
diagnostics on the same timeline as the loss, so the moment one diverges from
the other is visible rather than inferred.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ._backend import rerun

__all__ = ["training_callback", "log_history", "log_collapse", "log_summary"]


def _scalars(rec, entity: str, value: float) -> None:
    rec.log(entity, rerun().Scalars(float(value)))


def training_callback(recording, *, every: int = 1, prefix: str = "train"):
    """A :data:`xwm.training.Callback` that logs every metric the loss returns.

    ``Trainer.fit`` calls its callbacks on the same steps it appends to
    ``history`` (every ``log_every``), so ``every=1`` here means "every logged
    step", not every optimizer step.

    Args:
        recording: a :class:`~xwm.rerun.Recording`, or ``None`` for a no-op.
        every: log one in every ``every`` callback invocations.
        prefix: entity path prefix; metrics land at ``<prefix>/<metric>``.

    Example:
        >>> trainer.fit(batches, steps=1000,                       # doctest: +SKIP
        ...             callbacks=[xwm.rerun.training_callback(rec)])
    """
    if recording is None:
        return lambda state, metrics: None
    state_count = {"n": 0}

    def callback(state, metrics: dict[str, Any]) -> None:
        state_count["n"] += 1
        if (state_count["n"] - 1) % every:
            return
        recording.set_time("step", int(state.step))
        for name, value in metrics.items():
            # Metrics reach a callback as device arrays; history rows are floats.
            _scalars(recording, f"{prefix}/{name}", float(value))

    return callback


def log_history(recording, history, *, prefix: str = "train") -> None:
    """Log a finished ``history`` in one call per metric.

    Columnar rather than row-by-row: a 10,000-step history is one send per
    metric instead of 10,000 logs. Use this after ``fit`` returns, or to import
    a ``history.json`` from a run someone else produced.
    """
    if recording is None or not history:
        return
    rr = rerun()
    steps = np.asarray([row["step"] for row in history], dtype=np.int64)
    names = [k for k in history[0] if k != "step"]
    for name in names:
        values = np.asarray([float(row[name]) for row in history], dtype=np.float64)
        recording.send_columns(
            f"{prefix}/{name}",
            [rr.TimeColumn("step", sequence=steps)],
            rr.Scalars.columns(scalars=values),
        )


def log_collapse(recording, z, *, step: int | None = None, prefix: str = "collapse") -> None:
    """Log the collapse diagnostics, and the spectrum they summarise.

    Args:
        z: any array of embeddings; every leading axis is a sample, as in
            :func:`xwm.metrics.collapse_report`.
        step: sets the ``step`` timeline first. Omit it inside a callback that
            has already set the time.

    Logs ``<prefix>/{rankme,rank_ratio,feature_std,mean_cosine}`` as scalars and
    ``<prefix>/spectrum`` as a bar chart of the singular values. All four are
    reported because each misses a case the others catch.
    """
    if recording is None:
        return
    from ..metrics.representation import collapse_report, singular_values

    rr = rerun()
    if step is not None:
        recording.set_time("step", int(step))
    for name, value in collapse_report(z).items():
        _scalars(recording, f"{prefix}/{name}", float(value))
    recording.log(f"{prefix}/spectrum", rr.BarChart(np.asarray(singular_values(z), np.float64)))


def log_summary(recording, model, *, entity: str = "model/summary", max_depth: int = 2) -> None:
    """Log the model's parameter tree as a static text document.

    Static, so it is visible at every point on every timeline: it describes the
    run rather than a moment in it.
    """
    if recording is None:
        return
    from ..tools.summary import summary

    rr = rerun()
    recording.log(entity, rr.TextDocument(summary(model, max_depth=max_depth)), static=True)

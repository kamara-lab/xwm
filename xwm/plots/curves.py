"""Training-curve and spectrum plots."""

from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp

from ..core.types import Array
from ._backend import pyplot
from .style import (
    CURVE_PALETTE,
    MAGNITUDE_PALETTE,
    SEQUENTIAL_CMAP,
    plot_style,
    series_style,
)


def plot_history(
    history: Sequence[dict[str, float]],
    *,
    keys: Sequence[str] | None = None,
    ax=None,
    logy: bool = False,
    title: str | None = None,
    ylabel: str = "loss",
    label_last: bool = True,
    palette: str = CURVE_PALETTE,
):
    """Plot metrics from :meth:`xwm.training.Trainer.fit`'s history.

    Args:
        history: the list of metric dicts ``fit`` returns.
        keys: which metrics to draw; defaults to every ``loss*`` key.
        label_last: annotate each series' final value directly on the plot, so
            the reader does not have to match a colour to a legend entry to
            learn the number that matters.
    """
    plt = pyplot()
    if not history:
        raise ValueError("history is empty")
    if keys is None:
        keys = [k for k in history[0] if k.startswith("loss")]
    if not keys:
        raise ValueError("no metrics to plot")
    steps = [row["step"] for row in history]

    with plot_style(len(keys), palette):
        ax = ax or plt.subplots(figsize=(6, 3.8))[1]
        # Markers on every point become noise on long runs; show ~12 of them.
        every = max(1, len(steps) // 12)
        for i, key in enumerate(keys):
            values = [row[key] for row in history]
            style = series_style(i, len(keys), palette)
            ax.plot(steps, values, label=key, markevery=every, **style)
            if label_last:
                ax.annotate(
                    f"{values[-1]:.4g}",
                    (steps[-1], values[-1]),
                    textcoords="offset points",
                    xytext=(6, 0),
                    fontsize=8,
                    color=style["color"],
                    va="center",
                )
        ax.set_xlabel("step")
        ax.set_ylabel(ylabel)
        if logy:
            ax.set_yscale("log")
        if title:
            ax.set_title(title)
        # A legend is always present for two or more series.
        if len(keys) > 1:
            ax.legend(loc="upper right")
        ax.margins(x=0.12)
    return ax


def plot_spectrum(
    z: Array,
    *,
    ax=None,
    title: str | None = None,
    label: str | None = None,
    palette: str = CURVE_PALETTE,
):
    """Plot the normalised singular-value spectrum of an embedding matrix.

    A healthy representation decays gently; a collapsing one falls off a cliff
    after a handful of directions.
    """
    plt = pyplot()
    from ..metrics.representation import rankme, singular_values

    s = singular_values(z)
    with plot_style(1, palette):
        ax = ax or plt.subplots(figsize=(5.5, 3.8))[1]
        style = series_style(0, 1, palette)
        style.pop("marker")
        ax.plot(
            jnp.arange(1, s.shape[0] + 1),
            s / s[0],
            label=label or f"RankMe {float(rankme(z)):.1f} / {z.shape[-1]}",
            **style,
        )
        ax.set_yscale("log")
        ax.set_xlabel("singular value index")
        ax.set_ylabel("magnitude (normalised)")
        ax.set_title(title or f"RankMe = {float(rankme(z)):.1f} / {z.shape[-1]}")
    return ax


def plot_spectra(
    named: dict[str, Array],
    *,
    ax=None,
    title: str = "Embedding spectra",
    palette: str = MAGNITUDE_PALETTE,
):
    """Overlay several embedding spectra, one line per named model.

    The single most legible collapse diagnostic: a collapsed encoder's spectrum
    plunges after a few directions while a healthy one decays gently.
    """
    plt = pyplot()
    from ..metrics.representation import rankme, singular_values

    names = list(named)
    with plot_style(len(names), palette):
        ax = ax or plt.subplots(figsize=(6, 3.8))[1]
        for i, name in enumerate(names):
            s = singular_values(named[name])
            style = series_style(i, len(names), palette)
            style["marker"] = "None"
            ax.plot(
                jnp.arange(1, s.shape[0] + 1),
                s / s[0],
                label=f"{name} (RankMe {float(rankme(named[name])):.0f})",
                **style,
            )
        ax.set_yscale("log")
        ax.set_xlabel("singular value index")
        ax.set_ylabel("magnitude (normalised)")
        ax.set_title(title)
        if len(names) > 1:
            ax.legend(loc="lower left")
    return ax


def plot_bars(
    labels: Sequence[str],
    values: Sequence[float],
    *,
    ax=None,
    title: str | None = None,
    ylabel: str | None = None,
    annotate: bool = True,
    horizontal: bool = True,
    palette: str = MAGNITUDE_PALETTE,
):
    """A bar chart for comparing a single metric across a few configurations.

    Horizontal by default: configuration names are long, and rotated tick labels
    are harder to read than a horizontal bar's left-aligned label.
    """
    plt = pyplot()
    n = len(labels)
    if n != len(values):
        raise ValueError(f"{n} labels but {len(values)} values")
    with plot_style(min(n, 5), palette):
        ax = ax or plt.subplots(figsize=(6, 0.5 * n + 1.6))[1]
        from .style import palette_colors

        colors = palette_colors(min(n, 5), palette)
        colors = [colors[i % len(colors)] for i in range(n)]
        positions = range(n)
        if horizontal:
            ax.barh(list(positions), list(values), color=colors, height=0.62)
            ax.set_yticks(list(positions), labels)
            ax.invert_yaxis()
            ax.grid(axis="x", visible=True)
            ax.grid(axis="y", visible=False)
            if ylabel:
                ax.set_xlabel(ylabel)
            if annotate:
                span = max(values) - min(min(values), 0.0) or 1.0
                for pos, value in zip(positions, values, strict=True):
                    ax.annotate(
                        f"{value:.4g}",
                        (value, pos),
                        textcoords="offset points",
                        xytext=(5, 0),
                        va="center",
                        fontsize=8,
                    )
                ax.set_xlim(right=max(values) + 0.16 * span)
        else:
            ax.bar(list(positions), list(values), color=colors, width=0.62)
            ax.set_xticks(list(positions), labels, rotation=20, ha="right")
            if ylabel:
                ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title)
    return ax


def plot_horizon(
    horizons: Sequence[int],
    series: dict[str, Sequence[float]],
    *,
    ax=None,
    title: str = "Rollout error by horizon",
    ylabel: str = "latent L1 error",
    palette: str = CURVE_PALETTE,
):
    """Error-versus-horizon curves -- the compounding-error picture."""
    plt = pyplot()
    names = list(series)
    with plot_style(len(names), palette):
        ax = ax or plt.subplots(figsize=(6, 3.8))[1]
        for i, name in enumerate(names):
            ax.plot(
                list(horizons),
                list(series[name]),
                label=name,
                **series_style(i, len(names), palette),
            )
        ax.set_xlabel("prediction horizon (steps)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(list(horizons))
        if len(names) > 1:
            ax.legend(loc="lower right")
    return ax


__all__ = [
    "CURVE_PALETTE",
    "MAGNITUDE_PALETTE",
    "SEQUENTIAL_CMAP",
    "plot_bars",
    "plot_history",
    "plot_horizon",
    "plot_spectra",
    "plot_spectrum",
]

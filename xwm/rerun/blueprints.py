"""Default view layouts, one per kind of run.

A blueprint is what makes a recording readable the moment it opens rather than
after five minutes of dragging panels. These are deliberately coarse -- they
name origins, not every entity -- so a recording with extra entities still
lays out sensibly and the viewer's own controls remain useful.
"""

from __future__ import annotations

from ._backend import blueprint as _blueprint

__all__ = ["training", "episode", "franka"]


def training():
    """Losses beside the collapse diagnostics, with the parameter tree.

    The two time series sit side by side because the point of logging both is
    to see them disagree: a prediction loss falling while ``feature_std`` falls
    with it is collapse, not learning.
    """
    rrb = _blueprint()
    return rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(
                rrb.TimeSeriesView(name="loss", origin="/train"),
                rrb.TimeSeriesView(name="collapse", origin="/collapse"),
            ),
            rrb.Horizontal(
                rrb.BarChartView(name="spectrum", origin="/collapse/spectrum"),
                rrb.TextDocumentView(name="model", origin="/model/summary"),
            ),
            row_shares=[3, 2],
        ),
    )


def episode():
    """The observation, the distance to goal, and the plan behind them."""
    rrb = _blueprint()
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial2DView(name="observation", origin="/episode"),
            rrb.Vertical(
                rrb.TimeSeriesView(name="distance", origin="/episode"),
                rrb.TimeSeriesView(name="plan", origin="/plan"),
            ),
            column_shares=[2, 3],
        ),
    )


def franka():
    """The arm in 3-D, the camera's view of it, and the reach distance."""
    rrb = _blueprint()
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(name="scene", origin="/franka"),
            rrb.Vertical(
                rrb.Spatial2DView(name="camera", origin="/franka/camera"),
                rrb.TimeSeriesView(name="signals", origin="/franka/signal"),
            ),
            column_shares=[3, 2],
        ),
    )

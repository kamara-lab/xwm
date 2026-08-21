"""Lazy matplotlib import with an actionable error message."""

from __future__ import annotations


def pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "xwm.plots needs matplotlib; install it with `pip install xwm[plots]`"
        ) from exc
    return plt

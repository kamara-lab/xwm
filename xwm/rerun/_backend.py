"""Lazy Rerun import with an actionable error message.

The same shape as :mod:`xwm.plots._backend`, and for the same reason:
``xwm/__init__.py`` imports this package eagerly, so nothing here may touch
``rerun`` at module scope. A bare install must import cleanly and fail only
when a helper that genuinely needs the SDK is called.
"""

from __future__ import annotations

from importlib.util import find_spec

__all__ = ["available", "rerun", "blueprint"]


def rerun():
    """The ``rerun`` module, or an :exc:`ImportError` saying how to get it."""
    try:
        import rerun as rr
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "xwm.rerun needs the Rerun SDK; install it with `pip install xwm[rerun]`"
        ) from exc
    return rr


def blueprint():
    """The ``rerun.blueprint`` module, under the same contract as :func:`rerun`."""
    rerun()  # raise the actionable error first if the SDK is missing at all
    import rerun.blueprint as rrb

    return rrb


def available() -> bool:
    """Whether the Rerun SDK can be imported.

    The counterpart of :func:`xwm.envs.which_backends` and
    :func:`xwm.datasets.which_readers`: ask before a long run rather than
    finding out from a traceback at the end of one.
    """
    try:
        return find_spec("rerun") is not None
    except (ImportError, ValueError):  # pragma: no cover - broken installs
        return False

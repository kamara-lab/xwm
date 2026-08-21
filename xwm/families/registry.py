"""Family registry: build any model by name.

Useful for sweeps and config-driven experiments. The domain packages
(:mod:`xwm.families.jepa`, :mod:`~xwm.families.tdmpc2`,
:mod:`~xwm.families.muzero`) are the better entry point when you know which
model you want.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..core.module import WorldModel

__all__ = ["REGISTRY", "available", "create", "families"]


def _jepa_image(**kw):
    from .jepa.recipes import ijepa

    return ijepa(**kw)


def _jepa_image_lejepa(**kw):
    from .jepa.recipes import lejepa

    return lejepa(**kw)


def _jepa_video(**kw):
    from .jepa.recipes import vjepa

    return vjepa(**kw)


def _jepa_video_lejepa(**kw):
    from .jepa.recipes import video_lejepa

    return video_lejepa(**kw)


def _jepa_action(**kw):
    from .jepa.action import action_world_model

    return action_world_model(**kw)


def _tdmpc2(**kw):
    from .tdmpc2.model import tdmpc2

    return tdmpc2(**kw)


def _muzero(**kw):
    from .muzero.model import muzero

    return muzero(**kw)


#: name -> constructor. All are keyword-only; ``key`` is optional.
REGISTRY: dict[str, Callable[..., WorldModel]] = {
    "jepa/image": _jepa_image,
    "jepa/image-lejepa": _jepa_image_lejepa,
    "jepa/video": _jepa_video,
    "jepa/video-lejepa": _jepa_video_lejepa,
    "jepa/action": _jepa_action,
    "tdmpc2": _tdmpc2,
    "muzero": _muzero,
}


def available() -> list[str]:
    """Registered model names."""
    return sorted(REGISTRY)


def families() -> list[str]:
    """The family each name belongs to."""
    return sorted({name.split("/")[0] for name in REGISTRY})


def create(name: str, **kwargs: Any) -> WorldModel:
    """Build a registered model by name."""
    if name not in REGISTRY:
        raise KeyError(f"unknown model {name!r}; available: {available()}")
    return REGISTRY[name](**kwargs)

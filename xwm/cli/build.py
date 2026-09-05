"""Turning a config into a model and a task, with the shape arguments injected.

The one rule worth stating: a config never says ``action_dim``, ``img_size`` or
``in_channels``. Those come from the :class:`~xwm.tasks.TaskSpec`, so a model
cannot be built that disagrees with the data it is about to be trained on --
the failure mode being a model that trains cleanly on 10-wide grouped actions
because the config said 2, and then plans nonsense.
"""

from __future__ import annotations

from typing import Any

from ..config.schema import ExperimentConfig
from ..core.random import resolve_key

__all__ = ["build_model", "build_task", "model_kwargs"]


def build_task(config: ExperimentConfig):
    from .. import tasks

    return tasks.create(config.task.name, **config.task.overrides)


#: Families whose encoder takes images and has no state-vector variant.
_VISION_ONLY = ("jepa/", "dinowm")

#: Families that condition on a window of frames rather than one.
_NEEDS_HISTORY = ("jepa/lewm", "jepa/delta", "dinowm")


def model_kwargs(config: ExperimentConfig, task) -> dict[str, Any]:
    """The config's kwargs, plus everything the task determines.

    Shape arguments already present in the config are an error rather than an
    override: silently honouring them is how a model ends up disagreeing with
    its data.
    """
    spec = task.spec
    injected: dict[str, Any] = {}
    name = config.model.name
    if name.startswith("muzero"):
        injected["n_actions"] = 2 * spec.blocked_action_dim + 1
    else:
        injected["action_dim"] = spec.blocked_action_dim

    if name in _NEEDS_HISTORY:
        # A windowed model's history is the task's, not a free parameter: the
        # benchmark's Context and the predictor's temporal embedding have to
        # agree, and TaskSpec.clip_length is derived from the same number.
        injected["history"] = spec.history

    vision_only = name.startswith(_VISION_ONLY)
    if "video" in spec.observation:
        size = spec.image_size if spec.image_size is not None else spec.native_size[0]
        injected["img_size"] = size if isinstance(size, int) else size[0]
        if not vision_only:
            injected["observation"] = "image"
    elif vision_only:
        # The JEPA families build a VisionEncoder over a token grid; there is no
        # state-vector variant, and injecting state_dim would surface as a
        # TypeError from a constructor that never mentions the task.
        raise ValueError(
            f"model {name!r} encodes images, but task {spec.name!r} observes "
            f"{list(spec.observation)}. Give the task an image observation, or use a "
            "family with a state encoder (tdmpc2, muzero)."
        )
    else:
        injected["state_dim"] = spec.state_dim
        injected["observation"] = "state"

    clash = sorted(set(injected) & set(config.model.kwargs))
    if clash:
        raise ValueError(
            f"model.kwargs sets {clash}, which task {spec.name!r} determines "
            f"({', '.join(f'{k}={injected[k]!r}' for k in clash)}). Change the task, "
            "or its overrides, rather than the model."
        )
    return {**injected, **config.model.kwargs}


def build_model(config: ExperimentConfig, task, *, key=None):
    """Construct the model named by the config, sized by the task."""
    from .. import families

    kwargs = model_kwargs(config, task)
    model = families.create(config.model.name, key=resolve_key(key), **kwargs)
    if config.model.checkpoint:
        from ..tools.checkpoint import load

        model = load(config.model.checkpoint, model)
    return model


def batch_keys(model) -> dict[str, str]:
    """What this model calls the fields of a batch.

    Read off the model rather than inferred from its class: every family
    declares its keys as static fields (``video_key`` on the JEPA models,
    ``obs_key`` on TD-MPC2 and MuZero) precisely so a caller need not know which
    family it is holding.
    """
    return {
        "observation": getattr(model, "video_key", None) or getattr(model, "obs_key", "video"),
        "action": getattr(model, "action_key", "action"),
        "reward": getattr(model, "reward_key", "reward"),
    }

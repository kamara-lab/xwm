"""Saving and loading models.

Equinox serialises the *leaves* of a PyTree, not its structure, so loading
requires an object of the right shape to load into ("``like``"). That is a
feature, not a limitation: the architecture stays in code, and a checkpoint can
never silently reconstruct a model that differs from the one you asked for. The
config JSON written alongside is there so you can reconstruct that object.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import equinox as eqx

from ..core.types import PyTree


def save(path: str | Path, model: PyTree, *, config: dict[str, Any] | None = None) -> Path:
    """Serialise ``model`` to ``path``, optionally with a config sidecar.

    Args:
        path: destination file; parent directories are created.
        config: JSON-serialisable constructor arguments, written to
            ``<path>.json`` so :func:`load` has something to rebuild from.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(path, model)
    if config is not None:
        path.with_suffix(path.suffix + ".json").write_text(json.dumps(config, indent=2))
    return path


def load(path: str | Path, like: PyTree) -> PyTree:
    """Load parameters from ``path`` into a model with ``like``'s structure.

    Build ``like`` exactly as the saved model was built (same sizes, same keys
    are not required -- only the same shapes).
    """
    return eqx.tree_deserialise_leaves(Path(path), like)


def load_config(path: str | Path) -> dict[str, Any]:
    """Read the config sidecar written by :func:`save`."""
    path = Path(path)
    sidecar = path.with_suffix(path.suffix + ".json")
    if not sidecar.exists():
        raise FileNotFoundError(f"no config sidecar at {sidecar}")
    return json.loads(sidecar.read_text())


def save_state(path: str | Path, state: PyTree) -> Path:
    """Serialise a full :class:`~xwm.training.TrainState` (model, teacher, optimizer)."""
    return save(path, state)


def load_state(path: str | Path, like: PyTree) -> PyTree:
    """Restore a :class:`~xwm.training.TrainState`.

    Build ``like`` with ``Trainer.init()`` on a freshly constructed model, which
    allocates the teacher and optimizer state with the right shapes.
    """
    return load(path, like)

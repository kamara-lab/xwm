"""Shared type aliases and structural protocols for xwm.

Shape conventions used throughout the library
---------------------------------------------
Modules are written for a **single, unbatched sample** and are vmapped by the
caller (this is the idiomatic Equinox style and keeps the code free of leading
batch axes). Batch-level entry points -- everything named ``loss`` -- take
batched inputs and vmap internally, because objectives such as SIGReg and
VICReg are defined over batch statistics.

==========  ==========================================  ==========================
symbol      meaning                                     typical carrier
==========  ==========================================  ==========================
``B``       batch                                       batched ``loss`` inputs
``T``       time / frames                                video tensors
``N``       tokens (patches)                             encoder outputs
``D``       embedding width                              encoder outputs
``A``       action dimensionality                        action tensors
``K``       tokens selected by a mask                    mask index arrays
==========  ==========================================  ==========================

Images are ``(C, H, W)``, videos are ``(T, C, H, W)``, token sequences are
``(N, D)``, and masks are ``int32`` index arrays into the flattened token grid.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

import jax

Array = jax.Array
PRNGKey = jax.Array
PyTree = Any

# A batch is a dict so that models can pick out the fields they need
# ("image", "video", "action", "context", "targets", ...) without the trainer
# knowing anything about the modality.
Batch = dict[str, Any]

# Auxiliary scalars returned alongside a loss, for logging.
Metrics = dict[str, Array]

# One observation of an environment, in the vocabulary recorded datasets use:
# keys from :data:`xwm.datasets.spec.FRAME_FIELDS` -- ``"image"`` ``(C, H, W)``
# uint8, ``"state"`` ``(S,)``, ``"position"``. An environment emits the same
# thing a dataset reader does, so one preprocessing function serves both.
Frame = dict[str, Any]


@runtime_checkable
class Encoder(Protocol):
    """Maps one sample to a token sequence ``(N, D)``."""

    embed_dim: int

    def __call__(self, x: Array, *, key: PRNGKey | None = ...) -> Array: ...


@runtime_checkable
class Predictor(Protocol):
    """Predicts target-position embeddings from context tokens.

    ``context`` is ``(K_ctx, D)`` with flat grid positions ``context_idx``;
    the return value is ``(K_tgt, D)`` aligned with ``target_idx``.
    """

    def __call__(
        self,
        context: Array,
        context_idx: Array,
        target_idx: Array,
        *,
        key: PRNGKey | None = ...,
    ) -> Array: ...


@runtime_checkable
class LatentDynamics(Protocol):
    """One step of action-conditioned latent dynamics: ``(z, a) -> z'``."""

    def __call__(self, z: Array, action: Array, *, key: PRNGKey | None = ...) -> Array: ...


@runtime_checkable
class Plannable(Protocol):
    """What a world model must expose to be driven by :mod:`xwm.bench`.

    Every planner in :mod:`xwm.planning` consumes a ``(z, a) -> z'`` closure and
    a cost over ``z``. This protocol is the rest of the contract: how to build
    the first ``z`` from what the environment has shown so far, and which part
    of ``z`` a goal cost should compare. :class:`~xwm.core.module.WorldModel`
    supplies defaults for the Markov case (``history == 1``, ``z`` is the
    encoding of the latest frame, ``readout`` is the identity), so a model only
    overrides these when its latent state carries more than one frame.
    """

    history: int

    def initial_state(
        self, frames: Array, actions: Array | None = ..., *, key: PRNGKey | None = ...
    ) -> Array:
        """``(H, ...)`` observations and the ``(H - 1, A)`` actions between them -> ``z``."""
        ...

    def dynamics_fn(self, *, key: PRNGKey | None = ...) -> Callable[[Array, Array], Array]:
        """A ``(z, a) -> z'`` closure whose output has the shape of its input."""
        ...

    def readout(self, z: Array) -> Array:
        """The newest frame's latent inside ``z`` -- what a goal is compared to."""
        ...

    def goal_embedding(self, frame: Array, *, key: PRNGKey | None = ...) -> Array:
        """One observation -> the latent :meth:`readout` should reach."""
        ...


@runtime_checkable
class Objective(Protocol):
    """Batch-level training objective, returning ``(scalar_loss, metrics)``."""

    def __call__(self, batch: Batch, *, key: PRNGKey) -> tuple[Array, Metrics]: ...

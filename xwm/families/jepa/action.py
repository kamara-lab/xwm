"""Action-conditioned world models (the V-JEPA 2-AC recipe).

The two-stage structure this implements is the point:

1. Learn a representation of the world from passive video, with no actions --
   :mod:`xwm.video`. Video is abundant; action-labelled interaction is not.
2. *Freeze* that encoder and learn action-conditioned dynamics in its latent
   space, from a comparatively tiny amount of robot interaction data.

Freezing matters for more than compute. With the encoder fixed, the prediction
targets are constant functions of the observations, so the dynamics model has
nothing to gain from collapsing the representation -- the usual failure mode of
jointly training an encoder against its own output disappears.

Training combines two losses, and both are needed:

* **teacher forcing** -- one step ahead from ground-truth latents. Dense,
  well-conditioned signal, but it never shows the model its own mistakes.
* **rollout** -- the full horizon from a single starting latent, with the model
  consuming its own predictions. This is the regime a planner actually uses, and
  the only one that penalises compounding error.

References
----------
Assran et al., *V-JEPA 2: Self-Supervised Video Models Enable Understanding,
Prediction and Planning*, 2025. The two-stage recipe implemented here -- learn a
video encoder from passive data, freeze it, then learn action-conditioned
dynamics in its latent space -- is that paper's "V-JEPA 2-AC" stage.

Bardes et al., *Revisiting Feature Prediction for Learning Visual
Representations from Video* (V-JEPA), 2024. arXiv:2404.08471
"""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from ...core.ema import stop_gradient
from ...core.module import WorldModel
from ...core.random import resolve_key
from ...core.rollout import rollout, teacher_forced_rollout
from ...core.types import Array, Batch, Metrics, PRNGKey, PyTree
from ...dynamics.action_conditioned import ActionConditionedPredictor, Conditioning
from ...dynamics.action_embed import ContinuousActionEmbed
from ...encoders.image import ImageEncoder
from ...encoders.presets import preset
from ...encoders.vision import VisionEncoder
from ...objectives.prediction import LossKind, prediction_loss
from ...objectives.sigreg import sigreg


class ActionWorldModel(WorldModel):
    """A frozen (or fine-tuned) observation encoder plus learned latent dynamics.

    Args:
        encoder: maps one observation to ``(N, D)`` tokens. Applied per frame,
            so an :class:`~xwm.image.ImageEncoder` is the usual choice; a
            :class:`~xwm.video.VideoEncoder` works if you feed it short clips.
        dynamics: the one-step action-conditioned model.
        freeze_encoder: exclude the encoder from :meth:`trainable`, so the
            optimizer never touches it and allocates no state for it.
        teacher_forcing_weight, rollout_weight: mixture of the two losses.
        loss_kind: how latents are compared (V-JEPA 2-AC uses L1).
        collapse: ``"none"`` is correct with a frozen encoder. Use ``"sigreg"``
            if you unfreeze it, so the encoder cannot trivialise its own targets.
        reg_weight, n_proj: SIGReg settings, used only when ``collapse="sigreg"``.
    """

    encoder: VisionEncoder
    dynamics: ActionConditionedPredictor
    freeze_encoder: bool = eqx.field(static=True)
    teacher_forcing_weight: float = eqx.field(static=True)
    rollout_weight: float = eqx.field(static=True)
    loss_kind: LossKind = eqx.field(static=True)
    normalize_target: bool = eqx.field(static=True)
    collapse: str = eqx.field(static=True)
    reg_weight: float = eqx.field(static=True)
    n_proj: int = eqx.field(static=True)
    video_key: str = eqx.field(static=True)
    action_key: str = eqx.field(static=True)

    def __init__(
        self,
        encoder: VisionEncoder,
        dynamics: ActionConditionedPredictor,
        *,
        freeze_encoder: bool = True,
        teacher_forcing_weight: float = 1.0,
        rollout_weight: float = 1.0,
        loss_kind: LossKind = "l1",
        normalize_target: bool = False,
        collapse: str = "none",
        reg_weight: float = 1.0,
        n_proj: int = 256,
        video_key: str = "video",
        action_key: str = "action",
    ):
        if encoder.embed_dim != dynamics.embed_dim:
            raise ValueError(
                f"encoder width {encoder.embed_dim} != dynamics width {dynamics.embed_dim}"
            )
        self.encoder = encoder
        self.dynamics = dynamics
        self.freeze_encoder = freeze_encoder
        self.teacher_forcing_weight = teacher_forcing_weight
        self.rollout_weight = rollout_weight
        self.loss_kind = loss_kind
        self.normalize_target = normalize_target
        self.collapse = collapse
        self.reg_weight = reg_weight
        self.n_proj = n_proj
        self.video_key = video_key
        self.action_key = action_key
        self.uses_target = False

    # -- structure -----------------------------------------------------------
    def trainable(self) -> PyTree:
        """Everything inexact, minus the encoder when it is frozen."""
        spec = super().trainable()
        if self.freeze_encoder:
            spec = eqx.tree_at(
                lambda m: m.encoder,
                spec,
                jax.tree_util.tree_map(lambda _: False, spec.encoder),
            )
        return spec

    @property
    def latent_shape(self) -> tuple[int, int]:
        return (self.encoder.n_tokens, self.encoder.embed_dim)

    def dynamics_fn(self, *, key: PRNGKey | None = None) -> Callable[[Array, Array], Array]:
        """A plain ``(z, a) -> z'`` closure in eval mode, ready for a planner."""
        model = self.dynamics.eval_mode()
        return lambda z, a: model(z, a, key=key)

    # -- inference -----------------------------------------------------------
    def encode(self, observation: Array, *, key: PRNGKey | None = None) -> Array:
        """Encode one observation to ``(N, D)`` latent tokens."""
        return self.encoder(observation, key=key)

    def embed(self, observation: Array) -> Array:
        """Mean-pooled ``(D,)`` representation of one observation, in eval mode.

        Mirrors :meth:`xwm.JEPA.embed`, so a probe or a nearest-neighbour lookup
        works the same way whichever model family produced the encoder.
        """
        return jnp.mean(self.encoder.eval_mode()(observation), axis=0)

    def encode_sequence(self, frames: Array, *, key: PRNGKey | None = None) -> Array:
        """Encode ``(T, ...)`` observations independently to ``(T, N, D)``."""
        encoder = self.encoder if key is not None else self.encoder.eval_mode()
        if key is None:
            return jax.vmap(encoder)(frames)
        keys = jr.split(key, frames.shape[0])
        return jax.vmap(lambda f, k: encoder(f, key=k))(frames, keys)

    def imagine(self, z0: Array, actions: Array) -> Array:
        """Roll the dynamics forward from ``z0``: ``(H, A) -> (H, N, D)``.

        Wrap with :func:`equinox.filter_jit` rather than :func:`jax.jit` -- the
        bound method carries the model's parameters, which plain ``jit`` would
        try to treat as static.
        """
        return rollout(self.dynamics_fn(), z0, actions)

    # -- training ------------------------------------------------------------
    def loss(
        self,
        batch: Batch,
        *,
        key: PRNGKey,
        target: ActionWorldModel | None = None,
    ) -> tuple[Array, Metrics]:
        """Args:
            batch: ``{"video": (B, T, ...), "action": (B, T - 1, A)}``.
                ``action[b, t]`` is the action taken between frames ``t`` and
                ``t + 1``.
        """
        frames, actions = batch[self.video_key], batch[self.action_key]
        if actions.shape[1] != frames.shape[1] - 1:
            raise ValueError(
                f"expected {frames.shape[1] - 1} actions for {frames.shape[1]} frames, "
                f"got {actions.shape[1]}"
            )
        k_enc, k_tf, k_roll, k_reg = jr.split(key, 4)

        keys = jr.split(k_enc, frames.shape[0])
        z = jax.vmap(lambda f, k: self.encode_sequence(f, key=None if self.freeze_encoder else k))(
            frames, keys
        )  # (B, T, N, D)
        if self.freeze_encoder:
            z = stop_gradient(z)
        # Targets are always detached: the dynamics model must chase the
        # representation, never move it toward something easier to predict.
        targets = stop_gradient(z[:, 1:])

        metrics: Metrics = {}
        total = jnp.zeros(())

        if self.teacher_forcing_weight:
            pred = jax.vmap(
                lambda zb, ab, k: teacher_forced_rollout(
                    lambda s, a, *, key=None: self.dynamics(s, a, key=key), zb, ab, key=k
                )
            )(z, actions, jr.split(k_tf, z.shape[0]))
            loss_tf = prediction_loss(
                pred, targets, kind=self.loss_kind, normalize_target=self.normalize_target
            )
            metrics["loss_teacher_forcing"] = loss_tf
            total = total + self.teacher_forcing_weight * loss_tf

        if self.rollout_weight:
            pred = jax.vmap(
                lambda zb, ab, k: rollout(
                    lambda s, a, *, key=None: self.dynamics(s, a, key=key), zb[0], ab, key=k
                )
            )(z, actions, jr.split(k_roll, z.shape[0]))
            loss_roll = prediction_loss(
                pred, targets, kind=self.loss_kind, normalize_target=self.normalize_target
            )
            metrics["loss_rollout"] = loss_roll
            total = total + self.rollout_weight * loss_roll

        if self.collapse == "sigreg":
            reg = sigreg(z, k_reg, n_proj=self.n_proj)
            metrics["loss_reg"] = reg
            total = total + self.reg_weight * reg

        metrics["latent_std"] = jnp.mean(jnp.std(z.reshape(-1, z.shape[-1]), axis=0))
        metrics["loss"] = total
        return total, metrics


def action_world_model(
    *,
    key: PRNGKey | None = None,
    action_dim: int,
    encoder: VisionEncoder | None = None,
    size: str = "small",
    img_size: int | tuple[int, int] = 224,
    patch_size: int = 16,
    in_channels: int = 3,
    action_scale: float = 1.0,
    conditioning: Conditioning = "both",
    freeze_encoder: bool = True,
    dynamics_kwargs: dict | None = None,
    **model_kwargs,
) -> ActionWorldModel:
    """Assemble an :class:`ActionWorldModel`.

    Pass ``encoder`` to build dynamics on top of an existing (typically
    pretrained and frozen) encoder -- the V-JEPA 2-AC setup. Omit it and a fresh
    :class:`~xwm.image.ImageEncoder` of the given size is created, which is what
    you want for a from-scratch experiment.
    """
    k_enc, k_act, k_dyn = jr.split(resolve_key(key), 3)
    if encoder is None:
        encoder = ImageEncoder(
            key=k_enc,
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            **preset(size, "encoder"),
        )
    action_embed = ContinuousActionEmbed(
        action_dim, encoder.embed_dim, key=k_act, scale=action_scale
    )
    dyn_cfg = {**preset(size, "predictor"), **(dynamics_kwargs or {})}
    dynamics = ActionConditionedPredictor(
        encoder.grid,
        encoder.embed_dim,
        action_embed,
        key=k_dyn,
        conditioning=conditioning,
        **dyn_cfg,
    )
    return ActionWorldModel(
        encoder, dynamics, freeze_encoder=freeze_encoder, **model_kwargs
    )

"""Autoregressive JEPA: a window of frames, one causal pass, no decoder.

The family :mod:`xwm.families.jepa.action` could not express. ``ActionWorldModel``
is the V-JEPA 2-AC recipe -- freeze a pretrained encoder, learn Markov dynamics
in its latent space -- and it inherits that stage boundary: with a frozen
encoder the targets are fixed, so nothing can collapse, and with an unfrozen one
the model must be told how to avoid it.

:class:`ARWorldModel` is the same idea trained *end to end*, over a window
rather than a single frame. Three published models are this class with different
arguments:

============  =====================  ===================  ============================
model         encoder                collapse handled by  entry point
============  =====================  ===================  ============================
LeWorldModel  ViT, from scratch      SIGReg               :func:`lewm`
Delta-JEPA    ViT, from scratch      action decoding      :func:`delta_jepa`
DINO-WM       DINOv2, frozen         nothing to collapse  :func:`xwm.families.dinowm`
============  =====================  ===================  ============================

The middle column is the whole design space. Predicting a representation from a
representation admits the constant solution, and an encoder trained *through*
its own targets will find it unless something forbids it. LeWorldModel forbids
it distributionally: embeddings must look isotropic Gaussian, and a constant
does not. Delta-JEPA forbids it mechanically: the action must be recoverable
from the latent displacement, and a constant has no displacement. A frozen
encoder cannot collapse because it cannot move.

References
----------
Maes et al., *LeWorldModel: Stable End-to-End Joint-Embedding Predictive
Architecture from Pixels*, 2026. arXiv:2603.19312.

*Delta-JEPA: Learning Action-Sensitive World Models via Latent Difference
Decoding*, 2026. arXiv:2606.31232.

Zhou et al., *DINO-WM: World Models on Pre-trained Visual Features enable
Zero-shot Planning*, ICML 2025. arXiv:2411.04983.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jax.lax import stop_gradient

from ...core.module import WorldModel
from ...core.random import resolve_key
from ...core.rollout import rollout
from ...core.types import Array, Batch, Metrics, PRNGKey, PyTree
from ...dynamics.action_embed import ContinuousActionEmbed
from ...dynamics.autoregressive import ARConditioning, ARPredictor
from ...encoders.image import ImageEncoder
from ...encoders.presets import preset
from ...encoders.vision import VisionEncoder
from ...heads.action_decoder import LatentDifferenceActionDecoder
from ...nn.transformer import mean_pool
from ...objectives.prediction import LossKind, prediction_loss
from ...objectives.sigreg import sigreg

__all__ = ["ARWorldModel", "Window", "lewm", "delta_jepa"]

#: Which latent a frame becomes.
Tokens = Literal["pooled", "patch"]


class Window(NamedTuple):
    """The planning state of a model that conditions on several frames.

    Attributes:
        frames: ``(history, n_tokens, embed_dim)`` -- oldest first.
        actions: ``(history, action_dim)``. ``actions[t]`` is the action taken
            *from* ``frames[t]``; the last row is the slot the planner's next
            action goes into, and is zero until it does.

    A :class:`~typing.NamedTuple` rather than one packed array, so the actions
    are not broadcast across every token of every frame. It is a pytree, which
    is all :func:`xwm.core.rollout` and the planners require of a latent state.
    """

    frames: Array
    actions: Array


class ARWorldModel(WorldModel):
    """An encoder and a block-causal action-conditioned predictor, trained together.

    Args:
        encoder: applied per frame.
        dynamics: an :class:`~xwm.dynamics.ARPredictor` over ``history`` frames.
        tokens: what a frame becomes. ``"pooled"`` averages the token grid to a
            single vector and projects it -- what LeWorldModel does, and around
            two hundred times fewer tokens than the alternative, which is where
            its planning speed comes from. ``"patch"`` keeps the grid, which is
            what DINO-WM needs: pooling DINOv2 patches discards the spatial
            detail that makes them worth using.
        projector: a linear map applied after pooling. Ignored when
            ``tokens="patch"``. Deliberately *not* followed by a layer
            normalisation, which was measured to break the model: normalising
            each embedding to the unit sphere removes the scale SIGReg
            constrains, so the prediction loss can be driven to zero by
            collapsing to a single point on that sphere and the regularizer
            barely notices. LeWorldModel puts BatchNorm here instead, which
            constrains variance *across the batch* and is a different operation
            in the direction that matters. See ``docs/findings.md``.
        freeze_encoder: exclude the encoder from :meth:`trainable`.
        detach_target: stop gradients at the prediction targets. Unlike
            :class:`~xwm.families.jepa.ActionWorldModel` this defaults to
            ``False``, because the point of an end-to-end JEPA is that the
            encoder learns from the prediction task too. That is only safe with
            a countermeasure -- see ``collapse`` and ``action_weight`` -- and
            the constructor refuses the combination that has neither.
        loss_kind: how latents are compared. L2 for both papers.
        collapse: ``"sigreg"`` for LeWorldModel's isotropic-Gaussian penalty,
            ``"none"`` when a frozen encoder or the action decoder is doing the
            work.
        reg_weight, n_proj: SIGReg settings. ``reg_weight`` is the paper's
            single hyperparameter and ``0.1`` is its value there, which is what
            this defaults to -- but it is genuinely a knob and it does need
            turning. On this library's own synthetic pusher, 0.1 and 1.0 both
            let the latent *scale* collapse (``latent_std`` to 6e-4) while 10.0
            and 50.0 hold it near 0.85. Watch ``latent_std``; ``rankme`` will
            look healthy throughout, because the directions survive and only
            the scale goes. ``docs/findings.md`` has the measurements.
        action_weight: weight on decoding the action from the latent
            displacement. Non-zero makes this Delta-JEPA.

    There is deliberately no rollout term. ``ActionWorldModel`` mixes teacher
    forcing with a free rollout because its clips are long and its dynamics are
    Markov; here a clip is exactly ``history + 1`` frames, which is one window,
    so a rollout would have nothing to roll through. Compounding error is
    instead attacked by supervising the short prefixes -- position ``0`` of the
    window predicts from a single frame, and that is the regime a free rollout
    degrades into. Measure it with :meth:`imagine`.
    """

    encoder: VisionEncoder
    dynamics: ARPredictor
    projector: eqx.nn.Linear | None
    action_decoder: LatentDifferenceActionDecoder | None
    tokens: Tokens = eqx.field(static=True)
    freeze_encoder: bool = eqx.field(static=True)
    detach_target: bool = eqx.field(static=True)
    loss_kind: LossKind = eqx.field(static=True)
    normalize_target: bool = eqx.field(static=True)
    collapse: str = eqx.field(static=True)
    reg_weight: float = eqx.field(static=True)
    n_proj: int = eqx.field(static=True)
    action_weight: float = eqx.field(static=True)
    video_key: str = eqx.field(static=True)
    action_key: str = eqx.field(static=True)

    def __init__(
        self,
        encoder: VisionEncoder,
        dynamics: ARPredictor,
        *,
        projector: eqx.nn.Linear | None = None,
        action_decoder: LatentDifferenceActionDecoder | None = None,
        tokens: Tokens = "pooled",
        freeze_encoder: bool = False,
        detach_target: bool = False,
        loss_kind: LossKind = "l2",
        normalize_target: bool = False,
        collapse: str = "sigreg",
        reg_weight: float = 0.1,
        n_proj: int = 1024,
        action_weight: float = 0.0,
        video_key: str = "video",
        action_key: str = "action",
    ):
        # What the dynamics sees is the projector's output when there is one,
        # not the encoder's width: `latent_dim` is allowed to differ.
        width = encoder.embed_dim if projector is None else projector.out_features
        if width != dynamics.embed_dim:
            source = "encoder" if projector is None else "projector"
            raise ValueError(f"{source} width {width} != dynamics width {dynamics.embed_dim}")
        expected = 1 if tokens == "pooled" else encoder.n_tokens
        if dynamics.n_tokens != expected:
            raise ValueError(
                f"dynamics expects {dynamics.n_tokens} tokens per frame but tokens={tokens!r} "
                f"on this encoder gives {expected}"
            )
        if action_weight and action_decoder is None:
            raise ValueError("action_weight > 0 needs an action_decoder")
        # The one combination that trains a constant: a live encoder, targets it
        # can move, and nothing forbidding the trivial solution. Refused here
        # rather than discovered after an overnight run whose loss went to zero.
        if not freeze_encoder and not detach_target and collapse == "none" and not action_weight:
            raise ValueError(
                "an unfrozen encoder trained through its own targets will collapse: set "
                "collapse='sigreg' (LeWorldModel), action_weight>0 (Delta-JEPA), "
                "detach_target=True, or freeze_encoder=True"
            )
        self.encoder = encoder
        self.dynamics = dynamics
        self.projector = projector
        self.action_decoder = action_decoder
        self.tokens = tokens
        self.freeze_encoder = freeze_encoder
        self.detach_target = detach_target
        self.loss_kind = loss_kind
        self.normalize_target = normalize_target
        self.collapse = collapse
        self.reg_weight = reg_weight
        self.n_proj = n_proj
        self.action_weight = action_weight
        self.video_key = video_key
        self.action_key = action_key
        self.uses_target = False
        self.history = dynamics.history

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
    def latent_width(self) -> int:
        """Width of one frame's latent -- the projector's output where there is one."""
        return self.encoder.embed_dim if self.projector is None else self.projector.out_features

    @property
    def latent_shape(self) -> tuple[int, int]:
        """One frame's latent: ``(n_tokens, latent_width)``."""
        return (self.dynamics.n_tokens, self.latent_width)

    # -- inference -----------------------------------------------------------
    def encode(self, observation: Array, *, key: PRNGKey | None = None) -> Array:
        """Encode one observation to ``(n_tokens, embed_dim)``."""
        z = self.encoder(observation, key=key)
        if self.tokens == "patch":
            return z
        pooled = mean_pool(z)
        if self.projector is not None:
            pooled = self.projector(pooled)
        return pooled[None]

    def embed(self, observation: Array) -> Array:
        """Mean-pooled ``(D,)`` representation, in eval mode. Mirrors :meth:`xwm.JEPA.embed`."""
        return mean_pool(self.eval_mode().encode(observation))

    def encode_sequence(self, frames: Array, *, key: PRNGKey | None = None) -> Array:
        """Encode ``(T, ...)`` observations independently to ``(T, n_tokens, embed_dim)``."""
        model = self if key is not None else self.eval_mode()
        if key is None:
            return jax.vmap(model.encode)(frames)
        keys = jr.split(key, frames.shape[0])
        return jax.vmap(lambda f, k: model.encode(f, key=k))(frames, keys)

    def predict(self, z: Array, actions: Array, *, key: PRNGKey | None = None) -> Array:
        """One causal pass: ``(H, N, D) x (H, A) -> (H, N, D)`` of next-frame latents."""
        return self.dynamics(z, actions, key=key)

    # -- planning ------------------------------------------------------------
    def initial_state(
        self, frames: Array, actions: Array | None = None, *, key: PRNGKey | None = None
    ) -> Window:
        """Build the planning window from ``history`` frames and the actions between them.

        ``frames`` is ``(history, ...)`` oldest first, as
        :class:`xwm.bench.Context` supplies it, and ``actions`` is the
        ``history - 1`` actions that joined them. The newest action slot is left
        at zero: it is what the planner is about to choose.
        """
        z = self.encode_sequence(frames, key=key)
        history, action_dim = self.history, self.dynamics.action_embed.action_dim
        past = jnp.zeros((history, action_dim), z.dtype)
        if actions is not None and history > 1:
            past = past.at[: history - 1].set(jnp.asarray(actions, z.dtype)[-(history - 1) :])
        return Window(frames=z, actions=past)

    def readout(self, z: Window) -> Array:
        """The newest frame's latent -- what a goal cost should be compared against.

        Without this the cost would also charge the window's stale context for
        failing to move, which no action can fix.
        """
        return z.frames[-1]

    def goal_embedding(self, frame: Array, *, key: PRNGKey | None = None) -> Array:
        """Encode a goal observation into the space :meth:`readout` returns."""
        return self.encode(frame, key=key)

    def dynamics_fn(self, *, key: PRNGKey | None = None) -> Callable[[Window, Array], Window]:
        """A plain ``(window, a) -> window`` closure in eval mode, ready for a planner.

        Each call fills the empty action slot, predicts the next frame from the
        whole window, and slides: the oldest frame and its action fall off the
        back, the prediction joins the front.
        """
        model = self.dynamics.eval_mode()

        def step(window: Window, action: Array) -> Window:
            actions = window.actions.at[-1].set(action)
            predicted = model(window.frames, actions, key=key)[-1]
            return Window(
                frames=jnp.concatenate([window.frames[1:], predicted[None]], axis=0),
                actions=jnp.concatenate(
                    [actions[1:], jnp.zeros_like(actions[-1])[None]], axis=0
                ),
            )

        return step

    def imagine(self, window: Window, actions: Array) -> Window:
        """Roll the dynamics forward: ``(H, A)`` actions -> a stacked window per step."""
        return rollout(self.dynamics_fn(), window, actions)

    # -- training ------------------------------------------------------------
    def loss(
        self,
        batch: Batch,
        *,
        key: PRNGKey,
        target: ARWorldModel | None = None,
    ) -> tuple[Array, Metrics]:
        """Args:
            batch: ``{"video": (B, T, ...), "action": (B, T - 1, A)}`` with
                ``T = history + 1``. ``action[b, t]`` joins frames ``t`` and
                ``t + 1``.

        One causal pass predicts every frame from its prefix, so the window is
        supervised at each context length from one frame up to ``history`` --
        including the short prefixes the planner meets at the start of an
        episode, where a model trained only at full context is off-distribution.
        """
        del target
        frames, actions = batch[self.video_key], batch[self.action_key]
        if actions.shape[1] != frames.shape[1] - 1:
            raise ValueError(
                f"expected {frames.shape[1] - 1} actions for {frames.shape[1]} frames, "
                f"got {actions.shape[1]}"
            )
        if actions.shape[1] != self.history:
            raise ValueError(
                f"this model conditions on {self.history} frames, so it needs clips of "
                f"{self.history + 1}; got {frames.shape[1]}. Set the task's history, or "
                f"train.clip_length, to match."
            )
        k_enc, k_dyn, k_reg = jr.split(key, 3)

        keys = jr.split(k_enc, frames.shape[0])
        z = jax.vmap(
            lambda f, k: self.encode_sequence(f, key=None if self.freeze_encoder else k)
        )(frames, keys)  # (B, T, N, D)
        if self.freeze_encoder:
            z = stop_gradient(z)
        context, targets = z[:, :-1], z[:, 1:]
        if self.detach_target:
            targets = stop_gradient(targets)

        metrics: Metrics = {}
        total = jnp.zeros(())

        predicted = jax.vmap(lambda zb, ab, k: self.dynamics(zb, ab, key=k))(
            context, actions, jr.split(k_dyn, z.shape[0])
        )
        loss_pred = prediction_loss(
            predicted, targets, kind=self.loss_kind, normalize_target=self.normalize_target
        )
        metrics["loss_prediction"] = loss_pred
        total = total + loss_pred

        if self.action_weight:
            # Delta-JEPA: the displacement between *encoded* latents, so the
            # decoder constrains the representation rather than the predictor.
            delta = targets - context
            decode = jax.vmap(jax.vmap(self.action_decoder))
            loss_action = jnp.mean(jnp.square(decode(delta) - actions))
            metrics["loss_action"] = loss_action
            total = total + self.action_weight * loss_action

        if self.collapse == "sigreg":
            reg = sigreg(z, k_reg, n_proj=self.n_proj)
            metrics["loss_reg"] = reg
            total = total + self.reg_weight * reg

        metrics["latent_std"] = jnp.mean(jnp.std(z.reshape(-1, z.shape[-1]), axis=0))
        metrics["loss"] = total
        return total, metrics


def _build(
    *,
    key: PRNGKey | None,
    action_dim: int,
    history: int,
    encoder: VisionEncoder | None,
    size: str,
    img_size: int | tuple[int, int],
    patch_size: int,
    in_channels: int,
    tokens: Tokens,
    latent_dim: int | None,
    action_scale: float,
    conditioning: ARConditioning,
    encoder_kwargs: dict | None,
    dynamics_kwargs: dict | None,
    action_decoder_kwargs: dict | None,
    with_action_decoder: bool,
    **model_kwargs,
) -> ARWorldModel:
    """Assemble the encoder, predictor and optional decoder. Shared by the recipes."""
    k_enc, k_proj, k_act, k_dyn, k_dec = jr.split(resolve_key(key), 5)
    if encoder is None:
        encoder = ImageEncoder(
            key=k_enc,
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            **{**preset(size, "encoder"), **(encoder_kwargs or {})},
        )
    width = encoder.embed_dim
    projector = None
    if tokens == "pooled":
        # LeWorldModel projects the pooled embedding before predicting on it.
        # `latent_dim` is free to differ from the encoder's width, but the
        # predictor and the targets both live in the projected space, so the
        # dynamics is built at whatever this produces.
        width = latent_dim or encoder.embed_dim
        projector = eqx.nn.Linear(encoder.embed_dim, width, key=k_proj)
    elif latent_dim is not None:
        raise ValueError("latent_dim applies to tokens='pooled'; patch tokens keep their width")

    grid = (1,) if tokens == "pooled" else tuple(encoder.grid)
    action_embed = ContinuousActionEmbed(action_dim, width, key=k_act, scale=action_scale)
    dynamics = ARPredictor(
        grid,
        width,
        action_embed,
        history=history,
        key=k_dyn,
        conditioning=conditioning,
        **{**preset(size, "predictor"), **(dynamics_kwargs or {})},
    )
    decoder = (
        LatentDifferenceActionDecoder(
            width, action_dim, key=k_dec, **(action_decoder_kwargs or {})
        )
        if with_action_decoder
        else None
    )
    return ARWorldModel(
        encoder,
        dynamics,
        projector=projector,
        action_decoder=decoder,
        tokens=tokens,
        **model_kwargs,
    )


def lewm(
    *,
    key: PRNGKey | None = None,
    action_dim: int,
    history: int = 3,
    encoder: VisionEncoder | None = None,
    size: str = "tiny",
    img_size: int | tuple[int, int] = 224,
    patch_size: int = 14,
    in_channels: int = 3,
    latent_dim: int | None = None,
    action_scale: float = 1.0,
    conditioning: ARConditioning = "film",
    reg_weight: float = 0.1,
    n_proj: int = 1024,
    encoder_kwargs: dict | None = None,
    dynamics_kwargs: dict | None = None,
    **model_kwargs,
) -> ARWorldModel:
    """**LeWorldModel**: end to end from pixels, with two loss terms and one knob.

    A prediction loss and SIGReg, and nothing else -- no EMA teacher, no
    stop-gradient, no pretrained encoder, no schedule. The claim the paper
    makes and this reproduces is that isotropy is enough: if the embedding
    distribution must look like ``N(0, I)``, the constant solution is
    unavailable, and the six loss weights its predecessors needed collapse to
    ``reg_weight``.

    The latent is one pooled vector per frame, which is what makes it fast --
    around two hundred times fewer tokens than patch-level dynamics, and
    planning that finishes in under a second.

    Example:
        >>> model = xwm.families.jepa.lewm(action_dim=2, img_size=64, patch_size=8)
    """
    return _build(
        key=key,
        action_dim=action_dim,
        history=history,
        encoder=encoder,
        size=size,
        img_size=img_size,
        patch_size=patch_size,
        in_channels=in_channels,
        tokens="pooled",
        latent_dim=latent_dim,
        action_scale=action_scale,
        conditioning=conditioning,
        encoder_kwargs=encoder_kwargs,
        dynamics_kwargs=dynamics_kwargs,
        action_decoder_kwargs=None,
        with_action_decoder=False,
        collapse="sigreg",
        reg_weight=reg_weight,
        n_proj=n_proj,
        **model_kwargs,
    )


def delta_jepa(
    *,
    key: PRNGKey | None = None,
    action_dim: int,
    history: int = 3,
    encoder: VisionEncoder | None = None,
    size: str = "tiny",
    img_size: int | tuple[int, int] = 224,
    patch_size: int = 14,
    in_channels: int = 3,
    latent_dim: int | None = None,
    action_scale: float = 1.0,
    conditioning: ARConditioning = "film",
    action_weight: float = 10.0,
    encoder_kwargs: dict | None = None,
    dynamics_kwargs: dict | None = None,
    action_decoder_kwargs: dict | None = None,
    **model_kwargs,
) -> ARWorldModel:
    """**Delta-JEPA**: collapse prevented by making the action recoverable.

    Same architecture as :func:`lewm`, different second term. Instead of
    constraining the *distribution* of embeddings, it constrains their
    *differences*: ``a_t`` must be decodable from ``z_{t+1} - z_t``. Adjacent
    embeddings therefore cannot coincide, and distinct actions must produce
    distinguishable latent displacements -- which is the property a planner
    searching over action sequences actually depends on, and one isotropy does
    not by itself provide.

    ``collapse`` defaults to ``"none"`` here: the paper's ablations find the
    decoder sufficient on its own, and both terms together are untested.

    Example:
        >>> model = xwm.families.jepa.delta_jepa(action_dim=2, img_size=64, patch_size=8)
    """
    model_kwargs.setdefault("collapse", "none")
    return _build(
        key=key,
        action_dim=action_dim,
        history=history,
        encoder=encoder,
        size=size,
        img_size=img_size,
        patch_size=patch_size,
        in_channels=in_channels,
        tokens="pooled",
        latent_dim=latent_dim,
        action_scale=action_scale,
        conditioning=conditioning,
        encoder_kwargs=encoder_kwargs,
        dynamics_kwargs=dynamics_kwargs,
        action_decoder_kwargs=action_decoder_kwargs,
        with_action_decoder=True,
        action_weight=action_weight,
        **model_kwargs,
    )

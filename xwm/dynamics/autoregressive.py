"""Autoregressive latent dynamics over a window of frames.

:class:`~xwm.dynamics.ActionConditionedPredictor` is Markov: one latent, one
action, the next latent. That is the right model when a single observation
determines the state, and the wrong one whenever velocity matters -- a puck's
position tells you nothing about where it is going.

This module conditions on ``history`` frames at once. One block-causal pass
predicts *every* next frame in the window, so a clip of ``H + 1`` frames yields
``H`` prediction targets for the cost of a single forward pass, and the model is
trained at every context length from one frame up to ``H`` rather than only at
the full one -- which is what the planner meets at the start of an episode.

References
----------
Zhou et al., *DINO-WM: World Models on Pre-trained Visual Features enable
Zero-shot Planning*, ICML 2025. arXiv:2411.04983 -- causal attention over a
frame history with the action concatenated to every patch token
(``conditioning="concat"``).

Maes et al., *LeWorldModel: Stable End-to-End Joint-Embedding Predictive
Architecture from Pixels*, 2026. arXiv:2603.19312 -- the same window with the
action applied as a feature-wise modulation (``conditioning="film"``).
"""

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from ..core.module import Module
from ..core.random import resolve_key
from ..core.types import Array, PRNGKey
from ..nn.embed import LearnedPosEmbed, SinCosPosEmbed
from ..nn.transformer import Transformer
from .action_conditioned import RESIDUAL_INIT_SCALE, ActionEmbed

#: How the action reaches the tokens of its own frame.
ARConditioning = Literal["film", "concat", "both"]

__all__ = ["ARPredictor", "ARConditioning", "block_causal_mask"]


def block_causal_mask(history: int, n_tokens: int) -> Array:
    """``(H * N, H * N)`` boolean mask: a frame may attend to itself and earlier ones.

    *Block* causal rather than plain causal: tokens within one frame see each
    other in full, because they are one observation and imposing an order on the
    patches of a single image would be arbitrary. Only the frame index is
    ordered.
    """
    frame = jnp.repeat(jnp.arange(history), n_tokens)
    return frame[:, None] >= frame[None, :]


class ARPredictor(Module):
    """``(H, N, D) x (H, A) -> (H, N, D)``: predict every frame's successor.

    Output ``t`` is the model's prediction of frame ``t + 1``, formed from
    frames ``0..t`` and actions ``0..t``, where action ``t`` is the one taken
    *from* frame ``t``.

    Args:
        grid: token grid of one frame's latent.
        embed_dim: encoder width (input and output).
        action_embed: how an action becomes a vector.
        history: frames in the window. ``1`` makes this a slower spelling of
            :class:`~xwm.dynamics.ActionConditionedPredictor`.
        pred_dim: internal width.
        depth, num_heads: transformer size.
        conditioning: how the action reaches its frame's tokens.

            * ``"film"`` -- a feature-wise scale and shift, per frame. What
              LeWorldModel does (as AdaLN, applied at every layer; here it is
              applied once at the input, as elsewhere in :mod:`xwm.dynamics`).
            * ``"concat"`` -- appended to every token's features before the
              input projection. What DINO-WM does, and the only option that
              keeps the action distinguishable per token when a frame's tokens
              are many and its action is small.
            * ``"both"`` -- the default.
        residual: predict the change rather than the next state outright.
        pos: positional scheme *within* a frame. Position *between* frames is
            always a learned table of ``history`` entries: the window is short
            and its order is the whole point.
    """

    action_embed: ActionEmbed
    embed_in: eqx.nn.Linear
    embed_out: eqx.nn.Linear
    spatial_pos: SinCosPosEmbed | LearnedPosEmbed | None
    temporal_pos: LearnedPosEmbed
    film: eqx.nn.Linear | None
    blocks: Transformer
    history: int = eqx.field(static=True)
    n_tokens: int = eqx.field(static=True)
    embed_dim: int = eqx.field(static=True)
    pred_dim: int = eqx.field(static=True)
    conditioning: ARConditioning = eqx.field(static=True)
    residual: bool = eqx.field(static=True)

    def __init__(
        self,
        grid: tuple[int, ...],
        embed_dim: int,
        action_embed: ActionEmbed,
        *,
        history: int,
        pred_dim: int,
        depth: int,
        num_heads: int,
        key: PRNGKey | None = None,
        conditioning: ARConditioning = "both",
        residual: bool = True,
        pos: str = "sincos",
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        drop_path: float = 0.0,
        layer_scale: float | None = None,
        remat: bool = False,
    ):
        if history < 1:
            raise ValueError(f"history must be at least 1, got {history}")
        if conditioning not in ("film", "concat", "both"):
            raise ValueError(f"unknown conditioning {conditioning!r}")
        key = resolve_key(key)
        k_in, k_out, k_film, k_pos, k_time, k_blocks = jr.split(key, 6)

        n_tokens = 1
        for s in grid:
            n_tokens *= s
        uses_concat = conditioning in ("concat", "both")
        uses_film = conditioning in ("film", "both")

        self.action_embed = action_embed
        width = embed_dim + (action_embed.embed_dim if uses_concat else 0)
        self.embed_in = eqx.nn.Linear(width, pred_dim, key=k_in)

        embed_out = eqx.nn.Linear(pred_dim, embed_dim, key=k_out)
        if residual:
            # As in ActionConditionedPredictor: small, so the model starts near
            # the identity, but not zero, which would also zero the gradient
            # flowing back through the projection and stall everything upstream.
            embed_out = eqx.tree_at(
                lambda m: (m.weight, m.bias),
                embed_out,
                (RESIDUAL_INIT_SCALE * embed_out.weight, RESIDUAL_INIT_SCALE * embed_out.bias),
            )
        self.embed_out = embed_out

        if pos == "sincos":
            self.spatial_pos = SinCosPosEmbed(tuple(grid), pred_dim)
        elif pos == "learned":
            self.spatial_pos = LearnedPosEmbed(n_tokens, pred_dim, key=k_pos)
        elif pos == "none":
            self.spatial_pos = None
        else:
            raise ValueError(f"unknown pos {pos!r}; use 'sincos', 'learned' or 'none'")
        self.temporal_pos = LearnedPosEmbed(history, pred_dim, key=k_time)

        self.film = (
            eqx.tree_at(
                lambda m: (m.weight, m.bias),
                eqx.nn.Linear(action_embed.embed_dim, 2 * pred_dim, key=k_film),
                (
                    0.02 * jr.normal(k_film, (2 * pred_dim, action_embed.embed_dim)),
                    jnp.zeros((2 * pred_dim,)),
                ),
            )
            if uses_film
            else None
        )
        self.blocks = Transformer(
            pred_dim,
            depth,
            num_heads,
            key=k_blocks,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            drop_path=drop_path,
            layer_scale=layer_scale,
            remat=remat,
        )
        self.history = history
        self.n_tokens = n_tokens
        self.embed_dim = embed_dim
        self.pred_dim = pred_dim
        self.conditioning = conditioning
        self.residual = residual

    def __call__(self, z: Array, actions: Array, *, key: PRNGKey | None = None) -> Array:
        """Advance every frame in the window by one step.

        Args:
            z: ``(history, n_tokens, embed_dim)`` -- the window of latents.
            actions: ``(history, action_dim)`` -- ``actions[t]`` is taken *from*
                ``z[t]``. The last row is the action whose result is being
                predicted; earlier rows are the context that got us here.

        Returns:
            ``(history, n_tokens, embed_dim)``; row ``t`` predicts frame
            ``t + 1``, so the caller usually wants the last one.
        """
        if z.shape[0] != self.history or actions.shape[0] != self.history:
            raise ValueError(
                f"expected {self.history} frames and {self.history} actions, got "
                f"{z.shape[0]} and {actions.shape[0]}"
            )
        k_blocks, k_act = (None, None) if key is None else jr.split(key)
        keys = None if k_act is None else jr.split(k_act, self.history)
        a = (
            jax.vmap(self.action_embed)(actions)
            if keys is None
            else jax.vmap(lambda x, k: self.action_embed(x, key=k))(actions, keys)
        )  # (H, Ea)

        h = z
        if self.conditioning in ("concat", "both"):
            per_token = jnp.broadcast_to(a[:, None, :], (*h.shape[:2], a.shape[-1]))
            h = jnp.concatenate([h, per_token], axis=-1)
        h = jax.vmap(jax.vmap(self.embed_in))(h)  # (H, N, P)

        if self.spatial_pos is not None:
            h = h + self.spatial_pos()[None]
        h = h + self.temporal_pos()[:, None, :]
        if self.film is not None:
            scale, shift = jnp.split(jax.vmap(self.film)(a), 2, axis=-1)
            h = h * (1.0 + scale[:, None, :]) + shift[:, None, :]

        flat = h.reshape(self.history * self.n_tokens, self.pred_dim)
        flat = self.blocks(flat, mask=block_causal_mask(self.history, self.n_tokens), key=k_blocks)
        out = jax.vmap(self.embed_out)(flat).reshape(z.shape)
        return z + out if self.residual else out

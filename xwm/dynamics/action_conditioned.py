"""Action-conditioned latent dynamics.

This is the module that turns a *representation* of the world into a *model* of
it: given the latent state of the scene and an action, predict the latent state
that follows. Nothing is decoded to pixels, so the model is free to ignore
detail it cannot control or predict -- which is exactly what makes the learned
dynamics usable for planning.
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
from .action_embed import ContinuousActionEmbed, DiscreteActionEmbed, PoseActionEmbed

ActionEmbed = ContinuousActionEmbed | DiscreteActionEmbed | PoseActionEmbed
Conditioning = Literal["token", "film", "both"]

#: How far a freshly initialised residual dynamics model departs from the
#: identity, as a fraction of the default projection scale.
RESIDUAL_INIT_SCALE = 0.05


class ActionConditionedPredictor(Module):
    """``(z, a) -> z'`` over a token grid, satisfying :class:`~xwm.core.types.LatentDynamics`.

    Args:
        grid: token grid of the latent state.
        embed_dim: encoder width (input and output).
        action_embed: how the action becomes a vector.
        pred_dim: internal width.
        depth, num_heads: transformer size.
        conditioning: how the action reaches the tokens.

            * ``"token"`` -- prepend the action as an extra token and let
              attention route it. Most expressive, but slowest to get going:
              attention has to *learn* to read the token before the action
              influences anything.
            * ``"film"`` -- feature-wise scale and shift applied to every token.
              Cheapest, and reaches every token directly at depth zero.
            * ``"both"`` -- the default. All three converge to similar action
              sensitivity, and this one gets there first.
        residual: predict the *change* in latent state rather than the next
            state outright. On by default: consecutive latents are nearly
            identical, so starting close to the identity map is a far better
            prior than starting from noise. The output projection is scaled down
            by :data:`RESIDUAL_INIT_SCALE` in this mode.
        pos: positional scheme (additive only -- the action token has no grid
            position, so rotary embeddings do not apply here).
    """

    action_embed: ActionEmbed
    embed_in: eqx.nn.Linear
    embed_out: eqx.nn.Linear
    pos_embed: SinCosPosEmbed | LearnedPosEmbed | None
    action_token: eqx.nn.Linear | None
    film: eqx.nn.Linear | None
    blocks: Transformer
    grid: tuple[int, ...] = eqx.field(static=True)
    embed_dim: int = eqx.field(static=True)
    pred_dim: int = eqx.field(static=True)
    conditioning: Conditioning = eqx.field(static=True)
    residual: bool = eqx.field(static=True)

    def __init__(
        self,
        grid: tuple[int, ...],
        embed_dim: int,
        action_embed: ActionEmbed,
        *,
        pred_dim: int,
        depth: int,
        num_heads: int,
        key: PRNGKey | None = None,
        conditioning: Conditioning = "both",
        residual: bool = True,
        pos: Literal["sincos", "learned", "none"] = "sincos",
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        drop_path: float = 0.0,
        layer_scale: float | None = None,
        remat: bool = False,
    ):
        key = resolve_key(key)
        k_in, k_out, k_tok, k_film, k_pos, k_blocks = jr.split(key, 6)
        n_tokens = 1
        for s in grid:
            n_tokens *= s
        self.action_embed = action_embed
        self.embed_in = eqx.nn.Linear(embed_dim, pred_dim, key=k_in)
        embed_out = eqx.nn.Linear(pred_dim, embed_dim, key=k_out)
        if residual:
            # Shrink the output projection so the residual branch starts as a
            # small perturbation of the identity. Deliberately *small* and not
            # zero: a zero projection would also zero the gradient flowing back
            # through it, stalling every upstream parameter -- the transformer,
            # the action embedding, the FiLM layer -- until the projection
            # itself moved off zero.
            embed_out = eqx.tree_at(
                lambda m: (m.weight, m.bias),
                embed_out,
                (RESIDUAL_INIT_SCALE * embed_out.weight, RESIDUAL_INIT_SCALE * embed_out.bias),
            )
        self.embed_out = embed_out
        if pos == "sincos":
            self.pos_embed = SinCosPosEmbed(tuple(grid), pred_dim)
        elif pos == "learned":
            self.pos_embed = LearnedPosEmbed(n_tokens, pred_dim, key=k_pos)
        else:
            self.pos_embed = None
        uses_token = conditioning in ("token", "both")
        uses_film = conditioning in ("film", "both")
        self.action_token = (
            eqx.nn.Linear(action_embed.embed_dim, pred_dim, key=k_tok) if uses_token else None
        )
        # Small-but-nonzero weights with a zero bias: FiLM starts *near* the
        # identity (scale ~ 1, shift ~ 0) without starting *at* it. A fully
        # zero-initialised FiLM would make the dynamics exactly independent of
        # the action at step zero, which is the wrong place to begin for a model
        # whose entire job is to be action-conditioned.
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
        self.grid = tuple(grid)
        self.embed_dim = embed_dim
        self.pred_dim = pred_dim
        self.conditioning = conditioning
        self.residual = residual

    def __call__(self, z: Array, action: Array, *, key: PRNGKey | None = None) -> Array:
        """Advance one step.

        Args:
            z: ``(N, embed_dim)`` current latent state.
            action: the action, in whatever form ``action_embed`` accepts.

        Returns:
            ``(N, embed_dim)`` next latent state.
        """
        k_blocks, k_act = (None, None) if key is None else jr.split(key)
        h = jax.vmap(self.embed_in)(z)
        if self.pos_embed is not None:
            h = h + self.pos_embed()
        a = self.action_embed(action, key=k_act)

        if self.film is not None:
            scale, shift = jnp.split(self.film(a), 2)
            h = h * (1.0 + scale) + shift
        n_prefix = 0
        if self.action_token is not None:
            h = jnp.concatenate([self.action_token(a)[None], h], axis=0)
            n_prefix = 1

        h = self.blocks(h, key=k_blocks)
        out = jax.vmap(self.embed_out)(h[n_prefix:])
        return z + out if self.residual else out

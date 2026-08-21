"""A residual MLP latent dynamics model.

The transformer in :mod:`xwm.dynamics.action_conditioned` is the expressive
choice, but a planner evaluates dynamics thousands of times per decision -- CEM
with 512 samples over a horizon of 5 is 2560 forward passes for a single action.
At that rate the dynamics model has to be cheap, which is why TD-MPC2 and MuZero
both use a small MLP over a *pooled* latent rather than a sequence model over
tokens.

Latents here are flat vectors ``(D,)``, not token grids. :class:`MLPDynamics`
still satisfies :class:`~xwm.core.types.LatentDynamics`, so the planners and
rollout helpers work unchanged.

References
----------
Hansen, Su & Wang, *TD-MPC2: Scalable, Robust World Models for Continuous
Control*, ICLR 2024. arXiv:2310.16828 -- the MLP latent dynamics with SimNorm.

Schrittwieser et al., *MuZero*, Nature 2020. arXiv:1911.08265 -- the recurrent
dynamics function that emits a reward alongside the next latent.
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
from ..nn.norm import LayerNorm, SimNorm

__all__ = ["MLPDynamics"]

Normalization = Literal["simnorm", "layernorm", "none"]


class MLPDynamics(Module):
    """``(z, a) -> z'`` as a residual MLP on a flat latent.

    Args:
        latent_dim: width of the latent vector.
        action_dim: width of the action vector (continuous), or the embedding
            width if you embed a discrete action before calling.
        hidden_dim: MLP width.
        depth: number of hidden layers.
        residual: predict the change rather than the next state. Consecutive
            latents are nearly identical, so starting near the identity is a far
            better prior than starting from noise.
        normalize: what to apply to the output latent.

            * ``"simnorm"`` -- TD-MPC2's simplicial normalisation. This is the
              load-bearing detail of TD-MPC2's stability: it bounds the latent
              without collapsing it, so the dynamics cannot drift off to
              infinity during a long unroll and cannot shrink to a point either.
            * ``"layernorm"`` -- the usual choice elsewhere.
            * ``"none"`` -- unbounded; expect drift over long rollouts.
        simnorm_groups: group size for SimNorm; ``latent_dim`` must divide by it.
    """

    action_in: eqx.nn.Linear
    layers: list[eqx.nn.Linear]
    norms: list[LayerNorm]
    out: eqx.nn.Linear
    post: SimNorm | LayerNorm | None
    latent_dim: int = eqx.field(static=True)
    action_dim: int = eqx.field(static=True)
    residual: bool = eqx.field(static=True)

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        *,
        key: PRNGKey | None = None,
        hidden_dim: int = 512,
        depth: int = 2,
        residual: bool = True,
        normalize: Normalization = "simnorm",
        simnorm_groups: int = 8,
    ):
        key = resolve_key(key)
        keys = jr.split(key, depth + 2)
        self.action_in = eqx.nn.Linear(action_dim, hidden_dim, key=keys[0])
        dims = [latent_dim + hidden_dim] + [hidden_dim] * depth
        self.layers = [
            eqx.nn.Linear(dims[i], hidden_dim, key=keys[i + 1]) for i in range(depth)
        ]
        self.norms = [LayerNorm(hidden_dim) for _ in range(depth)]
        out = eqx.nn.Linear(hidden_dim, latent_dim, key=keys[-1])
        if residual:
            # Small, not zero: a zero output projection also zeroes the gradient
            # flowing back through it, stalling every upstream parameter.
            out = eqx.tree_at(
                lambda m: (m.weight, m.bias), out, (0.05 * out.weight, 0.05 * out.bias)
            )
        self.out = out
        if normalize == "simnorm":
            self.post = SimNorm(simnorm_groups)
        elif normalize == "layernorm":
            self.post = LayerNorm(latent_dim)
        elif normalize == "none":
            self.post = None
        else:
            raise ValueError(f"unknown normalization {normalize!r}")
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.residual = residual

    def __call__(self, z: Array, action: Array, *, key: PRNGKey | None = None) -> Array:
        """``z``: ``(latent_dim,)``, ``action``: ``(action_dim,)``."""
        h = jnp.concatenate([z, jax.nn.mish(self.action_in(jnp.atleast_1d(action)))])
        for layer, norm in zip(self.layers, self.norms, strict=True):
            h = jax.nn.mish(norm(layer(h)))
        out = self.out(h)
        out = z + out if self.residual else out
        return self.post(out) if self.post is not None else out

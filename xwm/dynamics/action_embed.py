"""Action encoders.

Actions arrive in whatever form the robot or environment speaks -- continuous
joint velocities, end-effector deltas, a discrete button set -- and have to
become a vector the latent dynamics can condition on. That is all these do.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from ..core.module import Module
from ..core.random import resolve_key
from ..core.types import Array, PRNGKey
from ..nn.mlp import Mlp


class ContinuousActionEmbed(Module):
    """Embed a continuous action vector with a small MLP.

    Args:
        action_dim: dimensionality of the raw action.
        embed_dim: output width.
        scale: divide the action by this before embedding. Set it to the
            action's typical magnitude; an unnormalised action of magnitude 100
            will otherwise dominate the conditioning signal.
    """

    mlp: Mlp
    scale: float = eqx.field(static=True)
    action_dim: int = eqx.field(static=True)
    embed_dim: int = eqx.field(static=True)

    def __init__(
        self,
        action_dim: int,
        embed_dim: int,
        *,
        key: PRNGKey | None = None,
        hidden_dim: int | None = None,
        scale: float = 1.0,
    ):
        key = resolve_key(key)
        self.mlp = Mlp(action_dim, hidden_dim or 4 * embed_dim, embed_dim, key=key)
        self.scale = scale
        self.action_dim = action_dim
        self.embed_dim = embed_dim

    def __call__(self, action: Array, *, key: PRNGKey | None = None) -> Array:
        """``action``: ``(action_dim,)``. Returns ``(embed_dim,)``."""
        return self.mlp(action / self.scale, key=key)


class DiscreteActionEmbed(Module):
    """Embed a discrete action index with a lookup table."""

    table: Array
    n_actions: int = eqx.field(static=True)
    embed_dim: int = eqx.field(static=True)

    def __init__(self, n_actions: int, embed_dim: int, *, key: PRNGKey):
        self.table = 0.02 * jr.normal(key, (n_actions, embed_dim))
        self.n_actions = n_actions
        self.embed_dim = embed_dim

    def __call__(self, action: Array, *, key: PRNGKey | None = None) -> Array:
        """``action``: scalar integer index. Returns ``(embed_dim,)``."""
        return self.table[jnp.asarray(action, jnp.int32)]


class PoseActionEmbed(Module):
    """Embed a rigid-body pose delta, splitting translation from rotation.

    End-effector actions mix units -- metres and radians -- and a single linear
    layer has to learn to rescale them. Embedding the parts separately removes
    that burden, which matters when translations are centimetre-scale.

    Args:
        translation_dim: usually 3.
        rotation_dim: 3 for axis-angle / Euler, 4 for a quaternion, 6 for the
            continuous 6-D rotation parameterisation.
        extra_dim: remaining scalars, e.g. a gripper command.
    """

    translation: eqx.nn.Linear
    rotation: eqx.nn.Linear
    extra: eqx.nn.Linear | None
    out: eqx.nn.Linear
    translation_dim: int = eqx.field(static=True)
    rotation_dim: int = eqx.field(static=True)
    extra_dim: int = eqx.field(static=True)
    embed_dim: int = eqx.field(static=True)

    def __init__(
        self,
        embed_dim: int,
        *,
        key: PRNGKey | None = None,
        translation_dim: int = 3,
        rotation_dim: int = 3,
        extra_dim: int = 0,
    ):
        key = resolve_key(key)
        k1, k2, k3, k4 = jr.split(key, 4)
        half = embed_dim // 2
        self.translation = eqx.nn.Linear(translation_dim, half, key=k1)
        self.rotation = eqx.nn.Linear(rotation_dim, half, key=k2)
        self.extra = eqx.nn.Linear(extra_dim, half, key=k3) if extra_dim else None
        n_parts = 3 if extra_dim else 2
        self.out = eqx.nn.Linear(half * n_parts, embed_dim, key=k4)
        self.translation_dim = translation_dim
        self.rotation_dim = rotation_dim
        self.extra_dim = extra_dim
        self.embed_dim = embed_dim

    @property
    def action_dim(self) -> int:
        return self.translation_dim + self.rotation_dim + self.extra_dim

    def __call__(self, action: Array, *, key: PRNGKey | None = None) -> Array:
        t = self.translation_dim
        r = t + self.rotation_dim
        parts = [self.translation(action[:t]), self.rotation(action[t:r])]
        if self.extra is not None:
            parts.append(self.extra(action[r:]))
        return self.out(jax.nn.gelu(jnp.concatenate(parts)))

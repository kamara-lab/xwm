"""MuZero: a latent model trained to make *search* consistent.

The distinguishing idea is that MuZero does not try to model the observation at
all -- not its pixels and not its embedding. It learns whatever latent makes
three predictions come out right: reward, value, and policy. And the policy
target is not the behaviour policy; it is the *improved* policy that tree search
produced. The model is trained to agree with a better version of itself.

Three networks, in MuZero's own terms:

====================  ==================================================
representation        observation -> latent ``s``  (:attr:`encoder`)
dynamics              ``(s, a) -> (s', r)``        (:attr:`dynamics`)
prediction            ``s -> (policy, value)``     (:attr:`policy`, :attr:`value`)
====================  ==================================================

Training unrolls the dynamics ``horizon`` steps from a real observation and
matches all three heads at every step against targets stored in the replay
buffer -- MCTS visit counts for the policy, an n-step bootstrapped return for the
value, and observed rewards. Because the unroll is recurrent and gradients flow
through it, the gradient is scaled by ``1 / horizon`` to keep its magnitude
independent of the unroll length.

MuZero is a discrete-action algorithm; that is what MCTS needs. For continuous
control, discretise per-dimension or use TD-MPC2, whose MPPI planner is native to
continuous actions.

References
----------
Schrittwieser et al., *Mastering Atari, Go, Chess and Shogi by Planning with a
Learned Model* (MuZero), Nature 2020. arXiv:1911.08265 -- the algorithm
implemented here: representation/dynamics/prediction trained against
search-improved targets.

Hubert et al., *Learning and Planning in Complex Action Spaces* (Sampled
MuZero), ICML 2021. arXiv:2104.06303 -- the continuous-action variant, and the
principled alternative to the action discretisation this module assumes.

Hafner et al., *Mastering Diverse Domains through World Models* (DreamerV3),
2023. arXiv:2301.04104 -- symlog plus two-hot categorical prediction, used by
the heads here.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from ...core.ema import stop_gradient
from ...core.module import WorldModel
from ...core.random import resolve_key
from ...core.types import Array, Batch, Metrics, PRNGKey
from ...dynamics.mlp_dynamics import MLPDynamics
from ...encoders.state import StateEncoder
from ...encoders.vision import VisionEncoder
from ...heads.categorical import CategoricalScalar, cross_entropy
from ...heads.scalar import ScalarHead
from ...nn.norm import LayerNorm

__all__ = ["MuZero", "PolicyHead", "muzero"]

Encoder = VisionEncoder | StateEncoder


class PolicyHead(WorldModel):
    """``s -> logits`` over a discrete action set."""

    layers: list[eqx.nn.Linear]
    norms: list[LayerNorm]
    out: eqx.nn.Linear
    n_actions: int = eqx.field(static=True)

    def __init__(
        self,
        latent_dim: int,
        n_actions: int,
        *,
        key: PRNGKey | None = None,
        hidden_dim: int = 256,
        depth: int = 2,
    ):
        key = resolve_key(key)
        keys = jr.split(key, depth + 1)
        dims = [latent_dim] + [hidden_dim] * depth
        self.layers = [eqx.nn.Linear(dims[i], hidden_dim, key=keys[i]) for i in range(depth)]
        self.norms = [LayerNorm(hidden_dim) for _ in range(depth)]
        self.out = eqx.nn.Linear(hidden_dim, n_actions, key=keys[-1])
        self.n_actions = n_actions

    def __call__(self, z: Array) -> Array:
        h = z
        for layer, norm in zip(self.layers, self.norms, strict=True):
            h = jax.nn.mish(norm(layer(h)))
        return self.out(h)


class MuZero(WorldModel):
    """A MuZero agent over a discrete action space.

    Args:
        encoder: the representation network.
        dynamics: the recurrent latent dynamics.
        reward: reward head, conditioned on the one-hot action.
        value: value head.
        policy: policy logits head.
        n_actions: size of the action set.
        horizon: unroll length during training.
        reward_coef, value_coef, policy_coef: loss weights.
    """

    encoder: Encoder
    dynamics: MLPDynamics
    reward: ScalarHead
    value: ScalarHead
    policy: PolicyHead
    n_actions: int = eqx.field(static=True)
    horizon: int = eqx.field(static=True)
    reward_coef: float = eqx.field(static=True)
    value_coef: float = eqx.field(static=True)
    policy_coef: float = eqx.field(static=True)
    obs_key: str = eqx.field(static=True)
    action_key: str = eqx.field(static=True)
    reward_key: str = eqx.field(static=True)
    value_key: str = eqx.field(static=True)
    policy_key: str = eqx.field(static=True)

    def __init__(
        self,
        encoder: Encoder,
        dynamics: MLPDynamics,
        reward: ScalarHead,
        value: ScalarHead,
        policy: PolicyHead,
        *,
        n_actions: int,
        horizon: int = 5,
        reward_coef: float = 1.0,
        value_coef: float = 0.25,
        policy_coef: float = 1.0,
        obs_key: str = "observation",
        action_key: str = "action",
        reward_key: str = "reward",
        value_key: str = "value_target",
        policy_key: str = "policy_target",
    ):
        self.encoder = encoder
        self.dynamics = dynamics
        self.reward = reward
        self.value = value
        self.policy = policy
        self.n_actions = n_actions
        self.horizon = horizon
        self.reward_coef = reward_coef
        self.value_coef = value_coef
        self.policy_coef = policy_coef
        self.obs_key = obs_key
        self.action_key = action_key
        self.reward_key = reward_key
        self.value_key = value_key
        self.policy_key = policy_key
        self.uses_target = False

    # -- the three MuZero functions -----------------------------------------
    def represent(self, observation: Array, *, key: PRNGKey | None = None) -> Array:
        """Observation -> latent ``(D,)``."""
        return jnp.mean(self.encoder(observation, key=key), axis=0)

    def recurrent(self, z: Array, action: Array) -> tuple[Array, Array]:
        """``(z, action_index) -> (z', reward)``. The signature MCTS wants."""
        one_hot = jax.nn.one_hot(action, self.n_actions)
        return self.dynamics(z, one_hot), self.reward.value(z, one_hot)

    def predict(self, z: Array) -> tuple[Array, Array]:
        """``z -> (policy_logits, value)``. The other signature MCTS wants."""
        return self.policy(z), self.value.value(z)

    def search_fns(self):
        """Eval-mode ``(recurrent, predict)`` closures for :class:`~xwm.planning.MCTS`."""
        model = self.eval_mode()
        return model.recurrent, model.predict

    # -- training ------------------------------------------------------------
    def loss(
        self,
        batch: Batch,
        *,
        key: PRNGKey,
        target: MuZero | None = None,
    ) -> tuple[Array, Metrics]:
        """Args:
            batch: exactly what :meth:`xwm.training.ReplayBuffer.sample` returns --
                ``{"observation": (B, H + 1, ...), "action": (B, H),
                "reward": (B, H), "value_target": (B, H + 1),
                "policy_target": (B, H + 1, n_actions)}``.

                Only ``observation[:, 0]`` is read. Everything after it comes
                from the model's own unroll, and that is the whole point: MuZero
                never re-encodes an observation mid-unroll, so the latent is free
                to be whatever makes the predictions work. The later
                observations are accepted (and ignored) so the same batch feeds
                both this family and TD-MPC2.
        """
        observations = batch[self.obs_key]
        # Take the root observation. A time axis is present because the batch
        # comes from a trajectory buffer; MuZero simply does not use the rest.
        if observations.ndim > 1 and observations.shape[1] == batch[self.action_key].shape[1] + 1:
            observations = observations[:, 0]
        actions = batch[self.action_key].astype(jnp.int32)
        rewards = batch[self.reward_key]
        value_targets = batch[self.value_key]
        policy_targets = batch[self.policy_key]
        horizon = min(self.horizon, actions.shape[1])

        z = jax.vmap(self.represent)(observations)  # (B, D)

        # Step 0 has no reward to predict: it is the root.
        logits = jax.vmap(self.policy)(z)
        policy_loss = cross_entropy(logits, policy_targets[:, 0])
        value_loss = jnp.mean(jax.vmap(self.value.loss)(z, value_targets[:, 0]))
        reward_loss = jnp.zeros(())

        for step in range(horizon):
            action = actions[:, step]
            one_hot = jax.nn.one_hot(action, self.n_actions)
            reward_loss = reward_loss + jnp.mean(
                jax.vmap(self.reward.loss)(z, rewards[:, step], one_hot)
            )
            z = jax.vmap(self.dynamics)(z, one_hot)
            # Scale the recurrent path so the gradient reaching the encoder does
            # not grow with the unroll length.
            z = 0.5 * z + 0.5 * stop_gradient(z)
            policy_loss = policy_loss + cross_entropy(
                jax.vmap(self.policy)(z), policy_targets[:, step + 1]
            )
            value_loss = value_loss + jnp.mean(
                jax.vmap(self.value.loss)(z, value_targets[:, step + 1])
            )

        scale = 1.0 / (horizon + 1)
        total = scale * (
            self.reward_coef * reward_loss
            + self.value_coef * value_loss
            + self.policy_coef * policy_loss
        )
        return total, {
            "loss": total,
            "loss_reward": scale * reward_loss,
            "loss_value": scale * value_loss,
            "loss_policy": scale * policy_loss,
            "latent_std": jnp.mean(jnp.std(z, axis=0)),
        }


def muzero(
    *,
    n_actions: int,
    encoder: Encoder | None = None,
    observation: str = "state",
    state_dim: int | None = None,
    img_size: int = 64,
    patch_size: int = 8,
    latent_dim: int = 256,
    hidden_dim: int = 256,
    horizon: int = 5,
    n_bins: int = 101,
    key: PRNGKey | None = None,
    **model_kwargs,
) -> MuZero:
    """Assemble a MuZero agent over a discrete action space."""
    from ...encoders.image import ImageEncoder

    key = resolve_key(key)
    k_enc, k_dyn, k_rew, k_val, k_pi = jr.split(key, 5)
    if encoder is None:
        if observation == "state":
            if state_dim is None:
                raise ValueError('observation="state" needs state_dim')
            encoder = StateEncoder(state_dim, latent_dim, key=k_enc)
        elif observation == "image":
            encoder = ImageEncoder(
                key=k_enc, img_size=img_size, patch_size=patch_size,
                embed_dim=latent_dim, depth=4, num_heads=8,
            )
        else:
            raise ValueError(f"unknown observation kind {observation!r}")
    latent_dim = encoder.embed_dim
    scalar = CategoricalScalar(n_bins=n_bins)
    return MuZero(
        encoder,
        # LayerNorm, not SimNorm: simplicial normalisation is TD-MPC2's
        # contribution, and applied here it pins every latent near the uniform
        # point of each simplex, leaving almost no variance for the prediction
        # heads to read.
        MLPDynamics(
            latent_dim, n_actions, key=k_dyn, hidden_dim=hidden_dim,
            normalize="layernorm",
        ),
        ScalarHead(latent_dim, action_dim=n_actions, key=k_rew,
                   hidden_dim=hidden_dim, scalar=scalar),
        ScalarHead(latent_dim, key=k_val, hidden_dim=hidden_dim, scalar=scalar),
        PolicyHead(latent_dim, n_actions, key=k_pi, hidden_dim=hidden_dim),
        n_actions=n_actions,
        horizon=horizon,
        **model_kwargs,
    )

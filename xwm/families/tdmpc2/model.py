"""TD-MPC2: a latent world model trained by reward and TD value.

Where a JEPA learns a representation by predicting its own future embeddings,
TD-MPC2 learns one by predicting *reward* and *value*. Nothing anchors the latent
to the observation except the tasks it has to support, which is the point: the
representation keeps exactly what is needed to predict return and discards the
rest. There is no decoder and no reconstruction anywhere.

Five components over a shared latent:

======================  ===============================================
encoder                 observation -> latent ``z``
dynamics                ``(z, a) -> z'``
reward                  ``(z, a) -> r``
Q-ensemble              ``(z, a) -> Q``, pessimistic aggregate
policy prior            ``z -> a``, used to seed the planner
======================  ===============================================

The loss unrolls the dynamics ``horizon`` steps from a real latent and sums three
terms per step: **consistency** (predicted latent against the encoded next
observation, with the target detached), **reward**, and **value**. The policy is
trained separately to maximise Q with an entropy bonus, on detached latents, so
its gradient never reshapes the world model.

Two details do most of the stability work:

* Latents are :class:`~xwm.nn.norm.SimNorm`-normalised, so a long unroll can
  neither blow up nor collapse.
* Reward and value are predicted as distributions over bins rather than scalars,
  which makes the loss scale-free across tasks (see :mod:`xwm.heads.categorical`).

References
----------
Hansen, Su & Wang, *TD-MPC2: Scalable, Robust World Models for Continuous
Control*, ICLR 2024. arXiv:2310.16828 -- the model implemented here, including
SimNorm, the categorical reward/value heads and the policy-prior-seeded planner.

Hansen, Wang & Su, *Temporal Difference Learning for Model Predictive Control*
(TD-MPC), ICML 2022. arXiv:2203.04955 -- the predecessor that introduced the
"latent dynamics + terminal value" planning objective.

Haarnoja et al., *Soft Actor-Critic*, ICML 2018. arXiv:1801.01290 -- the
entropy-regularised policy objective.

Chen et al., *Randomized Ensembled Double Q-Learning* (REDQ), ICLR 2021.
arXiv:2101.05982 -- min over a random subset of an ensemble as the pessimistic
aggregate.
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
from ...heads.categorical import CategoricalScalar
from ...heads.policy import GaussianPolicy
from ...heads.q_ensemble import QEnsemble
from ...heads.scalar import ScalarHead

__all__ = ["TDMPC2", "planner", "tdmpc2"]

Encoder = VisionEncoder | StateEncoder


class TDMPC2(WorldModel):
    """A TD-MPC2 agent.

    Args:
        encoder: observation encoder. Token output is mean-pooled to a flat
            latent, since the dynamics and heads here are MLPs.
        dynamics: latent dynamics; SimNorm-normalised by default.
        reward: reward head, action-conditioned.
        critic: Q-ensemble.
        policy: policy prior.
        horizon: unroll length for the consistency/reward/value losses.
        discount: RL discount.
        consistency_coef, reward_coef, value_coef: loss weights.
        rho: per-step decay on the unroll losses. Later steps are less reliable
            because they start from a predicted latent, so they count less.
        entropy_coef: SAC-style entropy bonus for the policy.
    """

    encoder: Encoder
    dynamics: MLPDynamics
    reward: ScalarHead
    critic: QEnsemble
    policy: GaussianPolicy
    horizon: int = eqx.field(static=True)
    discount: float = eqx.field(static=True)
    consistency_coef: float = eqx.field(static=True)
    reward_coef: float = eqx.field(static=True)
    value_coef: float = eqx.field(static=True)
    rho: float = eqx.field(static=True)
    entropy_coef: float = eqx.field(static=True)
    obs_key: str = eqx.field(static=True)
    action_key: str = eqx.field(static=True)
    reward_key: str = eqx.field(static=True)

    def __init__(
        self,
        encoder: Encoder,
        dynamics: MLPDynamics,
        reward: ScalarHead,
        critic: QEnsemble,
        policy: GaussianPolicy,
        *,
        horizon: int = 3,
        discount: float = 0.99,
        consistency_coef: float = 20.0,
        reward_coef: float = 0.1,
        value_coef: float = 0.1,
        rho: float = 0.5,
        entropy_coef: float = 1e-4,
        obs_key: str = "observation",
        action_key: str = "action",
        reward_key: str = "reward",
    ):
        self.encoder = encoder
        self.dynamics = dynamics
        self.reward = reward
        self.critic = critic
        self.policy = policy
        self.horizon = horizon
        self.discount = discount
        self.consistency_coef = consistency_coef
        self.reward_coef = reward_coef
        self.value_coef = value_coef
        self.rho = rho
        self.entropy_coef = entropy_coef
        self.obs_key = obs_key
        self.action_key = action_key
        self.reward_key = reward_key
        # The critic target is an EMA of the whole model; only `critic` is read.
        self.uses_target = True

    # -- inference -----------------------------------------------------------
    def encode(self, observation: Array, *, key: PRNGKey | None = None) -> Array:
        """Observation -> flat latent ``(D,)``.

        Encoders emit ``(N, D)`` tokens; the dynamics and heads here are MLPs
        over a single vector, so tokens are mean-pooled.
        """
        tokens = self.encoder(observation, key=key)
        return jnp.mean(tokens, axis=0)

    def dynamics_fn(self):
        """A plain ``(z, a) -> z'`` closure in eval mode, ready for a planner."""
        model = self.dynamics.eval_mode()
        return lambda z, a: model(z, a)

    def value(self, z: Array, action: Array, *, key: PRNGKey | None = None) -> Array:
        return self.critic.pessimistic(z, action, key=key)

    def act(self, observation: Array, *, key: PRNGKey | None = None) -> Array:
        """Policy-prior action, no planning. See :mod:`xwm.planning` to plan."""
        return self.policy.eval_mode().act(self.encode(observation), key=key)

    # -- training ------------------------------------------------------------
    def td_target(
        self,
        target: TDMPC2,
        z_next: Array,
        reward: Array,
        key: PRNGKey,
    ) -> Array:
        """``r + gamma * Q_target(z', pi(z'))``, fully detached.

        The next action comes from the *current* policy but the Q from the EMA
        target critic: bootstrapping off the online critic is what makes value
        learning diverge.
        """
        k_policy, k_q = jr.split(key)
        next_action = self.policy.act(z_next, key=k_policy)
        q_next = target.critic.pessimistic(z_next, next_action, key=k_q)
        return stop_gradient(reward + self.discount * q_next)

    def loss(
        self,
        batch: Batch,
        *,
        key: PRNGKey,
        target: TDMPC2 | None = None,
    ) -> tuple[Array, Metrics]:
        """Args:
            batch: ``{"observation": (B, H + 1, ...), "action": (B, H, A),
                "reward": (B, H)}`` -- a contiguous slice of a trajectory.
        """
        if target is None:
            raise ValueError("TD-MPC2 needs an EMA target; the Trainer supplies it")
        observations = batch[self.obs_key]
        actions = batch[self.action_key]
        rewards = batch[self.reward_key]
        horizon = min(self.horizon, actions.shape[1])
        k_enc, k_td, k_q, k_pi = jr.split(key, 4)

        # Encode the whole window once. Targets are detached: the consistency
        # loss must pull the *dynamics* toward the encoder, never the reverse,
        # or the pair collapses to a constant latent.
        encode = jax.vmap(jax.vmap(lambda o: self.encode(o)))
        latents = encode(observations)  # (B, H + 1, D)
        targets_z = stop_gradient(latents[:, 1:])

        consistency = jnp.zeros(())
        reward_loss = jnp.zeros(())
        value_loss = jnp.zeros(())
        z = latents[:, 0]
        for step in range(horizon):
            action = actions[:, step]
            weight = self.rho**step

            z_pred = jax.vmap(self.dynamics)(z, action)
            consistency = consistency + weight * jnp.mean(
                jnp.square(z_pred - targets_z[:, step])
            )
            reward_loss = reward_loss + weight * jnp.mean(
                jax.vmap(self.reward.loss)(z, rewards[:, step], action)
            )

            td = jax.vmap(lambda zn, r, k: self.td_target(target, zn, r, k))(
                targets_z[:, step],
                rewards[:, step],
                jr.split(jr.fold_in(k_td, step), z.shape[0]),
            )
            value_loss = value_loss + weight * jnp.mean(
                jax.vmap(self.critic.loss)(z, action, td)
            )
            z = z_pred

        # Policy: maximise Q with an entropy bonus, on detached latents so the
        # actor cannot reshape the world model to make itself look good.
        flat_z = stop_gradient(latents.reshape(-1, latents.shape[-1]))
        keys = jr.split(k_pi, flat_z.shape[0])
        sampled = jax.vmap(self.policy.sample)(flat_z, keys)
        q_keys = jr.split(k_q, flat_z.shape[0])
        q_values = jax.vmap(
            lambda z_, a_, k_: self.critic.pessimistic(z_, a_, key=k_)
        )(flat_z, sampled.action, q_keys)
        policy_loss = jnp.mean(self.entropy_coef * sampled.log_prob - q_values)

        total = (
            self.consistency_coef * consistency
            + self.reward_coef * reward_loss
            + self.value_coef * value_loss
            + policy_loss
        )
        return total, {
            "loss": total,
            "loss_consistency": consistency,
            "loss_reward": reward_loss,
            "loss_value": value_loss,
            "loss_policy": policy_loss,
            "q_mean": jnp.mean(q_values),
            "entropy": -jnp.mean(sampled.log_prob),
            "latent_std": jnp.mean(jnp.std(flat_z, axis=0)),
        }


def planner(
    model: TDMPC2,
    *,
    horizon: int | None = None,
    n_samples: int = 512,
    n_iters: int = 6,
    temperature: float = 0.5,
    noise_std: float = 0.5,
):
    """An MPPI planner wired to ``model``'s own reward and value heads.

    Returns ``(planner, cost_fn)``; call ``planner.plan(key, model.dynamics_fn(),
    z, cost_fn)``. The planner scores candidates by discounted predicted reward
    plus a terminal value bootstrap, evaluated at the pre-transition latent
    because that is how the reward head was trained.
    """
    from ...planning.cost import return_cost
    from ...planning.sampling import MPPI

    horizon = horizon or model.horizon
    reward = model.reward.eval_mode()
    critic = model.critic.eval_mode()
    policy = model.policy.eval_mode()

    search = MPPI(
        horizon,
        model.reward.action_dim,
        n_samples=n_samples,
        n_iters=n_iters,
        temperature=temperature,
        noise_std=noise_std,
        cost_on="current",
    )
    cost = return_cost(
        lambda z, a: reward.value(z, a),
        lambda z: critic.pessimistic(z, policy.act(z)),
        horizon=horizon,
        discount=model.discount,
    )
    return search, cost


def tdmpc2(
    *,
    action_dim: int,
    encoder: Encoder | None = None,
    observation: str = "state",
    state_dim: int | None = None,
    img_size: int = 64,
    patch_size: int = 8,
    latent_dim: int = 512,
    hidden_dim: int = 512,
    horizon: int = 3,
    n_bins: int = 101,
    key: PRNGKey | None = None,
    **model_kwargs,
) -> TDMPC2:
    """Assemble a TD-MPC2 agent.

    Args:
        observation: ``"state"`` for a vector observation (needs ``state_dim``)
            or ``"image"`` for pixels.
        encoder: supply your own -- e.g. a frozen JEPA encoder -- and the rest
            is built around it.
        latent_dim: width of the pooled latent the dynamics and heads act on.
    """
    from ...encoders.image import ImageEncoder

    key = resolve_key(key)
    k_enc, k_dyn, k_rew, k_q, k_pi = jr.split(key, 5)
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
    return TDMPC2(
        encoder,
        MLPDynamics(latent_dim, action_dim, key=k_dyn, hidden_dim=hidden_dim),
        ScalarHead(latent_dim, action_dim=action_dim, key=k_rew,
                   hidden_dim=hidden_dim, scalar=scalar),
        QEnsemble(latent_dim, action_dim, key=k_q, hidden_dim=hidden_dim, scalar=scalar),
        GaussianPolicy(latent_dim, action_dim, key=k_pi, hidden_dim=hidden_dim),
        horizon=horizon,
        **model_kwargs,
    )

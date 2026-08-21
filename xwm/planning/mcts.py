"""Monte-Carlo tree search over a learned model (MuZero's planner).

CEM and MPPI sample whole action sequences and score them. MCTS instead grows a
tree, spending its budget where the model and the value function disagree most.
That matters when actions are discrete and consequences are sharp: sampling
sequences uniformly wastes almost every sample, while a tree commits to a prefix
and refines it.

The search never touches the environment. Every node is a *latent* produced by
the learned dynamics, and every leaf is scored by the learned value head, so the
whole thing is one jittable computation over arrays.

Implementation notes that matter for correctness:

* **PUCT** balances a node's value against its prior and visit count.
* **Min-max normalised Q.** Values from a learned head have no fixed scale, so
  raw Q would swamp the exploration term on some tasks and vanish on others.
  Normalising by the tree's own observed range makes the constant portable.
* **Dirichlet noise at the root** keeps the search from committing to the
  policy's favourite action before it has evidence.

References
----------
Schrittwieser et al., *Mastering Atari, Go, Chess and Shogi by Planning with a
Learned Model* (MuZero), Nature 2020. arXiv:1911.08265 -- the search this
implements, including min-max normalised Q and the reward term in the selection
score.

Silver et al., *Mastering the Game of Go without Human Knowledge* (AlphaGo
Zero), Nature 2017 -- PUCT selection and Dirichlet root noise.

Rosin, *Multi-armed Bandits with Episode Context*, AMAI 2011 -- the PUCT rule.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from ..core.module import Module
from ..core.types import Array, PRNGKey

__all__ = ["MCTS", "SearchResult"]


class SearchResult(NamedTuple):
    """What a search returns.

    Attributes:
        policy: ``(n_actions,)`` visit-count distribution -- the improved policy
            MuZero trains the network's policy head against.
        value: root value, the visit-weighted mean of the children's returns.
        action: the most-visited action.
        visits: raw visit counts, for diagnostics.
    """

    policy: Array
    value: Array
    action: Array
    visits: Array


class MCTS(Module):
    """PUCT tree search over a learned latent model.

    Args:
        n_actions: size of the discrete action space.
        n_simulations: search budget.
        discount: RL discount used when backing values up the tree.
        c_puct: exploration constant.
        dirichlet_alpha, root_noise_fraction: root exploration noise.
        max_depth: hard cap on tree depth, which also bounds the arrays.
    """

    n_actions: int = eqx.field(static=True)
    n_simulations: int = eqx.field(static=True)
    discount: float = eqx.field(static=True)
    c_puct: float = eqx.field(static=True)
    dirichlet_alpha: float = eqx.field(static=True)
    root_noise_fraction: float = eqx.field(static=True)
    max_depth: int = eqx.field(static=True)

    def __init__(
        self,
        n_actions: int,
        *,
        n_simulations: int = 50,
        discount: float = 0.997,
        c_puct: float = 1.25,
        dirichlet_alpha: float = 0.3,
        root_noise_fraction: float = 0.25,
        max_depth: int = 50,
    ):
        if n_actions < 2:
            raise ValueError(f"need at least two actions, got {n_actions}")
        self.n_actions = n_actions
        self.n_simulations = n_simulations
        self.discount = discount
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.root_noise_fraction = root_noise_fraction
        self.max_depth = max_depth

    def search(
        self,
        key: PRNGKey,
        root_latent: Array,
        recurrent: Callable[[Array, Array], tuple[Array, Array]],
        predict: Callable[[Array], tuple[Array, Array]],
        *,
        add_noise: bool = True,
    ) -> SearchResult:
        """Run the search from ``root_latent``.

        Args:
            recurrent: ``(latent, action_index) -> (next_latent, reward)``.
            predict: ``latent -> (policy_logits, value)``.
            add_noise: Dirichlet noise at the root. On for self-play data
                collection, off for evaluation.

        Returns:
            A :class:`SearchResult`.
        """
        n_nodes = self.n_simulations + 1
        n_actions = self.n_actions

        # Flat arrays instead of node objects, so the whole search is jittable.
        latents = jnp.zeros((n_nodes, root_latent.shape[-1])).at[0].set(root_latent)
        priors = jnp.zeros((n_nodes, n_actions))
        values = jnp.zeros((n_nodes,))
        visits = jnp.zeros((n_nodes,), jnp.int32)
        value_sums = jnp.zeros((n_nodes,))
        children = jnp.full((n_nodes, n_actions), -1, jnp.int32)
        rewards = jnp.zeros((n_nodes,))
        parents = jnp.full((n_nodes,), -1, jnp.int32)
        parent_action = jnp.full((n_nodes,), -1, jnp.int32)

        root_logits, root_value = predict(root_latent)
        root_prior = jax.nn.softmax(root_logits)
        if add_noise:
            noise = jr.dirichlet(key, jnp.full((n_actions,), self.dirichlet_alpha))
            frac = self.root_noise_fraction
            root_prior = (1.0 - frac) * root_prior + frac * noise
        priors = priors.at[0].set(root_prior)
        values = values.at[0].set(root_value)
        visits = visits.at[0].set(1)
        value_sums = value_sums.at[0].set(root_value)

        state = (latents, priors, values, visits, value_sums, children, rewards,
                 parents, parent_action, jnp.asarray(1, jnp.int32))

        def simulate(carry, _):
            (latents, priors, values, visits, value_sums, children, rewards,
             parents, parent_action, n_used) = carry

            # -- select: walk down by PUCT until an unexpanded action.
            def cond(loop):
                _, _, depth, done = loop
                return jnp.logical_and(jnp.logical_not(done), depth < self.max_depth)

            def descend(loop):
                node, action, depth, _ = loop
                # The score of an action is the reward it earns *plus* the
                # discounted value of where it lands. Scoring by the child's
                # value alone makes the search blind to immediate reward, so it
                # cannot find a payoff that is one step away.
                q_mean = jnp.where(visits > 0, value_sums / jnp.maximum(visits, 1), 0.0)
                action_value = rewards + self.discount * q_mean
                # Normalise into [0, 1] using the tree's own range: a learned
                # value head has no fixed scale, so a raw Q would make c_puct
                # task-dependent.
                seen = visits > 0
                low = jnp.min(jnp.where(seen, action_value, jnp.inf))
                high = jnp.max(jnp.where(seen, action_value, -jnp.inf))
                span = jnp.maximum(high - low, 1e-8)
                child_idx = children[node]
                child_visits = jnp.where(child_idx >= 0, visits[child_idx], 0)
                child_q = jnp.where(
                    child_idx >= 0, (action_value[child_idx] - low) / span, 0.0
                )
                exploration = (
                    self.c_puct
                    * priors[node]
                    * jnp.sqrt(jnp.maximum(visits[node], 1))
                    / (1 + child_visits)
                )
                chosen = jnp.argmax(child_q + exploration)
                next_node = children[node, chosen]
                return (
                    jnp.where(next_node >= 0, next_node, node),
                    chosen,
                    depth + 1,
                    next_node < 0,
                )

            node, action, depth, _ = jax.lax.while_loop(
                cond, descend, (jnp.asarray(0, jnp.int32), jnp.asarray(0, jnp.int32),
                                jnp.asarray(0, jnp.int32), jnp.asarray(False))
            )

            # -- expand the chosen action into a new node.
            new_latent, reward = recurrent(latents[node], action)
            logits, leaf_value = predict(new_latent)
            index = n_used
            latents = latents.at[index].set(new_latent)
            priors = priors.at[index].set(jax.nn.softmax(logits))
            values = values.at[index].set(leaf_value)
            rewards = rewards.at[index].set(reward)
            parents = parents.at[index].set(node)
            parent_action = parent_action.at[index].set(action)
            children = children.at[node, action].set(index)

            # -- back up the discounted return along the path to the root.
            def step_back(carry_bu, _):
                current, value, visits_, sums_ = carry_bu
                alive = current >= 0
                safe = jnp.maximum(current, 0)
                visits_ = visits_.at[safe].add(jnp.where(alive, 1, 0))
                sums_ = sums_.at[safe].add(jnp.where(alive, value, 0.0))
                next_value = jnp.where(
                    alive, rewards[safe] + self.discount * value, value
                )
                return (parents[safe], next_value, visits_, sums_), None

            (_, _, visits, value_sums), _ = jax.lax.scan(
                step_back,
                (index, leaf_value, visits, value_sums),
                None,
                length=self.max_depth,
            )

            return (latents, priors, values, visits, value_sums, children, rewards,
                    parents, parent_action, n_used + 1), None

        state, _ = jax.lax.scan(simulate, state, None, length=self.n_simulations)
        _, _, _, visits, value_sums, children, rewards, _, _, _ = state

        root_children = children[0]
        child_visits = jnp.where(root_children >= 0, visits[root_children], 0).astype(
            jnp.float32
        )
        total = jnp.maximum(jnp.sum(child_visits), 1.0)
        policy = child_visits / total
        q_mean = jnp.where(visits > 0, value_sums / jnp.maximum(visits, 1), 0.0)
        action_value = rewards + self.discount * q_mean
        child_q = jnp.where(root_children >= 0, action_value[root_children], 0.0)
        return SearchResult(
            policy=policy,
            value=jnp.sum(policy * child_q),
            action=jnp.argmax(child_visits),
            visits=child_visits,
        )

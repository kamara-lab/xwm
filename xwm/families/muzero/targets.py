"""Building MuZero's training targets from self-play.

MuZero does not train on what the behaviour policy did; it trains on what the
*search* concluded. Two targets come out of a self-play episode:

* **policy** -- the MCTS visit distribution at each step. Search is a policy
  improvement operator, so this is a better policy than the network's own, and
  regressing onto it is what makes the loop improve.
* **value** -- an ``n``-step bootstrapped return, using the search's root value
  for the bootstrap rather than the network's raw value head. The search value is
  already an improvement over the head, so bootstrapping from it propagates that
  improvement backwards.

Getting the bootstrap index right is the fiddly part, which is why it lives here
rather than being rewritten in every training script.

References
----------
Schrittwieser et al., *Mastering Atari, Go, Chess and Shogi by Planning with a
Learned Model* (MuZero), Nature 2020. arXiv:1911.08265 -- Appendix G describes
the n-step value target and the use of MCTS visit counts as the policy target.
"""

from __future__ import annotations

import numpy as np

__all__ = ["n_step_value_targets"]


def n_step_value_targets(
    rewards: np.ndarray,
    root_values: np.ndarray,
    *,
    discount: float = 0.997,
    n_steps: int = 5,
) -> np.ndarray:
    """``(T + 1,)`` value targets for an episode of ``T`` transitions.

    Args:
        rewards: ``(T,)`` rewards received.
        root_values: ``(T + 1,)`` MCTS root values, one per visited state.
        discount: RL discount.
        n_steps: bootstrap horizon. Longer reduces bias and adds variance;
            beyond the episode end the sum simply truncates.

    Returns:
        ``target[t] = sum_{k<n} gamma^k r_{t+k} + gamma^n V_root[t+n]``, with the
        bootstrap dropped when ``t + n`` runs past the episode.
    """
    rewards = np.asarray(rewards, np.float32)
    root_values = np.asarray(root_values, np.float32)
    horizon = rewards.shape[0]
    if root_values.shape[0] != horizon + 1:
        raise ValueError(
            f"expected {horizon + 1} root values for {horizon} rewards, "
            f"got {root_values.shape[0]}"
        )

    targets = np.zeros((horizon + 1,), np.float32)
    for t in range(horizon + 1):
        bootstrap_index = t + n_steps
        if bootstrap_index <= horizon:
            value = (discount**n_steps) * root_values[bootstrap_index]
        else:
            value = 0.0  # past the end of the episode: nothing left to bootstrap
        for k in range(min(n_steps, horizon - t)):
            value += (discount**k) * rewards[t + k]
        targets[t] = value
    return targets

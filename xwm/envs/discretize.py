"""Turning a continuous action space into a discrete one.

MuZero's tree search enumerates actions, so it needs a finite set; a robot arm
has a continuous one. The cheap bridge is a small table of *axis-aligned* moves:
push one joint one way, or hold still.

The combinatorial alternative -- every combination of per-joint directions -- is
``3 ** n_joints``, which is 2187 actions for a 7-DoF arm and hopeless for a
search that must visit each at least once. Axis-aligned moves grow linearly, and
a sequence of them still reaches any direction; it just takes more steps.

For genuinely continuous control prefer TD-MPC2, whose MPPI planner samples the
action space directly and needs no discretisation at all.

References
----------
Hubert et al., *Learning and Planning in Complex Action Spaces* (Sampled
MuZero), ICML 2021. arXiv:2104.06303 -- the principled alternative to
discretising a continuous action space.

Tavakoli, Pardo & Kormushev, *Action Branching Architectures for Deep
Reinforcement Learning*, AAAI 2018. arXiv:1711.08946 -- on why per-dimension
factorisation beats the combinatorial product of per-joint choices.
"""

from __future__ import annotations

import numpy as np

__all__ = ["discrete_action_table"]


def discrete_action_table(
    action_dim: int,
    *,
    magnitude: float = 1.0,
    include_noop: bool = True,
) -> np.ndarray:
    """``(n_actions, action_dim)`` table of axis-aligned moves.

    Args:
        action_dim: number of continuous action dimensions.
        magnitude: how far each move pushes its dimension.
        include_noop: add an all-zeros action. Worth keeping: without it the
            agent cannot choose to stay put, which on a reach task means it can
            never stop once it arrives.

    Returns:
        ``2 * action_dim (+ 1)`` rows, each a continuous action vector.
    """
    if action_dim < 1:
        raise ValueError(f"action_dim must be positive, got {action_dim}")
    moves = []
    for axis in range(action_dim):
        for sign in (+1.0, -1.0):
            action = np.zeros((action_dim,), np.float32)
            action[axis] = sign * magnitude
            moves.append(action)
    if include_noop:
        moves.append(np.zeros((action_dim,), np.float32))
    return np.stack(moves)

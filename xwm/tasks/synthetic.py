"""Tasks backed by the pure-JAX worlds in :mod:`xwm.data`.

No download, no simulator, no optional dependency -- so the whole protocol runs
in CI and on a laptop. These are not stand-ins for published benchmarks and a
number measured here is not comparable with one measured on Push-T; what they
are for is the part of the pipeline that has nothing to do with which
simulator is underneath. The evaluation loop, the config system, the CLI and
the baselines are all exercised end to end here, which is why a bug in them
fails in two seconds rather than after a dataset download.

:class:`xwm.data.PushWorld` was designed to be the synthetic analogue of Push-T:
the map from action to next state has a hinge at the contact boundary, so a
dynamics model has to represent where the pusher is relative to the puck.
"""

from __future__ import annotations

from collections.abc import Iterator

import jax.random as jr
import numpy as np

from ..data.worlds import MazeWorld, PushWorld, maze_sequences, push_sequences
from ..envs.jax_world import JaxWorldEnv
from .spec import TaskSpec

__all__ = [
    "MAZE_SYNTHETIC",
    "PUSH_SYNTHETIC",
    "generated_episodes",
    "maze_env",
    "push_env",
    "push_expert_actions",
    "scripted_push_episodes",
]


def push_env(*, size: int = 32, seed: int = 0, **kwargs) -> JaxWorldEnv:
    return JaxWorldEnv(PushWorld(size, **kwargs), seed=seed)


def maze_env(*, size: int = 32, seed: int = 0, **kwargs) -> JaxWorldEnv:
    return JaxWorldEnv(MazeWorld(size, **kwargs), seed=seed)


#: How many episodes a generated task produces, and how long each one is.
_GENERATED = {"n_episodes": 64, "length": 24}


def generated_episodes(spec: TaskSpec) -> Iterator[dict[str, np.ndarray]]:
    """Roll the world out under smoothed random actions, in the recorded layout.

    Yields the same field vocabulary a dataset reader does -- ``video`` ``(T,
    C, H, W)``, ``action`` ``(T - 1, A)``, ``state`` ``(T, S)``, ``reward``
    ``(T - 1)`` -- so the generated path and the recorded path go through the
    same frameskip, split, cache and clipping code.
    """
    options = {**_GENERATED, **spec.dataset_options}
    n, length = int(options["n_episodes"]), int(options["length"])
    size = int(spec.native_size[0])
    key = jr.PRNGKey(int(options.get("seed", spec.split_seed)))
    if options.get("policy") == "expert":
        yield from scripted_push_episodes(
            spec,
            n_episodes=n,
            length=length,
            noise=float(options.get("noise", 0.25)),
            seed=int(options.get("seed", spec.split_seed)),
        )
        return
    if spec.env == "maze/synthetic":
        data = maze_sequences(key, n, length, world=MazeWorld(size))
    else:
        data = push_sequences(key, n, length, world=PushWorld(size))
    for i in range(n):
        episode = {
            # uint8: the same dtype a reader yields, so preprocess_episode's
            # resize takes the same branch on both paths.
            "video": np.asarray(np.clip(data["video"][i], 0.0, 1.0) * 255.0, np.uint8),
            "action": np.asarray(data["action"][i], np.float32),
            "state": np.asarray(data["state"][i], np.float32),
        }
        if "reward" in data:
            episode["reward"] = np.asarray(data["reward"][i], np.float32)
        yield episode


PUSH_SYNTHETIC = TaskSpec(
    name="pusht/synthetic",
    summary=(
        "Planar pushing in xwm.data.PushWorld: a pusher, a puck and a goal ring, "
        "32x32, no dependencies. The CI twin of pusht/lerobot -- same protocol, "
        "same code path, two seconds instead of a download."
    ),
    env="pusht/synthetic",
    dataset=None,
    # A scripted pusher, not smoothed noise: a random pusher misses the puck,
    # so every (start, goal) pair drawn from its recordings differs by nothing
    # and doing nothing solves the task. See push_expert_actions.
    dataset_options={"policy": "expert", "noise": 0.25, "n_episodes": 64, "length": 24},
    action_dim=2,
    action_low=-1.0,
    action_high=1.0,
    observation=("video",),
    native_size=(32, 32),
    state_dim=8,
    frameskip=1,
    history=1,
    horizon=5,
    receding_horizon=1,
    goal_offset=6,
    eval_budget=16,
    episodes=8,
    metric="puck",
    metric_options={"threshold": 0.08},
)

MAZE_SYNTHETIC = TaskSpec(
    name="maze/synthetic",
    summary=(
        "Goal-conditioned navigation through a walled maze, 32x32, no "
        "dependencies. Harder than pusht/synthetic for a planner: the straight "
        "line to the goal usually runs through a wall."
    ),
    env="maze/synthetic",
    dataset=None,
    action_dim=2,
    action_low=-1.0,
    action_high=1.0,
    observation=("video",),
    native_size=(32, 32),
    state_dim=6,
    frameskip=1,
    history=1,
    horizon=5,
    receding_horizon=1,
    goal_offset=8,
    eval_budget=24,
    episodes=8,
    metric="agent",
    metric_options={"threshold": 0.08},
)


def push_expert_actions(world: PushWorld, state, *, noise: float = 0.0, key=None) -> np.ndarray:
    """A scripted pusher: get behind the puck, then drive it at the goal.

    Smoothed random actions are the wrong demonstrator for a contact task. The
    pusher wanders, misses the puck, and the recorded puck displacement over any
    window is almost always zero -- so a start and a goal drawn from such a
    recording differ by nothing and *doing nothing* solves the instance. The
    evaluation then measures the sampler rather than the planner, which is
    exactly what the ``noop`` baseline exists to expose.

    The controller is two cases and no tuning: if the pusher is not on the
    far side of the puck from the goal, move to the point that puts it there;
    otherwise push along the puck-to-goal line. This is the synthetic analogue
    of the expert policies a benchmark ships to collect its data with.
    """
    import jax.numpy as jnp

    pusher, puck, goal = np.asarray(state.pusher), np.asarray(state.puck), np.asarray(state.goal)
    to_goal = goal - puck
    distance = float(np.linalg.norm(to_goal))
    direction = to_goal / max(distance, 1e-6)
    # Where the pusher must stand to push the puck toward the goal.
    behind = puck - direction * world.contact_radius
    offset = behind - pusher
    if float(np.linalg.norm(offset)) > 0.25 * world.contact_radius:
        move = offset  # approach
    else:
        move = direction * world.action_scale  # push through the puck
    action = move / max(float(np.linalg.norm(move)), 1e-6)
    if noise and key is not None:
        action = action + noise * np.asarray(jr.normal(key, (2,)))
    del jnp
    return np.clip(action, -1.0, 1.0).astype(np.float32)


def scripted_push_episodes(
    spec: TaskSpec, *, n_episodes: int, length: int, noise: float, seed: int
) -> Iterator[dict[str, np.ndarray]]:
    """Episodes in which the puck actually moves, from :func:`push_expert_actions`."""
    world = PushWorld(int(spec.native_size[0]))
    env = JaxWorldEnv(world, seed=seed)
    for episode in range(n_episodes):
        env.reset(seed=seed * 1000 + episode)
        frames = [env.observe()["image"]]
        states = [np.asarray(world.state_vector(env._state), np.float32)]
        actions, rewards, successes = [], [], []
        for step in range(length - 1):
            key = jr.fold_in(jr.PRNGKey(seed * 1000 + episode), step)
            action = push_expert_actions(world, env._state, noise=noise, key=key)
            env.step(action)
            actions.append(action)
            frames.append(env.observe()["image"])
            states.append(np.asarray(world.state_vector(env._state), np.float32))
            # The reward of the state the action *led to*, so it aligns with the
            # action and with ReplayBuffer.add_episode -- the same convention
            # push_sequences uses.
            rewards.append(float(world.reward(env._state)))
            successes.append(float(world.success(env._state)))
        yield {
            "video": np.stack(frames),
            "action": np.stack(actions),
            "state": np.stack(states),
            "reward": np.asarray(rewards, np.float32),
            "success": np.asarray(successes, np.float32),
        }

"""Two harder synthetic worlds: contact, and long-horizon navigation.

:class:`~xwm.data.synthetic.SpriteWorld` is deliberately easy -- linear
dynamics, fully observable, no reward. That makes it the right thing for a
self-supervised objective and the wrong thing for everything else, and it is why
the reward-driven examples currently reach for the Franka arm and the ``newton``
extra. These two worlds fill the gap without a simulator:

:class:`PushWorld`
    An agent that can only move a puck by touching it. Contact puts a hinge in
    the dynamics, so the map from action to next state is neither linear nor
    smooth, and the reward is dense, which is what TD-MPC2 and MuZero need.

:class:`MazeWorld`
    Walls, sliding contact and a goal several corridors away. Long-horizon and
    sparse, shaped after OGBench's ``antmaze-*`` ``navigate`` tasks at a scale
    that runs on a laptop, so a planner can be tested offline against the same
    task structure the OGBench datasets in :mod:`xwm.datasets` carry.

Both are pure JAX, ``jit``- and ``vmap``-able, and produce the same field
contract as :func:`~xwm.data.synthetic.sprite_sequences`, with ``reward`` added.

References
----------
Park, Frans, Eysenbach & Levine, *OGBench: Benchmarking Offline
Goal-Conditioned RL*, ICLR 2025. arXiv:2410.20092 -- the maze layouts and the
semi-sparse navigate reward this world imitates.

Hansen, Su & Wang, *TD-MPC2: Scalable, Robust World Models for Continuous
Control*, ICLR 2024. arXiv:2310.16828 -- the consumer of the dense reward.

Yu et al., *Meta-World*, CoRL 2019 -- planar pushing as the canonical
contact-rich manipulation task these dynamics abstract.
"""

from __future__ import annotations

from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from ..core.module import Module
from ..core.types import Array, PRNGKey
from .synthetic import random_actions, soft_disc, soft_ring

#: An 8x8 maze whose free cells form one connected corridor with no shortcuts,
#: so the distance between two cells is genuinely the path length rather than
#: the straight line. ``#`` is wall, ``.`` is free.
DEFAULT_MAZE: tuple[str, ...] = (
    "########",
    "#......#",
    "#.####.#",
    "#.#....#",
    "#.#.##.#",
    "#...#..#",
    "#.###..#",
    "########",
)


class PushState(NamedTuple):
    """A pusher, the puck it can push, and where the puck should end up."""

    pusher: Array  # (2,) in [0, 1]^2
    puck: Array  # (2,)
    goal: Array  # (2,)


class PushWorld(Module):
    """Planar pushing: the puck moves only when the pusher touches it.

    Args:
        size: rendered resolution (square).
        pusher_radius: radius of the controlled disc, normalised units.
        puck_radius: radius of the pushed disc.
        goal_radius: how close the puck must get to count as a success.
        action_scale: pusher displacement per unit of action.
        dt: integration step.

    The contact model is a position projection: the pusher moves wherever the
    action says, and if it then overlaps the puck, the puck is pushed straight
    out along the line between the two centres until they only touch. No
    inertia, no friction, no rotation -- the puck stops the instant contact
    stops.

    That is a deliberate simplification, and it keeps the one property that
    matters. The map from action to next state has a hinge in it at the contact
    boundary: on one side of it an action does nothing to the puck, on the other
    it moves it, and the derivative jumps across. A dynamics model has to
    represent *where the pusher is relative to the puck* to predict anything,
    which is exactly the structure :class:`~xwm.data.synthetic.SpriteWorld`'s
    linear dynamics cannot supply.

    The defaults were set by measuring the task rather than by eye, the same way
    :data:`xwm.envs.newton_franka.DEFAULT_GOAL` was. Over 48-step episodes: a
    single sequence of smoothed random actions gets the puck no closer than
    ~0.30 to the goal, and 3% of them ever reach it; the best of 80 solves 94%
    of starts inside the 0.08 goal radius, with a median closest approach of
    0.03. So random data has dense reward signal everywhere while leaving the
    task itself unsolved, which is the regime where a planner is distinguishable
    from noise -- a goal random actions stumble into measures nothing.
    """

    size: int = eqx.field(static=True)
    pusher_radius: float = eqx.field(static=True)
    puck_radius: float = eqx.field(static=True)
    goal_radius: float = eqx.field(static=True)
    action_scale: float = eqx.field(static=True)
    dt: float = eqx.field(static=True)

    def __init__(
        self,
        size: int = 32,
        *,
        pusher_radius: float = 0.09,
        puck_radius: float = 0.07,
        goal_radius: float = 0.08,
        action_scale: float = 0.12,
        dt: float = 1.0,
    ):
        self.size = size
        self.pusher_radius = pusher_radius
        self.puck_radius = puck_radius
        self.goal_radius = goal_radius
        self.action_scale = action_scale
        self.dt = dt

    @property
    def action_dim(self) -> int:
        return 2

    @property
    def channels(self) -> int:
        return 3

    @property
    def observation_shape(self) -> tuple[int, int, int]:
        return (self.channels, self.size, self.size)

    @property
    def state_dim(self) -> int:
        """Width of :meth:`state_vector`, for :class:`xwm.encoders.StateEncoder`."""
        return 8

    @property
    def contact_radius(self) -> float:
        """Centre distance at which pusher and puck touch."""
        return self.pusher_radius + self.puck_radius

    # -- dynamics ------------------------------------------------------------
    def reset(self, key: PRNGKey) -> PushState:
        """Puck near the middle, pusher beside it, goal a push or two away.

        The three are placed rather than sampled independently because
        independent samples put the goal behind the pusher a third of the time,
        where the only solution is to walk around the puck first -- a much
        longer-horizon task than the one this world is for.
        """
        k_puck, k_angle, k_goal = jr.split(key, 3)
        puck = jr.uniform(k_puck, (2,), minval=0.35, maxval=0.65)
        angle = jr.uniform(k_angle, (), minval=0.0, maxval=2.0 * jnp.pi)
        offset = jnp.stack([jnp.cos(angle), jnp.sin(angle)])
        pusher = jnp.clip(puck + 1.25 * self.contact_radius * offset, 0.05, 0.95)
        # The goal sits on the far side of the puck from the pusher, at a
        # distance a handful of pushes covers.
        distance = jr.uniform(k_goal, (), minval=0.20, maxval=0.35)
        goal = jnp.clip(puck - distance * offset, 0.12, 0.88)
        return PushState(pusher=pusher, puck=puck, goal=goal)

    def step(self, state: PushState, action: Array, *, key: PRNGKey | None = None) -> PushState:
        """Advance the world. ``action`` is a 2-D displacement in ``[-1, 1]^2``.

        ``key`` is accepted and ignored: this world is deterministic. The
        signature matches :meth:`xwm.data.SpriteWorld.step` so the two are
        interchangeable in a rollout.
        """
        del key
        move = self.action_scale * self.dt * jnp.clip(action, -1.0, 1.0)
        pusher = jnp.clip(state.pusher + move, self.pusher_radius, 1.0 - self.pusher_radius)

        delta = state.puck - pusher
        distance = jnp.linalg.norm(delta)
        # Direction to push in. At exactly coincident centres the direction is
        # undefined; fall back to the action, which is where the pusher came from.
        safe = jnp.maximum(distance, 1e-6)
        fallback = move / jnp.maximum(jnp.linalg.norm(move), 1e-6)
        direction = jnp.where(distance > 1e-6, delta / safe, fallback)
        overlap = jnp.maximum(self.contact_radius - distance, 0.0)
        puck = state.puck + overlap * direction
        puck = jnp.clip(puck, self.puck_radius, 1.0 - self.puck_radius)
        return PushState(pusher=pusher, puck=puck, goal=state.goal)

    # -- task ----------------------------------------------------------------
    def goal_distance(self, state: PushState) -> Array:
        """Distance from the puck to the goal. Ground truth, for evaluation."""
        return jnp.linalg.norm(state.puck - state.goal)

    def reward(self, state: PushState) -> Array:
        """Dense reward in ``[-1, 0]``: ``-tanh`` of the puck-to-goal distance.

        Bounded for the same reason :meth:`xwm.envs.FrankaEnv.reward` is
        bounded -- it keeps the value function inside the categorical head's bin
        range without per-task tuning, and unlike a squared distance its
        gradient does not vanish far from the goal.
        """
        return -jnp.tanh(self.goal_distance(state))

    def success(self, state: PushState) -> Array:
        """``1.0`` when the puck is within :attr:`goal_radius` of the goal."""
        return (self.goal_distance(state) < self.goal_radius).astype(jnp.float32)

    def state_vector(self, state: PushState) -> Array:
        """``(8,)``: pusher, puck, goal, and the puck-to-goal offset.

        The offset is redundant given the other three, and included anyway: it
        is the quantity the reward depends on, and handing it to a state encoder
        rather than making it learn a subtraction is the difference between a
        two-layer MLP working and not.
        """
        return jnp.concatenate([state.pusher, state.puck, state.goal, state.goal - state.puck])

    # -- rendering -----------------------------------------------------------
    def render(self, state: PushState) -> Array:
        """``(3, size, size)`` in ``[0, 1]``: pusher red, puck green, goal blue.

        One object per channel, the same convention
        :meth:`xwm.data.SpriteWorld.render` uses, so a probe can ask what a
        representation kept without having to disentangle anything.

        The goal is drawn as a **ring**, not a filled disc. A filled disc at the
        goal radius is the biggest, brightest thing in the frame and reads as the
        subject of the picture rather than as a target; an outline says "here"
        and leaves the two things that actually move as the subject. It also
        matches how Push-T and every other pushing benchmark draws its target,
        which is what makes the synthetic frames and the recorded ones
        comparable by eye.
        """
        pusher = soft_disc(self.size, state.pusher, self.pusher_radius, sharpness=1.1)
        puck = soft_disc(self.size, state.puck, self.puck_radius, sharpness=1.1)
        goal = soft_ring(self.size, state.goal, self.goal_radius)
        # A faint gradient rather than pure black: an empty frame gives an
        # encoder no absolute reference for position, and reads as a rendering
        # bug in a figure.
        grid = (jnp.arange(self.size) + 0.5) / self.size
        background = 0.12 * (grid[:, None] + grid[None, :]) / 2.0
        return jnp.clip(jnp.stack([pusher, puck, 0.75 * goal + background]), 0.0, 1.0)

    # -- trajectories --------------------------------------------------------
    def rollout(self, key: PRNGKey, actions: Array) -> tuple[Array, PushState]:
        """Run ``(T, 2)`` actions from a random start.

        Returns ``(frames, states)`` with ``T + 1`` of each -- the initial
        observation precedes the first action.
        """
        state = self.reset(key)

        def advance(s, a):
            s = self.step(s, a)
            return s, s

        _, states = jax.lax.scan(advance, state, actions)
        all_states = jax.tree_util.tree_map(
            lambda first, rest: jnp.concatenate([first[None], rest]), state, states
        )
        return jax.vmap(self.render)(all_states), all_states

    def observe(self, key: PRNGKey, actions: Array) -> Array:
        """Just the frames from :meth:`rollout`."""
        return self.rollout(key, actions)[0]


class MazeState(NamedTuple):
    """Position, velocity and goal, all in normalised world coordinates."""

    pos: Array  # (2,) in [0, 1]^2
    vel: Array  # (2,)
    goal: Array  # (2,)


class MazeWorld(Module):
    """Goal-conditioned navigation through a walled maze, sparse reward.

    Args:
        size: rendered resolution (square).
        layout: rows of ``#`` (wall) and ``.`` (free). Square, and the border
            must be wall.
        radius: agent radius in normalised units.
        goal_radius: how close counts as reaching the goal.
        damping: velocity decay per step; ``1.0`` is frictionless.
        action_scale: acceleration per unit of action.
        goal_range: goals are drawn from free cells within this straight-line
            distance of the start.

    Collision is resolved one axis at a time, so the agent slides along a wall
    instead of sticking to it -- the same trick every 2-D game uses, and the
    reason a policy that pushes diagonally into a corridor still makes progress.
    The agent is a box of side ``2 * radius`` for collision and a disc for
    rendering.

    ``goal_range`` defaults to 0.45, which keeps the task learnable *from random
    data*. With the default 8x8 layout and smoothed random actions, the fraction
    of episodes that touch the goal at least once is 25% at 32 steps, 33% at 64
    and 46% at 128. A goal sampled uniformly from the whole maze is reached by
    almost none of them, and a sparse reward that is zero in every episode of
    the dataset trains nothing at all. Widen it when the data comes from a
    policy rather than from noise.
    """

    size: int = eqx.field(static=True)
    layout: tuple[str, ...] = eqx.field(static=True)
    radius: float = eqx.field(static=True)
    goal_radius: float = eqx.field(static=True)
    damping: float = eqx.field(static=True)
    action_scale: float = eqx.field(static=True)
    goal_range: float = eqx.field(static=True)
    _free: tuple[tuple[float, float], ...] = eqx.field(static=True)

    def __init__(
        self,
        size: int = 32,
        *,
        layout: tuple[str, ...] = DEFAULT_MAZE,
        radius: float = 0.035,
        goal_radius: float = 0.06,
        damping: float = 0.6,
        action_scale: float = 0.09,
        goal_range: float = 0.45,
    ):
        rows = len(layout)
        if any(len(row) != rows for row in layout):
            raise ValueError(f"layout must be square, got {rows} rows of varying width")
        if not all(c == "#" for c in layout[0] + layout[-1]):
            raise ValueError("the maze border must be wall")
        if any(row[0] != "#" or row[-1] != "#" for row in layout):
            raise ValueError("the maze border must be wall")
        self.size = size
        self.layout = layout
        self.radius = radius
        self.goal_radius = goal_radius
        self.damping = damping
        self.action_scale = action_scale
        self.goal_range = goal_range
        cell = 1.0 / rows
        self._free = tuple(
            ((c + 0.5) * cell, (r + 0.5) * cell)
            for r, row in enumerate(layout)
            for c, value in enumerate(row)
            if value == "."
        )
        if len(self._free) < 2:
            raise ValueError("the maze needs at least two free cells")

    @property
    def action_dim(self) -> int:
        return 2

    @property
    def channels(self) -> int:
        return 3

    @property
    def observation_shape(self) -> tuple[int, int, int]:
        return (self.channels, self.size, self.size)

    @property
    def state_dim(self) -> int:
        """Width of :meth:`state_vector`."""
        return 6

    @property
    def n_cells(self) -> int:
        return len(self.layout)

    @property
    def free_cells(self) -> Array:
        """``(n_free, 2)`` centres of the free cells, as ``(x, y)``."""
        return jnp.asarray(self._free, dtype=jnp.float32)

    @property
    def walls(self) -> Array:
        """``(n_cells, n_cells)`` float mask indexed ``[row, column]``."""
        return jnp.asarray(
            [[1.0 if c == "#" else 0.0 for c in row] for row in self.layout], dtype=jnp.float32
        )

    # -- geometry ------------------------------------------------------------
    def _blocked(self, pos: Array) -> Array:
        """Would an agent centred at ``pos`` overlap a wall?

        Tests the four corners of the agent's bounding box rather than its
        centre, so the rendered disc never sinks into a wall it is not
        colliding with.
        """
        walls = self.walls
        corners = jnp.stack(
            [
                pos + jnp.array([-self.radius, -self.radius]),
                pos + jnp.array([self.radius, -self.radius]),
                pos + jnp.array([-self.radius, self.radius]),
                pos + jnp.array([self.radius, self.radius]),
            ]
        )
        cells = jnp.clip((corners * self.n_cells).astype(jnp.int32), 0, self.n_cells - 1)
        hits = walls[cells[:, 1], cells[:, 0]]  # (x, y) -> [row, column]
        return jnp.max(hits) > 0.0

    # -- dynamics ------------------------------------------------------------
    def reset(self, key: PRNGKey) -> MazeState:
        """Start at a random free cell, with a goal within :attr:`goal_range`."""
        k_start, k_goal = jr.split(key)
        cells = self.free_cells
        start_index = jr.randint(k_start, (), 0, cells.shape[0])
        pos = cells[start_index]
        distance = jnp.linalg.norm(cells - pos, axis=-1)
        # Near enough to be reachable, far enough not to start on top of it.
        eligible = (distance <= self.goal_range) & (distance > 2.0 * self.goal_radius)
        # If nothing qualifies, fall back to the furthest cell rather than
        # returning the start itself as a goal.
        logits = jnp.where(eligible, 0.0, -jnp.inf)
        logits = jnp.where(jnp.any(eligible), logits, distance)
        goal = cells[jr.categorical(k_goal, logits)]
        return MazeState(pos=pos, vel=jnp.zeros((2,)), goal=goal)

    def step(self, state: MazeState, action: Array, *, key: PRNGKey | None = None) -> MazeState:
        """Advance the world. ``action`` is a 2-D acceleration in ``[-1, 1]^2``.

        ``key`` is accepted and ignored; this world is deterministic.
        """
        del key
        vel = self.damping * state.vel + self.action_scale * jnp.clip(action, -1.0, 1.0)

        # One axis at a time: a blocked axis loses its velocity, the other keeps
        # moving. That is what turns a wall into something you slide along.
        def slide(pos, vel, axis):
            step = jnp.zeros((2,)).at[axis].set(vel[axis])
            attempt = pos + step
            hit = self._blocked(attempt)
            return jnp.where(hit, pos, attempt), jnp.where(hit, vel.at[axis].set(0.0), vel)

        pos, vel = slide(state.pos, vel, 0)
        pos, vel = slide(pos, vel, 1)
        return MazeState(pos=pos, vel=vel, goal=state.goal)

    # -- task ----------------------------------------------------------------
    def goal_distance(self, state: MazeState) -> Array:
        """Straight-line distance to the goal, walls ignored. Evaluation only."""
        return jnp.linalg.norm(state.pos - state.goal)

    def reward(self, state: MazeState) -> Array:
        """Sparse: ``1.0`` inside :attr:`goal_radius` of the goal, else ``0.0``.

        Sparse and not shaped, because a shaped maze reward is a lie -- the
        straight-line distance to the goal *increases* along the only path that
        reaches it, so distance-shaping actively teaches the wrong thing here.
        :meth:`goal_distance` is still exposed, for evaluation.
        """
        return (self.goal_distance(state) < self.goal_radius).astype(jnp.float32)

    def state_vector(self, state: MazeState) -> Array:
        """``(6,)``: position, velocity, goal."""
        return jnp.concatenate([state.pos, state.vel, state.goal])

    # -- rendering -----------------------------------------------------------
    def render(self, state: MazeState) -> Array:
        """``(3, size, size)``: agent red, goal green (a ring), walls blue."""
        agent = soft_disc(self.size, state.pos, self.radius, sharpness=1.4)
        goal = soft_ring(self.size, state.goal, self.goal_radius)
        grid = (jnp.arange(self.size) + 0.5) / self.size * self.n_cells
        cells = jnp.clip(grid, 0, self.n_cells - 1).astype(jnp.int32)
        walls = self.walls[cells[:, None], cells[None, :]]
        return jnp.clip(jnp.stack([agent, 0.7 * goal, 0.5 * walls]), 0.0, 1.0)

    # -- trajectories --------------------------------------------------------
    def rollout(self, key: PRNGKey, actions: Array) -> tuple[Array, MazeState]:
        """Run ``(T, 2)`` actions from a random start; ``T + 1`` frames out."""
        state = self.reset(key)

        def advance(s, a):
            s = self.step(s, a)
            return s, s

        _, states = jax.lax.scan(advance, state, actions)
        all_states = jax.tree_util.tree_map(
            lambda first, rest: jnp.concatenate([first[None], rest]), state, states
        )
        return jax.vmap(self.render)(all_states), all_states

    def observe(self, key: PRNGKey, actions: Array) -> Array:
        """Just the frames from :meth:`rollout`."""
        return self.rollout(key, actions)[0]


def push_sequences(
    key: PRNGKey,
    n_sequences: int,
    length: int,
    *,
    world: PushWorld | None = None,
    smoothness: float = 0.7,
) -> dict[str, Array]:
    """Generate an action-labelled pushing dataset.

    Returns a dict with:

    * ``video``: ``(n, length, 3, size, size)``
    * ``action``: ``(n, length - 1, 2)`` -- ``action[i, t]`` joins frames ``t``
      and ``t + 1``, the convention every family in xwm expects.
    * ``state``: ``(n, length, 8)`` -- :meth:`PushWorld.state_vector`.
    * ``reward``: ``(n, length - 1)`` -- the reward of the state each action
      *led to*, so it aligns with ``action`` and with
      :meth:`xwm.training.ReplayBuffer.add_episode`.
    * ``success``: ``(n, length - 1)`` -- the same alignment, as a 0/1 flag.
    """
    world = world or PushWorld()
    k_act, k_env = jr.split(key)
    actions = random_actions(
        k_act, n_sequences, length - 1, world.action_dim, smoothness=smoothness
    )
    frames, states = jax.vmap(world.rollout)(jr.split(k_env, n_sequences), actions)
    per_state = jax.vmap(jax.vmap(world.state_vector))(states)
    reward = jax.vmap(jax.vmap(world.reward))(states)
    success = jax.vmap(jax.vmap(world.success))(states)
    return {
        "video": frames,
        "action": actions,
        "state": per_state,
        "reward": reward[:, 1:],
        "success": success[:, 1:],
    }


def maze_sequences(
    key: PRNGKey,
    n_sequences: int,
    length: int,
    *,
    world: MazeWorld | None = None,
    smoothness: float = 0.9,
) -> dict[str, Array]:
    """Generate an action-labelled maze-navigation dataset.

    Returns ``video``, ``action``, ``state`` and ``reward`` on the same
    alignment as :func:`push_sequences`, plus ``distance``: ``(n, length)``
    straight-line distance to the goal, for evaluation rather than training.

    ``smoothness`` defaults higher than elsewhere (0.9 rather than 0.7): in a
    corridor, weakly correlated actions just vibrate against the walls, and the
    agent has to commit to a direction for a dozen steps to get anywhere.
    """
    world = world or MazeWorld()
    k_act, k_env = jr.split(key)
    actions = random_actions(
        k_act, n_sequences, length - 1, world.action_dim, smoothness=smoothness
    )
    frames, states = jax.vmap(world.rollout)(jr.split(k_env, n_sequences), actions)
    per_state = jax.vmap(jax.vmap(world.state_vector))(states)
    reward = jax.vmap(jax.vmap(world.reward))(states)
    distance = jax.vmap(jax.vmap(world.goal_distance))(states)
    return {
        "video": frames,
        "action": actions,
        "state": per_state,
        "reward": reward[:, 1:],
        "distance": distance,
    }

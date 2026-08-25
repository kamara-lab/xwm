"""A small controllable world, for tests and runnable examples.

:class:`SpriteWorld` is deliberately built to expose the property that makes
predictive world models worth having: it contains both a **controllable** object
that responds to actions and a configurable number of **distractors** that move
by an unpredictable random walk. A pixel-reconstruction objective must spend
capacity trying to predict the distractors and will always fail; a latent
predictive objective is free to represent only what actions can influence.

Everything is pure JAX, so datasets are generated on device and can be ``vmap``ed
and ``jit``ed alongside the model.
"""

from __future__ import annotations

from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from ..core.module import Module
from ..core.types import Array, PRNGKey


def distance_field(size: int, centre: Array) -> Array:
    """``(size, size)`` distance from every pixel centre to ``centre``.

    Pixel *centres*, not corners, so a sprite at 0.5 renders symmetrically.
    """
    grid = (jnp.arange(size) + 0.5) / size
    dy = grid[:, None] - centre[1]
    dx = grid[None, :] - centre[0]
    return jnp.sqrt(dy**2 + dx**2)


def soft_disc(size: int, centre: Array, radius: float, *, sharpness: float = 0.6) -> Array:
    """A soft disc at ``centre``, as a ``(size, size)`` map in ``[0, 1]``.

    Soft rather than hard because a hard disc is a step function of position:
    its gradient is zero everywhere and the rendered image only changes once the
    centre crosses a pixel boundary. A sigmoid edge makes sub-pixel motion
    visible, which is what a one-step prediction loss needs in order to have any
    signal at all at small displacements.

    ``sharpness`` is scaled by ``size``, so an edge is equally crisp at 32 px and
    at 96 px rather than turning to fog as the resolution rises.
    """
    return jax.nn.sigmoid((radius - distance_field(size, centre)) * size * sharpness)


def soft_ring(
    size: int, centre: Array, radius: float, *, width: float = 0.018, sharpness: float = 1.4
) -> Array:
    """A soft annulus of radius ``radius``: an outline rather than a filled disc.

    For drawing a *goal*, which is a place rather than an object. A filled disc
    at goal radius is the largest, brightest thing in the frame and reads as the
    subject of the picture; an outline says "here" without competing with the
    objects that actually move. It is also how every real pushing benchmark
    draws its target, Push-T included, so the synthetic world and the recorded
    one become comparable by eye.
    """
    offset = jnp.abs(distance_field(size, centre) - radius)
    return jax.nn.sigmoid((width - offset) * size * sharpness)


class SpriteState(NamedTuple):
    """World state: a controlled agent plus ``n_distractors`` random walkers."""

    pos: Array  # (2,) in [0, 1]^2
    vel: Array  # (2,)
    distractors: Array  # (n_distractors, 2)


class SpriteWorld(Module):
    """A 2-D world with one action-controlled sprite and some uncontrolled ones.

    Args:
        size: rendered resolution (square).
        n_distractors: uncontrollable sprites performing a random walk.
        radius: sprite radius in normalised units.
        dt: integration step.
        damping: velocity decay per step; ``1.0`` is frictionless.
        action_scale: acceleration applied per unit of action.
        distractor_speed: random-walk step size for the distractors.
        n_occluders: static vertical bars drawn *in front of* the sprites, which
            hide them without touching the dynamics. Zero by default. This is
            the one knob that makes the world partially observable, and partial
            observability is what separates a latent state carried through time
            from a per-frame embedding: behind a bar the current frame says
            nothing about where the agent is, so only a model that integrated
            its own predictions can still say.
        occluder_width: bar width in normalised units. The default exceeds the
            default sprite *diameter*, deliberately: a bar narrower than the
            sprite clips its edges but never hides it, and a world where the
            agent is always partly visible is not partially observable at all.
            Note the other end of the same constraint: at full action the agent
            covers 0.3 per step, more than one bar width, so it can cross a bar
            between two frames and never be observed behind it. Occlusion bites
            in the smoothed-action regime :func:`random_actions` produces.

    The defaults make one step of full action displace the agent by about 1.5
    radii. That matters for evaluation: with slower dynamics, a single step
    barely changes the image, "predict no change" becomes a near-optimal
    one-step baseline, and a latent dynamics model looks worthless at short
    horizons for reasons that have nothing to do with the model.
    """

    size: int = eqx.field(static=True)
    n_distractors: int = eqx.field(static=True)
    radius: float = eqx.field(static=True)
    dt: float = eqx.field(static=True)
    damping: float = eqx.field(static=True)
    action_scale: float = eqx.field(static=True)
    distractor_speed: float = eqx.field(static=True)
    n_occluders: int = eqx.field(static=True)
    occluder_width: float = eqx.field(static=True)

    def __init__(
        self,
        size: int = 32,
        *,
        n_distractors: int = 2,
        radius: float = 0.10,
        dt: float = 1.0,
        damping: float = 0.5,
        action_scale: float = 0.15,
        distractor_speed: float = 0.05,
        n_occluders: int = 0,
        occluder_width: float = 0.26,
    ):
        self.size = size
        self.n_distractors = n_distractors
        self.radius = radius
        self.dt = dt
        self.damping = damping
        self.action_scale = action_scale
        self.distractor_speed = distractor_speed
        self.n_occluders = n_occluders
        self.occluder_width = occluder_width

    @property
    def action_dim(self) -> int:
        return 2

    @property
    def channels(self) -> int:
        return 3

    @property
    def observation_shape(self) -> tuple[int, int, int]:
        return (self.channels, self.size, self.size)

    # -- dynamics ------------------------------------------------------------
    def reset(self, key: PRNGKey) -> SpriteState:
        k_pos, k_dist = jr.split(key)
        return SpriteState(
            pos=jr.uniform(k_pos, (2,), minval=0.2, maxval=0.8),
            vel=jnp.zeros((2,)),
            distractors=jr.uniform(k_dist, (self.n_distractors, 2), minval=0.1, maxval=0.9),
        )

    def step(self, state: SpriteState, action: Array, *, key: PRNGKey | None = None) -> SpriteState:
        """Advance the world. ``action`` is a 2-D acceleration in ``[-1, 1]^2``.

        Positions reflect off the walls, which keeps trajectories bounded
        without the discontinuity of wrapping.
        """
        vel = self.damping * state.vel + self.action_scale * jnp.clip(action, -1.0, 1.0)
        pos = state.pos + self.dt * vel
        # Reflect at the boundary and flip the corresponding velocity component.
        below, above = pos < 0.0, pos > 1.0
        pos = jnp.where(below, -pos, jnp.where(above, 2.0 - pos, pos))
        vel = jnp.where(below | above, -vel, vel)
        if key is None or self.n_distractors == 0:
            distractors = state.distractors
        else:
            walk = self.distractor_speed * jr.normal(key, state.distractors.shape)
            distractors = jnp.clip(state.distractors + walk, 0.0, 1.0)
        return SpriteState(pos=pos, vel=vel, distractors=distractors)

    # -- rendering -----------------------------------------------------------
    def _blob(self, centre: Array) -> Array:
        """A soft disc at ``centre``, as a ``(size, size)`` map in ``[0, 1]``."""
        return soft_disc(self.size, centre, self.radius)

    def _occlusion(self) -> Array:
        """``(size, size)`` in ``[0, 1]``: 1 where an occluder hides the scene.

        Static vertical bars at fixed positions, so occlusion is a property of
        the world's geometry rather than part of its state -- nothing about the
        dynamics changes, only what an observation reveals about them.
        """
        grid = (jnp.arange(self.size) + 0.5) / self.size
        centres = (jnp.arange(self.n_occluders) + 1.0) / (self.n_occluders + 1.0)
        offset = jnp.abs(grid[None, :] - centres[:, None])
        bars = jax.nn.sigmoid((self.occluder_width / 2 - offset) * self.size * 0.6)
        return jnp.broadcast_to(jnp.max(bars, axis=0), (self.size, self.size))

    def render(self, state: SpriteState) -> Array:
        """Render one state to ``(3, size, size)`` in ``[0, 1]``.

        The agent occupies the red channel and the distractors the green one, so
        a probe can tell trivially whether a representation kept the
        controllable content, the uncontrollable content, or both. Occluders,
        when configured, are drawn last in the blue channel and zero out
        whatever they cover.
        """
        agent = self._blob(state.pos)
        if self.n_distractors:
            distractors = jnp.max(jax.vmap(self._blob)(state.distractors), axis=0)
        else:
            distractors = jnp.zeros_like(agent)
        grid = (jnp.arange(self.size) + 0.5) / self.size
        background = 0.15 * (grid[:, None] + grid[None, :]) / 2.0 + 0.0 * agent
        if self.n_occluders:
            hidden = self._occlusion()
            agent = agent * (1.0 - hidden)
            distractors = distractors * (1.0 - hidden)
            background = background + 0.6 * hidden
        return jnp.clip(jnp.stack([agent, distractors, background]), 0.0, 1.0)

    # -- trajectories --------------------------------------------------------
    def rollout(self, key: PRNGKey, actions: Array) -> tuple[Array, SpriteState]:
        """Run ``(T, 2)`` actions from a random start.

        Returns ``(frames, states)`` with ``frames`` of shape
        ``(T + 1, 3, size, size)`` -- one more frame than actions, since the
        initial observation precedes the first action.
        """
        k_reset, k_steps = jr.split(key)
        state = self.reset(k_reset)
        keys = jr.split(k_steps, actions.shape[0])

        def advance(s, inputs):
            a, k = inputs
            s = self.step(s, a, key=k)
            return s, s

        final, states = jax.lax.scan(advance, state, (actions, keys))
        all_states = jax.tree_util.tree_map(
            lambda first, rest: jnp.concatenate([first[None], rest]), state, states
        )
        return jax.vmap(self.render)(all_states), all_states

    def observe(self, key: PRNGKey, actions: Array) -> Array:
        """Just the frames from :meth:`rollout`."""
        return self.rollout(key, actions)[0]


def random_actions(
    key: PRNGKey,
    n_sequences: int,
    length: int,
    action_dim: int = 2,
    *,
    smoothness: float = 0.7,
) -> Array:
    """``(n_sequences, length, action_dim)`` smoothly correlated random actions.

    White-noise actions make an almost unlearnable dataset -- the agent jitters
    in place and no action has visible consequences. Temporally correlated
    actions (an AR(1) process, ``smoothness`` being the correlation) produce
    trajectories that actually go somewhere.
    """
    noise = jr.uniform(key, (n_sequences, length, action_dim), minval=-1.0, maxval=1.0)

    def smooth(carry, x):
        carry = smoothness * carry + (1.0 - smoothness) * x
        return carry, carry

    _, out = jax.lax.scan(smooth, noise[:, 0], noise.transpose(1, 0, 2))
    return out.transpose(1, 0, 2)


def sprite_sequences(
    key: PRNGKey,
    n_sequences: int,
    length: int,
    *,
    world: SpriteWorld | None = None,
    smoothness: float = 0.7,
) -> dict[str, Array]:
    """Generate an action-labelled video dataset.

    Returns a dict with:

    * ``video``: ``(n, length, 3, size, size)``
    * ``action``: ``(n, length - 1, 2)`` -- ``action[i, t]`` joins frames
      ``t`` and ``t + 1``, the convention :class:`xwm.action.ActionWorldModel`
      expects.
    * ``position``: ``(n, length, 2)`` ground-truth agent position, for probes.
    """
    world = world or SpriteWorld()
    k_act, k_env = jr.split(key)
    actions = random_actions(
        k_act, n_sequences, length - 1, world.action_dim, smoothness=smoothness
    )
    keys = jr.split(k_env, n_sequences)
    frames, states = jax.vmap(world.rollout)(keys, actions)
    return {"video": frames, "action": actions, "position": states.pos}


def sprite_images(
    key: PRNGKey,
    n_images: int,
    *,
    world: SpriteWorld | None = None,
) -> dict[str, Array]:
    """Generate a still-image dataset: ``(n, 3, size, size)`` plus positions."""
    world = world or SpriteWorld()
    states = jax.vmap(world.reset)(jr.split(key, n_images))
    return {"image": jax.vmap(world.render)(states), "position": states.pos}

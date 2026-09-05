"""The evaluation protocol: does the loop measure control, and do the floors hold?

The load-bearing test here is the oracle. A model whose latent *is* the true
simulator state and whose dynamics *are* the real world must solve the task; if
it does not, the bug is in the loop -- the reset, the action scaling, the
frameskip, the success predicate -- and not in any world model. Every other test
in this file is downstream of that one.
"""

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import xwm
from xwm.bench.control import Context, run_episode
from xwm.bench.policies import make_policy
from xwm.data.worlds import PushState


class OracleModel(xwm.core.WorldModel):
    """Cheats completely: reads the simulator, steps the real world.

    It satisfies :class:`xwm.core.types.Plannable` and nothing else -- it cannot
    be trained and it never looks at a pixel. Its latent is the environment's
    own state vector, so ``readout`` selects the puck and the planner is
    searching over true dynamics. This is the control for the loop itself.
    """

    env: object
    goal: jnp.ndarray

    def __init__(self, env, goal):
        self.env = env
        self.goal = jnp.asarray(goal, jnp.float32)

    def loss(self, batch, *, key, target=None):
        raise NotImplementedError("the oracle is not trainable")

    def initial_state(self, frames, actions=None, *, key=None):
        del frames, actions, key
        return jnp.asarray(self.env.state(), jnp.float32)

    def readout(self, z):
        return z[2:4]  # PushState(pusher, puck, goal) -> the puck

    def goal_embedding(self, frame, *, key=None):
        del frame, key
        return self.goal

    def dynamics_fn(self, *, key=None):
        del key
        world = self.env.world

        def step(z, action):
            state = PushState(pusher=z[0:2], puck=z[2:4], goal=z[4:6])
            nxt = world.step(state, action)
            return jnp.concatenate([nxt.pusher, nxt.puck, nxt.goal])

        return step


def _oracle(env, goal):
    """The oracle for this task: its goal is where the puck should end up."""
    return OracleModel(env, goal[2:4])


@pytest.fixture(scope="module")
def task():
    return xwm.tasks.create("pusht/synthetic", episodes=6)


@pytest.fixture(scope="module")
def instances(task):
    return xwm.bench.sample_episodes(
        task.episodes(split="val"),
        n=6,
        goal_offset=task.spec.goal_offset,
        seed=0,
        frameskip=task.spec.frameskip,
    )


def _run(task, instances, policy_name, *, model_for=None, plan=None, budget=None):
    plan = plan or xwm.bench.PlanConfig(
        horizon=6,
        receding_horizon=1,
        planner_options={"n_samples": 256, "n_elites": 32, "n_iters": 4},
    )
    budget = task.spec.eval_budget if budget is None else budget
    env = task.make_env()
    planner = xwm.bench.build_planner(task, plan)
    results = []
    try:
        for i, instance in enumerate(instances):
            if policy_name == "planner":
                goal = task.env_state(
                    task.episodes(split="val")[instance.index]["state"][instance.goal], env
                )
                model = model_for(env, goal)
                policy = make_policy(
                    "planner",
                    task,
                    model=model,
                    planner=planner,
                    cost_fn=xwm.planning.goal_cost(
                        model.goal_embedding(None), kind="l2", readout=model.readout
                    ),
                    plan=plan,
                )
            else:
                policy = make_policy(policy_name, task)
            results.append(
                run_episode(
                    task,
                    env,
                    instance,
                    policy=policy,
                    plan=plan,
                    key=jr.PRNGKey(i),
                    budget=budget,
                )
            )
    finally:
        env.close()
    return results


# -- the control ------------------------------------------------------------
def test_oracle_dynamics_solve_the_task(task, instances):
    """True state, true dynamics: anything less than near-perfect is a loop bug."""
    results = _run(task, instances, "planner", model_for=_oracle)
    assert np.mean([r.success for r in results]) == 1.0
    assert np.mean([r.distance_closed for r in results]) > 0.9


def test_replay_reaches_the_goal(task, instances):
    """The reset-fidelity gate. Every model number is meaningless without it."""
    results = _run(task, instances, "replay")
    assert np.mean([r.success for r in results]) == 1.0


def test_noop_does_not_solve_the_task(task, instances):
    """If doing nothing succeeds, the goal is too close to the start to measure anything.

    Not *exactly* static: PushWorld resolves contact by projecting the puck out
    of the pusher every step, and the scripted demonstrator leaves the two
    touching, so an instance recorded mid-push can settle by a fraction of the
    goal radius under a zero action. A fraction of a percent of the gap is
    settling; anything more would mean the instances are trivial.
    """
    results = _run(task, instances, "noop")
    assert np.mean([r.success for r in results]) == 0.0
    assert max(abs(r.distance_closed) for r in results) < 0.01


def test_the_oracle_beats_the_floors(task, instances):
    oracle = _run(task, instances, "planner", model_for=_oracle)
    random = _run(task, instances, "random")
    assert np.mean([r.success for r in oracle]) > np.mean([r.success for r in random])
    assert np.mean([r.distance_closed for r in oracle]) > np.mean(
        [r.distance_closed for r in random]
    )


def test_instances_are_not_already_solved(task, instances):
    """The filter that keeps `noop` from scoring: every goal starts out of reach."""
    env = task.make_env()
    held = task.episodes(split="val")
    try:
        for instance in instances:
            start = task.env_state(held[instance.index]["state"][instance.start], env)
            goal = task.env_state(held[instance.index]["state"][instance.goal], env)
            assert not task.success(start, goal)
    finally:
        env.close()


def test_success_is_latched_but_the_episode_continues(task, instances):
    """``final_distance`` must stay honest about a planner that arrives and leaves."""
    results = _run(task, instances, "planner", model_for=_oracle)
    for r in results:
        assert r.steps == task.spec.eval_budget  # never stopped early
        if r.success:
            assert 0 < r.solved_at <= r.steps
            assert r.best_distance <= r.final_distance + 1e-6


# -- instances --------------------------------------------------------------
def test_sample_episodes_is_deterministic_and_model_free(task):
    held = task.episodes(split="val")
    a = xwm.bench.sample_episodes(held, n=5, goal_offset=6, seed=3)
    b = xwm.bench.sample_episodes(held, n=5, goal_offset=6, seed=3)
    assert a == b
    assert a != xwm.bench.sample_episodes(held, n=5, goal_offset=6, seed=4)


def test_sample_episodes_respects_the_goal_offset(task):
    held = task.episodes(split="val")
    for instance in xwm.bench.sample_episodes(held, n=20, goal_offset=6, seed=0):
        assert instance.goal - instance.start == 6
        assert instance.goal < held[instance.index]["state"].shape[0]


def test_sample_episodes_refuses_an_unreachable_offset(task):
    with pytest.raises(ValueError, match="longer than the goal offset"):
        xwm.bench.sample_episodes(task.episodes(split="val"), n=4, goal_offset=10_000)


# -- history windows --------------------------------------------------------
def test_context_pads_the_start_by_repeating_not_by_zeros():
    """A zero frame is not an observation; repeating the first one is the closest truth."""
    context = Context(history=3, action_width=2)
    first = jnp.ones((3, 4, 4))
    context.start(first)
    assert context.frames().shape == (3, 3, 4, 4)
    assert jnp.allclose(context.frames(), jnp.stack([first] * 3))
    assert context.actions().shape == (2, 2)


def test_context_rolls_frames_and_actions_together():
    context = Context(history=3, action_width=2)
    context.start(jnp.zeros((1,)))
    for i in (1.0, 2.0, 3.0):
        context.act(np.full(2, i, np.float32))
        context.observe(jnp.full((1,), i))
    assert jnp.allclose(context.frames().reshape(-1), jnp.array([1.0, 2.0, 3.0]))
    # Two actions retained for three frames: one per gap between them.
    assert jnp.allclose(context.actions().reshape(-1), jnp.array([2.0, 2.0, 3.0, 3.0]))


def test_markov_context_carries_no_actions():
    context = Context(history=1, action_width=2)
    context.start(jnp.zeros((1,)))
    assert context.actions() is None


# -- history-window models plan through the unchanged planners --------------
class WindowModel(xwm.core.WorldModel):
    """A latent that is a window of three frames, packed with its actions."""

    width: int = eqx.field(static=True)

    def __init__(self, width=4):
        self.width = width
        self.history = 3

    def loss(self, batch, *, key, target=None):
        raise NotImplementedError

    def initial_state(self, frames, actions=None, *, key=None):
        del key
        embeddings = jnp.stack([jnp.mean(f).repeat(self.width) for f in frames])
        past = (
            jnp.zeros((self.history, 2))
            if actions is None
            else jnp.concatenate([actions, jnp.zeros((1, 2))])
        )
        return jnp.concatenate([embeddings, past], axis=-1)

    def readout(self, z):
        return z[-1, : self.width]

    def goal_embedding(self, frame, *, key=None):
        del key
        return jnp.mean(frame).repeat(self.width)

    def dynamics_fn(self, *, key=None):
        del key

        def step(z, action):
            embeddings, past = z[:, : self.width], z[:, self.width :]
            past = past.at[-1].set(action)
            nxt = embeddings[-1] + jnp.mean(action)
            return jnp.concatenate(
                [
                    jnp.concatenate([embeddings[1:], nxt[None]]),
                    jnp.concatenate([past[1:], jnp.zeros((1, 2))]),
                ],
                axis=-1,
            )

        return step


def test_a_history_window_plans_through_an_unchanged_planner():
    """The whole point of packing the window into one array: CEM needs no changes."""
    model = WindowModel()
    z0 = model.initial_state(jnp.zeros((3, 2, 2)))
    assert z0.shape == (3, 6)
    cost = xwm.planning.goal_cost(jnp.full((4,), 2.0), kind="l2", readout=model.readout)
    plan = xwm.planning.CEM(4, 2, n_samples=64, n_elites=8, n_iters=3).plan(
        jr.PRNGKey(0), model.dynamics_fn(), z0, cost
    )
    assert plan.actions.shape == (4, 2)
    assert bool(jnp.isfinite(plan.cost))


def test_the_readout_ignores_stale_context():
    """Garbage in the older frames must not be charged against the plan."""
    model = WindowModel()
    goal = jnp.full((4,), 1.5)
    cost = xwm.planning.goal_cost(goal, kind="l2", readout=model.readout)
    clean = jnp.concatenate([jnp.full((3, 4), 1.5), jnp.zeros((3, 2))], axis=-1)
    stale = clean.at[:2, :4].set(-99.0)
    a, t = jnp.zeros((2,)), jnp.asarray(0)
    assert float(cost(clean, a, t)) == pytest.approx(0.0, abs=1e-6)
    assert float(cost(stale, a, t)) == pytest.approx(float(cost(clean, a, t)), abs=1e-6)


def test_the_window_shifts_by_exactly_one_frame():
    model = WindowModel()
    z = model.initial_state(jnp.arange(12.0).reshape(3, 2, 2))
    z_next = model.dynamics_fn()(z, jnp.array([0.5, 0.5]))
    assert z_next.shape == z.shape
    assert jnp.allclose(z_next[:-1, :4], z[1:, :4])  # older frames roll up
    assert not jnp.allclose(z_next[-1, :4], z[-1, :4])  # the newest is predicted


# -- objectives and planner/model compatibility -----------------------------
def _state_model(name, **kwargs):
    from xwm.cli.build import build_model, build_task

    config = xwm.config.from_dict(
        {
            "task": {"name": "pusht/synthetic", "overrides": {"observation": ["state"]}},
            "model": {"name": name, "kwargs": kwargs},
        }
    )
    task = build_task(config)
    return build_model(config, task, key=jr.PRNGKey(0)).eval_mode(), task


def test_the_return_objective_uses_the_family_that_owns_the_heads():
    """It cannot be assembled from outside: the reward head fixes the planner's convention.

    ``r(z_t, a_t)`` must be scored at the pre-transition latent, and the
    bootstrap needs the critic at the policy's own action. Reconstructing that
    here is how the two quietly disagree.
    """
    model, task = _state_model("tdmpc2", latent_dim=32, hidden_dim=32)
    plan = xwm.bench.PlanConfig(horizon=3, receding_horizon=1, objective="return")
    search, cost = xwm.bench.build_search(model, task, None, plan)
    assert search.cost_on == "current"
    z = model.encode(jr.normal(jr.PRNGKey(1), (task.spec.state_dim,)))
    plan_out = search.plan(jr.PRNGKey(2), model.dynamics_fn(), z, cost)
    assert plan_out.actions.shape == (3, task.spec.blocked_action_dim)
    assert bool(jnp.isfinite(plan_out.cost))


def test_the_return_objective_needs_a_model_that_has_the_heads():
    from xwm.cli.build import build_model, build_task

    config = xwm.config.from_dict(
        {
            "task": {"name": "pusht/synthetic"},
            "model": {"name": "jepa/action", "kwargs": {"size": "tiny", "patch_size": 8}},
        }
    )
    task = build_task(config)
    model = build_model(config, task, key=jr.PRNGKey(0))
    with pytest.raises(TypeError, match="reward and value heads"):
        xwm.bench.build_search(model, task, None, xwm.bench.PlanConfig(objective="return"))


def test_a_vision_model_refuses_a_state_task():
    """The mismatch has to name the task and the model, not a constructor kwarg."""
    from xwm.cli.build import build_task, model_kwargs

    config = xwm.config.from_dict(
        {
            "task": {"name": "pusht/synthetic", "overrides": {"observation": ["state"]}},
            "model": {"name": "jepa/action"},
        }
    )
    with pytest.raises(ValueError, match="encodes images, but task"):
        model_kwargs(config, build_task(config))


def test_a_discrete_model_refuses_a_continuous_planner():
    """A shape error from inside the dynamics names neither cause nor remedy."""
    model, task = _state_model("muzero", latent_dim=32, hidden_dim=32)
    with pytest.raises(TypeError, match="discrete actions"):
        xwm.bench.build_planner(task, xwm.bench.PlanConfig(), model)


def test_sample_episodes_explains_a_missing_state():
    episodes = [{"video": np.zeros((10, 3, 8, 8), np.uint8)}]
    with pytest.raises(ValueError, match="recorded 'state' per frame"):
        xwm.bench.sample_episodes(
            episodes, n=2, goal_offset=3, distance=lambda a, b: 1.0, min_distance=0.1
        )

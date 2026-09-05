"""Planners: do they actually find low-cost action sequences?

These use analytic dynamics rather than a learned model. A planner test should
fail when the *search* is broken, not when the world model happens to be badly
trained -- so the dynamics here are exactly solvable and the optimum is known.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import xwm.planning as P
from xwm.core.rollout import rollout, rollout_cost, teacher_forced_rollout

STEP = 0.2


def dynamics(z, a, *, key=None):
    """Latent is a position; the action moves it. Optimum is reachable exactly."""
    return z + STEP * a


def _planners(horizon=8, action_dim=2):
    return [
        P.CEM(horizon, action_dim, n_samples=256, n_elites=32, n_iters=6),
        P.MPPI(horizon, action_dim, n_samples=256, n_iters=6, temperature=0.05),
        P.GradientPlanner(horizon, action_dim, n_steps=150, learning_rate=0.1),
    ]


# -- goal costs over structured latents --------------------------------------
def test_goal_cost_readout_scores_only_the_selected_slice():
    """A history window must not be charged for its stale context frames."""
    goal = jnp.ones((2,))
    cost = P.goal_cost(goal, kind="l2", readout=lambda z: z[-1])
    matching = jnp.stack([jnp.full((2,), -99.0), goal])
    assert float(cost(matching, jnp.zeros((2,)), jnp.asarray(0))) == pytest.approx(0.0, abs=1e-6)
    stale_only = jnp.stack([goal, jnp.full((2,), -99.0)])
    assert float(cost(stale_only, jnp.zeros((2,)), jnp.asarray(0))) > 1.0


def test_goal_cost_without_readout_is_unchanged():
    goal = jnp.ones((1, 2))
    z = jnp.zeros((1, 2))
    plain = P.goal_cost(goal, kind="l2")
    identity = P.goal_cost(goal, kind="l2", readout=lambda z: z)
    a, t = jnp.zeros((2,)), jnp.asarray(0)
    assert float(plain(z, a, t)) == pytest.approx(float(identity(z, a, t)))


# -- rollout primitives -----------------------------------------------------
def test_rollout_returns_states_after_each_action():
    z0 = jnp.zeros((1, 2))
    actions = jnp.ones((3, 2))
    traj = rollout(dynamics, z0, actions)
    assert traj.shape == (3, 1, 2)
    assert jnp.allclose(traj[0], z0 + STEP)
    assert jnp.allclose(traj[-1], z0 + 3 * STEP)


def test_rollout_cost_matches_manual_sum():
    z0 = jnp.zeros((1, 2))
    actions = 0.5 * jnp.ones((4, 2))
    cost_fn = P.goal_cost(jnp.ones((1, 2)), kind="l2")
    traj = rollout(dynamics, z0, actions)
    manual = sum(cost_fn(traj[t], actions[t], jnp.asarray(t)) for t in range(4))
    assert jnp.allclose(rollout_cost(dynamics, z0, actions, cost_fn), manual, atol=1e-5)


def test_teacher_forced_rollout_uses_ground_truth():
    latents = jnp.arange(4.0).reshape(4, 1, 1)
    actions = jnp.ones((3, 1))
    out = teacher_forced_rollout(dynamics, latents, actions)
    assert out.shape == (3, 1, 1)
    # Each prediction starts from the *true* latent, not the previous prediction.
    assert jnp.allclose(out[:, 0, 0], latents[:3, 0, 0] + STEP)


# -- costs ------------------------------------------------------------------
def test_latent_distance_kinds():
    a, b = jnp.array([[1.0, 0.0]]), jnp.array([[1.0, 0.0]])
    for kind in ("l1", "l2", "cosine"):
        assert float(P.latent_distance(a, b, kind)) < 1e-6
    assert float(P.latent_distance(a, -b, "cosine")) == pytest.approx(2.0, abs=1e-5)
    with pytest.raises(ValueError, match="unknown distance"):
        P.latent_distance(a, b, "nope")


def test_terminal_only_cost_charges_one_step():
    goal = jnp.ones((1, 2))
    cost = P.goal_cost(goal, horizon=3, terminal_only=True)
    z = jnp.zeros((1, 2))
    assert float(cost(z, jnp.zeros((2,)), jnp.asarray(0))) == 0.0
    assert float(cost(z, jnp.zeros((2,)), jnp.asarray(2))) > 0.0


def test_terminal_only_requires_horizon():
    with pytest.raises(ValueError, match="requires horizon"):
        P.goal_cost(jnp.ones((1, 2)), terminal_only=True)


def test_action_penalty_discourages_large_actions():
    cost = P.goal_cost(jnp.zeros((1, 2)), action_penalty=1.0)
    z = jnp.zeros((1, 2))
    small = float(cost(z, jnp.zeros((2,)), jnp.asarray(0)))
    large = float(cost(z, jnp.ones((2,)), jnp.asarray(0)))
    assert large > small


def test_discount_prefers_earlier_progress():
    cost = P.goal_cost(jnp.ones((1, 2)), discount=0.5)
    z = jnp.zeros((1, 2))
    early = float(cost(z, jnp.zeros((2,)), jnp.asarray(0)))
    late = float(cost(z, jnp.zeros((2,)), jnp.asarray(3)))
    assert late < early


def test_sum_costs_weights():
    a = P.goal_cost(jnp.zeros((1, 1)))
    combined = P.sum_costs(a, a, weights=(1.0, 3.0))
    z, act, t = jnp.ones((1, 1)), jnp.zeros((1,)), jnp.asarray(0)
    assert float(combined(z, act, t)) == pytest.approx(4 * float(a(z, act, t)))
    with pytest.raises(ValueError, match="weights"):
        P.sum_costs(a, a, weights=(1.0,))


def test_reward_cost_negates():
    cost = P.reward_cost(lambda z: jnp.sum(z))
    assert float(cost(jnp.ones((2,)), jnp.zeros((1,)), jnp.asarray(0))) == -2.0


# -- planners ---------------------------------------------------------------
@pytest.mark.parametrize("planner", _planners(), ids=lambda p: type(p).__name__)
def test_planner_reaches_a_reachable_goal(planner):
    z0 = jnp.zeros((1, 2))
    goal = jnp.array([[1.0, -0.5]])
    plan = planner.plan(jr.PRNGKey(0), dynamics, z0, P.goal_cost(goal, kind="l2"))
    assert plan.actions.shape == (planner.horizon, planner.action_dim)
    final = z0
    for a in plan.actions:
        final = dynamics(final, a)
    assert float(jnp.max(jnp.abs(final - goal))) < 0.1


@pytest.mark.parametrize("planner", _planners(), ids=lambda p: type(p).__name__)
def test_planner_beats_doing_nothing(planner):
    z0 = jnp.zeros((1, 2))
    cost_fn = P.goal_cost(jnp.array([[1.0, -0.5]]), kind="l2")
    plan = planner.plan(jr.PRNGKey(0), dynamics, z0, cost_fn)
    idle = rollout_cost(dynamics, z0, jnp.zeros_like(plan.actions), cost_fn)
    assert float(plan.cost) < float(idle)


@pytest.mark.parametrize("planner", _planners(), ids=lambda p: type(p).__name__)
def test_planner_respects_bounds(planner):
    import equinox as eqx

    bounded = eqx.tree_at(
        lambda p: (p.low, p.high), planner, (jnp.full((2,), -0.1), jnp.full((2,), 0.1))
    )
    plan = bounded.plan(
        jr.PRNGKey(0), dynamics, jnp.zeros((1, 2)), P.goal_cost(jnp.full((1, 2), 10.0))
    )
    assert float(jnp.max(plan.actions)) <= 0.1 + 1e-6
    assert float(jnp.min(plan.actions)) >= -0.1 - 1e-6


def test_planners_are_jittable():
    planner = P.CEM(4, 2, n_samples=64, n_elites=8, n_iters=2)
    cost_fn = P.goal_cost(jnp.ones((1, 2)))
    f = jax.jit(lambda k: planner.plan(k, dynamics, jnp.zeros((1, 2)), cost_fn))
    assert f(jr.PRNGKey(0)).actions.shape == (4, 2)


def test_cem_rejects_more_elites_than_samples():
    with pytest.raises(ValueError, match="n_elites"):
        P.CEM(4, 2, n_samples=8, n_elites=16)


def test_cem_std_does_not_collapse_below_floor():
    planner = P.CEM(4, 2, n_samples=128, n_elites=16, n_iters=8, min_std=0.05)
    plan = planner.plan(jr.PRNGKey(0), dynamics, jnp.zeros((1, 2)), P.goal_cost(jnp.zeros((1, 2))))
    assert float(jnp.min(plan.std)) >= 0.05 - 1e-6


def test_warm_start_is_used():
    """A warm start at the known optimum must not be made worse."""
    planner = P.CEM(4, 2, n_samples=64, n_elites=8, n_iters=1)
    goal = jnp.array([[0.8, 0.8]])
    cost_fn = P.goal_cost(goal, kind="l2")
    optimal = jnp.full((4, 2), 1.0)
    cold = planner.plan(jr.PRNGKey(0), dynamics, jnp.zeros((1, 2)), cost_fn)
    warm = planner.plan(jr.PRNGKey(0), dynamics, jnp.zeros((1, 2)), cost_fn, init_mean=optimal)
    assert float(warm.cost) < float(cold.cost)


def test_shift_plan():
    mean = jnp.arange(6.0).reshape(3, 2)
    shifted = P.shift_plan(mean)
    assert jnp.allclose(shifted[:-1], mean[1:])
    assert jnp.allclose(shifted[-1], mean[-1])


def test_control_step_returns_first_action():
    planner = P.CEM(4, 2, n_samples=64, n_elites=8, n_iters=2)
    step = P.control_step(
        jr.PRNGKey(0), planner, dynamics, jnp.zeros((1, 2)), P.goal_cost(jnp.ones((1, 2)))
    )
    assert step.action.shape == (2,)
    assert jnp.allclose(step.action, step.plan.actions[0])
    assert step.warm_start.shape == (4, 2)


def test_mpc_drives_cost_down():
    planner = P.CEM(6, 2, n_samples=256, n_elites=32, n_iters=4)
    goal = jnp.array([[1.0, -0.5]])
    obs, actions, costs = P.run_mpc(
        jr.PRNGKey(0), planner, dynamics, P.goal_cost(goal, kind="l2"),
        encode=lambda o: o, step_env=dynamics,
        observation=jnp.zeros((1, 2)), n_steps=10,
    )
    assert len(obs) == 11 and len(actions) == 10 and len(costs) == 10
    assert float(costs[-1]) < float(costs[0]) / 10
    assert float(jnp.max(jnp.abs(obs[-1] - goal))) < 0.1


def test_planning_works_with_a_learned_model(key):
    """End to end: a real ActionWorldModel's dynamics must be plannable."""
    import xwm

    model = xwm.families.jepa.action_world_model(
        key=key, action_dim=2,
        encoder=xwm.encoders.ImageEncoder(
            key=key, img_size=16, patch_size=4, depth=2, embed_dim=64, num_heads=4
        ),
        dynamics_kwargs={"depth": 2, "pred_dim": 32, "num_heads": 4},
    )
    z0 = model.encode(jr.normal(key, (3, 16, 16)))
    z_goal = model.encode(jr.normal(jr.PRNGKey(1), (3, 16, 16)))
    planner = P.CEM(4, 2, n_samples=64, n_elites=8, n_iters=3)
    plan = planner.plan(key, model.dynamics_fn(), z0, P.goal_cost(z_goal, kind="l2"))
    assert plan.actions.shape == (4, 2)
    assert bool(jnp.isfinite(plan.cost))


def test_run_mpc_hands_its_whole_plan_to_an_observer(key):
    """``on_step`` sees the plan, not only the action taken from it.

    The costs ``run_mpc`` returns are one number per step; a diagnosis needs the
    proposal and its spread, which is why the hook passes the ``ControlStep``.
    """
    planner = P.CEM(horizon=3, action_dim=2, n_samples=32, n_elites=8, n_iters=2)
    cost = P.goal_cost(jnp.ones((4,)))
    seen = []

    P.run_mpc(
        key,
        planner,
        lambda z, a: z + jnp.pad(a, (0, z.shape[-1] - a.shape[-1])),
        cost,
        encode=lambda observation: observation,
        step_env=lambda observation, action: observation + jnp.pad(action, (0, 2)),
        observation=jnp.zeros((4,)),
        n_steps=4,
        on_step=lambda t, step: seen.append((t, step)),
    )

    assert [t for t, _ in seen] == [0, 1, 2, 3]
    for _, step in seen:
        assert step.plan.mean.shape == (3, 2)
        assert step.plan.std.shape == (3, 2)
        assert jnp.allclose(step.action, step.plan.actions[0])

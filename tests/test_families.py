"""The three families: TD-MPC2, MuZero, and what they share with JEPA."""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import xwm


def _grad_norm(tree):
    leaves = jax.tree_util.tree_leaves(eqx.filter(tree, eqx.is_inexact_array))
    return float(jnp.sqrt(sum(jnp.sum(x**2) for x in leaves))) if leaves else 0.0


# -- shared building blocks --------------------------------------------------
def test_simnorm_produces_group_simplices(key):
    """Bounded but not collapsed -- what a latent that gets unrolled needs."""
    out = xwm.nn.SimNorm(4)(jr.normal(key, (16,)))
    groups = out.reshape(4, 4)
    assert jnp.allclose(jnp.sum(groups, axis=-1), 1.0, atol=1e-5)
    assert bool(jnp.all(out >= 0)) and bool(jnp.all(out <= 1))


def test_simnorm_validates():
    with pytest.raises(ValueError, match="at least 2"):
        xwm.nn.SimNorm(1)
    with pytest.raises(ValueError, match="divisible"):
        xwm.nn.SimNorm(3)(jnp.zeros(16))


def test_two_hot_is_exact_and_invertible():
    """The encoding must reconstruct the value, or reward targets are biased."""
    bins, low, high = 11, -5.0, 5.0
    grid = jnp.linspace(low, high, bins)
    for value in (-5.0, -2.3, 0.0, 1.0, 3.7, 5.0):
        encoded = xwm.heads.two_hot(jnp.asarray(value), bins, low, high)
        assert jnp.isclose(jnp.sum(encoded), 1.0, atol=1e-5)
        assert jnp.isclose(jnp.sum(encoded * grid), value, atol=1e-4)


def test_two_hot_clips_out_of_range():
    encoded = xwm.heads.two_hot(jnp.asarray(99.0), 5, 0.0, 1.0)
    assert jnp.isclose(jnp.sum(encoded), 1.0)
    assert jnp.argmax(encoded) == 4


def test_symlog_roundtrip():
    for value in (0.0, 1.0, -1234.5, 1e5):
        x = jnp.asarray(value)
        assert jnp.allclose(xwm.heads.symexp(xwm.heads.symlog(x)), x, rtol=1e-4)


def test_categorical_scalar_decodes_across_magnitudes():
    scalar = xwm.heads.CategoricalScalar(101, -10.0, 10.0, transform=True)
    for value in (0.0, 1.0, -3.5, 250.0, -5000.0):
        logits = jnp.log(scalar.encode(jnp.asarray(value)) + 1e-12)
        assert jnp.isclose(scalar.decode(logits), value, rtol=0.05, atol=0.05)


def test_categorical_scalar_validates():
    with pytest.raises(ValueError, match="two bins"):
        xwm.heads.CategoricalScalar(1)
    with pytest.raises(ValueError, match="must exceed"):
        xwm.heads.CategoricalScalar(10, 1.0, 0.0)


def test_q_ensemble_pessimism(key):
    """The pessimistic aggregate must not exceed the mean of the members."""
    q = xwm.heads.QEnsemble(16, 3, key=key, n_members=5, subset_size=2)
    z, action = jr.normal(key, (16,)), jnp.zeros(3)
    values = q.values(z, action)
    assert values.shape == (5,)
    assert q.pessimistic(z, action) == pytest.approx(float(jnp.min(values)), abs=1e-5)
    subset = q.pessimistic(z, action, key=jr.PRNGKey(1))
    assert float(subset) >= float(jnp.min(values)) - 1e-5
    assert float(subset) <= float(jnp.mean(values)) + 1e-5


def test_q_ensemble_validates(key):
    with pytest.raises(ValueError, match="subset_size"):
        xwm.heads.QEnsemble(8, 2, key=key, n_members=2, subset_size=5)


def test_policy_is_bounded_with_a_corrected_log_prob(key):
    policy = xwm.heads.GaussianPolicy(16, 4, key=key)
    out = policy.sample(jr.normal(key, (16,)), jr.PRNGKey(1))
    assert out.action.shape == (4,)
    assert bool(jnp.all(jnp.abs(out.action) <= 1.0))
    assert jnp.ndim(out.log_prob) == 0 and bool(jnp.isfinite(out.log_prob))
    # Deterministic mean action when no key is given.
    assert jnp.allclose(policy.act(jr.normal(key, (16,))), policy.act(jr.normal(key, (16,))))


def test_mlp_dynamics_is_near_identity_and_bounded(key):
    dyn = xwm.dynamics.MLPDynamics(16, 3, key=key, normalize="simnorm", simnorm_groups=4)
    z = xwm.nn.SimNorm(4)(jr.normal(key, (16,)))
    z_next = dyn(z, jnp.zeros(3))
    assert z_next.shape == z.shape
    # SimNorm output: still on the simplices after a step.
    assert jnp.allclose(jnp.sum(z_next.reshape(4, 4), axis=-1), 1.0, atol=1e-5)


def test_mlp_dynamics_responds_to_the_action(key):
    dyn = xwm.dynamics.MLPDynamics(16, 3, key=key, normalize="layernorm")
    z = jr.normal(key, (16,))
    assert not jnp.allclose(dyn(z, jnp.ones(3)), dyn(z, -jnp.ones(3)))


def test_state_encoder_shapes(key):
    enc = xwm.encoders.StateEncoder(11, 32, key=key)
    out = enc(jr.normal(key, (11,)))
    assert out.shape == (1, 32) and enc.n_tokens == 1


# -- TD-MPC2 -----------------------------------------------------------------
def _tdmpc2(key, **kw):
    return xwm.families.tdmpc2.tdmpc2(
        action_dim=4, observation="state", state_dim=9, latent_dim=32,
        hidden_dim=64, horizon=3, key=key, **kw
    )


def _tdmpc2_batch(key, batch=4, horizon=3):
    return {
        "observation": jr.normal(key, (batch, horizon + 1, 9)),
        "action": jr.uniform(key, (batch, horizon, 4), minval=-1.0, maxval=1.0),
        "reward": jr.normal(key, (batch, horizon)),
    }


def test_tdmpc2_loss_and_metrics(key):
    model = _tdmpc2(key)
    loss, metrics = model.loss(_tdmpc2_batch(key), key=key, target=model)
    assert jnp.ndim(loss) == 0 and bool(jnp.isfinite(loss))
    assert {
        "loss_consistency", "loss_reward", "loss_value", "loss_policy", "q_mean", "entropy"
    } <= set(metrics)


def test_tdmpc2_requires_a_target(key):
    model = _tdmpc2(key)
    with pytest.raises(ValueError, match="EMA target"):
        model.loss(_tdmpc2_batch(key), key=key)
    assert model.uses_target


def test_tdmpc2_gradients_reach_every_component(key):
    model = _tdmpc2(key)
    batch = _tdmpc2_batch(key)
    grads = eqx.filter_grad(lambda m: m.loss(batch, key=key, target=model)[0])(model)
    for name in ("encoder", "dynamics", "reward", "critic", "policy"):
        assert _grad_norm(getattr(grads, name)) > 0, f"no gradient reached {name}"


def test_tdmpc2_acts_and_plans(key):
    model = _tdmpc2(key)
    observation = jr.normal(key, (9,))
    assert model.act(observation).shape == (4,)
    z = model.encode(observation)
    assert z.shape == (32,)
    planner = xwm.planning.MPPI(4, 4, n_samples=32, n_iters=2)
    cost = xwm.planning.goal_cost(model.encode(jr.normal(jr.PRNGKey(1), (9,))))
    plan = planner.plan(key, model.dynamics_fn(), z, cost)
    assert plan.actions.shape == (4, 4)


def test_tdmpc2_trains_with_the_trainer(key):
    """The family-agnostic Trainer must handle it, EMA target included."""
    model = _tdmpc2(key)
    trainer = xwm.training.Trainer(model, xwm.training.adamw(1e-3), ema_momentum=0.9)
    batches = (_tdmpc2_batch(jr.fold_in(key, i)) for i in range(20))
    state, history = trainer.fit(batches, steps=10, key=key, log_every=5)
    assert int(state.step) == 10
    assert state.target is not None
    assert history[-1]["loss"] != history[0]["loss"]


# -- MuZero ------------------------------------------------------------------
N_ACTIONS = 5


def _muzero(key, **kw):
    return xwm.families.muzero.muzero(
        n_actions=N_ACTIONS, observation="state", state_dim=8, latent_dim=32,
        hidden_dim=64, horizon=3, key=key, **kw
    )


def _muzero_batch(key, batch=4, horizon=3, time_axis=True):
    shape = (batch, horizon + 1, 8) if time_axis else (batch, 8)
    return {
        "observation": jr.normal(key, shape),
        "action": jr.randint(key, (batch, horizon), 0, N_ACTIONS),
        "reward": jr.normal(key, (batch, horizon)),
        "value_target": jr.normal(key, (batch, horizon + 1)),
        "policy_target": jax.nn.softmax(jr.normal(key, (batch, horizon + 1, N_ACTIONS))),
    }


def test_muzero_loss_and_metrics(key):
    model = _muzero(key)
    loss, metrics = model.loss(_muzero_batch(key), key=key)
    assert jnp.ndim(loss) == 0 and bool(jnp.isfinite(loss))
    assert {"loss_reward", "loss_value", "loss_policy"} <= set(metrics)
    assert not model.uses_target


def test_muzero_gradients_reach_every_component(key):
    model = _muzero(key)
    batch = _muzero_batch(key)
    grads = eqx.filter_grad(lambda m: m.loss(batch, key=key)[0])(model)
    for name in ("encoder", "dynamics", "reward", "value", "policy"):
        assert _grad_norm(getattr(grads, name)) > 0, f"no gradient reached {name}"


def test_muzero_latent_keeps_variance(key):
    """SimNorm here would pin every latent to the uniform point of each simplex."""
    model = _muzero(key)
    _, metrics = model.loss(_muzero_batch(key), key=key)
    assert float(metrics["latent_std"]) > 0.01


def test_muzero_accepts_the_replay_buffer_layout(key):
    """The same batch must feed both reward-driven families."""
    model = _muzero(key)
    with_time = model.loss(_muzero_batch(key, time_axis=True), key=key)[0]
    without = model.loss(_muzero_batch(key, time_axis=False), key=key)[0]
    assert bool(jnp.isfinite(with_time)) and bool(jnp.isfinite(without))


def test_muzero_trains_from_a_replay_buffer(key):
    """End to end through the buffer, with MuZero's extra search targets."""
    n_actions = N_ACTIONS
    buffer = xwm.training.ReplayBuffer(
        500, (8,), (), extra={"value_target": (), "policy_target": (n_actions,)}, seed=0
    )
    rng = np.random.default_rng(0)
    for _ in range(6):
        length = 8
        buffer.add_episode(
            rng.normal(size=(length + 1, 8)),
            rng.integers(0, n_actions, length),
            rng.normal(size=length),
            value_target=rng.normal(size=length + 1),
            policy_target=np.full((length + 1, n_actions), 1.0 / n_actions),
        )
    model = _muzero(key)
    trainer = xwm.training.Trainer(model, xwm.training.adamw(1e-3))
    batches = (buffer.sample(4, 3) for _ in range(6))
    state, history = trainer.fit(batches, steps=5, key=key, log_every=5)
    assert int(state.step) == 5 and bool(jnp.isfinite(jnp.asarray(history[-1]["loss"])))


def test_muzero_functions_have_the_shapes_mcts_needs(key):
    model = _muzero(key)
    z = model.represent(jr.normal(key, (8,)))
    z_next, reward = model.recurrent(z, jnp.asarray(2))
    logits, value = model.predict(z)
    assert z_next.shape == z.shape
    assert jnp.ndim(reward) == 0 and jnp.ndim(value) == 0
    assert logits.shape == (N_ACTIONS,)


def test_muzero_trains_with_the_trainer(key):
    model = _muzero(key)
    trainer = xwm.training.Trainer(model, xwm.training.adamw(1e-3))
    batches = (_muzero_batch(jr.fold_in(key, i)) for i in range(20))
    state, history = trainer.fit(batches, steps=10, key=key, log_every=5)
    assert int(state.step) == 10 and state.target is None
    assert history[-1]["loss"] < history[0]["loss"]


# -- MCTS --------------------------------------------------------------------
def test_mcts_concentrates_on_the_rewarding_action(key):
    """The search must find a payoff one step away.

    This is what catches scoring children by value alone: without the child's
    reward in the selection score, the search is blind to immediate reward.
    """
    mcts = xwm.planning.MCTS(N_ACTIONS, n_simulations=64, max_depth=8)
    flat = lambda z: (jnp.zeros(N_ACTIONS), jnp.asarray(0.0))  # noqa: E731
    for good in range(N_ACTIONS):
        recurrent = lambda z, a, g=good: (z, jnp.where(a == g, 5.0, -1.0))  # noqa: E731
        result = mcts.search(key, jnp.zeros(8), recurrent, flat, add_noise=False)
        assert int(result.action) == good, f"missed the payoff on action {good}"
        assert float(result.policy[good]) > 0.5


def test_mcts_policy_is_a_distribution_over_the_budget(key):
    mcts = xwm.planning.MCTS(N_ACTIONS, n_simulations=32, max_depth=6)
    result = mcts.search(
        key, jnp.zeros(8),
        lambda z, a: (z, jnp.asarray(0.0)),
        lambda z: (jnp.zeros(N_ACTIONS), jnp.asarray(0.0)),
    )
    assert jnp.isclose(jnp.sum(result.policy), 1.0, atol=1e-5)
    assert int(jnp.sum(result.visits)) == mcts.n_simulations


def test_mcts_is_jittable_and_drives_a_muzero_model(key):
    model = _muzero(key)
    recurrent, predict = model.search_fns()
    mcts = xwm.planning.MCTS(N_ACTIONS, n_simulations=16, max_depth=6)
    search = jax.jit(lambda k, z: mcts.search(k, z, recurrent, predict))
    result = search(key, model.represent(jr.normal(key, (8,))))
    assert result.policy.shape == (N_ACTIONS,)
    assert bool(jnp.isfinite(result.value))


def test_mcts_validates():
    with pytest.raises(ValueError, match="two actions"):
        xwm.planning.MCTS(1)


# -- registry ----------------------------------------------------------------
def test_registry_covers_every_family():
    names = xwm.families.available()
    assert xwm.families.families() == ["jepa", "muzero", "tdmpc2"]
    assert "tdmpc2" in names and "muzero" in names
    assert any(name.startswith("jepa/") for name in names)


def test_registry_builds_the_new_families():
    tdmpc2 = xwm.families.create(
        "tdmpc2", action_dim=3, observation="state", state_dim=6,
        latent_dim=16, hidden_dim=32,
    )
    assert isinstance(tdmpc2, xwm.families.tdmpc2.TDMPC2)
    muzero = xwm.families.create(
        "muzero", n_actions=4, observation="state", state_dim=6,
        latent_dim=16, hidden_dim=32,
    )
    assert isinstance(muzero, xwm.families.muzero.MuZero)
    with pytest.raises(KeyError, match="unknown model"):
        xwm.families.create("nope")


def test_families_share_encoders_and_planners(key):
    """The point of the layout: one encoder, three families."""
    encoder = xwm.encoders.StateEncoder(9, 32, key=key)
    tdmpc2 = xwm.families.tdmpc2.tdmpc2(action_dim=4, encoder=encoder, key=key, hidden_dim=64)
    muzero = xwm.families.muzero.muzero(n_actions=4, encoder=encoder, key=key, hidden_dim=64)
    observation = jr.normal(key, (9,))
    assert tdmpc2.encode(observation).shape == muzero.represent(observation).shape


def test_state_dim_and_agents_line_up_for_a_robot(key):
    """A robot's state_dim must build an agent without a manual shape fix."""
    pytest.importorskip("newton")
    env = xwm.envs.FrankaEnv(xwm.envs.FrankaConfig(image_size=32))
    agent = xwm.families.tdmpc2.tdmpc2(
        action_dim=env.action_dim, observation="state", state_dim=env.state_dim,
        latent_dim=32, hidden_dim=64, key=key,
    )
    z = agent.encode(jnp.asarray(env.state_observation()))
    assert z.shape == (32,)
    assert agent.act(jnp.asarray(env.state_observation())).shape == (env.action_dim,)


def test_muzero_dynamics_fn_agrees_with_its_search_closure(key):
    """The two spellings of MuZero's dynamics must not drift apart.

    ``recurrent`` is what MCTS calls and returns a reward alongside the latent;
    ``dynamics_fn`` is the ``(z, a) -> z'`` closure every planner and the
    goal-conditioned evaluator call. They are the same transition, so a change
    to one that misses the other has to fail here.
    """
    model = xwm.families.muzero.muzero(
        n_actions=5, observation="state", state_dim=6, latent_dim=32, hidden_dim=32, key=key
    )
    z = model.encode(jr.normal(key, (6,)))
    step = model.dynamics_fn()
    recurrent, _ = model.search_fns()
    for action in range(5):
        assert jnp.allclose(step(z, jnp.int32(action)), recurrent(z, jnp.int32(action))[0])
    # A one-hot vector is the same action as its index.
    assert jnp.allclose(step(z, jnp.int32(2)), step(z, jnp.eye(5)[2]))

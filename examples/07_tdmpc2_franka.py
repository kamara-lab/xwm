"""TD-MPC2 on a Franka reach task: learn the model *and* the value, then plan.

The contrast with `examples/05_planning.py` and `06_franka_newton.py` is the
learning signal. There, a JEPA learned a representation from observation alone
and a planner searched it against a goal *image*; the planner could only see as
far as its horizon, and on the Franka it could not beat doing nothing.

TD-MPC2 changes what trains the latent. Nothing anchors it to the observation --
no reconstruction, no embedding-prediction target. The latent is whatever makes
reward and value predictable, and the value head is what lets a horizon-3
planner act as though it could see much further: everything beyond step three is
summarised in ``V(z_H)``.

The loop is ordinary online RL: collect with the planner, store trajectory
slices, train on them, repeat. Evaluation is the only thing that touches ground
truth -- true tool distance to the goal, against three baselines.

Needs the Newton extra::

    pip install "xwm[newton]"

Run: python examples/07_tdmpc2_franka.py
"""

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from _common import describe_settings, figure_episode, report, setting, setup

import xwm

IMG_SIZE = setting("IMG_SIZE", 64)
EPISODE_LEN = setting("EPISODE_LEN", 12)
SEED_EPISODES = setting("SEED_EPISODES", 40)
ITERATIONS = setting("ITERATIONS", 25)
EPISODES_PER_ITER = setting("EPISODES_PER_ITER", 4)
STEPS_PER_ITER = setting("STEPS_PER_ITER", 40)
BATCH = setting("BATCH", 32)
HORIZON = setting("HORIZON", 3)
LATENT_DIM = setting("LATENT_DIM", 128)
HIDDEN_DIM = setting("HIDDEN_DIM", 256)
PLAN_SAMPLES = setting("PLAN_SAMPLES", 256)
PLAN_ITERS = setting("PLAN_ITERS", 4)
EVAL_EPISODES = setting("EVAL_EPISODES", 16)
EXPLORE_STD = setting("EXPLORE_STD", 0.3)


def rollout_policy(env, act, *, seed, length, rng=None):
    """Run one episode, choosing actions with ``act(state) -> action``."""
    env.reset(seed=seed)
    observations = [env.state_observation()]
    actions, rewards = [], []
    for _ in range(length):
        action = np.asarray(act(observations[-1]), dtype=np.float32)
        if rng is not None:  # exploration noise during collection
            action = np.clip(action + rng.normal(0.0, EXPLORE_STD, action.shape), -1.0, 1.0)
        env.step(action.astype(np.float32))
        observations.append(env.state_observation())
        actions.append(action)
        rewards.append(env.reward(action))
    return (
        np.stack(observations).astype(np.float32),
        np.stack(actions).astype(np.float32),
        np.asarray(rewards, np.float32),
        env.goal_distance(),
    )


# These two are jitted once, at module level, with the model passed in as an
# argument rather than closed over. That distinction is not stylistic: a
# filter_jit wrapper built inside the training loop closes over the weights,
# which makes them compile-time constants baked into that executable, so every
# iteration leaves another full copy of the model resident on the device. It
# also recompiles the planner every iteration. See docs/findings.md.
@eqx.filter_jit
def _planned_action(model, key, obs):
    """One MPPI-planned action for ``obs``."""
    search, cost = xwm.families.tdmpc2.planner(
        model, horizon=HORIZON, n_samples=PLAN_SAMPLES, n_iters=PLAN_ITERS
    )
    return search.plan(key, model.dynamics_fn(), model.encode(obs), cost).actions[0]


@eqx.filter_jit
def _prior_action(model, obs):
    """One action from the policy prior alone, with no search."""
    return model.policy.eval_mode().act(model.encode(obs))


def make_actor(model, *, key, plan: bool):
    """An ``act`` closure: plan with the learned model, or use the policy prior."""
    if not plan:
        return lambda obs: _prior_action(model, jnp.asarray(obs))
    return lambda obs: _planned_action(model, key, jnp.asarray(obs))


def evaluate(env, model, *, key, episodes, plan=True):
    """Mean true tool distance to the goal, plus the mean reward collected."""
    act = make_actor(model, key=key, plan=plan)
    distances, returns = [], []
    for i in range(episodes):
        _, _, rewards, distance = rollout_policy(env, act, seed=10_000 + i, length=EPISODE_LEN)
        distances.append(distance)
        returns.append(float(np.sum(rewards)))
    return float(np.mean(distances)), float(np.mean(returns))


def baselines(env, *, episodes):
    """No-op and random policies, on the same evaluation seeds."""
    results = {}
    for name, act in (
        ("no-op", lambda obs: np.zeros(env.action_dim, np.float32)),
        ("random", None),
    ):
        distances = []
        for i in range(episodes):
            if act is None:
                rng = np.random.default_rng(500 + i)
                actions = xwm.envs.smooth_actions(
                    rng, 1, EPISODE_LEN, env.action_dim, smoothness=0.5
                )[0]
                env.rollout(actions, seed=10_000 + i)
            else:
                rollout_policy(env, act, seed=10_000 + i, length=EPISODE_LEN)
            distances.append(env.goal_distance())
        results[name] = float(np.mean(distances))
    return results


def main():
    out = setup("07_tdmpc2_franka", n_series=3)
    key = jr.PRNGKey(0)
    k_model, k_collect, k_train, k_eval = jr.split(key, 4)
    xwm.set_seed(0)

    describe_settings(
        {
            "episode": EPISODE_LEN,
            "seed_episodes": SEED_EPISODES,
            "iterations": ITERATIONS,
            "steps/iter": STEPS_PER_ITER,
            "horizon": HORIZON,
            "latent": LATENT_DIM,
            "plan": f"{PLAN_SAMPLES}x{PLAN_ITERS}",
        }
    )
    env = xwm.envs.FrankaEnv(xwm.envs.FrankaConfig(image_size=IMG_SIZE))
    print(
        f"Franka FR3 | {env.action_dim} joints | state {env.state_dim} | "
        f"solver '{env.solver_name}' | goal {env.goal.round(3)}"
    )

    model = xwm.families.tdmpc2.tdmpc2(
        action_dim=env.action_dim,
        observation="state",
        state_dim=env.state_dim,
        latent_dim=LATENT_DIM,
        hidden_dim=HIDDEN_DIM,
        horizon=HORIZON,
        key=k_model,
    )
    print(f"  {model.n_params:,} params | discount {model.discount}")

    buffer = xwm.training.ReplayBuffer(
        capacity=200_000,
        observation_shape=(env.state_dim,),
        action_shape=(env.action_dim,),
        seed=0,
    )
    trainer = xwm.training.Trainer(
        model,
        xwm.training.adamw(3e-4, weight_decay=0.0),
        ema_momentum=0.99,
    )
    state = trainer.init()

    # Seed the buffer with random interaction: the planner is useless before the
    # model has seen anything, and bootstrapping off its own noise is worse.
    print(f"\nseeding {SEED_EPISODES} random episodes")
    for i in range(SEED_EPISODES):
        rng = np.random.default_rng(i)
        actions = xwm.envs.smooth_actions(rng, 1, EPISODE_LEN, env.action_dim, smoothness=0.5)[0]
        result = env.rollout(actions, seed=i)
        buffer.add_episode(result["state"], actions, result["reward"])
    print(f"  buffer: {len(buffer)} steps / {buffer.episodes} episodes")

    base = baselines(env, episodes=EVAL_EPISODES)
    start_distance, start_return = evaluate(
        env, state.model, key=k_eval, episodes=EVAL_EPISODES, plan=False
    )
    print(f"\nbaselines: no-op {base['no-op']:.3f} m  random {base['random']:.3f} m")
    print(f"untrained policy: {start_distance:.3f} m  (return {start_return:.2f})")

    history, curve = [], []
    print(f"\ntraining {ITERATIONS} iterations")
    for iteration in range(ITERATIONS):
        batches = (
            buffer.sample(BATCH, HORIZON) for _ in range(STEPS_PER_ITER)
        )
        state, rows = trainer.fit(
            batches,
            steps=STEPS_PER_ITER,
            key=jr.fold_in(k_train, iteration),
            state=state,
            log_every=STEPS_PER_ITER,
        )
        history.extend(rows)

        # Collect with the current model, planning through it.
        act = make_actor(state.model, key=jr.fold_in(k_collect, iteration), plan=True)
        for episode in range(EPISODES_PER_ITER):
            rng = np.random.default_rng(1000 * iteration + episode)
            observations, actions, rewards, _ = rollout_policy(
                env,
                act,
                seed=int(rng.integers(0, 2**31 - 1)),
                length=EPISODE_LEN,
                rng=rng,
            )
            buffer.add_episode(observations, actions, rewards)

        if iteration % 5 == 0 or iteration == ITERATIONS - 1:
            distance, ret = evaluate(
                env, state.model, key=k_eval, episodes=max(4, EVAL_EPISODES // 2)
            )
            curve.append({"iteration": iteration, "distance": distance, "return": ret})
            row = rows[-1]
            print(
                f"  iter {iteration:>3}  consistency {row['loss_consistency']:.4f}  "
                f"reward {row['loss_reward']:.3f}  value {row['loss_value']:.3f}  "
                f"| eval {distance:.3f} m  return {ret:+.2f}",
                flush=True,
            )

    final_distance, final_return = evaluate(
        env, state.model, key=k_eval, episodes=EVAL_EPISODES
    )
    policy_distance, _ = evaluate(
        env, state.model, key=k_eval, episodes=EVAL_EPISODES, plan=False
    )

    print("\ntrue tool distance to the goal (metres, lower is better):")
    rows = [
        ["no-op", base["no-op"]],
        ["random", base["random"]],
        ["policy prior (no planning)", policy_distance],
        ["TD-MPC2 + MPPI", final_distance],
    ]
    for name, value in rows:
        gap = 100.0 * (1.0 - value / base["no-op"])
        print(f"  {name:<28}{value:>8.3f}{gap:>+8.0f}% vs no-op")

    # -- artifacts ----------------------------------------------------------
    print("\nartifacts:")
    ax = xwm.plots.plot_history(
        history,
        keys=["loss_consistency", "loss_reward", "loss_value"],
        logy=True,
        title="TD-MPC2 losses",
    )
    report(xwm.plots.save_figure(ax.figure, out / "training_curve.png"))

    if len(curve) > 1:
        ax = xwm.plots.plot_horizon(
            [row["iteration"] for row in curve],
            {"planner": [row["distance"] for row in curve],
             "no-op baseline": [base["no-op"]] * len(curve)},
            title="Reach error while learning",
            ylabel="true tool distance (m)",
        )
        ax.set_xlabel("iteration")
        report(xwm.plots.save_figure(ax.figure, out / "learning_curve.png"))

    ax = xwm.plots.plot_bars(
        [name for name, _ in rows],
        [value for _, value in rows],
        ylabel="mean tool distance to goal (m)",
        title=f"Franka reach, {EVAL_EPISODES} episodes of {EPISODE_LEN} steps",
    )
    report(xwm.plots.save_figure(ax.figure, out / "policy_comparison.png"))

    act = make_actor(state.model, key=k_eval, plan=True)
    figure_episode(
        env,
        out,
        policy=lambda e: act(e.state_observation()),
        steps=EPISODE_LEN,
        seed=10_000,
        label="TD-MPC2 planning on the Franka reach task",
    )

    report(
        xwm.plots.save_table(
            out / "results",
            ["policy", "mean distance (m)", "% of gap closed"],
            [[name, value, 100.0 * (1.0 - value / base["no-op"])] for name, value in rows],
            caption=(
                f"TD-MPC2 on the Franka reach task after {ITERATIONS} iterations. "
                "Distances are ground truth; the agent only ever sees reward."
            ),
            label="tab:tdmpc2-franka",
        )
    )
    report(
        xwm.plots.save_metrics(
            out / "metrics",
            {
                "iterations": ITERATIONS,
                "episodes_collected": buffer.episodes,
                "buffer_steps": len(buffer),
                "horizon": HORIZON,
                "final_distance": final_distance,
                "final_return": final_return,
                "policy_only_distance": policy_distance,
                "noop_distance": base["no-op"],
                "random_distance": base["random"],
                "loss_consistency": history[-1]["loss_consistency"],
                "loss_reward": history[-1]["loss_reward"],
                "loss_value": history[-1]["loss_value"],
            },
        )
    )


if __name__ == "__main__":
    main()

"""MuZero on a Franka reach task: learn a model that agrees with its own search.

The loop that makes MuZero different: act by running MCTS through the learned
model, store what the *search* concluded rather than what the policy did, then
train the network to agree with it. Search is a policy-improvement operator, so
the target is always better than the network that produced it -- that is the
engine.

The latent here is anchored to nothing. There is no reconstruction and no
embedding target; the representation is whatever makes reward, value and policy
predictable. It never even re-encodes an observation mid-unroll.

MuZero needs discrete actions, and a 7-DoF arm has continuous ones, so actions
are the 15 axis-aligned moves from :func:`xwm.envs.discrete_action_table` --
push one joint either way, or hold still. (For genuinely continuous control use
TD-MPC2, whose planner needs no discretisation: `examples/07_tdmpc2_franka.py`.)

Needs the Newton extra::

    pip install "xwm[newton]"

Run: python examples/08_muzero_franka.py
"""

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from _common import describe_settings, figure_episode, report, setting, setup

import xwm

IMG_SIZE = setting("IMG_SIZE", 64)
EPISODE_LEN = setting("EPISODE_LEN", 12)
SEED_EPISODES = setting("SEED_EPISODES", 30)
ITERATIONS = setting("ITERATIONS", 20)
EPISODES_PER_ITER = setting("EPISODES_PER_ITER", 4)
STEPS_PER_ITER = setting("STEPS_PER_ITER", 40)
BATCH = setting("BATCH", 32)
HORIZON = setting("HORIZON", 3)
LATENT_DIM = setting("LATENT_DIM", 128)
HIDDEN_DIM = setting("HIDDEN_DIM", 256)
SIMULATIONS = setting("SIMULATIONS", 32)
N_STEP = setting("N_STEP", 5)
EVAL_EPISODES = setting("EVAL_EPISODES", 12)
TEMPERATURE = setting("TEMPERATURE", 1.0)


def search_fn(model, mcts, *, key):
    """A jitted MCTS call over the model's own recurrent/predict functions."""
    recurrent, predict = model.search_fns()

    @eqx.filter_jit
    def run(k, latent, noise: bool):
        return mcts.search(k, latent, recurrent, predict, add_noise=noise)

    encode = eqx.filter_jit(model.represent)
    return encode, run


def self_play(env, model, mcts, actions_table, *, key, seed, explore=True):
    """One episode driven by tree search, recording the search's own targets."""
    encode, search = search_fn(model, mcts, key=key)
    env.reset(seed=seed)
    observations = [env.state_observation()]
    action_indices, rewards, root_values, policies = [], [], [], []

    for step in range(EPISODE_LEN):
        latent = encode(jnp.asarray(observations[-1]))
        result = search(jr.fold_in(key, step), latent, explore)
        policy = np.asarray(result.policy, np.float32)
        root_values.append(float(result.value))
        policies.append(policy)

        if explore:
            # Sample from the visit distribution (temperature-scaled) so the
            # buffer sees more than the search's single favourite action.
            weights = policy ** (1.0 / max(TEMPERATURE, 1e-6))
            total = weights.sum()
            index = int(
                np.random.default_rng(seed * 1000 + step).choice(
                    len(policy), p=weights / total if total > 0 else None
                )
            )
        else:
            index = int(result.action)

        env.step(actions_table[index])
        observations.append(env.state_observation())
        action_indices.append(index)
        rewards.append(env.reward(actions_table[index]))

    # The final state gets a root value too, so the n-step bootstrap has
    # something to read at the end of the episode.
    latent = encode(jnp.asarray(observations[-1]))
    final = search(jr.fold_in(key, EPISODE_LEN), latent, False)
    root_values.append(float(final.value))
    policies.append(np.asarray(final.policy, np.float32))

    value_targets = xwm.families.muzero.n_step_value_targets(
        np.asarray(rewards, np.float32),
        np.asarray(root_values, np.float32),
        discount=0.997,
        n_steps=N_STEP,
    )
    return {
        "observations": np.stack(observations).astype(np.float32),
        "actions": np.asarray(action_indices, np.float32),
        "rewards": np.asarray(rewards, np.float32),
        "value_target": value_targets,
        "policy_target": np.stack(policies),
        "distance": env.goal_distance(),
    }


def evaluate(env, model, mcts, actions_table, *, key, episodes):
    """Mean true tool distance, acting greedily on the search's most-visited action."""
    distances = []
    for i in range(episodes):
        episode = self_play(
            env, model, mcts, actions_table, key=key, seed=10_000 + i, explore=False
        )
        distances.append(episode["distance"])
    return float(np.mean(distances))


def main():
    out = setup("08_muzero_franka", n_series=3)
    key = jr.PRNGKey(0)
    k_model, k_play, k_train, k_eval = jr.split(key, 4)
    xwm.set_seed(0)

    env = xwm.envs.FrankaEnv(xwm.envs.FrankaConfig(image_size=IMG_SIZE))
    actions_table = xwm.envs.discrete_action_table(env.action_dim)
    n_actions = actions_table.shape[0]
    describe_settings(
        {
            "episode": EPISODE_LEN,
            "actions": n_actions,
            "simulations": SIMULATIONS,
            "iterations": ITERATIONS,
            "horizon": HORIZON,
            "n_step": N_STEP,
        }
    )
    print(
        f"Franka FR3 | {n_actions} discrete actions | state {env.state_dim} | "
        f"solver '{env.solver_name}' | goal {env.goal.round(3)}"
    )

    model = xwm.families.muzero.muzero(
        n_actions=n_actions,
        observation="state",
        state_dim=env.state_dim,
        latent_dim=LATENT_DIM,
        hidden_dim=HIDDEN_DIM,
        horizon=HORIZON,
        key=k_model,
    )
    mcts = xwm.planning.MCTS(
        n_actions, n_simulations=SIMULATIONS, discount=0.997, max_depth=HORIZON + 4
    )
    print(f"  {model.n_params:,} params")

    buffer = xwm.training.ReplayBuffer(
        capacity=200_000,
        observation_shape=(env.state_dim,),
        action_shape=(),
        extra={"value_target": (), "policy_target": (n_actions,)},
        seed=0,
    )
    trainer = xwm.training.Trainer(model, xwm.training.adamw(3e-4, weight_decay=0.0))
    state = trainer.init()

    def store(episode):
        buffer.add_episode(
            episode["observations"],
            episode["actions"],
            episode["rewards"],
            value_target=episode["value_target"],
            policy_target=episode["policy_target"],
        )

    # Seed with random discrete actions: search through an untrained model is
    # no better than noise, and cheaper to skip.
    print(f"\nseeding {SEED_EPISODES} random episodes")
    uniform = np.full((EPISODE_LEN + 1, n_actions), 1.0 / n_actions, np.float32)
    for i in range(SEED_EPISODES):
        rng = np.random.default_rng(i)
        indices = rng.integers(0, n_actions, EPISODE_LEN)
        env.reset(seed=i)
        observations = [env.state_observation()]
        rewards = []
        for index in indices:
            env.step(actions_table[index])
            observations.append(env.state_observation())
            rewards.append(env.reward(actions_table[index]))
        store(
            {
                "observations": np.stack(observations).astype(np.float32),
                "actions": indices.astype(np.float32),
                "rewards": np.asarray(rewards, np.float32),
                "value_target": xwm.families.muzero.n_step_value_targets(
                    np.asarray(rewards, np.float32),
                    np.zeros(EPISODE_LEN + 1, np.float32),
                    n_steps=N_STEP,
                ),
                "policy_target": uniform,
            }
        )
    print(f"  buffer: {len(buffer)} steps / {buffer.episodes} episodes")

    noop_distance = []
    for i in range(EVAL_EPISODES):
        env.reset(seed=10_000 + i)
        for _ in range(EPISODE_LEN):
            env.step(np.zeros(env.action_dim, np.float32))
        noop_distance.append(env.goal_distance())
    noop = float(np.mean(noop_distance))
    print(f"\nno-op baseline: {noop:.3f} m")

    history, curve = [], []
    print(f"\ntraining {ITERATIONS} iterations")
    for iteration in range(ITERATIONS):
        batches = (buffer.sample(BATCH, HORIZON) for _ in range(STEPS_PER_ITER))
        state, rows = trainer.fit(
            batches,
            steps=STEPS_PER_ITER,
            key=jr.fold_in(k_train, iteration),
            state=state,
            log_every=STEPS_PER_ITER,
        )
        history.extend(rows)

        for episode_index in range(EPISODES_PER_ITER):
            episode = self_play(
                env,
                state.model,
                mcts,
                actions_table,
                key=jr.fold_in(k_play, iteration * 100 + episode_index),
                seed=1000 * iteration + episode_index,
            )
            store(episode)

        if iteration % 5 == 0 or iteration == ITERATIONS - 1:
            distance = evaluate(
                env, state.model, mcts, actions_table,
                key=k_eval, episodes=max(3, EVAL_EPISODES // 3),
            )
            curve.append({"iteration": iteration, "distance": distance})
            row = rows[-1]
            print(
                f"  iter {iteration:>3}  reward {row['loss_reward']:.3f}  "
                f"value {row['loss_value']:.3f}  policy {row['loss_policy']:.3f}  "
                f"| eval {distance:.3f} m",
                flush=True,
            )

    final = evaluate(
        env, state.model, mcts, actions_table, key=k_eval, episodes=EVAL_EPISODES
    )
    rows = [["no-op", noop], ["MuZero + MCTS", final]]
    print("\ntrue tool distance to the goal (metres, lower is better):")
    for name, value in rows:
        print(f"  {name:<20}{value:>8.3f}{100.0 * (1.0 - value / noop):>+8.0f}% vs no-op")

    # -- artifacts ----------------------------------------------------------
    print("\nartifacts:")
    ax = xwm.plots.plot_history(
        history,
        keys=["loss_reward", "loss_value", "loss_policy"],
        logy=True,
        title="MuZero losses",
    )
    report(xwm.plots.save_figure(ax.figure, out / "training_curve.png"))

    if len(curve) > 1:
        ax = xwm.plots.plot_horizon(
            [row["iteration"] for row in curve],
            {"MuZero + MCTS": [row["distance"] for row in curve],
             "no-op baseline": [noop] * len(curve)},
            title="Reach error while learning",
            ylabel="true tool distance (m)",
        )
        ax.set_xlabel("iteration")
        report(xwm.plots.save_figure(ax.figure, out / "learning_curve.png"))

    ax = xwm.plots.plot_bars(
        [name for name, _ in rows],
        [value for _, value in rows],
        ylabel="mean tool distance to goal (m)",
        title=f"Franka reach, {n_actions} discrete actions, {SIMULATIONS} simulations",
    )
    report(xwm.plots.save_figure(ax.figure, out / "policy_comparison.png"))

    # What the search decided, on one evaluation episode.
    encode, search = search_fn(state.model, mcts, key=k_eval)
    visits: list[np.ndarray] = []

    def searched_action(e):
        """One MCTS decision, keeping the visit counts for the entropy metric.

        Called once per step, on the pass that drives the episode; the renderers
        that follow replay the recorded actions, so the visits are collected
        exactly once however many times the episode is rendered.
        """
        latent = encode(jnp.asarray(e.state_observation()))
        result = search(jr.fold_in(k_eval, len(visits)), latent, False)
        visits.append(np.asarray(result.visits))
        return actions_table[int(result.action)]

    figure_episode(
        env,
        out,
        policy=searched_action,
        steps=EPISODE_LEN,
        seed=10_000,
        label="MuZero MCTS on the Franka reach task",
    )

    report(
        xwm.plots.save_table(
            out / "results",
            ["policy", "mean distance (m)", "% of gap closed"],
            [[name, value, 100.0 * (1.0 - value / noop)] for name, value in rows],
            caption=(
                f"MuZero on the Franka reach task, {n_actions} discrete actions, "
                f"{SIMULATIONS} simulations per move, after {ITERATIONS} iterations."
            ),
            label="tab:muzero-franka",
        )
    )
    report(
        xwm.plots.save_metrics(
            out / "metrics",
            {
                "iterations": ITERATIONS,
                "n_actions": n_actions,
                "simulations": SIMULATIONS,
                "episodes_collected": buffer.episodes,
                "buffer_steps": len(buffer),
                "final_distance": final,
                "noop_distance": noop,
                "mean_search_entropy": float(
                    np.mean([-np.sum((v / v.sum()) * np.log(v / v.sum() + 1e-9)) for v in visits])
                ),
                "loss_reward": history[-1]["loss_reward"],
                "loss_value": history[-1]["loss_value"],
                "loss_policy": history[-1]["loss_policy"],
            },
        )
    )


if __name__ == "__main__":
    main()

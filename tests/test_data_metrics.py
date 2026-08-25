"""The synthetic world, batching, and representation diagnostics."""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import xwm


# -- world ------------------------------------------------------------------
def test_actions_move_the_agent(key):
    """If actions don't change the observations, nothing downstream is learnable."""
    world = xwm.data.SpriteWorld(32, n_distractors=0)
    right = jnp.tile(jnp.array([1.0, 0.0]), (10, 1))
    left = -right
    _, s_right = world.rollout(key, right)
    _, s_left = world.rollout(key, left)
    assert float(s_right.pos[-1, 0]) > float(s_right.pos[0, 0]) + 0.2
    assert float(s_left.pos[-1, 0]) < float(s_left.pos[0, 0]) - 0.2


def test_agent_stays_in_bounds(key):
    world = xwm.data.SpriteWorld(16, n_distractors=1)
    actions = jnp.tile(jnp.array([1.0, 1.0]), (60, 1))  # drive hard into a corner
    _, states = world.rollout(key, actions)
    assert float(jnp.min(states.pos)) >= 0.0
    assert float(jnp.max(states.pos)) <= 1.0


def test_render_shape_and_range(key):
    world = xwm.data.SpriteWorld(24, n_distractors=2)
    frame = world.render(world.reset(key))
    assert frame.shape == (3, 24, 24) == world.observation_shape
    assert 0.0 <= float(frame.min()) and float(frame.max()) <= 1.0
    # The agent lives in channel 0 and must actually be visible.
    assert float(frame[0].max()) > 0.5


def test_distractors_are_unpredictable_from_actions(key):
    """Two rollouts with the same actions but different env keys must differ.

    This is the property that makes the dataset worth using: some of the pixels
    are genuinely not a function of the action sequence.
    """
    world = xwm.data.SpriteWorld(16, n_distractors=3)
    actions = jnp.zeros((8, 2))
    _, a = world.rollout(jr.PRNGKey(0), actions)
    _, b = world.rollout(jr.PRNGKey(1), actions)
    assert not jnp.allclose(a.distractors, b.distractors)


def test_no_distractors_means_static_background(key):
    world = xwm.data.SpriteWorld(16, n_distractors=0)
    frames, _ = world.rollout(key, jnp.zeros((4, 2)))
    assert jnp.allclose(frames[0], frames[-1], atol=1e-5)


def test_rollout_frame_count_is_one_more_than_actions(key):
    world = xwm.data.SpriteWorld(16, n_distractors=1)
    frames, states = world.rollout(key, jnp.zeros((5, 2)))
    assert frames.shape[0] == 6
    assert states.pos.shape == (6, 2)


def test_sprite_sequences_action_alignment(key):
    ds = xwm.data.sprite_sequences(key, 4, 7)
    assert ds["video"].shape[:2] == (4, 7)
    assert ds["action"].shape == (4, 6, 2)  # T - 1 actions for T frames
    assert ds["position"].shape == (4, 7, 2)


def test_random_actions_are_smooth(key):
    smooth = xwm.data.random_actions(key, 4, 40, 2, smoothness=0.95)
    rough = xwm.data.random_actions(key, 4, 40, 2, smoothness=0.0)
    d_smooth = float(jnp.mean(jnp.abs(jnp.diff(smooth, axis=1))))
    d_rough = float(jnp.mean(jnp.abs(jnp.diff(rough, axis=1))))
    assert d_smooth < d_rough / 3
    assert float(jnp.max(jnp.abs(smooth))) <= 1.0


def test_world_is_jittable_and_vmappable(key):
    world = xwm.data.SpriteWorld(16, n_distractors=1)
    actions = jnp.zeros((3, 2))
    batched = jax.jit(jax.vmap(world.observe, in_axes=(0, None)))
    assert batched(jr.split(key, 4), actions).shape == (4, 4, 3, 16, 16)


# -- occlusion --------------------------------------------------------------
def test_occluders_hide_the_agent_without_changing_its_dynamics(key):
    """The bars must be an observation change only, never a dynamics change."""
    plain = xwm.data.SpriteWorld(32, n_distractors=0)
    occluded = xwm.data.SpriteWorld(32, n_distractors=0, n_occluders=2)
    # Gently, at ~0.06 per step: at full action the agent covers 0.3 per step
    # and jumps clean over a 0.26 bar without ever being seen behind it.
    actions = jnp.tile(jnp.array([0.2, 0.0]), (30, 1))
    _, states_plain = plain.rollout(key, actions)
    frames, states_occluded = occluded.rollout(key, actions)
    assert jnp.allclose(states_plain.pos, states_occluded.pos)
    # Somewhere along the traverse the agent is behind a bar and nearly gone.
    visibility = jnp.max(frames[:, 0], axis=(1, 2))
    assert float(jnp.min(visibility)) < 0.25
    assert float(jnp.max(visibility)) > 0.5


def test_an_agent_at_a_bar_centre_is_hidden(key):
    """Directly, without relying on a trajectory to line up: 0.84 -> 0.18."""
    occluded = xwm.data.SpriteWorld(32, n_distractors=0, n_occluders=2)
    plain = xwm.data.SpriteWorld(32, n_distractors=0)
    behind = occluded.reset(key)._replace(pos=jnp.array([1.0 / 3.0, 0.5]))
    assert float(jnp.max(occluded.render(behind)[0])) < 0.25
    assert float(jnp.max(plain.render(behind)[0])) > 0.7


def test_no_occluders_is_the_default_and_changes_nothing(key):
    world = xwm.data.SpriteWorld(16, n_distractors=1)
    assert world.n_occluders == 0
    frames, _ = world.rollout(key, jnp.zeros((3, 2)))
    explicit = xwm.data.SpriteWorld(16, n_distractors=1, n_occluders=0)
    assert jnp.allclose(frames, explicit.rollout(key, jnp.zeros((3, 2)))[0])


# -- pushing ----------------------------------------------------------------
def test_the_puck_moves_only_on_contact(key):
    """The property the whole world exists for: a hinge at the contact boundary."""
    world = xwm.data.PushWorld()
    state = world.reset(key)
    outward = state.pusher - state.puck
    outward = outward / jnp.linalg.norm(outward)
    retreat = world.step(state, outward)
    approach = world.step(state, -outward)
    assert float(jnp.linalg.norm(retreat.puck - state.puck)) == 0.0
    assert float(jnp.linalg.norm(approach.puck - state.puck)) > 0.05


def test_push_dynamics_are_not_linear_in_the_action(key):
    """f(a) + f(-a) != 2 f(0) once contact is involved."""
    world = xwm.data.PushWorld()
    state = world.reset(key)
    towards = (state.puck - state.pusher) / jnp.linalg.norm(state.puck - state.pusher)
    forward = world.step(state, towards).puck
    backward = world.step(state, -towards).puck
    still = world.step(state, jnp.zeros((2,))).puck
    assert not jnp.allclose(forward + backward, 2.0 * still, atol=1e-3)


def test_push_reward_is_bounded_and_rises_as_the_puck_nears_the_goal(key):
    world = xwm.data.PushWorld()
    state = world.reset(key)
    near = state._replace(puck=state.goal)
    assert float(world.reward(near)) > float(world.reward(state))
    assert -1.0 <= float(world.reward(state)) <= 0.0
    assert float(world.success(near)) == 1.0
    assert float(world.success(state)) == 0.0


def test_puck_and_pusher_stay_in_bounds(key):
    world = xwm.data.PushWorld(16)
    _, states = world.rollout(key, jnp.tile(jnp.array([1.0, 1.0]), (40, 1)))
    assert float(jnp.min(states.puck)) >= 0.0 and float(jnp.max(states.puck)) <= 1.0
    assert float(jnp.min(states.pusher)) >= 0.0 and float(jnp.max(states.pusher)) <= 1.0


def test_push_sequences_contract(key):
    ds = xwm.data.push_sequences(key, 3, 9)
    assert ds["video"].shape == (3, 9, 3, 32, 32)
    assert ds["action"].shape == (3, 8, 2)  # T - 1 actions for T frames
    assert ds["state"].shape == (3, 9, 8)
    # Reward aligns with the actions, not the frames: it is the reward of the
    # state each action led to, which is what ReplayBuffer.add_episode wants.
    assert ds["reward"].shape == (3, 8)
    assert ds["success"].shape == (3, 8)


def test_push_search_beats_a_single_random_sequence(key):
    """A planner must have something to find, or the task measures nothing."""
    world = xwm.data.PushWorld()
    actions = xwm.data.random_actions(key, 64, 48, 2, smoothness=0.7)

    def closest(a):
        _, states = world.rollout(jr.PRNGKey(7), a)
        return jnp.min(jax.vmap(world.goal_distance)(states))

    reached = jax.jit(jax.vmap(closest))(actions)
    assert float(jnp.min(reached)) < world.goal_radius   # search solves it
    assert float(jnp.median(reached)) > world.goal_radius  # random does not


# -- maze -------------------------------------------------------------------
def test_the_maze_agent_cannot_walk_through_walls(key):
    world = xwm.data.MazeWorld(32)
    # Drive hard in every direction for a long time from many starts.
    for seed in range(8):
        for direction in ([1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]):
            actions = jnp.tile(jnp.array(direction), (40, 1))
            _, states = world.rollout(jr.PRNGKey(seed), actions)
            assert not bool(jnp.any(jax.vmap(world._blocked)(states.pos)))


def test_the_maze_agent_slides_along_a_wall(key):
    """A blocked axis must not freeze the other one, or corridors are traps."""
    world = xwm.data.MazeWorld(32)
    state = world.reset(jr.PRNGKey(0))
    diagonal = jnp.tile(jnp.array([1.0, 1.0]), (12, 1))
    _, states = world.rollout(jr.PRNGKey(0), diagonal)
    moved = jnp.abs(states.pos[-1] - states.pos[0])
    assert float(jnp.max(moved)) > 0.05
    del state


def test_maze_reward_is_sparse_and_reachable(key):
    ds = xwm.data.maze_sequences(key, 128, 64)
    reward = ds["reward"]
    assert set(np.unique(np.asarray(reward)).tolist()) <= {0.0, 1.0}
    touched = float(jnp.mean(jnp.max(reward, axis=1)))
    # Measured at ~0.33 for these defaults; assert the band, not the number.
    assert 0.15 < touched < 0.6


def test_maze_sequences_contract(key):
    ds = xwm.data.maze_sequences(key, 3, 9)
    assert ds["video"].shape == (3, 9, 3, 32, 32)
    assert ds["action"].shape == (3, 8, 2)
    assert ds["state"].shape == (3, 9, 6)
    assert ds["reward"].shape == (3, 8)
    assert ds["distance"].shape == (3, 9)


def test_maze_layout_is_validated():
    with pytest.raises(ValueError, match="border"):
        xwm.data.MazeWorld(16, layout=("....", "....", "....", "...."))
    with pytest.raises(ValueError, match="square"):
        xwm.data.MazeWorld(16, layout=("####", "#..#", "###"))


def test_new_worlds_are_jittable_and_vmappable(key):
    actions = jnp.zeros((3, 2))
    for world in (xwm.data.PushWorld(16), xwm.data.MazeWorld(16)):
        batched = jax.jit(jax.vmap(world.observe, in_axes=(0, None)))
        assert batched(jr.split(key, 4), actions).shape == (4, 4, 3, 16, 16)


# -- the reward-driven path -------------------------------------------------
def test_push_data_trains_tdmpc2_through_the_replay_buffer(key):
    """The whole point of PushWorld: a CPU reward task for the value families.

    Before this world existed, TD-MPC2 and MuZero had exactly one task to train
    on and it needed the `newton` extra.
    """
    world = xwm.data.PushWorld(16)
    data = xwm.data.push_sequences(key, 8, 9, world=world)

    buffer = xwm.training.ReplayBuffer(
        8 * 8, (world.state_dim,), (world.action_dim,), seed=0
    )
    for i in range(data["state"].shape[0]):
        buffer.add_episode(
            np.asarray(data["state"][i]),
            np.asarray(data["action"][i]),
            np.asarray(data["reward"][i]),
        )
    assert buffer.episodes == 8

    agent = xwm.families.tdmpc2.tdmpc2(
        key=key,
        action_dim=world.action_dim,
        observation="state",
        state_dim=world.state_dim,
    )
    trainer = xwm.training.Trainer(agent, xwm.training.adamw(1e-3))
    batches = (buffer.sample(4, 3) for _ in range(6))
    state, history = trainer.fit(batches, steps=5, key=key, log_every=5)
    assert int(state.step) == 5
    assert bool(jnp.isfinite(jnp.asarray(history[-1]["loss"])))


# -- batching ---------------------------------------------------------------
def test_iter_batches_covers_the_dataset(key):
    data = {"x": jnp.arange(20).reshape(20, 1)}
    seen = np.concatenate([np.asarray(b["x"]).reshape(-1) for b in
                           xwm.data.iter_batches(data, 5, key=key, epochs=1)])
    assert sorted(seen.tolist()) == list(range(20))


def test_iter_batches_repeats_forever(key):
    data = {"x": jnp.arange(8).reshape(8, 1)}
    it = xwm.data.iter_batches(data, 4, key=key, epochs=None)
    assert len([next(it) for _ in range(20)]) == 20


def test_iter_batches_shuffles(key):
    data = {"x": jnp.arange(32).reshape(32, 1)}
    ordered = next(xwm.data.iter_batches(data, 8, shuffle=False, epochs=1))
    shuffled = next(xwm.data.iter_batches(data, 8, key=key, epochs=1))
    assert jnp.array_equal(ordered["x"].reshape(-1), jnp.arange(8))
    assert not jnp.array_equal(shuffled["x"], ordered["x"])


def test_iter_batches_keeps_fields_aligned(key):
    data = {"x": jnp.arange(12).reshape(12, 1), "y": jnp.arange(12).reshape(12, 1) * 10}
    for b in xwm.data.iter_batches(data, 4, key=key, epochs=1):
        assert jnp.array_equal(b["y"], b["x"] * 10)


def test_iter_batches_validates(key):
    with pytest.raises(ValueError, match="disagree on length"):
        next(xwm.data.iter_batches({"a": jnp.zeros((4, 1)), "b": jnp.zeros((5, 1))}, 2, key=key))
    with pytest.raises(ValueError, match="exceeds dataset size"):
        next(xwm.data.iter_batches({"a": jnp.zeros((4, 1))}, 8, key=key))
    with pytest.raises(ValueError, match="needs a key"):
        next(xwm.data.iter_batches({"a": jnp.zeros((4, 1))}, 2, shuffle=True))


def test_clip_windows():
    video = jnp.arange(6).reshape(6, 1)
    windows = xwm.data.clip_windows(video, 3, stride=1)
    assert windows.shape == (4, 3, 1)
    assert jnp.array_equal(windows[0].reshape(-1), jnp.arange(3))
    assert xwm.data.clip_windows(video, 2, stride=2).shape == (3, 2, 1)
    with pytest.raises(ValueError, match="exceeds sequence length"):
        xwm.data.clip_windows(video, 9)


# -- metrics ----------------------------------------------------------------
def test_rankme_tracks_effective_rank(key):
    full = jr.normal(key, (512, 64))
    low = jr.normal(key, (512, 4)) @ jr.normal(jr.PRNGKey(1), (4, 64))
    assert float(xwm.metrics.rankme(full)) > 55
    assert 3.0 < float(xwm.metrics.rankme(low)) < 5.5
    assert float(xwm.metrics.effective_rank_ratio(full)) > 0.85


def test_feature_std_and_cosine_catch_constant_collapse(key):
    """A constant embedding is high-rank after centring; these must still catch it."""
    collapsed = jnp.tile(jr.normal(key, (1, 32)), (256, 1)) + 1e-4 * jr.normal(key, (256, 32))
    assert float(xwm.metrics.feature_std(collapsed)) < 1e-3
    assert float(xwm.metrics.mean_cosine_similarity(collapsed)) > 0.99
    healthy = jr.normal(key, (256, 32))
    assert float(xwm.metrics.feature_std(healthy)) > 0.9
    assert abs(float(xwm.metrics.mean_cosine_similarity(healthy))) < 0.1


def test_collapse_report_keys(key):
    report = xwm.metrics.collapse_report(jr.normal(key, (64, 16)))
    assert set(report) == {"rankme", "rank_ratio", "feature_std", "mean_cosine"}


def test_metrics_accept_token_sequences(key):
    z = jr.normal(key, (8, 16, 32))  # (B, N, D)
    assert jnp.ndim(xwm.metrics.rankme(z)) == 0
    assert jnp.ndim(xwm.metrics.feature_std(z)) == 0


def test_ridge_probe_recovers_a_linear_map(key):
    z = jr.normal(key, (512, 32))
    y = z @ jr.normal(jr.PRNGKey(1), (32, 3))
    good = xwm.metrics.ridge_probe(z[:400], y[:400], z[400:], y[400:])
    assert float(good["r2"]) > 0.99
    noise = xwm.metrics.ridge_probe(
        z[:400], jr.normal(key, (400, 3)), z[400:], jr.normal(jr.PRNGKey(2), (112, 3))
    )
    assert float(noise["r2"]) < 0.2


def test_ridge_probe_is_worse_on_a_collapsed_representation(key):
    z = jr.normal(key, (400, 32))
    y = z @ jr.normal(jr.PRNGKey(1), (32, 2))
    collapsed = jnp.zeros_like(z) + jnp.mean(z, axis=0, keepdims=True)
    good = xwm.metrics.ridge_probe(z[:300], y[:300], z[300:], y[300:])
    bad = xwm.metrics.ridge_probe(collapsed[:300], y[:300], collapsed[300:], y[300:])
    assert float(bad["r2"]) < float(good["r2"])


def test_knn_probe_regression_and_classification(key):
    z = jr.normal(key, (400, 8))
    y = jnp.sum(z, axis=-1, keepdims=True)
    reg = xwm.metrics.knn_probe(z[:300], y[:300], z[300:], y[300:], k=5)
    assert float(reg["r2"]) > 0.3
    labels = (y[:, 0] > 0).astype(jnp.int32)
    cls = xwm.metrics.knn_probe(
        z[:300], labels[:300], z[300:], labels[300:], k=5, classification=True, n_classes=2
    )
    assert float(cls["accuracy"]) > 0.6


def test_knn_classification_requires_n_classes(key):
    z = jr.normal(key, (10, 4))
    with pytest.raises(ValueError, match="n_classes"):
        xwm.metrics.knn_probe(z, jnp.zeros(10, int), z, jnp.zeros(10, int), classification=True)


def test_feature_std_and_mean_cosine_measure_independent_failures():
    """The two collapse metrics must be able to disagree.

    They are documented as independent, and a real run proved it: a Push-T
    encoder reached mean_cosine 0.73 (directions spreading) while feature_std sat
    at 0.0076 (magnitude dead). If either number were a proxy for the other,
    reporting one would be enough, and a reader would take that run for progress.
    """
    key = jr.PRNGKey(0)

    # Directions spread, magnitude dead: what the small-budget run looked like.
    tiny = 1e-3 * jr.normal(key, (64, 32))
    assert float(xwm.metrics.feature_std(tiny)) < 1e-2
    assert abs(float(xwm.metrics.mean_cosine_similarity(tiny))) < 0.2

    # Directions collapsed, magnitude healthy: one direction, varied lengths.
    direction = jr.normal(jr.fold_in(key, 1), (32,))
    scales = jnp.linspace(0.5, 5.0, 64)[:, None]
    rays = scales * direction
    assert float(xwm.metrics.feature_std(rays)) > 0.5
    assert float(xwm.metrics.mean_cosine_similarity(rays)) > 0.99

    # And the report carries both, so neither failure can hide behind the other.
    report = xwm.metrics.collapse_report(tiny)
    assert {"rankme", "rank_ratio", "feature_std", "mean_cosine"} <= set(report)

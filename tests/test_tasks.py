"""Tasks: the bindings that keep training and evaluation talking about the same thing.

Most of what a task does is bookkeeping -- resize this, group those, split here
-- and every one of those has a silent failure mode. A frameskip that drops the
last action, a split that leaks, an environment that renders at a different
resolution than the recording: none of them raise, and all of them produce a
number that looks fine.
"""

import itertools

import jax.random as jr
import numpy as np
import pytest

import xwm
from xwm.datasets.spec import check_episode
from xwm.envs.jax_world import JaxWorldEnv
from xwm.envs.protocol import Env
from xwm.tasks.data import (
    denormalize_action,
    frameskip_episode,
    normalize_action,
    split_episodes,
    unblock,
)


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path_factory, monkeypatch):
    monkeypatch.setenv("XWM_DATA_HOME", str(tmp_path_factory.mktemp("cache")))


# -- the Env protocol -------------------------------------------------------
@pytest.mark.parametrize(
    "world",
    [
        pytest.param(xwm.data.PushWorld(16), id="push"),
        pytest.param(xwm.data.MazeWorld(16), id="maze"),
        pytest.param(xwm.data.SpriteWorld(16), id="sprite"),
    ],
)
def test_jax_worlds_satisfy_the_env_protocol(world):
    env = JaxWorldEnv(world)
    assert isinstance(env, Env)
    frame = env.reset(seed=1)
    assert set(frame) <= {"image", "state", "position"}
    assert frame["image"].dtype == np.uint8
    assert frame["image"].shape == (3, *env.native_size)


@pytest.mark.parametrize(
    "world", [xwm.data.PushWorld(16), xwm.data.MazeWorld(16), xwm.data.SpriteWorld(16)]
)
def test_reset_to_state_round_trips_exactly(world):
    """The invariant the whole dataset-driven protocol rests on."""
    env = JaxWorldEnv(world)
    env.reset(seed=2)
    saved = env.state()
    for _ in range(5):
        env.step(np.full(env.action_dim, 0.4, np.float32))
    assert not np.allclose(env.state(), saved)  # the state really did move
    env.reset(state=saved)
    assert np.allclose(env.state(), saved, atol=1e-6)


def test_reset_is_reproducible_and_seed_dependent():
    env = JaxWorldEnv(xwm.data.PushWorld(16))
    env.reset(seed=7)
    first = env.state()
    env.reset(seed=7)
    assert np.allclose(env.state(), first)
    env.reset(seed=8)
    assert not np.allclose(env.state(), first)


def test_a_wrong_width_state_is_refused():
    env = JaxWorldEnv(xwm.data.PushWorld(16))
    with pytest.raises(ValueError, match="width 6"):
        env.reset(state=np.zeros(3, np.float32))


def test_a_goalless_world_reports_no_task_state():
    """SpriteWorld has no goal, so it can be stepped but cannot back a task."""
    assert JaxWorldEnv(xwm.data.SpriteWorld(16)).has_task_state is False
    assert JaxWorldEnv(xwm.data.PushWorld(16)).has_task_state is True


# -- frameskip --------------------------------------------------------------
@pytest.mark.parametrize(("n_frames", "k"), [(11, 5), (10, 5), (13, 5), (16, 1), (7, 3), (25, 4)])
def test_frameskip_groups_actions_without_losing_any(n_frames, k):
    """Every retained action must survive, in order, inside its block."""
    episode = {
        "state": np.arange(n_frames * 3, dtype=np.float32).reshape(n_frames, 3),
        "action": np.arange((n_frames - 1) * 2, dtype=np.float32).reshape(n_frames - 1, 2),
        "reward": np.ones(n_frames - 1, np.float32),
    }
    out = frameskip_episode(episode, k)
    check_episode(out)
    kept, transitions = out["state"].shape[0], out["action"].shape[0]

    assert transitions == kept - 1
    assert out["action"].shape[1] == 2 * k
    assert np.allclose(out["state"], episode["state"][: (kept - 1) * k + 1 : k])
    regrouped = np.concatenate([unblock(out["action"][i], 2) for i in range(transitions)])
    assert np.allclose(regrouped, episode["action"][: transitions * k])


def test_frameskip_sums_rewards_over_the_group():
    episode = {
        "state": np.zeros((11, 2), np.float32),
        "action": np.zeros((10, 2), np.float32),
        "reward": np.ones(10, np.float32),
    }
    out = frameskip_episode(episode, 5)
    assert np.allclose(out["reward"], 5.0)  # not averaged, not sampled


def test_frameskip_of_one_changes_nothing():
    episode = {
        "state": np.arange(12, dtype=np.float32).reshape(6, 2),
        "action": np.arange(10, dtype=np.float32).reshape(5, 2),
    }
    out = frameskip_episode(episode, 1)
    assert np.allclose(out["state"], episode["state"])
    assert np.allclose(out["action"], episode["action"])


def test_frameskip_refuses_an_episode_it_would_empty():
    episode = {"state": np.zeros((3, 2), np.float32), "action": np.zeros((2, 2), np.float32)}
    with pytest.raises(ValueError, match="at least 2 are needed"):
        frameskip_episode(episode, 10)


def test_unblock_reverses_the_grouping():
    blocked = np.arange(10, dtype=np.float32)
    assert unblock(blocked, 2).shape == (5, 2)
    assert np.allclose(unblock(blocked, 2).reshape(-1), blocked)
    with pytest.raises(ValueError, match="does not divide"):
        unblock(blocked, 3)


# -- action scaling ---------------------------------------------------------
@pytest.mark.parametrize("shape", [(2,), (10,), (1, 10), (4, 3, 10)])
def test_action_normalisation_round_trips(shape):
    low, high = np.zeros(2, np.float32), np.full(2, 512.0, np.float32)
    raw = np.random.default_rng(0).uniform(0, 512, shape).astype(np.float32)
    normalised = normalize_action(raw, low, high)
    assert normalised.shape == raw.shape
    assert -1.0 <= normalised.min() and normalised.max() <= 1.0
    assert np.allclose(denormalize_action(normalised, low, high), raw, atol=1e-3)


def test_action_normalisation_is_per_dimension():
    """Asymmetric bounds must not be collapsed into one scale."""
    low, high = np.array([0.0, -1.0], np.float32), np.array([512.0, 1.0], np.float32)
    assert np.allclose(
        normalize_action(np.array([0.0, -1.0, 512.0, 1.0], np.float32), low, high),
        [-1.0, -1.0, 1.0, 1.0],
    )


# -- splits -----------------------------------------------------------------
def test_split_is_deterministic_and_disjoint():
    train, val = split_episodes(20, holdout=0.2, seed=0)
    again, _ = split_episodes(20, holdout=0.2, seed=0)
    assert np.array_equal(train, again)
    assert not set(train) & set(val)
    assert len(train) + len(val) == 20
    assert len(val) == 4


def test_split_always_holds_something_out():
    _, val = split_episodes(3, holdout=0.01, seed=0)
    assert len(val) >= 1


def test_split_needs_something_to_split():
    with pytest.raises(ValueError, match="at least 2 episodes"):
        split_episodes(1, holdout=0.1, seed=0)


# -- the registry -----------------------------------------------------------
def test_registry_lists_and_describes_without_building():
    assert xwm.tasks.available()
    for name in xwm.tasks.available():
        spec = xwm.tasks.describe(name)
        assert spec.name == name
        assert spec.summary


def test_an_unknown_task_names_the_known_ones():
    with pytest.raises(KeyError, match="unknown task"):
        xwm.tasks.describe("no/such")


def test_overrides_reach_the_spec():
    task = xwm.tasks.create("pusht/synthetic", episodes=3, frameskip=2)
    assert task.spec.episodes == 3
    assert task.spec.blocked_action_dim == 4  # 2 raw dims x frameskip 2


def test_an_unknown_override_is_refused():
    with pytest.raises(TypeError, match="no field"):
        xwm.tasks.create("pusht/synthetic", epsiodes=3)


@pytest.mark.parametrize("name", xwm.tasks.available())
def test_every_task_agrees_with_its_environment(name):
    task = xwm.tasks.create(name)
    env = task.make_env()
    try:
        assert env.action_dim == task.spec.action_dim
        assert env.native_size == task.spec.native_size
        assert env.state_dim <= task.spec.state_dim
    finally:
        env.close()


# -- the data pipeline ------------------------------------------------------
def test_episodes_split_and_satisfy_the_recorded_contract():
    task = xwm.tasks.create("pusht/synthetic")
    train, val = task.episodes(split="train"), task.episodes(split="val")
    assert len(train) > len(val) > 0
    for episode in train[:3]:
        check_episode(episode)
        assert episode["video"].dtype == np.uint8
        assert -1.0 <= episode["action"].min() and episode["action"].max() <= 1.0


def test_batches_match_the_loss_contract():
    task = xwm.tasks.create("pusht/synthetic")
    batch = next(task.batches(batch_size=4, key=jr.PRNGKey(0)))
    frames, actions = batch["video"], batch["action"]
    assert frames.shape[0] == actions.shape[0] == 4
    assert actions.shape[1] == frames.shape[1] - 1  # the T / T-1 convention
    assert actions.shape[2] == task.spec.blocked_action_dim
    assert 0.0 <= float(frames.min()) and float(frames.max()) <= 1.0


def test_batches_do_not_run_out_after_one_epoch():
    """`Trainer.fit` stops when either the budget or the iterator ends.

    A finite stream silently caps a run at one epoch, and the run reports
    success exactly as loudly as one that trained for the requested number of
    steps -- which is how a 2000-step experiment quietly becomes a 41-step one.
    """
    task = xwm.tasks.create("pusht/synthetic")
    stream = task.batches(batch_size=32, key=jr.PRNGKey(0))
    assert sum(1 for _ in itertools.islice(stream, 500)) == 500


def test_the_cache_is_reused_rather_than_recomputed():
    task = xwm.tasks.create("pusht/synthetic")
    first = task.episodes(split="val")
    fresh = xwm.tasks.create("pusht/synthetic")  # new instance, warm disk cache
    assert np.allclose(fresh.episodes(split="val")[0]["state"], first[0]["state"])


def test_a_recorded_state_resets_the_environment():
    """What the `replay` baseline depends on, asserted directly."""
    task = xwm.tasks.create("pusht/synthetic")
    env = task.make_env()
    try:
        episode = task.episodes(split="val")[0]
        recorded = task.env_state(episode["state"][4], env)
        env.reset(state=recorded)
        assert np.allclose(env.state(), recorded, atol=1e-5)
        # And the recording replays from there.
        for action in episode["action"][4:8]:
            env.step(task.denormalize_action(action))
        assert np.allclose(env.state(), task.env_state(episode["state"][8], env), atol=1e-4)
    finally:
        env.close()


def test_env_frames_and_dataset_frames_preprocess_identically():
    """Preprocessing parity is what keeps evaluation in the training distribution."""
    task = xwm.tasks.create("pusht/synthetic")
    env = task.make_env()
    try:
        live = task.observation(env.reset(seed=0))
        recorded = task.observation({"image": task.episodes(split="val")[0]["video"][0]})
    finally:
        env.close()
    assert live.shape == recorded.shape
    assert live.dtype == recorded.dtype
    assert 0.0 <= float(live.min()) and float(live.max()) <= 1.0


@pytest.mark.parametrize(
    ("model_name", "observation", "kwargs", "expected_key"),
    [
        ("jepa/action", ("video",), {"size": "tiny", "patch_size": 8}, "video"),
        ("tdmpc2", ("state",), {}, "observation"),
    ],
)
def test_batches_are_named_the_way_each_family_reads_them(
    model_name, observation, kwargs, expected_key
):
    """The task chooses which array is observed; the model chooses what it is called."""
    import jax.numpy as jnp

    from xwm.cli.build import batch_keys, build_model, build_task

    config = xwm.config.from_dict(
        {
            "task": {"name": "pusht/synthetic", "overrides": {"observation": list(observation)}},
            "model": {"name": model_name, "kwargs": kwargs},
        }
    )
    task = build_task(config)
    model = build_model(config, task, key=jr.PRNGKey(0))
    assert batch_keys(model)["observation"] == expected_key

    batch = next(
        task.batches(batch_size=2, key=jr.PRNGKey(0), length=4, observation_key=expected_key)
    )
    assert expected_key in batch
    # A state-observation model must not be shipped a batch of frames it ignores.
    assert "video" not in batch or expected_key == "video"
    loss, _ = model.loss(batch, key=jr.PRNGKey(1), target=model if model.uses_target else None)
    assert bool(jnp.isfinite(loss))


def test_muzero_cannot_be_trained_from_a_recording(tmp_path):
    """Its targets come from its own search, so no recorded corpus contains them."""
    from xwm.cli.build import build_model, build_task
    from xwm.cli.train import _check_trainable_offline

    config = xwm.config.from_dict(
        {
            "task": {"name": "pusht/synthetic", "overrides": {"observation": ["state"]}},
            "model": {"name": "muzero"},
        }
    )
    task = build_task(config)
    model = build_model(config, task, key=jr.PRNGKey(0))
    with pytest.raises(ValueError, match="targets from its own search"):
        _check_trainable_offline("muzero", model)


def test_the_scripted_demonstrator_actually_performs_the_task():
    """Random actions leave the puck untouched, which makes every instance trivial."""
    task = xwm.tasks.create("pusht/synthetic")
    moved = [
        np.linalg.norm(e["state"][s + 6][2:4] - e["state"][s][2:4])
        for e in task.episodes(split="train")[:20]
        for s in range(e["state"].shape[0] - 6)
    ]
    threshold = task.spec.metric_options["threshold"]
    assert np.median(moved) > 2 * threshold


def test_a_limited_run_does_not_poison_the_cache():
    """A debug run with `limit=` must not truncate the corpus for every run after it.

    The cache key has to cover `limit`, because a smaller corpus trains and
    reports exactly like a large one -- there is nothing in the output of the
    second run to say it saw 12% of the data.
    """
    task = xwm.tasks.create("pusht/synthetic")
    limited = len(task.episodes(split="train", limit=8))
    full = len(xwm.tasks.create("pusht/synthetic").episodes(split="train"))
    assert limited < full


def test_both_splits_are_cached_in_one_pass():
    """Reading a corpus is the expensive half; it must not happen once per split."""
    from xwm.tasks.data import cache_path

    task = xwm.tasks.create("pusht/synthetic")
    task.episodes(split="train")
    assert cache_path(task.spec, "train").exists()
    assert cache_path(task.spec, "val").exists()

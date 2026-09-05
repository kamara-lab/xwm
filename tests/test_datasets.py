"""Readers for recorded datasets.

No network. Every reader is exercised against a fixture written into ``tmp_path``
in the real on-disk format -- a genuine parquet shard, a genuine mp4, a genuine
HDF5 -- because a reader tested against a mock of the format is a reader tested
against my belief about the format.

The tests that do touch the network live at the bottom, behind
``XWM_DATASET_TESTS=1``. They never run in CI.
"""

import dataclasses
import json
import os

import numpy as np
import pytest

import xwm
from xwm.datasets import spec

h5py = pytest.importorskip("h5py")
pq = pytest.importorskip("pyarrow.parquet")
pa = pytest.importorskip("pyarrow")
av = pytest.importorskip("av")


# -- fixture builders -------------------------------------------------------
def replace_source(entry, source):
    """A copy of a registry entry pointing at a local file."""
    return dataclasses.replace(entry, source=source)



def write_robomimic(path, *, n_demos=3, n_frames=12, size=16, next_obs=False, mask=False):
    """A LIBERO/RoboMimic-shaped file: data/demo_N/obs/..., actions, rewards."""
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as handle:
        data = handle.create_group("data")
        for i in range(n_demos):
            demo = data.create_group(f"demo_{i}")
            obs = demo.create_group("obs")
            obs["agentview_rgb"] = rng.integers(0, 255, (n_frames, size, size, 3), np.uint8)
            obs["joint_states"] = rng.normal(size=(n_frames, 7)).astype(np.float32)
            obs["gripper_states"] = rng.normal(size=(n_frames, 2)).astype(np.float32)
            if next_obs:
                following = demo.create_group("next_obs")
                following["agentview_rgb"] = rng.integers(
                    0, 255, (n_frames, size, size, 3), np.uint8
                )
                following["joint_states"] = rng.normal(size=(n_frames, 7)).astype(np.float32)
                following["gripper_states"] = rng.normal(size=(n_frames, 2)).astype(np.float32)
            # One action per frame, including a final one whose result is not
            # in the file. Every real dataset in this format does this.
            demo["actions"] = np.arange(n_frames * 7, dtype=np.float32).reshape(n_frames, 7)
            demo["rewards"] = np.zeros(n_frames, np.float32)
            demo["dones"] = np.eye(n_frames, dtype=np.float32)[-1]
        if mask:
            group = handle.create_group("mask")
            group["train"] = np.array([b"demo_0", b"demo_2"])
    return path


def write_ogbench(path, *, n_episodes=3, steps=10, obs_dim=4, visual=False, compact=False):
    """A flat OGBench/D4RL-shaped .npz."""
    n = n_episodes * steps
    terminals = np.zeros(n, np.float32)
    terminals[steps - 1 :: steps] = 1.0
    if visual:
        observations = np.zeros((n, 16, 16, 3), np.uint8)
        observations[:, :, :, 0] = np.arange(n, dtype=np.uint8)[:, None, None]
    else:
        observations = np.arange(n * obs_dim, dtype=np.float32).reshape(n, obs_dim)
    actions = np.arange(n * 2, dtype=np.float32).reshape(n, 2)
    arrays = {"observations": observations, "actions": actions}
    if compact:
        arrays["valids"] = 1.0 - terminals
    else:
        arrays["terminals"] = terminals
        arrays["next_observations"] = np.roll(observations, -1, axis=0)
        arrays["masks"] = np.ones(n, np.float32)
    np.savez(path, **arrays)
    return path


def _write_mp4(path, frames, fps=10):
    """Encode ``(T, H, W, 3)`` uint8 to h264, as a LeRobot video shard is."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.height, stream.width = frames.shape[1], frames.shape[2]
        stream.pix_fmt = "yuv420p"
        for frame in frames:
            picture = av.VideoFrame.from_ndarray(np.ascontiguousarray(frame), format="rgb24")
            for packet in stream.encode(picture):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def write_lerobot(
    root, *, version="v3.0", n_episodes=3, n_frames=12, size=16, fps=10, position=False
):
    """A LeRobot dataset, in either layout.

    v3: many episodes per parquet and per mp4, with the boundaries in
    ``meta/episodes/``. v2.1: one file per episode.
    """
    root.mkdir(parents=True, exist_ok=True)
    (root / "meta").mkdir(exist_ok=True)
    camera = "observation.images.top"
    v3 = version.startswith("v3")

    data_path = (
        "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
        if v3
        else "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    )
    video_path = (
        "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
        if v3
        else "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
    )
    (root / "meta" / "info.json").write_text(
        json.dumps(
            {
                "codebase_version": version,
                "fps": fps,
                "chunks_size": 1000,
                "total_episodes": n_episodes,
                "total_frames": n_episodes * n_frames,
                "data_path": data_path,
                "video_path": video_path,
                "features": {
                    "action": {"dtype": "float32", "shape": [7]},
                    "observation.state": {"dtype": "float32", "shape": [7]},
                    **(
                        {"observation.environment_state": {"dtype": "float32", "shape": [4]}}
                        if position
                        else {}
                    ),
                    camera: {"dtype": "video", "shape": [size, size, 3]},
                },
            }
        )
    )

    # Frame t of episode e is a flat grey ramp, so decoded order is checkable
    # through a lossy codec: brightness must increase along the episode.
    def episode_frames(index):
        base = np.linspace(20, 235, n_frames).astype(np.uint8)
        frames = np.zeros((n_frames, size, size, 3), np.uint8)
        frames[:] = base[:, None, None, None]
        frames[:, 0, 0, 0] = index
        return frames

    rows = {"action": [], "observation.state": [], "episode_index": [], "task_index": []}
    if position:
        rows["observation.environment_state"] = []
    records = []
    all_frames = []
    for episode in range(n_episodes):
        actions = (np.arange(n_frames * 7) + episode * 1000).astype(np.float32).reshape(n_frames, 7)
        states = actions + 0.5
        # Scene state, not proprioception: a different column with a different width.
        scene = (states[:, :4] * -1.0).astype(np.float32)
        if v3:
            rows["action"].extend(actions.tolist())
            rows["observation.state"].extend(states.tolist())
            if position:
                rows["observation.environment_state"].extend(scene.tolist())
            rows["episode_index"].extend([episode] * n_frames)
            rows["task_index"].extend([episode % 2] * n_frames)
            start = episode * n_frames
            records.append(
                {
                    "episode_index": episode,
                    "length": n_frames,
                    "tasks": ["do the thing"],
                    "task_index": episode % 2,
                    "data/chunk_index": 0,
                    "data/file_index": 0,
                    "data/from_index": start,
                    "data/to_index": start + n_frames,
                    f"videos/{camera}/chunk_index": 0,
                    f"videos/{camera}/file_index": 0,
                    f"videos/{camera}/from_timestamp": start / fps,
                    f"videos/{camera}/to_timestamp": (start + n_frames) / fps,
                }
            )
            all_frames.append(episode_frames(episode))
        else:
            table = pa.table(
                {
                    "action": actions.tolist(),
                    "observation.state": states.tolist(),
                    **(
                        {"observation.environment_state": scene.tolist()} if position else {}
                    ),
                    "episode_index": [episode] * n_frames,
                    "task_index": [episode % 2] * n_frames,
                }
            )
            target = root / data_path.format(episode_chunk=0, episode_index=episode)
            target.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, target)
            _write_mp4(
                root / video_path.format(episode_chunk=0, episode_index=episode, video_key=camera),
                episode_frames(episode),
                fps,
            )
            records.append(
                {"episode_index": episode, "length": n_frames, "tasks": ["do the thing"]}
            )

    if v3:
        target = root / data_path.format(chunk_index=0, file_index=0)
        target.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table(rows), target)
        _write_mp4(
            root / video_path.format(video_key=camera, chunk_index=0, file_index=0),
            np.concatenate(all_frames),
            fps,
        )
        meta = root / "meta" / "episodes" / "chunk-000"
        meta.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(records), meta / "file-000.parquet")
    else:
        (root / "meta" / "episodes.jsonl").write_text(
            "\n".join(json.dumps(r) for r in records) + "\n"
        )
    return root


# -- the contract -----------------------------------------------------------
def test_check_episode_catches_the_undropped_final_action():
    """The off-by-one this whole package is arranged around."""
    with pytest.raises(ValueError, match="drop the final recorded action"):
        spec.check_episode({"state": np.zeros((10, 4)), "action": np.zeros((10, 2))})
    assert spec.check_episode({"state": np.zeros((10, 4)), "action": np.zeros((9, 2))}) == 10


def test_clips_never_straddle_an_episode_boundary():
    """Overlapping windows, two episodes, and no clip may mix them."""
    episodes = [
        {"state": np.full((8, 1), float(i)), "action": np.full((7, 1), float(i))} for i in range(2)
    ]
    out = spec.clips(episodes, length=4, stride=2)
    for clip in out["state"]:
        assert len(set(clip.reshape(-1).tolist())) == 1


def test_clips_reports_what_it_dropped():
    episodes = [{"state": np.zeros((3, 1)), "action": np.zeros((2, 1))} for _ in range(4)]
    episodes.append({"state": np.zeros((9, 1)), "action": np.zeros((8, 1))})
    out = spec.clips(episodes, length=8)
    assert out["state"].shape[0] == 1
    # Per-clip, so the whole dict shares a leading axis and iter_batches works.
    assert out["dropped"].tolist() == [4]
    lengths = {v.shape[0] for v in out.values()}
    assert len(lengths) == 1


def test_clips_refuses_an_undeclared_field():
    """Otherwise it would be copied whole into every clip, silently misaligned."""
    episodes = [{"state": np.zeros((4, 1)), "action": np.zeros((3, 1)), "wat": np.zeros((3,))}]
    with pytest.raises(ValueError, match="unknown field 'wat'"):
        spec.clips(episodes, length=4)


def test_to_frames_normalises_layout_and_range():
    channels_last = np.full((5, 8, 8, 3), 255, np.uint8)
    out = spec.to_frames(channels_last)
    assert out.shape == (5, 3, 8, 8)
    assert out.dtype == np.float32 and float(out.max()) == 1.0
    assert spec.to_frames(channels_last, dtype="uint8").dtype == np.uint8
    assert spec.to_frames(channels_last, resize=4).shape == (5, 3, 4, 4)
    assert spec.to_frames(np.zeros((5, 8, 8), np.uint8)).shape == (5, 1, 8, 8)


def test_episode_slices_includes_an_unterminated_tail():
    terminals = np.array([0, 0, 1, 0, 1, 0, 0])
    assert list(spec.episode_slices(terminals)) == [(0, 3), (3, 5), (5, 7)]


# -- OGBench ----------------------------------------------------------------
def test_ogbench_action_alignment(tmp_path):
    """action[t] must be the action recorded at frame t, per episode."""
    path = write_ogbench(tmp_path / "d.npz", n_episodes=3, steps=10, obs_dim=4)
    episodes = list(xwm.datasets.offline.iter_episodes("x", path=path))
    assert len(episodes) == 3
    first = episodes[0]
    # next_observations supplies the final frame, so all 10 actions survive.
    assert first["state"].shape == (11, 4)
    assert first["action"].shape == (10, 2)
    assert np.allclose(first["action"][0], [0.0, 1.0])
    assert np.allclose(first["state"][0], [0.0, 1.0, 2.0, 3.0])
    # Episode 1 starts at flat step 10.
    assert np.allclose(episodes[1]["action"][0], [20.0, 21.0])


def test_ogbench_compact_variant_drops_the_last_step(tmp_path):
    """No next_observations means the final action has no observed result."""
    path = write_ogbench(tmp_path / "c.npz", steps=10, compact=True)
    episodes = list(xwm.datasets.offline.iter_episodes("x", path=path))
    assert episodes[0]["state"].shape[0] == 10
    assert episodes[0]["action"].shape[0] == 9


def test_ogbench_masks_are_kept_and_windowed(tmp_path):
    path = write_ogbench(tmp_path / "m.npz", steps=10)
    out = xwm.datasets.offline.load("x", path=path, length=6, stride=3)
    assert out["mask"].shape == (out["action"].shape[0], 5)


def test_ogbench_visual_variant(tmp_path):
    path = write_ogbench(tmp_path / "v.npz", steps=8, visual=True)
    out = xwm.datasets.offline.load("x", path=path, length=4, resize=8)
    assert out["video"].shape[1:] == (4, 3, 8, 8)
    assert "state" not in out


# -- LIBERO / RoboMimic -----------------------------------------------------
def test_hdf5_contract_and_alignment(tmp_path):
    path = write_robomimic(tmp_path / "libero.hdf5", n_demos=2, n_frames=12)
    episodes = list(xwm.datasets.hdf5.iter_episodes(path))
    assert len(episodes) == 2
    episode = episodes[0]
    assert episode["video"].shape == (12, 3, 16, 16)
    assert episode["state"].shape == (12, 9)  # gripper (2) + joint (7)
    assert episode["action"].shape == (11, 7)  # the final action is dropped
    assert np.allclose(episode["action"][0], np.arange(7))


def test_hdf5_next_obs_keeps_every_action(tmp_path):
    path = write_robomimic(tmp_path / "n.hdf5", n_demos=1, n_frames=10, next_obs=True)
    (episode,) = list(xwm.datasets.hdf5.iter_episodes(path))
    assert episode["video"].shape[0] == 11
    assert episode["action"].shape[0] == 10


def test_hdf5_demos_are_read_in_numeric_order(tmp_path):
    """demo_10 must not sort between demo_1 and demo_2, or `limit` is a lie."""
    path = write_robomimic(tmp_path / "many.hdf5", n_demos=12, n_frames=6)
    episodes = list(xwm.datasets.hdf5.iter_episodes(path, observation="state", limit=3))
    assert len(episodes) == 3
    # Every demo has identical contents here, so check the ordering directly.
    with h5py.File(path, "r") as handle:
        from xwm.datasets.hdf5 import _demo_order

        assert _demo_order(list(handle["data"]))[:3] == ["demo_0", "demo_1", "demo_2"]


def test_hdf5_split_and_state_only(tmp_path):
    path = write_robomimic(tmp_path / "s.hdf5", n_demos=3, n_frames=8, mask=True)
    episodes = list(xwm.datasets.hdf5.iter_episodes(path, observation="state", split="train"))
    assert len(episodes) == 2
    assert "video" not in episodes[0]
    with pytest.raises(KeyError, match="no camera"):
        list(xwm.datasets.hdf5.iter_episodes(path, cameras=("nope",)))


def test_hdf5_multiple_cameras_concatenate_on_channels(tmp_path):
    path = write_robomimic(tmp_path / "two.hdf5", n_demos=1, n_frames=6)
    with h5py.File(path, "a") as handle:
        obs = handle["data/demo_0/obs"]
        obs["eye_in_hand_rgb"] = np.zeros((6, 16, 16, 3), np.uint8)
    (episode,) = list(
        xwm.datasets.hdf5.iter_episodes(path, cameras=("agentview_rgb", "eye_in_hand_rgb"))
    )
    assert episode["video"].shape == (6, 6, 16, 16)


# -- LeRobot ----------------------------------------------------------------
@pytest.mark.parametrize("version", ["v3.0", "v2.1"])
def test_lerobot_both_layouts(tmp_path, version):
    root = write_lerobot(tmp_path / version, version=version, n_episodes=3, n_frames=12)
    episodes = list(xwm.datasets.lerobot.iter_episodes("local", root=root))
    assert len(episodes) == 3
    episode = episodes[0]
    assert episode["video"].shape == (12, 3, 16, 16)
    assert episode["state"].shape == (12, 7)
    assert episode["action"].shape == (11, 7)
    assert np.allclose(episode["action"][0], np.arange(7))


def test_lerobot_v3_slices_the_right_rows_per_episode(tmp_path):
    """Many episodes share one parquet file; the row range comes from metadata."""
    root = write_lerobot(tmp_path / "v3", n_episodes=3, n_frames=12)
    episodes = list(xwm.datasets.lerobot.iter_episodes("local", root=root, observation="state"))
    # Episode e's actions were written offset by e * 1000.
    for index, episode in enumerate(episodes):
        assert np.allclose(episode["action"][0], np.arange(7) + index * 1000)


def test_lerobot_decodes_frames_in_order(tmp_path):
    """Through a lossy codec, so check the monotone ramp rather than the pixels."""
    root = write_lerobot(tmp_path / "v3", n_episodes=2, n_frames=12)
    (episode, _) = list(xwm.datasets.lerobot.iter_episodes("local", root=root))
    brightness = episode["video"].mean(axis=(1, 2, 3))
    assert np.all(np.diff(brightness) > 0)


@pytest.mark.parametrize("version", ["v2.1", "v3.0"])
def test_lerobot_surfaces_environment_state_as_position(tmp_path, version):
    """``observation.environment_state`` is scene state, and must not be dropped.

    Push-T records the agent in ``observation.state`` and the pushed block
    nowhere else, so a reader that matches only the ``observation.state`` prefix
    silently yields a dataset with no ground truth for the object being pushed.
    """
    root = tmp_path / "position"
    write_lerobot(root, version=version, n_episodes=2, n_frames=8, position=True)
    episode = next(xwm.datasets.lerobot.iter_episodes(None, root=root, observation="state"))

    assert episode["position"].shape == (8, 4)
    assert episode["state"].shape == (8, 7)
    # Distinct columns, not a concatenation of one into the other.
    assert np.allclose(episode["position"], -episode["state"][:, :4])
    spec.check_episode(episode)


def test_lerobot_without_environment_state_has_no_position(tmp_path):
    root = tmp_path / "nopos"
    write_lerobot(root, n_episodes=1, n_frames=6)
    episode = next(xwm.datasets.lerobot.iter_episodes(None, root=root, observation="state"))
    assert "position" not in episode


def test_lerobot_info_and_camera_selection(tmp_path):
    root = write_lerobot(tmp_path / "v3", n_episodes=1, n_frames=6)
    meta = xwm.datasets.lerobot.info("local", root=root)
    assert meta["total_episodes"] == 1 and meta["fps"] == 10
    with pytest.raises(KeyError, match="no camera"):
        list(xwm.datasets.lerobot.iter_episodes("local", root=root, cameras=("nope",)))
    (state_only,) = list(
        xwm.datasets.lerobot.iter_episodes("local", root=root, observation="state")
    )
    assert "video" not in state_only


def test_lerobot_load_produces_clips(tmp_path):
    root = write_lerobot(tmp_path / "v3", n_episodes=3, n_frames=12)
    out = xwm.datasets.lerobot.load("local", root=root, length=6, stride=6, resize=8)
    assert out["video"].shape == (6, 6, 3, 8, 8)
    assert out["action"].shape == (6, 5, 7)
    assert out["task"].shape == (6,)


# -- registry ---------------------------------------------------------------
def test_the_registry_lists_datasets_without_importing_a_reader():
    names = xwm.datasets.available()
    assert names and all("/" in name for name in names)
    with pytest.raises(KeyError, match="unknown dataset"):
        xwm.datasets.create("nope/nope")


def test_describe_reports_the_cost_before_the_download():
    for name in xwm.datasets.available():
        entry = xwm.datasets.describe(name)
        assert entry.summary and entry.reader in xwm.datasets.READERS
        assert entry.citation


def test_which_readers_probes_the_environment():
    readers = xwm.datasets.which_readers()
    assert readers["offline"] is True  # urllib only, always available
    assert set(readers) == set(xwm.datasets.READERS)


# -- interop ----------------------------------------------------------------
def test_to_replay_buffer_round_trips_an_episode(tmp_path):
    path = write_ogbench(tmp_path / "d.npz", n_episodes=2, steps=10, obs_dim=4)
    data = xwm.datasets.offline.load("x", path=path, length=11)
    buffer = xwm.datasets.to_replay_buffer(data, reward=0.0)
    assert len(buffer) == 2 * 10
    assert buffer.episodes == 2
    batch = buffer.sample(4, horizon=3)
    assert batch["observation"].shape == (4, 4, 4)  # (batch, horizon + 1, obs)
    assert batch["action"].shape == (4, 3, 2)


def test_to_replay_buffer_needs_a_reward(tmp_path):
    path = write_ogbench(tmp_path / "d.npz", n_episodes=2, steps=10)
    data = xwm.datasets.offline.load("x", path=path, length=11)
    with pytest.raises(ValueError, match="no 'reward'"):
        xwm.datasets.to_replay_buffer(data)


def test_a_loaded_dataset_trains_a_family(tmp_path):
    """The real contract: it drops into a family with no adaptation.

    Mirrors tests/test_envs.py::test_franka_data_trains_an_action_world_model,
    with the data coming off disk instead of out of a simulator.
    """
    import jax.numpy as jnp
    import jax.random as jr

    key = jr.PRNGKey(0)
    path = write_robomimic(tmp_path / "libero.hdf5", n_demos=4, n_frames=10, size=16)
    data = xwm.datasets.hdf5.load(path, length=4, observation="video")
    model = xwm.families.jepa.action_world_model(
        key=key,
        action_dim=7,
        encoder=xwm.encoders.ImageEncoder(
            key=key, img_size=16, patch_size=8, depth=2, embed_dim=64, num_heads=4
        ),
        dynamics_kwargs={"depth": 2, "pred_dim": 32, "num_heads": 4},
    )
    batch = {"video": jnp.asarray(data["video"]), "action": jnp.asarray(data["action"])}
    loss, metrics = model.loss(batch, key=key)
    assert bool(jnp.isfinite(loss))
    assert "loss_rollout" in metrics


# -- network, opt-in --------------------------------------------------------
needs_network = pytest.mark.skipif(
    os.environ.get("XWM_DATASET_TESTS") != "1",
    reason="set XWM_DATASET_TESTS=1 to download real datasets",
)


@needs_network
def test_real_lerobot_v3_state():
    """lerobot/droid_100: the v3 layout, state only, so no video shard is pulled."""
    entry = xwm.datasets.describe("lerobot/droid-100")
    meta = xwm.datasets.lerobot.info(entry.source)
    assert meta["total_episodes"] == entry.episodes
    episodes = list(
        xwm.datasets.lerobot.iter_episodes(entry.source, observation="state", limit=3)
    )
    assert len(episodes) == 3
    for episode in episodes:
        assert spec.check_episode(episode) == episode["action"].shape[0] + 1
        assert episode["state"].shape[1] == 7
    # Distinct episodes, i.e. the row ranges really were sliced per episode.
    assert not np.allclose(episodes[0]["state"][0], episodes[1]["state"][0])


@needs_network
def test_real_lerobot_v3_video():
    """lerobot/pusht: small enough to decode an mp4 shard in a test."""
    data = xwm.datasets.create("lerobot/pusht", length=8, limit=2, resize=32)
    assert data["video"].shape[1:] == (8, 3, 32, 32)
    assert data["action"].shape[1:] == (7, 2)
    assert data["reward"].shape[1] == 7


@needs_network
def test_real_lerobot_v21_layout():
    """oxe/berkeley-ur5 is v2.1: one parquet and one mp4 per episode."""
    entry = xwm.datasets.describe("oxe/berkeley-ur5")
    assert xwm.datasets.lerobot.info(entry.source)["codebase_version"].startswith("v2")
    episodes = list(xwm.datasets.lerobot.iter_episodes(entry.source, limit=2, resize=32))
    assert len(episodes) == 2
    for episode in episodes:
        spec.check_episode(episode)
        assert episode["video"].shape[1:] == (3, 32, 32)


@needs_network
def test_real_lerobot_inline_png_frames():
    """LIBERO's LeRobot conversion stores PNG in the parquet, not mp4."""
    data = xwm.datasets.create("libero/10", length=8, limit=2, resize=32, max_clips=4)
    assert data["video"].shape[1:] == (8, 3, 32, 32)
    assert float(data["video"].std()) > 0.0


@needs_network
def test_real_ogbench_dataset():
    data = xwm.datasets.create("ogbench/pointmaze-medium-navigate", length=16, limit=4)
    assert data["state"].shape[1:] == (16, 2)
    assert data["action"].shape[1:] == (15, 2)


@needs_network
def test_real_robomimic_dataset():
    data = xwm.datasets.create("robomimic/lift", length=16, limit=4)
    assert data["state"].ndim == 3
    assert data["action"].shape[1:] == (15, 7)
    buffer = xwm.datasets.to_replay_buffer(data)
    assert buffer.episodes == data["action"].shape[0]


def test_to_frames_resizes_float_input_without_blacking_it_out():
    """Casting [0, 1] floats to uint8 for PIL truncates every pixel to zero."""
    grey = np.full((2, 8, 8, 3), 0.5, np.float32)
    out = spec.to_frames(grey, resize=4)
    assert out.shape == (2, 3, 4, 4)
    assert 0.45 < float(out.mean()) < 0.55
    with pytest.raises(ValueError, match=r"assumes the range \[0, 1\]"):
        spec.to_frames(grey * 10.0, resize=4)


@needs_network
def test_real_lerobot_v20_layout():
    """oxe/bridge is v2.0 -- older still than v2.1, and also one file per episode."""
    entry = xwm.datasets.describe("oxe/bridge")
    meta = xwm.datasets.lerobot.info(entry.source)
    assert meta["codebase_version"].startswith("v2")
    (episode,) = list(
        xwm.datasets.lerobot.iter_episodes(entry.source, limit=1, resize=32)
    )
    spec.check_episode(episode)
    assert episode["video"].shape[1:] == (3, 32, 32)
    assert episode["action"].shape[1] == entry.action_dim


def test_cache_dir_and_clear_cache(tmp_path, monkeypatch):
    """The cache honours XWM_DATA_HOME, and clearing reports what it freed."""
    from xwm.datasets import cache

    monkeypatch.setenv("XWM_DATA_HOME", str(tmp_path / "home"))
    root = cache.cache_dir()
    assert root == tmp_path / "home" and root.is_dir()
    nested = cache.cache_dir("ogbench")
    (nested / "a.npz").write_bytes(b"x" * 100)
    assert cache.clear_cache("ogbench") == 100
    assert not nested.exists()
    # A single file, not a directory -- rglob would be the wrong tool.
    (root / "loose.npz").write_bytes(b"y" * 7)
    assert cache.clear_cache("loose.npz") == 7
    assert cache.clear_cache("never-existed") == 0


def test_create_passes_the_spec_modality_to_the_reader(tmp_path):
    """The registry knows what a dataset holds; the reader has to accept it.

    Regression: the spec-driven `observation=` default reached a reader that had
    no such parameter, so every `create("ogbench/...")` raised TypeError.
    """
    path = write_ogbench(tmp_path / "d.npz", n_episodes=2, steps=10)
    entry = xwm.datasets.describe("ogbench/pointmaze-medium-navigate")
    assert entry.observation == ("state",)
    data = xwm.datasets.create(entry.name, length=8, path=path)
    assert data["state"].shape[1] == 8
    # And asking a state file for pixels is an error, not silence.
    with pytest.raises(ValueError, match="holds state observations"):
        xwm.datasets.offline.load("x", path=path, length=8, observation="video")


def test_create_reaches_every_reader_kind_without_network(tmp_path, monkeypatch):
    """`create` dispatch, for the two readers a fixture can stand in for."""
    npz = write_ogbench(tmp_path / "d.npz", n_episodes=2, steps=8)
    assert "state" in xwm.datasets.create(
        "ogbench/antmaze-large-navigate", length=6, path=npz
    )
    hdf5_path = write_robomimic(tmp_path / "lift.hdf5", n_demos=2, n_frames=8)
    monkeypatch.setitem(
        xwm.datasets.DATASETS,
        "robomimic/lift",
        replace_source(xwm.datasets.DATASETS["robomimic/lift"], str(hdf5_path)),
    )
    data = xwm.datasets.create("robomimic/lift", length=6)
    assert "state" in data and "video" not in data  # spec says state-only


def test_hdf5_reads_many_files_and_limit_spans_them(tmp_path):
    """LIBERO-90 is 90 per-task files, so `limit` has to count across files."""
    first = write_robomimic(tmp_path / "a.hdf5", n_demos=2, n_frames=8)
    second = write_robomimic(tmp_path / "b.hdf5", n_demos=2, n_frames=8)
    both = list(xwm.datasets.hdf5.iter_episodes([first, second], observation="state"))
    assert len(both) == 4
    # A directory is the same thing, discovered by glob.
    assert len(list(xwm.datasets.hdf5.iter_episodes(tmp_path, observation="state"))) == 4
    # And `limit` stops mid-stream rather than per file.
    assert len(list(xwm.datasets.hdf5.iter_episodes(tmp_path, observation="state", limit=3))) == 3


def test_hdf5_multi_file_resolves_lazily(tmp_path):
    """`limit` must not resolve files it never reads -- 90 downloads vs one."""
    paths = [write_robomimic(tmp_path / f"{i}.hdf5", n_demos=2, n_frames=8) for i in range(4)]
    touched = []

    def lazy():
        for path in paths:
            touched.append(path)
            yield path

    episodes = list(xwm.datasets.hdf5.iter_episodes(lazy(), observation="state", limit=3))
    assert len(episodes) == 3
    # Two files hold four demos; the third and fourth must never be reached.
    assert len(touched) == 2


def test_minari_dict_observations_flatten_recursively():
    """D4RL/kitchen nests dicts inside achieved_goal; one level is not enough."""
    from xwm.datasets.offline import _flatten_observations

    flat = _flatten_observations(np.zeros((5, 3), np.float64))
    assert flat.shape == (5, 3) and flat.dtype == np.float32
    one_level = _flatten_observations(
        {"observation": np.zeros((5, 4)), "desired_goal": np.zeros((5, 2))}
    )
    assert one_level.shape == (5, 6)  # sorted: desired_goal | observation
    nested = _flatten_observations(
        {
            "achieved_goal": {"kettle": np.zeros((5, 7)), "microwave": np.zeros((5, 1))},
            "observation": np.zeros((5, 9)),
        }
    )
    assert nested.shape == (5, 17)
    # A trailing scalar axis still becomes a column rather than vanishing.
    assert _flatten_observations(np.zeros((5,))).shape == (5, 1)


def test_minari_rejects_a_pixel_request():
    with pytest.raises(ValueError, match="state-based"):
        xwm.datasets.offline.load_minari("whatever", observation="video")


@needs_network
def test_real_minari_dataset():
    """The smallest dataset in the registry, and the nested-dict case."""
    pytest.importorskip("minari")
    entry = xwm.datasets.describe("minari/kitchen-complete")
    data = xwm.datasets.create(entry.name, length=8, limit=4)
    assert data["action"].shape[1:] == (7, entry.action_dim)
    assert data["state"].shape[-1] == 81
    assert data["reward"].shape[1] == 7


@needs_network
def test_real_libero_90_streams_files_lazily():
    """LIBERO-90 has no LeRobot conversion: 90 raw HDF5 files, read on demand."""
    entry = xwm.datasets.describe("libero/90")
    assert entry.source.startswith("hf://") and entry.source.endswith("/")
    data = xwm.datasets.create("libero/90", length=8, limit=3, resize=32, max_clips=6)
    assert data["video"].shape[1:] == (8, 3, 32, 32)
    assert data["action"].shape[1:] == (7, entry.action_dim)
    assert float(data["video"].std()) > 0.0


# -- mixtures ---------------------------------------------------------------
def test_mixture_interleaves_and_pads_actions(tmp_path):
    """Cross-embodiment mixing: narrower actions padded, one shared array.

    Both members read the same fixture here -- what is under test is the
    interleaving and the padding, not the readers, which have their own tests.
    """
    path = write_ogbench(tmp_path / "n.npz", n_episodes=6, steps=10, obs_dim=4)
    out = xwm.datasets.mixture(
        {"ogbench/pointmaze-medium-navigate": 0.5, "ogbench/antmaze-large-navigate": 0.5},
        action_dim=7,
        length=6,
        resize=None,
        path=path,
    )
    assert out["action"].shape[1:] == (5, 7)
    assert np.all(out["action"][:, :, 2:] == 0.0)  # padded on the right
    # Both members contributed, and `source` records which.
    assert set(out["task"].tolist()) == {0, 1}


def test_mixture_requires_resize_and_refuses_to_truncate(tmp_path):
    path = write_ogbench(tmp_path / "n.npz", n_episodes=2, steps=8)
    with pytest.raises(ValueError, match="needs resize="):
        xwm.datasets.mixture({"ogbench/pointmaze-medium-navigate": 1.0}, action_dim=7)
    with pytest.raises(ValueError, match="wider than action_dim"):
        xwm.datasets.mixture(
            {"ogbench/pointmaze-medium-navigate": 1.0},
            action_dim=1,
            length=4,
            resize=None,
            path=path,
        )


def test_oxe_mixture_weights_name_registered_datasets():
    assert abs(sum(xwm.datasets.OXE_MIXTURE.values()) - 1.0) < 1e-9
    for name in xwm.datasets.OXE_MIXTURE:
        assert xwm.datasets.describe(name).reader == "lerobot"


def test_stream_yields_episodes_without_materialising(tmp_path):
    path = write_ogbench(tmp_path / "d.npz", n_episodes=3, steps=10)
    episodes = xwm.datasets.stream("ogbench/pointmaze-medium-navigate", path=path)
    first = next(iter(episodes))
    assert spec.check_episode(first) == 11
    with pytest.raises(NotImplementedError, match="no streaming path"):
        xwm.datasets.stream("minari/kitchen-complete")

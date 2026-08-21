"""The Newton Franka environment.

Skipped entirely when Newton is not installed. These tests check the properties
a world model actually depends on: actions must change the observation,
different actions must lead to different observations, the arm must hold still
when idle, and the simulation must never hand back NaNs.
"""

import numpy as np
import pytest

pytest.importorskip("newton")
pytest.importorskip("warp")

import xwm  # noqa: E402
from xwm.envs import FrankaConfig, FrankaEnv, franka_sequences, smooth_actions  # noqa: E402


@pytest.fixture(scope="module")
def env():
    """One env for the module -- building it downloads and compiles."""
    return FrankaEnv(FrankaConfig(image_size=32))


def test_env_shapes(env):
    assert env.action_dim == 7
    assert env.observation_shape == (3, 32, 32)
    obs = env.reset(seed=0)
    assert obs.shape == (3, 32, 32) and obs.dtype == np.float32
    assert 0.0 <= obs.min() and obs.max() <= 1.0


def test_tool_body_resolved_by_name(env):
    """The end effector must be found by label, not a hardcoded index."""
    labels = [str(x) for x in env.model.body_label]
    assert "hand" in labels[env.tool_body_index]


def test_servo_gains_reached_the_model(env):
    """Regression: JointDofConfig silently drops target_kd, which diverges."""
    kd = np.asarray(env.model.joint_target_kd.numpy())
    assert np.all(kd[: env.action_dim] > 0)


def test_actions_change_the_observation(env):
    before = env.reset(seed=1)
    after = before
    for _ in range(6):
        after = env.step(np.array([0.8, -0.5, 0.0, 0.4, 0.0, 0.5, 0.0], np.float32))
    assert env.is_finite()
    assert float(np.abs(after - before).mean()) > 1e-3


def test_opposite_actions_diverge(env):
    action = np.array([0.8, -0.5, 0.0, 0.4, 0.0, 0.5, 0.0], np.float32)
    env.reset(seed=1)
    for _ in range(6):
        forward = env.step(action)
    env.reset(seed=1)
    for _ in range(6):
        backward = env.step(-action)
    assert float(np.abs(forward - backward).mean()) > 1e-3


def test_actions_move_the_tool(env):
    env.reset(seed=2)
    start = env.tool_position().copy()
    for _ in range(8):
        env.step(np.array([0.8, -0.5, 0.0, 0.4, 0.0, 0.5, 0.0], np.float32))
    assert float(np.linalg.norm(env.tool_position() - start)) > 0.05


def test_arm_holds_still_when_idle(env):
    """If gravity dominates, the model learns gravity instead of the action."""
    env.reset(seed=3)
    start = env.joint_positions().copy()
    for _ in range(12):
        env.step(np.zeros(7, np.float32))
    assert env.is_finite()
    assert float(np.abs(env.joint_positions() - start).mean()) < 0.05


def test_joint_targets_respect_limits(env):
    env.reset(seed=4)
    for _ in range(40):
        env.step(np.ones(7, np.float32))
    q = env.joint_positions()
    assert env.is_finite()
    assert np.all(q <= env.joint_limit_upper[: env.action_dim] + 0.35)
    assert np.all(q >= env.joint_limit_lower[: env.action_dim] - 0.35)


def test_step_validates_action_shape(env):
    env.reset(seed=0)
    with pytest.raises(ValueError, match="action dims"):
        env.step(np.zeros(3))


def test_reset_is_reproducible_and_varied(env):
    a = env.reset(seed=7)
    b = env.reset(seed=7)
    c = env.reset(seed=8)
    assert np.allclose(a, b)
    assert not np.allclose(a, c)


def test_rollout_returns_one_more_frame_than_actions(env):
    result = env.rollout(np.zeros((5, 7), np.float32), seed=0)
    assert result["video"].shape == (6, 3, 32, 32)
    assert result["joint_q"].shape == (6, 7)
    assert result["tool"].shape == (6, 3)


def test_smooth_actions_are_bounded_and_correlated():
    rng = np.random.default_rng(0)
    smooth = smooth_actions(rng, 3, 40, 7, smoothness=0.95)
    rough = smooth_actions(rng, 3, 40, 7, smoothness=0.0)
    assert smooth.shape == (3, 40, 7)
    assert np.abs(smooth).max() <= 1.0
    assert np.abs(np.diff(smooth, axis=1)).mean() < np.abs(np.diff(rough, axis=1)).mean() / 3


def test_franka_sequences_matches_the_action_world_model_contract(env):
    data = franka_sequences(env, 3, 5, seed=0)
    assert data["video"].shape == (3, 5, 3, 32, 32)
    assert data["action"].shape == (3, 4, 7)  # T - 1 actions for T frames
    assert data["joint_q"].shape == (3, 5, 7)
    assert data["tool"].shape == (3, 5, 3)
    assert np.all(np.isfinite(data["video"]))


def test_franka_data_trains_an_action_world_model(env, key):
    """The env output must drop straight into xwm.action with no adaptation."""
    import jax.numpy as jnp

    data = franka_sequences(env, 4, 4, seed=1)
    model = xwm.families.jepa.action_world_model(
        key=key,
        action_dim=env.action_dim,
        encoder=xwm.encoders.ImageEncoder(
            key=key, img_size=32, patch_size=8, depth=2, embed_dim=64, num_heads=4
        ),
        dynamics_kwargs={"depth": 2, "pred_dim": 32, "num_heads": 4},
    )
    batch = {"video": jnp.asarray(data["video"]), "action": jnp.asarray(data["action"])}
    loss, metrics = model.loss(batch, key=key)
    assert bool(jnp.isfinite(loss))
    assert "loss_rollout" in metrics


def test_ground_truth_accessors_return_copies(env):
    """Warp's .numpy() is a view on live device memory; these must not alias it.

    Without copying, a value read now mutates as the simulation advances, and a
    collected trajectory becomes N references to the final state -- a corruption
    that shape and finiteness checks cannot see.
    """
    env.reset(seed=0)
    action = np.array([0.8, -0.5, 0.0, 0.4, 0.0, 0.5, 0.0], np.float32)
    held_tool = env.tool_position()
    held_joints = env.joint_positions()
    held_obs = env.observe()
    for _ in range(6):
        env.step(action)
    assert not np.allclose(held_tool, env.tool_position()), "tool_position aliases live state"
    assert not np.allclose(held_joints, env.joint_positions()), "joint_positions aliases"
    assert not np.allclose(held_obs, env.observe()), "observe aliases"


def test_rollout_records_distinct_states(env):
    """Regression: every row of a rollout used to be the final state."""
    action = np.tile(np.array([0.8, -0.5, 0.0, 0.4, 0.0, 0.5, 0.0], np.float32), (6, 1))
    result = env.rollout(action, seed=0)
    for field in ("tool", "joint_q", "video"):
        rows = result[field]
        assert not np.all(rows == rows[0]), f"{field} rows are all identical"
    # The tool must actually travel over the episode.
    assert np.linalg.norm(result["tool"][-1] - result["tool"][0]) > 0.05


def test_sequences_have_within_episode_variation(env):
    data = franka_sequences(env, 2, 6, seed=0)
    for field in ("tool", "joint_q"):
        for episode in data[field]:
            assert not np.all(episode == episode[0]), f"{field} constant within an episode"


def test_look_at_matches_newtons_convention():
    """Camera forward is local -z, up is +y: reproduce Newton's own quaternion.

    Newton's examples hardcode (0.5, 0.5, 0.5, 0.5) for a camera on +x looking
    back along -x with +z up, which pins the convention down.
    """
    from xwm.envs.newton_franka import look_at_quaternion

    q = look_at_quaternion(np.array([1.7, 0.0, 0.9]), np.array([0.0, 0.0, 0.9]))
    assert np.allclose(np.abs(q), 0.5, atol=1e-5)


def test_look_at_is_normalised_and_validated():
    from xwm.envs.newton_franka import look_at_quaternion

    for target in ([0.0, 0.0, 0.0], [0.0, 1.0, 0.5], [-1.0, -1.0, 2.0]):
        q = look_at_quaternion(np.array([1.0, 0.0, 1.0]), np.array(target))
        assert np.isclose(np.linalg.norm(q), 1.0, atol=1e-5)
    with pytest.raises(ValueError, match="coincide"):
        look_at_quaternion(np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]))
    # Looking straight down must not produce a degenerate basis.
    q = look_at_quaternion(np.array([0.0, 0.0, 2.0]), np.array([0.0, 0.0, 0.0]))
    assert np.all(np.isfinite(q))


def test_camera_frames_the_robot(env):
    """Framing is functional, not cosmetic.

    With the arm small in frame, one control step moves a handful of pixels,
    'predict no change' becomes an excellent baseline, and there is nearly
    nothing for a predictive model to learn.
    """
    action = np.tile(np.array([0.8, -0.5, 0.0, 0.4, 0.0, 0.5, 0.0], np.float32), (6, 1))
    video = env.rollout(action, seed=3)["video"]
    # The robot is bright against a dark floor; it must occupy a real share.
    occupancy = (video.mean(axis=1) > 0.55).mean()
    assert occupancy > 0.02, f"robot occupies only {occupancy:.1%} of the frame"
    # And consecutive frames must differ visibly.
    assert np.abs(np.diff(video, axis=0)).mean() > 5e-3


# -- task: reward and state observations -------------------------------------
def test_state_dim_matches_the_observation(env):
    """A mismatch here silently mis-sizes any agent built with env.state_dim."""
    assert env.state_dim == env.state_observation().shape[0]


def test_state_observation_is_finite_and_copies(env):
    env.reset(seed=0)
    held = env.state_observation()
    for _ in range(4):
        env.step(np.array([0.5, -0.3, 0.0, 0.2, 0.0, 0.3, 0.0], np.float32))
    assert np.all(np.isfinite(held))
    assert not np.allclose(held, env.state_observation()), "state_observation aliases"


def test_reward_is_monotone_in_distance(env):
    """Closer must always mean higher reward, whatever the trajectory did.

    Tested on the reward's *shape* rather than by random search: sampling in a
    7-D action space rarely improves a specific distance, so a search-based test
    would be measuring the sampler, not the reward.
    """
    env.reset(seed=0)
    pairs = []
    action = np.array([0.6, -0.4, 0.2, 0.3, 0.0, 0.4, 0.0], np.float32)
    for _ in range(10):
        env.step(action)
        pairs.append((env.goal_distance(), env.reward()))
    by_distance = [r for _, r in sorted(pairs)]
    assert by_distance == sorted(by_distance, reverse=True), (
        "reward must decrease monotonically with distance"
    )


def test_goal_is_reachable_and_not_already_solved(env):
    """A goal outside the reachable set trains nothing; one at home measures nothing."""
    env.reset(seed=0)
    home_distance = env.goal_distance()
    assert home_distance > 0.1, "goal is already solved at the home pose"

    best = home_distance
    for trial in range(20):
        actions = xwm.envs.smooth_actions(
            np.random.default_rng(100 + trial), 1, 6, env.action_dim, smoothness=0.5
        )[0]
        env.rollout(actions, seed=trial)
        best = min(best, env.goal_distance())
    assert best < 0.8 * home_distance, f"nothing got closer than {best:.3f} of {home_distance:.3f}"


def test_reward_is_bounded(env):
    """A bounded reward keeps the value head inside its bin range."""
    env.reset(seed=0)
    for _ in range(10):
        env.step(np.ones(env.action_dim, np.float32))
        assert -2.0 <= env.reward(np.ones(env.action_dim)) <= 0.0


def test_action_penalty_discourages_thrashing(env):
    env.reset(seed=0)
    idle = env.reward(np.zeros(env.action_dim))
    thrash = env.reward(np.ones(env.action_dim))
    assert thrash < idle


def test_rollout_carries_rewards_and_states(env):
    result = env.rollout(np.zeros((5, env.action_dim), np.float32), seed=1)
    assert result["reward"].shape == (5,)
    assert result["state"].shape == (6, env.state_dim)
    assert np.all(np.isfinite(result["reward"]))


# -- rendering ---------------------------------------------------------------
def test_render_at_any_resolution(env):
    """Figures should be rendered, not upscaled from the training resolution."""
    env.reset(seed=0)
    for size in (32, 64, 96):
        assert env.render(size).shape == (3, size, size)
    # observe() keeps using the resolution the model is trained on.
    assert env.observe().shape == env.observation_shape


def test_render_target_is_cached(env):
    env.reset(seed=0)
    env.render(48)
    before = len(env._targets)
    env.render(48)
    assert len(env._targets) == before, "re-rendering the same size rebuilt the buffers"


def test_render_validates_size(env):
    with pytest.raises(ValueError, match="at least 8"):
        env.render(4)


def test_high_resolution_render_carries_more_detail(env):
    """A native render must be sharper than an upscaled one, not merely bigger."""
    import numpy as np

    env.reset(seed=0)
    native = env.render(96)
    small = env.render(32)
    upscaled = np.repeat(np.repeat(small, 3, axis=1), 3, axis=2)
    # Block upscaling zeroes two of every three horizontal differences, so an
    # upscaled frame necessarily carries less local structure than a real render
    # at the same size. Measured ratio is ~1.37; 1.2 leaves room for the scene.
    detail = lambda f: float(np.abs(np.diff(f, axis=2)).mean())  # noqa: E731
    assert detail(native) > detail(upscaled) * 1.2


# -- rendering backends ------------------------------------------------------
def test_which_backends_reports_availability():
    """Check before a long run, not after."""
    backends = xwm.envs.which_backends()
    assert set(backends) == {"warp", "rtx", "usd"}
    assert backends["warp"] is True  # always, it is the Warp raytracer itself
    assert all(isinstance(v, bool) for v in backends.values())


def test_supersample_averages_blocks():
    frame = np.zeros((3, 9, 9), np.float32)
    frame[:, 0:3, 0:3] = 1.0
    out = xwm.envs.supersample(frame, 3)
    assert out.shape == (3, 3, 3)
    assert out[0, 0, 0] == pytest.approx(1.0)
    assert out[0, 1, 1] == pytest.approx(0.0)
    # A half-covered block averages to a half -- that is the anti-aliasing.
    half = np.zeros((3, 2, 2), np.float32)
    half[:, :, 0] = 1.0
    assert xwm.envs.supersample(half, 2)[0, 0, 0] == pytest.approx(0.5)


def test_supersample_validates():
    with pytest.raises(ValueError, match="divisible"):
        xwm.envs.supersample(np.zeros((3, 10, 10)), 3)
    identity = np.zeros((3, 4, 4))
    assert xwm.envs.supersample(identity, 1) is identity


def test_render_samples_smooths_edges(env):
    """Supersampling must actually change the pixels, not just cost more."""
    env.reset(seed=0)
    one = env.render(48, samples=1)
    three = env.render(48, samples=3)
    assert one.shape == three.shape
    assert not np.allclose(one, three)
    # Averaging can only reduce the extremes, never widen them.
    assert three.max() <= one.max() + 1e-6
    assert three.min() >= one.min() - 1e-6


def test_render_validates_samples(env):
    with pytest.raises(ValueError, match="at least 1"):
        env.render(32, samples=0)


def test_high_quality_renderer_rejects_the_fast_backend(env):
    with pytest.raises(ValueError, match="'rtx' or 'usd'"):
        env.high_quality_renderer(backend="warp")


def test_high_quality_renderer_reports_missing_dependencies(env):
    if xwm.envs.which_backends()["rtx"]:
        pytest.skip("ovrtx is installed, so there is no error to check")
    with pytest.raises(ImportError, match="ovrtx"):
        env.high_quality_renderer(backend="rtx")


def test_usd_export_writes_an_animated_stage(env, tmp_path):
    """The portable route to a publication figure: a stage rendered offline."""
    pytest.importorskip("pxr")
    from pxr import Usd

    path = tmp_path / "episode.usd"
    with env.high_quality_renderer(backend="usd", output_path=path, fps=10) as renderer:
        env.reset(seed=0)
        renderer.add(env.state)
        for _ in range(4):
            env.step(np.zeros(env.action_dim, np.float32))
            renderer.add(env.state)
    assert path.exists() and path.stat().st_size > 1000

    stage = Usd.Stage.Open(str(path))
    assert stage is not None
    assert len(list(stage.Traverse())) > 10, "stage has no geometry"
    assert stage.GetEndTimeCode() > stage.GetStartTimeCode(), "stage is not animated"


def test_usd_backend_requires_an_output_path(env):
    pytest.importorskip("pxr")
    with pytest.raises(ValueError, match="output_path"):
        env.high_quality_renderer(backend="usd")


# -- the path-traced rendering path ---------------------------------------


def test_look_at_angles_invert_the_viewers_forward_vector():
    """The angles must reproduce Newton's own camera formula, per up-axis.

    This is the one piece of the RTX path that cannot be caught by looking at
    an image: a sign error here aims the camera somewhere plausible-looking but
    wrong, and every path-traced figure silently frames the wrong thing.
    """
    from xwm.envs import look_at_angles

    def forward(pitch, yaw, up_axis):
        pitch, yaw = np.radians(pitch), np.radians(yaw)
        horizontal, vertical = np.cos(pitch), np.sin(pitch)
        components = {
            "X": (vertical, np.cos(yaw) * horizontal, np.sin(yaw) * horizontal),
            "Y": (np.cos(yaw) * horizontal, vertical, np.sin(yaw) * horizontal),
            "Z": (np.cos(yaw) * horizontal, np.sin(yaw) * horizontal, vertical),
        }
        return np.array(components[up_axis])

    rng = np.random.default_rng(0)
    for up_axis in ("X", "Y", "Z"):
        for _ in range(16):
            eye, target = rng.normal(size=3), rng.normal(size=3)
            pitch, yaw = look_at_angles(eye, target, up_axis)
            want = (target - eye) / np.linalg.norm(target - eye)
            assert np.allclose(forward(pitch, yaw, up_axis), want, atol=1e-9)


def test_look_at_angles_rejects_a_degenerate_camera():
    from xwm.envs import look_at_angles

    with pytest.raises(ValueError, match="coincide"):
        look_at_angles([1.0, 0.0, 0.5], [1.0, 0.0, 0.5])


def test_frame_conversion_drops_alpha_and_scales_to_unit_range():
    from xwm.envs.render import _to_frame

    image = np.zeros((4, 6, 4), dtype=np.uint8)
    image[..., 0] = 255  # pure red, opaque
    image[..., 3] = 128  # a half-alpha that must not survive
    frame = _to_frame(image)

    assert frame.shape == (3, 4, 6)
    assert frame.dtype == np.float32
    assert np.allclose(frame[0], 1.0)
    assert np.allclose(frame[1:], 0.0)

    with pytest.raises(ValueError, match="H, W, C"):
        _to_frame(np.zeros((4, 6)))


def test_the_high_quality_camera_matches_the_training_camera():
    """The two renderers must look at the same place from the same place."""
    env = xwm.envs.FrankaEnv(xwm.envs.FrankaConfig(image_size=32))
    eye, target = env.camera_framing

    assert np.allclose(eye, [env.config.camera_distance, 0.0, env.config.camera_height])
    assert np.allclose(target, [0.0, 0.0, env.config.camera_target_height])

    # The Warp camera's quaternion and the viewer's pitch/yaw are two encodings
    # of one direction; both must decode to eye -> target.
    from xwm.envs import look_at_angles

    pitch, yaw = look_at_angles(eye, target, "Z")
    horizontal = np.cos(np.radians(pitch))
    forward = np.array(
        [np.cos(np.radians(yaw)) * horizontal,
         np.sin(np.radians(yaw)) * horizontal,
         np.sin(np.radians(pitch))]
    )
    assert np.allclose(forward, (target - eye) / np.linalg.norm(target - eye), atol=1e-6)


def test_frames_are_only_available_on_the_rtx_backend():
    from xwm.envs.render import HighQualityRenderer

    renderer = HighQualityRenderer.__new__(HighQualityRenderer)
    renderer.backend, renderer._frames = "usd", []
    with pytest.raises(AttributeError, match="writes a file"):
        _ = renderer.frames

    renderer.backend = "rtx"
    with pytest.raises(RuntimeError, match="no frames captured"):
        _ = renderer.frames


class _FakeViewer:
    """A stand-in for ``ViewerRTX``, which needs a graphics-capable GPU.

    The capture path cannot be exercised against the real path tracer on a
    compute-only container, so the frame bookkeeping -- frames in order, one per
    state, time advancing at the requested rate -- is verified against a stub
    that speaks the same three-call protocol.
    """

    def __init__(self, *, private_grab=True):
        self.times: list[float] = []
        self.states: list[object] = []
        self.private_grab = private_grab
        self._next = 0
        if private_grab:
            self._capture_screenshot_pixels = self._pixels

    def _pixels(self):
        self._next += 1
        return np.full((4, 6, 4), self._next, dtype=np.uint8)

    def begin_frame(self, time):
        self.times.append(time)

    def log_state(self, state):
        self.states.append(state)

    def end_frame(self):
        pass

    def save_screenshot(self, path):
        from PIL import Image

        Image.fromarray(self._pixels()).save(path)

    def close(self):
        pass


def _rtx_renderer(viewer, fps=10):
    from xwm.envs.render import HighQualityRenderer

    renderer = HighQualityRenderer.__new__(HighQualityRenderer)
    renderer.backend, renderer.fps, renderer.up_axis = "rtx", fps, "Z"
    renderer._time, renderer._frames, renderer._viewer = 0.0, [], viewer
    return renderer


def test_rtx_capture_collects_one_frame_per_state_in_order():
    viewer = _FakeViewer()
    renderer = _rtx_renderer(viewer, fps=10)

    for state in ("a", "b", "c"):
        renderer.add(state)

    frames = renderer.frames
    assert frames.shape == (3, 3, 4, 6)
    assert viewer.states == ["a", "b", "c"]
    assert np.allclose(viewer.times, [0.0, 0.1, 0.2])  # fps=10
    # Frame i must be the i-th render, not a repeat of the last one -- the bug
    # that silently turns an animation into a still.
    assert np.allclose([f[0, 0, 0] for f in frames], np.arange(1, 4) / 255, atol=1e-6)


def test_rtx_capture_falls_back_to_save_screenshot():
    """Newton may drop the private grab; the documented path must still work."""
    viewer = _FakeViewer(private_grab=False)
    assert not hasattr(viewer, "_capture_screenshot_pixels")
    renderer = _rtx_renderer(viewer)

    renderer.add("only")
    frames = renderer.frames

    assert frames.shape == (1, 3, 4, 6)
    assert np.allclose(frames[0], 1 / 255, atol=1e-6)


def test_only_the_rtx_backend_captures_frames():
    viewer = _FakeViewer()
    renderer = _rtx_renderer(viewer)
    renderer.backend = "usd"

    renderer.add("state")

    assert viewer.states == ["state"]  # still logged to the stage
    assert renderer._frames == []  # but no pixels buffered

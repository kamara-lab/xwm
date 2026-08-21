"""A Franka arm in Newton, as a source of action-labelled video.

`Newton <https://github.com/newton-physics/newton>`_ is a GPU-accelerated
robotics physics engine built on NVIDIA Warp. This module wraps a Franka
Emika FR3 arm in it and exposes exactly what a predictive world model needs:
RGB observations, continuous joint-space actions, and ground-truth state for
evaluation only.

Why this and not the toy world in :mod:`xwm.data`? Because the toy world's
dynamics are linear and fully observable, so it cannot show whether latent
prediction actually buys anything. A 7-DoF arm rendered to pixels is
genuinely hard in the ways that matter: the mapping from joint angles to
image is nonlinear and many-to-one, self-occlusion hides the state, and
contact makes the dynamics non-smooth.

**This is not a JAX object.** Newton and Warp own mutable device state, so
:class:`FrankaEnv` is an ordinary Python class that steps in place, and the
boundary to xwm is NumPy arrays. Keeping the simulator outside the JAX
world is deliberate -- it is the one part of the pipeline that cannot be
``jit``ed, and pretending otherwise would only hide that.

Install the extra first::

    pip install "xwm[newton]"

Rendering uses Newton's Warp-based raytracer, which runs on CPU, so this
works headless and without a GPU (slower, but it works).

References
----------
Newton: a GPU-accelerated robotics physics engine built on NVIDIA Warp.
https://github.com/newton-physics/newton

Featherstone, *Rigid Body Dynamics Algorithms*, Springer 2008 -- the
articulated-body solver used when the MuJoCo backend is unavailable.

Todorov, Erez & Tassa, *MuJoCo: A Physics Engine for Model-Based Control*, IROS
2012 -- the preferred backend, via ``mujoco_warp``.

Haddadin et al. and the Franka Emika FR3 -- the 7-DoF arm modelled here.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# Franka FR3 joint limits are enforced by the URDF; these are the *actuated*
# arm joints, excluding the two gripper fingers.
N_ARM_JOINTS = 7

#: A neutral, well-conditioned arm pose to perturb around (radians).
HOME_POSE = np.array([0.0, -0.4, 0.0, -2.0, 0.0, 1.6, 0.8], dtype=np.float32)

#: World up axis, for camera framing.
WORLD_UP = np.array([0.0, 0.0, 1.0], dtype=np.float32)

#: Default reach target. Chosen by measuring the reachable set rather than by
#: eye: from the home pose it sits 0.25 m away, ~30% of random smooth action
#: sequences get closer to it, and the best of 80 came within 0.08 m. That makes
#: it a task a learner has signal on and a planner can actually solve -- a goal
#: outside the reachable set trains nothing, and one at the home pose measures
#: nothing.
DEFAULT_GOAL = np.array([0.30, 0.15, 0.55], dtype=np.float32)


def look_at_quaternion(
    eye: np.ndarray,
    target: np.ndarray,
    up: np.ndarray | None = None,
) -> np.ndarray:
    """Camera orientation as ``(x, y, z, w)`` for a camera at ``eye`` facing ``target``.

    Newton's camera convention is the usual OpenGL one -- forward is local
    ``-z``, up is local ``+y``, right is local ``+x`` -- so the rotation whose
    columns are ``[right, up, -forward]`` aims the camera. Deriving it beats
    hardcoding a quaternion: the framing then follows from where you want to
    look, which is what actually matters for an image-based world model.
    """
    eye = np.asarray(eye, dtype=np.float64)
    forward = np.asarray(target, dtype=np.float64) - eye
    norm = np.linalg.norm(forward)
    if norm < 1e-9:
        raise ValueError("camera position and target coincide")
    forward /= norm
    up = np.asarray(WORLD_UP if up is None else up, dtype=np.float64)
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-6:  # looking straight along `up`
        right = np.cross(forward, np.array([1.0, 0.0, 0.0]))
    right /= np.linalg.norm(right)
    true_up = np.cross(right, forward)

    rotation = np.stack([right, true_up, -forward], axis=1)
    trace = np.trace(rotation)
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (rotation[2, 1] - rotation[1, 2]) / scale
        y = (rotation[0, 2] - rotation[2, 0]) / scale
        z = (rotation[1, 0] - rotation[0, 1]) / scale
    else:  # pick the largest diagonal element for numerical stability
        i = int(np.argmax(np.diag(rotation)))
        j, k = (i + 1) % 3, (i + 2) % 3
        scale = np.sqrt(1.0 + rotation[i, i] - rotation[j, j] - rotation[k, k]) * 2.0
        q = np.zeros(3)
        q[i] = 0.25 * scale
        q[j] = (rotation[j, i] + rotation[i, j]) / scale
        q[k] = (rotation[k, i] + rotation[i, k]) / scale
        w = (rotation[k, j] - rotation[j, k]) / scale
        x, y, z = q
    quaternion = np.array([x, y, z, w], dtype=np.float64)
    return (quaternion / np.linalg.norm(quaternion)).astype(np.float32)


def _require_newton() -> tuple[Any, Any]:
    try:
        import newton
        import warp as wp
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "xwm.envs.newton_franka needs Newton and Warp; install them with "
            '`pip install "xwm[newton]"`'
        ) from exc
    return newton, wp


@dataclass(frozen=True)
class FrankaConfig:
    """Simulation and rendering settings.

    Args:
        image_size: square observation resolution. CPU raytracing cost grows
            with the pixel count, so 64 is a sensible default for training.
        action_scale: radians of joint-target change per unit action. Large
            enough that one step is visible in the image, which matters: if a
            single step barely changes the observation, "predict no change"
            becomes a near-perfect baseline and the benchmark is vacuous.
        fps: control rate.
        substeps: physics substeps per control step. What matters is the
            resulting ``dt / substeps``; Featherstone integration of this arm
            diverges above roughly 1/480 s.
        joint_armature: added rotor inertia. The single most important
            stabiliser here -- without it the solver produces NaNs.
        target_ke, target_kd: joint position-servo gains. The defaults were
            chosen so the arm *holds* its commanded pose (~0.007 rad of drift
            over 12 idle steps) while a commanded sweep still moves the tool
            ~0.65 m. Both halves matter: a servo too weak and the model learns
            gravity instead of the action; too stiff and the solver diverges.
        pose_noise: radians of uniform noise on the initial pose at reset,
            which is what gives the dataset its variety.
        enable_shadows, enable_textures: raytracing quality. Both cost render
            time, which is why they are off by default on CPU; on a GPU they are
            close to free and they add real information to the image -- shading
            disambiguates depth, and texture distinguishes links that are
            otherwise identically white.
        solver: ``"mujoco"``, ``"featherstone"``, or ``"auto"`` to prefer MuJoCo
            (GPU-only, via ``mujoco_warp``) and fall back to Featherstone.
    """

    image_size: int = 64
    action_scale: float = 0.25
    fps: int = 30
    substeps: int = 16
    joint_armature: float = 0.1
    target_ke: float = 800.0
    target_kd: float = 40.0
    pose_noise: float = 0.35
    goal: tuple[float, float, float] = (0.30, 0.15, 0.55)
    action_penalty: float = 0.01
    camera_distance: float = 1.15
    camera_height: float = 0.65
    camera_target_height: float = 0.45
    camera_fov_degrees: float = 60.0
    enable_shadows: bool = False
    enable_textures: bool = False
    solver: str = "auto"
    asset_name: str = "franka_emika_panda"
    urdf_relative_path: str = "urdf/fr3_franka_hand.urdf"


class FrankaEnv:
    """A Franka FR3 arm with RGB observations and joint-space actions.

    Example:
        >>> env = FrankaEnv()
        >>> env.reset(seed=0)
        >>> frame = env.observe()               # (3, H, W) in [0, 1]
        >>> env.step(np.zeros(env.action_dim))  # joint-target deltas in [-1, 1]

    Attributes:
        config: the :class:`FrankaConfig` in use.
        model: the underlying ``newton.Model``, for callers who need it.
    """

    def __init__(self, config: FrankaConfig | None = None, *, urdf_path: str | Path | None = None):
        newton, wp = _require_newton()
        self.config = config or FrankaConfig()
        self._newton, self._wp = newton, wp

        if urdf_path is None:
            import newton.utils

            asset = newton.utils.download_asset(self.config.asset_name)
            urdf_path = Path(asset) / self.config.urdf_relative_path
        urdf_path = Path(urdf_path)
        if not urdf_path.exists():
            raise FileNotFoundError(f"Franka URDF not found at {urdf_path}")

        builder = newton.ModelBuilder()
        builder.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
            armature=self.config.joint_armature,
            limit_ke=1.0e4,
            limit_kd=1.0e2,
        )
        builder.add_urdf(str(urdf_path), floating=False)
        builder.add_ground_plane()
        # Set the servo gains on the builder arrays directly. Passing them
        # through JointDofConfig does *not* reach the arm joints in Newton 1.5 --
        # target_kd silently stays 0, which makes the position servo an
        # undamped spring and sends the Featherstone solver to NaN.
        for i in range(len(builder.joint_target_ke)):
            builder.joint_target_ke[i] = self.config.target_ke
            builder.joint_target_kd[i] = self.config.target_kd
        self.model = builder.finalize()
        self._verify_gains()

        self.solver, self.solver_name = self._make_solver()
        self._state_0 = self.model.state()
        self._state_1 = self.model.state()
        self._control = self.model.control()

        self._n_dofs = int(self.model.joint_dof_count)
        limits_lower = np.asarray(self.model.joint_limit_lower.numpy(), dtype=np.float32)
        limits_upper = np.asarray(self.model.joint_limit_upper.numpy(), dtype=np.float32)
        self.joint_limit_lower = limits_lower[: self._n_dofs]
        self.joint_limit_upper = limits_upper[: self._n_dofs]

        self.tool_body_index = self._find_tool_body()
        self._setup_camera()
        self._target = np.zeros(self._n_dofs, dtype=np.float32)
        self.reset(seed=0)

    def _make_solver(self):
        """Pick a solver, preferring MuJoCo where it is available.

        MuJoCo (through ``mujoco_warp``) is the better-conditioned articulated
        solver and needs a CUDA GPU; Featherstone runs anywhere. ``"auto"``
        tries MuJoCo and falls back, so the same code runs on a laptop and on a
        GPU box without a flag change.
        """
        newton = self._newton
        wanted = self.config.solver
        if wanted not in ("auto", "mujoco", "featherstone"):
            raise ValueError(f"unknown solver {wanted!r}")

        self.solver_fallback_reason: str | None = None
        if wanted in ("auto", "mujoco"):
            try:
                return newton.solvers.SolverMuJoCo(self.model), "mujoco"
            except Exception as exc:
                reason = f"{type(exc).__name__}: {exc}"
                if wanted == "mujoco":
                    raise RuntimeError(
                        "SolverMuJoCo is unavailable (it needs a CUDA GPU and the "
                        f"mujoco_warp backend): {reason}"
                    ) from exc
                # Say why. A silent downgrade to a different solver changes the
                # physics of every result that follows.
                self.solver_fallback_reason = reason
                warnings.warn(
                    f"SolverMuJoCo unavailable, falling back to Featherstone -- {reason}",
                    RuntimeWarning,
                    stacklevel=2,
                )
        return newton.solvers.SolverFeatherstone(self.model), "featherstone"

    def _verify_gains(self) -> None:
        """Fail loudly if the servo gains did not reach the model."""
        kd = np.asarray(self.model.joint_target_kd.numpy())
        if self.config.target_kd > 0 and not np.any(kd[: self.action_dim] > 0):
            raise RuntimeError(
                "joint_target_kd did not reach the model; the position servo would be "
                "undamped and the solver would diverge"
            )

    def _find_tool_body(self) -> int:
        """Index of the tool-centre-point body, by label rather than guesswork."""
        labels = [str(x) for x in getattr(self.model, "body_label", [])]
        for needle in ("hand_tcp", "hand", "link7"):
            for i, label in enumerate(labels):
                if needle in label:
                    return i
        return self.model.body_count - 1

    # -- setup ---------------------------------------------------------------
    def _setup_camera(self) -> None:
        from newton.sensors import SensorTiledCamera

        wp = self._wp
        self._camera = SensorTiledCamera(model=self.model)
        self._camera.default_render_config.enable_shadows = self.config.enable_shadows
        self._camera.default_render_config.enable_textures = self.config.enable_textures
        self._camera.utils.create_default_light(enable_shadows=self.config.enable_shadows)
        # Ray grids and output buffers are cached per resolution, so rendering a
        # high-resolution frame for a figure costs one extra buffer rather than a
        # second environment.
        self._targets: dict[int, tuple] = {}
        # Aim the camera at the workspace rather than at the horizon. Framing
        # is not cosmetic here: with the arm small in the frame, one control
        # step moves only a handful of pixels, "predict no change" becomes an
        # excellent baseline, and there is almost nothing for a predictive model
        # to learn.
        eye, target = self.camera_framing
        qx, qy, qz, qw = (float(v) for v in look_at_quaternion(eye, target))
        pose = wp.transformf(wp.vec3f(*eye.tolist()), wp.quatf(qx, qy, qz, qw))
        self._camera_transform = wp.array([[pose]], dtype=wp.transformf)
        self._clear = SensorTiledCamera.GRAY_CLEAR_DATA

    @property
    def camera_framing(self) -> tuple[np.ndarray, np.ndarray]:
        """``(eye, target)`` in world metres, shared by every renderer.

        One definition so that a path-traced figure and the observations the
        model trains on show the same view from the same place.
        """
        eye = np.array(
            [self.config.camera_distance, 0.0, self.config.camera_height], dtype=np.float32
        )
        target = np.array([0.0, 0.0, self.config.camera_target_height], dtype=np.float32)
        return eye, target

    def _render_target(self, size: int):
        """Cached ``(rays, colour buffer)`` for a square render of ``size``."""
        if size not in self._targets:
            if size < 8:
                raise ValueError(f"render size must be at least 8, got {size}")
            utils = self._camera.utils
            self._targets[size] = (
                utils.compute_camera_rays_pinhole(
                    size, size, camera_fovs=math.radians(self.config.camera_fov_degrees)
                ),
                utils.create_color_image_output(size, size, 1),
            )
        return self._targets[size]

    # -- properties ----------------------------------------------------------
    @property
    def action_dim(self) -> int:
        """Number of controlled arm joints (the gripper is held fixed)."""
        return N_ARM_JOINTS

    @property
    def observation_shape(self) -> tuple[int, int, int]:
        return (3, self.config.image_size, self.config.image_size)

    @property
    def dt(self) -> float:
        return 1.0 / self.config.fps

    # -- dynamics ------------------------------------------------------------
    def reset(self, seed: int | None = None) -> np.ndarray:
        """Reset to a randomly perturbed home pose. Returns the observation."""
        newton, wp = self._newton, self._wp
        rng = np.random.default_rng(seed)
        q = np.zeros(self._n_dofs, dtype=np.float32)
        n = min(N_ARM_JOINTS, self._n_dofs)
        q[:n] = HOME_POSE[:n] + rng.uniform(
            -self.config.pose_noise, self.config.pose_noise, size=n
        ).astype(np.float32)
        q = np.clip(q, self.joint_limit_lower, self.joint_limit_upper)

        self._state_0.joint_q.assign(wp.array(q, dtype=wp.float32))
        self._state_0.joint_qd.assign(wp.zeros(self._n_dofs, dtype=wp.float32))
        newton.eval_fk(self.model, self._state_0.joint_q, self._state_0.joint_qd, self._state_0)
        self._target = q.copy()
        self._control.joint_target_q = wp.array(self._target, dtype=wp.float32)
        return self.observe()

    def step(self, action: np.ndarray) -> np.ndarray:
        """Apply one control step. Returns the resulting observation.

        Args:
            action: ``(7,)`` in ``[-1, 1]``; interpreted as a *delta* on the
                joint position targets, scaled by ``config.action_scale`` and
                clipped to the URDF joint limits. Deltas rather than absolute
                targets so the action distribution is state-independent, which
                is what makes a learned ``(z, a) -> z'`` well-posed.
        """
        wp = self._wp
        action = np.clip(np.asarray(action, dtype=np.float32).reshape(-1), -1.0, 1.0)
        if action.shape[0] != self.action_dim:
            raise ValueError(f"expected {self.action_dim} action dims, got {action.shape[0]}")

        n = min(N_ARM_JOINTS, self._n_dofs)
        self._target[:n] = np.clip(
            self._target[:n] + self.config.action_scale * action[:n],
            self.joint_limit_lower[:n],
            self.joint_limit_upper[:n],
        )
        self._control.joint_target_q = wp.array(self._target, dtype=wp.float32)

        sub_dt = self.dt / self.config.substeps
        for _ in range(self.config.substeps):
            self._state_0.clear_forces()
            self.solver.step(self._state_0, self._state_1, self._control, None, sub_dt)
            self._state_0, self._state_1 = self._state_1, self._state_0
        return self.observe()

    # -- observation ---------------------------------------------------------
    def observe(self) -> np.ndarray:
        """Render the current state to ``(3, H, W)`` float32 in ``[0, 1]``.

        Uses ``config.image_size`` -- the resolution the model is trained on.
        """
        return self.render()

    def render(self, size: int | None = None, *, samples: int = 1) -> np.ndarray:
        """Render at any resolution, ``(3, size, size)`` float32 in ``[0, 1]``.

        Pass a larger ``size`` for figures: upscaling a training frame turns
        every pixel into a block and adds no detail, while the raytracer will
        render at whatever resolution you ask for.

        Args:
            size: output resolution. Defaults to ``config.image_size``.
            samples: supersampling factor. The Warp raytracer casts one ray per
                pixel, so silhouettes and shadow boundaries come out as hard
                staircases; rendering at ``samples x`` and averaging down fixes
                that at ``samples ** 2`` the cost. Leave at 1 for training
                observations, use 2-3 for figures.

        For photorealistic output -- soft shadows, ambient occlusion, materials --
        see :class:`xwm.envs.HighQualityRenderer`, which drives Newton's OVRTX
        path tracer or exports a USD stage.
        """
        size = self.config.image_size if size is None else int(size)
        if samples < 1:
            raise ValueError(f"samples must be at least 1, got {samples}")
        if samples > 1:
            from .render import supersample

            return supersample(self._render_once(size * samples), samples)
        return self._render_once(size)

    def _render_once(self, size: int) -> np.ndarray:
        rays, color = self._render_target(size)
        self.model.bvh_refit_shapes(self._state_0)
        self._camera.update(
            self._state_0,
            self._camera_transform,
            rays,
            color_image=color,
            clear_data=self._clear,
        )
        rgba = np.array(self._camera.utils.to_rgba_from_color(color).numpy(), copy=True)
        frame = rgba.reshape(-1, size, size, 4)[0, ..., :3]
        return np.ascontiguousarray(frame.transpose(2, 0, 1), dtype=np.float32) / 255.0

    # -- ground truth (evaluation only) --------------------------------------
    #
    # Every accessor below returns a *copy*. Warp's ``.numpy()`` hands back a
    # view onto live device memory, so a value read now and kept for later
    # silently mutates as the simulation advances -- collecting a trajectory
    # would yield N references to the final state rather than N states. That
    # failure is invisible to shape and finiteness checks, so it is copied here
    # once and asserted in the tests.
    def joint_positions(self) -> np.ndarray:
        """``(7,)`` arm joint angles. For evaluation -- the model never sees these."""
        q = np.array(self._state_0.joint_q.numpy(), dtype=np.float32, copy=True)
        return q[: self.action_dim]

    def body_positions(self) -> np.ndarray:
        """``(n_bodies, 3)`` world-frame body origins."""
        q = np.array(self._state_0.body_q.numpy(), dtype=np.float32, copy=True)
        return q[:, :3]

    def tool_position(self) -> np.ndarray:
        """``(3,)`` world position of the tool centre point (the hand)."""
        return np.array(self.body_positions()[self.tool_body_index], copy=True)

    def state_observation(self) -> np.ndarray:
        """``(20,)`` proprioceptive observation: joints, velocities, tool, goal delta.

        The cheap alternative to pixels. TD-MPC2 and MuZero on state converge in
        minutes rather than hours, which makes them testable; swap in an image
        encoder once the pipeline is known to work.
        """
        q = self.joint_positions()
        qd = np.array(self._state_0.joint_qd.numpy(), dtype=np.float32, copy=True)
        return np.concatenate(
            [q, qd[: self.action_dim], self.tool_position(), self.tool_position() - self.goal]
        ).astype(np.float32)

    @property
    def state_dim(self) -> int:
        """Length of :meth:`state_observation`: joints, velocities, tool, delta."""
        return 2 * self.action_dim + 6

    def is_finite(self) -> bool:
        """Whether the simulation is still numerically healthy."""
        return bool(np.all(np.isfinite(np.array(self._state_0.joint_q.numpy(), copy=True))))

    def high_quality_renderer(self, **kwargs):
        """A :class:`xwm.envs.HighQualityRenderer` bound to this environment's model.

        Feed it simulation states as the episode runs::

            with env.high_quality_renderer(backend="usd",
                                           output_path="episode.usd") as renderer:
                env.reset(seed=0)
                for action in actions:
                    env.step(action)
                    renderer.add(env.state)
        """
        from .render import HighQualityRenderer

        kwargs.setdefault("fps", self.config.fps)
        kwargs.setdefault("up_axis", "XYZ"[int(np.argmax(WORLD_UP))])
        renderer = HighQualityRenderer(self.model, **kwargs)
        renderer.look_at(*self.camera_framing)
        return renderer

    @property
    def state(self):
        """The current ``newton.State``, for a renderer or a custom sensor."""
        return self._state_0

    # -- task ----------------------------------------------------------------
    @property
    def goal(self) -> np.ndarray:
        """``(3,)`` reach target for the tool."""
        return np.asarray(self.config.goal, dtype=np.float32)

    def goal_distance(self) -> float:
        """Metres from the tool to the goal. Ground truth, for evaluation."""
        return float(np.linalg.norm(self.tool_position() - self.goal))

    def reward(self, action: np.ndarray | None = None) -> float:
        """Dense reach reward in roughly ``[-1, 0]``, minus an action penalty.

        ``-tanh(distance)`` rather than ``-distance``: a bounded reward keeps the
        value function inside the categorical head's bin range without per-task
        tuning, and its gradient does not vanish far from the goal the way a
        squared distance's does.
        """
        shaped = -float(np.tanh(self.goal_distance()))
        if action is None or self.config.action_penalty == 0.0:
            return shaped
        cost = self.config.action_penalty * float(np.mean(np.square(action)))
        return shaped - cost

    # -- trajectories --------------------------------------------------------
    def rollout(self, actions: np.ndarray, *, seed: int | None = None) -> dict[str, np.ndarray]:
        """Execute an action sequence from a fresh reset.

        Args:
            actions: ``(T, 7)``.

        Returns:
            ``{"video": (T + 1, 3, H, W), "joint_q": (T + 1, 7),
            "tool": (T + 1, 3)}`` -- one more observation than actions, since
            the initial frame precedes the first action.
        """
        actions = np.asarray(actions, dtype=np.float32)
        frames = [self.reset(seed=seed)]
        joints = [self.joint_positions()]
        tools = [self.tool_position()]
        states = [self.state_observation()]
        rewards = []
        for action in actions:
            frames.append(self.step(action))
            joints.append(self.joint_positions())
            tools.append(self.tool_position())
            states.append(self.state_observation())
            rewards.append(self.reward(action))
        return {
            "video": np.stack(frames),
            "joint_q": np.stack(joints),
            "tool": np.stack(tools),
            "state": np.stack(states),
            "reward": np.asarray(rewards, dtype=np.float32),
        }


def smooth_actions(
    rng: np.random.Generator,
    n_sequences: int,
    length: int,
    action_dim: int,
    *,
    smoothness: float = 0.7,
) -> np.ndarray:
    """``(n, length, action_dim)`` temporally correlated actions in ``[-1, 1]``.

    White noise makes an arm jitter in place and go nowhere; an AR(1) process
    produces trajectories that actually sweep through the workspace.
    """
    noise = rng.uniform(-1.0, 1.0, size=(n_sequences, length, action_dim)).astype(np.float32)
    out = np.empty_like(noise)
    carry = noise[:, 0]
    for t in range(length):
        carry = smoothness * carry + (1.0 - smoothness) * noise[:, t]
        out[:, t] = carry
    return np.clip(out, -1.0, 1.0)


def franka_sequences(
    env: FrankaEnv,
    n_sequences: int,
    length: int,
    *,
    seed: int = 0,
    smoothness: float = 0.7,
    progress_every: int = 0,
) -> dict[str, np.ndarray]:
    """Collect an action-labelled video dataset from ``env``.

    The returned dict matches what :class:`xwm.action.ActionWorldModel` expects,
    so it is a drop-in replacement for :func:`xwm.data.sprite_sequences`:

    * ``video``: ``(n, length, 3, H, W)``
    * ``action``: ``(n, length - 1, 7)`` -- ``action[i, t]`` joins frames
      ``t`` and ``t + 1``
    * ``joint_q``: ``(n, length, 7)`` ground truth, for probes
    * ``tool``: ``(n, length, 3)`` ground-truth hand position, for probes

    Diverged rollouts (NaN from the solver) are discarded and resampled, so the
    dataset never contains a corrupt sequence.
    """
    rng = np.random.default_rng(seed)
    videos, actions, joints, tools = [], [], [], []
    attempts = 0
    max_attempts = 4 * n_sequences + 16

    while len(videos) < n_sequences and attempts < max_attempts:
        attempts += 1
        action = smooth_actions(rng, 1, length - 1, env.action_dim, smoothness=smoothness)[0]
        result = env.rollout(action, seed=int(rng.integers(0, 2**31 - 1)))
        if not (env.is_finite() and np.all(np.isfinite(result["video"]))):
            continue
        videos.append(result["video"])
        actions.append(action)
        joints.append(result["joint_q"])
        tools.append(result["tool"])
        if progress_every and len(videos) % progress_every == 0:
            print(f"  collected {len(videos)}/{n_sequences} sequences", flush=True)

    if len(videos) < n_sequences:
        raise RuntimeError(
            f"only {len(videos)}/{n_sequences} rollouts stayed finite after {attempts} attempts; "
            "raise substeps or joint_armature in FrankaConfig"
        )
    return {
        "video": np.stack(videos),
        "action": np.stack(actions),
        "joint_q": np.stack(joints),
        "tool": np.stack(tools),
    }

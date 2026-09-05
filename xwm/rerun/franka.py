"""The Franka arm as a 3-D scene: meshes, joints, camera, goal.

Newton ships its own ``newton.viewer.ViewerRerun``, and for an interactive
session it is the better tool -- it draws every collision shape and it knows
about contacts. It is not usable from a library, though: it calls the global
``rr.init`` and, unless handed a server address, spawns or serves a viewer. A
training run on a headless GPU box wants neither. These functions log the same
scene into an explicit :class:`~xwm.rerun.Recording`, from the URDF the
environment already resolved, and write to a file like everything else.
"""

from __future__ import annotations

import math
import re
from hashlib import md5
from pathlib import Path

import numpy as np

from ..tools.cache import cache_dir
from ._backend import rerun

__all__ = ["log_franka_scene", "franka_hook", "log_franka_step", "resolve_urdf"]

#: Where the scene lives in the entity tree.
ROOT = "franka"


def log_franka_scene(recording, env, *, root: str = ROOT) -> None:
    """Log everything about the environment that does not change: mesh, camera, goal.

    All static, so it is present at every point on every timeline. The camera is
    logged as a :class:`rerun.Pinhole` under a :class:`rerun.Transform3D`, both
    derived from ``env.camera_framing`` and ``config.camera_fov_degrees`` -- the
    same definition the training observations are rendered from, so the frustum
    in the 3-D view is the view the model actually sees.
    """
    if recording is None:
        return
    rr = rerun()
    from ..envs.newton_franka import WORLD_UP, look_at_quaternion

    up = "XYZ"[int(np.argmax(np.abs(np.asarray(WORLD_UP))))]
    recording.log(
        root, getattr(rr.ViewCoordinates, f"RIGHT_HAND_{up}_UP"), static=True
    )

    tree = _urdf_tree(env, root)
    if tree is not None:
        # Rerun's own URDF importer: it loads the meshes *and* fixes the frame
        # names that `UrdfJoint.compute_transform` later addresses, so the
        # articulation in `log_franka_step` lands on the geometry logged here.
        tree.log_urdf_to_recording(recording.stream)

    eye, target = env.camera_framing
    quaternion = np.asarray(look_at_quaternion(eye, target), np.float32)
    size = int(env.config.image_size)
    recording.log(
        f"{root}/camera",
        rr.Transform3D(translation=np.asarray(eye, np.float32), quaternion=quaternion),
        static=True,
    )
    # Rerun wants a focal length in pixels, not a field of view, so convert the
    # one the renderer is configured with rather than logging a second number
    # that could drift from it.
    focal = 0.5 * size / math.tan(0.5 * math.radians(env.config.camera_fov_degrees))
    recording.log(
        f"{root}/camera",
        rr.Pinhole(resolution=[size, size], focal_length=[focal, focal]),
        static=True,
    )
    recording.log(
        f"{root}/goal",
        rr.Points3D([np.asarray(env.goal, np.float32)], colors=[[255, 140, 60]], radii=[0.02]),
        static=True,
    )


def log_franka_step(recording, env, *, root: str = ROOT, observe: bool = True) -> None:
    """Log the arm's current pose, its tool, its reach distance and its view.

    Joint angles are sent as URDF joint transforms, so Rerun articulates the
    meshes loaded by :func:`log_franka_scene`. Where the URDF cannot be read the
    body origins are logged as points instead, which still shows the motion.
    """
    if recording is None:
        return
    rr = rerun()
    joints = np.asarray(env.joint_positions(), np.float32)
    tree = _urdf_tree(env, root)
    if tree is not None:
        # One Transform3D per movable joint. Each carries its own parent and
        # child frame ids, so they all go to one entity and Rerun resolves the
        # kinematic chain itself.
        for angle, joint in zip(joints, _movable_joints(tree), strict=False):
            recording.log(f"{root}/joints", joint.compute_transform(float(angle)))
    else:
        recording.log(
            f"{root}/bodies",
            rr.Points3D(np.asarray(env.body_positions(), np.float32), radii=[0.015]),
        )
    recording.log(
        f"{root}/tool",
        rr.Points3D(
            [np.asarray(env.tool_position(), np.float32)],
            colors=[[80, 160, 255]],
            radii=[0.02],
        ),
    )
    recording.log(f"{root}/signal/goal_distance", rr.Scalars(float(env.goal_distance())))
    for i, angle in enumerate(joints):
        recording.log(f"{root}/signal/joint/{i}", rr.Scalars(float(angle)))
    if observe:
        from ..plots.export import frames_to_uint8

        recording.log(f"{root}/camera/image", rr.Image(frames_to_uint8(env.render()[None])[0]))


def franka_hook(recording, env, *, root: str = ROOT, timeline: str = "env_step", observe=True):
    """A ``record()``-shaped closure that logs one frame per call.

    Shaped to drop into the callbacks the examples already use --
    ``examples/_common.py``'s ``play(record)`` and
    :func:`xwm.bench.run_episode`'s ``on_step`` both call something once per
    step. It owns its own counter, so nothing has to pass it a time.
    """
    if recording is None:
        return lambda *args, **kwargs: None
    log_franka_scene(recording, env, root=root)
    count = {"t": 0}

    def hook(*args, **kwargs) -> None:
        recording.set_time(timeline, count["t"])
        count["t"] += 1
        log_franka_step(recording, env, root=root, observe=observe)

    return hook


def _urdf_tree(env, root: str = ROOT):
    """Rerun's parsed URDF for this environment, or ``None`` if it cannot be read.

    Cached on the environment, because parsing the file and resolving its meshes
    is slow enough to matter once per logged frame.
    """
    cached = getattr(env, "_xwm_urdf_tree", "missing")
    if cached != "missing":
        return cached
    tree = None
    path = getattr(env, "urdf_path", None)
    if path is not None:
        try:
            tree = rerun().urdf.UrdfTree.from_file_path(
                str(resolve_urdf(path)), entity_path_prefix=root
            )
        except Exception:  # pragma: no cover - a viewer aid must not stop a run
            tree = None
    try:
        env._xwm_urdf_tree = tree
    except AttributeError:  # pragma: no cover - a frozen env
        pass
    return tree


def resolve_urdf(urdf: Path) -> Path:
    """A copy of ``urdf`` whose ``package://`` meshes point at real files.

    The Franka URDF addresses its meshes as
    ``package://franka_emika_panda/meshes/...``, the ROS convention. Newton's
    asset cache holds exactly that layout, but not the ``package.xml`` manifest
    a ROS resolver looks for, so ``ROS_PACKAGE_PATH`` does not rescue it and
    Rerun loads an arm with no geometry at all -- an empty 3-D view, logged
    without an error.

    The rewrite is a copy in xwm's own cache rather than an edit in place:
    Newton owns that directory and re-downloads it, and a library that mutates
    another one's cache is a bug waiting for a version bump. Meshes keep their
    original location, so the textures they reference relatively still resolve.
    Returns the original path unchanged when there is nothing to rewrite.
    """
    urdf = Path(urdf).resolve()
    text = urdf.read_text()
    if "package://" not in text:
        return urdf
    root = urdf.parent.parent.parent  # <cache>/<asset>/<package>/urdf/x.urdf

    def absolute(match: re.Match) -> str:
        target = root / match.group(1)
        return str(target) if target.exists() else match.group(0)

    rewritten = re.sub(r"package://([^\"\'\s>]+)", absolute, text)
    if rewritten == text:  # nothing resolved: leave the original to warn honestly
        return urdf
    out = cache_dir("urdf") / f"{urdf.stem}-{md5(str(urdf).encode()).hexdigest()[:8]}.urdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    if not out.exists() or out.read_text() != rewritten:
        out.write_text(rewritten)
    return out


def _movable_joints(tree):
    """The tree's joints that have an angle, in URDF order."""
    joints = getattr(tree, "joints", None)
    joints = list(joints() if callable(joints) else joints or [])
    return [j for j in joints if getattr(j, "joint_type", "") not in ("fixed",)]

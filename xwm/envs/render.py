"""Two rendering paths, for two different jobs.

A world model and a figure want opposite things from a renderer. Training needs
thousands of frames per second at 64-128 px and does not care how they look;
a figure needs one frame that looks right. Newton exposes both, and forcing one
renderer to do both jobs is what makes observations slow *and* figures ugly.

============  ==================================================================
``"warp"``    The Warp raytracer (``SensorTiledCamera``). Returns arrays
              in-process, runs on CPU or GPU, milliseconds per frame. This is
              what :meth:`~xwm.envs.FrankaEnv.observe` uses, and what a training
              loop should always use. Supports supersampling for figures that
              need to be merely *clean* rather than photoreal.
``"rtx"``     ``ViewerRTX``, NVIDIA's OVRTX path tracer, headless. Soft shadows,
              ambient occlusion, real materials and studio lighting -- the
              quality of the renders in Newton's own documentation. Needs an
              NVIDIA GPU and ``pip install ovrtx``.
``"usd"``     ``ViewerUSD``. Writes the scene to a USD stage instead of pixels,
              to be rendered offline in Omniverse, Blender or ``usdview`` at
              whatever quality you like. Needs ``pip install usd-core``, works
              headless on any platform, and is the portable route to a
              publication figure.
============  ==================================================================

The backends deliberately do not share a return type: ``"warp"`` and ``"rtx"``
hand back frames, ``"usd"`` writes a file. Pretending otherwise would mean
buffering an entire animation in memory to look uniform.

References
----------
Newton's viewer backends: https://newton-physics.github.io/newton/
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np

__all__ = [
    "RENDER_BACKENDS",
    "HighQualityRenderer",
    "look_at_angles",
    "supersample",
    "which_backends",
]

Backend = Literal["warp", "rtx", "usd"]

#: Backend -> the package it needs beyond Newton itself.
RENDER_BACKENDS: dict[str, str | None] = {"warp": None, "rtx": "ovrtx", "usd": "pxr"}


def which_backends() -> dict[str, bool]:
    """Which rendering backends are usable here.

    Worth calling before a long run: discovering that the high-quality path is
    unavailable after four hours of simulation is a poor use of a GPU.
    """
    import importlib.util

    available = {}
    for backend, requirement in RENDER_BACKENDS.items():
        available[backend] = requirement is None or (
            importlib.util.find_spec(requirement) is not None
        )
    return available


def supersample(frame: np.ndarray, factor: int) -> np.ndarray:
    """Box-downsample a ``(3, H*f, W*f)`` frame by ``factor``.

    Rendering above the target resolution and averaging down is the cheapest
    anti-aliasing there is, and the Warp raytracer has no built-in
    multisampling: one ray per pixel means every silhouette is a hard staircase.
    At ``factor=3`` each output pixel integrates nine rays, which is enough to
    make edges and shadow boundaries read as smooth.
    """
    if factor <= 1:
        return frame
    channels, height, width = frame.shape
    if height % factor or width % factor:
        raise ValueError(f"frame {height}x{width} is not divisible by factor {factor}")
    reshaped = frame.reshape(channels, height // factor, factor, width // factor, factor)
    return reshaped.mean(axis=(2, 4))


def look_at_angles(eye, target, up_axis: str = "Z") -> tuple[float, float]:
    """``(pitch, yaw)`` in degrees for a viewer camera at ``eye`` facing ``target``.

    Newton's viewer camera is parameterised by position, pitch and yaw rather
    than by a look-at target. For a Z-up scene its forward vector is
    ``(cos yaw cos pitch, sin yaw cos pitch, sin pitch)``, which inverts to
    ``pitch = asin(dz)`` and ``yaw = atan2(dy, dx)``; the other two up-axes
    permute which components play those roles. Deriving the angles beats
    guessing them -- a camera aimed at the horizon puts the robot in a handful
    of pixels, which is a figure of nothing.
    """
    eye = np.asarray(eye, dtype=np.float64).reshape(3)
    direction = np.asarray(target, dtype=np.float64).reshape(3) - eye
    norm = float(np.linalg.norm(direction))
    if norm == 0.0:
        raise ValueError("camera position and target coincide")
    direction /= norm
    vertical = {"X": 0, "Y": 1, "Z": 2}[str(up_axis).upper()]
    # The two ground-plane axes, in the order the viewer's yaw sweeps them.
    first, second = {0: (1, 2), 1: (0, 2), 2: (0, 1)}[vertical]
    pitch = float(np.degrees(np.arcsin(np.clip(direction[vertical], -1.0, 1.0))))
    yaw = float(np.degrees(np.arctan2(direction[second], direction[first])))
    return pitch, yaw


def _to_frame(pixels: np.ndarray) -> np.ndarray:
    """A viewer's ``(H, W, C)`` image as a ``(3, H, W)`` float32 frame in ``[0, 1]``.

    The rest of the library -- :func:`xwm.plots.save_gif`,
    :func:`xwm.plots.plot_frames`, every encoder -- speaks channels-first float,
    so the conversion belongs here rather than at each call site.
    """
    pixels = np.asarray(pixels)
    if pixels.ndim != 3:
        raise ValueError(f"expected an (H, W, C) image, got shape {pixels.shape}")
    frame = pixels[..., :3]  # drop alpha; the viewer renders opaque
    if not np.issubdtype(frame.dtype, np.floating):
        frame = frame.astype(np.float32) / 255.0
    return np.ascontiguousarray(np.transpose(frame, (2, 0, 1)), dtype=np.float32)


class HighQualityRenderer:
    """Render or export a trajectory with Newton's high-quality viewers.

    Args:
        model: the ``newton.Model`` to render.
        backend: ``"rtx"`` for path-traced frames, ``"usd"`` for a USD stage.
        size: output resolution (``"rtx"`` only).
        output_path: destination stage (``"usd"`` only).
        environment: OVRTX lighting environment -- ``"studio"`` is the lit
            backdrop used for presentation renders.
        fps: playback rate recorded in the output.

    Example:
        >>> renderer = HighQualityRenderer(env.model, backend="usd",
        ...                                output_path="episode.usd")
        >>> for state in states:
        ...     renderer.add(state)
        >>> renderer.close()
    """

    def __init__(
        self,
        model,
        *,
        backend: Backend = "usd",
        size: tuple[int, int] = (1280, 720),
        output_path: str | Path | None = None,
        environment: str = "studio",
        fps: int = 30,
        up_axis: str = "Z",
    ):
        if backend not in ("rtx", "usd"):
            raise ValueError(
                f"backend must be 'rtx' or 'usd' (got {backend!r}); for fast "
                "in-process frames use FrankaEnv.render, which is the 'warp' path"
            )
        requirement = RENDER_BACKENDS[backend]
        if not which_backends()[backend]:
            raise ImportError(
                f"the {backend!r} backend needs `pip install "
                f"{'ovrtx' if backend == 'rtx' else 'usd-core'}`"
                + (" and an NVIDIA GPU" if backend == "rtx" else "")
                + f" (missing module {requirement!r})"
            )

        import newton.viewer as viewer

        self.backend = backend
        self.size = size
        self.fps = fps
        self.up_axis = str(up_axis).upper()
        self._time = 0.0
        self._frames: list[np.ndarray] = []

        if backend == "rtx":
            self._viewer = viewer.ViewerRTX(
                width=size[0],
                height=size[1],
                headless=True,
                up_axis=up_axis,
                environment=environment,
                async_rendering=False,
            )
        else:
            if output_path is None:
                raise ValueError("the 'usd' backend needs an output_path")
            self.output_path = Path(output_path)
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            self._viewer = viewer.ViewerUSD(
                output_path=str(self.output_path), fps=fps, up_axis=up_axis
            )
        self._viewer.set_model(model)

    def set_camera(self, *args, **kwargs) -> None:
        """Forwarded to the viewer, where supported."""
        setter = getattr(self._viewer, "set_camera", None)
        if setter is not None:
            setter(*args, **kwargs)

    def look_at(self, eye, target) -> None:
        """Aim the camera from ``eye`` at ``target``, both in world metres.

        Without this the path-traced view points at the horizon while the
        training camera looks at the arm, and the two renders are not of the
        same scene. See :func:`look_at_angles` for the conversion.
        """
        pitch, yaw = look_at_angles(eye, target, self.up_axis)
        eye = np.asarray(eye, dtype=np.float64).reshape(3)
        self.set_camera(tuple(float(v) for v in eye), float(pitch), float(yaw))

    def add(self, state) -> None:
        """Record one simulation state as a frame."""
        self._viewer.begin_frame(self._time)
        self._viewer.log_state(state)
        self._viewer.end_frame()
        self._time += 1.0 / self.fps
        if self.backend == "rtx":
            self._frames.append(self._capture())

    def _capture(self) -> np.ndarray:
        """The frame just rendered, as ``(3, H, W)`` float32.

        ``ViewerRTX`` offers pixels two ways: an in-memory grab of the OVRTX
        colour output, and ``save_screenshot``. Prefer the former -- a PNG
        encode-decode per frame is pure overhead on a path tracer -- but it is a
        private entry point, so fall back to the documented one rather than
        break outright on a Newton upgrade.
        """
        grab = getattr(self._viewer, "_capture_screenshot_pixels", None)
        if grab is not None:
            return _to_frame(grab())

        import tempfile

        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "frame.png"
            self._viewer.save_screenshot(str(path))
            return _to_frame(Image.open(path))

    @property
    def frames(self) -> np.ndarray:
        """Captured frames as ``(T, 3, H, W)`` float32 -- the ``"rtx"`` path only.

        Feeds :func:`xwm.plots.save_gif` and :func:`xwm.plots.plot_frames`
        directly, so a path-traced episode is written exactly like a Warp one.
        """
        if self.backend != "rtx":
            raise AttributeError(
                f"the {self.backend!r} backend writes a file, not frames; "
                "see close() for the path it wrote"
            )
        if not self._frames:
            raise RuntimeError("no frames captured yet -- call add() first")
        return np.stack(self._frames)

    def close(self) -> Path | None:
        """Finish the output. Returns the written path for ``"usd"``."""
        self._viewer.close()
        return getattr(self, "output_path", None)

    def __enter__(self) -> HighQualityRenderer:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

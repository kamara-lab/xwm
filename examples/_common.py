"""Shared plumbing for the examples: output paths and the plot style.

Every example writes its artifacts to ``examples/outputs/<name>/``:

* ``*.png`` -- figures
* ``*.gif`` -- animations
* ``*.json`` -- the numbers, at full precision, for anything that reads them back
* ``*.tex`` -- the same numbers formatted for a paper
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TypeVar

import xwm

T = TypeVar("T")

#: Where artifacts are written. Overridable with ``XWM_OUTPUT_ROOT`` so a remote
#: launcher (see ``deploy/modal_app.py``) can point it at a mounted volume
#: without the examples knowing they are not running locally.
OUTPUT_ROOT = Path(
    os.environ.get("XWM_OUTPUT_ROOT") or Path(__file__).resolve().parent / "outputs"
)


def setting(name: str, default: T, cast=None) -> T:
    """Read ``XWM_<NAME>`` from the environment, falling back to ``default``.

    Lets one example script run at laptop fidelity by default and at GPU
    fidelity when a launcher raises the knobs, without a second copy of the
    pipeline to keep in sync.
    """
    raw = os.environ.get(f"XWM_{name.upper()}")
    if raw is None:
        return default
    if cast is None:
        cast = type(default)
    if cast is bool:
        return raw.strip().lower() in {"1", "true", "yes", "on"}  # type: ignore[return-value]
    return cast(raw)


def describe_settings(names: dict[str, object]) -> None:
    """Print the resolved knobs, so a run's output records how it was produced."""
    body = "  ".join(f"{k}={v}" for k, v in names.items())
    print(f"settings: {body}\n", flush=True)


def outputs(name: str) -> Path:
    """Create and return ``examples/outputs/<name>/``."""
    path = OUTPUT_ROOT / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def setup(name: str, *, n_series: int = xwm.plots.MAX_CATEGORICAL) -> Path:
    """Apply the viridis style globally and return this example's output directory."""
    import matplotlib

    matplotlib.use("Agg")  # examples run headless
    xwm.plots.use_viridis(n_series)
    return outputs(name)


def figure_episode(
    env,
    out: Path,
    name: str = "episode",
    *,
    policy=None,
    actions=None,
    steps: int | None = None,
    seed: int = 0,
    fps: int = 5,
    usd_name: str | None = None,
    label: str = "Franka FR3 in Newton",
):
    """Render one episode at figure quality and write it as a GIF and a strip.

    The renderer is chosen at run time, not assumed: OVRTX path tracing where the
    host can do it (soft shadows, ambient occlusion, real materials), and
    otherwise the Warp raytracer at ``XWM_RTX_SIZE`` with ``XWM_RTX_SAMPLES``x
    supersampling, plus a USD stage that can be path traced offline. A
    compute-only GPU container cannot run OVRTX at all -- see
    ``docs/findings.md`` -- so every Franka example has to degrade rather than
    fail.

    Drive the episode either open loop with ``actions``, or closed loop with
    ``policy(env) -> action`` and ``steps``. In the closed-loop case the actions
    taken are recorded on the first pass and replayed for the others, so a
    stochastic policy still yields one episode across every renderer rather than
    three similar-looking ones.

    Returns the frames as ``(T, 3, H, W)``.
    """
    import numpy as np

    if (policy is None) == (actions is None):
        raise ValueError("pass exactly one of policy= or actions=")
    if policy is not None and steps is None:
        raise ValueError("policy= needs steps=")

    enabled = setting("RTX", 1)
    size = setting("RTX_SIZE", 512)
    samples = setting("RTX_SAMPLES", 3)
    dpi = setting("FIGURE_DPI", 200)
    max_panels = setting("STRIP_PANELS", 8)
    usd_name = name if usd_name is None else usd_name

    taken: list[np.ndarray] = []

    def play(record, replay=None):
        """Run the episode once, calling ``record()`` after reset and each step."""
        env.reset(seed=seed)
        record()
        source = replay if replay is not None else actions
        if source is not None:
            for action in source:
                env.step(np.asarray(action, np.float32))
                record()
        else:
            for _ in range(steps):
                action = np.asarray(policy(env), np.float32)
                taken.append(action)
                env.step(action)
                record()

    frames, kind = None, ""
    backends = xwm.envs.which_backends()
    if enabled and backends["rtx"]:
        try:
            with env.high_quality_renderer(backend="rtx", size=(size, size)) as renderer:
                play(lambda: renderer.add(env.state))
                frames = renderer.frames
            kind = f"{size}x{size} OVRTX path traced"
        except Exception as exc:  # noqa: BLE001 - a figure is not worth the run
            print(f"  (OVRTX failed: {type(exc).__name__}: {exc})")
    elif not enabled:
        print("  (high-quality figures disabled: RTX=0)")

    if frames is None:
        collected: list[np.ndarray] = []
        play(lambda: collected.append(env.render(size, samples=samples)))
        frames = np.stack(collected)
        kind = f"{size}x{size} Warp raytraced, {samples}x supersampled"
        if backends["usd"]:
            with env.high_quality_renderer(
                backend="usd", output_path=out / f"{usd_name}.usd", fps=fps
            ) as renderer:
                play(lambda: renderer.add(env.state), replay=taken or actions)
            report(out / f"{usd_name}.usd")

    report(xwm.plots.save_gif(out / f"{name}.gif", frames, fps=fps))

    # Evenly spaced panels rather than the first N: a 21-step episode truncated
    # to its first 8 frames would show only the opening of the motion.
    picked = np.unique(np.linspace(0, len(frames) - 1, min(max_panels, len(frames))).round())
    picked = picked.astype(int)
    fig = xwm.plots.plot_frames(
        frames[picked],
        titles=[f"t={i}" for i in picked],
        max_frames=len(picked),
        scale=size / dpi,
        suptitle=f"{label}, {kind}",
    )
    report(xwm.plots.save_figure(fig, out / f"{name}_frames.png", dpi=dpi))
    return frames


def report(paths: dict[str, Path] | Path, label: str = "") -> None:
    """Print what was written, relative to the repository root where possible."""
    root = Path(__file__).resolve().parent.parent
    items = paths.values() if isinstance(paths, dict) else [paths]
    for path in items:
        try:
            shown = path.relative_to(root)
        except ValueError:
            shown = path
        print(f"  wrote {shown}" + (f"  ({label})" if label else ""))

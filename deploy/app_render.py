"""Modal app: render the Franka at presentation quality.

Separate from the experiment apps because it answers a different question -- not
"does the model learn?" but "what can this scene look like?". It reports which
rendering backends the container actually has, then produces frames with each
available one so the difference is visible rather than asserted.

Usage::

    modal run deploy/app_render.py                    # all available backends
    modal run deploy/app_render.py --size 1920 --frames 24
    modal run deploy/app_render.py::vulkan_probe        # can OVRTX see a GPU?
"""

from __future__ import annotations

import modal
from _shared import GPU, OUTPUT_DIR, collect_files, download, image, volume_for

EXPERIMENT = "render"

app = modal.App(f"xwm-{EXPERIMENT}", image=image)
volume = volume_for(EXPERIMENT)


@app.function(gpu=GPU, timeout=20 * 60)
def vulkan_probe() -> dict:
    """Can OVRTX get a GPU at all? Answers in a minute instead of in a render.

    OVRTX is a Vulkan application, and a CUDA container is not automatically a
    Vulkan one: it needs the loader (``libvulkan1``) plus an ICD manifest
    pointing at the NVIDIA driver's Vulkan implementation. When either is
    missing the failure surfaces late and unhelpfully, as ``createDevices
    failed`` from inside a scene build, so probe it directly.
    """
    import glob
    import os
    import pathlib
    import subprocess

    info: dict = {}
    info["icd_files"] = sorted(glob.glob("/usr/share/vulkan/icd.d/*")) + sorted(
        glob.glob("/etc/vulkan/icd.d/*")
    )
    info["vk_icd_filenames"] = os.environ.get("VK_ICD_FILENAMES", "<unset>")
    info["driver_capabilities"] = os.environ.get("NVIDIA_DRIVER_CAPABILITIES", "<unset>")
    # Vulkan needs the graphics device nodes, not just the compute ones.
    info["device_nodes"] = sorted(glob.glob("/dev/nvidia*")) + sorted(glob.glob("/dev/dri/*"))
    # The full NVIDIA library set, not just the ones I expect: hardware ray
    # tracing needs the OptiX and RT-core libraries (libnvoptix, libnvidia-rtcore)
    # on top of the Vulkan ICD, and a CUDA-only driver injection ships neither.
    patterns = (
        "libGLX_nvidia.so*",
        "libnvidia-*",
        "libnvoptix*",
        "libvulkan.so*",
        "nvidia_icd*",
        "libEGL_nvidia.so*",
    )
    roots = ("/usr/lib/x86_64-linux-gnu", "/usr/lib64", "/usr/local/lib", "/run/nvidia")
    found: list[str] = []
    for root in roots:
        for pattern in patterns:
            found += glob.glob(f"{root}/**/{pattern}", recursive=True)
    names = sorted({pathlib.Path(path).name for path in found})
    info["driver_libs"] = names
    # The three that decide whether path tracing is possible at all.
    info["ray_tracing_libs"] = {
        need: any(name.startswith(need) for name in names)
        for need in ("libnvoptix", "libnvidia-rtcore", "libnvidia-glvkspirv")
    }

    # The loader rejected libGLX_nvidia.so.0 for not exporting
    # vk_icdGetInstanceProcAddr, so find which library does export it rather
    # than guessing at manifest contents. Scanning the file for the symbol name
    # avoids needing binutils: ELF keeps dynamic symbol names as literal strings.
    exporters = []
    for path in sorted(set(found)):
        try:
            if b"vk_icdGetInstanceProcAddr" in pathlib.Path(path).read_bytes():
                exporters.append(path)
        except OSError:
            continue
    info["vulkan_icd_exporters"] = exporters

    for name, cmd in (
        ("nvidia_smi", ["nvidia-smi", "-L"]),
        ("vulkaninfo", ["vulkaninfo", "--summary"]),
        # Does the loader actually pick up the NVIDIA ICD? The summary above
        # reports what was enumerated; this reports what was tried.
        ("icd_manifest", ["cat", "/usr/share/vulkan/icd.d/nvidia_icd.json"]),
    ):
        try:
            done = subprocess.run(cmd, capture_output=True, text=True, check=False)  # noqa: S603
            text = (done.stdout or done.stderr).strip()
            info[name] = text.splitlines()[:80]
        except FileNotFoundError:
            info[name] = "not installed"

    try:
        import ovrtx

        config = ovrtx.RendererConfig()
        config.log_level = "verbose"
        ovrtx.Renderer(config=config)
        info["ovrtx_renderer"] = "ok"
    except Exception as exc:  # noqa: BLE001 - the whole point is to report it
        info["ovrtx_renderer"] = f"{type(exc).__name__}: {exc}"

    for key, value in info.items():
        print(f"{key}: {value}", flush=True)
    return info


@app.function(gpu=GPU, volumes={OUTPUT_DIR: volume}, timeout=60 * 60)
def render(size: int = 1280, frames: int = 16, samples: int = 3) -> dict:
    import sys

    sys.path.insert(0, "/root")
    from pathlib import Path

    import numpy as np

    import xwm

    out = Path(OUTPUT_DIR) / "render"
    out.mkdir(parents=True, exist_ok=True)

    backends = xwm.envs.which_backends()
    print(f"rendering backends available: {backends}", flush=True)

    env = xwm.envs.FrankaEnv(
        xwm.envs.FrankaConfig(image_size=128, enable_shadows=True, enable_textures=True)
    )
    print(f"solver: {env.solver_name}", flush=True)
    actions = xwm.envs.smooth_actions(
        np.random.default_rng(3), 1, frames, env.action_dim, smoothness=0.5
    )[0]

    report: dict = {"backends": backends, "solver": env.solver_name, "written": []}

    # -- the fast path, supersampled -----------------------------------------
    warp_frames = []
    env.reset(seed=3)
    warp_frames.append(env.render(size // 2, samples=samples))
    for action in actions:
        env.step(action)
        warp_frames.append(env.render(size // 2, samples=samples))
    stack = np.stack(warp_frames)
    xwm.plots.save_gif(out / "warp.gif", stack, fps=8)
    xwm.plots.save_figure(
        xwm.plots.plot_frames(stack, max_frames=6, scale=2.4,
                              suptitle=f"Warp raytracer, {size // 2}px, {samples}x supersampled"),
        out / "warp_frames.png",
    )
    report["written"] += ["warp.gif", "warp_frames.png"]

    # -- the photoreal path ---------------------------------------------------
    if backends.get("rtx"):
        try:
            with env.high_quality_renderer(
                backend="rtx", size=(size, size * 9 // 16), environment="studio"
            ) as renderer:
                env.reset(seed=3)
                renderer.add(env.state)
                for action in actions:
                    env.step(action)
                    renderer.add(env.state)
                frames = renderer.frames
            xwm.plots.save_gif(out / "rtx.gif", frames, fps=8)
            fig = xwm.plots.plot_frames(frames, suptitle="OVRTX path traced")
            xwm.plots.save_figure(fig, out / "rtx_frames.png")
            report["written"] += ["rtx.gif", "rtx_frames.png"]
            report["rtx"] = "ok"
            report["rtx_frame_shape"] = list(frames.shape)
            print(f"rtx render completed: {frames.shape}", flush=True)
        except Exception as exc:  # noqa: BLE001 - report, do not abort the run
            report["rtx"] = f"failed: {type(exc).__name__}: {exc}"
            print(f"rtx render failed: {exc}", flush=True)
    else:
        report["rtx"] = "unavailable"

    if backends.get("usd"):
        with env.high_quality_renderer(
            backend="usd", output_path=out / "franka.usd", fps=8
        ) as renderer:
            env.reset(seed=3)
            renderer.add(env.state)
            for action in actions:
                env.step(action)
                renderer.add(env.state)
        report["usd"] = "ok"
        report["written"].append("franka.usd")
    else:
        report["usd"] = "unavailable"

    xwm.plots.save_metrics(out / "render_report", report)
    volume.commit()
    return report


@app.function(volumes={OUTPUT_DIR: volume}, timeout=30 * 60)
def collect() -> dict[str, bytes]:
    return collect_files()


@app.local_entrypoint()
def main(size: int = 1280, frames: int = 16, samples: int = 3, fetch: bool = True):
    report = render.remote(size, frames, samples)
    print("\n=== render report ===")
    for key, value in report.items():
        print(f"  {key}: {value}")
    if fetch:
        print(f"\ndownloaded {download(collect.remote())} files into examples/outputs/")

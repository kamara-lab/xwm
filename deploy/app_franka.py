"""Modal app: Franka FR3 world model in Newton.

One app per experiment, so the six can run concurrently on separate GPUs and be
started, watched and stopped independently.

Usage::

    modal run deploy/app_franka.py                     # gpu preset
    modal run deploy/app_franka.py --preset xl
    modal run deploy/app_franka.py --preset cpu-parity # laptop knobs, on a GPU
"""

from __future__ import annotations

import modal
from _shared import GPU, OUTPUT_DIR, collect_files, download, execute, image, volume_for

EXPERIMENT = "franka"

app = modal.App(f"xwm-{EXPERIMENT}", image=image)
volume = volume_for(EXPERIMENT)


@app.function(gpu=GPU, volumes={OUTPUT_DIR: volume}, timeout=6 * 60 * 60)
def run(preset: str = "gpu") -> dict:
    result = execute(EXPERIMENT, preset)
    volume.commit()
    return result


@app.function(volumes={OUTPUT_DIR: volume}, timeout=30 * 60)
def collect() -> dict[str, bytes]:
    return collect_files()


@app.local_entrypoint()
def main(preset: str = "gpu", fetch: bool = True):
    result = run.remote(preset)
    status = "ok" if result["returncode"] == 0 else f"FAILED ({result['returncode']})"
    print(f"\n{EXPERIMENT}: {status} in {result['seconds']:.1f}s")
    if fetch:
        print(f"downloaded {download(collect.remote())} files into examples/outputs/")

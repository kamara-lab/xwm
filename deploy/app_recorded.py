"""Modal app: train on a recorded dataset, against its synthetic twin.

The only app whose data comes off the network rather than out of a simulator, so
it carries a second volume: the dataset cache. Without it every run re-downloads
Push-T, and a run on DROID or LIBERO would re-download tens of gigabytes.

Usage::

    modal run deploy/app_recorded.py                        # gpu-recorded preset
    modal run deploy/app_recorded.py --preset cpu-parity    # laptop knobs, on a GPU
    modal run deploy/app_recorded.py --preset gpu-recorded --dataset libero/10
"""

from __future__ import annotations

import modal
from _shared import GPU, OUTPUT_DIR, collect_files, download, execute, image, volume_for

EXPERIMENT = "recorded"
CACHE_DIR = "/data"

app = modal.App(f"xwm-{EXPERIMENT}", image=image)
volume = volume_for(EXPERIMENT)
#: Datasets are immutable once downloaded and shared across runs, so they live in
#: their own volume rather than beside the artifacts -- `collect_files` globs the
#: output volume, and a 30 MB dataset in there would come back down with the
#: figures every time.
cache = modal.Volume.from_name("xwm-dataset-cache", create_if_missing=True)


@app.function(
    gpu=GPU,
    volumes={OUTPUT_DIR: volume, CACHE_DIR: cache},
    timeout=6 * 60 * 60,
)
def run(preset: str = "gpu-recorded", dataset: str | None = None) -> dict:
    import os

    os.environ["XWM_DATA_HOME"] = f"{CACHE_DIR}/datasets"
    os.environ["HF_HOME"] = f"{CACHE_DIR}/huggingface"
    if dataset:
        os.environ["XWM_DATASET"] = dataset
    result = execute(EXPERIMENT, preset)
    volume.commit()
    cache.commit()
    return result


@app.function(volumes={OUTPUT_DIR: volume}, timeout=30 * 60)
def collect() -> dict[str, bytes]:
    return collect_files()


@app.local_entrypoint()
def main(preset: str = "gpu-recorded", dataset: str = "", fetch: bool = True):
    result = run.remote(preset, dataset or None)
    status = "ok" if result["returncode"] == 0 else f"FAILED ({result['returncode']})"
    print(f"\n{EXPERIMENT}: {status} in {result['seconds']:.1f}s")
    if fetch:
        print(f"downloaded {download(collect.remote())} files into examples/outputs/")

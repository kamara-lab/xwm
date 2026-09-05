"""Which readers can run here, plus the shared cache helpers.

The cache itself -- :func:`cache_dir`, :func:`http_fetch`, :func:`require` --
lives in :mod:`xwm.tools.cache`, because :mod:`xwm.tasks` and :mod:`xwm.envs`
need the same three rules (one user-named directory, resumable atomic
downloads, no new dependency for plain HTTP) and should not import a dataset
package to get them. They are re-exported here so existing call sites keep
working.
"""

from __future__ import annotations

from ..tools.cache import cache_dir, clear_cache, http_fetch, require

__all__ = [
    "READERS",
    "cache_dir",
    "clear_cache",
    "http_fetch",
    "require",
    "which_readers",
]

#: Which third-party module each reader needs, for :func:`which_readers`.
READERS: dict[str, tuple[str, ...]] = {
    "offline": (),  # OGBench: urllib only
    "hdf5": ("h5py",),
    "lerobot": ("huggingface_hub", "pyarrow", "av"),
    "rlds": ("tensorflow_datasets",),
    "minari": ("minari",),
}


def which_readers() -> dict[str, bool]:
    """Which readers this environment can actually run.

    The counterpart of :func:`xwm.envs.which_backends`: ask before you plan a
    run, rather than finding out from a traceback halfway through one.
    """
    from importlib.util import find_spec

    available = {}
    for reader, modules in READERS.items():
        try:
            available[reader] = all(find_spec(m) is not None for m in modules)
        except (ImportError, ValueError):  # pragma: no cover - broken installs
            available[reader] = False
    return available

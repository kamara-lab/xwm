"""Where downloaded and derived files live, and how they get there.

Shared by :mod:`xwm.datasets` (recorded corpora) and :mod:`xwm.tasks`
(preprocessed episodes for the benchmark). Three rules:

* **One directory, and the user names it.** ``$XWM_DATA_HOME`` if set, else
  ``~/.cache/xwm/datasets``. A cache belongs where the user's other caches are,
  not beside their source tree, and never inside the package.
* **Resumable and atomic.** Downloads land in ``<name>.part`` and are renamed
  only once complete, so an interrupted 40 GB pull resumes instead of restarting
  and a truncated file is never mistaken for a finished one.
* **No new dependency for plain HTTP.** ``urllib`` is in the standard library
  and OGBench is a static file server, so that reader costs nothing to install.
  Only the hub, parquet, video and HDF5 paths need ``xwm[data]``.
"""

from __future__ import annotations

import os
import shutil
import sys
import urllib.error
import urllib.request
from collections.abc import Iterable
from importlib import import_module
from pathlib import Path
from typing import Any

__all__ = ["cache_dir", "clear_cache", "http_fetch", "require"]

_CHUNK = 1 << 20  # 1 MiB


def cache_dir(*parts: str) -> Path:
    """The cache directory, created if absent.

    ``$XWM_DATA_HOME`` overrides the default ``~/.cache/xwm/datasets``. Extra
    ``parts`` are appended and created too, so a reader can ask for its own
    subdirectory in one call.
    """
    root = os.environ.get("XWM_DATA_HOME")
    base = Path(root).expanduser() if root else Path.home() / ".cache" / "xwm" / "datasets"
    path = base.joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path


def clear_cache(name: str | None = None) -> int:
    """Delete cached data and report how many bytes were freed.

    Args:
        name: a subdirectory or file under :func:`cache_dir`. ``None`` clears
            everything, which is why it is not the default of a function that
            deletes things.
    """
    target = cache_dir() if name is None else cache_dir() / name
    if not target.exists():
        return 0
    if target.is_dir():
        freed = sum(f.stat().st_size for f in target.rglob("*") if f.is_file())
        shutil.rmtree(target)
    else:
        freed = target.stat().st_size
        target.unlink()
    return freed


def require(
    modules: str | Iterable[str], extra: str | None = "data", *, what: str = "this reader"
) -> tuple[Any, ...]:
    """Import ``modules``, or raise an :class:`ImportError` naming the extra.

    The same contract as :func:`xwm.envs.newton_franka._require_newton`: fail at
    *call* time with an actionable message, so importing :mod:`xwm` never needs
    any of this installed.

    Args:
        modules: import names, e.g. ``"pyarrow.parquet"`` or ``("gymnasium", "gym_pusht")``.
        extra: the ``xwm[...]`` extra that provides them; ``None`` to name the
            bare package instead.
        what: who is asking, for the message -- ``"this reader"``, ``"PushTEnv"``.
    """
    names = (modules,) if isinstance(modules, str) else tuple(modules)
    imported = []
    for name in names:
        try:
            imported.append(import_module(name))
        except ImportError as exc:  # pragma: no cover - depends on the environment
            hint = (
                f'`pip install "xwm[{extra}]"`'
                if extra
                else f"`pip install {name.replace('_', '-')}`"
            )
            raise ImportError(f"{what} needs {name}; install it with {hint}") from exc
    return tuple(imported)


def http_fetch(
    url: str,
    destination: Path | None = None,
    *,
    expected_bytes: int | None = None,
    progress: bool = True,
    timeout: float = 30.0,
) -> Path:
    """Download ``url`` into the cache, resuming a partial download if there is one.

    Args:
        url: what to fetch.
        destination: where to put it. Defaults to the URL's filename under
            :func:`cache_dir`.
        expected_bytes: if given, a completed file of a different size is
            treated as corrupt and re-fetched. Worth passing for anything large:
            a truncated ``.npz`` fails much later and much less clearly.
        progress: print progress to stderr. On by default because these files
            are measured in gigabytes and silence for twenty minutes reads as a
            hang.

    Returns the path to the complete file. An already-complete file is returned
    without touching the network.
    """
    destination = cache_dir() / url.rsplit("/", 1)[-1] if destination is None else destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if expected_bytes is None or destination.stat().st_size == expected_bytes:
            return destination
        destination.unlink()  # wrong size: a truncated download from last time

    partial = destination.with_suffix(destination.suffix + ".part")
    have = partial.stat().st_size if partial.exists() else 0
    request = urllib.request.Request(url, headers={"User-Agent": "xwm"})
    if have:
        request.add_header("Range", f"bytes={have}-")

    try:
        response = urllib.request.urlopen(request, timeout=timeout)  # noqa: S310 - fixed schemes
    except urllib.error.HTTPError as exc:
        if have and exc.code == 416:  # already have the whole thing
            partial.rename(destination)
            return destination
        raise
    # A server that ignores Range replies 200 and starts from zero; appending
    # to what we have would corrupt the file, so start over instead.
    resuming = response.status == 206
    if have and not resuming:
        have = 0

    total = response.headers.get("Content-Length")
    total = (int(total) + have) if total is not None else expected_bytes
    with response, open(partial, "ab" if resuming else "wb") as handle:
        done, step = have, 0
        while chunk := response.read(_CHUNK):
            handle.write(chunk)
            done += len(chunk)
            if progress and total and done - step > total / 20:
                step = done
                print(
                    f"\r{destination.name}: {done / 1e6:.0f}/{total / 1e6:.0f} MB",
                    end="",
                    file=sys.stderr,
                    flush=True,
                )
    if progress and total:
        print(f"\r{destination.name}: {done / 1e6:.0f} MB", file=sys.stderr)

    if expected_bytes is not None and partial.stat().st_size != expected_bytes:
        raise OSError(
            f"{url} gave {partial.stat().st_size} bytes, expected {expected_bytes}; "
            f"the partial file is kept at {partial} so the next call can resume"
        )
    partial.rename(destination)
    return destination

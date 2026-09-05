"""A Rerun recording, as an object rather than a global.

Every logger in :mod:`xwm.rerun` takes one of these first and accepts ``None``,
so a call site that logs and one that does not are the same line of code::

    rec = xwm.rerun.Recording("xwm", path=directory / "train.rrd")
    xwm.rerun.log_collapse(rec, z, step=0)     # writes
    xwm.rerun.log_collapse(None, z, step=0)    # does nothing

Rerun's Python SDK also offers a process-global stream (``rr.init``, ``rr.log``).
This wraps :class:`rerun.RecordingStream` instead, because a library that
installs itself into a global has no way to coexist with the application that
imported it -- Newton's own ``ViewerRerun`` calls ``rr.init``, and two of those
in one process fight over the same sink.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ._backend import rerun

__all__ = ["Recording"]


class Recording:
    """A stream of logged data, going to a file, a viewer, or both.

    Args:
        app: the application id. Rerun keys blueprints and window layout off it,
            so runs that should share a layout should share an app id.
        path: write an ``.rrd`` file here. Self-contained: a run on a GPU box
            produces a file that opens on a laptop with ``rerun run.rrd``.
        spawn: launch the viewer as a child process and stream to it.
        connect: a gRPC address of an already-running viewer, e.g.
            ``"rerun+http://127.0.0.1:9876/proxy"``. Combine with ``path`` to
            watch a remote run live *and* keep the recording.
        blueprint: a :mod:`rerun.blueprint` object describing the default view
            layout. :mod:`xwm.rerun.blueprints` has one per kind of run.
        recording_id: set it to make several processes append to one recording.

    Example:
        >>> with Recording("xwm", path="run.rrd") as rec:      # doctest: +SKIP
        ...     rec.set_time("step", 0)
        ...     rec.log("train/loss", rec.rr.Scalars(1.0))
    """

    def __init__(
        self,
        app: str = "xwm",
        *,
        path: str | Path | None = None,
        spawn: bool = False,
        connect: str | None = None,
        blueprint: Any = None,
        recording_id: str | None = None,
    ):
        rr = rerun()
        self.rr = rr
        self.app = app
        self.path = Path(path) if path is not None else None
        self._closed = False
        self._stream = rr.RecordingStream(app, recording_id=recording_id)
        # The blueprint goes to each sink as its *default*, rather than being
        # logged once: a sink attached later would otherwise open with no
        # layout, and a viewer the user has already arranged is not overridden.
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._stream.save(str(self.path), default_blueprint=blueprint)
        if spawn:
            self._stream.spawn(default_blueprint=blueprint)
        if connect is not None:
            self._stream.connect_grpc(connect, default_blueprint=blueprint)

    # -- the stream ----------------------------------------------------------
    @property
    def stream(self):
        """The underlying :class:`rerun.RecordingStream`, for anything not wrapped."""
        return self._stream

    def log(self, entity: str, *archetypes, static: bool = False) -> None:
        """Log one or more archetypes at ``entity``, on the current times."""
        self._stream.log(entity, *archetypes, static=static)

    def set_time(self, timeline: str, value: int | float, *, kind: str = "sequence") -> None:
        """Move ``timeline`` to ``value``.

        ``kind`` is ``"sequence"`` for an integer index (a training step, an
        environment step) or ``"duration"`` for seconds.
        """
        if kind == "sequence":
            self._stream.set_time(timeline, sequence=int(value))
        elif kind == "duration":
            self._stream.set_time(timeline, duration=float(value))
        else:  # pragma: no cover - programmer error
            raise ValueError(f"unknown time kind {kind!r}; use 'sequence' or 'duration'")

    def send_columns(self, entity: str, indexes, columns) -> None:
        """Bulk-log a whole column at once -- a finished history, a full episode."""
        self._stream.send_columns(entity, indexes=indexes, columns=columns)

    def send_blueprint(self, blueprint) -> None:
        """Replace the default view layout."""
        self._stream.send_blueprint(blueprint, make_active=True)

    def notebook(self, *, width: int | None = None, height: int | None = None) -> None:
        """Embed the viewer in the current notebook cell.

        Needs ``rerun-sdk[notebook]``. Draining behaviour is Rerun's: the call
        shows what has been logged so far and streams what follows.
        """
        kwargs = {k: v for k, v in (("width", width), ("height", height)) if v is not None}
        self._stream.notebook_show(**kwargs)

    def log_file(self, path: str | Path, *, static: bool = True) -> None:
        """Hand a file to Rerun's own loaders -- a URDF, a mesh, an image."""
        self._stream.log_file_from_path(Path(path), static=static)

    # -- lifecycle -----------------------------------------------------------
    def flush(self) -> None:
        """Block until everything logged so far has reached the sinks."""
        self._stream.flush()

    def close(self) -> None:
        """Flush and disconnect. Idempotent, and safe to call twice."""
        if self._closed:
            return
        self._closed = True
        try:
            self._stream.flush()
        finally:
            self._stream.disconnect()

    def __enter__(self) -> Recording:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __repr__(self) -> str:
        where = str(self.path) if self.path is not None else "stream"
        return f"Recording({self.app!r}, {where})"

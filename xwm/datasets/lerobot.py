"""The LeRobot dataset layout: parquet for signals, mp4 for cameras.

This is the format that matters most, because it is the one the field converged
on. DROID ships in it, the Open X-Embodiment collections have been mirrored into
it, and every teleoperated SO-101 dataset recorded since is native to it. Reading
it means reading almost everything.

The layout is metadata-driven rather than filename-driven, which is what lets it
scale to millions of episodes and also what makes a reader more than a glob:

* ``meta/info.json`` holds the feature schema, the frame rate, and *format
  templates* for where data and video shards live. The templates are read from
  the file rather than hardcoded, because they are the part that changed between
  versions.
* ``meta/episodes/*.parquet`` (v3) or ``meta/episodes.jsonl`` (v2.1) says which
  shard each episode is in and which row and timestamp range inside it. Episode
  boundaries are metadata, not file boundaries -- one parquet file holds many
  episodes.
* Cameras are mp4 shards, also many episodes to a file, so a single episode is a
  *time span* to seek to and decode rather than a file to open.

Both versions are handled, because the mirrors are mid-migration: a v2.1 dataset
is one file per episode, a v3 one is not, and which you get depends on when the
dataset was last pushed.

References
----------
Cadene et al., *LeRobot: State-of-the-art machine learning for real-world
robotics in PyTorch*, 2024. https://github.com/huggingface/lerobot

Khazatsky et al., *DROID: A Large-Scale In-the-Wild Robot Manipulation
Dataset*, RSS 2024. arXiv:2403.12945.

Open X-Embodiment Collaboration, *Open X-Embodiment: Robotic Learning Datasets
and RT-X Models*, ICRA 2024. arXiv:2310.08864.
"""

from __future__ import annotations

import io
import json
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .cache import require
from .spec import clips, to_frames

__all__ = ["info", "iter_episodes", "load"]

#: Column prefixes that are proprioception rather than pixels or bookkeeping.
_STATE_PREFIX = "observation.state"
#: Columns describing the *scene* rather than the robot -- object keypoints,
#: for instance. Push-T's block pose lives here and nowhere else, so dropping
#: it would leave the dataset with no ground truth for the thing being pushed.
_POSITION_PREFIX = "observation.environment_state"
_BOOKKEEPING = frozenset(
    {"index", "frame_index", "episode_index", "task_index", "timestamp", "next.done", "next.reward"}
)


class _Source:
    """Either a Hub repo or a local directory, behind one ``path()`` call.

    Every reader in this module asks for relative paths like
    ``meta/info.json``; only this class knows whether that means a download or a
    directory join. Tests use the local branch and exercise the same code the
    Hub branch runs.
    """

    def __init__(self, repo_id: str | None = None, root: str | Path | None = None, revision=None):
        if (repo_id is None) == (root is None):
            raise ValueError("pass exactly one of repo_id or root")
        self.repo_id = repo_id
        self.root = Path(root).expanduser() if root is not None else None
        self.revision = revision

    def path(self, relative: str) -> Path:
        if self.root is not None:
            local = self.root / relative
            if not local.exists():
                raise FileNotFoundError(local)
            return local
        (hub,) = require("huggingface_hub", extra="data")
        try:
            return Path(
                hub.hf_hub_download(
                    self.repo_id, relative, repo_type="dataset", revision=self.revision
                )
            )
        except hub.errors.EntryNotFoundError as exc:
            # A missing file is a missing file whichever backend it came from.
            # This one is load-bearing: probing for `meta/episodes.jsonl` is how
            # the v2.1 layout is detected, and the Hub signals its absence with
            # its own exception type rather than with FileNotFoundError.
            raise FileNotFoundError(f"{self.repo_id}:{relative}") from exc

    def glob(self, prefix: str, suffix: str) -> list[str]:
        """Relative paths under ``prefix`` ending in ``suffix``, sorted."""
        if self.root is not None:
            base = self.root / prefix
            found = sorted(p.relative_to(self.root).as_posix() for p in base.rglob(f"*{suffix}"))
            return found
        (hub,) = require("huggingface_hub", extra="data")
        files = hub.list_repo_files(self.repo_id, repo_type="dataset", revision=self.revision)
        return sorted(f for f in files if f.startswith(prefix) and f.endswith(suffix))


def info(source: str, *, root: str | Path | None = None, revision: str | None = None) -> dict:
    """Read a dataset's ``meta/info.json``.

    Cheap -- one small file -- and the right thing to call before a download:
    it reports ``total_episodes``, ``total_frames``, ``fps`` and the feature
    schema, so you can see what you are about to pull.
    """
    where = _Source(None if root is not None else source, root, revision)
    return json.loads(where.path("meta/info.json").read_text())


def _episode_records(where: _Source, meta: dict) -> list[dict[str, Any]]:
    """Per-episode metadata, from whichever of the two layouts this dataset uses."""
    try:
        lines = where.path("meta/episodes.jsonl").read_text().splitlines()
        return [json.loads(line) for line in lines if line.strip()]
    except FileNotFoundError:
        pass

    (pyarrow,) = require("pyarrow.parquet", extra="data")
    records: list[dict[str, Any]] = []
    shards = where.glob("meta/episodes", ".parquet")
    if not shards:
        raise FileNotFoundError(
            "no meta/episodes.jsonl and no meta/episodes/*.parquet; this does not "
            "look like a LeRobot dataset"
        )
    for shard in shards:
        table = pyarrow.read_table(where.path(shard))
        records.extend(table.to_pylist())
    records.sort(key=lambda r: r.get("episode_index", 0))
    del meta
    return records


def _field(record: dict[str, Any], *names: str) -> Any:
    """First present key among ``names``, in order.

    Ordered exact candidates rather than a suffix match, because the v3 episode
    metadata has *several* columns ending in ``chunk_index`` -- one per camera,
    plus one for the data shard, plus one for the metadata shard itself -- and a
    suffix match picks whichever the dict happens to yield first. The row range
    is spelled ``dataset_from_index`` in the wild and ``data/from_index`` in the
    documentation, so both are listed wherever it is read.
    """
    for name in names:
        if name in record:
            return record[name]
    return None


#: Where the row range of an episode is recorded, most-current spelling first.
_FROM = ("dataset_from_index", "data/from_index", "from_index")
_TO = ("dataset_to_index", "data/to_index", "to_index")


def _video_span(record: dict[str, Any], camera: str) -> tuple[Any, ...]:
    """``(chunk, file, from_timestamp, to_timestamp)`` for one camera, v3 style."""
    prefix = f"videos/{camera}/"
    scoped = {k[len(prefix) :]: v for k, v in record.items() if k.startswith(prefix)}
    return (
        scoped.get("chunk_index"),
        scoped.get("file_index"),
        scoped.get("from_timestamp"),
        scoped.get("to_timestamp"),
    )


def _decode_column(values: list) -> np.ndarray:
    """Decode a parquet image column to ``(T, H, W, C)`` uint8.

    A feature declared ``dtype: "image"`` is stored *in the parquet*, one
    encoded PNG per row, as a ``{"bytes": ..., "path": ...}`` struct. Only
    ``dtype: "video"`` means an mp4 shard. Both spellings are in active use --
    ``lerobot/libero_10_image`` is the first, ``lerobot/droid_100`` the second.
    """
    (Image,) = require("PIL.Image", extra="data")
    frames = []
    for value in values:
        if isinstance(value, dict):
            value = value.get("bytes")
        if value is None:
            raise ValueError("image column row has no bytes")
        frames.append(np.asarray(Image.open(io.BytesIO(bytes(value))).convert("RGB")))
    return np.stack(frames)


def _decode_video(path: Path, start: float | None, stop: float | None) -> np.ndarray:
    """Decode ``[start, stop)`` seconds of an mp4 as ``(T, H, W, 3)`` uint8.

    ``start is None`` decodes the whole file, which is what a v2.1 dataset needs
    -- there, one file *is* one episode.
    """
    (av,) = require("av", extra="data")
    frames = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        if start:
            # Seek slightly early: seeking lands on a keyframe, and decoding
            # from after the span's first frame would silently drop frames.
            container.seek(max(int((start - 0.5) / stream.time_base), 0), stream=stream)
        for frame in container.decode(stream):
            moment = float(frame.pts * stream.time_base) if frame.pts is not None else 0.0
            if start is not None and moment < start - 1e-6:
                continue
            if stop is not None and moment >= stop - 1e-6:
                break
            frames.append(frame.to_ndarray(format="rgb24"))
    if not frames:
        raise ValueError(f"decoded no frames from {path} over [{start}, {stop})")
    return np.stack(frames)


def iter_episodes(
    source: str,
    *,
    root: str | Path | None = None,
    revision: str | None = None,
    cameras: Sequence[str] | None = None,
    observation: str = "both",
    limit: int | None = None,
    resize: int | None = None,
    dtype: str = "float32",
) -> Iterator[dict[str, np.ndarray]]:
    """Stream a LeRobot dataset as per-episode dicts.

    Args:
        source: a Hub dataset id, e.g. ``"lerobot/droid_100"``.
        root: read from this local directory instead of the Hub.
        revision: Hub revision, for pinning a dataset the way you would pin code.
        cameras: video feature keys, e.g. ``("observation.images.top",)``.
            ``None`` takes the first camera in the schema. Several are
            concatenated on the channel axis.
        observation: ``"video"``, ``"state"`` or ``"both"``.
        limit: stop after this many episodes. DROID is 76k of them; this is not
            an optional argument in practice.
        resize, dtype: passed to :func:`~xwm.datasets.spec.to_frames`.

    Yields dicts with ``video`` and/or ``state`` (``T`` entries), ``position``
    where the dataset records scene state (``observation.environment_state``),
    ``action`` (``T - 1``), ``reward`` where the dataset has one, and ``task``.
    """
    if observation not in ("video", "state", "both"):
        raise ValueError(f"observation must be 'video', 'state' or 'both', got {observation!r}")
    (pyarrow,) = require("pyarrow.parquet", extra="data")

    where = _Source(None if root is not None else source, root, revision)
    meta = json.loads(where.path("meta/info.json").read_text())
    features = meta.get("features", {})
    chunks_size = int(meta.get("chunks_size", 1000))
    default_data = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
    data_template = meta.get("data_path", default_data)
    video_template = meta.get("video_path", "")

    video_keys = [k for k, v in features.items() if v.get("dtype") in ("video", "image")]
    if cameras is not None:
        missing = [c for c in cameras if c not in video_keys]
        if missing:
            raise KeyError(f"no camera(s) {missing}; this dataset has {video_keys}")
        video_keys = list(cameras)
    elif video_keys:
        video_keys = video_keys[:1]
    want_video = observation in ("video", "both") and bool(video_keys)

    state_keys = sorted(
        k
        for k in features
        if k.startswith(_STATE_PREFIX) and features[k].get("dtype") not in ("video", "image")
    )
    position_keys = sorted(
        k
        for k in features
        if k.startswith(_POSITION_PREFIX) and features[k].get("dtype") not in ("video", "image")
    )

    records = _episode_records(where, meta)
    # One shard at a time, not a growing dict of them. Episodes are ordered by
    # shard, so a single-entry cache gets the same hit rate as an unbounded one
    # -- and an unbounded one would hold every parquet file in a 147-shard
    # dataset in memory, which is exactly what iter_episodes exists to avoid.
    cached: tuple[Path, Any] | None = None

    for count, record in enumerate(records):
        if limit is not None and count >= limit:
            return
        # Not `or count`: an episode_index of 0 is legitimate and falsy.
        recorded_index = _field(record, "episode_index")
        episode_index = count if recorded_index is None else int(recorded_index)

        # Where the rows are. v3 names the shard and the row range; v2.1 puts
        # one episode per file and the range is the whole thing.
        from_index = _field(record, *_FROM)
        to_index = _field(record, *_TO)
        formatted = data_template.format(
            episode_chunk=episode_index // chunks_size,
            episode_index=episode_index,
            chunk_index=_field(record, "data/chunk_index", "chunk_index") or 0,
            file_index=_field(record, "data/file_index", "file_index") or 0,
            # (both default to 0, so the `or` is safe here as it is not above)
        )
        shard = where.path(formatted)
        if cached is None or cached[0] != shard:
            cached = (shard, pyarrow.read_table(shard))
        table = cached[1]
        rows = table.slice(int(from_index), int(to_index) - int(from_index)) if (
            from_index is not None and to_index is not None
        ) else table

        columns = set(rows.column_names)
        actions = np.asarray(rows.column("action").to_pylist(), np.float32)
        n_rows = actions.shape[0]

        episode: dict[str, np.ndarray] = {}
        if want_video:
            channels = []
            for camera in video_keys:
                if features.get(camera, {}).get("dtype") == "image" and camera in columns:
                    decoded = _decode_column(rows.column(camera).to_pylist())
                    channels.append(to_frames(decoded, resize=resize, dtype=dtype))
                    continue
                chunk, file_index, start, stop = _video_span(record, camera)
                relative = video_template.format(
                    episode_chunk=episode_index // chunks_size,
                    episode_index=episode_index,
                    video_key=camera,
                    chunk_index=chunk or 0,
                    file_index=file_index or 0,
                )
                decoded = _decode_video(where.path(relative), start, stop)
                channels.append(to_frames(decoded, resize=resize, dtype=dtype))
            shortest = min(c.shape[0] for c in channels)
            video = np.concatenate([c[:shortest] for c in channels], axis=1)
            episode["video"] = video

        if observation in ("state", "both") and state_keys:
            vectors = []
            for key in state_keys:
                if key in columns:
                    value = np.asarray(rows.column(key).to_pylist(), np.float32)
                    vectors.append(value.reshape(value.shape[0], -1))
            if vectors:
                episode["state"] = np.concatenate(vectors, axis=-1)
        if observation in ("state", "both") and position_keys:
            vectors = []
            for key in position_keys:
                if key in columns:
                    value = np.asarray(rows.column(key).to_pylist(), np.float32)
                    vectors.append(value.reshape(value.shape[0], -1))
            if vectors:
                episode["position"] = np.concatenate(vectors, axis=-1)

        # A decoded video and a parquet slice can disagree by a frame at the
        # edges of a span. Trim to the shortest rather than trusting either.
        n_frames = min([v.shape[0] for v in episode.values()] or [n_rows])
        episode = {k: v[:n_frames] for k, v in episode.items()}
        episode["action"] = actions[: n_frames - 1]
        if "next.reward" in columns:
            reward = np.asarray(rows.column("next.reward").to_pylist(), np.float32).reshape(-1)
            episode["reward"] = reward[: n_frames - 1]
        task = _field(record, "task_index")
        if task is None and "task_index" in columns:
            task = rows.column("task_index").to_pylist()[0]
        if task is not None:
            episode["task"] = np.int32(task)
        yield episode


def load(
    source: str,
    *,
    length: int = 16,
    stride: int | None = None,
    limit: int | None = None,
    max_clips: int | None = None,
    **kwargs,
) -> dict[str, np.ndarray]:
    """Load a LeRobot dataset as fixed-length clips.

    See :func:`iter_episodes` for the arguments, and call :func:`info` first if
    you are not sure how much data ``limit=None`` would mean.
    """
    episodes = iter_episodes(source, limit=limit, **kwargs)
    return clips(episodes, length=length, stride=stride, max_clips=max_clips)

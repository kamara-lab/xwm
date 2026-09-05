"""Recorded datasets: read what someone else collected.

The third leg of a boundary the other two modules already draw.
:mod:`xwm.data` generates synthetic arrays in pure JAX, on device.
:mod:`xwm.envs` wraps an external simulator and steps it outside the ``jit``
boundary. This package reads data that was recorded once and will not change:
host-side NumPy, optional dependencies, network and disk.

Why it exists: until now every number the library could produce came from data
the library itself generated. A world model that has only ever seen a sprite and
a simulated Franka has not been tested against real contact, real latency or
real camera noise, and the corpora that supply those are all in one of four
formats. So there are four readers, one registry over them, and one function
that pours the result into the replay buffer the reward-driven families want.

**Everything lands in the same field layout**, the one
:func:`xwm.data.sprite_sequences` already produces -- ``video``, ``state``,
``action`` at ``T - 1``, ``reward`` at ``T - 1``. See :mod:`xwm.datasets.spec`
for the contract and why the ``T - 1`` matters.

>>> import xwm
>>> xwm.datasets.available()                       # doctest: +SKIP
>>> xwm.datasets.describe("lerobot/droid-100")     # doctest: +SKIP
>>> data = xwm.datasets.create("lerobot/droid-100", length=16, limit=8)  # doctest: +SKIP

Nothing here is imported eagerly in the sense that matters: the reader modules
themselves are pure-Python and cheap, and every third-party import happens
inside the function that needs it, via :func:`xwm.datasets.cache.require`. So
``import xwm`` works, :func:`available` works, and :func:`describe` works on an
install with no extras at all -- only actually reading data needs
``pip install "xwm[data]"``.

Every ``source`` below was resolved against the live Hub or host, with the
episode counts and file sizes read from the datasets themselves rather than from
their papers. A registry entry that 404s is worse than no entry.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import hdf5, lerobot, offline, rlds, spec
from .cache import READERS, cache_dir, clear_cache, http_fetch, require, which_readers
from .spec import DatasetSpec, check_episode, clips, to_frames

__all__ = [
    "DATASETS",
    "READERS",
    "DatasetSpec",
    "available",
    "cache_dir",
    "check_episode",
    "clear_cache",
    "clips",
    "create",
    "describe",
    "hdf5",
    "OXE_MIXTURE",
    "http_fetch",
    "lerobot",
    "mixture",
    "offline",
    "readers",
    "rlds",
    "spec",
    "stream",
    "to_frames",
    "to_replay_buffer",
    "which_readers",
]

_OXE = "Open X-Embodiment Collaboration, ICRA 2024. arXiv:2310.08864"
_OGBENCH = "Park et al., OGBench, ICLR 2025. arXiv:2410.20092"
_LIBERO = "Liu et al., LIBERO, NeurIPS 2023. arXiv:2306.03310"
_ROBOMIMIC = "Mandlekar et al., CoRL 2021. arXiv:2108.03298"
_DROID = "Khazatsky et al., DROID, RSS 2024. arXiv:2403.12945"
#: Minari's own citation; also referenced from :mod:`xwm.datasets.offline`.
MINARI_CITATION = "Younis et al., Minari, 2024 -- the successor to D4RL"

#: name -> what it is. Slash-namespaced by corpus, as
#: :data:`xwm.families.registry.REGISTRY` is by family.
DATASETS: dict[str, DatasetSpec] = {
    # -- LeRobot layout: parquet + mp4 or inline PNG ------------------------
    "lerobot/droid-100": DatasetSpec(
        name="lerobot/droid-100",
        reader="lerobot",
        source="lerobot/droid_100",
        summary=(
            "100 episodes of DROID, three exterior/wrist cameras at 180x320, 15 Hz. "
            "The first thing to reach for: small enough to download over coffee and "
            "identical in layout to the full 92k-episode set."
        ),
        action_dim=7,
        observation=("video", "state"),
        episodes=100,
        size_gb=0.4,
        licence="MIT",
        citation=_DROID,
    ),
    "lerobot/droid": DatasetSpec(
        name="lerobot/droid",
        reader="lerobot",
        source="IPEC-COMMUNITY/droid_lerobot",
        summary=(
            "DROID in full: 92,233 episodes, 27M frames, 564 scenes, one Franka. "
            "One robot in many scenes, which makes it the testbed for scene "
            "generalisation rather than for cross-embodiment transfer."
        ),
        action_dim=7,
        observation=("video", "state"),
        episodes=92233,
        licence="MIT",
        citation=_DROID,
    ),
    "lerobot/pusht": DatasetSpec(
        name="lerobot/pusht",
        reader="lerobot",
        source="lerobot/pusht",
        summary=(
            "Push-T: 206 episodes of 2-D pushing at 96x96 with reward and success "
            "flags. Tiny, and the closest recorded analogue of "
            ":class:`xwm.data.PushWorld` -- useful for checking that a result on "
            "the synthetic world survives contact with real data."
        ),
        action_dim=2,
        observation=("video", "state"),
        episodes=206,
        size_gb=0.03,
        licence="MIT",
        citation="Chi et al., Diffusion Policy, RSS 2023. arXiv:2303.04137",
    ),
    "lerobot/pusht-keypoints": DatasetSpec(
        name="lerobot/pusht-keypoints",
        reader="lerobot",
        source="lerobot/pusht_keypoints",
        summary=(
            "Push-T as 8 T-block keypoints (``position``, 16-D) plus the agent "
            "position (``state``, 2-D). The same 206 episodes as ``lerobot/pusht`` "
            "in the same order, without video. ``lerobot/pusht`` records only the "
            "agent, so this is the only source of the block pose -- which is what "
            ":mod:`xwm.tasks` joins the two on to evaluate goal reaching."
        ),
        action_dim=2,
        observation=("state", "position"),
        episodes=206,
        size_gb=0.01,
        licence="MIT",
        citation="Chi et al., Diffusion Policy, RSS 2023. arXiv:2303.04137",
    ),
    # -- Open X-Embodiment, through its LeRobot mirrors ---------------------
    "oxe/rt1": DatasetSpec(
        name="oxe/rt1",
        reader="lerobot",
        source="IPEC-COMMUNITY/fractal20220817_data_lerobot",
        summary=(
            "RT-1 (fractal20220817): 87,212 episodes on Everyday Robots mobile "
            "manipulators. The largest single contributor to the OXE mixture."
        ),
        action_dim=7,
        observation=("video", "state"),
        episodes=87212,
        licence="Apache-2.0",
        citation=_OXE,
    ),
    "oxe/bridge": DatasetSpec(
        name="oxe/bridge",
        reader="lerobot",
        source="IPEC-COMMUNITY/bridge_orig_lerobot",
        summary=(
            "Bridge Data V2: 53,192 episodes of WidowX manipulation across 24 "
            "scenes, four camera views. The standard cross-scene transfer set."
        ),
        action_dim=7,
        observation=("video", "state"),
        episodes=53192,
        licence="CC-BY-4.0",
        citation="Walke et al., BridgeData V2, CoRL 2023. arXiv:2308.12952",
    ),
    "oxe/language-table": DatasetSpec(
        name="oxe/language-table",
        reader="lerobot",
        source="IPEC-COMMUNITY/language_table_lerobot",
        summary=(
            "Language-Table: 442,226 short episodes of language-conditioned "
            "tabletop rearrangement. The largest episode count in OXE, and the "
            "one to use when the question is about task conditioning."
        ),
        action_dim=7,
        observation=("video", "state"),
        episodes=442226,
        licence="Apache-2.0",
        citation="Lynch et al., Interactive Language, CoRL 2022. arXiv:2210.06407",
    ),
    "oxe/taco-play": DatasetSpec(
        name="oxe/taco-play",
        reader="lerobot",
        source="IPEC-COMMUNITY/taco_play_lerobot",
        summary=(
            "Taco-Play: 3,242 episodes, static and gripper cameras. Small enough "
            "to hold in memory, which makes it the practical choice for a first "
            "real-data run."
        ),
        action_dim=7,
        observation=("video", "state"),
        episodes=3242,
        size_gb=3.0,
        licence="CC-BY-4.0",
        citation="Rosete-Beas et al., TACO-RL, CoRL 2022. arXiv:2209.13962",
    ),
    "oxe/berkeley-ur5": DatasetSpec(
        name="oxe/berkeley-ur5",
        reader="lerobot",
        source="IPEC-COMMUNITY/berkeley_autolab_ur5_lerobot",
        summary=(
            "Berkeley AUTOLab UR5: 896 episodes at 5 Hz, scene and hand cameras. "
            "A different embodiment from everything else here, and stored in the "
            "older v2.1 layout, so it also exercises that reader path."
        ),
        action_dim=7,
        observation=("video", "state"),
        episodes=896,
        size_gb=1.0,
        licence="CC-BY-4.0",
        citation=_OXE,
    ),
    # -- LIBERO, already converted to the LeRobot layout --------------------
    "libero/10": DatasetSpec(
        name="libero/10",
        reader="lerobot",
        source="lerobot/libero_10_image",
        summary=(
            "LIBERO-10 (long-horizon): 379 episodes, 256x256 agentview and wrist "
            "cameras stored as PNG inside the parquet. Ten multi-stage tasks."
        ),
        action_dim=7,
        observation=("video", "state"),
        episodes=379,
        size_gb=3.4,
        licence="MIT",
        citation=_LIBERO,
    ),
    "libero/spatial": DatasetSpec(
        name="libero/spatial",
        reader="lerobot",
        source="lerobot/libero_spatial_image",
        summary="LIBERO-Spatial: 432 episodes; same objects, different layouts.",
        action_dim=7,
        observation=("video", "state"),
        episodes=432,
        size_gb=1.8,
        licence="MIT",
        citation=_LIBERO,
    ),
    "libero/object": DatasetSpec(
        name="libero/object",
        reader="lerobot",
        source="lerobot/libero_object_image",
        summary="LIBERO-Object: 454 episodes; same layout, different objects.",
        action_dim=7,
        observation=("video", "state"),
        episodes=454,
        size_gb=2.2,
        licence="MIT",
        citation=_LIBERO,
    ),
    "libero/goal": DatasetSpec(
        name="libero/goal",
        reader="lerobot",
        source="lerobot/libero_goal_image",
        summary="LIBERO-Goal: 428 episodes; same scene, ten different goals.",
        action_dim=7,
        observation=("video", "state"),
        episodes=428,
        size_gb=1.7,
        licence="MIT",
        citation=_LIBERO,
    ),
    "libero/90": DatasetSpec(
        name="libero/90",
        reader="hdf5",
        source="hf://yifengzhu-hf/LIBERO-datasets/libero_90/",
        summary=(
            "LIBERO-90: the 90-task short-horizon suite, and the only LIBERO "
            "split with no LeRobot conversion -- it is read from the original "
            "HDF5, 90 per-task files (50 demos each), downloaded one at a time "
            "as the reader reaches them. Pass `limit` unless you want all of it."
        ),
        action_dim=7,
        observation=("video", "state"),
        episodes=4500,
        licence="MIT",
        citation=_LIBERO,
    ),
    # -- RoboMimic, raw HDF5 -----------------------------------------------
    # Only the low-dim observations are distributed; images are rendered
    # locally from the recorded simulator states with robomimic's
    # dataset_states_to_obs.py, so these entries are state-only by nature.
    "robomimic/lift": DatasetSpec(
        name="robomimic/lift",
        reader="hdf5",
        source="http://downloads.cs.stanford.edu/downloads/rt_benchmark/lift/ph/low_dim_v141.hdf5",
        summary=(
            "RoboMimic Lift, 200 proficient-human demos, low-dim observations "
            "only. 22 MB: the cheapest real dataset in the registry, and the "
            "fastest way to check a state-encoder pipeline end to end."
        ),
        action_dim=7,
        observation=("state",),
        episodes=200,
        size_gb=0.022,
        licence="MIT",
        citation=_ROBOMIMIC,
    ),
    "robomimic/can": DatasetSpec(
        name="robomimic/can",
        reader="hdf5",
        source="http://downloads.cs.stanford.edu/downloads/rt_benchmark/can/ph/low_dim_v141.hdf5",
        summary="RoboMimic Can, 200 proficient-human demos, low-dim only.",
        action_dim=7,
        observation=("state",),
        episodes=200,
        size_gb=0.048,
        licence="MIT",
        citation=_ROBOMIMIC,
    ),
    "robomimic/square": DatasetSpec(
        name="robomimic/square",
        reader="hdf5",
        source=(
            "http://downloads.cs.stanford.edu/downloads/rt_benchmark/square/ph/low_dim_v141.hdf5"
        ),
        summary=(
            "RoboMimic Square (nut assembly), 200 proficient-human demos. The "
            "hardest of the three, and the one where planning horizon shows up."
        ),
        action_dim=7,
        observation=("state",),
        episodes=200,
        size_gb=0.053,
        licence="MIT",
        citation=_ROBOMIMIC,
    ),
    # -- OGBench: state and pixel offline RL, plain HTTP --------------------
    "ogbench/pointmaze-medium-navigate": DatasetSpec(
        name="ogbench/pointmaze-medium-navigate",
        reader="offline",
        source="pointmaze-medium-navigate-v0",
        summary=(
            "OGBench pointmaze-medium, navigate: 1M transitions of 2-D maze "
            "navigation, 20 MB. The registry's smoke test -- it downloads in "
            "seconds and has the same structure as everything else here."
        ),
        action_dim=2,
        observation=("state",),
        size_gb=0.02,
        licence="MIT",
        citation=_OGBENCH,
    ),
    "ogbench/antmaze-large-navigate": DatasetSpec(
        name="ogbench/antmaze-large-navigate",
        reader="offline",
        source="antmaze-large-navigate-v0",
        summary=(
            "OGBench antmaze-large, navigate: a 29-D quadruped crossing a large "
            "maze. The long-horizon offline RL benchmark, and the recorded "
            "counterpart of :class:`xwm.data.MazeWorld`."
        ),
        action_dim=8,
        observation=("state",),
        size_gb=0.24,
        licence="MIT",
        citation=_OGBENCH,
    ),
    "ogbench/humanoidmaze-medium-navigate": DatasetSpec(
        name="ogbench/humanoidmaze-medium-navigate",
        reader="offline",
        source="humanoidmaze-medium-navigate-v0",
        summary=(
            "OGBench humanoidmaze-medium: a 69-D humanoid in a maze, 1.1 GB. "
            "Where horizon reduction stops being optional."
        ),
        action_dim=21,
        observation=("state",),
        size_gb=1.07,
        licence="MIT",
        citation=_OGBENCH,
    ),
    "ogbench/cube-single-play": DatasetSpec(
        name="ogbench/cube-single-play",
        reader="offline",
        source="cube-single-play-v0",
        summary=(
            "OGBench cube-single, play: pick-and-place with one cube, from "
            "unlabelled play data. The manipulation entry point."
        ),
        action_dim=5,
        observation=("state",),
        size_gb=0.26,
        licence="MIT",
        citation=_OGBENCH,
    ),
    "ogbench/cube-double-play": DatasetSpec(
        name="ogbench/cube-double-play",
        reader="offline",
        source="cube-double-play-v0",
        summary="OGBench cube-double, play: two cubes, up to four sequential subtasks.",
        action_dim=5,
        observation=("state",),
        size_gb=0.30,
        licence="MIT",
        citation=_OGBENCH,
    ),
    "ogbench/scene-play": DatasetSpec(
        name="ogbench/scene-play",
        reader="offline",
        source="scene-play-v0",
        summary=(
            "OGBench scene, play: a cube, a drawer, a window and two buttons -- "
            "eight subtasks, and the tasks that need a tool used before an object "
            "can be reached."
        ),
        action_dim=5,
        observation=("state",),
        size_gb=0.27,
        licence="MIT",
        citation=_OGBENCH,
    ),
    "ogbench/puzzle-3x3-play": DatasetSpec(
        name="ogbench/puzzle-3x3-play",
        reader="offline",
        source="puzzle-3x3-play-v0",
        summary=(
            "OGBench puzzle-3x3, play: 'Lights Out' with a robot arm. Combinatorial "
            "rather than geometric, which breaks distance-based shaping entirely."
        ),
        action_dim=5,
        observation=("state",),
        size_gb=0.24,
        licence="MIT",
        citation=_OGBENCH,
    ),
    "ogbench/visual-cube-single-play": DatasetSpec(
        name="ogbench/visual-cube-single-play",
        reader="offline",
        source="visual-cube-single-play-v0",
        summary=(
            "OGBench cube-single from pixels: 64x64x3 observations, 1.7 GB. The "
            "pixel-based offline RL setting, for an image encoder rather than an "
            "MLP."
        ),
        action_dim=5,
        observation=("video",),
        size_gb=1.68,
        licence="MIT",
        citation=_OGBENCH,
    ),
    # -- Minari: D4RL's successor (needs `pip install minari`) -------------
    "minari/pointmaze-umaze": DatasetSpec(
        name="minari/pointmaze-umaze",
        reader="minari",
        source="D4RL/pointmaze/umaze-v2",
        summary=(
            "D4RL pointmaze umaze through Minari: 13,210 episodes, 1M steps, a "
            "goal-conditioned dict observation flattened to 8 dims "
            "(achieved_goal | desired_goal | observation). The cheapest "
            "goal-conditioned dataset in the registry to iterate on."
        ),
        action_dim=2,
        observation=("state",),
        episodes=13210,
        licence="Apache-2.0",
        citation=MINARI_CITATION,
    ),
    "minari/antmaze-large-play": DatasetSpec(
        name="minari/antmaze-large-play",
        reader="minari",
        source="D4RL/antmaze/large-play-v2",
        summary=(
            "D4RL antmaze large-play through Minari: 2,052 episodes, 1M steps, "
            "31-dim flattened state. The D4RL counterpart of "
            "`ogbench/antmaze-large-navigate`, useful for checking a result "
            "against the older benchmark as well as the newer one."
        ),
        action_dim=8,
        observation=("state",),
        episodes=2052,
        licence="Apache-2.0",
        citation=MINARI_CITATION,
    ),
    "minari/halfcheetah-expert": DatasetSpec(
        name="minari/halfcheetah-expert",
        reader="minari",
        source="mujoco/halfcheetah/expert-v0",
        summary=(
            "MuJoCo halfcheetah, expert policy: 1,000 episodes, 1M steps, 17-dim "
            "state, no goal. The plain locomotion control case."
        ),
        action_dim=6,
        observation=("state",),
        episodes=1000,
        licence="Apache-2.0",
        citation=MINARI_CITATION,
    ),
    "minari/kitchen-complete": DatasetSpec(
        name="minari/kitchen-complete",
        reader="minari",
        source="D4RL/kitchen/complete-v2",
        summary=(
            "D4RL FrankaKitchen, complete demonstrations: 19 episodes and 4,209 "
            "steps, the smallest dataset here. Its observation space is a "
            "*nested* dict (per-object goals inside achieved_goal), flattened to "
            "81 dims -- the case that breaks a one-level flattener."
        ),
        action_dim=9,
        observation=("state",),
        episodes=19,
        size_gb=0.01,
        licence="Apache-2.0",
        citation=MINARI_CITATION,
    ),
    # -- Open X-Embodiment, native RLDS (needs tensorflow-datasets) --------
    "rlds/rt1": DatasetSpec(
        name="rlds/rt1",
        reader="rlds",
        source="fractal20220817_data",
        summary=(
            "RT-1 as published: RLDS TFRecord from the OXE bucket. The same data "
            "as ``oxe/rt1`` without the conversion, at the cost of a TensorFlow "
            "install -- `pip install tensorflow-datasets`."
        ),
        action_dim=7,
        observation=("video",),
        episodes=87212,
        licence="Apache-2.0",
        citation=_OXE,
        options={"version": "0.1.0"},
    ),
    "rlds/bridge": DatasetSpec(
        name="rlds/bridge",
        reader="rlds",
        source="bridge",
        summary="Bridge Data V2 as published, RLDS TFRecord from the OXE bucket.",
        action_dim=7,
        observation=("video",),
        licence="CC-BY-4.0",
        citation=_OXE,
        options={"version": "0.1.0"},
    ),
}


#: A spec's ``observation`` tuple as the readers' ``observation=`` argument.
_MODALITY = {
    ("video",): "video",
    ("state",): "state",
    ("video", "state"): "both",
    ("state", "position"): "state",
}


def available() -> list[str]:
    """Registered dataset names."""
    return sorted(DATASETS)


def readers() -> list[str]:
    """The reader each name uses."""
    return sorted({entry.reader for entry in DATASETS.values()})


def describe(name: str) -> DatasetSpec:
    """What a registered dataset is, without downloading any of it.

    Call this first. ``size_gb`` and ``episodes`` are why it exists: the
    difference between ``lerobot/droid-100`` and ``lerobot/droid`` is 400 MB
    against most of a terabyte, and the names differ by four characters.
    """
    if name not in DATASETS:
        raise KeyError(f"unknown dataset {name!r}; available: {available()}")
    return DATASETS[name]


def _resolve_hdf5(entry: DatasetSpec, *, progress: bool = True):
    """A local path -- or a lazy stream of them -- for an HDF5 dataset.

    Three source forms. A plain URL is one file. ``hf://repo/path.hdf5`` is one
    file on the Hub. ``hf://repo/prefix/`` is a *directory* of them, and that one
    returns a generator that downloads each file only when the reader reaches
    it: LIBERO-90 is 90 per-task files with no LeRobot conversion, and
    ``limit=4`` has to cost four downloads rather than ninety.
    """
    source = entry.source
    if source.startswith("hf://"):
        (hub,) = require("huggingface_hub", extra="data")
        # A Hub repo id is owner/name -- two segments -- so the path inside the
        # repo starts at the third. Splitting at the first slash yields the
        # owner as the repo and a 401.
        owner, name, path = (source[len("hf://") :].split("/", 2) + ["", ""])[:3]
        repo = f"{owner}/{name}"
        if not path.endswith("/"):
            return hub.hf_hub_download(repo, path, repo_type="dataset")

        def stream():
            files = hub.list_repo_files(repo, repo_type="dataset")
            wanted = sorted(
                f for f in files if f.startswith(path) and f.endswith((".hdf5", ".h5"))
            )
            if not wanted:
                raise FileNotFoundError(f"no HDF5 files under {source}")
            for name in wanted:
                yield hub.hf_hub_download(repo, name, repo_type="dataset")

        return stream()
    if source.startswith(("http://", "https://")):
        target = cache_dir(entry.reader) / f"{entry.name.replace('/', '-')}.hdf5"
        return str(http_fetch(source, target, progress=progress))
    return source


def create(name: str, **kwargs: Any) -> dict[str, np.ndarray]:
    """Load a registered dataset by name, as fixed-length clips.

    Keyword arguments go to the reader -- ``length``, ``stride``, ``limit``,
    ``resize``, ``dtype``, ``observation``, ``cameras``. See the reader modules
    for what each accepts.

    This downloads. :func:`describe` first if you do not know how much.
    """
    entry = describe(name)
    options = dict(entry.options)
    # The spec records which modalities the dataset actually has, so a
    # state-only corpus does not need the caller to say so. RoboMimic's low_dim
    # releases have no cameras at all, and the default of "both" would look for
    # one.
    options.setdefault("observation", _MODALITY.get(tuple(entry.observation), "both"))
    options.update(kwargs)
    if entry.reader == "lerobot":
        return lerobot.load(entry.source, **options)
    if entry.reader == "offline":
        return offline.load(entry.source, **options)
    if entry.reader == "minari":
        return offline.load_minari(entry.source, **options)
    if entry.reader == "rlds":
        return rlds.load(entry.source, **options)
    if entry.reader == "hdf5":
        progress = options.pop("progress", True)
        return hdf5.load(_resolve_hdf5(entry, progress=progress), **options)
    raise KeyError(f"dataset {name!r} names an unknown reader {entry.reader!r}")


#: The Open X-Embodiment mixture weights, restricted to the members registered
#: here. The published "Magic Soup" recipe spans ~25 datasets; five of them have
#: LeRobot mirrors in :data:`DATASETS`, and these are their weights renormalised
#: over that subset. It is **not** the full mixture, and a result on it is not a
#: result on OXE -- it is the shape of the thing, at a scale a laptop can reach.
OXE_MIXTURE: dict[str, float] = {
    "oxe/rt1": 0.35,
    "oxe/bridge": 0.30,
    "oxe/language-table": 0.15,
    "oxe/taco-play": 0.13,
    "oxe/berkeley-ur5": 0.07,
}


def stream(name: str, **kwargs: Any):
    """Stream a registered dataset as per-episode dicts, without materialising it.

    :func:`create` holds the whole thing in memory; this does not. Same
    arguments, minus the windowing ones (``length``, ``stride``, ``max_clips``),
    which only apply once episodes are being cut into clips.
    """
    entry = describe(name)
    options = dict(entry.options)
    options.setdefault("observation", _MODALITY.get(tuple(entry.observation), "both"))
    options.update(kwargs)
    if entry.reader == "lerobot":
        return lerobot.iter_episodes(entry.source, **options)
    if entry.reader == "offline":
        return offline.iter_episodes(entry.source, **options)
    if entry.reader == "rlds":
        return rlds.iter_episodes(entry.source, **options)
    if entry.reader == "hdf5":
        progress = options.pop("progress", True)
        return hdf5.iter_episodes(_resolve_hdf5(entry, progress=progress), **options)
    if entry.reader == "minari":
        raise NotImplementedError(
            "the Minari reader has no streaming path: minari.load_dataset materialises "
            "the dataset before xwm sees it, so use create() instead"
        )
    raise KeyError(f"dataset {name!r} names an unknown reader {entry.reader!r}")


def mixture(
    weights: dict[str, float],
    *,
    action_dim: int,
    length: int = 16,
    stride: int | None = None,
    max_clips: int | None = None,
    limit: int | None = None,
    seed: int = 0,
    **kwargs: Any,
) -> dict[str, np.ndarray]:
    """Interleave several registered datasets in proportion to ``weights``.

    This is what "training on Open X-Embodiment" actually means: not one dataset
    but a weighted mixture of many, which is why :data:`OXE_MIXTURE` exists.

    Two things have to be reconciled before episodes from different robots can
    share an array, and both are decisions rather than details:

    * **Observation size.** Pass ``resize`` -- it is required here, because the
      members disagree (180x320 for DROID, 480x640 for the UR5) and ``np.stack``
      will not tell you politely.
    * **Action width.** Cross-embodiment datasets have different action
      dimensions. Every action is zero-padded on the right to ``action_dim``,
      the convention the RT-X models use; an action *wider* than
      ``action_dim`` is an error rather than a truncation, because silently
      dropping a gripper channel is not a thing to do quietly.

    Args:
        weights: registered name -> relative weight. Normalised internally.
        action_dim: common action width to pad to.
        limit: episodes per member, not in total. The members are wildly
            different sizes, so a total would be dominated by whichever one the
            sampler happened to favour.
        seed: seeds the member sampler.
        kwargs: passed to every member's reader. ``resize`` is required.

    Returns the same field layout as :func:`create`, plus ``source``: the index
    into ``sorted(weights)`` that each clip came from, so a per-member breakdown
    is still possible afterwards.
    """
    if "resize" not in kwargs:
        raise ValueError(
            "mixture() needs resize=: its members record at different resolutions "
            "and clips from them have to share an array"
        )
    names = sorted(weights)
    if not names:
        raise ValueError("mixture() needs at least one dataset")
    total = float(sum(weights[n] for n in names))
    if total <= 0:
        raise ValueError(f"weights must be positive, got {weights}")
    probabilities = np.array([weights[n] / total for n in names])

    streams = {n: iter(stream(n, limit=limit, **kwargs)) for n in names}
    rng = np.random.default_rng(seed)

    def interleaved():
        alive = list(range(len(names)))
        while alive:
            weights_now = probabilities[alive] / probabilities[alive].sum()
            choice = int(rng.choice(alive, p=weights_now))
            try:
                episode = next(streams[names[choice]])
            except StopIteration:
                alive.remove(choice)
                continue
            actions = episode["action"]
            if actions.shape[-1] > action_dim:
                raise ValueError(
                    f"{names[choice]} has {actions.shape[-1]}-dimensional actions, "
                    f"wider than action_dim={action_dim}; raise it rather than "
                    "truncating a channel away"
                )
            pad = action_dim - actions.shape[-1]
            if pad:
                episode = dict(episode)
                episode["action"] = np.pad(actions, ((0, 0), (0, pad)))
            yield {**episode, "task": np.int32(choice)}

    return clips(interleaved(), length=length, stride=stride, max_clips=max_clips)


def to_replay_buffer(
    data: dict[str, np.ndarray],
    *,
    reward: float | np.ndarray | None = None,
    observation: str | None = None,
    capacity: int | None = None,
    seed: int = 0,
):
    """Pour a loaded dataset into a :class:`xwm.training.ReplayBuffer`.

    This is the join between this package and the reward-driven families. Each
    clip becomes one buffer episode, which is exactly right: clips already never
    straddle an episode boundary, and the buffer refuses slices that do, so an
    unroll sampled out of the buffer is guaranteed contiguous in the original
    recording.

    Args:
        data: the output of a reader's ``load``.
        reward: what to use when ``data`` has no ``reward`` field. A scalar fills
            it; an array must be shaped like ``data["action"][..., 0]``.
        observation: which field is the observation -- ``"video"``, ``"state"``
            or ``"image"``. Defaults to the first of those present.
        capacity: buffer capacity in steps. Defaults to exactly what the data
            needs, so nothing is evicted.
        seed: sampling seed for the buffer.

    Raises:
        ValueError: if there is no reward anywhere. Goal-conditioned corpora
            such as OGBench genuinely have none -- the reward belongs to the
            task you evaluate against, not to the data -- so this asks rather
            than quietly filling zeros and letting a value function train on
            nothing.
    """
    from ..training import ReplayBuffer

    if observation is None:
        for candidate in ("video", "state", "image"):
            if candidate in data:
                observation = candidate
                break
        else:
            raise ValueError(f"no observation field in {sorted(data)}")
    if "action" not in data:
        raise ValueError(f"no 'action' field in {sorted(data)}")

    observations = np.asarray(data[observation], np.float32)
    actions = np.asarray(data["action"], np.float32)
    n_clips, n_steps = actions.shape[0], actions.shape[1]

    if "reward" in data:
        rewards = np.asarray(data["reward"], np.float32).reshape(n_clips, n_steps)
    elif reward is None:
        raise ValueError(
            f"the data has no 'reward' field (it has {sorted(data)}) and none was "
            "given. Goal-conditioned datasets carry no reward by design -- pass "
            "reward=<scalar or array> to say what it should be."
        )
    elif np.ndim(reward) == 0:
        rewards = np.full((n_clips, n_steps), float(reward), np.float32)
    else:
        rewards = np.asarray(reward, np.float32).reshape(n_clips, n_steps)

    buffer = ReplayBuffer(
        capacity=capacity or max(n_clips * n_steps, 2),
        observation_shape=observations.shape[2:],
        action_shape=actions.shape[2:],
        seed=seed,
    )
    for clip in range(n_clips):
        buffer.add_episode(observations[clip], actions[clip], rewards[clip])
    return buffer

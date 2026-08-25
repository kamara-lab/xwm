# Datasets

Until recently `xwm` could only be run on data it generated itself: the sprite
world in [`xwm.data`](../reference/data.md) and the Franka arm in
[`xwm.envs`](../reference/envs.md). [`xwm.datasets`](../reference/datasets.md)
reads corpora that someone else recorded, so a result can be reported on the
data the field benchmarks on.

```bash
pip install "xwm[data]"
```

## Look before you download

```python
import xwm

xwm.datasets.available()                      # every registered name
xwm.datasets.which_readers()                  # what this environment can read
xwm.datasets.describe("lerobot/droid-100")
```

`describe` returns a
[`DatasetSpec`](../reference/datasets.md#xwm.datasets.DatasetSpec) — episode
count, size on disk, action dimension, licence, citation — without touching the
network. Worth the habit: `lerobot/droid-100` and `lerobot/droid` differ by four
characters and by three orders of magnitude.

```python
data = xwm.datasets.create("lerobot/droid-100", length=16, limit=8, resize=64)
```

Downloads are cached under `$XWM_DATA_HOME`, or `~/.cache/xwm/datasets` if that
is unset, and resume if interrupted. `xwm.datasets.clear_cache(name)` frees them
again and reports how many bytes it freed.

## What comes back

The same field layout `xwm.data` produces, so nothing downstream has to know
where a batch came from.

| field | shape | meaning |
| --- | --- | --- |
| `video` | `(n, T, C, H, W)` | frames, `[0, 1]` at `dtype="float32"` |
| `state` | `(n, T, S)` | proprioception, for a state encoder |
| `action` | `(n, T-1, A)` | `action[i, t]` joins frames `t` and `t+1` |
| `reward` | `(n, T-1)` | the reward of the state each action led to |
| `task` | `(n,)` | index into the dataset's task list |
| `episode` | `(n,)` | which source episode each clip came from |
| `dropped` | `(n,)` | episodes too short for `length`, and thus skipped |

!!! warning "The `T-1` is not cosmetic"

    Every recorded format stores one action *per frame*, including a final action
    whose result was never recorded. The readers drop it. Keeping it shifts the
    whole dataset by one step, and the symptom — a dynamics model that learns a
    slightly blurred identity — reads as underfitting rather than as a bug.
    [`check_episode`](../reference/datasets.md#xwm.datasets.check_episode)
    refuses an episode that did not drop it.

Check `dropped` on a first run. A `length` larger than most episodes is the
easiest way to draw a conclusion from 4% of a corpus without noticing.

## The four readers

=== "LeRobot"

    Parquet for signals, mp4 (or inline PNG) for cameras. The format the field
    converged on: DROID ships in it, the Open X-Embodiment datasets are mirrored
    into it, and every SO-101 dataset recorded since is native to it.

    ```python
    xwm.datasets.lerobot.info("lerobot/droid_100")      # schema, fps, counts
    xwm.datasets.create("libero/10", length=16, limit=20, resize=64)
    xwm.datasets.create("oxe/bridge", length=8, limit=100, observation="state")
    ```

    Both layouts are handled — v3, where many episodes share one parquet and one
    mp4 and the boundaries live in `meta/episodes/`, and v2.1, where each episode
    is its own pair of files. Which you get depends on when the dataset was last
    pushed, and the mirrors are mid-migration.

=== "OGBench"

    State and pixel offline RL over plain HTTP. No client library, no hub, **no
    dependency at all** beyond the standard library, so this reader works on a
    bare `pip install xwm`.

    ```python
    data = xwm.datasets.create("ogbench/antmaze-large-navigate", length=32, limit=200)
    ```

    OGBench distinguishes `terminals` (end of trajectory) from `masks` (task
    completion). They are not interchangeable: bootstrap across the wrong one and
    the value function learns to predict through a reset, silently — the loss
    still falls. Episodes are segmented on `terminals`; `masks` is passed through
    untouched.

    These datasets carry **no reward**, by design. The reward belongs to the task
    you evaluate against, not to the data, which is the premise of a
    goal-conditioned benchmark — so `to_replay_buffer` asks you for one.

=== "Minari"

    D4RL's successor, and the way to reach the older benchmark's datasets.
    Needs `pip install minari`, which is why it is not in the `data` extra — it
    brings Gymnasium and a set of environment dependencies most users already
    pin themselves.

    ```python
    data = xwm.datasets.create("minari/antmaze-large-play", length=32, limit=200)
    data = xwm.datasets.offline.load_minari("D4RL/pen/expert-v2", length=16)   # any name
    ```

    Four datasets are registered; `load_minari` takes any of the 343 names in
    `minari.list_remote_datasets()`. Minari already stores `T + 1` observations
    against `T` actions, which is this library's convention exactly, so the
    reader is nearly a transcription. The one thing it does is flatten dict
    observation spaces: every goal-conditioned dataset has one, and
    `D4RL/kitchen` nests them, so `achieved_goal | desired_goal | observation`
    is concatenated in sorted-key order into a single state vector — the goal
    becomes part of the state, which is what a goal-conditioned value function
    wants.

=== "HDF5"

    LIBERO and RoboMimic, in the flat `data/demo_N/obs/...` layout they share.
    Cheapest of the four to read: one dependency, random access, no codec.

    ```python
    data = xwm.datasets.create("robomimic/lift", length=16)       # 22 MB
    data = xwm.datasets.hdf5.load("mine.hdf5", cameras=("agentview_rgb",), split="train")
    ```

    Only RoboMimic's low-dim observations are distributed; its image variants are
    rendered locally from the recorded simulator states, which is why those
    entries are state-only. LIBERO is easier to reach through its LeRobot
    conversion (`libero/10`, `libero/spatial`, `libero/object`, `libero/goal`).

=== "RLDS"

    Open X-Embodiment as published, from the public GCS bucket. Needs
    `pip install tensorflow-datasets`, which is why it is not in the `data`
    extra — TensorFlow's CUDA pins fight JAX's. Prefer the `oxe/*` mirrors
    unless you specifically want the original bytes.

    ```python
    data = xwm.datasets.create("rlds/rt1", length=8, limit=50)
    ```

## Mixtures

"Training on Open X-Embodiment" does not mean one dataset. It means a weighted
mixture of many, which is what
[`mixture`](../reference/datasets.md#xwm.datasets.mixture) is for:

```python
data = xwm.datasets.mixture(
    xwm.datasets.OXE_MIXTURE,       # rt1, bridge, language-table, taco-play, berkeley-ur5
    action_dim=7,
    resize=64,
    length=16,
    limit=20,                       # episodes *per member*
)
data["task"]                        # which member each clip came from
```

Two things have to be reconciled before episodes from different robots can share
an array, and both are decisions rather than details. `resize` is **required**,
because the members disagree (180×320 for DROID, 480×640 for the UR5). And
actions are zero-padded on the right to `action_dim`, the convention the RT-X
models use — an action *wider* than `action_dim` raises, because silently
dropping a gripper channel is not something to do quietly.

`OXE_MIXTURE` is not the published Magic Soup. That recipe spans about 25
datasets; five have LeRobot mirrors registered here, and `OXE_MIXTURE` is their
weights renormalised over that subset. A result on it is not a result on OXE.

[`stream`](../reference/datasets.md#xwm.datasets.stream) is the same
registry-by-name access without materialising anything, and is what `mixture`
is built on:

```python
for episode in xwm.datasets.stream("oxe/bridge", resize=64):
    ...
```

## Feeding the reward-driven families

JEPA trains on a shuffled pile of clips, so a loaded dataset goes straight into
[`iter_batches`](../reference/data.md#xwm.data.iter_batches). TD-MPC2 and MuZero
need contiguous slices of one episode, which is what
[`ReplayBuffer`](../reference/training.md#xwm.training.ReplayBuffer) provides:

```python
data = xwm.datasets.create("robomimic/lift", length=32, limit=100)
buffer = xwm.datasets.to_replay_buffer(data)

agent = xwm.families.tdmpc2.tdmpc2(action_dim=7, observation="state", state_dim=data["state"].shape[-1])
batch = buffer.sample(64, horizon=3)
```

Each clip becomes one buffer episode. That is exactly right rather than merely
convenient: clips already never straddle an episode boundary, and the buffer
refuses slices that do, so every unroll sampled out of it is contiguous in the
original recording.

For a dataset with no reward, say what it should be:

```python
buffer = xwm.datasets.to_replay_buffer(data, reward=-1.0)          # a constant
buffer = xwm.datasets.to_replay_buffer(data, reward=my_rewards)    # (n, T-1)
```

## Memory

`create` and `load` hold everything in memory. A LIBERO suite at
`dtype="float32"` and full resolution is tens of gigabytes, and the three knobs
that matter are `limit` (episodes), `resize` (pixels) and `dtype="uint8"` (four
times less, converted at the `jit` boundary instead).

When it will not fit at all, stream it:

```python
for episode in xwm.datasets.lerobot.iter_episodes("lerobot/droid_100", limit=None):
    ...     # one episode at a time
```

## Staying honest about scale

Every registered `source` was resolved against the live Hub or host, and the
episode counts in the registry were read from the datasets themselves rather
than from their papers. But a `limit=8` run on DROID is a laptop result, and
`describe(name).episodes` is there to make the gap between what you ran and what
exists impossible to overlook. Report the scale you actually used.

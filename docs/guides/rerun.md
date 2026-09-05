# Inspecting a run with Rerun

Everything else `xwm` writes is static: a PNG of a loss curve, a GIF of an
episode, a JSON of the numbers. That is the right format for a paper and the
wrong one for a diagnosis. *Where* an imagined rollout leaves the real
trajectory, *whether* a planner's proposal ever converged, *what* the arm was
doing when the distance stopped falling -- all are shapes over time, and a shape
over time wants a timeline you can scrub.

```bash
pip install "xwm[rerun]"
```

```bash
xwm train configs/pusht/jepa.toml --rerun --eval
rerun runs/pusht-synthetic-jepa-action-s0-9f2c1a0b4e7d/train.rrd
```

A `.rrd` is self-contained. A run on a GPU box produces one file that opens on a
laptop, which is the point: `deploy/` runs on Modal, and the recording comes back
with the rest of the artifacts.

## What the flag does

`--rerun` writes `train.rrd` from `xwm train` and `eval.rrd` from `xwm eval`,
both into the run directory beside `history.json` and `eval.json`. The
equivalent config block, for a file that should always record:

```toml
[log]
rerun = true
spawn = false                                  # also open the viewer
connect = "rerun+http://127.0.0.1:9876/proxy"  # also stream to a running one
every = 1                                      # thin the training callback
```

`log` is the one section [`config_hash`](../reference/config.md) ignores. Two
runs that differ only in whether a viewer was watching are the same experiment,
so they produce the same digest and land in the same run directory. Watching a
run cannot change where it is written, and therefore cannot change what it is
compared against.

`--rerun-spawn` is `--rerun` plus a viewer window, for a laptop. `xwm doctor`
reports whether the SDK is installed at all.

## Using it from your own code

Nothing above is special-cased in the CLI: it calls the same public functions.

```python
import xwm

rec = xwm.rerun.Recording("xwm", path="run.rrd",
                          blueprint=xwm.rerun.blueprints.training())

state, history = trainer.fit(
    batches, steps=10_000, callbacks=[xwm.rerun.training_callback(rec)],
)
xwm.rerun.log_collapse(rec, model.encode(frames), step=int(state.step))
rec.close()
```

`Recording` wraps a [`rerun.RecordingStream`][stream] rather than the SDK's
process-global one, so it coexists with anything else in the process that logs
-- Newton's own `ViewerRerun`, for instance. Sinks compose: pass `path=` and
`connect=` together to watch a remote run live *and* keep its recording.

**Every logger accepts `None` for the recording and does nothing with it.** That
is what keeps instrumentation from becoming branching:

```python
rec = xwm.rerun.Recording(...) if watching else None
xwm.rerun.log_plan(rec, plan)          # writes, or does not
```

## What gets logged

| function | what it puts in the recording |
| --- | --- |
| `training_callback(rec)` | every metric `loss` returns, on `step` |
| `log_history(rec, history)` | a finished history, one send per metric |
| `log_collapse(rec, z)` | `rankme`, `rank_ratio`, `feature_std`, `mean_cosine`, and the spectrum |
| `log_summary(rec, model)` | the parameter tree, static |
| `log_latent_trajectory(rec, true, imagined)` | both paths in one PCA basis, plus the full-dimensional error per step |
| `log_plan(rec, plan)` | the proposal mean, its spread, the actions taken, the cost |
| `log_clip(rec, frames)` | a `(T, C, H, W)` clip on `frame` |
| `episode_hook(rec)` | an `on_step` hook for `xwm.bench.evaluate` |
| `log_episode(rec, result)` | a finished `EpisodeResult`, after the fact |
| `franka_hook(rec, env)` | the arm's meshes, joints, camera frustum and reach distance |

Five timelines, so several things can be logged into one recording without
their steps colliding:

| timeline | counts |
| --- | --- |
| `step` | optimizer steps -- training metrics, collapse diagnostics |
| `episode` | which evaluation instance |
| `env_step` | raw environment steps within an episode |
| `horizon` | position within one imagined rollout or plan |
| `frame` | position within a logged clip |

`env_step` counts **raw** environment steps, not planned ones, because that is
the unit the budget and the protocol are stated in. One planned action covers
`frameskip` of them.

## Reading an evaluation

`episode_hook` namespaces entities by policy -- `episode/planner/distance`,
`episode/noop/distance`, and so on -- so the planner and its three floors plot
on one axis. That is the comparison the
[benchmark](benchmark.md) exists to make, and it is the first thing to look at:
a `replay` trace that does not reach zero means the environment is not being
reset faithfully and no model number from that task means anything yet.

`episode/planner/plan/spread` is the planner's own uncertainty. A search whose
spread never shrinks did not converge, which looks identical to a bad model in
the success rate alone.

## The Franka arm in 3-D

```python
env = xwm.envs.FrankaEnv(xwm.envs.FrankaConfig(image_size=64))
hook = xwm.rerun.franka_hook(rec, env)     # logs the static scene immediately
env.reset(seed=0)
for action in actions:
    env.step(action)
    hook()
```

The meshes come from the same URDF Newton built the arm from, the joint
transforms from `env.joint_positions()`, and the camera frustum from
`env.camera_framing` and `config.camera_fov_degrees` -- the one definition every
renderer shares, so the frustum in the 3-D view is the view the model trains on.

!!! note "Why not Newton's own viewer"

    Newton ships `newton.viewer.ViewerRerun`, and for an interactive session it
    is the better tool: it draws contacts and every collision shape. It is not
    usable from a library, because it calls the process-global `rr.init` and,
    unless handed a server address, spawns or serves a viewer. A headless
    training run wants neither. Use it directly when you want it:

    ```python
    viewer = newton.viewer.ViewerRerun(keep_historical_data=True)
    ```

!!! note "The URDF is rewritten, not edited"

    The Franka URDF addresses its meshes as
    `package://franka_emika_panda/meshes/...`, the ROS convention. Newton's
    asset cache has that layout but not the `package.xml` a ROS resolver looks
    for, so Rerun would log an arm with no geometry -- an empty 3-D view, and no
    error. `xwm.rerun.franka.resolve_urdf` writes a copy with absolute paths
    into xwm's own cache. Newton owns its cache and re-downloads it; a library
    that edits another's cache is a bug waiting for a version bump.

## Examples and notebooks

The examples record when `XWM_RERUN=1` is set, writing beside their other
artifacts:

```bash
XWM_RERUN=1 uv run python examples/05_planning.py     # plan, latent rollout
XWM_RERUN=1 uv run python examples/06_franka_newton.py   # the arm in 3-D
```

In a notebook, `pip install "rerun-sdk[notebook]"` and embed the viewer in the
cell:

```python
rec = xwm.rerun.Recording("xwm")
...
rec.notebook()
```

[stream]: https://rerun.io/docs/concepts/logging/recordings

# Running on GPU

Each experiment is its own [Modal](https://modal.com) app, so the nine can run
concurrently on separate GPUs and be started, watched and stopped independently.

```bash
./deploy/run_all.sh                            # all nine, gpu preset
./deploy/run_all.sh xl                         # all nine, higher fidelity
modal run deploy/app_tdmpc2.py --preset xl     # just one
modal run deploy/app_recorded.py               # recorded data, gpu-recorded preset
```

Logs land in `deploy/logs/<experiment>.log`. One app per experiment means they get
separate GPUs and separate volumes, so a failure in one does not touch the others.

## Presets

Presets raise resolution, episode count, model size and render quality through
`XWM_*` environment variables, so there is **one copy of each pipeline** rather
than a laptop version and a cluster version.

| preset | for |
| --- | --- |
| `cpu-parity` | the laptop settings, on GPU hardware |
| `gpu` | the default: 128 px, 4000 steps, ViT depth 6 / width 384, 120 RL iterations |
| `xl` | 224 px, 12 000 steps, ViT depth 12, 400 RL iterations, 1024 px path tracing |
| `gpu-recorded` | example 09 only: 96 px, 6000 steps per stage, all 206 Push-T episodes, `reg_weight` 100 |

`cpu-parity` exists to **isolate hardware from settings**. If a GPU run disagrees
with a laptop run, `cpu-parity` tells you whether the hardware or the configuration
is responsible. Without it, every comparison confounds the two.

`gpu-recorded` is separate from `gpu` rather than folded into it because
[example 09](examples.md)'s input resolution is fixed by its data: Push-T records
at 96×96, and upsampling that to the shared preset's 128 px buys tokens and no
information. It also runs a smaller batch than `gpu`, which is not a typo — see
[Findings](../findings.md#xla_python_client_preallocatefalse-does-not-make-a-gpu-bigger)
for the 12 GiB allocation that established it.

!!! note "The dataset cache"

    `app_recorded.py` is the only app whose data comes off the network, so it
    mounts a second volume, `xwm-dataset-cache`, at `/data` with `XWM_DATA_HOME`
    and `HF_HOME` pointed into it. Without it every run re-downloads, and a run
    on DROID or LIBERO would re-download tens of gigabytes. It is deliberately
    not the artifacts volume: `collect_files` globs that one, so a cached dataset
    in there would come back down with the figures every time.

## How the overrides reach the examples

Each example reads its knobs through one helper:

```python
IMG_SIZE = setting("IMG_SIZE", 32)      # reads XWM_IMG_SIZE, falls back to 32
STEPS = setting("STEPS", 300)
```

The `XWM_` prefix lives in exactly one place (`deploy/_shared.py`), because
writing prefixed names by hand in each script is precisely how a preset silently
stops applying to one of them.

So a preset is a dict of strings, and running an example locally with an override
needs no Modal at all:

```bash
XWM_IMG_SIZE=64 XWM_STEPS=1000 python examples/01_image_ijepa.py
```

## JAX on the device

Install the accelerator wheel first; `xwm` does not pin one:

```bash
pip install -U "jax[cuda12]"
```

Nothing in the library is CUDA-specific, and nothing needs changing to move a
script to a GPU. Two things do change behaviour:

- **The physics solver.** `FrankaConfig(solver="auto")` picks `mujoco_warp` where a
  CUDA GPU is available and Featherstone otherwise.
- **The renderer.** Path tracing is seconds per frame, so the presets enable it
  (`XWM_RTX=1`) only on GPU, and it still needs graphics access, not just CUDA.
  See [Rendering](rendering.md#when-rtx-is-unavailable).

## Sizing a run

The knobs that actually cost time, in rough order:

| knob | effect |
| --- | --- |
| `IMG_SIZE` / `PATCH_SIZE` | tokens go as \((\text{img}/\text{patch})^2\), the dominant term |
| `ENC_DEPTH`, `ENC_DIM` | encoder size, linear in depth |
| `STEPS`, `PRETRAIN_STEPS`, `DYNAMICS_STEPS` | wall clock, linear |
| `ITERATIONS` × `EPISODES_PER_ITER` | **simulation** time for the RL families, not GPU time |
| `PLAN_SAMPLES`, `PLAN_ITERS`, `SIMULATIONS` | planning cost per decision |

For the reward-driven families, collection is usually the bottleneck rather than
training: the arm steps on the host, one substep at a time, while the GPU waits.
That is the argument for `observation="state"` while you are still debugging the
pipeline.

## Checkpoints and artifacts

Examples write to `examples/outputs/<name>/`, which the Modal apps mount as a
volume so results survive the container. To keep a model rather than a figure:

```python
xwm.tools.save(out / "model", state.model, config={"img_size": IMG_SIZE})
```

See [Figures and tables](figures.md) for what the examples emit, and
[Findings](../findings.md) for what the GPU runs measured.

# Examples

Nine runnable scripts, in the order they build on each other. **01–05 run on CPU**
against the synthetic worlds in [`xwm.data`](../reference/data.md), so there is no
dataset to download. **06–08** need the `newton` extra and download the Franka asset
on first run. **09** needs the `data` extra and downloads ~30 MB of recorded robot
data.

```bash
python examples/01_image_ijepa.py
```

| example | shows |
| --- | --- |
| `01_image_ijepa.py` | I-JEPA pretraining, a probe, collapse diagnostics |
| `02_video_vjepa.py` | tube masking, short- vs long-range |
| `03_collapse_strategies.py` | `ema` vs `sigreg` vs `vicreg` vs `none` |
| `04_action_world_model.py` | frozen encoder + latent dynamics, compounding error |
| `05_planning.py` | the full JEPA pipeline, measured against baselines |
| `06_franka_newton.py` | the same pipeline on a Franka arm |
| `07_tdmpc2_franka.py` | TD-MPC2: learn the model *and* the value |
| `08_muzero_franka.py` | MuZero: a model that agrees with its own search |
| `09_recorded_data.py` | a recorded dataset, against the synthetic world built to abstract it |

Measured results, including the negative ones, are collected in
**[Findings](../findings.md)**.

## What they have in common

Every example follows the same shape, through `examples/_common.py`:

```python
IMG_SIZE = setting("IMG_SIZE", 32)      # reads XWM_IMG_SIZE, defaults to a laptop value
out = setup("01_image_ijepa")           # makes examples/outputs/01_image_ijepa/
```

So the same file runs on a laptop and on an A100. See
[Running on GPU](gpu.md). Defaults are deliberately small: 32 px, a few hundred
steps, minutes on CPU.

`09` is the exception and the slowest of the nine, because it trains four models
— two stages on each of two data sources. At its defaults it took 1494 s of CPU
time on an M-series laptop; `XWM_EPISODES`, `XWM_PRETRAIN_STEPS` and
`XWM_DYNAMICS_STEPS` are the knobs that shrink it. Its GPU run is 20× the work
in 2515 s on one A10 (`modal run deploy/app_recorded.py`), and the two disagree
about the headline result rather than merely about precision — see
[Findings](../findings.md#scaling-the-recipe-does-rescue-it-on-a-gpu).

Each writes `*.png` figures, `*.gif` animations, `*.json` metrics at full precision
and `*.tex` tables of the same numbers, into `examples/outputs/<name>/`.

The three Franka examples render their episode figures through one shared helper,
`examples/_common.py::figure_episode`, so they pick a renderer at run time and
degrade the same way when path tracing is unavailable. See
[Rendering](rendering.md).

## 01 · I-JEPA on images

Pretrain, probe, and inspect. The point of the script is the *disagreement* between
its outputs: the loss falls by an order of magnitude while the collapse diagnostics
say the encoder collapsed, and the probe saturates before training even starts.

<figure markdown="span">
  ![Mask draws and training curve](../outputs/01_image_ijepa/training_curve.png){ width="600" }
</figure>

Writes: `masks.png`, `mask_draws.gif`, `training_curve.png`, `spectrum.png`,
`latent_pca.png`. → [Findings](../findings.md#i-jepa-on-images)

## 02 · V-JEPA on clips

Tube masking on 8-frame clips, comparing the short-range and long-range presets.

<figure markdown="span">
  ![Tube masks over a clip](../outputs/02_video_vjepa/tube_masks.gif){ width="420" }
  <figcaption>The tube is held across frames, so no visible frame contains the answer.</figcaption>
</figure>

Writes: `tube_masks.gif`, `tube_mask_frames.png`, `clip.gif`, `training_curves.png`,
`spectra.png`. → [Findings](../findings.md#video-tube-masking)

## 03 · Anti-collapse strategies

The same encoder trained five ways: `ema`, `sigreg` at two weights, `vicreg`, and
`none` as a control. This is the example to read first if you are deciding what
`collapse=` to use.

<figure markdown="span">
  ![Training curves by strategy](../outputs/03_collapse_strategies/training_curves.png){ width="600" }
  <figcaption>The run with the lowest loss is the one that learned nothing.</figcaption>
</figure>

Writes: `training_curves.png`, `diagnostics.png`, `spectra.png`.
→ [Findings](../findings.md#anti-collapse-strategies)

## 04 · An action-conditioned world model

Freeze the encoder from a JEPA, learn `(z, a) → z'` in its latent space, then
measure a free rollout against a no-op baseline at every horizon.

<figure markdown="span">
  ![Real against imagined rollout](../outputs/04_action_world_model/episode.gif){ width="420" }
  <figcaption>Real frames against the model's imagined latents, decoded only for the figure.</figcaption>
</figure>

Writes: `episode.gif`, `episode_frames.png`, `horizon_error.png`, `training_curve.png`.
→ [Findings](../findings.md#compounding-error)

## 05 · Planning

The full JEPA pipeline end to end: pretrain, learn dynamics, plan with CEM toward a
goal *image*, and evaluate against no-op, random and an oracle. Step 4 is what makes
it an evaluation rather than a demo. Latent cost going down proves nothing, since
the planner optimises latent cost by construction, so the script scores true
distance instead.

<figure markdown="span">
  ![Per-episode planning results](../outputs/05_planning/per_episode.png){ width="620" }
</figure>

Writes: `planning_episode.gif`, `policy_comparison.png`, `per_episode.png`.
→ [Findings](../findings.md#planning-in-the-synthetic-world)

## 06 · The same pipeline on a Franka arm

Everything from 04 and 05, on a real robot model in Newton. Needs
`pip install "xwm[newton]"`.

<figure markdown="span">
  ![Planning on the Franka arm](../outputs/06_franka_newton/planning_episode.gif){ width="480" }
</figure>

It picks its renderer from
[`which_backends()`](../reference/envs.md#xwm.envs.which_backends) at run time: where
OVRTX is available it writes path-traced GIFs, and where it is not it writes
supersampled Warp figures plus a USD stage to render offline. `XWM_RTX=0` skips the
path tracing; `XWM_RTX_SIZE` changes its resolution. See [Rendering](rendering.md).

Writes: `episode.gif`, `planning_episode.gif`, `episode_frames.png`,
`horizon_error.png`, `latent_pca.png`, `policy_comparison.png`, `training_curves.png`,
plus, at native render resolution, `episode_hq.gif` and `episode_frames_hq.png`, and
`episode.usd` for path tracing the same episode offline.
→ [Findings](../findings.md#franka-planning)

## 07 · TD-MPC2

Collect, train, plan, repeat, with reward instead of self-supervision. Compares the
planner against the policy prior alone, which is the comparison that justifies
planning at all.

<figure markdown="span">
  ![TD-MPC2 learning curve](../outputs/07_tdmpc2_franka/learning_curve.png){ width="600" }
</figure>

<figure markdown="span">
  [![TD-MPC2 planned episode](../outputs/07_tdmpc2_franka/episode_frames.png){ width="700" }](../outputs/07_tdmpc2_franka/episode_frames.png)
  <figcaption markdown="span">
  One planned episode: eight panels spaced evenly across its 21 frames rather than
  taken from the opening. The arm extends across the frame as the plan unfolds.
  Click through for the full render at 768 px per panel.
  </figcaption>
</figure>

Writes: `episode.gif` and `episode_frames.png` at figure quality,
`learning_curve.png`, `training_curve.png`, `policy_comparison.png`, and
`episode.usd` for path tracing the same episode offline.
→ [Findings](../findings.md#td-mpc2-on-the-franka-arm)

## 08 · MuZero

Act by running MCTS through the learned model, store what the *search* concluded
rather than what the policy did, and train the model to agree with it.

<figure markdown="span">
  ![MuZero learning curve](../outputs/08_muzero_franka/learning_curve.png){ width="600" }
</figure>

<figure markdown="span">
  [![MuZero searched episode](../outputs/08_muzero_franka/episode_frames.png){ width="700" }](../outputs/08_muzero_franka/episode_frames.png)
  <figcaption markdown="span">
  One episode driven by tree search, panelled the same way. The arm lifts and
  settles rather than extending: with 15 discrete actions the search found a short
  path, so there is less motion here than in 07's continuous plan. That is the
  policy, not the renderer.
  </figcaption>
</figure>

Writes: `episode.gif` and `episode_frames.png` at figure quality,
`learning_curve.png`, `training_curve.png`, `policy_comparison.png`, and
`episode.usd` for path tracing the same episode offline.
→ [Findings](../findings.md#muzero-on-the-franka-arm)

## 09 · A recorded dataset

Train on data a real robot produced — `lerobot/pusht`, in the same LeRobot layout
DROID and the Open X-Embodiment mirrors use — and put the horizon curve beside
the synthetic world built to abstract it.

<figure markdown="span">
  ![One episode from each source](../outputs/09_recorded_data/episode_frames.png){ width="820" }
  <figcaption>
    Recorded and synthetic, each at the resolution it was recorded. Figures come
    from the data's own resolution rather than the downsampled tensor the model
    trains on; the synthetic panel is driven by a scripted controller, so the
    frames show what the world is for rather than what random actions do.
  </figcaption>
</figure>

<figure markdown="span">
  ![Embedding spectra](../outputs/09_recorded_data/spectra.png){ width="560" }
  <figcaption>
    The most legible collapse diagnostic, and the reason this example prints
    <code>collapse_report</code> beside its horizon ratio: a ratio near 1 cannot
    tell a hard task from a dead encoder.
  </figcaption>
</figure>

Writes: `episode.gif` and `episode_frames.png`, `horizon_error.png`,
`spectra.png`, `latent_pca_{recorded,synthetic}.png`, both training curves, and
`horizon_error` / `collapse` as JSON and LaTeX. **The committed artifacts are
from the `gpu-recorded` preset**, not the laptop defaults — the two disagree
about the result, which is the finding.
→ [Findings](../findings.md#a-falling-prediction-loss-on-recorded-data-and-a-dead-encoder-under-it)

## Running at higher fidelity

```bash
XWM_IMG_SIZE=128 XWM_STEPS=4000 python examples/01_image_ijepa.py   # locally
modal run deploy/app_image.py --preset gpu                          # on a GPU
./deploy/run_all.sh                                                 # all nine
```

See [Running on GPU](gpu.md) for what each preset changes.

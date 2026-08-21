---
title: xwm
description: Action-conditioned world models for robotics, in JAX.
hide:
  - navigation
---

<div class="xwm-hero" markdown="0">
  <div class="xwm-hero__grid">
    <div class="xwm-hero__cell"></div>
    <div class="xwm-hero__cell xwm-hero__cell--masked"></div>
    <div class="xwm-hero__cell"></div>
    <div class="xwm-hero__cell xwm-hero__cell--masked"></div>
    <div class="xwm-hero__cell"></div>
    <div class="xwm-hero__cell xwm-hero__cell--masked"></div>
    <div class="xwm-hero__cell"></div>
    <div class="xwm-hero__cell xwm-hero__cell--masked"></div>
    <div class="xwm-hero__cell"></div>
  </div>
  <div>
    <div class="xwm-hero__word">xwm</div>
    <div class="xwm-hero__kicker">World Models</div>
  </div>
  <p class="xwm-hero__tagline">
    Action-conditioned latent world models in JAX. One dynamics interface,
    three training signals, four planners, and no decoder anywhere.
  </p>
</div>

## Overview

`xwm` is a JAX library for action-conditioned latent world models. A model is
three parts: an encoder \(h: O \rightarrow Z\), a dynamics model
\(d: Z \times A \rightarrow Z\), and whatever prediction heads the training signal
requires. Every objective and every planner operates in \(Z\).

The library contains no decoder and no pixel-reconstruction loss. Encoders accept
an arbitrary subset of the token grid, so masked positions cost nothing to
compute, and planners reach the dynamics through a single
\((z, a) \rightarrow z\) closure. That one closure is what lets a single set of
planners serve every model in the library.

<figure class="xwm-diagram" markdown="1">
  ![An action-conditioned latent world model](assets/world-model-light.svg#only-light){ width="820" } ![An action-conditioned latent world model](assets/world-model-dark.svg#only-dark){ width="820" }
  <figcaption>
    The observation is a real frame from the Franka FR3 environment in
    <code>xwm.envs</code>, at the resolution the encoder is given. Everything after
    the encoder happens inside <em>Z</em>, which is why one dynamics closure serves
    every family and every planner in the library.
  </figcaption>
</figure>

## Install

```bash
pip install xwm                 # core
pip install "xwm[dev]"          # core plus tests
pip install "xwm[newton]"       # core plus the Franka robot environment
```

Python 3.11 or newer, with `jax`, `equinox`, `optax` and `einops`. The optional
extras are listed in [Installation](getting-started/installation.md).

## Quick start

```python
import xwm

xwm.set_seed(0)

# Self-supervised: learns from observation alone, no reward.
model = xwm.families.jepa.lejepa(size="small", img_size=224, patch_size=16)

# Reward-driven, continuous actions: the natural fit for a robot arm.
agent = xwm.families.tdmpc2.tdmpc2(action_dim=7, observation="state", state_dim=20)

# Reward-driven, discrete actions, plans with tree search.
agent = xwm.families.muzero.muzero(n_actions=15, observation="state", state_dim=20)

trainer = xwm.training.Trainer(model, xwm.training.adamw(1e-4))
state, history = trainer.fit(batches, steps=10_000)
```

The [Quickstart](getting-started/quickstart.md) takes this from an untrained model
to a working planner in about twenty lines.

## Models

Three families share the same encoders, latent dynamics and planners. What
separates them is the signal that trains the latent space.

| family | learning signal | reward? | actions | planner |
| --- | --- | --- | --- | --- |
| [`jepa`](families/jepa.md) | its own future embeddings | no | continuous | CEM / MPPI |
| [`tdmpc2`](families/tdmpc2.md) | reward and TD value | yes | continuous | MPPI |
| [`muzero`](families/muzero.md) | search-improved targets | yes | discrete | MCTS |

They are complementary rather than competing. JEPA needs no reward, so it can
pretrain on passive video, which is abundant and unlabelled. TD-MPC2 and MuZero
need interaction, but they learn a value function, so their planner can see past
its own horizon. A JEPA encoder is a reasonable initialisation for either, and it
is one argument: `tdmpc2(encoder=pretrained)`.

## Documentation

<div class="grid cards" markdown>

-   :material-rocket-launch-outline:{ .lg .middle } __Getting started__

    ---

    Install the right extras, build a model, train it, and plan with it.

    [:octicons-arrow-right-24: Installation](getting-started/installation.md) ·
    [Quickstart](getting-started/quickstart.md) ·
    [Conventions](getting-started/conventions.md)

-   :material-lightbulb-outline:{ .lg .middle } __Concepts__

    ---

    What a JEPA predicts, why it does not collapse, how planning works, and what
    to measure instead of the loss.

    [:octicons-arrow-right-24: Concepts](concepts/index.md) ·
    [Masking](concepts/masking.md) ·
    [Collapse](concepts/collapse.md) ·
    [Planning](concepts/planning.md)

-   :material-robot-industrial-outline:{ .lg .middle } __Guides__

    ---

    Training, a Franka FR3 arm in Newton, rendering, figures, and running the
    whole thing on a GPU.

    [:octicons-arrow-right-24: Training](guides/training.md) ·
    [Robotics](guides/robotics.md) ·
    [Rendering](guides/rendering.md) ·
    [Examples](guides/examples.md)

-   :material-chart-line-variant:{ .lg .middle } __Results and reference__

    ---

    Every number this library claims, including the negative ones, and the full
    API with the paper each module follows.

    [:octicons-arrow-right-24: Findings](findings.md) ·
    [API reference](reference/index.md)

</div>

## Library layout

| module | contents |
| --- | --- |
| [`xwm.core`](reference/core.md) | types, base modules, EMA targets, rollouts, the default key |
| [`xwm.nn`](reference/nn.md) | attention, transformers, RoPE, patch and tubelet embeddings, SimNorm |
| [`xwm.encoders`](reference/encoders.md) | observation to latent: image, video, state |
| [`xwm.dynamics`](reference/dynamics.md) | **`(z, a) → z'`**, transformer or MLP |
| [`xwm.heads`](reference/heads.md) | reward, value, policy, Q-ensemble, categorical scalars |
| [`xwm.masking`](reference/masking.md) | what a JEPA predicts: blocks, tubes, temporal splits |
| [`xwm.objectives`](reference/objectives.md) | latent prediction, SIGReg, VICReg, InfoNCE |
| [`xwm.families`](reference/families.md) | `jepa`, `tdmpc2`, `muzero`, and a registry |
| [`xwm.planning`](reference/planning.md) | CEM, MPPI, gradient planning, MPC, MCTS, latent costs |
| [`xwm.training`](reference/training.md) | `Trainer`, schedules, `TrainState`, `ReplayBuffer` |
| [`xwm.envs`](reference/envs.md) | a Franka FR3 arm in [Newton](https://github.com/newton-physics/newton) |
| [`xwm.data`](reference/data.md) | batch streams and a synthetic controllable world |
| [`xwm.metrics`](reference/metrics.md) | probes and collapse diagnostics |
| [`xwm.plots`](reference/plots.md) | figures, GIFs, JSON and LaTeX tables |
| [`xwm.tools`](reference/tools.md) | checkpointing, model summaries |

`xwm.dynamics` is the centre of the library rather than an add-on. Every family
consumes a \((z, a) \rightarrow z\) model from it, and every planner consumes
nothing else. Changing family changes how that model is *trained*, never how it is
*used*.

## Project

- **Source**: [github.com/Kleyt0n/xwm](https://github.com/Kleyt0n/xwm)
- **Package**: [pypi.org/project/xwm](https://pypi.org/project/xwm/)
- **License**: Apache-2.0
- **Citations**: [one per module](reference/index.md#citations), sitting beside the
  implementation it describes

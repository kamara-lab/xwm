<p align="center">
  <img src="https://raw.githubusercontent.com/kamara-lab/xwm/refs/heads/main/assets/logo.svg" alt="xwm" width="300">
</p>

<p align="center"><strong>Action-conditioned world models for robotics.</strong></p>

<p align="center">
  <a href="https://pypi.org/project/xwm/"><img alt="PyPI version" src="https://img.shields.io/pypi/v/xwm?style=flat-square&color=059669&labelColor=ffffff"></a>
  <a href="https://www.python.org/downloads/"><img alt="Python 3.11+" src="https://img.shields.io/badge/python-3.11%2B-059669?style=flat-square&labelColor=ffffff"></a>
  <a href="https://github.com/jax-ml/jax"><img alt="JAX" src="https://img.shields.io/badge/built%20on-JAX%20%20-059669?style=flat-square&labelColor=ffffff"></a>
  <a href="#tests"><img alt="Tests" src="https://img.shields.io/badge/tests-388%20passing-059669?style=flat-square&labelColor=ffffff"></a>
  <a href="https://docs.astral.sh/ruff/"><img alt="Ruff" src="https://img.shields.io/badge/lint-ruff-059669?style=flat-square&labelColor=ffffff"></a>
  <a href="#license"><img alt="Apache 2.0" src="https://img.shields.io/badge/license-Apache--2.0-059669?style=flat-square&labelColor=ffffff"></a>
</p>

---

`xwm` is a JAX-based library for **action-conditioned latent world models**: an encoder $h: O \rightarrow Z$, a dynamics model $d: Z \times A \rightarrow Z$, and whatever prediction heads the training signal requires. Every objective and every planner operates in $Z$. The library contains no decoder and no pixel-reconstruction loss. Encoders accept an arbitrary subset of the token grid, so masked positions cost nothing to compute, and planners reach the dynamics through a single $(z, a) \rightarrow z$ closure, which is what lets one set of planners serve every model.

The components are independently useful: ViT encoders over 2D patches or 3D tubelets and MLP encoders over state vectors; transformer or residual-MLP dynamics; categorical reward and value heads, pessimistic Q-ensembles, squashed-Gaussian policies; latent-prediction, SIGReg, VICReg, InfoNCE and TD objectives; CEM, MPPI, gradient and PUCT-MCTS planners; a family-agnostic trainer with EMA targets, parameter freezing and a trajectory replay buffer.

<p align="center">
  <img src="https://raw.githubusercontent.com/kamara-lab/xwm/refs/heads/main/docs/assets/world-model-dark.svg" alt="Encoder, latent dynamics and heads" width="720">
</p>

## Install

Add it to your own project:

```bash
uv add xwm                     # core
uv add "xwm[plots]"            # figures, GIFs, tables
uv add "xwm[newton]"           # the Franka robot environment
uv add "xwm[data]"             # recorded datasets: DROID, LIBERO, OGBench, OXE
```

Or work on it from a clone, where `uv.lock` pins the whole environment:

```bash
git clone https://github.com/kamara-lab/xwm && cd xwm
uv sync --extra dev                       # core + tests
uv sync --extra dev --extra newton        # + the Franka robot environment
uv run pytest                             # run in that environment
```

`--all-extras` is the one combination to avoid: it pulls in `render`, whose `ovrtx` ships as an sdist and wants a graphics-capable NVIDIA GPU, so it will try to build on machines that can never use it. Add `--extra render` deliberately, on a host that has one.

Python ≥ 3.11, `jax`, `equinox`, `optax`, `einops`.

## Quick start

```python
import xwm

xwm.set_seed(0)

# Self-supervised: learns from observation alone, no reward.
model = xwm.families.jepa.lejepa(size="small", img_size=224, patch_size=16)

# Reward-driven, continuous actions -- the natural fit for a robot arm.
agent = xwm.families.tdmpc2.tdmpc2(action_dim=7, observation="state", state_dim=20)

# Reward-driven, discrete actions, plans with tree search.
agent = xwm.families.muzero.muzero(n_actions=15, observation="state", state_dim=20)

trainer = xwm.training.Trainer(model, xwm.training.adamw(1e-4))
state, history = trainer.fit(batches, steps=10_000)
```

## Models

All three share the same encoders, latent dynamics and planners. What separates them is **what signal trains the latent space**.

| family | learning signal | reward? | planner |
| --- | --- | --- | --- |
| [`jepa`](xwm/families/jepa) | its own future embeddings | no | CEM / MPPI |
| [`tdmpc2`](xwm/families/tdmpc2) | reward + TD value | yes | MPPI |
| [`muzero`](xwm/families/muzero) | search-improved targets | yes | MCTS |

They are complementary rather than competing. JEPA needs no reward, so it can pretrain on passive video, abundant and unlabelled. TD-MPC2 and MuZero need interaction, but they learn a value function, so their planner can see past its own horizon. A JEPA encoder is a reasonable initialisation for either: `tdmpc2(encoder=pretrained)` is one argument.

`xwm.families.available()` lists every registered model;
`xwm.families.create(name, **kwargs)` builds one by name.

## Layout

| module | contents |
| --- | --- |
| `xwm.core` | types, base modules, EMA targets, rollouts, the default key |
| `xwm.nn` | attention, transformers, RoPE, patch/tubelet embeddings, SimNorm |
| `xwm.encoders` | observation → latent: image, video, state |
| `xwm.dynamics` | **`(z, a) → z'`** — transformer or MLP |
| `xwm.heads` | reward, value, policy, Q-ensemble, categorical scalars |
| `xwm.masking` | what a JEPA predicts: blocks, tubes, temporal splits |
| `xwm.objectives` | latent prediction, SIGReg, VICReg, InfoNCE |
| `xwm.families` | `jepa`, `tdmpc2`, `muzero`, and a registry |
| `xwm.planning` | CEM, MPPI, gradient planning, MPC, MCTS, latent costs |
| `xwm.training` | `Trainer`, schedules, `TrainState`, `ReplayBuffer` |
| `xwm.envs` | a Franka FR3 arm in [Newton](https://github.com/newton-physics/newton), and the `Env` protocol |
| `xwm.data` | batch streams and synthetic controllable worlds |
| `xwm.tasks` | benchmark tasks: dataset + environment + metric |
| `xwm.bench` | goal-reaching evaluation under a step budget |
| `xwm.config` | experiment configs, TOML files and overrides |
| `xwm.metrics` | probes and collapse diagnostics |
| `xwm.plots` | figures, GIFs, JSON/LaTeX tables |
| `xwm.tools` | checkpointing, model summaries |

`xwm.dynamics` is the centre of the library rather than an add-on: every family consumes a $(z, a) \rightarrow z$ model from it, and every planner consumes nothing else. Changing family changes how that model is *trained*, never how it is *used*.

## Concepts

### What a JEPA predicts

A mask sampler splits the token grid into a visible context and target blocks. The context encoder computes only the visible tokens, which is where the speedup over reconstruction comes from.

| sampler | used by | idea |
| --- | --- | --- |
| `MultiBlockMask2d` | I-JEPA | large 2-D blocks, too big to interpolate from neighbours |
| `TubeMask3d` | V-JEPA | a spatial region extended through time, so no visible frame contains the answer |
| `TemporalSplit` | V-JEPA 2-AC | see a prefix, predict whole future frames |
| `RandomMask` | baselines | uniform random tokens |

Masks are batch-shared and statically shaped, so a training step compiles once. Sampling is combinatorial host-side work and happens in `model.prepare_batch()`, outside `jit`; the `Trainer` calls it for you.

### Why it doesn't collapse

Predicting a representation from a representation has a trivial solution: emit a constant. `collapse=` selects the countermeasure.

| option | used by | mechanism | teacher? |
| --- | --- | --- | --- |
| `"ema"` | I-JEPA, V-JEPA | targets from a slowly-moving copy, gradients cut | yes |
| `"sigreg"` | LeJEPA | a distributional penalty forbids the constant solution | **no** |
| `"vicreg"` | VICReg | variance + covariance penalties | no |
| `"none"` | — | control, for watching collapse happen | no |

**SIGReg** replaces EMA teachers, stop-gradients, centering and sharpening with one statement: *the embedding distribution should be an isotropic Gaussian*. It is enforced by a sketch — for `z ~ N(0, I_D)` and any unit vector `v`, the projection `⟨z, v⟩` is exactly `N(0, 1)` regardless of `D` — so it draws random directions, projects the batch onto each, and penalises deviation from a standard normal. Isotropy and unit scale both fall out, the cost is linear in batch size, and there is one coefficient instead of a schedule.

```python
xwm.objectives.sigreg(z, key, n_proj=256, statistic="epps_pulley")
```

### Planning

`model.dynamics_fn()` hands a planner a plain `(z, a) -> z'` closure. Everything in `xwm.planning` is jittable — candidates are `vmap`ed and refinement is a `lax.fori_loop` — so a plan is one device call.

```python
planner = xwm.planning.CEM(horizon=8, action_dim=7, n_samples=512, n_elites=64)
cost = xwm.planning.goal_cost(model.encode(goal_image), kind="l2")
plan = planner.plan(key, model.dynamics_fn(), model.encode(observation), cost)
```

| planner | actions | notes |
| --- | --- | --- |
| `CEM`, `MPPI` | continuous | sample whole sequences; what JEPA and TD-MPC2 use |
| `GradientPlanner` | continuous | differentiates the rollout; happy to exploit model error |
| `MCTS` | discrete | grows a tree; what MuZero uses |

`run_mpc` closes the loop with replanning and warm starts. For value-based agents, `return_cost` scores candidates by predicted reward plus a terminal value bootstrap — the term that lets a horizon-3 planner act as though it saw further.

### Diagnostics

**The loss is not the metric.** A collapsing encoder drives its prediction loss *down* — it is predicting its own degenerate output.

```python
xwm.metrics.collapse_report(z)
# {'rankme': ..., 'rank_ratio': ..., 'feature_std': ..., 'mean_cosine': ...}
```

`feature_std → 0` and `mean_cosine → 1` both mean collapse; `rankme` is the effective rank of the spectrum. All are reported because each misses a case the others catch — `rankme` is computed after centring, so a constant offset is invisible to it. A linear probe is *not* a collapse detector: `ridge_probe` standardises features, so it amplifies a nearly-dead signal back to full scale.

## Robotics

`xwm.envs` wraps a **Franka Emika FR3** in Newton (NVIDIA Warp), observed either as pixels or as a 20-D proprioceptive state vector. A dense reach task supplies the reward the value-based families need.

```python
env = xwm.envs.FrankaEnv(xwm.envs.FrankaConfig(image_size=64))

data = xwm.envs.franka_sequences(env, 320, 8, seed=0)   # for JEPA
env.state_observation(), env.reward(action), env.goal_distance()   # for RL
```

`franka_sequences` returns exactly what `xwm.data.sprite_sequences` does, so it drops straight into any family.

Two synthetic worlds cover the same ground on CPU, without a simulator. `xwm.data.PushWorld` is planar pushing: the puck moves only when the pusher touches it, which puts a hinge in the dynamics that `SpriteWorld`'s linear ones cannot have, and it carries a dense reward. `xwm.data.MazeWorld` is sparse-reward navigation through corridors, shaped after OGBench's `antmaze` navigate tasks. Both were tuned by measuring the task rather than by eye: on `PushWorld`, a single sequence of random actions solves 3% of starts and the best of 80 solves 94%, which is the gap a planner has to exploit.

### Recorded datasets

`xwm.datasets` reads the corpora the field benchmarks on, in the four formats they come in, and returns exactly the field layout `sprite_sequences` does.

```python
xwm.datasets.describe("lerobot/droid-100")               # size, episodes, licence, citation
data = xwm.datasets.create("libero/10", length=16, limit=20, resize=64)
buffer = xwm.datasets.to_replay_buffer(xwm.datasets.create("robomimic/lift", length=32))
```

| reader | corpora | needs |
| --- | --- | --- |
| `lerobot` | DROID, LIBERO, Push-T, and the Open X-Embodiment mirrors (RT-1, Bridge V2, Language-Table, TACO-Play, Berkeley UR5) | `xwm[data]` |
| `offline` | OGBench — 2-D mazes to a 69-D humanoid, state and pixels | nothing at all |
| `hdf5` | LIBERO (including the 90-task suite, which has no LeRobot conversion) and RoboMimic, in their native HDF5 | `xwm[data]` |
| `minari` | D4RL through its successor — pointmaze, antmaze, halfcheetah, FrankaKitchen | `minari` |
| `rlds` | Open X-Embodiment as published | `tensorflow-datasets` |

Mixtures are what OXE is actually for, so `xwm.datasets.mixture(xwm.datasets.OXE_MIXTURE, action_dim=7, resize=64)` interleaves several members by weight, zero-padding narrower action spaces to a common width; `xwm.datasets.stream(name)` is the same registry access without materialising anything.

Every registered source was resolved against the live host, and the episode counts come from the datasets' own metadata rather than from their papers. The one thing to know before using it: recorded formats store one action per frame, including a last one whose result was never recorded, and `xwm` needs `T - 1` actions for `T` frames. The readers drop it and the contract check refuses an episode that did not — a silent one-step shift looks exactly like underfitting.

### Rendering

Training and figures want opposite things from a renderer, so there are two paths. Use `xwm.envs.which_backends()` to see what is installed.

| backend | speed | quality | needs |
| --- | --- | --- | --- |
| `warp` | ms/frame | hard shadows, flat ambient | nothing — CPU or GPU |
| `rtx` | seconds/frame | path-traced: soft shadows, ambient occlusion, materials | `ovrtx`, `pyglet`, a graphics-capable NVIDIA GPU |
| `usd` | export only | whatever your offline renderer does | `usd-core` |

```python
env.observe()                        # warp, at config.image_size -- for training
env.render(384, samples=3)           # warp, supersampled -- for a clean figure

with env.high_quality_renderer(backend="rtx", size=(768, 768)) as r:
    r.add(env.state)                 # path traced, one frame per state
frames = r.frames                    # (T, 3, H, W) -- feeds save_gif directly

with env.high_quality_renderer(backend="usd", output_path="ep.usd") as r:
    r.add(env.state)                 # a stage to render in Omniverse or Blender
```

`env.render` casts one ray per pixel, so `samples` renders at `samples×` and averages down — the only anti-aliasing the Warp raytracer has. It is the right tool for observations and for tidy figures, but it will not produce a photorealistic image: for that use `rtx`, or export a USD stage and render it offline. Every backend shares one camera definition (`env.camera_framing`), so the path-traced figure and the observations the model trains on show the same view from the same place.

Because the physics is deterministic given a seed and an action sequence, a path-traced figure is produced by *replaying* an episode rather than by storing its pixels — the render is of the same episode the numbers came from. Example 06 writes both: `episode_frames.png` is what the encoder sees, `episode_rtx.gif` and `planning_episode_rtx.gif` are what the robot is doing. Set `XWM_RTX=0` to skip them, or `XWM_RTX_SIZE` to change the resolution. `deploy/app_render.py` runs every available backend on a GPU and writes the results side by side; `app_render.py::vulkan_probe` reports in about a minute whether OVRTX can get a device at all.

`rtx` needs more than an NVIDIA GPU: it needs graphics access. Many GPU cloud containers — Modal's among them — expose a compute-only device set (no `/dev/nvidia-modeset`), which satisfies CUDA but not NVIDIA's Vulkan driver, so OVRTX cannot create an instance there however complete the library stack is. Example 06 therefore picks its renderer from `which_backends()` at run time, and where OVRTX is unavailable it writes supersampled Warp figures plus `episode.usd` to path trace offline. See `docs/findings.md` for the diagnosis.

The two-stage V-JEPA 2-AC recipe — learn a representation from passive video, freeze it, learn action-conditioned dynamics in its latent space — is one call. Freezing is not only a compute saving: with the encoder fixed the prediction targets are fixed functions of the observations, so the dynamics model has nothing to gain from degrading the representation.

```python
model = xwm.families.jepa.action_world_model(
    action_dim=7, encoder=pretrained_encoder, freeze_encoder=True,
)
trainer.n_trainable == model.dynamics.n_params   # the encoder gets no optimizer state
```

Training mixes **teacher forcing** (one step from ground-truth latents — a dense signal) with **rollout** (the full horizon from a single latent, the model consuming its own predictions — the only term that penalises compounding error).

## Benchmark

One protocol, so two world models can be compared. Reset the environment to a
state a held-out recording visited, show the model -- as an *observation* -- a
state that recording reached later, and give its planner a fixed budget of
environment steps. Success is a ground-truth predicate on simulator state the
model never sees.

```bash
xwm tasks list                              # what can be run
xwm train configs/pusht/jepa.toml --eval    # train, then measure
xwm eval runs/jepa-3afec466e5a2             # measure again, or differently
```

```
 policy  success  gap closed  final dist  best dist  steps  seconds
planner    0.000       0.012       0.283      0.216      -   71.900
 replay    1.000       0.125       0.233      0.000  5.250    0.090
   noop    0.000       0.002       0.309      0.309      -    0.090
 random    0.000       0.029       0.305      0.287      -    0.090
```

Three baselines run through the same loop on the same instances, because a
success rate without its floor is unreadable. **Read `replay` first**: it
executes the demonstrator's own recorded actions, so it should succeed almost
always -- and when it does not, the environment is not being reset faithfully
and no model number from that task means anything yet.

| module | contents |
| --- | --- |
| `xwm.tasks` | a task: dataset + environment + metric, and the frameskip, resolution, goal offset and budget written down once |
| `xwm.bench` | instance sampling, the control loop, the four policies, the results schema |
| `xwm.config` | frozen dataclasses, TOML files, `key.path=value` overrides |

See **[docs/guides/benchmark.md](docs/guides/benchmark.md)** for why the goal
comes from a recording, why actions are grouped rather than repeated, and what
`replay` is protecting you from; **[docs/guides/metrics.md](docs/guides/metrics.md)**
defines every number above, with equations — including the two that are
misleading on their own.

## Training

`Trainer` is family-agnostic: it needs only `loss`, `prepare_batch` and `trainable`. It owns the `jit` boundary, the EMA teacher, and the parameter filter, so frozen submodules never reach the optimizer.

```python
trainer = xwm.training.Trainer(model, xwm.training.adamw(xwm.training.cosine_warmup(1e-3, 1000)))
state, history = trainer.fit(batches, steps=1000)
```

Batches come from `xwm.data.iter_batches` for a fixed dataset, or from `xwm.training.ReplayBuffer` for the reward-driven families, whose losses need contiguous slices of a single episode. The buffer rejects slices that straddle an episode boundary — training a dynamics model to predict through a reset is the one transition it can never get right. `xwm.datasets.to_replay_buffer` fills it from a recorded corpus, one clip per episode.

### Keys

`key=` is optional wherever a model is *built*. Omit it and the key comes from an ambient source; pass one and nothing ambient is touched.

```python
xwm.set_seed(0)
model = xwm.families.jepa.ijepa(img_size=64)                    # ambient
other = xwm.families.jepa.ijepa(img_size=64, key=jr.PRNGKey(7)) # explicit

with xwm.seed(123):                                             # scoped
    model = xwm.families.jepa.ijepa(img_size=64)
```

The source advances on every draw — it has to, or every transformer block would be initialised identically — so a fixed sequence of calls under a fixed seed is reproducible, but inserting a construction shifts everything built after it. Pass explicit keys for anything that must survive refactors.

Only *construction* defaults. `loss`, `sigreg` and the planners still require a key, because those are consumed inside `jit`, where a key drawn at trace time would be baked in as a constant and reused for every step.

## Examples

Examples 01–05 run on CPU against the synthetic worlds in `xwm.data`, so there is no dataset to download. 06–08 need the `newton` extra and download the Franka asset on first run. 09 needs the `data` extra and downloads ~30 MB of recorded robot data.

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

Measured results, including the negative ones, are collected in **[docs/findings.md](docs/findings.md)**.

## Running on GPU

Each experiment is its own [Modal](https://modal.com) app, so the nine can run
concurrently on separate GPUs and be started, watched and stopped independently.

```bash
./deploy/run_all.sh                            # all nine, gpu preset
modal run deploy/app_tdmpc2.py --preset xl     # one, at higher fidelity
```

Presets (`cpu-parity`, `gpu`, `xl`) raise resolution, episode count, model size
and render quality through `XWM_*` environment variables, so there is one copy of
each pipeline rather than a laptop version and a cluster version. `cpu-parity`
exists to isolate hardware from settings when comparing runs.

## Conventions

- **Modules are unbatched.** Written for a single sample and `vmap`ed by the caller, the Equinox idiom. Batch-level entry points are the methods named `loss`.
- **Shapes.** Images `(C, H, W)`, clips `(T, C, H, W)`, token sequences `(N, D)`, flat latents `(D,)`, actions `(A,)`. Masks are `int32` index arrays.
- **Immutability.** `model.eval_mode()` *returns* a dropout-free copy.

## Tests

```bash
uv run pytest
```

The tests are written to fail on broken *behaviour*, not just broken shapes: mask samplers must never leak a target token into the context, dynamics must respond to their action input, planners must reach a reachable goal, MCTS must find a payoff one step away, frozen parameters must not move, SIGReg must actually pull a skewed distribution toward isotropy, and a dataset reader must drop the final recorded action rather than shift a whole corpus by one step.

Two groups are gated. The Franka tests need the `newton` extra and skip without it. The dataset tests that download run only under `XWM_DATASET_TESTS=1`; the rest of them exercise every reader against fixtures written into `tmp_path` in the real on-disk formats — a real parquet shard, a real mp4, a real HDF5 — because a reader tested against a mock of a format is a reader tested against a belief about the format.

## References

Every module carries a `References` block in its docstring naming the paper the code follows, so the citation sits beside the implementation — try `help(xwm.families.tdmpc2.model)`.

| model | paper |
| --- | --- |
| I-JEPA | Assran et al., CVPR 2023 · [arXiv:2301.08243](https://arxiv.org/abs/2301.08243) |
| V-JEPA | Bardes et al., 2024 · [arXiv:2404.08471](https://arxiv.org/abs/2404.08471) |
| V-JEPA 2 / -AC | Assran et al., *V-JEPA 2*, 2025 |
| LeJEPA | Balestriero & LeCun, 2025 |
| TD-MPC2 | Hansen, Su & Wang, ICLR 2024 · [arXiv:2310.16828](https://arxiv.org/abs/2310.16828) |
| TD-MPC | Hansen, Wang & Su, ICML 2022 · [arXiv:2203.04955](https://arxiv.org/abs/2203.04955) |
| MuZero | Schrittwieser et al., Nature 2020 · [arXiv:1911.08265](https://arxiv.org/abs/1911.08265) |
| Sampled MuZero | Hubert et al., ICML 2021 · [arXiv:2104.06303](https://arxiv.org/abs/2104.06303) |
| VICReg | Bardes, Ponce & LeCun, ICLR 2022 · [arXiv:2105.04906](https://arxiv.org/abs/2105.04906) |

Component-level citations — SimNorm, two-hot categorical scalars, REDQ, SAC, MPPI, PUCT, Epps–Pulley, RankMe, ViT/ViViT, MAE, RoPE, LayerScale, Mish — live in the docstrings of the modules that implement them.

Simulation: [Newton](https://github.com/newton-physics/newton) with a Franka Emika FR3; MuJoCo via `mujoco_warp` where a CUDA GPU is available, Featherstone otherwise.

## Contributors

<a href="https://github.com/kamara-lab/xwm/graphs/contributors">
  <img alt="Contributors to surface" src="https://contrib.rocks/image?repo=kamara-lab/xwm">
</a>

## Supported by

Get in touch kleyton.vsc@gmail.com

## License

Apache-2.0

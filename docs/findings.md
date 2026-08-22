# Findings

Measured results and the diagnoses behind them. Numbers here come from runs whose
settings are recorded in `deploy/_shared.py`, unless a section says otherwise —
at least one records a smoke run at the laptop defaults, and says so, because a
labelled negative result beats a quiet omission. Where a claim was overturned by
a later run, both are kept.

## OVRTX path tracing does not run on a compute-only GPU container

The Warp raytracer that produces observations casts one ray per pixel: hard
shadows, flat ambient, no global illumination. Reaching the quality of Newton's
own documentation renders means OVRTX, Newton's path tracer, so `xwm.envs`
supports it as a second backend alongside a USD export. It renders on a
workstation GPU. It does not render on Modal, and the reason is worth recording
because nothing in the error message says it.

OVRTX is a Vulkan application. Getting to the real failure took four diagnostic
layers, each of which looked like the answer:

| symptom | cause | fix |
| --- | --- | --- |
| `No module named 'pyglet'` | `ViewerRTX` imports `pyglet.math` for camera vectors even headless | install `pyglet` |
| `libOpenGL.so.0: cannot open shared object` | the USD resolver plugin links it | `apt install libopengl0` |
| `createDevices failed`, "GPUs do not support RayTracing" | `vulkan-tools` pulls in Mesa's software ICDs and the loader enumerated llvmpipe | pin `VK_DRIVER_FILES` to the NVIDIA manifest |
| `ERROR_INCOMPATIBLE_DRIVER`, "Found no drivers!" | n/a | none |

The last one is the wall. The loader opens the NVIDIA ICD and then gets a null
`vkCreateInstance` back from it:

```
loader_scanned_icd_add: Could not get 'vkCreateInstance' via
'vk_icdGetInstanceProcAddr' for ICD libGLX_nvidia.so.0
```

The container is not missing libraries. `libnvoptix.so.1`,
`libnvidia-rtcore.so` and `libnvidia-glvkspirv.so` are all present, so the entire
ray-tracing stack is injected. What is missing is a device node:

```
/dev/nvidia2  /dev/nvidiactl  /dev/nvidia-uvm     # present: compute
/dev/nvidia-modeset                               # absent: graphics
```

That is a compute-only device set. CUDA is satisfied by it; NVIDIA's Vulkan
implementation is not, and refuses to create an instance. Setting
`NVIDIA_DRIVER_CAPABILITIES=all` in the image does not help, because that
variable is read by the container runtime when it creates the container and
injects device nodes. By the time an image layer or the process can set it, the
decision has been made. It is a property of the platform, not of the GPU model.

So the backends split by host, not by preference: OVRTX on a workstation or any
container with graphics capability, and on Modal the best available quality is
the Warp raytracer at figure resolution with supersampling, plus a USD stage that
can be path traced offline. Example 06 chooses at run time from
`which_backends()` rather than assuming, and `app_render.py::vulkan_probe`
answers the question in a minute instead of in a render.

## The Franka planning result varies by more than its headline number

Three GPU runs of example 06 at the identical `gpu` preset, differing only in
run-to-run nondeterminism (JAX reductions on GPU are not bit-reproducible):

| run | probe R² | no-op | best planner | gap closed |
| --- | --- | --- | --- | --- |
| first | 0.413 | 0.246 m | 0.178 m | +28% |
| second | 0.385 | 0.246 m | 0.216 m | +12% |
| third | 0.366 | 0.246 m | 0.193 m | +22% |

Two things are stable across all three runs: the ordering, in that the planner
beats both the no-op and the random baseline every time, and the action penalty's
sign, in that `a_pen=0.05` wins every time. The magnitude is not: +28%, +12%,
+22%, a spread of 16 points around an effect of roughly 21. Quoting any single
run's number overstates the precision by more than a factor of two, so the honest
claim is that planning over these latent dynamics closes something in the region
of 10-30% of the gap, with a spread comparable to the effect itself.

The no-op baseline is identical to the millimetre in all three runs (0.246 m),
because it is seeded and model-independent. The spread is therefore entirely in
what was learned, not in how it was measured.

This is also why example 06 derives its printed conclusion from the measured
numbers instead of stating one. The hardcoded verdict ("the planner does not beat
doing nothing") was written when it was true at laptop scale, and survived into
GPU runs where the table printed immediately above it said otherwise.

## The MuJoCo solver drops contacts on this scene

Every GPU Franka run prints hundreds of solver buffer overflows:

```
narrowphase overflow - please increase nconmax to 134 or naconmax to 134
nefc overflow - please increase njmax to 69
```

These are not cosmetic. The contact and constraint buffers are sized from the
initial state, and when the arm folds into self-contact the extra contacts are
discarded rather than solved, so the physics being learned is quietly not the
physics MuJoCo would produce with adequate buffers. `SolverMuJoCo` accepts
`nconmax` and `njmax`, and uses the larger of the supplied and estimated values,
so this is a one-line fix, but it changes the dynamics, and therefore every
number above, so it has not been applied to the results on this page.

## I-JEPA on images

128×128 sprite images, 4000 steps, 384-dim embeddings, EMA teacher.

| quantity | value |
| --- | --- |
| prediction loss, first → last | 0.5526 → 0.0433 |
| probe R², before → after training | 0.9991 → 0.8216 |
| rankme | 10.61 |
| rank_ratio | 0.028 |
| feature_std | 0.0398 |
| mean_cosine | 0.9902 |

Two things are wrong here in ways worth stating plainly.

**The loss fell by an order of magnitude and the encoder collapsed.**
`mean_cosine = 0.990` and `rank_ratio = 0.028`, which is 28 effective dimensions
out of 384, describe a largely degenerate representation. The prediction loss reports success
because predicting a nearly-constant target is nearly free. This is not a bug in
I-JEPA or in this implementation; it is what the EMA-teacher mechanism costs at
this scale, and it is why LeJEPA exists.

**The probe went *down* and means nothing either way.** It starts at R² = 0.999 on
a *randomly initialised* encoder, because in this toy world the agent's position is
nearly a linear function of the red channel and
[`ridge_probe`](reference/metrics.md#xwm.metrics.ridge_probe) standardises features
before fitting. A high probe score is not evidence of a healthy representation, and
this row is kept precisely to show that.

## Anti-collapse strategies

2048 sprite images, 4000 steps, 384-dim embeddings. The same encoder, trained five
ways.

| strategy | pred_loss | reg | feature_std | mean_cosine | rankme (of 384) | probe R² |
| --- | --- | --- | --- | --- | --- | --- |
| ema (I-JEPA) | 0.0615 | 0.0 | 0.1381 | 0.9565 | 21.40 | 0.1880 |
| sigreg w=5 (LeJEPA) | 0.0006 | 0.0128 | 0.0137 | 0.2802 | 56.93 | 0.9408 |
| sigreg w=20 (LeJEPA) | 0.0319 | 0.0058 | 0.0379 | 0.1051 | 93.08 | 0.9602 |
| vicreg w=1 | 0.0772 | 0.6216 | 0.1369 | 0.8752 | 210.18 | 0.3293 |
| none (control) | 2.0e-08 | 0.0 | 1.7e-05 | 1.0000 | 91.16 | 0.9818 |

**The best prediction loss belongs to the worst representation.** The control run
beats every real strategy by seven orders of magnitude and has learned nothing at
all: every embedding is the same vector (`mean_cosine = 1.0000`,
`feature_std = 1.7e-05`). Its `rankme` of 91 is the clearest available
demonstration that rankme is computed after centring: the numerical dust around a
constant has structure, and rankme measures it.

Reading the rest: SIGReg at `w=20` is the healthiest run on the page, with the
highest effective rank among the non-degenerate rows, the lowest mean cosine and
the best probe. The
EMA teacher at `mean_cosine = 0.957` has largely collapsed too. VICReg reaches the
highest raw rank of any row but spreads it thin: `mean_cosine` stays at 0.875 and
the probe only reaches 0.329.

`reg_weight` also behaves monotonically for SIGReg: 5 → 20 trades prediction loss
(0.0006 → 0.0319) for rank (57 → 93). That is the trade-off the single coefficient
is meant to expose.

## Video tube masking

V-JEPA on 128×128×8 clips, 4000 steps.

| preset | tubes | tokens/tube | context | loss | feature_std | mean_cosine | rankme |
| --- | --- | --- | --- | --- | --- | --- | --- |
| short-range | 8 | 36 | 69 | 0.4512 | 0.7331 | 0.2100 | 15.14 |
| long-range | 2 | 100 | 86 | 0.2726 | 0.5900 | 0.4688 | 22.68 |

The long-range preset wins on both axes at once, with lower loss *and* higher
effective rank, which is not the trade-off one might expect from "fewer, harder
targets".
The mechanism is the context: 2 large tubes leave 86 visible tokens against 69 for
8 small ones, so each individual prediction is better supported. Its `mean_cosine`
is higher (0.469 vs 0.210), so the representation is less spread out; on this
evidence the short-range preset produces more decorrelated features and the
long-range preset a better-fit predictor.

Neither is collapsed, which is the more important observation: `feature_std` at
0.73 and 0.59 is a working representation in both cases, unlike the image runs
above. Eight frames of a moving sprite is a harder prediction problem than a still
image, and the constant solution is correspondingly less attractive.

## Compounding error

A free rollout on a frozen encoder, predicting \(h\) steps from a single latent with
the model consuming its own predictions, scored against a **no-op baseline** that
assumes nothing changes. Held-out data. Ratio below 1 beats the baseline.

=== "Synthetic world"

    | horizon | model L1 | no-op L1 | ratio |
    | --- | --- | --- | --- |
    | 1 | 0.1177 | 0.1382 | 0.85 |
    | 2 | 0.1205 | 0.1569 | 0.77 |
    | 3 | 0.1281 | 0.1670 | 0.77 |
    | 4 | 0.1272 | 0.1703 | 0.75 |
    | 5 | 0.1332 | 0.1734 | 0.77 |
    | 6 | 0.1501 | 0.1879 | 0.80 |
    | 7 | 0.1475 | 0.1847 | 0.80 |
    | 8 | 0.1437 | 0.1808 | 0.79 |

=== "Franka arm"

    | horizon | model L1 | no-op L1 | ratio |
    | --- | --- | --- | --- |
    | 1 | 0.0570 | 0.0588 | 0.97 |
    | 2 | 0.0705 | 0.0916 | 0.77 |
    | 3 | 0.0723 | 0.0998 | 0.72 |
    | 4 | 0.0773 | 0.1097 | 0.70 |
    | 5 | 0.0796 | 0.1118 | 0.71 |
    | 6 | 0.0855 | 0.1197 | 0.71 |
    | 7 | 0.0911 | 0.1235 | 0.74 |

**The no-op baseline is the finding.** Absolute rollout error grows with horizon in
both settings, which looks like compounding error and is usually reported as such.
But the *baseline* grows too, and faster. The ratio is flat to slightly improving
out to 8 steps, so what the absolute number mostly measures is how far the system
moved, not how wrong the model was.

The Franka `h=1` ratio of 0.97 is the sharpest case: at one step of a 30 fps arm,
"assume nothing changes" is almost exactly as good as a trained dynamics model,
because almost nothing does change. A raw L1 number at short horizon is therefore
close to uninterpretable on its own, and every horizon table in this library
carries its baseline.

## Planning in the synthetic world

True distance to the goal after 6 steps, 32 episodes, CEM over latent dynamics.
The planner optimises a latent cost and never sees these positions.

| policy | mean | median | worst | % of gap closed |
| --- | --- | --- | --- | --- |
| no-op | 0.4204 | 0.4148 | 0.8240 | 0.0 |
| random | 0.6276 | 0.6551 | 1.1673 | −49.3 |
| planner | 0.1861 | 0.0941 | 1.2058 | **+55.7** |
| oracle | 0.0 | 0.0 | 0.0 | 100.0 |

Note the *worst* column: the planner's worst episode (1.206) is worse than random's
worst (1.167), while its median (0.094) is four times better than anything else on
the page. The distribution is strongly bimodal, in that the planner either solves
the episode almost exactly or fails badly. A mean alone conceals that, and it is
the expected signature of planning through a model that is accurate in most of the
state space and wrong in a corner of it.

## Franka planning

Tool distance to a **goal image** after 5 steps. The planner optimises a latent
cost; it never sees a coordinate.

| policy | mean (m) | median (m) | worst (m) | % of gap closed | mean \|a\| |
| --- | --- | --- | --- | --- | --- |
| no-op | 0.2465 | 0.2308 | 0.4691 | 0.0 | n/a |
| random | 0.3447 | 0.3757 | 0.8154 | −39.8 | n/a |
| planner (`a_pen=0`) | 0.2201 | 0.2003 | 0.5567 | +10.7 | 0.2131 |
| planner (`a_pen=0.05`) | 0.1933 | 0.1829 | 0.3636 | +21.6 | 0.0896 |
| oracle | 0.0 | 0.0 | 0.0 | 100.0 | n/a |

!!! warning "Read the run-to-run spread before quoting the headline"

    This is the third of three runs at the identical preset, which closed +28%,
    +12% and +22% respectively: a mean near 21% and a spread comparable to the
    effect. What replicates is the ordering and the action penalty's sign, not the
    magnitude, so the honest claim is 10–30%. See
    [the variance section](#the-franka-planning-result-varies-by-more-than-its-headline-number)
    for why the no-op baseline is identical to the millimetre in all three.

**The action penalty doubled the result and more than halved the motion.**
`a_pen=0.05` closes +21.6% of the gap against +10.7% unpenalised, at 0.090 mean
action magnitude against 0.213, and it cuts the worst episode from 0.56 m to
0.36 m. That last column is the tell: the penalty helps most where the model is
worst. A planner with nothing to lose will thrash, because large opposing actions
cancel in the true dynamics while the model scores them as progress, so the
penalty buys model-error robustness rather than aesthetics. Of the numbers on this
page it is the one that replicated in every run.

Supporting numbers from the same run: joint-angle probe R² of 0.366 from the frozen
encoder, encoder loss 0.128, teacher-forcing loss 0.0530, rollout loss 0.0558. The
rollout and teacher-forcing losses being this close is the two-term training
objective working: the free rollout has not drifted away from the one-step signal.

## TD-MPC2 on the Franka arm

Two runs at the identical `gpu` preset: 120 iterations, 1160 episodes collected,
23 200 buffer steps, horizon 3. Distances are ground truth; the agent only ever
sees reward.

| policy | run A (m) | run A | run B (m) | run B |
| --- | --- | --- | --- | --- |
| no-op | 0.2455 | 0.0 | 0.2455 | 0.0 |
| random | 0.4958 | −102.0 | 0.4958 | −102.0 |
| policy prior (no planning) | 0.8942 | −264.3 | 0.8846 | −260.3 |
| **TD-MPC2 + MPPI** | **0.2031** | **+17.2** | **0.4173** | **−70.0** |

**The sign of the headline result does not reproduce.** Same preset, same code: the
planner closed 17% of the gap in one run and finished 70% *worse* than doing
nothing in the other. That is not a spread around an effect, it is both signs of
one, so neither number can be quoted as the result. The defensible claim is that
TD-MPC2 at this budget on this task lands somewhere between clearly better and
clearly worse than not moving, and that a single run tells you nothing about which.

The baselines locate the variance. `no-op` and `random` are identical to four
decimals across both runs, because both are seeded and model-independent, so every
bit of the movement is in what was learned rather than in how it was scored.

**What does replicate is the planner-against-prior gap.** Planning beats the policy
prior by a factor of 2.1 in the worse run and 4.4 in the better one, and the prior
alone is worse than random actions in both. So *the prior is a proposal
distribution and not a controller* is a stable finding; *the planner beats doing
nothing* is not. Reading the prior's −260% as a failed agent would still be a
misreading of what it is for.

**The training trace favours the negative reading over an unlucky evaluation.** In
run B the consistency loss rose monotonically, 0.87 at iteration 0 to 1.68 at
iteration 115, while the evaluation distance oscillated between 0.34 m and 1.11 m
with no trend; both runs end with it near 1.7 (1.741 and 1.705). The term whose job
is to make the latent dynamics predictive got worse throughout training, so MPPI
was planning through a model that was degrading under it. With dynamics that poor,
whether search helps or hurts is not pinned down by the configuration, which is
what a sign flip between identical runs looks like. Value loss is flat at 0.350 in
both; reward loss 0.973 and 1.006.

That last point corrects a reading in an earlier version of this page, which took
the consistency term still being the largest as a sign that the dynamics were
"still improving". A large loss under a 20× weight says the term dominates the
gradient, not that it is on its way down. The trace says it was going up.

## MuZero on the Franka arm

!!! warning "A smoke run, not a result"

    Unlike every other section on this page, these numbers are the **laptop
    defaults**: 2 iterations, 8 simulations per move, 13 episodes, a 65-step
    buffer. No preset in `deploy/_shared.py` produced them. They record that the
    pipeline runs end to end, and they are not evidence about MuZero. The GPU run
    that would be is not in yet.

| policy | mean distance (m) | % of gap closed |
| --- | --- | --- |
| no-op | 0.2143 | 0.0 |
| MuZero + MCTS | 0.5051 | **−135.7** |

At this budget the agent ends up more than twice as far from the goal as doing
nothing. Sixty-five buffer steps is not a MuZero budget: the algorithm is the most
sample-hungry of the three families, and it needs the search to be better than its
own policy before its targets mean anything.

**One diagnostic is legible and one is not, and the difference is instructive.**

`mean_search_entropy` is **not** interpretable here. With 8 simulations the root
can visit at most 8 distinct actions, so the entropy is capped at
\(\ln 8 \approx 2.08\), nowhere near the \(\ln 15 \approx 2.71\) of a uniform
policy over all 15 actions. The observed 1.386 is exactly \(\ln 4\), which is
eight visits spread evenly across four actions. That is what the simulation budget
forces, not something the search decided, so this row cannot be read as evidence
of concentration in either direction.

The **policy loss** is interpretable, and it is the one to read. At 2.689 it sits
at 99.3% of \(\ln 15 = 2.708\), the cross-entropy of a head that outputs a uniform
distribution over the 15 actions. So the policy head had learned essentially
nothing after two iterations, which is exactly what two iterations should buy.
Value loss 2.807, reward loss 0.948.

!!! note "The two arms' baselines are not interchangeable"

    The no-op baseline here is 0.214 m, while
    [TD-MPC2's](#td-mpc2-on-the-franka-arm) is 0.245 m. Same task and same goal,
    but different episode lengths, so a longer episode drifts further from the
    start. Percentages of gap closed are comparable within a page section and not
    across them.

It is on this page because a library that only shows its wins is not much use for
deciding what to run, and because a smoke run labelled as one is more useful than
a gap.

## Plot palettes

Sampling the viridis ramp over `VIRIDIS_SPAN = (0.06, 0.94)` and checking
adjacent-pair separation in OKLab:

| slots | CVD separation ΔE | normal-vision ΔE |
| --- | --- | --- |
| 2 | 62.8 | 63.7 |
| 3 | 28.3 | 33.0 |
| 4 | 18.6 | 22.4 |
| 5 | 13.6 | 16.3 |
| 6 | 10.4 | **13.2, too low** |

A normal-vision ΔE below 15 means readers with full colour vision cannot reliably
tell the pair apart, and no secondary encoding fixes that. So
[`viridis_colors`](reference/plots.md#xwm.plots.viridis_colors) **raises** above
`MAX_CATEGORICAL = 5` rather than degrading quietly: the limit is enforced instead
of documented.

Curves use `"blue-orange"` instead: Wong's colourblind-safe family, adjacent-pair
ΔE of 24–36, passing every check in light and dark mode. Line charts carry two to
four series and need maximum separation; a sequential ramp is the wrong tool for
them, and its dark and light extremes have poor contrast against the plot surface
at either end. See [Figures and tables](guides/figures.md).

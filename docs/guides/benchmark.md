# The benchmark protocol

A world model is easy to score badly. Prediction error in latent space says
nothing on its own -- a collapsed encoder has the lowest error of all -- and a
success rate quoted without the budget, the goal distance and the floor it beats
is not a measurement. `xwm.bench` fixes those things, once, so that two models
can be compared.

The question it asks is deliberately narrow:

> Reset the environment to a state some held-out recording actually visited.
> Show the model, **as an observation**, a state that same recording reached
> `goal_offset` steps later. Give its planner `eval_budget` environment steps.
> Did the simulator's own state get within the success threshold?

Everything else follows from that sentence.

## Why the goal comes from a recording

Because it guarantees the task is solvable. A randomly sampled goal may be
unreachable in the budget, or unreachable at all, and a model that fails it has
told you nothing. A goal the demonstrator reached, from a start the
demonstrator was at, in exactly the budget allowed, has a known solution --
which the `replay` baseline executes.

It also removes an argument. Two models evaluated on instances sampled with the
same seed are evaluated on *literally the same instances*, so the difference
between their numbers is a difference between the models.

## The four policies, and which one to read first

| policy | what it does | what it tells you |
| --- | --- | --- |
| `planner` | receding-horizon planning through the model | the number |
| `replay` | the demonstrator's own recorded actions | **the gate** |
| `noop` | nothing | how much of the gap the start already closes |
| `random` | uniform actions | the floor a search must beat |

**Read `replay` first.** It should succeed essentially always, because the goal
is by construction where those actions led. When it does not, the environment
is not being reset to the recorded state faithfully -- an unrecorded velocity, a
contact that has to re-settle, a state vector that does not determine the
simulator -- and no model number from that task means anything until that is
fixed.

```bash
xwm eval runs/my-run --policy replay --policy noop
```

That is one command and it is worth running before any training.

## What the loop does, and does not, do

**It counts raw environment steps.** One planned action covers `frameskip` of
them. The budget, the distance trace and steps-to-success are all in raw steps,
because that is the unit a task's difficulty is stated in.

**It does not stop on success.** Arrival is latched as `solved_at` and the
episode runs to the budget, so `final_distance` stays honest about a planner
that reaches the goal and then drifts off it. Report `success_rate` with
`distance_closed_mean` beside it for exactly this reason.

**It ignores the environment's own `terminated`.** That flag answers a
different question -- Push-T raises it on coverage of a *fixed* target, not on
reaching the recorded goal -- and letting it end an episode would silently
change the metric.

## Actions are grouped, not repeated

At `frameskip = k`, one model action carries all `k` raw actions concatenated:
`blocked_action_dim = action_dim * k`. The alternative, repeating one action `k`
times, throws away four fifths of the recorded control signal at the Push-T
default and leaves the model unable to express what the demonstrator did.

This is the single easiest thing to get wrong, because getting it wrong raises
nothing: the model trains cleanly and plans badly. `frameskip` therefore lives
only on the `TaskSpec`, so training and evaluation cannot disagree about it.

## Preprocessing parity

An environment emits the same field vocabulary a dataset does -- `image`,
`state`, `position` -- so `Task.preprocess_frame` is called with a recorded
frame during training and a live frame during evaluation, with identical
arguments. Parity is structural rather than a convention someone maintains.

The rule that makes it work: **the environment renders at the recording's
resolution**, and both paths resize from there. A natively-rendered 224x224
frame is sharper and differently anti-aliased than a 96x96 recording upsampled
to 224, and an encoder trained on the second and evaluated on the first is being
evaluated out of distribution, with nothing in the output to say so.

## Instances that measure something

An instance whose goal is already within the success threshold of its start is
one `noop` solves. `sample_episodes` refuses to draw them:

```python
xwm.bench.sample_episodes(episodes, n=50, goal_offset=25,
                          distance=task.distance, min_distance=threshold)
```

This matters more than it sounds, and it is not only a synthetic-data problem:
recorded corpora contain idle windows at the start and end of episodes, and in
a contact task any window where the demonstrator was not touching the object
contributes a start and a goal that differ by nothing.

The synthetic Push-T task was built twice for this reason. Under smoothed random
actions -- the obvious way to generate data -- `PushWorld`'s pusher usually
misses the puck, whose median displacement over any window is **0.000**; both
`noop` and `random` scored 100%. The task now ships a scripted pusher
(`push_expert_actions`) that performs the task, which is what a benchmark's
`expert_policy` is for.

## What the numbers mean

Every metric here is defined with its equation in **[Metrics](metrics.md)**:
success rate, distance closed, best-versus-final distance, the prediction
losses, the task criteria and the collapse diagnostics. Two are worth knowing
before reading any table: `distance_closed` goes **negative** when a policy ends
further from the goal than it started, and `best_distance` below `final_distance`
is the signature of a planner that reached the goal and could not hold it.

## Results

One file per evaluation, schema `xwm.bench.goal_reaching/1`, holding everything
needed to re-run it: the task and its full spec, the model and the hash of the
config that built it, the planner settings, and the exact instance list. A
results file that cannot be re-run from its own contents is a screenshot.

```
runs/<name>-<config hash>/
  config.toml       the source, verbatim
  config.json       the resolved tree, which `xwm eval` reads back
  checkpoint/       weights plus the sidecar that rebuilds them
  history.json      training loss
  eval.json         the schema above
  eval_table.tex    the policy comparison, for a paper
```

## Adding a task

1. Write a `TaskSpec` -- dataset, environment, action bounds, resolution,
   frameskip, goal offset, budget, metric.
2. Register an environment builder satisfying `xwm.envs.Env`. The invariant to
   check is `env.reset(state=env.state())` reproducing the state.
3. Register a metric: a distance and a threshold predicate over *simulator*
   state, never over observations or latents.
4. Run the floors. `replay` at ~100% and `noop` at ~0% mean the task measures
   control. Anything else means it does not, yet.

## Comparable numbers, and what they are not

Published goal-reaching success rates at a 50-step budget, for orientation:

| model | Push-T | OGBench Cube |
| --- | --- | --- |
| LeWM | 96 | 74 |
| PLDM | 78 | 65 |
| DINO-WM | 74 | 86 |
| TD-MPC2 | 12 | 4 |

These are **PyTorch implementations, different encoders, different data, and
not xwm results**. They are here as targets, and as a reminder that TD-MPC2's
low numbers are what a reward-driven model scores when it is asked a
goal-reaching question with its reward head unused -- a handicap of the
protocol, stated rather than hidden.

## References

- Zhou et al., *DINO-WM*, 2024. [arXiv:2411.04983](https://arxiv.org/abs/2411.04983)
- Maes et al., *stable-worldmodel*, 2026. [arXiv:2605.21800](https://arxiv.org/abs/2605.21800)
- Chi et al., *Diffusion Policy*, RSS 2023. [arXiv:2303.04137](https://arxiv.org/abs/2303.04137) -- Push-T
- Park et al., *OGBench*, ICLR 2025. [arXiv:2410.20092](https://arxiv.org/abs/2410.20092)

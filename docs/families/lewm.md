# Autoregressive JEPA

**A window of frames, one block-causal pass, and the encoder trained through its
own prediction loss.** Two published models are this class with different second
loss terms.

```python
model = xwm.families.jepa.lewm(action_dim=2, img_size=64, patch_size=8)
model = xwm.families.jepa.delta_jepa(action_dim=2, img_size=64, patch_size=8)
```

[`ActionWorldModel`](jepa.md) — the V-JEPA 2-AC recipe — is Markov, and inherits
a stage boundary: pretrain a representation, freeze it, then learn dynamics in
its latent space. Freezing is what makes it safe, because a fixed encoder cannot
degrade its own targets.
[`ARWorldModel`](../reference/families.md#xwm.families.jepa.ARWorldModel)
removes both restrictions at once, and each removal costs something that has to
be paid for.

## Components

| component | signature | class |
| --- | --- | --- |
| encoder | `(C, H, W) → (N, D)` | [`ImageEncoder`](../reference/encoders.md#xwm.encoders.ImageEncoder) |
| projector | `(D,) → (D',)` | `equinox.nn.Linear` |
| dynamics | `(H, N, D) × (H, A) → (H, N, D)` | [`ARPredictor`](../reference/dynamics.md#xwm.dynamics.ARPredictor) |
| action decoder | `(N, D) → (A,)` | [`LatentDifferenceActionDecoder`](../reference/heads.md#xwm.heads.LatentDifferenceActionDecoder) |

## Why a window

A single frame does not determine the state. The puck's position is in the
image; its velocity is not. A Markov model has to infer contact from position
alone, and there are configurations where that is impossible in principle rather
than merely hard.

`ARPredictor` conditions on `history` frames and predicts **every** next frame in
one pass, using a *block* causal mask: frame `t` attends to frames `0..t` in
full, and the tokens within one frame attend to each other in both directions,
because imposing an order on the patches of a single image would be arbitrary.

```python
predicted = model.dynamics(z, actions)   # (H, N, D) -> (H, N, D)
predicted[t]                             # the model's guess at frame t + 1
```

One consequence is worth stating, because it is free and it matters at
evaluation time. Position `0` of the window predicts from a single frame,
position `1` from two, and so on. The model is therefore trained at **every**
context length up to `history`, which is exactly the regime the planner meets in
the first few steps of an episode, before the window has filled.

## The loss

| term | weight | what it does |
| --- | --- | --- |
| `loss_prediction` | 1 | L2 between predicted and encoded next-frame latents |
| `loss_reg` | `reg_weight` (0.1) | SIGReg: embeddings should look isotropic Gaussian |
| `loss_action` | `action_weight` (10 in Delta-JEPA) | the action, decoded from the latent displacement |

```python
loss, metrics = model.loss(batch, key=key)
# loss_prediction, loss_reg | loss_action, latent_std, loss
```

Watch **`latent_std`**, not `loss`. A collapsing encoder drives its prediction
loss *down*, because it is predicting its own constant output.

## What stops it collapsing

This is the entire design space, and the choice is the model.

| recipe | countermeasure | mechanism |
| --- | --- | --- |
| `lewm` | SIGReg | the embedding *distribution* must be isotropic Gaussian, and a point mass is not |
| `delta_jepa` | action decoding | `a_t` must be recoverable from `z_{t+1} − z_t`, and a constant has no displacement |
| `dinowm` | a frozen encoder | it cannot move, so it cannot degrade its targets |

The constructor refuses the fourth combination — a live encoder, movable
targets, and nothing forbidding the trivial solution — rather than letting it be
discovered after an overnight run whose loss went to zero.

```python
xwm.families.jepa.ARWorldModel(encoder, dynamics, collapse="none")
# ValueError: an unfrozen encoder trained through its own targets will collapse: ...
```

!!! note "The two are not interchangeable"

    SIGReg constrains where embeddings *are*; the action decoder constrains how
    they *move*. Isotropy does not by itself guarantee that two different
    actions produce distinguishable latent displacements, which is the property
    a planner searching over action sequences actually depends on. Delta-JEPA's
    ablations report the difference; `docs/findings.md` records what it is on
    this library's own task.

## Planning

The planning state is a `Window`, not a latent — the first multi-frame state in
the library.

```python
window = model.initial_state(context.frames(), context.actions())
plan = planner.plan(key, model.dynamics_fn(), window, cost)
```

Each step of `dynamics_fn` writes the proposed action into the empty slot,
predicts, and slides: the oldest frame and its action fall off the back, the
prediction joins the front. `readout` selects the newest frame, and the goal cost
**must** be given it:

```python
cost = xwm.planning.goal_cost(goal, readout=model.readout)
```

Without `readout` the cost also charges the window's stale context for failing
to move, which no action can fix.

## Training

Clips are exactly `history + 1` frames — one window — and the model says so by
name if they are not:

```toml
[task.overrides]
history = 3

[train]
clip_length = 4
```

End-to-end training is not free, and the cost is in the encoder rather than the
predictor. `jepa/action` freezes its encoder, so a step is one forward pass
through it and a backward pass through the dynamics alone; `jepa/lewm` runs the
encoder over every frame of the window *and* backpropagates through all of them.
On the synthetic pusher at laptop settings that is roughly an order of magnitude
more time per step. It buys a representation shaped by the prediction task,
which is the whole claim -- but budget for it, and prefer `dinowm` when you want
the dynamics measured against a representation you did not have to fit.

There is deliberately no rollout term. With one window per clip there is nothing
to roll through; compounding error is attacked by supervising the short prefixes
instead. Measure it with `model.imagine(window, actions)`.

## References

| model | paper |
| --- | --- |
| LeWorldModel | Maes et al., 2026 · [arXiv:2603.19312](https://arxiv.org/abs/2603.19312) |
| Delta-JEPA | 2026 · [arXiv:2606.31232](https://arxiv.org/abs/2606.31232) |
| SIGReg / LeJEPA | Balestriero & LeCun, 2025 |

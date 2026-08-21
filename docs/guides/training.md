# Training

[`Trainer`](../reference/training.md#xwm.training.Trainer) is **family-agnostic**:
it needs only `loss`, `prepare_batch` and `trainable`. It owns the `jit` boundary,
the EMA teacher and the parameter filter, so frozen submodules never reach the
optimizer.

```python
trainer = xwm.training.Trainer(
    model,
    xwm.training.adamw(xwm.training.cosine_warmup(1e-3, 1000, warmup_steps=100)),
)
state, history = trainer.fit(batches, steps=1000)
```

`state` is a [`TrainState`](../reference/training.md#xwm.training.TrainState):
`state.model`, `state.target`, `state.opt_state`, `state.step`. `history` is a list
of dicts, one per logged step, ready for
[`plot_history`](../reference/plots.md#xwm.plots.plot_history).

## What the Trainer does for you

| concern | how |
| --- | --- |
| compilation | one `eqx.filter_jit`ed step, built once at construction |
| mask sampling | calls `model.prepare_batch(batch, key)` on the host each step |
| EMA teacher | maintained iff the model sets `uses_target` |
| frozen parameters | the optimizer sees `model.trainable` and nothing else |
| per-step keys | `jr.fold_in(key, step)`, so one seed reproduces the run |
| metrics | kept on device between logged steps, so the loop does not block |

That last one matters more than it sounds: `log_every=10` means metrics are pulled
back to the host every tenth step, and in between the Python loop dispatches
without waiting on the step it just queued.

## Optimisers and schedules

```python
opt = xwm.training.adamw(1e-4)                            # constant
opt = xwm.training.adamw(1e-4, weight_decay=0.0)          # for the RL families
opt = xwm.training.adamw(xwm.training.cosine_warmup(1e-3, 4000, warmup_steps=400))
```

[`adamw`](../reference/training.md#xwm.training.adamw) defaults to
`weight_decay=0.05`, `b2=0.95` and gradient clipping at global norm 1.0, which is
the ViT recipe rather than optax's defaults.
[`weight_decay_schedule`](../reference/training.md#xwm.training.weight_decay_schedule)
and [`ema_momentum`](../reference/training.md#xwm.training.ema_momentum) cover the
two schedules the JEPA papers ramp:

```python
trainer = xwm.training.Trainer(
    model, opt, ema_momentum=xwm.training.ema_momentum(steps, base=0.996),
)
```

## Where batches come from

Two sources, for two different needs:

=== "A fixed dataset"

    ```python
    batches = xwm.data.iter_batches(data, batch_size=64, key=key, epochs=None)
    ```

    `epochs=None` gives an endless shuffled stream; `epochs=1` gives one pass.
    Any dict of arrays with a common leading axis works.

=== "A replay buffer"

    ```python
    buffer = xwm.training.ReplayBuffer(
        100_000, observation_shape=(20,), action_shape=(7,),
    )
    buffer.add_episode(observations, actions, rewards)

    batch = buffer.sample(batch_size=64, horizon=3)
    ```

    The reward-driven families need **contiguous slices of a single episode**,
    which a shuffled dataset cannot provide.

!!! note "The buffer refuses to cross an episode boundary"

    A slice that straddles a reset would ask the dynamics model to predict
    through it, the one transition it can never get right. `sample` only draws
    starts with `horizon` steps of the same episode ahead of them.

## Callbacks

```python
state, history = trainer.fit(
    batches, steps=1000, log_every=10,
    callbacks=[xwm.training.print_metrics(every=50, keys=["loss", "embed_std"])],
)
```

A callback is `cb(state, metrics)`, called on logged steps. Anything you want fits:
checkpointing, an early-stopping check, a custom logger.

```python
def checkpoint_every(n, path):
    def cb(state, metrics):
        if int(state.step) % n == 0:
            xwm.tools.save(f"{path}/step_{int(state.step)}", state.model)
    return cb
```

## Freezing

Freezing is expressed by the model, through `trainable`, and the `Trainer` respects
it without being told:

```python
model = xwm.families.jepa.action_world_model(
    action_dim=7, encoder=pretrained, freeze_encoder=True,
)

trainer = xwm.training.Trainer(model, xwm.training.adamw(1e-4))
trainer.n_trainable == model.dynamics.n_params    # the encoder has no optimizer state
```

`n_trainable` is worth asserting on in your own scripts. A frozen module that
quietly receives gradients is a silent bug. The tests assert frozen parameters do
not move, and so should you.

## A manual loop

`fit` is a convenience. When the loop has to interleave with data collection, as
the reward-driven families do, drive `step` yourself:

```python
state = trainer.init()

for iteration in range(iterations):
    for episode in collect(env, agent, n=8):
        buffer.add_episode(*episode)

    for i in range(steps_per_iter):
        batch = buffer.sample(64, horizon=agent.horizon)
        state, metrics = trainer.step(state, batch, jr.fold_in(key, iteration * 1000 + i))
```

`trainer.step` is the jitted step; `trainer.init()` builds the initial state
(including the EMA copy, if the model needs one).

## Evaluating

```python
scores = trainer.evaluate(state, held_out_batches)
```

`evaluate` puts the model in eval mode with dropout off, averages the loss metrics
over the batches, and calls `prepare_batch` per batch so masked families get their
masks. It reports the *loss*, so read
[Diagnostics](../concepts/diagnostics.md) before concluding anything from it.

## Checkpoints

```python
xwm.tools.save("runs/jepa", state.model, config={"img_size": 64, "patch_size": 8})

model = xwm.families.jepa.ijepa(img_size=64, patch_size=8)   # same architecture
model = xwm.tools.load("runs/jepa", like=model)
```

`load` needs a `like`, an instance of the right architecture to pour the leaves
into. That is why `save` takes a `config`: store what you need to rebuild the
skeleton, and [`load_config`](../reference/tools.md#xwm.tools.load_config) reads it
back.

## Model size

```python
print(xwm.tools.summary(model, max_depth=2))
model.n_params
xwm.tools.param_bytes(model)
```

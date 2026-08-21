# Randomness

`key=` is **optional wherever a model is built** and **required wherever a key is
consumed inside `jit`**. That line is not a convenience choice; it is the `jit`
boundary.

## Construction

```python
xwm.set_seed(0)
model = xwm.families.jepa.ijepa(img_size=64)                     # ambient
other = xwm.families.jepa.ijepa(img_size=64, key=jr.PRNGKey(7))   # explicit
```

Pass a key and nothing ambient is touched. Omit it and the key comes from an
ambient [`KeySource`](../reference/core.md#xwm.core.KeySource).

For a scoped seed:

```python
with xwm.seed(123):
    model = xwm.families.jepa.ijepa(img_size=64)
```

## The source advances on every draw

It has to, or every transformer block would be initialised identically.
The consequence is worth stating plainly:

!!! warning "A fixed sequence of calls under a fixed seed is reproducible; inserting a construction is not"

    ```python
    xwm.set_seed(0)
    a = xwm.families.jepa.ijepa(img_size=64)     # draw 1
    b = xwm.families.jepa.ijepa(img_size=64)     # draw 2 -- different weights
    ```

    Add a third model between them and everything built after it shifts. Pass
    explicit keys for anything that must survive a refactor.

That is the rule of thumb: **ambient keys for scripts, explicit keys for
libraries, tests and checkpoints**.

## Only construction defaults

`loss`, [`sigreg`](../reference/objectives.md#xwm.objectives.sigreg.sigreg) and every
planner still require a key:

```python
loss, metrics = model.loss(batch, key)          # required
plan = planner.plan(key, step, z0, cost)        # required
```

The reason is `jit`. These are consumed *inside* a traced function, where a key
drawn at trace time would be captured as a **constant** and reused for every
step: the same dropout mask, the same SIGReg projections, the same CEM samples,
forever. It would not error; it would silently stop being random. Making the key
an argument makes that impossible.

## Splitting

```python
key = jr.PRNGKey(0)
k_data, k_model, k_train = jr.split(key, 3)
```

The examples all follow this shape: split once at the top, then
[`jr.fold_in`](https://docs.jax.dev/en/latest/_autosummary/jax.random.fold_in.html)
for per-item derivations (`jr.fold_in(k_data, epoch)`), so adding a consumer never
disturbs the others. [`xwm.core.split`](../reference/core.md#xwm.core.split) is the
same operation with the library's optional-key semantics.

## Inside the training loop

The [`Trainer`](../reference/training.md#xwm.training.Trainer) folds the step
number into the key it was given, so every step gets a fresh key and the whole run
is a pure function of one seed:

```python
state, history = trainer.fit(batches, steps=1000, key=jr.PRNGKey(0))
```

Host-side randomness, meaning mask sampling, is separate: it happens in
`model.prepare_batch(batch, key)`, outside `jit`, because rejection-sampling
non-overlapping blocks is combinatorial work that does not belong on an
accelerator.

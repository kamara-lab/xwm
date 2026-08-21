# Conventions

Five rules cover almost everything. They are worth reading once, because they
explain most of the API's shape.

## Modules are unbatched

Every module is written for a **single sample** and `vmap`ed by the caller,
following the Equinox idiom. Batch-level entry points are the methods named `loss`.

```python
z = model.encode(observation)                      # one observation
z = jax.vmap(model.encode)(observations)           # a batch
z = xwm.core.batched_apply(jax.jit(jax.vmap(model.embed)), images, batch_size=64)
```

`batched_apply` exists for the case where the batch does not fit in memory: it
chunks, applies, and concatenates.

## Shapes

| thing | shape |
| --- | --- |
| image | `(C, H, W)` |
| clip | `(T, C, H, W)` |
| token sequence | `(N, D)` |
| flat latent | `(D,)` |
| action | `(A,)` |
| action sequence | `(T, A)` |
| mask | `int32` index array |

Masks are **index arrays**, not booleans, which is what makes a gather cheap and
a shape static. [`xwm.masking.boolean_mask`](../reference/masking.md) converts
when you need the other form.

## Immutability

Modules are PyTrees and nothing mutates in place. Anything that looks like a
setter *returns a copy*:

```python
eval_model = model.eval_mode()      # a dropout-free copy; `model` is unchanged
```

## Keys are optional at construction, required inside `jit`

```python
xwm.set_seed(0)
model = xwm.families.jepa.ijepa(img_size=64)                     # ambient key
other = xwm.families.jepa.ijepa(img_size=64, key=jr.PRNGKey(7))   # explicit
```

`loss`, `sigreg` and the planners still require a key, because those are consumed
inside `jit`, where a key drawn at trace time would be baked in as a constant and
reused for every step. See [Randomness](../concepts/randomness.md).

## The dynamics interface

Every family exposes its dynamics as a plain closure, and every planner consumes
nothing else:

```python
step = model.dynamics_fn()      # (z, a) -> z'
```

That single signature is why one set of planners serves all three families, and
why swapping family changes how a dynamics model is *trained*, never how it is
*used*.

## Naming

| name | means |
| --- | --- |
| `encode` | observation → latent, the model's own preferred latent |
| `embed` | observation → representation, for probing and diagnostics |
| `predict` | latent → latent, or latent → head outputs |
| `imagine` | latent + action sequence → latent trajectory |
| `loss` | the batch-level training objective, `loss(batch, *, key, target=None)` |
| `prepare_batch` | host-side work that must happen outside `jit` (mask sampling) |
| `*_fn` | returns a closure, for handing to a planner or a rollout |

## References live beside the code

Every module carries a `References` block in its docstring naming the paper the
code follows:

```python
help(xwm.families.tdmpc2.model)
```

Component-level citations live in the docstrings of the modules that implement
them, and are reproduced in the [API reference](../reference/index.md): SimNorm,
two-hot categorical scalars, REDQ, SAC, MPPI, PUCT, Epps–Pulley, RankMe,
ViT/ViViT, MAE, RoPE, LayerScale, Mish.

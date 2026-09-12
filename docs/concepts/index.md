# Concepts

A world model answers *"what happens if I do this?"*. An **action-conditioned**
one answers it in a latent space of its own choosing, which is what makes it
usable for robotics: you can search over imagined action sequences without
rendering a single pixel.

## Three pieces and no decoder

<figure class="xwm-diagram" markdown="1">
  ![Observation passes through the encoder and dynamics to a planner, which chooses an action](../assets/world-model-light.svg#only-light){ width="1040" } ![Observation passes through the encoder and dynamics to a planner, which chooses an action](../assets/world-model-dark.svg#only-dark){ width="1040" }
  <figcaption>
    The shared encoder and dynamics support planning; optional heads estimate rewards and values.
  </figcaption>
</figure>

| piece | signature | lives in |
| --- | --- | --- |
| encoder | \(h: O \rightarrow Z\) | [`xwm.encoders`](../reference/encoders.md) |
| dynamics | \(d: Z \times A \rightarrow Z\) | [`xwm.dynamics`](../reference/dynamics.md) |
| heads | \(Z \rightarrow \mathbb{R}\), \(Z \rightarrow \Delta(A)\) | [`xwm.heads`](../reference/heads.md) |

There is no decoder \(Z \rightarrow O\) and no pixel-reconstruction loss anywhere in
the library. That is a design commitment, not an omission: reconstruction spends
capacity on the parts of an observation that are unpredictable *and* irrelevant
(the exact texture of the floor, the grain of a shadow), and a planner never needs
them.

## Why the dynamics model is the centre

Every family consumes a \((z, a) \rightarrow z\) model, and every planner consumes
nothing else:

```python
step = model.dynamics_fn()     # (z, a) -> z'
plan = planner.plan(key, step, model.encode(observation), cost)
```

Changing family changes how that closure is *trained*. It never changes how it is
*used*, which is why one implementation of CEM, MPPI, gradient planning and MCTS
serves the whole library.

## What differs between families

| | trains the latent space with | needs reward? | acts by |
| --- | --- | --- | --- |
| [JEPA](../families/jepa.md) | its own future embeddings | no | CEM / MPPI over a latent cost |
| [TD-MPC2](../families/tdmpc2.md) | reward + TD value | yes | MPPI over reward + terminal value |
| [MuZero](../families/muzero.md) | search-improved targets | yes | MCTS |

## Read in this order

<div class="grid cards" markdown>

-   __1. [Masking](masking.md)__

    ---

    What a JEPA is asked to predict, and why the mask has to be hard enough to
    stop it interpolating from neighbours.

-   __2. [Collapse](collapse.md)__

    ---

    Predicting a representation from a representation has a trivial solution.
    Four ways to forbid it, one of which needs no teacher.

-   __3. [Planning](planning.md)__

    ---

    How a plain closure plus a latent cost becomes a single device call.

-   __4. [Diagnostics](diagnostics.md)__

    ---

    The loss is not the metric. What to measure instead, and what each measure
    misses.

-   __5. [Randomness](randomness.md)__

    ---

    Where `key=` is optional, where it is mandatory, and why the difference is
    exactly the `jit` boundary.

</div>

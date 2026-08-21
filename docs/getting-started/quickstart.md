# Quickstart

Twenty lines, end to end: build a model, train it, then plan with it. Everything
here runs on CPU against the synthetic world in [`xwm.data`](../reference/data.md),
so there is no dataset to download.

## 1. A model

```python
import xwm

xwm.set_seed(0)

model = xwm.families.jepa.lejepa(size="tiny", img_size=64, patch_size=8)
```

`lejepa` is a *recipe*: it assembles an encoder, a predictor and a mask sampler
into a [`JEPA`](../families/jepa.md), and chooses SIGReg as the anti-collapse
term. `size=` selects a ViT preset (`tiny`, `small`, `base`, `large`).

## 2. Data

```python
import jax.random as jr

world = xwm.data.SpriteWorld(64, n_distractors=2)
data = xwm.data.sprite_images(jr.PRNGKey(0), 2048, world=world)
batches = xwm.data.iter_batches(data, batch_size=64, epochs=None)
```

`data` is a dict: `{"image": (n, 3, 64, 64), "position": (n, 2)}`. The position
is there for a probe, never for training.

`epochs=None` gives an endless stream, which is what `Trainer.fit(steps=...)`
wants.

## 3. Training

```python
trainer = xwm.training.Trainer(model, xwm.training.adamw(1e-3))
state, history = trainer.fit(batches, steps=1000, callbacks=[xwm.training.print_metrics(100)])
```

The `Trainer` owns the `jit` boundary, the EMA teacher and the parameter filter.
It also calls `model.prepare_batch()` for you, which is where the host-side mask
sampling happens.

## 4. Check the representation

The loss is not the metric. A collapsing encoder drives its prediction loss
*down*, because it is predicting its own degenerate output. Ask the
representation instead:

```python
import jax

embed = jax.jit(jax.vmap(state.model.embed))
z = xwm.core.batched_apply(embed, data["image"][:512])

xwm.metrics.collapse_report(z)
# {'rankme': ..., 'rank_ratio': ..., 'feature_std': ..., 'mean_cosine': ...}
```

`feature_std → 0` and `mean_cosine → 1` both mean collapse. See
[Diagnostics](../concepts/diagnostics.md).

## 5. Planning

A JEPA on its own has no actions. Freeze its encoder and learn action-conditioned
dynamics in its latent space. That is the V-JEPA 2-AC recipe, and it is one call:

```python
world = xwm.families.jepa.action_world_model(
    action_dim=2, encoder=state.model.encoder, freeze_encoder=True,
    img_size=64, patch_size=8,
)
```

Then plan. The planner never sees a pixel or a coordinate; it optimises a cost
defined purely in \(Z\):

```python
planner = xwm.planning.CEM(horizon=8, action_dim=2, n_samples=512, n_elites=64)
cost = xwm.planning.goal_cost(world.encode(goal_image), kind="l2")

plan = planner.plan(jr.PRNGKey(1), world.dynamics_fn(), world.encode(observation), cost)
plan.actions  # (horizon, action_dim)
```

## Reward-driven instead

If you have a reward, the value-based families skip the two-stage dance: one
model learns representation, dynamics, reward and value together:

=== "TD-MPC2 (continuous)"

    ```python
    agent = xwm.families.tdmpc2.tdmpc2(action_dim=7, observation="state", state_dim=20)
    mppi = xwm.families.tdmpc2.planner(agent, n_samples=512)

    action = agent.act(observation)          # policy prior, no planning
    ```

=== "MuZero (discrete)"

    ```python
    agent = xwm.families.muzero.muzero(n_actions=15, observation="state", state_dim=20)
    search = xwm.planning.MCTS(n_actions=15, n_simulations=50)
    ```

Both need trajectories rather than shuffled samples, because their losses take
contiguous slices of a single episode:

```python
buffer = xwm.training.ReplayBuffer(100_000, observation_shape=(20,), action_shape=(7,))
buffer.add_episode(observations, actions, extra={"reward": rewards})

batch = buffer.sample(batch_size=64, horizon=3)
```

## Next steps

<div class="grid cards" markdown>

-   __Understand the pieces__

    ---

    What a JEPA predicts, why it does not collapse, and how a planner reaches the
    dynamics.

    [:octicons-arrow-right-24: Concepts](../concepts/index.md)

-   __Put it on a robot__

    ---

    A Franka FR3 in Newton, pixels or proprioception, with a dense reach reward.

    [:octicons-arrow-right-24: Robotics](../guides/robotics.md)

-   __Read the runnable versions__

    ---

    Eight examples, from I-JEPA pretraining to MuZero on the arm.

    [:octicons-arrow-right-24: Examples](../guides/examples.md)

-   __Check the API__

    ---

    Every public symbol, with the paper it follows.

    [:octicons-arrow-right-24: Reference](../reference/index.md)

</div>

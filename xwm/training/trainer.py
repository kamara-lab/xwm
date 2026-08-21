"""The training loop.

One class, one contract: the model exposes :meth:`~xwm.core.WorldModel.loss`,
:meth:`~xwm.core.WorldModel.prepare_batch` and
:meth:`~xwm.core.WorldModel.trainable`, and the trainer knows nothing else about
it. The same loop trains I-JEPA, V-JEPA, LeJEPA and an action-conditioned model.

The trainer owns three things the model should not:

* **the jit boundary** -- mask sampling is combinatorial host-side work, so it
  runs in ``prepare_batch`` *before* the compiled step, which is then traced
  exactly once because every mask shape is static.
* **the EMA teacher** -- maintained here, outside the optimizer, and only for
  models that ask for one.
* **the parameter filter** -- frozen submodules never reach the optimizer, so no
  state is allocated for them.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax

from ..core.ema import ema_init, ema_update
from ..core.module import WorldModel
from ..core.random import resolve_key
from ..core.types import Batch, Metrics, PRNGKey
from .state import TrainState

Callback = Callable[[TrainState, Metrics], None]


def global_norm(tree: Any) -> jnp.ndarray:
    leaves = jax.tree_util.tree_leaves(eqx.filter(tree, eqx.is_inexact_array))
    if not leaves:
        return jnp.zeros(())
    return jnp.sqrt(sum(jnp.sum(jnp.square(x)) for x in leaves))


class Trainer:
    """Trains any :class:`~xwm.core.WorldModel`.

    Args:
        model: the model to train.
        optimizer: an optax transformation (see
            :func:`xwm.training.adamw` for sensible JEPA defaults).
        ema_momentum: teacher momentum. A float is constant; a
            :class:`optax.Schedule` is evaluated per step (see
            :func:`xwm.training.ema_momentum`). Ignored unless the model sets
            ``uses_target``.

    Example:
        >>> trainer = Trainer(model, adamw(cosine_warmup(1e-3, 1000)))
        >>> state, history = trainer.fit(batches, steps=1000, key=key)
    """

    def __init__(
        self,
        model: WorldModel,
        optimizer: optax.GradientTransformation,
        *,
        ema_momentum: float | optax.Schedule = 0.996,
    ):
        self.model = model
        self.optimizer = optimizer
        self.trainable_spec = model.trainable()
        self.ema_momentum = ema_momentum
        self._step_fn = self._compile()

    # -- setup ---------------------------------------------------------------
    def init(self, model: WorldModel | None = None) -> TrainState:
        """Fresh training state, with the teacher and optimizer state allocated."""
        model = self.model if model is None else model
        params, _ = eqx.partition(model, self.trainable_spec)
        return TrainState(
            model=model,
            target=ema_init(model) if model.uses_target else None,
            opt_state=self.optimizer.init(params),
            step=0,
        )

    @property
    def n_trainable(self) -> int:
        params, _ = eqx.partition(self.model, self.trainable_spec)
        leaves = jax.tree_util.tree_leaves(eqx.filter(params, eqx.is_inexact_array))
        return sum(int(x.size) for x in leaves)

    def momentum_at(self, step: int) -> float:
        m = self.ema_momentum
        return float(m(step)) if callable(m) else float(m)

    # -- one step ------------------------------------------------------------
    def _compile(self):
        optimizer, spec = self.optimizer, self.trainable_spec

        def objective(params, static, target, batch, key):
            model = eqx.combine(params, static)
            return model.loss(batch, key=key, target=target)

        def step(state: TrainState, batch: Batch, key: PRNGKey, momentum):
            params, static = eqx.partition(state.model, spec)
            (loss, metrics), grads = jax.value_and_grad(objective, has_aux=True)(
                params, static, state.target, batch, key
            )
            updates, opt_state = optimizer.update(grads, state.opt_state, params)
            model = eqx.combine(eqx.apply_updates(params, updates), static)
            target = None if state.target is None else ema_update(state.target, model, momentum)
            metrics = {**metrics, "grad_norm": global_norm(grads)}
            return (
                TrainState(model=model, target=target, opt_state=opt_state, step=state.step + 1),
                metrics,
            )

        return eqx.filter_jit(step)

    def step(self, state: TrainState, batch: Batch, key: PRNGKey) -> tuple[TrainState, Metrics]:
        """Prepare the batch, take one optimizer step, update the teacher."""
        k_prep, k_loss = jr.split(key)
        batch = state.model.prepare_batch(batch, k_prep)
        momentum = jnp.asarray(self.momentum_at(int(state.step)), jnp.float32)
        return self._step_fn(state, batch, k_loss, momentum)

    # -- loops ---------------------------------------------------------------
    def fit(
        self,
        batches: Iterable[Batch],
        *,
        key: PRNGKey | None = None,
        steps: int | None = None,
        state: TrainState | None = None,
        log_every: int = 10,
        callbacks: Sequence[Callback] = (),
    ) -> tuple[TrainState, list[dict[str, float]]]:
        """Train over an iterable of batches.

        Args:
            batches: any iterable of batch dicts (see :mod:`xwm.data`).
            steps: stop after this many steps; ``None`` exhausts the iterable.
            state: resume from this state instead of a fresh one.
            log_every: how often to pull metrics back to the host. Metrics stay
                on device otherwise, so the loop does not block on the step it
                just dispatched.
            callbacks: called as ``cb(state, metrics)`` on logged steps.

        Returns:
            ``(final_state, history)``.
        """
        # Safe to default: this runs once per run, outside jit. The per-step
        # keys below are derived from it, so the whole run stays reproducible.
        key = resolve_key(key)
        state = self.init() if state is None else state
        history: list[dict[str, float]] = []
        for i, batch in enumerate(batches):
            if steps is not None and i >= steps:
                break
            state, metrics = self.step(state, batch, jr.fold_in(key, i))
            if log_every and (i % log_every == 0 or (steps is not None and i == steps - 1)):
                row = {"step": int(state.step), **{k: float(v) for k, v in metrics.items()}}
                history.append(row)
                for cb in callbacks:
                    cb(state, metrics)
        return state, history

    def evaluate(
        self,
        state: TrainState,
        batches: Iterable[Batch],
        *,
        key: PRNGKey | None = None,
    ) -> dict[str, float]:
        """Average the loss over ``batches`` with the model in eval mode."""
        key = resolve_key(key)
        model = state.model.eval_mode()
        target = None if state.target is None else state.target.eval_mode()

        @eqx.filter_jit
        def one(batch, key):
            return model.loss(batch, key=key, target=target)[1]

        totals: dict[str, float] = {}
        count = 0
        for i, batch in enumerate(batches):
            batch = model.prepare_batch(batch, jr.fold_in(key, i))
            metrics = one(batch, jr.fold_in(key, i + 1_000_000))
            for k, v in metrics.items():
                totals[k] = totals.get(k, 0.0) + float(v)
            count += 1
        return {k: v / max(count, 1) for k, v in totals.items()}


def print_metrics(every: int = 1, keys: Sequence[str] | None = None) -> Callback:
    """A callback that prints selected metrics."""
    seen = {"n": 0}

    def cb(state: TrainState, metrics: Metrics) -> None:
        seen["n"] += 1
        if (seen["n"] - 1) % every:
            return
        items = metrics if keys is None else {k: metrics[k] for k in keys if k in metrics}
        body = "  ".join(f"{k}={float(v):.4f}" for k, v in items.items())
        print(f"step {int(state.step):>6}  {body}")

    return cb

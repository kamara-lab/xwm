"""Base classes shared by every xwm model."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from .types import Array, Batch, Metrics, PRNGKey, PyTree


class Module(eqx.Module):
    """An :class:`equinox.Module` with a couple of conveniences.

    Everything in xwm subclasses this, so :attr:`n_params` and
    :meth:`eval_mode` are available on individual layers and on whole world
    models alike.

    Note that ``inference`` is deliberately *not* the name of the method here:
    Equinox treats an ``inference`` attribute as the flag it toggles, so a
    method by that name would shadow it and break
    :func:`equinox.nn.inference_mode`.
    """

    @property
    def n_params(self) -> int:
        """Number of trainable (inexact-array) scalars in this subtree."""
        leaves = jax.tree_util.tree_leaves(eqx.filter(self, eqx.is_inexact_array))
        return sum(int(x.size) for x in leaves)

    def eval_mode(self):
        """A copy with dropout and drop-path disabled.

        Modules are immutable, so this *returns* the eval-mode model rather
        than mutating in place: ``model = model.eval_mode()``.
        """
        return eqx.nn.inference_mode(self, value=True)

    def train_mode(self):
        """A copy with dropout and drop-path re-enabled."""
        return eqx.nn.inference_mode(self, value=False)


class WorldModel(Module):
    """A trainable world model.

    Subclasses implement :meth:`loss`, the single contract the trainer relies
    on. Models whose objective needs a slowly-moving *teacher* (I-JEPA, V-JEPA,
    BYOL-style asymmetry) set :attr:`uses_target` to ``True``; the trainer then
    maintains an EMA copy of the model and passes it in as ``target``. Models
    that do not need one (LeJEPA, VICReg) leave it ``False`` and receive
    ``None``.
    """

    uses_target: bool = eqx.field(static=True, default=False)
    #: Frames of context the latent state carries. 1 for a Markov model.
    history: int = eqx.field(static=True, default=1)

    def loss(
        self,
        batch: Batch,
        *,
        key: PRNGKey,
        target: WorldModel | None = None,
    ) -> tuple[Array, Metrics]:
        """Scalar loss for a *batched* input, plus scalars to log.

        Args:
            batch: modality-specific dict (see :mod:`xwm.data`).
            key: RNG for masking, dropout and any stochastic objective.
            target: EMA copy of ``self`` when :attr:`uses_target`, else ``None``.
        """
        raise NotImplementedError

    def trainable(self) -> PyTree:
        """Boolean tree marking which leaves the optimizer may update.

        Defaults to every inexact array. Override to freeze part of the model --
        an action-conditioned stage trained on top of a fixed video encoder is
        the canonical case, and freezing it here means the optimizer never
        allocates state for those parameters at all.
        """
        return jax.tree_util.tree_map(eqx.is_inexact_array, self)

    def prepare_batch(self, batch: Batch, key: PRNGKey) -> Batch:
        """Host-side hook, run before the jitted step.

        Where mask sampling and any other combinatorial, non-jittable batch
        preparation belongs. It may read the model's *static* configuration but
        must not depend on its parameters -- it runs outside ``jit`` and outside
        the gradient.
        """
        return batch

    # -- planning contract (xwm.core.types.Plannable) -------------------------
    # Defaults for the Markov case. A model whose latent state is a window of
    # several frames overrides all three together; see
    # :class:`xwm.families.jepa.ARWorldModel`.
    def initial_state(
        self, frames: Array, actions: Array | None = None, *, key: PRNGKey | None = None
    ) -> Array:
        """Latent state from ``(H, ...)`` observations: by default, encode the newest.

        ``actions`` are the ``(H - 1, A)`` actions between those frames; the
        default ignores them because a single frame has no history to condition
        on.
        """
        del actions
        return self._encode_for_planning(frames[-1], key=key)

    def readout(self, z: Array) -> Array:
        """The part of ``z`` a goal is compared to. Identity for a Markov model."""
        return z

    def goal_embedding(self, frame: Array, *, key: PRNGKey | None = None) -> Array:
        """Encode a goal observation to what :meth:`readout` should reach."""
        return self._encode_for_planning(frame, key=key)

    def _encode_for_planning(self, frame: Array, *, key: PRNGKey | None) -> Array:
        encode = getattr(self, "encode", None)
        if encode is None:
            raise NotImplementedError(
                f"{type(self).__name__} has no encode(); implement initial_state and "
                "goal_embedding to make it plannable"
            )
        return encode(frame, key=key)


def batched_apply(fn, x, *, batch_size: int = 64) -> PyTree:
    """Apply ``fn`` over the leading axis of ``x`` in chunks, then concatenate.

    ``fn`` maps a batch to a batch (already vmapped or jitted). Use this instead
    of one enormous call whenever the leading axis is a dataset rather than a
    minibatch: encoding 16k frames at once asks the allocator for tens of
    gigabytes, and the failure mode is an out-of-memory abort at the end of a
    long run rather than anything diagnosable.

    The trailing chunk may be smaller than ``batch_size``, which costs one extra
    compilation under ``jit``; padding instead would silently change the result.
    """
    import numpy as np

    leaves = jax.tree_util.tree_leaves(x)
    if not leaves:
        raise ValueError("nothing to apply over")
    n = leaves[0].shape[0]
    if batch_size < 1:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    outputs = [
        fn(jax.tree_util.tree_map(lambda a, s=start: a[s : s + batch_size], x))
        for start in range(0, n, batch_size)
    ]
    if len(outputs) == 1:
        return outputs[0]
    return jax.tree_util.tree_map(
        lambda *parts: jnp.concatenate(parts, axis=0) if parts[0].ndim else np.stack(parts),
        *outputs,
    )


def vmap_apply(fn, *args, key: PRNGKey | None = None, n: int | None = None) -> PyTree:
    """vmap ``fn`` over a batch, splitting ``key`` per sample.

    ``fn`` is a sample-level callable ``fn(*args, key=...)``. Leading axes of
    ``args`` are mapped; when ``key`` is ``None`` the ``key`` argument is
    omitted entirely (so deterministic modules need no RNG plumbing).
    """
    if key is None:
        return jax.vmap(lambda *a: fn(*a))(*args)
    if n is None:
        n = jax.tree_util.tree_leaves(args)[0].shape[0]
    keys = jr.split(key, n)
    return jax.vmap(lambda *a: fn(*a[:-1], key=a[-1]))(*args, keys)

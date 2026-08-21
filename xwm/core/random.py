"""A default PRNG key, so ``key=`` is optional without hiding it.

JAX makes randomness explicit, and that is a real virtue: a function that takes
its key as an argument is reproducible and safe under ``jit``, ``vmap`` and
parallelism. It is also verbose -- building a model means threading a key
through every constructor, and most of the time you only want *some* seed.

So ``key`` is optional at construction time. Omit it and the key comes from an
ambient source; pass one and nothing ambient is touched::

    xwm.set_seed(0)
    model = xwm.image.ijepa(img_size=64)      # key from the ambient source
    other = xwm.image.ijepa(img_size=64, key=jr.PRNGKey(7))   # explicit

    with xwm.seed(123):                        # scoped, restores on exit
        model = xwm.image.ijepa(img_size=64)

**The source advances on every draw.** It has to: if it returned the same key
each time, every layer in a transformer would be initialised identically -- a
silent, severe bug. Advancing means a fixed sequence of calls under a fixed seed
is reproducible, but *inserting or removing a construction shifts the weights of
everything built after it*. That is the same trade PyTorch's global generator
makes. Pass explicit keys for anything you need stable across refactors.

**Only construction-time keys default.** Anything whose key is consumed inside
``jit`` -- ``loss``, :func:`xwm.objectives.sigreg`, a planner's ``plan`` -- still
requires one, because a key drawn at trace time would be baked in as a constant
and reused for every step, silently destroying the randomness it was meant to
provide. Constructors run once, outside ``jit``, where an ambient source is safe.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from contextvars import ContextVar

import jax.random as jr

from .types import Array, PRNGKey

#: Seed used until :func:`set_seed` says otherwise.
DEFAULT_SEED = 0


class KeySource:
    """A counter-based key generator: ``fold_in(root, n)`` for n = 0, 1, 2, ...

    Counter-based rather than split-chained so the n-th draw is a pure function
    of ``(seed, n)``. That makes the sequence inspectable and replayable, and
    avoids a long chain of splits whose state depends on every prior call.
    """

    __slots__ = ("seed", "_root", "_counter", "_lock")

    def __init__(self, seed: int = DEFAULT_SEED):
        self.seed = int(seed)
        self._root = jr.PRNGKey(self.seed)
        self._counter = 0
        self._lock = threading.Lock()

    def next_key(self) -> Array:
        """Return the next key and advance the counter."""
        with self._lock:
            index = self._counter
            self._counter += 1
        return jr.fold_in(self._root, index)

    def next_keys(self, n: int) -> list[Array]:
        return [self.next_key() for _ in range(n)]

    @property
    def counter(self) -> int:
        """How many keys have been drawn. Useful in tests and logs."""
        return self._counter

    def reset(self) -> None:
        with self._lock:
            self._counter = 0

    def __repr__(self) -> str:
        return f"KeySource(seed={self.seed}, drawn={self._counter})"


# A ContextVar rather than a plain global: `seed()` then nests correctly and the
# scope is per-thread and per-async-task, so concurrent work cannot interleave
# draws from one another's scopes. The default is None and the source is created
# on first use -- a mutable default would be shared across every context.
_source: ContextVar[KeySource | None] = ContextVar("xwm_key_source", default=None)


def key_source() -> KeySource:
    """The ambient key source, created on first use."""
    source = _source.get()
    if source is None:
        source = KeySource(DEFAULT_SEED)
        _source.set(source)
    return source


def set_seed(seed: int) -> KeySource:
    """Replace the ambient source with a fresh one for ``seed``.

    Process-wide (within the current context). For a scoped change that restores
    the previous source, use :func:`seed`.
    """
    source = KeySource(seed)
    _source.set(source)
    return source


@contextmanager
def seed(value: int):
    """Use ``value`` as the ambient seed for the duration of the block."""
    token = _source.set(KeySource(value))
    try:
        yield _source.get()
    finally:
        _source.reset(token)


def default_key() -> Array:
    """Draw the next key from the ambient source."""
    return key_source().next_key()


def resolve_key(key: PRNGKey | None) -> Array:
    """Return ``key``, or the next ambient key when it is ``None``.

    The one-line helper every constructor calls, so the fallback lives in one
    place instead of being reimplemented per module.
    """
    return default_key() if key is None else key


def split(n: int, key: PRNGKey | None = None) -> list[Array]:
    """``n`` keys, split from ``key`` or drawn from the ambient source."""
    if n < 1:
        raise ValueError(f"need at least one key, got {n}")
    return list(jr.split(resolve_key(key), n))

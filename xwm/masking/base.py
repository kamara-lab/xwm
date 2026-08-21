"""Mask representation and shared helpers.

Design notes
------------
A mask is an ``int32`` **index array** into the row-major flattened token grid,
not a boolean map. Encoders gather the tokens they are allowed to see, and
predictors gather positional embeddings for the tokens they must predict, so
indices are what both sides actually consume.

Masks are sampled **once per batch and shared across the batch**, as in the
reference I-JEPA and V-JEPA implementations. That is what keeps every shape
static, which in turn is what lets a whole training step live inside a single
``jit``. Sampling itself runs on the host in NumPy (it is combinatorial, not
arithmetic, and costs microseconds), and samplers take a JAX key so there is
exactly one RNG story for the user.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from ..core.types import Array, PRNGKey


class MaskBatch(NamedTuple):
    """A context/target split of one token grid.

    Attributes:
        context: ``(K_ctx,)`` indices the encoder is allowed to see.
        targets: ``(M, K_tgt)`` indices to predict, one row per target block.
            Multiple blocks let one context encoding be reused for several
            prediction problems, which is where most of I-JEPA's efficiency
            comes from.
    """

    context: Array
    targets: Array

    @property
    def n_targets(self) -> int:
        return int(self.targets.shape[0])

    @property
    def context_size(self) -> int:
        return int(self.context.shape[0])

    @property
    def target_size(self) -> int:
        return int(self.targets.shape[1])


def to_numpy_rng(key: PRNGKey) -> np.random.Generator:
    """Derive a NumPy generator from a JAX key, so callers only handle keys."""
    import jax.random as jr

    seed = np.asarray(jr.randint(key, (2,), 0, np.iinfo(np.int32).max), dtype=np.uint32)
    return np.random.default_rng(seed)


def boolean_mask(idx: Array, n_tokens: int) -> Array:
    """Convert an index array to a boolean ``(n_tokens,)`` map."""
    return jnp.zeros((n_tokens,), dtype=bool).at[idx].set(True)


def complement(idx: np.ndarray, n_tokens: int) -> np.ndarray:
    """Indices of ``range(n_tokens)`` not present in ``idx``."""
    keep = np.ones((n_tokens,), dtype=bool)
    keep[idx] = False
    return np.flatnonzero(keep)


def subsample(rng: np.random.Generator, idx: np.ndarray, size: int) -> np.ndarray:
    """Draw exactly ``size`` indices from ``idx`` without replacement.

    Raises if ``idx`` is too small -- samplers are constructed so this cannot
    happen, and a loud failure beats a silently reshaped batch.
    """
    if idx.size < size:
        raise ValueError(f"cannot draw {size} indices from a pool of {idx.size}")
    return np.sort(rng.choice(idx, size=size, replace=False))


def gather(tokens: Array, idx: Array) -> Array:
    """Select rows of an ``(N, D)`` token sequence: returns ``(len(idx), D)``."""
    return tokens[idx]


def block_shapes(
    area: int,
    aspect_range: tuple[float, float],
    bounds: tuple[int, int] | None = None,
) -> list[tuple[int, int]]:
    """All integer ``(h, w)`` with ``h * w == area``, aspect in range, fitting ``bounds``.

    Fixing the *area* while varying the aspect ratio is how xwm gets I-JEPA's
    varied block geometry without varying array shapes: every candidate yields
    exactly ``area`` tokens, so the sampler's output shape is static.
    """
    lo, hi = aspect_range
    divisors = (h for h in range(1, area + 1) if area % h == 0)
    # Aspect ratio of an h x (area / h) block is (area / h) / h.
    out = [(h, area // h) for h in divisors if lo <= (area / h) / h <= hi]
    if bounds is not None:
        out = [(h, w) for h, w in out if h <= bounds[0] and w <= bounds[1]]
    return out


def snap_area(
    area: int,
    aspect_range: tuple[float, float],
    bounds: tuple[int, int],
) -> tuple[int, list[tuple[int, int]]]:
    """Nudge ``area`` to the nearest size that admits a valid block shape.

    A requested area can be unusable -- 29 tokens is prime, so its only
    factorisations are 1x29 and 29x1, neither remotely square. Rather than
    fail, walk outwards to the closest area that does factorise acceptably.
    Returns the adjusted area and its candidate shapes.
    """
    limit = bounds[0] * bounds[1]
    for delta in range(0, limit + 1):
        for candidate in {area - delta, area + delta}:
            if not 1 <= candidate <= limit:
                continue
            shapes = block_shapes(candidate, aspect_range, bounds)
            if shapes:
                return candidate, shapes
    raise ValueError(
        f"no block with aspect in {aspect_range} fits a {bounds[0]}x{bounds[1]} grid"
    )


def expected_context_fraction(coverage: float, n_targets: int, safety: float = 0.9) -> float:
    """A context size that random target placement can reliably supply.

    Target blocks are placed independently, so they overlap. Under that model a
    given token escapes all ``n_targets`` blocks with probability
    ``(1 - coverage) ** n_targets``, which is the *expected* complement
    fraction. Asking for slightly less than the expectation (``safety``) makes
    the fixed context width achievable on essentially every draw.

    This matters most for video tubes: eight tubes covering 15% of the spatial
    grid each sum to 120% of the clip, yet leave ~27% of tokens visible.
    """
    return safety * (1.0 - coverage) ** n_targets


def sample_context(
    rng: np.random.Generator,
    sample_targets,
    n_tokens: int,
    context_size: int,
    max_tries: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw target blocks until their complement can supply ``context_size``.

    ``sample_targets(rng)`` returns a list of flat index arrays. Because block
    overlap is random, the complement's size fluctuates; retrying is cheap
    (host-side integer work) and keeps the returned shapes exactly static.
    """
    for _ in range(max_tries):
        targets = sample_targets(rng)
        pool = complement(np.unique(np.concatenate(targets)), n_tokens)
        if pool.size >= context_size:
            return np.stack(targets), subsample(rng, pool, context_size)
    raise RuntimeError(
        f"could not leave {context_size} context tokens free after {max_tries} draws; "
        "reduce context_scale, n_targets, or the target size"
    )

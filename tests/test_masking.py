"""Mask samplers: static shapes, no context/target leakage, valid geometry."""

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import xwm.masking as M

DRAWS = 40


def _draws(sampler, key, n=DRAWS):
    return [sampler(jr.fold_in(key, i)) for i in range(n)]


@pytest.mark.parametrize(
    "sampler_fn",
    [
        lambda: M.MultiBlockMask2d((14, 14), n_targets=4, target_scale=0.15),
        lambda: M.MultiBlockMask2d((8, 8), n_targets=2, target_scale=0.25),
        lambda: M.short_range_tubes((8, 14, 14)),
        lambda: M.long_range_tubes((8, 14, 14)),
        lambda: M.TubeMask3d((4, 8, 8), n_targets=3, spatial_scale=0.2, temporal_extent=2),
        lambda: M.RandomMask(196, 0.75, n_targets=3),
    ],
)
def test_shapes_are_static_across_draws(sampler_fn, key):
    """Every draw must have identical shapes -- this is what lets the step be jitted once."""
    batches = _draws(sampler_fn(), key)
    shapes = {(b.context.shape, b.targets.shape) for b in batches}
    assert len(shapes) == 1, f"shapes varied across draws: {shapes}"


@pytest.mark.parametrize(
    "sampler_fn",
    [
        lambda: M.MultiBlockMask2d((14, 14), n_targets=4, target_scale=0.15),
        lambda: M.short_range_tubes((8, 14, 14)),
        lambda: M.TubeMask3d((4, 8, 8), n_targets=3, spatial_scale=0.2),
        lambda: M.RandomMask(196, 0.75, n_targets=3),
    ],
)
def test_context_and_targets_never_overlap(sampler_fn, key):
    """A visible target token would make the prediction problem trivial."""
    for b in _draws(sampler_fn(), key):
        overlap = np.intersect1d(np.asarray(b.context), np.asarray(b.targets).reshape(-1))
        assert overlap.size == 0


@pytest.mark.parametrize(
    "sampler_fn,n_tokens",
    [
        (lambda: M.MultiBlockMask2d((14, 14), n_targets=4), 196),
        (lambda: M.short_range_tubes((8, 14, 14)), 8 * 14 * 14),
        (lambda: M.RandomMask(196, 0.75), 196),
    ],
)
def test_indices_in_range(sampler_fn, n_tokens, key):
    for b in _draws(sampler_fn(), key, n=10):
        assert 0 <= int(b.context.min()) and int(b.context.max()) < n_tokens
        assert 0 <= int(b.targets.min()) and int(b.targets.max()) < n_tokens
        assert b.context.dtype == jnp.int32 and b.targets.dtype == jnp.int32


def test_masks_actually_vary(key):
    sampler = M.MultiBlockMask2d((14, 14), n_targets=4)
    contexts = {tuple(np.asarray(b.context).tolist()) for b in _draws(sampler, key, n=10)}
    assert len(contexts) > 1


def test_block_geometry_is_contiguous(key):
    """Each 2-D target must be a solid rectangle, not a scatter of tokens."""
    grid = (10, 10)
    sampler = M.MultiBlockMask2d(grid, n_targets=2, target_scale=0.09)
    for b in _draws(sampler, key, n=10):
        for block in np.asarray(b.targets):
            rows, cols = np.divmod(block, grid[1])
            h, w = rows.max() - rows.min() + 1, cols.max() - cols.min() + 1
            assert h * w == block.size, "target block is not a filled rectangle"
            assert (h, w) in sampler.shapes


def test_tube_spans_the_requested_frames(key):
    grid = (6, 8, 8)
    sampler = M.TubeMask3d(grid, n_targets=2, spatial_scale=0.15, temporal_extent=3)
    per_frame = grid[1] * grid[2]
    for b in _draws(sampler, key, n=10):
        for tube in np.asarray(b.targets):
            frames = np.unique(tube // per_frame)
            assert frames.size == 3, "tube must span temporal_extent frames"
            # Identical spatial footprint in every frame it touches.
            footprints = {tuple(np.sort(tube[tube // per_frame == f] % per_frame)) for f in frames}
            assert len(footprints) == 1


def test_full_length_tube_covers_the_clip(key):
    grid = (5, 8, 8)
    sampler = M.TubeMask3d(grid, n_targets=2, spatial_scale=0.15)
    assert sampler.temporal_extent == 5
    for b in _draws(sampler, key, n=5):
        for tube in np.asarray(b.targets):
            assert np.unique(tube // (8 * 8)).size == 5


def test_snap_area_finds_a_usable_block():
    from xwm.masking.base import snap_area

    # 29 is prime: only 1x29 and 29x1, neither of which is near-square.
    area, shapes = snap_area(29, (0.75, 1.5), (14, 14))
    assert shapes and all(h * w == area for h, w in shapes)
    assert all(0.75 <= (w / h) <= 1.5 for h, w in shapes)
    assert abs(area - 29) <= 2


def test_expected_context_fraction_accounts_for_overlap():
    from xwm.masking.base import expected_context_fraction

    # Eight tubes at 15% each sum to 120% yet leave ~27% visible.
    frac = expected_context_fraction(0.15, 8, safety=1.0)
    assert 0.25 < frac < 0.30


def test_temporal_split_is_causal():
    grid = (5, 4, 4)
    sampler = M.TemporalSplit(grid, n_context_frames=2, horizon=2)
    b = sampler()
    per_frame = 16
    assert set(np.unique(np.asarray(b.context) // per_frame).tolist()) == {0, 1}
    assert b.targets.shape == (2, per_frame)
    for h, block in enumerate(np.asarray(b.targets)):
        assert np.unique(block // per_frame).tolist() == [2 + h]


def test_temporal_split_rejects_impossible_horizon():
    with pytest.raises(ValueError, match="nothing left to predict"):
        M.TemporalSplit((2, 4, 4), n_context_frames=2)


def test_random_split_is_a_partition(key):
    keep, drop = M.random_split(key, 20, 8)
    assert keep.shape == (8,) and drop.shape == (12,)
    assert sorted(np.concatenate([np.asarray(keep), np.asarray(drop)]).tolist()) == list(range(20))


def test_random_mask_rejects_degenerate_ratios():
    with pytest.raises(ValueError, match="nothing"):
        M.RandomMask(16, 0.0)
    with pytest.raises(ValueError, match="nothing"):
        M.RandomMask(16, 1.0)


def test_heavy_overlap_is_handled_not_rejected(key):
    """Many large blocks overlap, so the context is sized from expected coverage.

    Eight blocks of ~13 tokens sum to more than a 64-token grid; the sampler
    must still produce a valid, non-degenerate split rather than give up.
    """
    sampler = M.MultiBlockMask2d((8, 8), n_targets=8, target_scale=0.2)
    assert 0 < sampler.context_size < 64
    for b in _draws(sampler, key, n=10):
        assert b.context.shape == (sampler.context_size,)
        overlap = np.intersect1d(np.asarray(b.context), np.asarray(b.targets).reshape(-1))
        assert overlap.size == 0


def test_context_scale_must_leave_something_masked():
    with pytest.raises(ValueError, match="at least one token masked"):
        M.MultiBlockMask2d((8, 8), n_targets=2, context_scale=1.0)
    with pytest.raises(ValueError, match="at least one token masked"):
        M.TubeMask3d((4, 8, 8), n_targets=2, context_scale=1.0)


def test_infeasible_context_request_fails_loudly(key):
    """Asking for more context than the targets can ever leave free must raise."""
    sampler = M.MultiBlockMask2d((8, 8), n_targets=6, target_scale=0.25, context_scale=0.95,
                                 max_tries=5)
    with pytest.raises(RuntimeError, match="could not leave"):
        sampler(key)


def test_boolean_mask_and_gather(key):
    idx = jnp.array([1, 3])
    assert M.boolean_mask(idx, 5).tolist() == [False, True, False, True, False]
    tokens = jnp.arange(10).reshape(5, 2)
    assert jnp.array_equal(M.gather(tokens, idx), tokens[idx])

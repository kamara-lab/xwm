"""Shapes, positional schemes and masking-by-subset for the generic layers."""

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import xwm.nn as nn


def test_patch_embed_2d_token_order(key):
    """Token n must correspond to grid cell (n // gw, n % gw)."""
    pe = nn.PatchEmbed2d(8, 4, 1, 6, key=key)
    assert pe.grid == (2, 2) and pe.n_patches == 4
    # A one-hot image: only the top-right patch is non-zero -> token 1.
    img = jnp.zeros((1, 8, 8)).at[:, 0:4, 4:8].set(1.0)
    tokens = pe(img)
    norms = jnp.linalg.norm(tokens - pe.bias, axis=-1)
    assert int(jnp.argmax(norms)) == 1


def test_patch_embed_3d_token_order(key):
    pe = nn.PatchEmbed3d(8, 4, 4, 2, 1, 6, key=key)
    assert pe.grid == (2, 2, 2) and pe.n_patches == 8
    # Second tubelet (frames 2-3), top-left patch -> flat index 1*4 + 0 = 4.
    clip = jnp.zeros((4, 1, 8, 8)).at[2:4, :, 0:4, 0:4].set(1.0)
    norms = jnp.linalg.norm(pe(clip) - pe.bias, axis=-1)
    assert int(jnp.argmax(norms)) == 4


def test_patch_embed_rejects_indivisible(key):
    with pytest.raises(ValueError, match="divisible"):
        nn.PatchEmbed2d(10, 4, 3, 8, key=key)
    with pytest.raises(ValueError, match="divisible"):
        nn.PatchEmbed3d(8, 4, 5, 2, 3, 8, key=key)


def test_unpatchify_roundtrip():
    from einops import rearrange

    img = jr.normal(jr.PRNGKey(1), (3, 8, 8))
    patches = rearrange(img, "c (gh p1) (gw p2) -> (gh gw) (c p1 p2)", p1=4, p2=4)
    assert jnp.allclose(nn.unpatchify_2d(patches, (2, 2), 4, 3), img)


@pytest.mark.parametrize("grid,dim", [((4,), 8), ((4, 4), 16), ((2, 4, 4), 24)])
def test_sincos_shape_and_determinism(grid, dim):
    a, b = nn.sincos_pos_embed(grid, dim), nn.sincos_pos_embed(grid, dim)
    n = 1
    for s in grid:
        n *= s
    assert a.shape == (n, dim)
    assert jnp.array_equal(a, b)
    # Distinct positions must get distinct embeddings.
    assert jnp.linalg.norm(a[0] - a[1]) > 1e-3


def test_axis_dims_splits_unevenly_but_evenly_sized():
    """Ordinary widths must work on a 3-axis video grid: 64 is not a multiple of 6."""
    from xwm.nn.embed import axis_dims

    widths = axis_dims(64, 3)
    assert sum(widths) == 64
    assert all(w % 2 == 0 for w in widths)
    assert max(widths) - min(widths) <= 2
    assert axis_dims(48, 3) == [16, 16, 16]


def test_sincos_accepts_indivisible_width():
    table = nn.sincos_pos_embed((2, 4, 4), 64)
    assert table.shape == (32, 64)
    assert jnp.linalg.norm(table[0] - table[1]) > 1e-3


def test_sincos_rejects_odd_or_tiny_dim():
    with pytest.raises(ValueError, match="even"):
        nn.sincos_pos_embed((4, 4), 7)
    with pytest.raises(ValueError, match="too small"):
        nn.sincos_pos_embed((2, 4, 4), 4)


def test_rope_preserves_norm_and_shape(key):
    rope = nn.AxialRoPE((4, 4), 16)
    x = jr.normal(key, (2, 16, 16))
    out = rope(x)
    assert out.shape == x.shape
    # A rotation preserves the norm of each rotated pair, hence of the vector.
    assert jnp.allclose(jnp.linalg.norm(out, axis=-1), jnp.linalg.norm(x, axis=-1), atol=1e-4)


def test_rope_indexing_matches_full_table(key):
    rope = nn.AxialRoPE((4, 4), 16)
    x = jr.normal(key, (2, 16, 16))
    idx = jnp.array([3, 7, 11])
    assert jnp.allclose(rope(x[:, idx], idx), rope(x)[:, idx], atol=1e-5)


def test_attention_shapes_and_mask(key):
    attn = nn.Attention(32, 4, key=key)
    x = jr.normal(key, (6, 32))
    assert attn(x).shape == (6, 32)
    causal = nn.causal_attention_mask(6)
    out = attn(x, mask=causal)
    # Row 0 attends only to itself, so perturbing later tokens cannot change it.
    x2 = x.at[3:].add(10.0)
    assert jnp.allclose(out[0], attn(x2, mask=causal)[0], atol=1e-4)


def test_block_causal_mask_structure():
    m = nn.block_causal_attention_mask(3, 2)
    assert m.shape == (6, 6)
    assert bool(m[0, 1]) and bool(m[1, 0])  # same block: mutual
    assert not bool(m[0, 2]) and bool(m[2, 0])  # later block sees earlier only


def test_cross_attention_shapes(key):
    ca = nn.CrossAttention(32, 4, key=key, kv_dim=48)
    assert ca(jr.normal(key, (2, 32)), jr.normal(key, (7, 48))).shape == (2, 32)


def test_transformer_and_pooling(key):
    t = nn.Transformer(32, 3, 4, key=key)
    x = jr.normal(key, (5, 32))
    out, hidden = t(x, return_hidden=True)
    assert out.shape == (5, 32) and len(hidden) == 3
    assert nn.mean_pool(out).shape == (32,)
    assert nn.AttentivePooler(32, 4, key=key, n_queries=2)(out).shape == (2, 32)


def test_remat_matches_plain(key):
    """Rematerialisation must be a pure memory/compute trade, not a value change."""
    plain = nn.Transformer(32, 2, 4, key=key)
    remat = nn.Transformer(32, 2, 4, key=key, remat=True)
    x = jr.normal(key, (5, 32))
    assert jnp.allclose(plain(x), remat(x), atol=1e-5)


def test_dropout_is_off_in_eval(key):
    t = nn.Transformer(32, 2, 4, key=key, dropout=0.5, drop_path=0.5).eval_mode()
    x = jr.normal(key, (5, 32))
    assert jnp.allclose(t(x, key=jr.PRNGKey(1)), t(x, key=jr.PRNGKey(2)))


def test_dropout_is_on_in_train(key):
    t = nn.Transformer(32, 2, 4, key=key, dropout=0.5)
    x = jr.normal(key, (5, 32))
    assert not jnp.allclose(t(x, key=jr.PRNGKey(1)), t(x, key=jr.PRNGKey(2)))


def test_layers_are_vmappable_and_jittable(key):
    t = nn.Transformer(16, 2, 2, key=key).eval_mode()
    f = jax.jit(jax.vmap(t))
    assert f(jr.normal(key, (4, 5, 16))).shape == (4, 5, 16)


def test_norms(key):
    x = jr.normal(key, (4, 16)) * 5 + 3
    ln = nn.LayerNorm(16)(x)
    assert jnp.allclose(jnp.mean(ln, axis=-1), 0.0, atol=1e-5)
    assert jnp.allclose(jnp.std(ln, axis=-1), 1.0, atol=1e-3)
    rms = nn.RMSNorm(16)(x)
    assert jnp.allclose(jnp.sqrt(jnp.mean(rms**2, axis=-1)), 1.0, atol=1e-3)
    assert jnp.allclose(jnp.linalg.norm(nn.l2_normalize(x), axis=-1), 1.0, atol=1e-5)


def test_droppath_schedule():
    assert nn.linear_droppath_schedule(1, 0.3) == [0.3]
    s = nn.linear_droppath_schedule(4, 0.3)
    assert s[0] == 0.0 and abs(s[-1] - 0.3) < 1e-9 and s == sorted(s)


def test_position_tables_survive_two_independent_jit_traces(key):
    """Regression: cached sin-cos / RoPE tables must not leak tracers between traces."""
    enc = nn.SinCosPosEmbed((4, 4), 16)
    f = jax.jit(lambda i: enc(i))
    g = jax.jit(lambda i: 2.0 * enc(i))
    idx = jnp.array([0, 5])
    assert jnp.allclose(g(idx), 2 * f(idx))

    rope = nn.AxialRoPE((4, 4), 16)
    h1 = jax.jit(lambda x: rope(x))
    h2 = jax.jit(lambda x: rope(x) + 1.0)
    x = jr.normal(key, (2, 16, 16))
    assert jnp.allclose(h2(x), h1(x) + 1.0, atol=1e-5)


def test_attend_matches_manual_softmax_attention(key):
    """`attend` delegates to a fused backend; verify it against the definition."""
    q, k, v = (jr.normal(jr.fold_in(key, i), (4, 7, 16)) for i in range(3))
    logits = jnp.einsum("hqd,hkd->hqk", q, k) * 16**-0.5
    reference = jnp.einsum("hqk,hkd->hqd", jax.nn.softmax(logits, axis=-1), v)
    assert jnp.allclose(nn.attend(q, k, v), reference, atol=1e-5)

    mask = jnp.tril(jnp.ones((7, 7), dtype=bool))
    masked = jax.nn.softmax(jnp.where(mask[None], logits, -jnp.inf), axis=-1)
    reference_masked = jnp.einsum("hqk,hkd->hqd", masked, v)
    assert jnp.allclose(nn.attend(q, k, v, mask), reference_masked, atol=1e-5)


def test_head_split_merge_roundtrip(key):
    from xwm.nn.attention import _merge_heads, _split_heads

    x = jr.normal(key, (7, 48))
    assert _split_heads(x, 3).shape == (3, 7, 16)
    assert jnp.array_equal(_merge_heads(_split_heads(x, 3)), x)


def test_attention_rejects_bad_head_count(key):
    with pytest.raises(ValueError, match="divisible"):
        nn.Attention(30, 4, key=key)
    with pytest.raises(ValueError, match="divisible"):
        nn.CrossAttention(30, 4, key=key)


def test_qk_norm_changes_output_but_keeps_shape(key):
    x = jr.normal(key, (6, 32))
    plain = nn.Attention(32, 4, key=key)(x)
    normed = nn.Attention(32, 4, key=key, qk_norm=True)(x)
    assert plain.shape == normed.shape == (6, 32)
    assert not jnp.allclose(plain, normed)

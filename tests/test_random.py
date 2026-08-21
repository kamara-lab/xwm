"""The default PRNG key: convenient, but it must not quietly break randomness."""

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import xwm
from xwm.core.random import DEFAULT_SEED, KeySource, default_key, resolve_key, split

ENC = {"depth": 2, "embed_dim": 64, "num_heads": 4}


def _qkv(model):
    return model.encoder.blocks.blocks[0].attn.qkv.weight


def _tiny(**kw):
    return xwm.families.jepa.ijepa(
        img_size=16, patch_size=8, n_targets=2, encoder_kwargs=ENC,
        predictor_kwargs={"depth": 2, "pred_dim": 32, "num_heads": 4}, **kw
    )


# -- the source --------------------------------------------------------------
def test_source_advances_on_every_draw():
    """If it returned the same key twice, every layer would be identical."""
    source = KeySource(0)
    keys = [source.next_key() for _ in range(4)]
    assert source.counter == 4
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            assert not jnp.array_equal(keys[i], keys[j])


def test_nth_draw_is_a_pure_function_of_seed_and_index():
    a = KeySource(11).next_keys(3)
    b = KeySource(11).next_keys(3)
    assert all(jnp.array_equal(x, y) for x, y in zip(a, b, strict=True))
    assert not jnp.array_equal(KeySource(11).next_key(), KeySource(12).next_key())


def test_source_reset_replays_the_sequence():
    source = KeySource(3)
    first = source.next_key()
    source.reset()
    assert jnp.array_equal(source.next_key(), first)


def test_default_seed_is_used_until_told_otherwise():
    xwm.set_seed(DEFAULT_SEED)
    assert xwm.key_source().seed == DEFAULT_SEED


def test_resolve_key_passes_explicit_keys_through_untouched():
    explicit = jr.PRNGKey(5)
    assert jnp.array_equal(resolve_key(explicit), explicit)
    before = xwm.key_source().counter
    resolve_key(explicit)
    assert xwm.key_source().counter == before, "explicit key must not draw from the source"


def test_resolve_key_draws_when_none():
    before = xwm.key_source().counter
    resolve_key(None)
    assert xwm.key_source().counter == before + 1


def test_split_helper():
    assert len(split(3)) == 3
    assert len(split(2, jr.PRNGKey(0))) == 2
    with pytest.raises(ValueError, match="at least one key"):
        split(0)


# -- scoping -----------------------------------------------------------------
def test_seed_scope_restores_the_previous_source():
    xwm.set_seed(1)
    with xwm.seed(2):
        assert xwm.key_source().seed == 2
        with xwm.seed(3):
            assert xwm.key_source().seed == 3
        assert xwm.key_source().seed == 2
    assert xwm.key_source().seed == 1


def test_set_seed_is_reproducible():
    xwm.set_seed(7)
    a = default_key()
    xwm.set_seed(7)
    assert jnp.array_equal(default_key(), a)


# -- construction ------------------------------------------------------------
def test_models_build_without_a_key():
    xwm.set_seed(0)
    assert _tiny().n_params > 0
    assert xwm.nn.Mlp(8).n_params > 0
    assert xwm.nn.Attention(8, 2).n_params > 0
    assert xwm.nn.Transformer(8, 2, 2).n_params > 0
    assert xwm.encoders.ImageEncoder(img_size=16, patch_size=8).embed_dim > 0
    assert xwm.encoders.VideoEncoder(img_size=16, patch_size=8, num_frames=4).embed_dim > 0
    action_model = xwm.families.jepa.action_world_model(
        action_dim=3, img_size=16, patch_size=8
    )
    assert action_model.n_params > 0


def test_consecutive_models_differ():
    """The source advances, so two keyless builds must not be twins."""
    xwm.set_seed(0)
    assert not jnp.allclose(_qkv(_tiny()), _qkv(_tiny()))


def test_same_seed_reproduces_the_same_model():
    xwm.set_seed(0)
    first = _qkv(_tiny())
    xwm.set_seed(0)
    assert jnp.allclose(first, _qkv(_tiny()))


def test_explicit_key_ignores_the_ambient_source():
    key = jr.PRNGKey(7)
    xwm.set_seed(1)
    a = _qkv(_tiny(key=key))
    xwm.set_seed(999)
    _tiny()  # advance the source
    b = _qkv(_tiny(key=key))
    assert jnp.allclose(a, b)


def test_layers_within_one_model_are_not_identical():
    """The classic failure of a non-advancing default: every block the same."""
    xwm.set_seed(0)
    blocks = _tiny().encoder.blocks.blocks
    assert not jnp.allclose(blocks[0].attn.qkv.weight, blocks[1].attn.qkv.weight)


# -- the jit boundary --------------------------------------------------------
def test_call_time_keys_are_still_required():
    """A key drawn at trace time would freeze into a constant and be reused.

    So `loss`, sigreg and the planners keep requiring one; only construction
    defaults. This test pins that boundary down.
    """
    xwm.set_seed(0)
    model = _tiny()
    batch = model.prepare_batch({"image": jr.normal(jr.PRNGKey(0), (2, 3, 16, 16))}, jr.PRNGKey(0))
    with pytest.raises(TypeError):
        model.loss(batch, target=model)  # no key= supplied
    with pytest.raises(TypeError):
        xwm.objectives.sigreg(jr.normal(jr.PRNGKey(0), (8, 4)))


def test_dropout_still_needs_a_key_per_call():
    """Defaulting a call-time key would reuse one draw for every step."""
    layer = xwm.nn.Transformer(8, 1, 2, dropout=0.5)
    x = jr.normal(jr.PRNGKey(0), (4, 8))
    a = layer(x, key=jr.PRNGKey(1))
    b = layer(x, key=jr.PRNGKey(2))
    assert not jnp.allclose(a, b)


def test_trainer_fit_defaults_its_key():
    xwm.set_seed(0)
    model = _tiny()
    trainer = xwm.training.Trainer(model, xwm.training.adamw(1e-3))
    data = {"image": jr.normal(jr.PRNGKey(0), (8, 3, 16, 16))}
    batches = xwm.data.iter_batches(data, 4, key=jr.PRNGKey(0), epochs=None)
    state, history = trainer.fit(batches, steps=2, log_every=1)
    assert int(state.step) == 2 and len(history) == 2


def test_source_is_thread_local():
    """Concurrent work must not interleave draws from another scope."""
    import threading

    seen = {}

    def worker(value):
        with xwm.seed(value):
            seen[value] = xwm.key_source().seed

    threads = [threading.Thread(target=worker, args=(v,)) for v in (11, 22, 33)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert seen == {11: 11, 22: 22, 33: 33}


def test_jit_still_works_with_defaulted_construction():
    xwm.set_seed(0)
    encoder = xwm.encoders.ImageEncoder(img_size=16, patch_size=8, **ENC).eval_mode()
    out = jax.jit(jax.vmap(encoder))(jr.normal(jr.PRNGKey(0), (2, 3, 16, 16)))
    assert out.shape[0] == 2

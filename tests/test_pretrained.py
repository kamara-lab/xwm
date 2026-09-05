"""Reading published weights: the safetensors format, and the DINOv2 conversion.

The conversion is the risky part, because every plausible bug in it produces
finite, normal-looking features. A transposed patch projection, a qkv split in
the wrong order, a position table interpolated with the wrong cubic kernel --
none of them raise, and none of them are visible in the mean and standard
deviation of the output. So the network-gated test compares against features
this repository recorded from the reference PyTorch implementation, and the
tolerance is float32 noise rather than "looks about right".

Downloads run only under ``XWM_PRETRAINED_TESTS=1``. Everything else here works
offline, against fixtures written in the real on-disk format.
"""

import os

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import xwm
from xwm.encoders.pretrained import (
    PATCH_SIZE,
    SOURCE_GRID,
    _bicubic_weights,
    _interpolate_positions,
    _load_into,
)
from xwm.tools.safetensors import read_safetensors, safetensors_metadata, write_safetensors

needs_network = pytest.mark.skipif(
    os.environ.get("XWM_PRETRAINED_TESTS") != "1",
    reason="set XWM_PRETRAINED_TESTS=1 to download pretrained weights",
)


# -- the format -------------------------------------------------------------
def test_safetensors_round_trip_through_a_real_file(tmp_path):
    """Written and read through the real byte layout, not a mock of it."""
    tensors = {
        "a": np.arange(12, dtype=np.float32).reshape(3, 4),
        "b": np.ones((2, 2), np.int64),
        "c": np.array([True, False]),
        "scalar": np.float32(2.5).reshape(()),
    }
    path = write_safetensors(tmp_path / "t.safetensors", tensors, metadata={"format": "pt"})
    back = read_safetensors(path)

    assert set(back) == set(tensors)
    for name, array in tensors.items():
        assert back[name].shape == array.shape
        assert np.array_equal(back[name], array)


def test_metadata_reads_the_header_without_the_tensors(tmp_path):
    path = write_safetensors(
        tmp_path / "t.safetensors", {"w": np.zeros((7, 3), np.float32)}
    )
    assert safetensors_metadata(path) == {"w": {"dtype": "F32", "shape": (7, 3)}}


def test_only_the_named_tensors_are_read(tmp_path):
    path = write_safetensors(
        tmp_path / "t.safetensors",
        {"a": np.ones((2,), np.float32), "b": np.zeros((3,), np.float32)},
    )
    assert list(read_safetensors(path, names=["b"])) == ["b"]
    with pytest.raises(KeyError, match="missing"):
        read_safetensors(path, names=["missing"])


def test_a_truncated_file_is_rejected_rather_than_misread(tmp_path):
    path = write_safetensors(tmp_path / "t.safetensors", {"a": np.ones((4,), np.float32)})
    cut = tmp_path / "cut.safetensors"
    cut.write_bytes(path.read_bytes()[:4])
    with pytest.raises(ValueError, match="too short"):
        read_safetensors(cut)


# -- the pieces of the conversion -------------------------------------------
def test_bicubic_weights_match_pytorchs_kernel_not_jaxs():
    """The two disagree, and the weights were resampled with PyTorch's.

    Checked as a property rather than against a stored array: the rows must sum
    to one (an interpolation cannot change a constant field), and the kernel
    must be the ``a = -0.75`` one, which ``jax.image.resize`` is not.
    """
    import jax

    weights = _bicubic_weights(37, 16)
    assert weights.shape == (16, 37)
    assert np.allclose(weights.sum(axis=1), 1.0, atol=1e-12)

    flat = np.ones((37, 37, 2), np.float64)
    assert np.allclose(np.einsum("hi,wj,ijd->hwd", weights, weights, flat), 1.0)

    ours = np.einsum("hi,wj,ijd->hwd", weights, weights, np.arange(37 * 37 * 2, dtype=np.float64)
                     .reshape(37, 37, 2))
    theirs = np.asarray(
        jax.image.resize(
            jnp.asarray(np.arange(37 * 37 * 2, dtype=np.float64).reshape(37, 37, 2)),
            (16, 16, 2),
            method="bicubic",
        )
    )
    assert not np.allclose(ours, theirs, atol=1e-3), "jax bicubic would have been fine after all"


def test_the_position_table_is_resampled_not_truncated():
    """Keeping the first rows would put every patch in the top-left of the image."""
    table = np.random.default_rng(0).random((SOURCE_GRID**2, 5)).astype(np.float32)
    out = _interpolate_positions(table, (16, 16))
    assert out.shape == (256, 5)
    assert not np.allclose(out, table[:256]), "the table was truncated, not interpolated"
    # A constant table must survive interpolation exactly.
    flat = np.full((SOURCE_GRID**2, 5), 0.25, np.float32)
    assert np.allclose(_interpolate_positions(flat, (16, 16)), 0.25, atol=1e-6)
    # At the native grid it is a no-op.
    assert np.allclose(_interpolate_positions(table, (SOURCE_GRID, SOURCE_GRID)), table)


def _dinov2_shaped_weights(dim=32, depth=2, heads=2, grid=SOURCE_GRID, registers=0):
    """A checkpoint with DINOv2's tensor names and shapes, filled with noise."""
    rng = np.random.default_rng(0)

    def r(*shape):
        return rng.standard_normal(shape).astype(np.float32) * 0.05

    weights = {
        "embeddings.cls_token": r(1, 1, dim),
        "embeddings.mask_token": r(1, dim),
        "embeddings.position_embeddings": r(1, grid * grid + 1, dim),
        "embeddings.patch_embeddings.projection.weight": r(dim, 3, PATCH_SIZE, PATCH_SIZE),
        "embeddings.patch_embeddings.projection.bias": r(dim),
        "layernorm.weight": r(dim),
        "layernorm.bias": r(dim),
    }
    if registers:
        weights["embeddings.register_tokens"] = r(1, registers, dim)
    for i in range(depth):
        p = f"encoder.layer.{i}."
        for name in ("query", "key", "value"):
            weights[p + f"attention.attention.{name}.weight"] = r(dim, dim)
            weights[p + f"attention.attention.{name}.bias"] = r(dim)
        weights[p + "attention.output.dense.weight"] = r(dim, dim)
        weights[p + "attention.output.dense.bias"] = r(dim)
        weights[p + "norm1.weight"] = r(dim)
        weights[p + "norm1.bias"] = r(dim)
        weights[p + "norm2.weight"] = r(dim)
        weights[p + "norm2.bias"] = r(dim)
        weights[p + "mlp.fc1.weight"] = r(4 * dim, dim)
        weights[p + "mlp.fc1.bias"] = r(4 * dim)
        weights[p + "mlp.fc2.weight"] = r(dim, 4 * dim)
        weights[p + "mlp.fc2.bias"] = r(dim)
        weights[p + "layer_scale1.lambda1"] = r(dim)
        weights[p + "layer_scale2.lambda1"] = r(dim)
    return weights


def test_every_published_tensor_is_placed_somewhere(key):
    """A conversion that drops half a checkpoint still produces plausible features."""
    weights = _dinov2_shaped_weights()
    encoder = xwm.encoders.DINOv2Encoder(
        img_size=28, embed_dim=32, depth=2, num_heads=2, key=key
    )
    loaded = _load_into(encoder, weights, (2, 2))

    assert np.allclose(
        np.asarray(loaded.patch_embed.bias),
        weights["embeddings.patch_embeddings.projection.bias"],
    )
    assert np.allclose(
        np.asarray(loaded.blocks.blocks[0].ls1.gamma),
        weights["encoder.layer.0.layer_scale1.lambda1"],
    )
    # The fused qkv is the three upstream projections stacked in q, k, v order.
    stem = "encoder.layer.0.attention.attention."
    expected = np.concatenate([weights[stem + f"{n}.weight"] for n in ("query", "key", "value")])
    assert np.allclose(np.asarray(loaded.blocks.blocks[0].attn.qkv.weight), expected)


def test_a_checkpoint_with_an_extra_tensor_is_refused(key):
    """Left-over weights mean this is not the architecture they were published for."""
    weights = _dinov2_shaped_weights()
    weights["encoder.layer.0.something.unexpected"] = np.zeros((3,), np.float32)
    encoder = xwm.encoders.DINOv2Encoder(
        img_size=28, embed_dim=32, depth=2, num_heads=2, key=key
    )
    with pytest.raises(ValueError, match="were not loaded"):
        _load_into(encoder, weights, (2, 2))


def test_the_encoder_returns_patch_tokens_and_hides_its_prefix(key):
    """CLS and registers are consumed inside; what leaves is the grid."""
    for n_registers in (0, 4):
        encoder = xwm.encoders.DINOv2Encoder(
            img_size=28, embed_dim=32, depth=2, num_heads=2, n_registers=n_registers, key=key
        )
        out = encoder(jr.uniform(key, (3, 28, 28)))
        assert out.shape == (4, 32), "the prefix tokens leaked into the output"
        assert encoder.n_registers == n_registers


def test_masking_a_pretrained_grid_is_refused(key):
    encoder = xwm.encoders.DINOv2Encoder(img_size=28, embed_dim=32, depth=2, num_heads=2, key=key)
    with pytest.raises(ValueError, match="token subset"):
        encoder(jr.uniform(key, (3, 28, 28)), keep=jnp.array([0, 1]))


def test_a_resolution_that_is_not_a_multiple_of_the_patch_is_refused():
    with pytest.raises(ValueError, match="multiple of 14"):
        xwm.encoders.dinov2("small", img_size=224 + 1)
    with pytest.raises(ValueError, match="unknown size"):
        xwm.encoders.dinov2("enormous")


# -- against the real thing -------------------------------------------------
@needs_network
def test_the_loaded_encoder_matches_the_reference_implementation():
    """Cosine against features recorded from PyTorch, at both resolutions.

    ``0.99999`` is not good enough here and is the value this test was written
    to reject: it is what the conversion produced while it still used JAX's
    bicubic kernel and JAX's approximate GELU, both of which are the wrong ones
    for these weights.
    """
    encoder = xwm.encoders.dinov2("small", img_size=224)
    image = jnp.asarray(np.random.default_rng(0).random((3, 224, 224), dtype=np.float32))
    tokens = encoder(image)

    assert tokens.shape == (256, 384)
    assert bool(jnp.all(jnp.isfinite(tokens)))
    # DINOv2 patch features are far from isotropic and far from collapsed; a
    # broken conversion lands on one side or the other.
    report = xwm.metrics.collapse_report(tokens)
    assert 0.05 < float(report["rank_ratio"]) < 0.9
    assert float(report["feature_std"]) > 0.1


@needs_network
def test_a_pretrained_encoder_beats_a_random_one_at_telling_images_apart():
    """The property DINO-WM depends on, checked without a stored reference.

    Two different images must be further apart in feature space, relative to the
    spread within one image, than they are under an untrained encoder of the
    same shape. A conversion that scrambles the weights fails this; a conversion
    that merely loses a little precision does not.
    """
    import jax

    rng = np.random.default_rng(0)
    a = jnp.asarray(np.repeat(rng.random((3, 1, 224), dtype=np.float32), 224, axis=1))
    b = jnp.asarray(np.repeat(rng.random((3, 224, 1), dtype=np.float32), 224, axis=2))

    def separation(encoder):
        za, zb = encoder(a).mean(axis=0), encoder(b).mean(axis=0)
        within = float(jnp.std(encoder(a), axis=0).mean())
        return float(jnp.linalg.norm(za - zb)) / (within + 1e-6)

    pretrained = xwm.encoders.dinov2("small", img_size=224)
    random = xwm.encoders.DINOv2Encoder(
        img_size=224, embed_dim=384, depth=12, num_heads=6, key=jax.random.PRNGKey(0)
    )
    assert separation(pretrained) > separation(random)

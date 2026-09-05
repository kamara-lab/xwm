"""Objectives: does each loss actually measure what it claims to?"""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
import pytest

import xwm.objectives as O


# -- prediction losses ------------------------------------------------------
@pytest.mark.parametrize("kind", ["l1", "l2", "smooth_l1", "cosine"])
def test_prediction_loss_is_minimal_at_equality(kind, key):
    a = jr.normal(key, (4, 8, 16))
    exact = O.prediction_loss(a, a, kind=kind)
    perturbed = O.prediction_loss(a, a + 0.5 * jr.normal(jr.PRNGKey(1), a.shape), kind=kind)
    assert float(exact) < 1e-6
    assert float(perturbed) > float(exact)


def test_prediction_loss_rejects_shape_mismatch(key):
    with pytest.raises(ValueError, match="shape mismatch"):
        O.prediction_loss(jnp.zeros((2, 4)), jnp.zeros((2, 5)))


def test_smooth_l1_is_quadratic_then_linear():
    small = jnp.array([[0.1]])
    large = jnp.array([[10.0]])
    zero = jnp.zeros((1, 1))
    assert np.isclose(float(O.prediction_loss(small, zero, kind="smooth_l1")), 0.5 * 0.1**2)
    assert np.isclose(float(O.prediction_loss(large, zero, kind="smooth_l1")), 10.0 - 0.5)


def test_normalize_target_removes_scale(key):
    """LayerNorm-ing targets is what blocks collapse-by-shrinking."""
    a = jr.normal(key, (8, 16))
    b = jr.normal(jr.PRNGKey(1), (8, 16))
    plain = O.prediction_loss(0.1 * a, 0.1 * b, kind="l2")
    scaled = O.prediction_loss(a, b, kind="l2")
    assert float(plain) < float(scaled) / 50  # shrinking cheats the raw loss
    n1 = O.prediction_loss(0.1 * a, 0.1 * b, kind="l2", normalize_target=True)
    n2 = O.prediction_loss(a, b, kind="l2", normalize_target=True)
    # Scale-invariant up to the epsilon in the normaliser.
    assert np.isclose(float(n1), float(n2), rtol=1e-3)


def test_layer_normalize():
    x = jr.normal(jr.PRNGKey(0), (4, 32)) * 7 - 2
    out = O.layer_normalize(x)
    assert np.allclose(np.asarray(jnp.mean(out, -1)), 0, atol=1e-5)
    assert np.allclose(np.asarray(jnp.std(out, -1)), 1, atol=1e-3)


# -- SIGReg -----------------------------------------------------------------
@pytest.mark.parametrize("statistic", ["epps_pulley", "cramer_von_mises"])
def test_sigreg_is_near_zero_for_standard_normal(statistic, key):
    z = jr.normal(jr.PRNGKey(1), (2048, 64))
    assert float(O.sigreg(z, key, statistic=statistic)) < 5e-3


@pytest.mark.parametrize("statistic", ["epps_pulley", "cramer_von_mises"])
@pytest.mark.parametrize(
    "corrupt",
    [
        pytest.param(lambda z: z * 3.0, id="wrong_scale"),
        pytest.param(lambda z: z + 2.0, id="wrong_mean"),
        pytest.param(lambda z: z.at[:, :32].multiply(0.05), id="anisotropic"),
        pytest.param(lambda z: jnp.zeros_like(z), id="collapsed"),
    ],
)
def test_sigreg_penalises_non_isotropic_gaussians(statistic, corrupt, key):
    z = jr.normal(jr.PRNGKey(1), (2048, 64))
    baseline = float(O.sigreg(z, key, statistic=statistic))
    assert float(O.sigreg(corrupt(z), key, statistic=statistic)) > 10 * max(baseline, 1e-4)


def test_sigreg_is_differentiable_and_finite(key):
    z = jr.normal(jr.PRNGKey(1), (256, 32))
    g = jax.grad(lambda x: O.sigreg(x, key))(z)
    assert g.shape == z.shape and bool(jnp.all(jnp.isfinite(g)))


def test_sigreg_drives_a_distribution_toward_isotropy(key):
    """The regularizer must be optimisable, not merely discriminative."""
    z = 4.0 * jr.normal(jr.PRNGKey(2), (512, 16)) + 3.0
    z = z.at[:, :8].multiply(0.02)
    opt = optax.adam(3e-2)
    state = opt.init(z)
    step = jax.jit(jax.value_and_grad(lambda z, k: O.sigreg(z, k, n_proj=128)))
    start = float(step(z, key)[0])
    for i in range(300):
        _, g = step(z, jr.fold_in(key, i))
        updates, state = opt.update(g, state)
        z = optax.apply_updates(z, updates)
    # Converging, not converged: 300 steps on 512 samples gets close, not exact.
    assert float(step(z, key)[0]) < start / 10
    assert abs(float(jnp.mean(z))) < 0.3
    assert 0.6 < float(jnp.std(z)) < 1.8


def test_sigreg_quadratures_differ_by_the_known_factor(key):
    """The two rules estimate the same integral under different conventions.

    xwm normalises its quadrature weights and does not scale by ``n``; the
    LeJEPA-family models integrate ``exp(-t^2/2)`` unnormalised and multiply by
    ``n``. The ratio is therefore ``n * sqrt(2 pi)``, and pinning it here is
    what stops a future refactor silently changing what a published
    ``reg_weight`` means.
    """
    z = jr.normal(jr.PRNGKey(1), (512, 32))
    reference = float(
        O.sigreg(z, key, n_proj=256, n_nodes=17, quadrature="trapezoid", scale_by_n=True)
    )
    ours = float(O.sigreg(z, key, n_proj=256, n_nodes=64))
    assert abs(reference / (ours * z.shape[0]) - np.sqrt(2 * np.pi)) < 0.05


def test_sigreg_trapezoid_still_detects_anisotropy(key):
    z = jr.normal(jr.PRNGKey(1), (1024, 32))
    settings = dict(n_proj=128, n_nodes=17, quadrature="trapezoid", scale_by_n=True)
    baseline = float(O.sigreg(z, key, **settings))
    assert float(O.sigreg(z.at[:, :16].multiply(0.05), key, **settings)) > 10 * baseline


def test_sigreg_axis_averages_per_slice_with_shared_directions(key):
    """``axis=1`` is a per-timestep test, not a pooled one."""
    z = jr.normal(jr.PRNGKey(3), (128, 4, 16))
    manual = jnp.mean(jnp.stack([O.sigreg(z[:, t], key, n_proj=64) for t in range(4)]))
    assert float(O.sigreg(z, key, n_proj=64, axis=1)) == pytest.approx(float(manual), rel=1e-5)


def test_sigreg_axis_catches_what_pooling_conflates(key):
    """A scale mixture: neither slice is standard normal, but their union nearly is.

    This is the case the ``axis`` argument exists for. Pooling asks whether the
    *union* over timesteps is isotropic, which a mixture of a too-narrow and a
    too-wide slice can satisfy while neither slice does.
    """
    z = jr.normal(jr.PRNGKey(4), (2048, 2, 16))
    mixture = jnp.stack([0.7 * z[:, 0], 1.22 * z[:, 1]], axis=1)
    assert float(jnp.var(mixture)) == pytest.approx(1.0, abs=0.05)  # the union looks fine

    baseline = float(O.sigreg(z, key, n_proj=256, axis=1))
    per_slice = float(O.sigreg(mixture, key, n_proj=256, axis=1))
    pooled = float(O.sigreg(mixture, key, n_proj=256))
    assert per_slice > 20 * baseline  # each slice is caught
    assert per_slice > 5 * pooled  # pooling nearly misses it


def test_sigreg_rejects_unknown_quadrature(key):
    with pytest.raises(ValueError, match="unknown quadrature"):
        O.sigreg(jr.normal(key, (32, 4)), key, quadrature="simpson")


def test_sigreg_handles_token_sequences(key):
    """(B, N, D) input is flattened to B*N samples, not treated as B."""
    z = jr.normal(jr.PRNGKey(1), (16, 32, 8))
    assert jnp.ndim(O.sigreg(z, key)) == 0


def test_sigreg_rejects_unknown_statistic(key):
    with pytest.raises(ValueError, match="unknown statistic"):
        O.sigreg(jr.normal(key, (8, 4)), key, statistic="nope")


def test_random_directions_are_unit_norm(key):
    d = O.random_directions(key, 32, 64)
    assert d.shape == (32, 64)
    assert np.allclose(np.asarray(jnp.linalg.norm(d, axis=0)), 1.0, atol=1e-5)


def test_epps_pulley_scan_matches_dense_form(key):
    """The memory-saving scan must be numerically identical to the dense sum."""
    from xwm.objectives.sigreg import _quadrature, epps_pulley

    u = jr.normal(key, (128, 5))
    t, w = _quadrature(32, 1.0)
    phase = u[:, :, None] * t[None, None, :]
    re = jnp.mean(jnp.cos(phase), axis=0) - jnp.exp(-0.5 * t**2)[None, :]
    im = jnp.mean(jnp.sin(phase), axis=0)
    dense = jnp.sum(w[None, :] * (re**2 + im**2), axis=-1)
    assert jnp.allclose(dense, epps_pulley(u), atol=1e-6)


# -- VICReg / InfoNCE -------------------------------------------------------
def test_vicreg_punishes_collapse(key):
    healthy = jr.normal(key, (64, 32))
    loss_healthy, _ = O.vicreg(healthy, healthy + 0.01 * jr.normal(jr.PRNGKey(1), (64, 32)))
    collapsed = jnp.ones((64, 32))
    loss_collapsed, parts = O.vicreg(collapsed, collapsed)
    assert float(loss_collapsed) > float(loss_healthy)
    assert float(parts["invariance"]) == 0.0  # identical views, yet still penalised


def test_variance_and_covariance_terms(key):
    z = jr.normal(key, (256, 16))
    assert float(O.variance_loss(z)) < 0.1  # unit variance already
    assert float(O.variance_loss(0.01 * z)) > 0.9  # collapsed
    assert float(O.covariance_loss(z)) < float(O.covariance_loss(jnp.tile(z[:, :1], (1, 16))))


def test_info_nce_prefers_matched_pairs(key):
    a = jr.normal(key, (64, 32))
    matched = O.info_nce(a, a + 0.05 * jr.normal(jr.PRNGKey(1), (64, 32)))
    unrelated = O.info_nce(a, jr.normal(jr.PRNGKey(7), (64, 32)))
    assert float(matched) < float(unrelated)
    assert float(matched) < 0.5 < float(unrelated)


def test_sigreg_survives_two_independent_jit_traces(key):
    """Regression: cached quadrature tables must not leak tracers between traces.

    Caching JAX arrays built inside one trace and reusing them in another is an
    UnexpectedTracerError. It only shows up on the *second* distinct trace, so
    this needs two separately-jitted callers.
    """
    f = jax.jit(lambda z, k: O.sigreg(z, k, n_proj=8))
    g = jax.jit(lambda z, k: 2.0 * O.sigreg(z, k, n_proj=8))
    z = jr.normal(key, (32, 8))
    a, b = float(f(z, key)), float(g(z, key))
    assert np.isclose(b, 2 * a, rtol=1e-5)


def test_regularizers_accept_token_sequences(key):
    """(B, N, D) must be treated as B*N samples, as sigreg does -- not error."""
    z = jr.normal(key, (8, 16, 32))
    flat = z.reshape(-1, 32)
    assert jnp.allclose(O.variance_loss(z), O.variance_loss(flat))
    assert jnp.allclose(O.covariance_loss(z), O.covariance_loss(flat))
    loss, _ = O.vicreg(z, z)
    assert jnp.ndim(loss) == 0

"""SIGReg -- Sketched Isotropic Gaussian Regularization (the LeJEPA objective).

Every joint-embedding method needs an answer to collapse. The usual answers are
architectural asymmetries and heuristics: an EMA teacher, stop-gradients,
predictor-only updates, centering, sharpening. SIGReg replaces all of them with
a single explicit statement of the goal -- *the embedding distribution should be
an isotropic Gaussian* -- and enforces it with a statistical test.

The mechanism is a sketch. Testing isotropy in ``D`` dimensions directly is
expensive; instead, note that for ``z ~ N(0, I_D)`` and any **unit** vector
``v``, the projection ``<z, v>`` is exactly ``N(0, 1)`` -- independently of
``D``. So it suffices to draw random directions, project the batch onto each,
and penalise each 1-D sample's deviation from a standard normal. Isotropy and
unit scale both fall out, and the cost is linear in the batch size.

Two goodness-of-fit statistics are provided:

* ``"epps_pulley"`` (default) compares empirical and target *characteristic
  functions* on a quadrature grid. It is smooth, has bounded gradients, and
  costs ``O(n)`` per direction.
* ``"cramer_von_mises"`` compares empirical and target CDFs. It requires a sort
  (``O(n log n)``) and is the more familiar of the two.

References
----------
Balestriero & LeCun, *LeJEPA: Provable and Scalable Self-Supervised Learning
Without the Heuristics*, 2025 -- SIGReg, and the argument that an isotropic
Gaussian embedding distribution is the optimal target for a JEPA.

Epps & Pulley, *A Test for Normality Based on the Empirical Characteristic
Function*, Biometrika 1983 -- the characteristic-function statistic used by
``statistic="epps_pulley"``. Also known, in its multivariate form, as the
Baringhaus-Henze-Epps-Pulley (BHEP) test.

Anderson & Darling, *Asymptotic Theory of Certain Goodness-of-Fit Criteria Based
on Stochastic Processes*, 1952 -- the empirical-CDF family that
``statistic="cramer_von_mises"`` belongs to.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy.stats as jstats
import numpy as np

from ..core.types import Array, PRNGKey

Statistic = Literal["epps_pulley", "cramer_von_mises"]
Quadrature = Literal["gauss_hermite", "trapezoid"]


def random_directions(key: PRNGKey, dim: int, n_proj: int) -> Array:
    """``(dim, n_proj)`` directions drawn uniformly from the unit sphere."""
    v = jr.normal(key, (dim, n_proj))
    return v / (jnp.linalg.norm(v, axis=0, keepdims=True) + 1e-8)


@lru_cache(maxsize=16)
def _quadrature_np(n_nodes: int, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Hermite nodes/weights for ``int f(t) exp(-t^2 / (2 sigma^2)) dt``.

    ``hermgauss`` targets the weight ``exp(-x^2)``; substituting
    ``t = sigma * sqrt(2) * x`` gives the Gaussian weight we want. The weights
    are normalised to sum to one so the statistic's scale does not depend on
    ``n_nodes``.

    Cached as NumPy rather than JAX arrays on purpose: a cache holding arrays
    built inside one ``jit`` trace would hand those tracers to the next trace,
    which JAX rejects as a leak. NumPy values are trace-independent and become
    ordinary constants wherever they are used.
    """
    x, w = np.polynomial.hermite.hermgauss(n_nodes)
    return sigma * np.sqrt(2.0) * x, w / w.sum()


@lru_cache(maxsize=16)
def _trapezoid_np(n_nodes: int, sigma: float, t_max: float) -> tuple[np.ndarray, np.ndarray]:
    """Trapezoid nodes/weights on ``[0, t_max]`` for the same integral.

    The integrand is even in ``t``, so integrating the half-line and doubling
    the interior weights covers ``[-t_max, t_max]``. The weights are left
    **unnormalised** -- they sum to ``~sqrt(2 pi)`` -- because that is the
    convention the published SIGReg weights were tuned under; see
    :func:`epps_pulley`.
    """
    t = np.linspace(0.0, t_max, n_nodes)
    dt = t_max / (n_nodes - 1)
    w = np.full(n_nodes, 2.0 * dt)
    w[0] = w[-1] = dt
    return t, w * np.exp(-(t**2) / (2.0 * sigma**2))


def _quadrature(
    n_nodes: int,
    sigma: float,
    quadrature: Quadrature = "gauss_hermite",
    t_max: float = 3.0,
) -> tuple[Array, Array]:
    if quadrature == "gauss_hermite":
        t, w = _quadrature_np(n_nodes, sigma)
    elif quadrature == "trapezoid":
        t, w = _trapezoid_np(n_nodes, sigma, t_max)
    else:
        raise ValueError(f"unknown quadrature {quadrature!r}")
    return jnp.asarray(t, jnp.float32), jnp.asarray(w, jnp.float32)


def epps_pulley(
    u: Array,
    *,
    n_nodes: int = 32,
    sigma: float = 1.0,
    quadrature: Quadrature = "gauss_hermite",
    t_max: float = 3.0,
) -> Array:
    """Characteristic-function distance from ``u`` to ``N(0, 1)``.

    Args:
        u: ``(n, P)`` -- ``P`` independent 1-D samples of size ``n``.
        n_nodes: quadrature nodes; 32 is ample for so smooth an integrand.
            The LeJEPA-derived models call this the number of *knots* and use 17.
        sigma: width of the quadrature weight, i.e. which frequencies the test
            emphasises. Larger values probe finer structure in the tails.
        quadrature: ``"gauss_hermite"`` places nodes at ``sigma*sqrt(2)*x_i``
            with weights normalised to sum to one, so the statistic estimates
            ``E_{t ~ N(0, sigma^2)}[err(t)]`` and its scale does not depend on
            ``n_nodes``. ``"trapezoid"`` uses an evenly spaced grid on
            ``[0, t_max]`` with *unnormalised* weights, estimating the integral
            ``int err(t) exp(-t^2 / 2) dt`` itself. The two differ by a factor of
            ``sqrt(2 pi) ~ 2.5066``, which is why a published ``reg_weight`` only
            means what it says under the rule it was tuned with.
        t_max: upper limit of the trapezoid grid; ignored otherwise.

    Returns:
        ``(P,)`` non-negative statistics, zero iff the empirical characteristic
        function matches ``exp(-t^2 / 2)`` on the weighted grid.

    The quadrature is a :func:`jax.lax.scan` rather than one batched einsum:
    the dense form would allocate ``n * P * n_nodes`` floats, which at a real
    batch size (``n`` in the tens of thousands once tokens are counted) reaches
    hundreds of megabytes for a term that is only a scalar penalty. Scanning
    keeps the footprint at ``n * P``.
    """
    t, w = _quadrature(n_nodes, sigma, quadrature, t_max)

    def node(_, tw):
        t_k, w_k = tw
        phase = t_k * u  # (n, P)
        re = jnp.mean(jnp.cos(phase), axis=0) - jnp.exp(-0.5 * t_k**2)
        im = jnp.mean(jnp.sin(phase), axis=0)
        return _, w_k * (re**2 + im**2)

    _, per_node = jax.lax.scan(node, None, (t, w))  # (nodes, P)
    return jnp.sum(per_node, axis=0)


def cramer_von_mises(u: Array) -> Array:
    """CDF distance from ``u`` to ``N(0, 1)``.

    Args:
        u: ``(n, P)`` -- ``P`` independent 1-D samples of size ``n``.

    Returns:
        ``(P,)`` non-negative statistics (the ``omega^2`` form, so the scale is
        independent of ``n``).
    """
    n = u.shape[0]
    cdf = jstats.norm.cdf(jnp.sort(u, axis=0))
    ranks = (2.0 * jnp.arange(1, n + 1) - 1.0) / (2.0 * n)
    return jnp.sum((cdf - ranks[:, None]) ** 2, axis=0) / n + 1.0 / (12.0 * n**2)


def _sigreg_one(
    z: Array,
    key: PRNGKey,
    *,
    n_proj: int,
    statistic: Statistic,
    n_nodes: int,
    sigma: float,
    center: bool,
    quadrature: Quadrature,
    t_max: float,
    scale_by_n: bool,
) -> Array:
    """One pool of samples, one statistic. See :func:`sigreg`."""
    z = z.reshape(-1, z.shape[-1])
    if center:
        z = z - jnp.mean(z, axis=0, keepdims=True)
    u = z @ random_directions(key, z.shape[-1], n_proj)  # (n, n_proj)
    if statistic == "epps_pulley":
        stats = epps_pulley(u, n_nodes=n_nodes, sigma=sigma, quadrature=quadrature, t_max=t_max)
    elif statistic == "cramer_von_mises":
        stats = cramer_von_mises(u)
    else:
        raise ValueError(f"unknown statistic {statistic!r}")
    return jnp.mean(stats) * (u.shape[0] if scale_by_n else 1.0)


def sigreg(
    z: Array,
    key: PRNGKey,
    *,
    n_proj: int = 512,
    statistic: Statistic = "epps_pulley",
    n_nodes: int = 32,
    sigma: float = 1.0,
    center: bool = False,
    quadrature: Quadrature = "gauss_hermite",
    t_max: float = 3.0,
    scale_by_n: bool = False,
    axis: int | None = None,
) -> Array:
    """Sketched isotropic-Gaussian regularizer for a batch of embeddings.

    Args:
        z: ``(..., D)``. Every leading axis is treated as a sample, so token
            sequences ``(B, N, D)`` contribute ``B * N`` samples.
        key: RNG for the projection directions. Redraw it each step -- fresh
            directions are what make the sketch cover all of ``R^D`` over
            training rather than only a fixed subspace.
        n_proj: number of random directions.
        statistic: which goodness-of-fit test to use.
        n_nodes, sigma, quadrature, t_max: quadrature settings for
            ``"epps_pulley"``. The defaults are xwm's; the published
            LeJEPA-family weights assume ``n_nodes=17, quadrature="trapezoid",
            scale_by_n=True``.
        center: subtract the batch mean before testing. Off by default,
            because driving the mean to zero is part of the job.
        scale_by_n: multiply by the number of samples entering each empirical
            characteristic function -- the classic ``n * omega^2`` normalisation,
            under which the statistic has a fixed asymptotic distribution.
            Off by default because it makes the penalty's scale depend on the
            batch size; on when matching a published ``reg_weight``.
        axis: compute an independent statistic for each index along this axis
            and average, *sharing the projection directions*. ``axis=1`` on a
            ``(B, T, D)`` tensor asks "is each timestep's distribution
            isotropic?", which is what the LeJEPA-family world models do;
            ``None`` pools every leading axis, which is a stronger claim and
            conflates it with "the union over time is isotropic".

    Returns:
        A scalar, minimised when the embeddings look isotropic Gaussian.
    """
    settings = dict(
        n_proj=n_proj,
        statistic=statistic,
        n_nodes=n_nodes,
        sigma=sigma,
        center=center,
        quadrature=quadrature,
        t_max=t_max,
        scale_by_n=scale_by_n,
    )
    if axis is None:
        return _sigreg_one(z, key, **settings)
    # One statistic per slice, with the *same* key: the directions must be
    # shared, or the per-slice statistics are not comparable and their mean is
    # noisier than it needs to be.
    sliced = jnp.moveaxis(z, axis, 0)
    return jnp.mean(jax.vmap(lambda zi: _sigreg_one(zi, key, **settings))(sliced))

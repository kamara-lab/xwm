"""Linear and nearest-neighbour probes.

The standard way to ask what a frozen representation knows. Both probes here
are closed-form, so evaluating a checkpoint costs one matrix solve rather than
a training run -- which makes it cheap enough to run as a callback during
pretraining rather than only at the end.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ..core.types import Array


def ridge_probe(
    z_train: Array,
    y_train: Array,
    z_test: Array,
    y_test: Array,
    *,
    alpha: float = 1e-3,
) -> dict[str, Array]:
    """Closed-form ridge regression from embeddings to targets.

    Args:
        z_train: ``(N, D)`` embeddings.
        y_train: ``(N, K)`` regression targets.
        alpha: ridge penalty, on the scale of the (standardised) features.

    Returns:
        ``{"r2", "mse", "train_mse"}``. ``r2`` is computed against the *test*
        mean, so a probe that has learned nothing scores ~0 and a harmful one
        scores below 0.
    """
    z_train, z_test = jnp.atleast_2d(z_train), jnp.atleast_2d(z_test)
    y_train, y_test = jnp.atleast_2d(y_train), jnp.atleast_2d(y_test)
    mu = jnp.mean(z_train, axis=0, keepdims=True)
    sigma = jnp.std(z_train, axis=0, keepdims=True) + 1e-6
    a = jnp.concatenate([(z_train - mu) / sigma, jnp.ones((z_train.shape[0], 1))], axis=1)
    b = jnp.concatenate([(z_test - mu) / sigma, jnp.ones((z_test.shape[0], 1))], axis=1)
    d = a.shape[1]
    w = jnp.linalg.solve(a.T @ a + alpha * a.shape[0] * jnp.eye(d), a.T @ y_train)
    pred_test, pred_train = b @ w, a @ w
    mse = jnp.mean(jnp.square(pred_test - y_test))
    variance = jnp.mean(jnp.square(y_test - jnp.mean(y_test, axis=0, keepdims=True)))
    return {
        "r2": 1.0 - mse / (variance + 1e-12),
        "mse": mse,
        "train_mse": jnp.mean(jnp.square(pred_train - y_train)),
    }


def knn_probe(
    z_train: Array,
    y_train: Array,
    z_test: Array,
    y_test: Array,
    *,
    k: int = 5,
    classification: bool = False,
    n_classes: int | None = None,
) -> dict[str, Array]:
    """k-nearest-neighbour probe in cosine distance.

    Unlike :func:`ridge_probe` this reads *local* structure, so the two together
    distinguish "linearly decodable" from "merely clustered".

    Args:
        classification: treat ``y`` as integer labels and report accuracy;
            otherwise average the neighbours' values and report ``r2``.
        n_classes: required when ``classification`` is set.
    """
    a = z_train / (jnp.linalg.norm(z_train, axis=-1, keepdims=True) + 1e-8)
    b = z_test / (jnp.linalg.norm(z_test, axis=-1, keepdims=True) + 1e-8)
    neighbours = jnp.argsort(-(b @ a.T), axis=1)[:, :k]  # (N_test, k)

    if classification:
        if n_classes is None:
            raise ValueError("classification=True requires n_classes")
        labels = jnp.asarray(y_train, jnp.int32)[neighbours]  # (N_test, k)
        votes = jnp.sum(jax.nn.one_hot(labels, n_classes), axis=1)
        pred = jnp.argmax(votes, axis=-1)
        return {"accuracy": jnp.mean(pred == jnp.asarray(y_test, jnp.int32))}

    y_train = jnp.atleast_2d(y_train)
    y_test = jnp.atleast_2d(y_test)
    pred = jnp.mean(y_train[neighbours], axis=1)
    mse = jnp.mean(jnp.square(pred - y_test))
    variance = jnp.mean(jnp.square(y_test - jnp.mean(y_test, axis=0, keepdims=True)))
    return {"mse": mse, "r2": 1.0 - mse / (variance + 1e-12)}

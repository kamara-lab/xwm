import jax.random as jr
import pytest


@pytest.fixture
def key():
    return jr.PRNGKey(0)


@pytest.fixture(scope="session")
def world():
    import xwm

    return xwm.data.SpriteWorld(32, n_distractors=2)


#: Tiny architecture used across the model tests, so they stay fast.
TINY_ENCODER = {"depth": 2, "embed_dim": 64, "num_heads": 4}
TINY_PREDICTOR = {"depth": 2, "pred_dim": 32, "num_heads": 4}

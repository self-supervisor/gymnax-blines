import pytest
from utils.models import build_RND_models
import jax
import jax.numpy as jnp


@pytest.fixture
def RND():
    rng = jax.random.PRNGKey(0)
    rng_1, rng_2 = jax.random.split(rng)
    RND_model, RND_params, distiller_model, distiller_params = build_RND_models(
        obs_shape=(4,), rng_rnd=rng_1, rng_distiller=rng_2
    )
    return RND_model, RND_params, distiller_model, distiller_params


def test_build_RND_models(RND):
    RND_model, RND_params, distiller_model, distiller_params = RND

    key = jax.random.PRNGKey(4)
    obs = jax.random.normal(key, (1, 4))
    rnd_output = RND_model.apply(RND_params, obs)
    distiller_output = distiller_model.apply(distiller_params, obs)
    assert jnp.array_equal(rnd_output, distiller_output) == False


def test_compute_novelty(RND):
    from utils.ppo import compute_novelty

    return


def test_update_rnd(RND):
    from utils.ppo import update_rnd

    return


def test_log_value_predictions(RND):
    from utils.ppo import log_value_predictions

    return


def test_update_epoch(RND):
    from utils.ppo import update_epoch

    return


def test_update_RND(RND):
    from utils.ppo import update_RND

    return

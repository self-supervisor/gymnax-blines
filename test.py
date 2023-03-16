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

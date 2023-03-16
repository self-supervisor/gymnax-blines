import jax
import jax.numpy as jnp
import pytest
from flax.training.train_state import TrainState

from utils.helpers import load_config
from utils.models import CategoricalSeparateMLP, build_RND_models, get_model_ready
import gymnax


@pytest.fixture
def RND():
    rng = jax.random.PRNGKey(0)
    rng_1, rng_2 = jax.random.split(rng)
    RND_model, RND_params, distiller_model, distiller_params = build_RND_models(
        obs_shape=(4,), rng_rnd=rng_1, rng_distiller=rng_2
    )
    return RND_model, RND_params, distiller_model, distiller_params


@pytest.fixture
def config():
    config = load_config("agents/FourRooms-misc/ppo.yaml")
    return config


@pytest.fixture
def gymnax_env(config):
    env, env_params = gymnax.make(
        config.train_config.env_name, **config.train_config.env_kwargs
    )
    return env, env_params


@pytest.fixture
def PPO(config, gymnax_env, observation):

    env, env_params = gymnax_env
    rng = jax.random.PRNGKey(0)
    PPO_model = CategoricalSeparateMLP(
        **config.train_config.network_config,
        num_output_units=env.num_actions,
        scale=1,
        novelty_switch=int(1e4),
    )
    obs_shape = env.observation_space(env_params).shape
    counts_shape = env.get_counts().shape
    PPO_params = PPO_model.init(
        rng, jnp.zeros(obs_shape), jnp.zeros(counts_shape), jnp.array([0, 0]), rng=rng,
    )
    return PPO_model, PPO_params


@pytest.fixture
def PPO_train_state(config, PPO):
    from utils.ppo import get_optimiser

    PPO_model, PPO_params = PPO
    tx = get_optimiser(config)
    PPO_train_state = TrainState.create(
        apply_fn=PPO_model.apply, params=PPO_params, tc=tx
    )
    return PPO_train_state


@pytest.fixture
def observation():
    key = jax.random.PRNGKey(4)
    return jax.random.normal(key, (1, 4))


@pytest.fixture
def rollout_manager(gymnax_env, config, PPO):
    from utils.ppo import RolloutManager

    PPO_model, _ = PPO
    rollout_manager = RolloutManager(
        PPO_model,
        config.train_config.env_name,
        config.train_config.env_kwargs,
        config.train_config.env_params,
    )
    return rollout_manager


def test_build_RND_models(RND, observation):
    RND_model, RND_params, distiller_model, distiller_params = RND

    key = jax.random.PRNGKey(4)
    obs = jax.random.normal(key, (1, 4))
    rnd_output = RND_model.apply(RND_params, obs)
    distiller_output = distiller_model.apply(distiller_params, obs)
    assert jnp.array_equal(rnd_output, distiller_output) == False


def test_compute_novelty(RND, observation):
    from utils.ppo import compute_novelty

    RND_model, RND_params, distiller_model, distiller_params = RND
    novelty = compute_novelty(
        observation, RND_model, RND_params, distiller_model, distiller_params
    )
    assert jnp.allclose(novelty, jnp.zeros_like(novelty)) == False


def test_log_value_predictions(RND, PPO, rollout_manager):
    from utils.ppo import log_value_predictions, compute_novelty

    PPO_model, PPO_params = PPO
    RND_model, RND_params, distiller_model, distiller_params = RND
    novelty_signal = compute_novelty(
        observation, RND_model, RND_params, distiller_model, distiller_params
    )
    rng = jax.random.PRNGKey(0)
    log_value_predictions(
        use_wandb=False,
        model=PPO_model,
        train_state=PPO_train_state,
        rollout_manage=rollout_manager,
        rng=rng,
        novelty_signal=novelty_signal,
    )


def test_update_epoch(RND):
    from utils.ppo import update_epoch

    pass


def test_update_RND(RND):
    from utils.ppo import update_RND

    pass

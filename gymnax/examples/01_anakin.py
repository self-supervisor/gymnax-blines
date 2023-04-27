import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1"
import jax
from jax import random
from jax import numpy as jnp
from utils import TimeIt, make_gif, TimeStep

print("devices", jax.devices())
import gymnax
from flax.serialization import to_state_dict, from_state_dict
from gymnax.environments.minatar.space_invaders import EnvState
from learner import get_learner_fn
from models import get_network_fn, get_transition_model_fn
import optax

env, env_params = gymnax.make("SpaceInvaders-MinAtar")
print("devices", jax.devices())


def run_experiment(env, batch_size, rollout_len, step_size, iterations, seed):
    """Runs experiment."""
    cores_count = len(jax.devices())  # get available TPU cores.
    network = get_network_fn(env.num_actions)  # define network.
    forward_transition_model = get_transition_model_fn(600)
    backward_transition_model = get_transition_model_fn(600)
    optim_network = optax.adam(step_size)  # define optimiser.
    optim_forward_transition_model = optax.adam(step_size)
    optim_backward_transition_model = optax.adam(step_size)

    rng, rng_e, rng_p, rng_t = random.split(random.PRNGKey(seed), num=4)  # prng keys.
    obs, state = env.reset(rng_e)
    dummy_obs = obs[
        None,
    ]  # dummy for net init.
    dummy_obs_and_action = jnp.concatenate(
        (obs.reshape(-1), jnp.array([0]).reshape(-1))
    )  # dummy for net init.
    network_params = network.init(rng_p, dummy_obs, None)  # initialise params.
    forward_transition_model_params = forward_transition_model.init(
        rng_p, dummy_obs_and_action, None
    )
    backward_transition_model_params = backward_transition_model.init(
        rng_p, dummy_obs_and_action, None
    )

    opt_state_network = optim_network.init(
        network_params
    )  # initialise optimiser stats.
    opt_state_forward_transition_model = optim_forward_transition_model.init(
        forward_transition_model_params
    )
    opt_state_backward_transition_model = optim_backward_transition_model.init(
        backward_transition_model_params
    )

    learn = get_learner_fn(  # get batched iterated update.
        env,
        network.apply,
        forward_transition_model.apply,
        backward_transition_model.apply,
        optim_network.update,
        optim_forward_transition_model.update,
        optim_backward_transition_model.update,
        rollout_len=rollout_len,
        agent_discount=1,
        lambda_=0.99,
        iterations=iterations,
    )
    learn = jax.pmap(learn, axis_name="i")  # replicate over multiple cores.

    broadcast = lambda x: jnp.broadcast_to(x, (cores_count, batch_size) + x.shape)
    network_params = jax.tree_map(
        broadcast, network_params
    )  # broadcast to cores and batch.
    forward_transition_model_params = jax.tree_map(
        broadcast, forward_transition_model_params
    )  # broadcast to cores and batch.
    backward_transition_model_params = jax.tree_map(
        broadcast, backward_transition_model_params
    )  # broadcast to cores and batch.
    opt_state_network = jax.tree_map(
        broadcast, opt_state_network
    )  # broadcast to cores and batch
    opt_state_forward_transition_model = jax.tree_map(
        broadcast, opt_state_forward_transition_model
    )  # broadcast to cores and batch
    opt_state_backward_transition_model = jax.tree_map(
        broadcast, opt_state_backward_transition_model
    )  # broadcast to cores and batch

    rng, *env_rngs = jax.random.split(rng, cores_count * batch_size + 1)
    env_obs, env_states = jax.vmap(env.reset)(jnp.stack(env_rngs))  # init envs.
    rng, *step_rngs = jax.random.split(rng, cores_count * batch_size + 1)

    reshape = lambda x: x.reshape((cores_count, batch_size) + x.shape[1:])
    step_rngs = reshape(jnp.stack(step_rngs))  # add dimension to pmap over.
    env_obs = reshape(env_obs)  # add dimension to pmap over.
    env_states_re = to_state_dict(env_states)
    env_states = {k: reshape(env_states_re[k]) for k in env_states_re.keys()}
    env_states = EnvState(**env_states)
    with TimeIt(tag="COMPILATION"):
        learn(
            network_params,
            opt_state_network,
            forward_transition_model_params,
            opt_state_forward_transition_model,
            backward_transition_model_params,
            opt_state_backward_transition_model,
            step_rngs,
            env_states,
        )  # compiles

    num_frames = cores_count * iterations * rollout_len * batch_size
    with TimeIt(tag="EXECUTION", frames=num_frames):
        (
            network_params,
            opt_state_network,
            forward_transition_model_params,
            opt_state_forward_transition_model,
            backward_transition_model_params,
            opt_state_backward_transition_model,
            step_rngs,
            env_states,
        ) = learn(  # runs compiled fn
            network_params,
            opt_state_network,
            forward_transition_model_params,
            opt_state_forward_transition_model,
            backward_transition_model_params,
            opt_state_backward_transition_model,
            step_rngs,
            env_states,
        )
    return (
        network_params,
        forward_transition_model_params,
        backward_transition_model_params,
    )


def main():
    print("Running on", len(jax.devices()), "cores.", flush=True)
    batch_params = run_experiment(env, 128, 16, 3e-4, 10000, 42)
    # Get model ready for evaluation - squeeze broadcasted params
    network = get_network_fn(env.num_actions)
    forward_transition_model = get_transition_model_fn(600)
    squeeze = lambda x: x[0][0]
    (
        network_params,
        forward_transition_model_params,
        backward_transition_model_params,
    ) = jax.tree_map(squeeze, batch_params)

    rng = jax.random.PRNGKey(0)

    from tqdm import tqdm

    obs, state = env.reset(rng)
    cum_ret = 0

    ground_truth_obs = []
    pred_obs = []
    for step in tqdm(range(env_params.max_steps_in_episode)):
        rng, key_step = jax.random.split(rng)
        q_values = network.apply(network_params, obs[None,], None)
        action = jnp.argmax(q_values)
        n_obs, n_state, reward, done, _ = env.step(key_step, state, action, env_params)
        pred_obs.append(
            forward_transition_model.apply(
                forward_transition_model_params,
                jnp.concatenate((obs.reshape(-1), jnp.array([action]).reshape(-1)))[
                    None,
                ],
                None,
            )
        )
        ground_truth_obs.append(n_obs)
        cum_ret += reward

        if done:
            break
        else:
            state = n_state
            obs = n_obs

    import numpy as np

    np.save("ground_truth_obs.npy", np.array(ground_truth_obs))
    np.save(
        "pred_obs.npy", np.array(pred_obs).reshape(len(ground_truth_obs), 10, 10, -1)
    )

    make_gif()


if __name__ == "__main__":
    main()

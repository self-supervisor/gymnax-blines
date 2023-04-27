import chex
import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1"
import jax
import haiku as hk
from jax import lax
from jax import random
from jax import numpy as jnp
import optax
import rlax
import timeit

print("devices", jax.devices())
import gymnax
from flax.serialization import to_state_dict, from_state_dict
from gymnax.environments.minatar.space_invaders import EnvState

env, env_params = gymnax.make("SpaceInvaders-MinAtar")
print("devices", jax.devices())


@chex.dataclass(frozen=True)
class TimeStep:
    q_values: chex.Array
    action: chex.Array
    discount: chex.Array
    reward: chex.Array
    pred_next_obs: chex.Array
    actual_obs: chex.Array


def get_network_fn(num_outputs: int):
    """Define a fully connected multi-layer haiku network."""

    def network_fn(obs: chex.Array, rng: chex.PRNGKey) -> chex.Array:
        return hk.Sequential(
            [  # flatten, 2x hidden + relu, output layer.
                hk.Flatten(),
                hk.Linear(256),
                jax.nn.relu,
                hk.Linear(256),
                jax.nn.relu,
                hk.Linear(num_outputs),
            ]
        )(obs)

    return hk.without_apply_rng(hk.transform(network_fn))


def get_transition_model_fn(num_outputs: int):
    def transition_model_fn(
        obs_and_action: chex.Array, rng: chex.PRNGKey
    ) -> chex.Array:
        return hk.Sequential(
            [  # flatten, 2x hidden + relu, output layer.
                hk.Flatten(),
                hk.Linear(256),
                jax.nn.relu,
                hk.Linear(256),
                jax.nn.relu,
                hk.Linear(num_outputs),
            ]
        )(obs_and_action)

    return hk.without_apply_rng(hk.transform(transition_model_fn))


def get_learner_fn(
    env,
    forward_pass_q_network,
    forward_pass_transition_model,
    opt_update_network,
    opt_update_transition_model,
    rollout_len,
    agent_discount,
    lambda_,
    iterations,
):
    """Define the minimal unit of computation in Anakin."""

    def loss_fn(
        network_params,
        transition_model_params,
        outer_rng,
        env_state,
        q_learning_weight,
        transition_model_weight,
    ):
        """Compute the loss on a single trajectory."""

        def step_fn(env_state, rng):
            obs = env.get_obs(env_state)
            q_values = forward_pass_q_network(network_params, obs[None,], None)[
                0
            ]  # forward pass.
            action = jnp.argmax(q_values)  # greedy policy.
            obs_and_action = jnp.concatenate((obs.reshape(-1), action.reshape(-1)))
            pred_next_obs = forward_pass_transition_model(
                transition_model_params, obs_and_action[None,], None
            )[0]
            obs, env_state, reward, terminal, info = env.step(
                rng, env_state, action
            )  # step environment.
            return (
                env_state,
                TimeStep(  # return env state and transition data.
                    q_values=q_values,
                    action=action,
                    discount=1.0 - terminal,
                    reward=reward,
                    pred_next_obs=pred_next_obs,
                    actual_obs=obs,
                ),
            )

        step_rngs = random.split(outer_rng, rollout_len)
        env_state, rollout = lax.scan(step_fn, env_state, step_rngs)  # trajectory.
        qa_tm1 = rlax.batched_index(rollout.q_values[:-1], rollout.action[:-1])
        td_error = rlax.td_lambda(  # compute multi-step temporal diff error.
            v_tm1=qa_tm1,  # predictions.
            r_t=rollout.reward[1:],  # rewards.
            discount_t=agent_discount * rollout.discount[1:],  # discount.
            v_t=jnp.max(rollout.q_values[1:], axis=-1),  # bootstrap values.
            lambda_=lambda_,
        )  # mixing hyper-parameter lambda.
        q_learning_loss = jnp.mean(td_error ** 2)  # mean squared error.
        transition_model_loss = jnp.mean(
            jnp.sum(
                (
                    rollout.pred_next_obs
                    - rollout.actual_obs.reshape(rollout.actual_obs.shape[0], -1)
                )
                ** 2,
                axis=-1,
            )
        )
        total_loss = (
            q_learning_weight * q_learning_loss
            + transition_model_weight * transition_model_loss
        )
        return total_loss, env_state

    def update_fn(
        network_params,
        network_opt_state,
        transition_model_params,
        transition_model_opt_state,
        rng,
        env_state,
    ):
        """Compute a gradient update from a single trajectory."""
        rng, loss_rng = random.split(rng)
        grads, new_env_state = jax.grad(  # compute gradient on a single trajectory.
            loss_fn, argnums=0, has_aux=True
        )(
            network_params,
            transition_model_params,
            loss_rng,
            env_state,
            q_learning_weight=1.0,
            transition_model_weight=0.0,
        )
        grads = lax.pmean(grads, axis_name="j")  # reduce mean across cores.
        grads = lax.pmean(grads, axis_name="i")  # reduce mean across batch.
        network_updates, new_network_opt_state = opt_update_network(
            grads, network_opt_state
        )  # transform grads.
        grads, new_env_state = jax.grad(  # compute gradient on a single trajectory.
            loss_fn, argnums=1, has_aux=True
        )(
            network_params,
            transition_model_params,
            loss_rng,
            env_state,
            q_learning_weight=0.0,
            transition_model_weight=1.0,
        )
        grads = lax.pmean(grads, axis_name="j")  # reduce mean across cores.
        grads = lax.pmean(grads, axis_name="i")  # reduce mean across batch.
        (
            transition_model_updates,
            new_transition_model_opt_state,
        ) = opt_update_transition_model(
            grads, transition_model_opt_state
        )  # transform grads.
        # ensures that the networks are not updated until they have both computed
        # equivalent rollouts
        new_network_params = optax.apply_updates(
            network_params, network_updates
        )  # update parameters.
        new_transition_model_params = optax.apply_updates(
            transition_model_params, transition_model_updates
        )  # update parameters.
        return (
            new_network_params,
            new_network_opt_state,
            new_transition_model_params,
            new_transition_model_opt_state,
            rng,
            new_env_state,
        )

    def learner_fn(
        network_params,
        network_opt_state,
        transition_model_params,
        transition_model_opt_state,
        rngs,
        env_states,
    ):
        """Vectorise and repeat the update."""
        batched_update_fn = jax.vmap(
            update_fn, axis_name="j"
        )  # vectorize across batch.

        def iterate_fn(_, val):  # repeat many times to avoid going back to Python.
            (
                network_params,
                network_opt_state,
                transition_model_params,
                transition_model_opt_state,
                rngs,
                env_states,
            ) = val
            return batched_update_fn(
                network_params,
                network_opt_state,
                transition_model_params,
                transition_model_opt_state,
                rngs,
                env_states,
            )

        return lax.fori_loop(
            0,
            iterations,
            iterate_fn,
            (
                network_params,
                network_opt_state,
                transition_model_params,
                transition_model_opt_state,
                rngs,
                env_states,
            ),
        )

    return learner_fn


class TimeIt:
    def __init__(self, tag, frames=None):
        self.tag = tag
        self.frames = frames

    def __enter__(self):
        self.start = timeit.default_timer()
        return self

    def __exit__(self, *args):
        self.elapsed_secs = timeit.default_timer() - self.start
        msg = self.tag + (": Elapsed time=%.2fs" % self.elapsed_secs)
        if self.frames:
            msg += ", FPS=%.2e" % (self.frames / self.elapsed_secs)
        print(msg)


def run_experiment(env, batch_size, rollout_len, step_size, iterations, seed):
    """Runs experiment."""
    cores_count = len(jax.devices())  # get available TPU cores.
    network = get_network_fn(env.num_actions)  # define network.
    transition_model = get_transition_model_fn(600)
    optim_network = optax.adam(step_size)  # define optimiser.
    optim_transition_model = optax.adam(step_size)

    rng, rng_e, rng_p, rng_t = random.split(random.PRNGKey(seed), num=4)  # prng keys.
    obs, state = env.reset(rng_e)
    dummy_obs = obs[
        None,
    ]  # dummy for net init.
    dummy_obs_and_action = jnp.concatenate(
        (obs.reshape(-1), jnp.array([0]).reshape(-1))
    )  # dummy for net init.
    network_params = network.init(rng_p, dummy_obs, None)  # initialise params.
    transition_model_params = transition_model.init(rng_p, dummy_obs_and_action, None)

    opt_state_network = optim_network.init(
        network_params
    )  # initialise optimiser stats.
    opt_state_transition_model = optim_transition_model.init(transition_model_params)

    learn = get_learner_fn(  # get batched iterated update.
        env,
        network.apply,
        transition_model.apply,
        optim_network.update,
        optim_transition_model.update,
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
    transition_model_params = jax.tree_map(
        broadcast, transition_model_params
    )  # broadcast to cores and batch.
    opt_state_network = jax.tree_map(
        broadcast, opt_state_network
    )  # broadcast to cores and batch
    opt_state_transition_model = jax.tree_map(
        broadcast, opt_state_transition_model
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
            transition_model_params,
            opt_state_transition_model,
            step_rngs,
            env_states,
        )  # compiles

    num_frames = cores_count * iterations * rollout_len * batch_size
    with TimeIt(tag="EXECUTION", frames=num_frames):
        (
            network_params,
            opt_state_network,
            transition_model_params,
            opt_state_transition_model,
            step_rngs,
            env_states,
        ) = learn(  # runs compiled fn
            network_params,
            opt_state_network,
            transition_model_params,
            opt_state_transition_model,
            step_rngs,
            env_states,
        )
    return network_params, transition_model_params


print("Running on", len(jax.devices()), "cores.", flush=True)
batch_params = run_experiment(env, 128, 16, 3e-4, 10000, 42)
# Get model ready for evaluation - squeeze broadcasted params
network = get_network_fn(env.num_actions)
transition_model = get_transition_model_fn(600)
squeeze = lambda x: x[0][0]
network_params, transition_model_params = jax.tree_map(squeeze, batch_params)

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
        transition_model.apply(
            transition_model_params,
            jnp.concatenate((obs.reshape(-1), jnp.array([action]).reshape(-1)))[None,],
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
np.save("pred_obs.npy", np.array(pred_obs).reshape(len(ground_truth_obs), 10, 10, -1))
print(cum_ret)

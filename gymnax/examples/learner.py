import jax.numpy as jnp
from utils import TimeStep
import optax
from jax import lax, random
import rlax
import jax


def get_learner_fn(
    env,
    forward_pass_q_network,
    forward_pass_forward_transition_model,
    forward_pass_backward_transition_model,
    opt_update_network,
    opt_update_forward_transition_model,
    opt_update_backward_transition_model,
    rollout_len,
    agent_discount,
    lambda_,
    iterations,
):
    """Define the minimal unit of computation in Anakin."""

    def loss_fn(
        network_params,
        forward_transition_model_params,
        backward_transition_model_params,
        outer_rng,
        env_state,
        q_learning_weight,
        forward_transition_model_weight,
        backward_transition_model_weight,
    ):
        """Compute the loss on a single trajectory."""

        def step_fn(env_state, rng):
            obs = env.get_obs(env_state)
            q_values = forward_pass_q_network(network_params, obs[None,], None)[
                0
            ]  # forward pass.
            prev_obs = obs.copy()
            # copy a observation
            action = jnp.argmax(q_values)  # greedy policy.
            obs_and_action = jnp.concatenate((obs.reshape(-1), action.reshape(-1)))
            pred_next_obs = forward_pass_forward_transition_model(
                forward_transition_model_params, obs_and_action[None,], None
            )[0]
            obs, env_state, reward, terminal, info = env.step(
                rng, env_state, action
            )  # step environment.
            next_obs_and_action = jnp.concatenate((obs.reshape(-1), action.reshape(-1)))
            pred_prev_obs = forward_pass_backward_transition_model(
                backward_transition_model_params, next_obs_and_action[None,], None
            )[0]
            return (
                env_state,
                TimeStep(  # return env state and transition data.
                    q_values=q_values,
                    action=action,
                    discount=1.0 - terminal,
                    reward=reward,
                    pred_next_obs=pred_next_obs,
                    pred_prev_obs=pred_prev_obs,
                    actual_obs=obs,
                    prev_obs=prev_obs,
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
        forward_transition_model_loss = jnp.mean(
            jnp.sum(
                (
                    rollout.pred_next_obs
                    - rollout.actual_obs.reshape(rollout.actual_obs.shape[0], -1)
                )
                ** 2,
                axis=-1,
            )
        )
        backward_transition_model_loss = jnp.mean(
            jnp.sum(
                (
                    rollout.pred_prev_obs
                    - rollout.prev_obs.reshape(rollout.actual_obs.shape[0], -1)
                )
                ** 2,
                axis=-1,
            )
        )
        total_loss = (
            q_learning_weight * q_learning_loss
            + forward_transition_model_weight * forward_transition_model_loss
            + backward_transition_model_weight * backward_transition_model_loss
        )
        return total_loss, env_state

    def update_fn(
        network_params,
        network_opt_state,
        forward_transition_model_params,
        forward_transition_model_opt_state,
        backward_transition_model_params,
        backward_transition_model_opt_state,
        rng,
        env_state,
    ):
        """Compute a gradient update from a single trajectory."""
        rng, loss_rng = random.split(rng)
        grads, new_env_state = jax.grad(  # compute gradient on a single trajectory.
            loss_fn, argnums=0, has_aux=True
        )(
            network_params,
            forward_transition_model_params,
            backward_transition_model_params,
            loss_rng,
            env_state,
            q_learning_weight=1.0,
            forward_transition_model_weight=0.0,
            backward_transition_model_weight=0.0,
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
            forward_transition_model_params,
            backward_transition_model_params,
            loss_rng,
            env_state,
            q_learning_weight=0.0,
            forward_transition_model_weight=1.0,
            backward_transition_model_weight=0.0,
        )
        grads = lax.pmean(grads, axis_name="j")  # reduce mean across cores.
        grads = lax.pmean(grads, axis_name="i")  # reduce mean across batch.
        (
            forward_transition_model_updates,
            new_forward_transition_model_opt_state,
        ) = opt_update_forward_transition_model(
            grads, forward_transition_model_opt_state
        )  # transform grads.
        grads, new_env_state = jax.grad(  # compute gradient on a single trajectory.
            loss_fn, argnums=2, has_aux=True
        )(
            network_params,
            forward_transition_model_params,
            backward_transition_model_params,
            loss_rng,
            env_state,
            q_learning_weight=0.0,
            forward_transition_model_weight=0.0,
            backward_transition_model_weight=1.0,
        )
        grads = lax.pmean(grads, axis_name="j")  # reduce mean across cores.
        grads = lax.pmean(grads, axis_name="i")  # reduce mean across batch.
        (
            backward_transition_model_updates,
            new_backward_transition_model_opt_state,
        ) = opt_update_backward_transition_model(
            grads, backward_transition_model_opt_state
        )  # transform grads.
        # ensures that the networks are not updated until they have both computed
        # equivalent rollouts
        new_network_params = optax.apply_updates(
            network_params, network_updates
        )  # update parameters.
        new_forward_transition_model_params = optax.apply_updates(
            forward_transition_model_params, forward_transition_model_updates
        )  # update parameters.
        new_backward_transition_model_params = optax.apply_updates(
            backward_transition_model_params, backward_transition_model_updates
        )  # update parameters.
        return (
            new_network_params,
            new_network_opt_state,
            new_forward_transition_model_params,
            new_forward_transition_model_opt_state,
            new_backward_transition_model_params,
            new_backward_transition_model_opt_state,
            rng,
            new_env_state,
        )

    def learner_fn(
        network_params,
        network_opt_state,
        forward_transition_model_params,
        forward_transition_model_opt_state,
        backward_transition_model_params,
        backward_transition_model_opt_state,
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
                forward_transition_model_params,
                forward_transition_model_opt_state,
                backward_transition_model_params,
                backward_transition_model_opt_state,
                rngs,
                env_states,
            ) = val
            return batched_update_fn(
                network_params,
                network_opt_state,
                forward_transition_model_params,
                forward_transition_model_opt_state,
                backward_transition_model_params,
                backward_transition_model_opt_state,
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
                forward_transition_model_params,
                forward_transition_model_opt_state,
                backward_transition_model_params,
                backward_transition_model_opt_state,
                rngs,
                env_states,
            ),
        )

    return learner_fn

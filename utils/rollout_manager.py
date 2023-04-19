from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import gymnax
import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
import flax

# STR_TO_JAX_ARR = {
#     "high": jnp.array([1, 0, 0]),
#     "low": jnp.array([0, 1, 0]),
#     "mixed": jnp.array([0, 0, 1]),
# }


# @partial(jax.jit, static_argnums=0)
# def policy(
#     apply_fn: Callable[..., Any],
#     params: flax.core.frozen_dict.FrozenDict,
#     obs: jnp.ndarray,
#     counts: jnp.ndarray,
#     low_mixed_or_high: jnp.ndarray,
#     rng,
# ):
#     value, pi = apply_fn(params, obs, counts, low_mixed_or_high, rng)
#     return value, pi
@partial(jax.jit, static_argnums=0)
def policy(
    apply_fn: Callable[..., Any],
    params: flax.core.frozen_dict.FrozenDict,
    obs: jnp.ndarray,
    rng,
):
    value, pi = apply_fn(params, obs, rng)
    return value, pi


class RolloutManager(object):
    def __init__(self, model, env_name, env_kwargs, env_params):
        # Setup functionalities for vectorized batch rollout
        self.env_name = env_name
        self.env, self.env_params = gymnax.make(env_name, **env_kwargs)
        self.env_params = self.env_params.replace(**env_params)
        self.observation_space = self.env.observation_space(self.env_params)
        self.action_size = self.env.action_space(self.env_params).shape
        self.apply_fn = model.apply
        self.select_action = self.select_action_ppo

    @partial(jax.jit, static_argnums=0)
    def select_action_ppo(
        self, train_state: TrainState, obs: jnp.ndarray, rng: jax.random.PRNGKey,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jax.random.PRNGKey]:
        value, pi = policy(train_state.apply_fn, train_state.params, obs, rng)
        action = pi.sample(seed=rng)
        log_prob = pi.log_prob(action)
        return action, log_prob, value[:, 0], rng

    @partial(jax.jit, static_argnums=0)
    def batch_reset(self, keys):
        return jax.vmap(self.env.reset, in_axes=(0, None))(keys, self.env_params)

    @partial(jax.jit, static_argnums=0)
    def batch_step(self, keys, state, action):
        return jax.vmap(self.env.step, in_axes=(0, 0, 0, None))(
            keys, state, action, self.env_params
        )

    @partial(jax.jit, static_argnums=(0, 3))
    def batch_evaluate(self, rng_input, train_state, num_envs):
        """Rollout an episode with lax.scan."""
        # Reset the environment
        rng_reset, rng_episode = jax.random.split(rng_input)
        obs, state = self.batch_reset(jax.random.split(rng_reset, num_envs))

        def policy_step(state_input, _):
            """lax.scan compatible step transition in jax env."""
            obs, state, train_state, rng, cum_reward, valid_mask = state_input
            rng, rng_step, rng_net = jax.random.split(rng, 3)
            action, _, _, rng = self.select_action(train_state, obs, rng_net)
            next_o, next_s, reward, done, _ = self.batch_step(
                jax.random.split(rng_step, num_envs), state, action.squeeze(),
            )
            new_cum_reward = cum_reward + reward * valid_mask
            new_valid_mask = valid_mask * (1 - done)
            carry, y = (
                [next_o, next_s, train_state, rng, new_cum_reward, new_valid_mask,],
                [new_valid_mask],
            )
            return carry, y

        # Scan over episode step loop
        carry_out, scan_out = jax.lax.scan(
            policy_step,
            [
                obs,
                state,
                train_state,
                rng_episode,
                jnp.array(num_envs * [0.0]),
                jnp.array(num_envs * [1.0]),
            ],
            (),
            self.env_params.max_steps_in_episode,
        )

        cum_return = carry_out[-2].squeeze()
        return jnp.mean(cum_return)

    # @partial(jax.jit, static_argnums=(0, 3))
    # def batch_evaluate(self, rng_input, train_state, num_envs):
    #     """Rollout an episode with lax.scan."""
    #     # Reset the environment
    #     rng_reset, rng_episode = jax.random.split(rng_input)
    #     obs, state = self.batch_reset(jax.random.split(rng_reset, num_envs))

    #     def policy_step(state_input, _):
    #         """lax.scan compatible step transition in jax env."""
    #         obs, state, train_state, rng, cum_reward, valid_mask = state_input
    #         rng, rng_step, rng_net = jax.random.split(rng, 3)
    #         action, _, _, rng = self.select_action(train_state, obs, rng_net)
    #         next_o, next_s, reward, done, _ = self.batch_step(
    #             jax.random.split(rng_step, num_envs), state, action.squeeze(),
    #         )
    #         new_cum_reward = cum_reward + reward * valid_mask
    #         new_valid_mask = valid_mask * (1 - done)
    #         carry, y = (
    #             [next_o, next_s, train_state, rng, new_cum_reward, new_valid_mask,],
    #             [new_valid_mask],
    #         )
    #         return carry, y

    #     # Scan over episode step loop
    #     carry_out, scan_out = jax.lax.scan(
    #         policy_step,
    #         [
    #             obs,
    #             state,
    #             train_state,
    #             rng_episode,
    #             jnp.array(num_envs * [0.0]),
    #             jnp.array(num_envs * [1.0]),
    #         ],
    #         (),
    #         self.env_params.max_steps_in_episode,
    #     )

    #     cum_return = carry_out[-2].squeeze()
    #     return jnp.mean(cum_return)


# class RolloutManager(object):
#     def __init__(
#         self, model, env_name, env_kwargs, env_params, map_params, map_all_locations
#     ):
#         # Setup functionalities for vectorized batch rollout
#         self.env_name = env_name
#         self.env, self.env_params = gymnax.make(
#             env_name,
#             # train_map=map_params,
#             # test_map=map_params,
#             # map_all_locations=map_all_locations,
#             **env_kwargs
#         )
#         self.env_params = self.env_params.replace(**env_params)
#         self.observation_space = self.env.observation_space(self.env_params)
#         self.action_size = self.env.action_space(self.env_params).shape
#         self.apply_fn = model.apply
#         self.select_action = self.select_action_ppo

#     @partial(jax.jit, static_argnums=0)
#     def select_action_ppo(
#         self,
#         train_state: TrainState,
#         obs: jnp.ndarray,
#         counts: jnp.ndarray,
#         low_mixed_or_high: jnp.ndarray,
#         rng: jax.random.PRNGKey,
#     ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jax.random.PRNGKey]:
#         value, pi = policy(
#             train_state.apply_fn,
#             train_state.params,
#             obs,
#             counts,
#             low_mixed_or_high,
#             rng,
#         )
#         action = pi.sample(seed=rng)
#         log_prob = pi.log_prob(action)
#         return action, log_prob, value[:, 0], rng

#     @partial(jax.jit, static_argnums=0)
#     def batch_reset(self, keys, training):
#         return jax.vmap(self.env.reset, in_axes=(0, 0, None))(
#             keys, jnp.array([training] * keys.shape[0]), self.env_params
#         )

#     @partial(jax.jit, static_argnums=0)
#     def batch_step(self, keys, state, action):
#         return jax.vmap(self.env.step, in_axes=(0, 0, 0, None))(
#             keys, state, action, self.env_params
#         )

#     @partial(jax.jit, static_argnums=(0, 3))
#     def batch_evaluate(self, rng_input, train_state, num_envs, counts, training):
#         """Rollout an episode with lax.scan."""
#         # Reset the environment
#         rng_reset, rng_episode = jax.random.split(rng_input)
#         obs, state = self.batch_reset(jax.random.split(rng_reset, num_envs), training)

#         def policy_step(state_input, _):
#             """lax.scan compatible step transition in jax env."""
#             obs, state, train_state, rng, cum_reward, valid_mask, td_error = state_input
#             rng, rng_step, rng_net = jax.random.split(rng, 3)
#             action, _, _, rng = self.select_action(
#                 train_state, obs, counts, STR_TO_JAX_ARR["mixed"], rng_net
#             )
#             next_o, next_s, reward, done, _ = self.batch_step(
#                 jax.random.split(rng_step, num_envs), state, action.squeeze(),
#             )
#             next_o_val, _ = policy(
#                 train_state.apply_fn,
#                 train_state.params,
#                 obs,
#                 counts,
#                 STR_TO_JAX_ARR["mixed"],
#                 rng_net,
#             )
#             o_val, _ = policy(
#                 train_state.apply_fn,
#                 train_state.params,
#                 next_o,
#                 counts,
#                 STR_TO_JAX_ARR["mixed"],
#                 rng_net,
#             )

#             td_error = (
#                 (o_val.squeeze() - (reward + 0.99 * next_o_val.squeeze())) ** 2
#             ).mean()
#             new_cum_reward = cum_reward + reward * valid_mask
#             new_valid_mask = valid_mask * (1 - done)
#             carry, y = (
#                 [
#                     next_o,
#                     next_s,
#                     train_state,
#                     rng,
#                     new_cum_reward,
#                     new_valid_mask,
#                     td_error,
#                 ],
#                 [new_valid_mask],
#             )

#             return carry, y

#         # Scan over episode step loop
#         carry_out, scan_out = jax.lax.scan(
#             policy_step,
#             [
#                 obs,
#                 state,
#                 train_state,
#                 rng_episode,
#                 jnp.array(num_envs * [0.0]),
#                 jnp.array(num_envs * [1.0]),
#                 jnp.array(0.0),
#             ],
#             (),
#             self.env_params.max_steps_in_episode,
#         )

#         cum_return = carry_out[-3].squeeze()
#         # mean_novelty = counts[obs[:, :2][:, 0], obs[:, :2][:, 1]].mean()
#         return jnp.mean(cum_return), carry_out[-1]  # , mean_novelty

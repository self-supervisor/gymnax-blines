from collections import defaultdict
from functools import partial
from typing import Any, Callable, Tuple
import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tqdm
from flax.training.train_state import TrainState

from .batch_manager import BatchManager
from .logging import (
    log_counts,
    log_error_vs_counts,
    log_value_predictions,
    log_frequency_stats,
    compute_difference_with_perfect_policy,
    log_histograms,
)
from .rollout_manager import RolloutManager

STR_TO_JAX_ARR = {
    "high": jnp.array([1, 0, 0]),
    "low": jnp.array([0, 1, 0]),
    "mixed": jnp.array([0, 0, 1]),
}


HIGH_FREQ_ERRORS = []
LOW_FREQ_ERRORS = []
HIGH_FREQ_COUNTS = []
LOW_FREQ_COUNTS = []


def update_counts(counts, obs):
    obs = np.array(obs)
    for i in range(obs.shape[0]):
        counts[obs[i][0], obs[i][1]] += 1
    return counts


def update_values(values, obs, new_values):
    obs = np.array(obs)
    for i in range(obs.shape[0]):
        values[obs[i][0], obs[i][1]] = new_values[i]
    return values


def train_ppo(
    rng,
    config,
    model,
    params,
    mle_log,
    use_wandb,
    perfect_network_params,
    perfect_network,
    train_state=None,
):
    """Training loop for PPO based on https://github.com/bmazoure/ppo_jax."""
    # counts = np.zeros((13, 13))
    values = np.zeros((13, 13))
    num_total_epochs = int(config.num_train_steps // config.num_train_envs + 1)
    num_steps_warm_up = int(config.num_train_steps * config.lr_warmup)
    schedule_fn = optax.linear_schedule(
        init_value=-float(config.lr_begin),
        end_value=-float(config.lr_end),
        transition_steps=num_steps_warm_up,
    )

    tx = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.scale_by_adam(eps=1e-5),
        optax.scale_by_schedule(schedule_fn),
    )

    if train_state == None:
        train_state = TrainState.create(apply_fn=model.apply, params=params, tx=tx,)
    # Setup the rollout manager -> Collects data in vmapped-fashion over envs
    rollout_manager = RolloutManager(
        model,
        config.env_name,
        config.env_kwargs,
        config.env_params,
        # config.map_params,
        # config.map_all_locations,
    )

    batch_manager = BatchManager(
        discount=config.gamma,
        gae_lambda=config.gae_lambda,
        n_steps=config.n_steps + 1,
        num_envs=config.num_train_envs,
        action_size=rollout_manager.action_size,
        state_space=rollout_manager.observation_space,
    )

    @partial(jax.jit, static_argnums=5)
    def get_transition(
        train_state: TrainState,
        obs: jnp.ndarray,
        state: dict,
        batch,
        rng: jax.random.PRNGKey,
        num_train_envs: int,
        # counts: jnp.ndarray,
    ):
        action, log_pi, value, new_key = rollout_manager.select_action(
            train_state, obs, rng
        )
        # print(action.shape)
        new_key, key_step = jax.random.split(new_key)
        b_rng = jax.random.split(key_step, num_train_envs)
        # Automatic env resetting in gymnax step!
        next_obs, next_state, reward, done, _ = rollout_manager.batch_step(
            b_rng, state, action
        )
        batch = batch_manager.append(batch, obs, action, reward, done, log_pi, value)
        return train_state, next_obs, next_state, batch, value, new_key

    batch = batch_manager.reset()

    rng, rng_step, rng_reset, rng_eval, rng_update = jax.random.split(rng, 5)
    # obs, state = rollout_manager.batch_reset(
    #     jax.random.split(rng_reset, config.num_train_envs), training=1
    # )

    obs, state = rollout_manager.batch_reset(
        jax.random.split(rng_reset, config.num_train_envs)
    )

    total_steps = 0
    (
        log_steps,
        log_return_train,
        log_return_test,
        log_td_error_train,
        log_td_error_test,
        log_mean_novelty_train,
        log_mean_novelty_test,
        log_kl_div,
        log_mse,
        log_critic_mean_abs_activation,
        log_actor_mean_abs_activation,
        log_critic_mean_RMS_activation,
        log_actor_mean_RMS_activation,
        # log_mean_counts,
    ) = ([], [], [], [], [], [], [], [], [], [], [], [], [])
    t = tqdm.tqdm(range(1, num_total_epochs + 1), desc="PPO", leave=True)

    for step in t:
        train_state, obs, state, batch, new_values, rng_step = get_transition(
            train_state,
            obs,
            state,
            batch,
            rng_step,
            config.num_train_envs,
            # jnp.array(counts),
        )
        # update_counts(counts, obs)
        # update_values(values, obs, new_values)
        total_steps += config.num_train_envs
        if step % (config.n_steps + 1) == 0 and step > 1:
            metric_dict, train_state, rng_update = update(
                train_state,
                batch_manager.get(batch),
                config.num_train_envs,
                config.n_steps,
                config.n_minibatch,
                config.epoch_ppo,
                config.clip_eps,
                config.entropy_coeff,
                config.critic_coeff,
                rng_update,
                # counts,
            )
            batch = batch_manager.reset()

        if step % config.evaluate_every_epochs == 0 or step == 1:
            rng, rng_eval = jax.random.split(rng)
            # (
            #     rewards_test,
            #     td_error_test,
            #     mean_novelty_test,
            # ) = rollout_manager.batch_evaluate(
            #     rng_eval, train_state, config.num_test_rollouts, counts, training=0
            # )

            # (
            #     rewards_test,
            #     td_error_test,
            #     mean_novelty_test,
            # ) = rollout_manager.batch_evaluate(
            #     rng_eval, train_state, config.num_test_rollouts
            # )
            # (
            #     rewards_train,
            #     td_error_train,
            #     mean_novelty_train,
            # ) = rollout_manager.batch_evaluate(
            #     rng_eval, train_state, config.num_test_rollouts, counts, training=1
            # )
            rewards_train = rollout_manager.batch_evaluate(
                rng_eval, train_state, config.num_test_rollouts  # , training=1
            )

            log_steps.append(total_steps)
            log_return_train.append(np.mean(rewards_train))
            # log_return_test.append(np.mean(rewards_test))
            # log_td_error_train.append(td_error_train)
            # log_td_error_test.append(td_error_test)
            # log_mean_novelty_train.append(mean_novelty_train)
            # log_mean_novelty_test.append(mean_novelty_test)
            t.set_description(f"R: {str(rewards_train)}")
            t.refresh()
            KL_div, MSE = compute_difference_with_perfect_policy(
                training_state=train_state,
                training_network=model,
                perfect_network_params=perfect_network_params,
                perfect_network=perfect_network,
                obs=obs,
                # counts=counts,
                rng=rng_eval,
            )
            # KL_div, MSE = compute_difference_with_perfect_policy(
            #     training_state=train_state,
            #     training_network=model,
            #     perfect_network_params=perfect_network_params,
            #     obs=obs,
            #     rng=rng_eval,
            # )
            log_kl_div.append(KL_div)
            log_mse.append(MSE)
            # log_mean_counts.append(mean_counts)
            # log_value_predictions(
            #     use_wandb,
            #     model,
            #     train_state,
            #     rollout_manager,
            #     rng,
            #     counts,
            #     STR_TO_JAX_ARR,
            # )

            # log_value_predictions(
            #     use_wandb, model, train_state, rollout_manager, rng, STR_TO_JAX_ARR,
            # )

            # log_error_vs_counts(
            #     freq_counts=HIGH_FREQ_COUNTS,
            #     freq_errors=HIGH_FREQ_ERRORS,
            #     log_str="high",
            #     use_wandb=use_wandb,
            # )
            # log_error_vs_counts(
            #     freq_counts=LOW_FREQ_COUNTS,
            #     freq_errors=LOW_FREQ_ERRORS,
            #     log_str="low",
            #     use_wandb=use_wandb,
            # )
            # log_counts(counts=counts, use_wandb=use_wandb)
            if config.SIRENs == True:
                (
                    critic_activation_abs_mean,
                    policy_activation_abs_mean,
                    critic_activation_RMS_mean,
                    policy_activation_RMS_mean,
                ) = log_frequency_stats(train_state, obs)
                log_histograms(train_state, obs, config)
                log_critic_mean_abs_activation.append(critic_activation_abs_mean)
                log_actor_mean_abs_activation.append(policy_activation_abs_mean)
                log_critic_mean_RMS_activation.append(critic_activation_RMS_mean)
                log_actor_mean_RMS_activation.append(policy_activation_RMS_mean)
            # log the std dev of the first layer of the policy network

            # model.apply(train_state.params, obs, counts, STR_TO_JAX_ARR["mixed"], rng)

            model.apply(train_state.params, obs, rng)
            if mle_log is not None:
                mle_log.update(
                    {"num_steps": total_steps},
                    {"return": rewards_train},
                    model=train_state.params,
                    save=True,
                )

    return (
        log_steps,
        log_return_train,
        # log_return_test,
        # log_td_error_train,
        # log_td_error_test,
        # log_mean_novelty_train,
        # log_mean_novelty_test,
        log_kl_div,
        log_mse,
        log_critic_mean_abs_activation,
        log_actor_mean_abs_activation,
        log_critic_mean_RMS_activation,
        log_actor_mean_RMS_activation,
        train_state.params,
        train_state,
    )


@jax.jit
def flatten_dims(x):
    return x.swapaxes(0, 1).reshape(x.shape[0] * x.shape[1], *x.shape[2:])


def loss_actor_and_critic(
    params_model: flax.core.frozen_dict.FrozenDict,
    apply_fn: Callable[..., Any],
    obs: jnp.ndarray,
    target: jnp.ndarray,
    value_old: jnp.ndarray,
    log_pi_old: jnp.ndarray,
    gae: jnp.ndarray,
    action: jnp.ndarray,
    clip_eps: float,
    critic_coeff: float,
    entropy_coeff: float,
    # counts: jnp.ndarray,
    # high_low_or_mixed: jnp.ndarray,
) -> jnp.ndarray:

    # value_pred, pi = apply_fn(params_model, obs, counts, high_low_or_mixed, rng=None)
    value_pred, pi = apply_fn(params_model, obs, rng=None)
    value_pred = value_pred[:, 0]

    # TODO: Figure out why training without 0 breaks categorical model
    # And why with 0 breaks gaussian model pi
    log_prob = pi.log_prob(action[..., -1])

    value_pred_clipped = value_old + (value_pred - value_old).clip(-clip_eps, clip_eps)
    value_losses = jnp.square(value_pred - target)
    value_losses_clipped = jnp.square(value_pred_clipped - target)
    value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

    ratio = jnp.exp(log_prob - log_pi_old)
    gae_mean = gae.mean()
    gae = (gae - gae_mean) / (gae.std() + 1e-8)
    loss_actor1 = ratio * gae
    loss_actor2 = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * gae
    loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
    loss_actor = loss_actor.mean()

    entropy = pi.entropy().mean()

    total_loss = loss_actor + critic_coeff * value_loss - entropy_coeff * entropy

    return (
        total_loss,
        (
            value_loss,
            loss_actor,
            entropy,
            value_pred.mean(),
            target.mean(),
            gae_mean,
            (value_pred - value_old) / value_pred,
        ),
    )


def update(
    train_state: TrainState,
    batch: Tuple,
    num_envs: int,
    n_steps: int,
    n_minibatch: int,
    epoch_ppo: int,
    clip_eps: float,
    entropy_coeff: float,
    critic_coeff: float,
    rng: jax.random.PRNGKey,
    # counts: jnp.ndarray,
):
    """Perform multiple epochs of updates with multiple updates."""
    obs, action, log_pi_old, value, target, gae = batch
    size_batch = num_envs * n_steps
    size_minibatch = size_batch // n_minibatch
    idxes = jnp.arange(num_envs * n_steps)
    avg_metrics_dict = defaultdict(int)

    for _ in range(epoch_ppo):
        idxes = jax.random.permutation(rng, idxes)
        idxes_list = [
            idxes[start : start + size_minibatch]
            for start in jnp.arange(0, size_batch, size_minibatch)
        ]

        # train_state, total_loss = update_epoch(
        #     train_state,
        #     idxes_list,
        #     flatten_dims(obs),
        #     flatten_dims(action),
        #     flatten_dims(log_pi_old),
        #     flatten_dims(value),
        #     jnp.array(flatten_dims(target)),
        #     jnp.array(flatten_dims(gae)),
        #     clip_eps,
        #     entropy_coeff,
        #     critic_coeff,
        #     counts=counts,
        #     high_low_or_mixed=STR_TO_JAX_ARR["low"],
        # )

        train_state, total_loss = update_epoch(
            train_state,
            idxes_list,
            flatten_dims(obs),
            flatten_dims(action),
            flatten_dims(log_pi_old),
            flatten_dims(value),
            jnp.array(flatten_dims(target)),
            jnp.array(flatten_dims(gae)),
            clip_eps,
            entropy_coeff,
            critic_coeff,
            # high_low_or_mixed=STR_TO_JAX_ARR["low"],
        )

        (
            total_loss,
            (
                value_loss,
                loss_actor,
                entropy,
                value_pred,
                target_val,
                gae_val,
                value_error,
            ),
        ) = total_loss

        # global HIGH_FREQ_COUNTS
        # global HIGH_FREQ_ERRORS
        # global LOW_FREQ_COUNTS
        # global LOW_FREQ_ERRORS

        # obs_to_index = obs.reshape(obs.shape[0] * obs.shape[1], -1)
        # counts_to_log = counts[
        #     obs_to_index[:, :2].astype(int)[:, 0], obs_to_index[:, :2].astype(int)[:, 1]
        # ]
        # LOW_FREQ_ERRORS = np.abs(np.asarray(value_error))
        # LOW_FREQ_COUNTS = np.asarray(counts_to_log)
        avg_metrics_dict["total_loss"] += np.asarray(total_loss)
        avg_metrics_dict["value_loss"] += np.asarray(value_loss)
        avg_metrics_dict["actor_loss"] += np.asarray(loss_actor)
        avg_metrics_dict["entropy"] += np.asarray(entropy)
        avg_metrics_dict["value_pred"] += np.asarray(value_pred)
        avg_metrics_dict["target"] += np.asarray(target_val)
        avg_metrics_dict["gae"] += np.asarray(gae_val)

        # train_state, total_loss = update_epoch(
        #     train_state,
        #     idxes_list,
        #     flatten_dims(obs),
        #     flatten_dims(action),
        #     flatten_dims(log_pi_old),
        #     flatten_dims(value),
        #     jnp.array(flatten_dims(target)),
        #     jnp.array(flatten_dims(gae)),
        #     clip_eps,
        #     entropy_coeff,
        #     critic_coeff,
        #     # counts=counts,
        #     high_low_or_mixed=STR_TO_JAX_ARR["high"],
        # )

        # (
        #     total_loss,
        #     (
        #         value_loss,
        #         loss_actor,
        #         entropy,
        #         value_pred,
        #         target_val,
        #         gae_val,
        #         value_error,
        #     ),
        # ) = total_loss

        # obs_to_index = obs.reshape(obs.shape[0] * obs.shape[1], -1)
        # counts_to_log = counts[
        #     obs_to_index[:, :2].astype(int)[:, 0], obs_to_index[:, :2].astype(int)[:, 1]
        # ]
        # HIGH_FREQ_ERRORS = np.abs(np.asarray(value_error))
        # HIGH_FREQ_COUNTS = np.asarray(counts_to_log)
        # avg_metrics_dict["total_loss"] += np.asarray(total_loss)
        # avg_metrics_dict["value_loss"] += np.asarray(value_loss)
        # avg_metrics_dict["actor_loss"] += np.asarray(loss_actor)
        # avg_metrics_dict["entropy"] += np.asarray(entropy)
        # avg_metrics_dict["value_pred"] += np.asarray(value_pred)
        # avg_metrics_dict["target"] += np.asarray(target_val)
        # avg_metrics_dict["gae"] += np.asarray(gae_val)

    for k, v in avg_metrics_dict.items():
        avg_metrics_dict[k] = v / (epoch_ppo)

    return avg_metrics_dict, train_state, rng


@jax.jit
def update_epoch(
    train_state: TrainState,
    idxes: jnp.ndarray,
    obs,
    action,
    log_pi_old,
    value,
    target,
    gae,
    clip_eps: float,
    entropy_coeff: float,
    critic_coeff: float,
    # counts: jnp.ndarray,
    # high_low_or_mixed: jnp.ndarray,
):
    for idx in idxes:
        # print(action[idx].shape, action[idx].reshape(-1, 1).shape)
        grad_fn = jax.value_and_grad(loss_actor_and_critic, has_aux=True)
        total_loss, grads = grad_fn(
            train_state.params,
            train_state.apply_fn,
            obs=obs[idx],
            target=target[idx],
            value_old=value[idx],
            log_pi_old=log_pi_old[idx],
            gae=gae[idx],
            # action=action[idx].reshape(-1, 1),
            action=jnp.expand_dims(action[idx], -1),
            clip_eps=clip_eps,
            critic_coeff=critic_coeff,
            entropy_coeff=entropy_coeff,
            # counts=counts,
            # high_low_or_mixed=high_low_or_mixed,
        )
        train_state = train_state.apply_gradients(grads=grads)
    return train_state, total_loss

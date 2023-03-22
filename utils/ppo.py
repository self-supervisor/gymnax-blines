from collections import defaultdict
from functools import partial
from typing import Any, Callable, Tuple

import flax
import gymnax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tqdm
from flax.training.train_state import TrainState
import wandb

STR_TO_JAX_ARR = {
    "high": jnp.array([1, 0, 0]),
    "low": jnp.array([0, 1, 0]),
    "mixed": jnp.array([0, 0, 1]),
}


class BatchManager:
    def __init__(
        self,
        discount: float,
        gae_lambda: float,
        n_steps: int,
        num_envs: int,
        action_size,
        state_space,
    ):
        self.num_envs = num_envs
        self.action_size = action_size
        self.buffer_size = num_envs * n_steps
        self.num_envs = num_envs
        self.n_steps = n_steps
        self.discount = discount
        self.gae_lambda = gae_lambda

        try:
            temp = state_space.shape[0]
            self.state_shape = state_space.shape
        except Exception:
            self.state_shape = [state_space]
        self.reset()

    @partial(jax.jit, static_argnums=0)
    def reset(self):
        return {
            "states": jnp.empty(
                (self.n_steps, self.num_envs, *self.state_shape), dtype=jnp.float32,
            ),
            "actions": jnp.empty((self.n_steps, self.num_envs, *self.action_size),),
            "rewards": jnp.empty((self.n_steps, self.num_envs), dtype=jnp.float32),
            "dones": jnp.empty((self.n_steps, self.num_envs), dtype=jnp.uint8),
            "log_pis_old": jnp.empty((self.n_steps, self.num_envs), dtype=jnp.float32),
            "values_old": jnp.empty((self.n_steps, self.num_envs), dtype=jnp.float32),
            "_p": 0,
        }

    @partial(jax.jit, static_argnums=0)
    def append(self, buffer, state, action, reward, done, log_pi, value):
        return {
            "states": buffer["states"].at[buffer["_p"]].set(state),
            "actions": buffer["actions"].at[buffer["_p"]].set(action),
            "rewards": buffer["rewards"].at[buffer["_p"]].set(reward.squeeze()),
            "dones": buffer["dones"].at[buffer["_p"]].set(done.squeeze()),
            "log_pis_old": buffer["log_pis_old"].at[buffer["_p"]].set(log_pi),
            "values_old": buffer["values_old"].at[buffer["_p"]].set(value),
            "_p": (buffer["_p"] + 1) % self.n_steps,
        }

    @partial(jax.jit, static_argnums=0)
    def get(self, buffer):
        gae, target = self.calculate_gae(
            value=buffer["values_old"], reward=buffer["rewards"], done=buffer["dones"],
        )
        batch = (
            buffer["states"][:-1],
            buffer["actions"][:-1],
            buffer["log_pis_old"][:-1],
            buffer["values_old"][:-1],
            target,
            gae,
        )
        return batch

    @partial(jax.jit, static_argnums=0)
    def calculate_gae(
        self, value: jnp.ndarray, reward: jnp.ndarray, done: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        advantages = []
        gae = 0.0
        for t in reversed(range(len(reward) - 1)):
            value_diff = self.discount * value[t + 1] * (1 - done[t]) - value[t]
            delta = reward[t] + value_diff
            gae = delta + self.discount * self.gae_lambda * (1 - done[t]) * gae
            advantages.append(gae)
        advantages = advantages[::-1]
        advantages = jnp.array(advantages)
        return advantages, advantages + value[:-1]


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
        self,
        train_state: TrainState,
        obs: jnp.ndarray,
        counts: jnp.ndarray,
        low_mixed_or_high: jnp.ndarray,
        rng: jax.random.PRNGKey,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jax.random.PRNGKey]:
        value, pi = policy(
            train_state.apply_fn,
            train_state.params,
            obs,
            counts,
            low_mixed_or_high,
            rng,
        )
        action = pi.sample(seed=rng)
        log_prob = pi.log_prob(action)
        return action, log_prob, value[:, 0], rng

    @partial(jax.jit, static_argnums=0)
    def batch_reset(self, keys, training):
        return jax.vmap(self.env.reset, in_axes=(0, 0, None))(
            keys, jnp.array([training] * keys.shape[0]), self.env_params
        )

    @partial(jax.jit, static_argnums=0)
    def batch_step(self, keys, state, action):
        return jax.vmap(self.env.step, in_axes=(0, 0, 0, None))(
            keys, state, action, self.env_params
        )

    @partial(jax.jit, static_argnums=(0, 3))
    def batch_evaluate(self, rng_input, train_state, num_envs, counts, training):
        """Rollout an episode with lax.scan."""
        # Reset the environment
        rng_reset, rng_episode = jax.random.split(rng_input)
        obs, state = self.batch_reset(jax.random.split(rng_reset, num_envs), training)

        def policy_step(state_input, _):
            """lax.scan compatible step transition in jax env."""
            obs, state, train_state, rng, cum_reward, valid_mask = state_input
            rng, rng_step, rng_net = jax.random.split(rng, 3)
            action, _, _, rng = self.select_action(
                train_state, obs, counts, STR_TO_JAX_ARR["mixed"], rng_net
            )
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


@partial(jax.jit, static_argnums=0)
def policy(
    apply_fn: Callable[..., Any],
    params: flax.core.frozen_dict.FrozenDict,
    obs: jnp.ndarray,
    counts: jnp.ndarray,
    low_mixed_or_high: jnp.ndarray,
    rng,
):
    value, pi = apply_fn(params, obs, counts, low_mixed_or_high, rng)
    return value, pi


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


def train_ppo(rng, config, model, params, mle_log, use_wandb):
    """Training loop for PPO based on https://github.com/bmazoure/ppo_jax."""
    counts = np.zeros((13, 13))
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

    train_state = TrainState.create(apply_fn=model.apply, params=params, tx=tx,)
    # Setup the rollout manager -> Collects data in vmapped-fashion over envs
    rollout_manager = RolloutManager(
        model, config.env_name, config.env_kwargs, config.env_params
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
        counts: jnp.ndarray,
    ):
        action, log_pi, value, new_key = rollout_manager.select_action(
            train_state, obs, counts, STR_TO_JAX_ARR["mixed"], rng
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
    obs, state = rollout_manager.batch_reset(
        jax.random.split(rng_reset, config.num_train_envs), training=1
    )

    total_steps = 0
    log_steps, log_return_train, log_return_test = [], [], []
    t = tqdm.tqdm(range(1, num_total_epochs + 1), desc="PPO", leave=True)
    for step in t:
        train_state, obs, state, batch, new_values, rng_step = get_transition(
            train_state,
            obs,
            state,
            batch,
            rng_step,
            config.num_train_envs,
            jnp.array(counts),
        )
        update_counts(counts, obs)
        update_values(values, obs, new_values)
        total_steps += config.num_train_envs
        if step % (config.n_steps + 1) == 0:
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
                counts,
            )
            batch = batch_manager.reset()

        if (step + 1) % config.evaluate_every_epochs == 0:
            rng, rng_eval = jax.random.split(rng)
            rewards_train = rollout_manager.batch_evaluate(
                rng_eval, train_state, config.num_test_rollouts, counts, training=1
            )
            rewards_test = rollout_manager.batch_evaluate(
                rng_eval, train_state, config.num_test_rollouts, counts, training=0
            )

            log_steps.append(total_steps)
            log_return_train.append(rewards_train)
            log_return_test.append(rewards_test)
            t.set_description(f"R: {str(rewards_train)}")
            t.refresh()
            log_value_predictions(
                use_wandb, model, train_state, rollout_manager, rng, counts
            )

            model.apply(train_state.params, obs, counts, STR_TO_JAX_ARR["mixed"], rng)
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
        log_return_test,
        train_state.params,
    )


def log_value_predictions(
    use_wandb: bool, model, train_state, rollout_manager, rng, counts: np.ndarray,
) -> None:
    for key in STR_TO_JAX_ARR.keys():
        values = np.zeros((13, 13))
        preds = model.apply(
            train_state.params,
            rollout_manager.env.coords,
            counts,
            STR_TO_JAX_ARR[key],
            rng,
        )
        all_coordinates = [
            [np.array(i)[0], np.array(i)[1]] for i in rollout_manager.env.coords
        ]
        for index, val in enumerate(all_coordinates):
            i, j = val[0], val[1]
            values[i, j] = np.array(preds[0][index])

        if use_wandb:
            import plotly.graph_objects as go

            fig = go.Figure(
                data=go.Heatmap(
                    z=values,
                    x=np.arange(0, 13),
                    y=np.arange(0, 13),
                    colorscale="Viridis",
                )
            )
            wandb.log({f"value_predictions_{key}": fig})

    if use_wandb:
        fig = go.Figure(
            data=go.Heatmap(
                z=counts, x=np.arange(0, 13), y=np.arange(0, 13), colorscale="Viridis",
            )
        )
        wandb.log({"counts": fig})


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
    counts: jnp.ndarray,
    high_low_or_mixed: jnp.ndarray,
) -> jnp.ndarray:

    value_pred, pi = apply_fn(params_model, obs, counts, high_low_or_mixed, rng=None)
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
        (value_loss, loss_actor, entropy, value_pred.mean(), target.mean(), gae_mean,),
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
    counts: jnp.ndarray,
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
            counts=counts,
            high_low_or_mixed=STR_TO_JAX_ARR["low"],
        )

        (
            total_loss,
            (value_loss, loss_actor, entropy, value_pred, target_val, gae_val,),
        ) = total_loss

        avg_metrics_dict["total_loss"] += np.asarray(total_loss)
        avg_metrics_dict["value_loss"] += np.asarray(value_loss)
        avg_metrics_dict["actor_loss"] += np.asarray(loss_actor)
        avg_metrics_dict["entropy"] += np.asarray(entropy)
        avg_metrics_dict["value_pred"] += np.asarray(value_pred)
        avg_metrics_dict["target"] += np.asarray(target_val)
        avg_metrics_dict["gae"] += np.asarray(gae_val)

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
            counts=counts,
            high_low_or_mixed=STR_TO_JAX_ARR["high"],
        )

        (
            total_loss,
            (value_loss, loss_actor, entropy, value_pred, target_val, gae_val,),
        ) = total_loss

        avg_metrics_dict["total_loss"] += np.asarray(total_loss)
        avg_metrics_dict["value_loss"] += np.asarray(value_loss)
        avg_metrics_dict["actor_loss"] += np.asarray(loss_actor)
        avg_metrics_dict["entropy"] += np.asarray(entropy)
        avg_metrics_dict["value_pred"] += np.asarray(value_pred)
        avg_metrics_dict["target"] += np.asarray(target_val)
        avg_metrics_dict["gae"] += np.asarray(gae_val)

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
    counts: jnp.ndarray,
    high_low_or_mixed: jnp.ndarray,
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
            counts=counts,
            high_low_or_mixed=high_low_or_mixed,
        )
        train_state = train_state.apply_gradients(grads=grads)
    return train_state, total_loss

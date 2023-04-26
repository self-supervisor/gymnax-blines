from typing import Dict

import numpy as np
import plotly.graph_objects as go
import wandb
from flax.training.train_state import TrainState
from typing import Tuple
import jax.numpy as jnp
import flax
import jax
from scipy.stats import entropy

STR_TO_JAX_ARR = {
    "high": jnp.array([1, 0, 0]),
    "low": jnp.array([0, 1, 0]),
    "mixed": jnp.array([0, 0, 1]),
}


def compute_difference_with_perfect_policy(
    training_state: TrainState,
    training_network,
    perfect_network_params,
    perfect_network,
    obs,
    rng,
) -> Tuple[jnp.array, jnp.array]:
    """compute kl divergence between policy predictions and MSE between value predictions"""
    perfect_network_values, perfect_network_policy = perfect_network.apply(
        perfect_network_params, obs, rng=rng,
    )
    training_network_values, training_network_policy = training_network.apply(
        training_state.params, obs, rng=rng,
    )
    training_network_policy_probs = np.array(
        jax.nn.softmax(training_network_policy.logits)
    )
    perfect_network_policy_probs = np.array(
        jax.nn.softmax(perfect_network_policy.logits)
    )
    kl_div = entropy(perfect_network_policy_probs, training_network_policy_probs).mean()
    MSE = np.array(jnp.square(training_network_values - perfect_network_values).mean())
    wandb.log(
        {"kl_div_with_perfect_policy": kl_div, "mse_with_perfect_value_network": MSE}
    )
    return kl_div, MSE


def make_arrows(actions: np.ndarray, all_coordinates: np.ndarray, direction: int):
    x = []
    y = []

    for index, val in enumerate(all_coordinates):
        if actions[index] == direction:
            i, j = val[0], val[1]
            x.append(i)
            y.append(j)

    return x, y


def get_predictions_for_all_states(
    input_positions: jnp.array,
    goal_positions: jnp.array,
    model: flax.linen.Module,
    params: jnp.array,
    rng,
    counts,
) -> jnp.array:
    model_inputs = jnp.concatenate([input_positions, goal_positions], axis=1)
    preds = model.apply(
        params,
        model_inputs,
        counts,
        high_low_or_mixed=STR_TO_JAX_ARR["mixed"],
        rng=rng,
    )
    return preds


def log_value_predictions(
    use_wandb: bool,
    model,
    train_state,
    rollout_manager,
    rng,
    counts: np.ndarray,
    STR_TO_JAX_ARR: Dict,
) -> None:
    for key in STR_TO_JAX_ARR.keys():
        values = np.zeros((13, 13))
        goal_positions = np.repeat(
            [[2, 9]], rollout_manager.env.coords.shape[0], axis=0
        )
        input_positions = rollout_manager.env.coords
        value_preds, pi = get_predictions_for_all_states(
            input_positions=input_positions,
            goal_positions=goal_positions,
            model=model,
            params=train_state.params,
            rng=rng,
            counts=counts,
        )
        all_coordinates = [
            [np.array(i)[0], np.array(i)[1]] for i in rollout_manager.env.coords
        ]
        for index, val in enumerate(all_coordinates):
            i, j = val[0], val[1]
            values[i, j] = np.array(value_preds[index])

        actions = pi.sample(seed=rng)
        x_arrows_up, y_arrows_up = make_arrows(actions, all_coordinates, 0)
        x_arrows_down, y_arrows_down = make_arrows(actions, all_coordinates, 2)
        x_arrows_left, y_arrows_left = make_arrows(actions, all_coordinates, 1)
        x_arrows_right, y_arrows_right = make_arrows(actions, all_coordinates, 3)

        if use_wandb:

            fig = go.Figure(
                data=go.Heatmap(
                    z=values,
                    x=np.arange(0, 13),
                    y=np.arange(0, 13),
                    colorscale="Viridis",
                    text=actions,
                )
            )

            # log actions as arrows on the heatmap
            fig.add_trace(
                go.Scatter(
                    y=x_arrows_up,
                    x=y_arrows_up,
                    mode="markers",
                    marker=dict(
                        size=10,
                        color="black",
                        symbol="arrow-down",
                        opacity=0.5,
                        line=dict(width=1, color="black"),
                    ),
                )
            )
            fig.add_trace(
                go.Scatter(
                    y=x_arrows_down,
                    x=y_arrows_down,
                    mode="markers",
                    marker=dict(
                        size=10,
                        color="black",
                        symbol="arrow-up",
                        opacity=0.5,
                        line=dict(width=1, color="black"),
                    ),
                )
            )
            fig.add_trace(
                go.Scatter(
                    y=x_arrows_left,
                    x=y_arrows_left,
                    mode="markers",
                    marker=dict(
                        size=10,
                        color="black",
                        symbol="arrow-right",
                        opacity=0.5,
                        line=dict(width=1, color="black"),
                    ),
                )
            )
            fig.add_trace(
                go.Scatter(
                    y=x_arrows_right,
                    x=y_arrows_right,
                    mode="markers",
                    marker=dict(
                        size=10,
                        color="black",
                        symbol="arrow-left",
                        opacity=0.5,
                        line=dict(width=1, color="black"),
                    ),
                )
            )

            wandb.log({f"value_predictions_{key}": fig})


def log_error_vs_counts(
    freq_counts: np.ndarray, freq_errors: np.ndarray, log_str: str, use_wandb: bool
) -> None:
    if use_wandb:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=freq_counts, y=freq_errors, mode="markers", name="freq_counts",
            )
        )
        wandb.log({f"scatter_plot_{log_str}": fig})


def log_counts(counts: np.ndarray, use_wandb: bool):
    if use_wandb:
        fig = go.Figure(
            data=go.Heatmap(
                z=counts, x=np.arange(0, 13), y=np.arange(0, 13), colorscale="Viridis",
            )
        )
        wandb.log({"counts": fig})


def save_plotly_to_json(fig, title) -> None:
    import json

    fig_json = fig.to_json()
    with open("fig.json", "w") as f:
        json.dump(fig_json, f)


def make_plotly_histogram(datapoints, config, title):
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=np.array(datapoints).reshape(-1), name=title))
    wandb.log({f"{title}_histogram": fig})

    save_plotly_to_json(
        fig, title=f"three_dim_plots/{config.env_name}_{config.scale}_histogram.json"
    )


def log_median(datapoints, config, title):
    wandb.log({f"{title}_median": np.median(datapoints)})
    np.save(
        f"three_dim_plots/{config.env_name}_{config.scale}_{title}_median.npy",
        np.median(datapoints),
    )


def log_histograms(train_state: TrainState, obs: jnp.array, config) -> None:
    if len(obs) > 2:
        obs = obs.reshape(obs.shape[0], -1)
    policy_initial_weight_matrix = np.array(
        [i for i in train_state.params.items()][0][1]["lff_actor"]["dense"]["kernel"]
    )
    datapoints = jnp.array(
        [policy_initial_weight_matrix.T @ obs[i] for i in range(len(obs))]
    )
    make_plotly_histogram(datapoints, config, title="policy")
    log_median(datapoints, config, title="policy")

    critic_initial_weight_matrix = np.array(
        [i for i in train_state.params.items()][0][1]["lff_actor"]["dense"]["kernel"]
    )
    datapoints = jnp.array(
        [critic_initial_weight_matrix.T @ obs[i] for i in range(len(obs))]
    )
    make_plotly_histogram(datapoints, config, title="critic")
    log_median(datapoints, config, title="critic")


def log_frequency_stats(train_state: TrainState, obs: jnp.array) -> None:
    if len(obs) > 2:
        obs = obs.reshape(obs.shape[0], -1)
    critic_initial_weight_matrix = np.array(
        [i for i in train_state.params.items()][0][1]["lff_critic"]["dense"]["kernel"]
    )
    critic_std = jnp.std(critic_initial_weight_matrix)
    critic_norm = jnp.linalg.norm(critic_initial_weight_matrix)
    critic_average_activation = jnp.mean(
        jnp.abs(
            jnp.array(
                [critic_initial_weight_matrix.T @ obs[i] for i in range(len(obs))]
            )
        )
    )
    critic_norm_activation = jnp.linalg.norm(
        jnp.array([critic_initial_weight_matrix.T @ obs[i] for i in range(len(obs))])
    )

    critic_rms_activation = jnp.sqrt(
        jnp.mean(
            jnp.square(
                jnp.array(
                    [critic_initial_weight_matrix.T @ obs[i] for i in range(len(obs))]
                )
            )
        )
    )
    critic_after_sin = jnp.mean(
        jnp.pi
        * jnp.sin(
            jnp.array(
                [critic_initial_weight_matrix.T @ obs[i] for i in range(len(obs))]
            )
        )
    )

    policy_initial_weight_matrix = np.array(
        [i for i in train_state.params.items()][0][1]["lff_actor"]["dense"]["kernel"]
    )
    policy_std = jnp.std(policy_initial_weight_matrix)
    policy_norm = jnp.linalg.norm(policy_initial_weight_matrix)
    policy_average_activation = jnp.mean(
        jnp.abs(
            jnp.array(
                [policy_initial_weight_matrix.T @ obs[i] for i in range(len(obs))]
            )
        )
    )
    policy_norm_activation = jnp.linalg.norm(
        jnp.array([policy_initial_weight_matrix.T @ obs[i] for i in range(len(obs))])
    )
    policy_rms_activation = jnp.sqrt(
        jnp.square(
            jnp.mean(
                jnp.array(
                    [policy_initial_weight_matrix.T @ obs[i] for i in range(len(obs))]
                )
            )
        )
    )
    policy_after_sin = jnp.mean(
        jnp.pi
        * jnp.sin(
            jnp.array(
                [policy_initial_weight_matrix.T @ obs[i] for i in range(len(obs))]
            )
        )
    )

    wandb.log({"critic_std": np.array(critic_std)})
    wandb.log({"critic_norm": np.array(critic_norm)})

    wandb.log({"policy_std": np.array(policy_std)})
    wandb.log({"policy_norm": np.array(policy_norm)})

    wandb.log({"critic_average_activation": np.array(critic_average_activation)})
    wandb.log({"critic_norm_activation": np.array(critic_norm_activation)})

    wandb.log({"policy_average_activation": np.array(policy_average_activation)})
    wandb.log({"policy_norm_activation": np.array(policy_norm_activation)})

    wandb.log({"critic_rms_activation": np.array(critic_rms_activation)})
    wandb.log({"policy_rms_activation": np.array(policy_rms_activation)})

    wandb.log({"critic_after_sin": np.array(critic_after_sin)})
    wandb.log({"policy_after_sin": np.array(policy_after_sin)})

    if np.isnan(np.array(critic_after_sin)):
        jax.debug.breakpoint()
    return (
        critic_average_activation,
        policy_average_activation,
        critic_rms_activation,
        policy_rms_activation,
    )

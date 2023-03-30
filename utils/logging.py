from typing import Dict

import numpy as np
import plotly.graph_objects as go
import wandb
import jax


def make_arrows(actions: np.ndarray, all_coordinates: np.ndarray, direction: int):
    x = []
    y = []

    for index, val in enumerate(all_coordinates):
        if actions[index] == direction:
            i, j = val[0], val[1]
            x.append(i)
            y.append(j)

    return x, y


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
        network_inputs = np.concatenate(
            [input_positions, goal_positions], axis=1
        ).astype(np.float32)

        preds = model.apply(
            train_state.params, network_inputs, counts, STR_TO_JAX_ARR[key], rng,
        )
        all_coordinates = [
            [np.array(i)[0], np.array(i)[1]] for i in rollout_manager.env.coords
        ]
        for index, val in enumerate(all_coordinates):
            i, j = val[0], val[1]
            values[i, j] = np.array(preds[0][index])

        actions = preds[1].sample(seed=rng)
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

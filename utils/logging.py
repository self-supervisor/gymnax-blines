from typing import Dict

import numpy as np
import plotly.graph_objects as go
import wandb
import jax


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
            [[4, 4]], rollout_manager.env.coords.shape[0], axis=0
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

        if use_wandb:

            fig = go.Figure(
                data=go.Heatmap(
                    z=values,
                    x=np.arange(0, 13),
                    y=np.arange(0, 13),
                    colorscale="Viridis",
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

import chex
import haiku as hk
import jax


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

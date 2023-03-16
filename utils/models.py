import flax.linen as nn
import gymnax
import jax
import jax.numpy as jnp
from evosax import NetworkMapper
from tensorflow_probability.substrates import jax as tfp


def build_RND_models(obs_shape, rng_rnd, rng_distiller):
    RND_model = MLP()
    RND_params = RND_model.init(rng_rnd, jnp.zeros(obs_shape))

    distiller_model = MLP()
    distiller_params = distiller_model.init(rng_distiller, jnp.zeros(obs_shape))
    return RND_model, RND_params, distiller_model, distiller_params


def get_model_ready(rng, config, scale, novelty_switch, speed=False):
    """Instantiate a model according to obs shape of environment."""
    # Get number of desired output units
    env, env_params = gymnax.make(config.env_name, **config.env_kwargs)

    rng, rng_rnd, rng_distiller = jax.random.split(rng, 3)
    # Instantiate model class (flax-based)
    if config.train_type == "ES":
        model = NetworkMapper[config.network_name](
            **config.network_config, num_output_units=env.num_actions
        )
    elif config.train_type == "PPO":
        if config.network_name == "Categorical-MLP":
            PPO_model = CategoricalSeparateMLP(
                **config.network_config,
                num_output_units=env.num_actions,
                scale=scale,
                novelty_switch=novelty_switch,
            )
        elif config.network_name == "Gaussian-MLP":
            model = GaussianSeparateMLP(
                **config.network_config, num_output_units=env.num_actions
            )

    # Only use feedforward MLP in speed evaluations!
    if speed and config.network_name == "LSTM":
        model = NetworkMapper["MLP"](
            num_hidden_units=64,
            num_hidden_layers=2,
            hidden_activation="relu",
            output_activation="categorical"
            if config.env_name != "PointRobot-misc"
            else "identity",
            num_output_units=env.num_actions,
        )

    # Initialize the network based on the observation shape
    obs_shape = env.observation_space(env_params).shape
    counts_shape = env.get_counts().shape
    if config.network_name != "LSTM" or speed:
        PPO_params = PPO_model.init(
            rng,
            jnp.zeros(obs_shape),
            jnp.zeros(counts_shape),
            jnp.array([0, 0]),
            rng=rng,
        )
    else:
        PPO_params = PPO_model.init(
            rng, jnp.zeros(obs_shape), model.initialize_carry(), rng=rng
        )

    RND_model, RND_params, distiller_model, distiller_params = build_RND_models(
        obs_shape, rng_model, rng_distiller
    )

    return (
        PPO_model,
        PPO_params,
        RND_model,
        RND_params,
        distiller_model,
        distiller_params,
    )


def default_mlp_init():
    return nn.initializers.uniform(scale=0.05)


def lff_weight_init(scale: float, num_inputs: int):
    return nn.initializers.normal(stddev=scale / num_inputs)


def lff_bias_init():
    return nn.initializers.uniform(scale=2)


class LFF(nn.Module):
    num_output_features: int
    num_input_features: int
    scale: float

    def setup(self):
        self.dense = nn.Dense(
            features=self.num_output_features,
            kernel_init=lff_weight_init(
                scale=self.scale, num_inputs=self.num_input_features
            ),
            bias_init=lff_bias_init(),
        )

    def __call__(self, x):
        return jnp.pi * jnp.sin(self.dense(x) - 1)


class MLP(nn.Module):
    num_hidden_units: int = 64
    num_hidden_layers: int = 2
    hidden_activation: str = "relu"
    output_activation: str = "identity"
    num_output_units: int = 64

    @nn.compact
    def __call__(self, x):
        for _ in range(self.num_hidden_layers):
            x = nn.relu(
                nn.Dense(
                    features=self.num_hidden_units,
                    kernel_init=default_mlp_init(),
                    bias_init=default_mlp_init(),
                )(x)
            )
        x = nn.Dense(
            features=self.num_output_units,
            kernel_init=default_mlp_init(),
            bias_init=default_mlp_init(),
        )(x)
        return x


class CategoricalSeparateMLP(nn.Module):
    """Split Actor-Critic Architecture for PPO."""

    num_output_units: int
    num_hidden_units: int
    num_hidden_layers: int
    scale: float
    prefix_actor: str = "actor"
    prefix_critic: str = "critic"
    model_name: str = "separate-mlp"
    flatten_2d: bool = False  # Catch case
    flatten_3d: bool = False  # Rooms/minatar case
    novelty_switch: int = int(2e3)

    @nn.compact
    def __call__(self, x, novelty_vector, high_low_or_mixed, rng):
        if self.flatten_2d and len(x.shape) == 2:
            x = x.reshape(-1)
        if self.flatten_2d and len(x.shape) > 2:
            x = x.reshape(x.shape[0], -1)
        if self.flatten_3d and len(x.shape) == 3:
            x = x.reshape(-1)
        if self.flatten_3d and len(x.shape) > 3:
            x = x.reshape(x.shape[0], -1)

        x = (x / 13) - 0.5
        if len(x.shape) > 1:
            low_frequency = jnp.copy(x[:, :2])
            low_frequency = LFF(
                num_output_features=self.num_hidden_units,
                num_input_features=x.shape[-1],
                name=self.prefix_critic + "_fc_1_low_frequency",
                scale=self.scale * 0.001,
            )(low_frequency)
            high_frequency = jnp.copy(x[:, :2])
            high_frequency = LFF(
                num_output_features=self.num_hidden_units,
                num_input_features=x.shape[-1],
                name=self.prefix_critic + "_fc_1_high_frequency",
                scale=self.scale * 100,
            )(high_frequency)

        else:
            low_frequency = LFF(
                num_output_features=self.num_hidden_units,
                num_input_features=x.shape[-1],
                name=self.prefix_critic + "_fc_1_low_frequency",
                scale=self.scale * 0.001,
            )(x[:2])
            high_frequency = LFF(
                num_output_features=self.num_hidden_units,
                num_input_features=x.shape[-1],
                name=self.prefix_critic + "_fc_1_high_frequency",
                scale=self.scale * 1000,
            )(x[:2])

        for i in range(1, self.num_hidden_layers):
            x_v_high_frequency = nn.relu(
                nn.Dense(
                    self.num_hidden_units,
                    name=self.prefix_critic + f"_fc_{i+1}_high_frequency",
                    bias_init=default_mlp_init(),
                )(high_frequency)
            )
            x_v_low_frequency = nn.relu(
                nn.Dense(
                    self.num_hidden_units,
                    name=self.prefix_critic + f"_fc_{i+1}_low_frequency",
                    bias_init=default_mlp_init(),
                )(low_frequency)
            )
        v_low_frequency = nn.Dense(
            1,
            name=self.prefix_critic + "_fc_v_low_frequency",
            bias_init=default_mlp_init(),
        )(x_v_low_frequency)

        v_high_frequency = nn.Dense(
            1,
            name=self.prefix_critic + "_fc_v_high_frequency",
            bias_init=default_mlp_init(),
        )(x_v_high_frequency)

        if len(x.shape) == 1:
            v_low = jnp.copy(v_low_frequency)
            v_high = jnp.copy(v_high_frequency)
            v_low_frequency = v_low_frequency * (
                novelty_vector < self.novelty_switch
            ).astype(int)
            v_high_frequency = v_high_frequency * (
                novelty_vector > self.novelty_switch
            ).astype(int)
            v_mixed = v_high_frequency + v_low_frequency
            v = (
                high_low_or_mixed[0] * v_high
                + high_low_or_mixed[1] * v_low
                + high_low_or_mixed[2] * v_mixed
            )
        else:
            v_low = jnp.copy(v_low_frequency)
            v_high = jnp.copy(v_high_frequency)

            v_low_frequency = v_low_frequency * jnp.expand_dims(
                (novelty_vector < self.novelty_switch).astype(int), 1
            )
            v_high_frequency = v_high_frequency * jnp.expand_dims(
                (novelty_vector > self.novelty_switch).astype(int), 1
            )
            v_mixed = v_high_frequency + v_low_frequency
            v = (
                high_low_or_mixed[0] * v_high
                + high_low_or_mixed[1] * v_low
                + high_low_or_mixed[2] * v_mixed
            )

        if len(x.shape) > 1:
            x = x[:, :2]
        else:
            x = x[:2]

        x_a = nn.relu(nn.Dense(self.num_hidden_units, bias_init=default_mlp_init(),)(x))
        for i in range(1, self.num_hidden_layers):
            x_a = nn.relu(
                nn.Dense(self.num_hidden_units, bias_init=default_mlp_init(),)(x_a)
            )
        logits = nn.Dense(self.num_output_units, bias_init=default_mlp_init(),)(x_a)
        pi = tfp.distributions.Categorical(logits=logits)
        return v, pi


class GaussianSeparateMLP(nn.Module):
    """Split Actor-Critic Architecture for PPO."""

    num_output_units: int
    num_hidden_units: int
    num_hidden_layers: int
    prefix_actor: str = "actor"
    prefix_critic: str = "critic"
    min_std: float = 0.001
    model_name: str = "separate-mlp"

    @nn.compact
    def __call__(self, x, rng):
        x_v = nn.relu(
            nn.Dense(
                self.num_hidden_units,
                name=self.prefix_critic + "_fc_1",
                bias_init=default_mlp_init(),
            )(x)
        )
        # Loop over rest of intermediate hidden layers
        for i in range(1, self.num_hidden_layers):
            x_v = nn.relu(
                nn.Dense(
                    self.num_hidden_units,
                    name=self.prefix_critic + f"_fc_{i+1}",
                    bias_init=default_mlp_init(),
                )(x_v)
            )
        v = nn.Dense(
            1, name=self.prefix_critic + "_fc_v", bias_init=default_mlp_init(),
        )(x_v)

        x_a = nn.relu(
            nn.Dense(
                self.num_hidden_units,
                name=self.prefix_actor + "_fc_1",
                bias_init=default_mlp_init(),
            )(x)
        )
        # Loop over rest of intermediate hidden layers
        for i in range(1, self.num_hidden_layers):
            x_a = nn.relu(
                nn.Dense(
                    self.num_hidden_units,
                    name=self.prefix_actor + f"_fc_{i+1}",
                    bias_init=default_mlp_init(),
                )(x_a)
            )
        mu = nn.Dense(
            self.num_output_units,
            name=self.prefix_actor + "_fc_mu",
            bias_init=default_mlp_init(),
        )(x_a)
        log_scale = nn.Dense(
            self.num_output_units,
            name=self.prefix_actor + "_fc_scale",
            bias_init=default_mlp_init(),
        )(x_a)
        scale = jax.nn.softplus(log_scale) + self.min_std
        pi = tfp.distributions.MultivariateNormalDiag(mu, scale)
        return v, pi

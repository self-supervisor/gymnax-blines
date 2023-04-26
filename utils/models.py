import flax.linen as nn
import gymnax
import jax
import jax.numpy as jnp
from evosax import NetworkMapper
from tensorflow_probability.substrates import jax as tfp


def get_model_ready(rng, config, scale, speed=False, force_ReLU=False):
    """Instantiate a model according to obs shape of environment."""
    # Get number of desired output units
    env, env_params = gymnax.make(config.env_name, **config.env_kwargs)

    # Instantiate model class (flax-based)
    if config.train_type == "ES":
        model = NetworkMapper[config.network_name](
            **config.network_config, num_output_units=env.num_actions
        )
    elif config.train_type == "PPO":
        if config.network_name == "Categorical-MLP":
            # model = VanillaCategoricalSeparateMLP(
            #     **config.network_config,
            #     num_output_units=env.num_actions,
            #     scale=scale,
            #     high_freq_multiplier=high_freq_multiplier,
            #     count_switch=count_switch,
            # )
            if force_ReLU == True or config.SIRENs == False:
                model = CategoricalSeparateMLP(
                    **config.network_config,
                    num_output_units=env.num_actions,
                    scale=scale,
                    high=env.observation_space(env_params).high,
                    low=env.observation_space(env_params).low,
                    # high_freq_multiplier=high_freq_multiplier,
                    # count_switch=count_switch,
                )
            elif config.SIRENs == True:
                model = CategoricalSeparateMLPSIREN(
                    **config.network_config,
                    num_output_units=env.num_actions,
                    scale=scale,
                    high=env.observation_space(env_params).high,
                    low=env.observation_space(env_params).low,
                    # high_freq_multiplier=high_freq_multiplier,
                    # count_switch=count_switch,
                )
        elif config.network_name == "Gaussian-MLP":
            if force_ReLU == True or config.SIRENs == False:
                model = GaussianSeparateMLP(
                    **config.network_config,
                    scale=scale,
                    num_output_units=env.num_actions,
                    high=env.observation_space(env_params).high,
                    low=env.observation_space(env_params).low,
                )
            elif config.SIRENs == True:
                model = GaussianSeparateMLPSIREN(
                    **config.network_config,
                    scale=scale,
                    num_output_units=env.num_actions,
                    high=env.observation_space(env_params).high,
                    low=env.observation_space(env_params).low,
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
    # counts_shape = env.get_counts().shape
    if config.network_name != "LSTM" or speed:
        params = model.init(
            rng,
            jnp.zeros(obs_shape),
            # jnp.zeros(counts_shape),
            # jnp.array([0, 0]),
            rng=rng,
        )
    else:
        params = model.init(
            rng, jnp.zeros(obs_shape), model.initialize_carry(), rng=rng
        )
    return model, params


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


class VanillaCategoricalSeparateMLP(nn.Module):
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
    relu: bool = False
    not_relu: bool = True

    @nn.compact
    def __call__(self, x, rng):
        # Flatten a single 2D image
        if self.flatten_2d and len(x.shape) == 2:
            x = x.reshape(-1)
        # Flatten a batch of 2d images into a batch of flat vectors
        if self.flatten_2d and len(x.shape) > 2:
            x = x.reshape(x.shape[0], -1)

        # Flatten a single 3D image
        if self.flatten_3d and len(x.shape) == 3:
            x = x.reshape(-1)
        # Flatten a batch of 3d images into a batch of flat vectors
        if self.flatten_3d and len(x.shape) > 3:
            x = x.reshape(x.shape[0], -1)

        x_v = LFF(
            num_output_features=self.num_hidden_units,
            num_input_features=x.shape[-1],
            name="lff_critic",
            scale=self.scale,
        )(x)
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

        x_a = LFF(
            num_output_features=self.num_hidden_units,
            num_input_features=x.shape[-1],
            name="lff_critic",
            scale=self.scale,
        )(x)
        x_a = nn.relu(nn.Dense(self.num_hidden_units, bias_init=default_mlp_init(),)(x))
        # Loop over rest of intermediate hidden layers
        for i in range(1, self.num_hidden_layers):
            x_a = nn.relu(
                nn.Dense(self.num_hidden_units, bias_init=default_mlp_init(),)(x_a)
            )
        logits = nn.Dense(self.num_output_units, bias_init=default_mlp_init(),)(x_a)
        # pi = distrax.Categorical(logits=logits)
        pi = tfp.distributions.Categorical(logits=logits)
        return v, pi


class CategoricalSeparateMLPSIREN(nn.Module):
    """Split Actor-Critic Architecture for PPO."""

    num_output_units: int
    num_hidden_units: int
    num_hidden_layers: int
    scale: float
    high: jnp.ndarray
    low: jnp.ndarray
    prefix_actor: str = "actor"
    prefix_critic: str = "critic"
    model_name: str = "separate-mlp"
    flatten_2d: bool = False  # Catch case
    flatten_3d: bool = False  # Rooms/minatar case
    # count_switch: int = int(2e3)

    @nn.compact
    def __call__(self, x, rng):
        # Flatten a single 2D image
        if self.flatten_2d and len(x.shape) == 2:
            x = x.reshape(-1)
        # Flatten a batch of 2d images into a batch of flat vectors
        if self.flatten_2d and len(x.shape) > 2:
            x = x.reshape(x.shape[0], -1)

        # Flatten a single 3D image
        if self.flatten_3d and len(x.shape) == 3:
            x = x.reshape(-1)
        # Flatten a batch of 3d images into a batch of flat vectors
        if self.flatten_3d and len(x.shape) > 3:
            x = x.reshape(x.shape[0], -1)

        x = (x - self.low) / (self.high - self.low)
        x_v = LFF(
            num_output_features=self.num_hidden_units,
            num_input_features=x.shape[-1],
            name="lff_critic",
            scale=self.scale,
        )(x)
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

        x_a = LFF(
            num_output_features=self.num_hidden_units,
            num_input_features=x.shape[-1],
            name="lff_actor",
            scale=self.scale,
        )(x)
        # Loop over rest of intermediate hidden layers
        for i in range(1, self.num_hidden_layers):
            x_a = nn.relu(
                nn.Dense(self.num_hidden_units, bias_init=default_mlp_init(),)(x_a)
            )
        logits = nn.Dense(self.num_output_units, bias_init=default_mlp_init(),)(x_a)
        # pi = distrax.Categorical(logits=logits)
        pi = tfp.distributions.Categorical(logits=logits)
        return v, pi


class CategoricalSeparateMLP(nn.Module):
    """Split Actor-Critic Architecture for PPO."""

    num_output_units: int
    num_hidden_units: int
    num_hidden_layers: int
    scale: float
    high: jnp.ndarray
    low: jnp.ndarray
    prefix_actor: str = "actor"
    prefix_critic: str = "critic"
    model_name: str = "separate-mlp"
    flatten_2d: bool = False  # Catch case
    flatten_3d: bool = False  # Rooms/minatar case
    # count_switch: int = int(2e3)

    @nn.compact
    def __call__(self, x, rng):
        # Flatten a single 2D image
        if self.flatten_2d and len(x.shape) == 2:
            x = x.reshape(-1)
        # Flatten a batch of 2d images into a batch of flat vectors
        if self.flatten_2d and len(x.shape) > 2:
            x = x.reshape(x.shape[0], -1)

        # Flatten a single 3D image
        if self.flatten_3d and len(x.shape) == 3:
            x = x.reshape(-1)
        # Flatten a batch of 3d images into a batch of flat vectors
        if self.flatten_3d and len(x.shape) > 3:
            x = x.reshape(x.shape[0], -1)

        x = (x - self.low) / (self.high - self.low)
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

        x_a = nn.relu(nn.Dense(self.num_hidden_units, bias_init=default_mlp_init(),)(x))
        # Loop over rest of intermediate hidden layers
        for i in range(1, self.num_hidden_layers):
            x_a = nn.relu(
                nn.Dense(self.num_hidden_units, bias_init=default_mlp_init(),)(x_a)
            )
        logits = nn.Dense(self.num_output_units, bias_init=default_mlp_init(),)(x_a)
        # pi = distrax.Categorical(logits=logits)
        pi = tfp.distributions.Categorical(logits=logits)
        return v, pi


# class CategoricalSeparateMLP(nn.Module):
#     """Split Actor-Critic Architecture for PPO."""

#     num_output_units: int
#     num_hidden_units: int
#     num_hidden_layers: int
#     scale: float
#     high_freq_multiplier: float
#     prefix_actor: str = "actor"
#     prefix_critic: str = "critic"
#     model_name: str = "separate-mlp"
#     flatten_2d: bool = False  # Catch case
#     flatten_3d: bool = False  # Rooms/minatar case
#     count_switch: int = int(2e3)

#     @nn.compact
#     def __call__(self, x, counts, high_low_or_mixed, rng):
#         if self.flatten_2d and len(x.shape) == 2:
#             x = x.reshape(-1)
#         if self.flatten_2d and len(x.shape) > 2:
#             x = x.reshape(x.shape[0], -1)
#         if self.flatten_3d and len(x.shape) == 3:
#             x = x.reshape(-1)
#         if self.flatten_3d and len(x.shape) > 3:
#             x = x.reshape(x.shape[0], -1)

#         if len(x.shape) > 1:
#             positions = x[:, :2].astype(int)
#             extracted_counts = counts[positions[:, 0], positions[:, 1]]
#         else:
#             positions = x[:2].astype(int)
#             extracted_counts = counts[positions[0], positions[1]]

#         x = (x / 13) - 0.5
#         if len(x.shape) > 1:
#             low_frequency = jnp.copy(x)
#             low_frequency = LFF(
#                 num_output_features=self.num_hidden_units,
#                 num_input_features=x.shape[-1],
#                 name=self.prefix_critic + "_fc_1_low_frequency",
#                 scale=self.scale,
#             )(low_frequency)
#             high_frequency = jnp.copy(x)
#             high_frequency = LFF(
#                 num_output_features=self.num_hidden_units,
#                 num_input_features=x.shape[-1],
#                 name=self.prefix_critic + "_fc_1_high_frequency",
#                 scale=self.scale * self.high_freq_multiplier,
#             )(high_frequency)

#         else:
#             low_frequency = LFF(
#                 num_output_features=self.num_hidden_units,
#                 num_input_features=x.shape[-1],
#                 name=self.prefix_critic + "_fc_1_low_frequency",
#                 scale=self.scale,
#             )(x)
#             high_frequency = LFF(
#                 num_output_features=self.num_hidden_units,
#                 num_input_features=x.shape[-1],
#                 name=self.prefix_critic + "_fc_1_high_frequency",
#                 scale=self.scale * self.high_freq_multiplier,
#             )(x)

#         # for i in range(1, self.num_hidden_layers):
#         x_v_high_frequency = nn.relu(
#             nn.Dense(
#                 self.num_hidden_units,
#                 name=self.prefix_critic + "_fc_2_high_frequency",
#                 bias_init=default_mlp_init(),
#             )(high_frequency)
#         )
#         x_v_low_frequency = nn.relu(
#             nn.Dense(
#                 self.num_hidden_units,
#                 name=self.prefix_critic + "_fc_2_low_frequency",
#                 bias_init=default_mlp_init(),
#             )(low_frequency)
#         )
#         v_low_frequency = nn.Dense(
#             1,
#             name=self.prefix_critic + "_fc_3_low_frequency",
#             bias_init=default_mlp_init(),
#         )(x_v_low_frequency)

#         v_high_frequency = nn.Dense(
#             1,
#             name=self.prefix_critic + "_fc_3_high_frequency",
#             bias_init=default_mlp_init(),
#         )(x_v_high_frequency)

#         if len(x.shape) == 1:
#             v_low = jnp.copy(v_low_frequency)
#             v_high = jnp.copy(v_high_frequency)
#             v_low_frequency = v_low_frequency * (
#                 extracted_counts < self.count_switch
#             ).astype(int)
#             v_high_frequency = v_high_frequency * (
#                 extracted_counts > self.count_switch
#             ).astype(int)
#             v_mixed = v_high_frequency + v_low_frequency
#             v = (
#                 high_low_or_mixed[0] * v_high
#                 + high_low_or_mixed[1] * v_low
#                 + high_low_or_mixed[2] * v_mixed
#             )
#         else:
#             v_low = jnp.copy(v_low_frequency)
#             v_high = jnp.copy(v_high_frequency)

#             v_low_frequency = v_low_frequency * jnp.expand_dims(
#                 (extracted_counts < self.count_switch).astype(int), 1
#             )
#             v_high_frequency = v_high_frequency * jnp.expand_dims(
#                 (extracted_counts > self.count_switch).astype(int), 1
#             )
#             v_mixed = v_high_frequency + v_low_frequency
#             v = (
#                 high_low_or_mixed[0] * v_high
#                 + high_low_or_mixed[1] * v_low
#                 + high_low_or_mixed[2] * v_mixed
#             )

#         # if len(x.shape) > 1:
#         #     x = x
#         # else:
#         #     x = x

#         # x_a = nn.relu(nn.Dense(self.num_hidden_units, bias_init=default_mlp_init(),)(x))
#         # v_a = nn.Dense(self.num_hidden_units, bias_init=default_mlp_init(),)(x_a)
#         # x_a = LFF(
#         #     num_output_features=self.num_hidden_units,
#         #     num_input_features=x.shape[-1],
#         #     name=self.prefix_critic + "_fc_1_low_frequency_initial",
#         #     scale=self.scale * self.high_freq_multiplier,
#         # )(x)
#         # for i in range(1, self.num_hidden_layers):
#         #     x_a = LFF(
#         #         num_output_features=self.num_hidden_units,
#         #         num_input_features=x.shape[-1],
#         #         name=self.prefix_critic + f"_fc_1_low_frequency_{i}",
#         #         scale=self.scale * self.high_freq_multiplier,
#         #     )(x_a)
#         # logits = nn.Dense(self.num_output_units, bias_init=default_mlp_init(),)(x_a)
#         # pi = tfp.distributions.Categorical(logits=logits)
#         # return v, pi

#         # x_a = nn.relu(nn.Dense(self.num_hidden_units, bias_init=default_mlp_init(),)(x))
#         x_a = LFF(
#             num_output_features=self.num_hidden_units,
#             num_input_features=x.shape[-1],
#             name=self.prefix_critic + "_fc_1_action",
#             scale=self.scale * self.high_freq_multiplier,
#         )(x)
#         # # Loop over rest of intermediate hidden layers
#         for i in range(1, self.num_hidden_layers):
#             # x_a = LFF(
#             #     num_output_features=self.num_hidden_units,
#             #     num_input_features=x.shape[-1],
#             #     name=self.prefix_critic + f"_fc_{i+1}_action",
#             #     scale=self.scale * self.high_freq_multiplier,
#             # )(x_a)
#             x_a = nn.relu(
#                 nn.Dense(self.num_hidden_units, bias_init=default_mlp_init(),)(x_a)
#             )
#         logits = nn.Dense(self.num_output_units, bias_init=default_mlp_init(),)(x_a)
#         # pi = distrax.Categorical(logits=logits)
#         pi = tfp.distributions.Categorical(logits=logits)
#         return v, pi

#         # Flatten a single 2D image
#         # if self.flatten_2d and len(x.shape) == 2:
#         #     x = x.reshape(-1)
#         # # Flatten a batch of 2d images into a batch of flat vectors
#         # if self.flatten_2d and len(x.shape) > 2:
#         #     x = x.reshape(x.shape[0], -1)

#         # # Flatten a single 3D image
#         # if self.flatten_3d and len(x.shape) == 3:
#         #     x = x.reshape(-1)
#         # # Flatten a batch of 3d images into a batch of flat vectors
#         # if self.flatten_3d and len(x.shape) > 3:
#         #     x = x.reshape(x.shape[0], -1)

#         # if len(x.shape) > 1:
#         #     x = x[:, :2]
#         # else:
#         #     x = x[:2]
#         # x_v = nn.relu(
#         #     nn.Dense(
#         #         self.num_hidden_units,
#         #         name=self.prefix_critic + "_fc_1",
#         #         bias_init=default_mlp_init(),
#         #     )(x)
#         # )
#         # # Loop over rest of intermediate hidden layers
#         # for i in range(1, self.num_hidden_layers):
#         #     x_v = nn.relu(
#         #         nn.Dense(
#         #             self.num_hidden_units,
#         #             name=self.prefix_critic + f"_fc_{i+1}",
#         #             bias_init=default_mlp_init(),
#         #         )(x_v)
#         #     )
#         # v = nn.Dense(
#         #     1, name=self.prefix_critic + "_fc_v", bias_init=default_mlp_init(),
#         # )(x_v)

#         # x_a = nn.relu(nn.Dense(self.num_hidden_units, bias_init=default_mlp_init(),)(x))
#         # # Loop over rest of intermediate hidden layers
#         # for i in range(1, self.num_hidden_layers):
#         #     x_a = nn.relu(
#         #         nn.Dense(self.num_hidden_units, bias_init=default_mlp_init(),)(x_a)
#         #     )
#         # logits = nn.Dense(self.num_output_units, bias_init=default_mlp_init(),)(x_a)
#         # # pi = distrax.Categorical(logits=logits)
#         # pi = tfp.distributions.Categorical(logits=logits)
#         # return v, pi


class GaussianSeparateMLPSIREN(nn.Module):
    """Split Actor-Critic Architecture for PPO."""

    num_output_units: int
    num_hidden_units: int
    num_hidden_layers: int
    scale: float
    high: jnp.ndarray
    low: jnp.ndarray
    prefix_actor: str = "actor"
    prefix_critic: str = "critic"
    min_std: float = 0.001
    model_name: str = "separate-mlp"

    @nn.compact
    def __call__(self, x, rng):
        # x_v = nn.relu(
        #     nn.Dense(
        #         self.num_hidden_units,
        #         name=self.prefix_critic + "_fc_1",
        #         bias_init=default_mlp_init(),
        #     )(x)
        # )

        x = (x - self.low) / (self.high - self.low)
        x_v = LFF(
            num_output_features=self.num_hidden_units,
            num_input_features=x.shape[-1],
            name="lff_critic",
            scale=self.scale,
        )(x)
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

        x_a = LFF(
            num_output_features=self.num_hidden_units,
            num_input_features=x.shape[-1],
            name="lff_actor",
            scale=self.scale,
        )(x)
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


class GaussianSeparateMLP(nn.Module):
    """Split Actor-Critic Architecture for PPO."""

    num_output_units: int
    num_hidden_units: int
    num_hidden_layers: int
    scale: float
    high: jnp.ndarray
    low: jnp.ndarray
    prefix_actor: str = "actor"
    prefix_critic: str = "critic"
    min_std: float = 0.001
    model_name: str = "separate-mlp"

    @nn.compact
    def __call__(self, x, rng):
        x = (x - self.low) / (self.high - self.low)
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

        # Loop over rest of intermediate hidden layers
        x_a = nn.relu(
            nn.Dense(
                self.num_hidden_units,
                name=self.prefix_actor + "_fc_1",
                bias_init=default_mlp_init(),
            )(x)
        )
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

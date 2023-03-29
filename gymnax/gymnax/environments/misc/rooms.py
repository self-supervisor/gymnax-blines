# import jax
# import jax.numpy as jnp
# from jax import lax
# from gymnax.environments import environment, spaces
# from typing import Tuple, Optional, List
# import chex
# from flax import struct


# @struct.dataclass
# class EnvState:
#     pos: chex.Array
#     goal: chex.Array
#     time: int


# @struct.dataclass
# class EnvParams:
#     fail_prob: float = 0.0  # 1.0 / 3
#     resample_init_pos: bool = True
#     resample_goal_pos: bool = False
#     max_steps_in_episode: int = 500


# four_rooms_map = """
# xxxxxxxxxxxxxxxxxxxxxxxx
# x               x      x
# x                      x
# x               x      x
# x               x      x
# x               x      x
# xx xxxxxxxxxxxxxxxxxx xx
# x               xxxx  xx
# x               xxxx xxx
# x               xxxx  xx
# x                   x xx
# x                     xx
# xxxxxxxxxxxxxxxxxxxxxxxx"""
# # four_rooms_map = """
# # xxxxxxxxxxxxx
# # x     x     x
# # x     x     x
# # x           x
# # x     x     x
# # x     x     x
# # xx xxxx     x
# # x     xxx xxx
# # x     x     x
# # x     x     x
# # x           x
# # x     x     x
# # xxxxxxxxxxxxx"""

# train_four_rooms_map = """
# xxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxx xx
# xxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxx"""

# test_four_rooms_map = """
# xxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxx
# x xxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxx"""

# # test_four_rooms_map = """
# # xxxxxxxxxxxxx
# # xxxxxxxxxxxxx
# # xxxxxxxxxxxxx
# # xxxxxxxxxxxxx
# # xxxxxxxxxxxxx
# # xxxxxxxxxxxxx
# # xxxxxxxxxxxxx
# # x     xxxxxxx
# # x     xxxxxxx
# # x     xxxxxxx
# # x     xxxxxxx
# # x     xxxxxxx
# # xxxxxxxxxxxxx"""


# def string_to_bool_map(str_map: str) -> chex.Array:
#     """Convert string map into boolean walking map."""
#     bool_map = []
#     for row in str_map.split("\n")[1:]:
#         bool_map.append([r == " " for r in row])
#     return jnp.array(bool_map)


# class FourRooms(environment.Environment):
#     """
#     JAX Compatible version of Four Rooms environment (Sutton et al., 1999).
#     Source: Comparable to https://github.com/howardh/gym-fourrooms
#     Since gymnax automatically resets env at done, we abstract different resets
#     """

#     def __init__(
#         self,
#         use_visual_obs: bool = False,
#         goal_fixed: List[int] = [10, 21],
#         pos_fixed: List[int] = [4, 1],
#     ):
#         super().__init__()
#         self.env_map = string_to_bool_map(four_rooms_map)
#         self.train_map = string_to_bool_map(train_four_rooms_map)
#         self.test_map = string_to_bool_map(test_four_rooms_map)
#         self.occupied_map = 1 - self.env_map
#         coords = []
#         train_indices = []
#         test_indices = []
#         count = 0

#         for y in range(self.env_map.shape[0]):
#             for x in range(self.env_map.shape[1]):
#                 if self.env_map[y, x]:  # If it's an open space
#                     coords.append([y, x])
#                 if self.train_map[y, x]:
#                     train_indices.append([y, x])
#                 if self.test_map[y, x]:
#                     test_indices.append([y, x])
#                 count += 1

#         self.coords = jnp.array(coords)
#         self.train_indices = jnp.array(train_indices)
#         self.test_indices = jnp.array(test_indices)
#         self.counts = jnp.zeros_like(jnp.array(coords))
#         self.directions = jnp.array([[-1, 0], [0, 1], [1, 0], [0, -1]])

#         # Any open space in the map can be a goal for the agent
#         self.available_goals = self.coords

#         # Whether to use 3D visual observation
#         # Channel ID 0 - Wall (1) or not occupied (0)
#         # Channel ID 1 - Agent location in maze
#         self.use_visual_obs = use_visual_obs

#         # Set fixed goal and position if we dont resample each time
#         self.goal_fixed = jnp.array(goal_fixed)
#         self.pos_fixed = jnp.array(pos_fixed)

#     @property
#     def default_params(self) -> EnvParams:
#         # Default environment parameters
#         return EnvParams()

#     def step_env(
#         self, key: chex.PRNGKey, state: EnvState, action: int, params: EnvParams
#     ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
#         """Perform single timestep state transition."""
#         key_random, key_action = jax.random.split(key)
#         # Sample whether to choose a random action
#         choose_random = jax.random.uniform(key_random, ()) < params.fail_prob * 4 / 3
#         action = jax.lax.select(
#             choose_random, self.action_space(params).sample(key_action), action
#         )

#         p = state.pos + self.directions[action]
#         in_map = self.env_map[p[0], p[1]]
#         new_pos = jax.lax.select(in_map, p, state.pos)
#         in_lava = self.env_map[p[0], p[1]]  # self.env_map[new_pos[0], new_pos[1]]
#         reward = (
#             100
#             * jnp.logical_and(new_pos[0] == state.goal[0], new_pos[1] == state.goal[1])
#             - 1
#             + in_lava * -5
#         )

#         # Update state dict and evaluate termination conditions
#         state = EnvState(new_pos, state.goal, state.time + 1)
#         done = self.is_terminal(state, params)
#         return (
#             lax.stop_gradient(self.get_obs(state)),
#             lax.stop_gradient(state),
#             reward,
#             done,
#             {"discount": self.discount(state, params)},
#         )

#         # key_random, key_action = jax.random.split(key)
#         # # Sample whether to choose a random action
#         # choose_random = jax.random.uniform(key_random, ()) < params.fail_prob * 4 / 3
#         # action = jax.lax.select(
#         #     choose_random, self.action_space(params).sample(key_action), action
#         # )

#         # p = state.pos + self.directions[action]
#         # in_map = self.env_map[p[0], p[1]]
#         # new_pos = jax.lax.select(in_map, p, state.pos)
#         # self.counts = self.counts.at[new_pos[0], new_pos[1]].add(1)

#         # reward = (
#         #     # -0.1
#         #     # + 10
#         #     jnp.logical_and(new_pos[0] == state.goal[0], new_pos[1] == state.goal[1])
#         #     # + -0.1 * in_lava
#         # )
#         # # reward = -0.1 * self.counts[new_pos[0], new_pos[1]]

#         # # Update state dict and evaluate termination conditions
#         # state = EnvState(new_pos, state.goal, state.time + 1)

#         # done = self.is_terminal(state, params)
#         # return (
#         #     lax.stop_gradient(self.get_obs(state)),
#         #     lax.stop_gradient(state),
#         #     reward,
#         #     done,
#         #     {"discount": self.discount(state, params)},
#         # )

#     def reset_env(
#         self, key: chex.PRNGKey, training: int, params: EnvParams,
#     ) -> Tuple[chex.Array, EnvState]:
#         """Reset environment state by sampling initial position."""
#         # Reset both the agents position and the goal location
#         rng_goal, rng_pos = jax.random.split(key, 2)
#         goal_new = reset_goal(rng_goal, self.available_goals, params)
#         # Only use resampled position if specified in EnvParams
#         goal = jax.lax.select(params.resample_goal_pos, goal_new, self.goal_fixed)

#         pos_new = reset_pos(
#             rng_pos, self.coords, self.train_indices, self.test_indices, goal, training
#         )
#         pos = jax.lax.select(params.resample_init_pos, pos_new, self.pos_fixed)
#         # jax.debug.print("params.resample_init_pos {}", params.resample_init_pos)
#         # jax.debug.print("pos {}", pos)
#         state = EnvState(pos, goal, 0)
#         return self.get_obs(state), state

#     def get_obs(self, state: EnvState) -> chex.Array:
#         """Return observation from raw state trafo."""
#         if not self.use_visual_obs:
#             return jnp.array(
#                 [state.pos[0], state.pos[1], state.goal[0], state.goal[1],]
#             )
#         else:
#             agent_map = jnp.zeros(self.occupied_map.shape)
#             agent_map = agent_map.at[state.pos[1], state.pos[0]].set(1)
#             obs_array = jnp.stack([self.occupied_map, agent_map], axis=2)
#             return obs_array

#     # def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
#     #     """Check whether state is terminal."""
#     #     # Check number of steps in episode termination condition
#     #     done_steps = state.time >= params.max_steps_in_episode
#     #     # Check if agent has found the goal
#     #     done_goal = jnp.logical_and(
#     #         state.pos[0] == state.goal[0], state.pos[1] == state.goal[1],
#     #     )
#     #     done_not_dead = jnp.logical_or(done_goal, done_steps)
#     #     in_map = self.env_map[state.pos[0], state.pos[1]]
#     #     done_dead = jax.lax.select(in_map, True, False)
#     #     done = jnp.logical_or(done_not_dead, done_dead)
#     #     return done_goal
#     def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
#         """Check whether state is terminal."""
#         # Check number of steps in episode termination condition
#         done_steps = state.time >= params.max_steps_in_episode
#         # Check if agent has found the goal
#         done_goal = jnp.logical_and(
#             state.pos[0] == state.goal[0], state.pos[1] == state.goal[1],
#         )
#         done = jnp.logical_or(done_goal, done_steps)
#         return done

#     def get_counts(self) -> chex.Array:
#         """Return count frequencies of agent positions."""
#         return self.counts

#     @property
#     def name(self) -> str:
#         """Environment name."""
#         return "FourRooms-misc"

#     @property
#     def num_actions(self) -> int:
#         """Number of actions possible in environment."""
#         return 4

#     def action_space(self, params: Optional[EnvParams] = None) -> spaces.Discrete:
#         """Action space of the environment."""
#         return spaces.Discrete(4)

#     def observation_space(self, params: EnvParams) -> spaces.Box:
#         """Observation space of the environment."""
#         if self.use_visual_obs:
#             return spaces.Box(0, 1, (13, 23, 2), jnp.float32)
#         else:
#             return spaces.Box(
#                 jnp.min(self.coords), jnp.max(self.coords), (4,), jnp.float32
#             )

#     def state_space(self, params: EnvParams) -> spaces.Dict:
#         """State space of the environment."""
#         return spaces.Dict(
#             {
#                 "pos": spaces.Box(
#                     jnp.min(self.coords), jnp.max(self.coords), (2,), jnp.float32,
#                 ),
#                 "goal": spaces.Box(
#                     jnp.min(self.coords), jnp.max(self.coords), (2,), jnp.float32,
#                 ),
#                 "time": spaces.Discrete(params.max_steps_in_episode),
#             }
#         )

#     def render(self, state: EnvState, params: EnvParams):
#         """Small utility for plotting the agent's state."""
#         import matplotlib.pyplot as plt

#         fig, ax = plt.subplots()
#         ax.imshow(self.occupied_map, cmap="Greys")
#         ax.annotate(
#             "A",
#             fontsize=20,
#             xy=(state.pos[1], state.pos[0]),
#             xycoords="data",
#             xytext=(state.pos[1] - 0.3, state.pos[0] + 0.25),
#         )
#         ax.annotate(
#             "G",
#             fontsize=20,
#             xy=(state.goal[1], state.goal[0]),
#             xycoords="data",
#             xytext=(state.goal[1] - 0.3, state.goal[0] + 0.25),
#         )
#         ax.set_xticks([])
#         ax.set_yticks([])
#         return fig, ax


# def reset_goal(
#     rng: chex.PRNGKey, available_goals: chex.Array, params: EnvParams
# ) -> chex.Array:
#     """Reset the goal state/position in the environment."""
#     goal_index = jax.random.randint(rng, (), 0, available_goals.shape[0])
#     goal = available_goals[goal_index][:]
#     return goal


# # def reset_pos(rng: chex.PRNGKey, coords: chex.Array, goal: chex.Array) -> chex.Array:
# #     """Reset the position of the agent."""
# # pos_index = jax.random.randint(rng, (), 0, coords.shape[0] - 1)
# # collision = jnp.logical_and(
# #     coords[pos_index][0] == goal[0], coords[pos_index][1] == goal[1]
# # )
# # pos_index = jax.lax.select(collision, coords.shape[0] - 1, pos_index)
# # return coords[pos_index][:]


# def reset_pos(
#     rng: chex.PRNGKey,
#     coords: chex.Array,
#     train_indices: chex.Array,
#     test_indices: chex.Array,
#     goal: chex.Array,
#     training: chex.Array,
# ) -> chex.Array:
#     """Reset the position of the agent."""
#     rng, rng_train, rng_test = jax.random.split(rng, 3)
#     train_pos = jax.random.choice(rng_train, train_indices)
#     test_pos = jax.random.choice(rng_test, test_indices)
#     pos_index = jax.lax.select(training == 1, train_pos, test_pos)
#     collision = jnp.logical_and(pos_index[0] == goal[0], pos_index[1] == goal[1])
#     pos_index = jax.lax.select(collision, goal - 1, pos_index)
#     return pos_index

#     # pos_index = jax.random.randint(rng, (), 0, coords.shape[0] - 1)
#     # collision = jnp.logical_and(
#     #     coords[pos_index][0] == goal[0], coords[pos_index][1] == goal[1]
#     # )
#     # pos_index = jax.lax.select(collision, coords.shape[0] - 1, pos_index)
#     # return coords[pos_index][:]

import jax
import jax.numpy as jnp
from jax import lax
from gymnax.environments import environment, spaces
from typing import Tuple, Optional, List
import chex
from flax import struct


@struct.dataclass
class EnvState:
    pos: chex.Array
    goal: chex.Array
    time: int


@struct.dataclass
class EnvParams:
    fail_prob: float = 0.0  # 1.0 / 3
    resample_init_pos: bool = True
    resample_goal_pos: bool = False
    max_steps_in_episode: int = 500


four_rooms_map = """
xxxxxxxxxxxxx
x     x     x
x     x     x
x           x
x     x     x
x     x     x
xx xxxx     x
x     xxx xxx
x     x     x
x     x     x
x           x
x     x     x
xxxxxxxxxxxxx"""

train_four_rooms_map = """
xxxxxxxxxxxxx
xxxxxxxxxxxxx
xxxxxxxxxxxxx
xxxxxxxxxxxxx
xxxxxxxxxxxxx
xxxxxxxxxxxxx
xxxxxxxxxxxxx
xxxxxxxxxxxxx
xxxxxxxxxxxxx
xxxxxxxxxxxxx
xxxxxxxxx xxx
xxxxxxxxxxxxx
xxxxxxxxxxxxx"""

test_four_rooms_map = """
xxxxxxxxxxxxx
x     x     x
x     x     x
x           x
x     x     x
x     x     x
xx xxxx     x
x     xxx xxx
x     x     x
x     x     x
x           x
x     x     x
xxxxxxxxxxxxx"""


def string_to_bool_map(str_map: str) -> chex.Array:
    """Convert string map into boolean walking map."""
    bool_map = []
    for row in str_map.split("\n")[1:]:
        bool_map.append([r == " " for r in row])
    return jnp.array(bool_map)


class FourRooms(environment.Environment):
    """
    JAX Compatible version of Four Rooms environment (Sutton et al., 1999).
    Source: Comparable to https://github.com/howardh/gym-fourrooms
    Since gymnax automatically resets env at done, we abstract different resets
    """

    def __init__(
        self,
        use_visual_obs: bool = False,
        goal_fixed: List[int] = [1, 11],
        pos_fixed: List[int] = [4, 1],
    ):
        super().__init__()
        self.env_map = string_to_bool_map(four_rooms_map)
        self.train_map = string_to_bool_map(train_four_rooms_map)
        self.test_map = string_to_bool_map(test_four_rooms_map)
        self.occupied_map = 1 - self.env_map
        coords = []
        train_indices = []
        test_indices = []
        goal_indices = []
        count = 0

        for y in range(self.env_map.shape[0]):
            for x in range(self.env_map.shape[1]):
                if self.env_map[y, x]:  # If it's an open space
                    coords.append([y, x])
                if self.train_map[y, x]:
                    train_indices.append([y, x])
                if self.test_map[y, x]:
                    test_indices.append([y, x])
                if
                count += 1

        self.coords = jnp.array(coords)
        self.train_indices = jnp.array(train_indices)
        self.test_indices = jnp.array(test_indices)
        self.counts = jnp.zeros_like(jnp.array(coords))
        self.directions = jnp.array([[-1, 0], [0, 1], [1, 0], [0, -1]])

        # Any open space in the map can be a goal for the agent
        self.available_goals = self.coords

        # Whether to use 3D visual observation
        # Channel ID 0 - Wall (1) or not occupied (0)
        # Channel ID 1 - Agent location in maze
        self.use_visual_obs = use_visual_obs

        # Set fixed goal and position if we dont resample each time
        self.goal_fixed = jnp.array(goal_fixed)
        self.pos_fixed = jnp.array(pos_fixed)

    @property
    def default_params(self) -> EnvParams:
        # Default environment parameters
        return EnvParams()

    def step_env(
        self, key: chex.PRNGKey, state: EnvState, action: int, params: EnvParams
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        """Perform single timestep state transition."""
        key_random, key_action = jax.random.split(key)
        # Sample whether to choose a random action
        choose_random = jax.random.uniform(key_random, ()) < params.fail_prob * 4 / 3
        action = jax.lax.select(
            choose_random, self.action_space(params).sample(key_action), action
        )

        p = state.pos + self.directions[action]
        in_map = self.env_map[p[0], p[1]]
        new_pos = jax.lax.select(in_map, p, state.pos)
        in_lava = self.env_map[p[0], p[1]]  # self.env_map[new_pos[0], new_pos[1]]
        self.counts = self.counts.at[new_pos[0], new_pos[1]].add(1)

        reward = -0.1 + 10 * jnp.logical_and(
            new_pos[0] == state.goal[0], new_pos[1] == state.goal[1]
        )
        # reward = -0.1 * self.counts[new_pos[0], new_pos[1]]

        # Update state dict and evaluate termination conditions
        state = EnvState(new_pos, state.goal, state.time + 1)
        done = self.is_terminal(state, params)
        return (
            lax.stop_gradient(self.get_obs(state)),
            lax.stop_gradient(state),
            reward,
            done,
            {"discount": self.discount(state, params)},
        )

    def reset_env(
        self, key: chex.PRNGKey, training: int, params: EnvParams,
    ) -> Tuple[chex.Array, EnvState]:
        """Reset environment state by sampling initial position."""
        # Reset both the agents position and the goal location
        rng_goal, rng_pos = jax.random.split(key, 2)
        goal_new = reset_goal(rng_goal, self.available_goals, params)
        # Only use resampled position if specified in EnvParams
        goal = jax.lax.select(params.resample_goal_pos, goal_new, self.goal_fixed)

        pos_new = reset_pos(
            rng_pos, self.coords, self.train_indices, self.test_indices, goal, training
        )

        pos = jax.lax.select(params.resample_init_pos, pos_new, self.pos_fixed)
        state = EnvState(pos, goal, 0)
        return self.get_obs(state), state

    def get_obs(self, state: EnvState) -> chex.Array:
        """Return observation from raw state trafo."""
        if not self.use_visual_obs:
            return jnp.array(
                [state.pos[0], state.pos[1], state.goal[0], state.goal[1],]
            )
        else:
            agent_map = jnp.zeros(self.occupied_map.shape)
            agent_map = agent_map.at[state.pos[1], state.pos[0]].set(1)
            obs_array = jnp.stack([self.occupied_map, agent_map], axis=2)
            return obs_array

    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        """Check whether state is terminal."""
        # Check number of steps in episode termination condition
        done_steps = state.time >= params.max_steps_in_episode
        # Check if agent has found the goal
        done_goal = jnp.logical_and(
            state.pos[0] == state.goal[0], state.pos[1] == state.goal[1],
        )
        # done_not_dead = jnp.logical_or(done_goal, done_steps)
        # in_map = self.env_map[state.pos[0], state.pos[1]]
        # done_dead = jax.lax.select(in_map, True, False)
        done = jnp.logical_or(done_goal, done_steps)
        return done

    def get_counts(self) -> chex.Array:
        """Return count frequencies of agent positions."""
        return self.counts

    @property
    def name(self) -> str:
        """Environment name."""
        return "FourRooms-misc"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return 4

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Discrete(4)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        if self.use_visual_obs:
            return spaces.Box(0, 1, (13, 13, 2), jnp.float32)
        else:
            return spaces.Box(
                jnp.min(self.coords), jnp.max(self.coords), (4,), jnp.float32
            )

    def state_space(self, params: EnvParams) -> spaces.Dict:
        """State space of the environment."""
        return spaces.Dict(
            {
                "pos": spaces.Box(
                    jnp.min(self.coords), jnp.max(self.coords), (2,), jnp.float32,
                ),
                "goal": spaces.Box(
                    jnp.min(self.coords), jnp.max(self.coords), (2,), jnp.float32,
                ),
                "time": spaces.Discrete(params.max_steps_in_episode),
            }
        )

    def render(self, state: EnvState, params: EnvParams):
        """Small utility for plotting the agent's state."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.imshow(self.occupied_map, cmap="Greys")
        ax.annotate(
            "A",
            fontsize=20,
            xy=(state.pos[1], state.pos[0]),
            xycoords="data",
            xytext=(state.pos[1] - 0.3, state.pos[0] + 0.25),
        )
        ax.annotate(
            "G",
            fontsize=20,
            xy=(state.goal[1], state.goal[0]),
            xycoords="data",
            xytext=(state.goal[1] - 0.3, state.goal[0] + 0.25),
        )
        ax.set_xticks([])
        ax.set_yticks([])
        return fig, ax


def reset_goal(
    rng: chex.PRNGKey, available_goals: chex.Array, params: EnvParams
) -> chex.Array:
    """Reset the goal state/position in the environment."""
    goal_index = jax.random.randint(rng, (), 0, available_goals.shape[0])
    goal = available_goals[goal_index][:]
    return goal


def reset_pos(
    rng: chex.PRNGKey,
    coords: chex.Array,
    train_indices: chex.Array,
    test_indices: chex.Array,
    goal: chex.Array,
    training: chex.Array,
) -> chex.Array:
    """Reset the position of the agent."""
    train_pos = jax.random.choice(rng, train_indices)
    test_pos = jax.random.choice(rng, test_indices)
    pos_index = jax.lax.select(training == 1, train_pos, test_pos)
    collision = jnp.logical_and(train_pos[0] == goal[0], test_pos[0] == goal[1])
    pos_index = jax.lax.select(collision, goal, pos_index)
    return pos_index

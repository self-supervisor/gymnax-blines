import numpy as np
import matplotlib.pyplot as plt
import imageio
import chex
import timeit


def init_minatar(ax, obs, state):
    import seaborn as sns
    import matplotlib.colors as colors

    n_channels = obs.shape[-1]
    # The seaborn color_palette cubhelix is used to assign visually distinct colors to each channel for the env
    cmap = sns.color_palette("cubehelix", n_channels)
    cmap.insert(0, (0, 0, 0))
    cmap = colors.ListedColormap(cmap)
    bounds = [i for i in range(n_channels + 2)]
    norm = colors.BoundaryNorm(bounds, n_channels + 2)
    numerical_state = (
        np.amax(obs * np.reshape(np.arange(n_channels) + 1, (1, 1, -1)), 2) + 0.5
    )
    # ax.set_xticks([])
    # ax.set_yticks([])
    # return ax.imshow(
    #     numerical_state, cmap=cmap, norm=norm, interpolation="none"
    # ), numerical_state
    return numerical_state


def update_minatar(im, obs):
    n_channels = obs.shape[-1]
    numerical_state = (
        np.amax(obs * np.reshape(np.arange(n_channels) + 1, (1, 1, -1)), 2) + 0.5
    )
    im.set_data(numerical_state)
    return numerical_state


def make_gif():
    title_str_list = ["ground_truth_obs", "pred_obs"]
    all_frames_list = []
    for title_str in title_str_list:
        filename_list = []
        numerical_states = []
        frames = np.load(f"{title_str}.npy")

        for i in range(len(frames)):
            numerical_states.append(init_minatar(plt.gca(), frames[i], 0))

        for i in range(len(numerical_states)):
            plt.imshow(numerical_states[i])
            filename = f"{title_str}_{i}.png"
            plt.savefig(filename)
            filename_list.append(filename)
            plt.close()

        images = [imageio.imread(filename) for filename in filename_list]

        imageio.mimsave(f"animation_{title_str}.gif", images, duration=0.1)
        all_frames_list.append(numerical_states)
    return all_frames_list


class TimeIt:
    def __init__(self, tag, frames=None):
        self.tag = tag
        self.frames = frames

    def __enter__(self):
        self.start = timeit.default_timer()
        return self

    def __exit__(self, *args):
        self.elapsed_secs = timeit.default_timer() - self.start
        msg = self.tag + (": Elapsed time=%.2fs" % self.elapsed_secs)
        if self.frames:
            msg += ", FPS=%.2e" % (self.frames / self.elapsed_secs)
        print(msg)


@chex.dataclass(frozen=True)
class TimeStep:
    q_values: chex.Array
    action: chex.Array
    discount: chex.Array
    reward: chex.Array
    pred_next_obs: chex.Array
    pred_prev_obs: chex.Array
    actual_obs: chex.Array
    prev_obs: chex.Array

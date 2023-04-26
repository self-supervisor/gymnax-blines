from utils.helpers import load_config, save_pkl_object, get_perfect_params_from_pickle
from utils.models import get_model_ready
import wandb
import numpy as np
from distutils.util import strtobool
import jax

print(jax.devices())


def main(
    config,
    mle_log,
    scale,
    SIRENs,
    # count_switch,
    # high_freq_multiplier,
    log_ext="",
    use_wandb: bool = False,
    save_checkpoint: bool = False,
):
    """Run training with ES or PPO. Store logs and agent ckpt."""
    rng = jax.random.PRNGKey(config.seed_id)
    # Setup the model architecture
    rng, rng_init = jax.random.split(rng)
    perfect_network_params = get_perfect_params_from_pickle(
        filepath=f"agents/{config.env_name}/{config.train_type.lower()}{log_ext}.pkl"
    )
    perfect_network, _ = get_model_ready(rng_init, config, scale, force_ReLU=True)

    config.scale = scale
    config.SIRENs = SIRENs
    model, params = get_model_ready(
        rng_init, config, scale  # , count_switch, high_freq_multiplier
    )

    # config.count_switch = count_switch
    # config.high_freq_multiplier = high_freq_multiplier
    if use_wandb:
        wandb.init(config=config, project="gymnax")

    # Run the training loop (either evosax ES or PPO)
    if config.train_type == "ES":
        from utils.es import train_es as train_fn
    elif config.train_type == "PPO":
        from utils.ppo import train_ppo as train_fn
    else:
        raise ValueError("Unknown train_type. Has to be in ('ES', 'PPO').")

    # Log and store the results.
    (
        log_steps,
        log_return_train,
        log_kl_div,
        log_MSE,
        log_mean_abs_critic,
        log_mean_abs_actor,
        log_mean_RMS_critic,
        log_mean_RMS_actor,
        # log_counts,
        network_ckpt,
        train_state,
    ) = train_fn(
        rng,
        config,
        model,
        params,
        mle_log,
        use_wandb,
        perfect_network_params,
        perfect_network,
    )

    if use_wandb:
        log_return_train = [np.array(i) for i in log_return_train]
        # log_return_test = [np.array(i) for i in log_return_test]
        # log_td_error_train = [np.array(i) for i in log_td_error_train]
        # log_td_error_test = [np.array(i) for i in log_td_error_test]
        # log_mean_novelty_train = [np.array(i) for i in log_mean_novelty_train]
        # log_mean_novelty_test = [np.array(i) for i in log_mean_novelty_test]
        # log_counts = [np.array(i) for i in log_counts]

        for i in range(len(log_return_train)):
            wandb.log(
                {
                    "steps": log_steps[i],
                    "return_train": log_return_train[i],
                    # "test_return": log_return_test[i],
                    # "td_error_train": log_td_error_train[i],
                    # "td_error_test": log_td_error_test[i],
                    # "novelty_train": log_mean_novelty_train[i],
                    # "novelty_test": log_mean_novelty_test[i],
                }
            )
        wandb.log({"total_return_train": np.sum(log_return_train)})
        # wandb.log({"total_return_train": np.sum(log_return_test)})

    # np.save(f"three_dim_plots/{scale}_kl_div.npy", np.array(log_kl_div))
    # np.save(f"three_dim_plots/{scale}_MSE.npy", np.array(log_MSE))
    np.save(f"three_dim_plots/{config.env_name}_{scale}_steps.npy", np.array(log_steps))
    # np.save(
    #     f"three_dim_plots/{scale}_log_mean_novelty_training.npy",
    #     np.array(log_mean_novelty_train),
    # )
    # np.save(
    #     f"three_dim_plots/{scale}_log_mean_novelty_test.npy",
    #     np.array(log_mean_novelty_test),
    # )
    np.save(
        f"three_dim_plots/{config.env_name}_{scale}_log_mean_abs_critic.npy",
        np.array(log_mean_abs_critic),
    )
    np.save(
        f"three_dim_plots/{config.env_name}_{scale}_log_mean_abs_actor.npy",
        np.array(log_mean_abs_actor),
    )
    np.save(
        f"three_dim_plots/{config.env_name}_{scale}_log_mean_RMS_critic.npy",
        np.array(log_mean_RMS_critic),
    )
    np.save(
        f"three_dim_plots/{config.env_name}_{scale}_log_mean_RMS_actor.npy",
        np.array(log_mean_RMS_actor),
    )
    # np.save(
    #     f"three_dim_plots/{scale}_counts.npy", np.array(log_counts),
    # )

    np.save(
        f"three_dim_plots/{config.env_name}_{scale}_return.npy",
        np.array(log_return_train),
    )

    data_to_store = {
        "log_steps": log_steps,
        "log_return_train": log_return_train,
        # "log_return_test": log_return_test,
        "network": network_ckpt,
        "train_config": config,
    }

    if args.save_checkpoint:
        save_pkl_object(
            data_to_store,
            f"agents/{config.env_name}/{config.train_type.lower()}{log_ext}.pkl",
        )


if __name__ == "__main__":
    import argparse
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-config",
        "--config_fname",
        type=str,
        default="agents/Asterix-MinAtar/ppo.yaml",
        help="Path to configuration yaml.",
    )
    parser.add_argument(
        "-seed", "--seed_id", type=int, default=100, help="Random seed of experiment.",
    )
    parser.add_argument(
        "-lr", "--lrate", type=float, default=5e-04, help="learning rate of PPO agent",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1,
        help="Scale of the frequency in the SIREN network",
    )
    parser.add_argument(
        "--wandb",
        type=lambda x: bool(strtobool(x)),
        default=True,
        help="whether to log with wandb",
    )
    parser.add_argument(
        "--SIRENs",
        type=lambda x: bool(strtobool(x)),
        default=True,
        help="whether to use SIREN layers",
    )

    # parser.add_argument(
    #     "--high_freq_multiplier",
    #     type=float,
    #     default=1,
    #     help="Multiplier for the high frequency in the SIREN network",
    # )
    # parser.add_argument(
    #     "--count_switch",
    #     type=int,
    #     default=1,
    #     help="Number of steps before switching to high frequency",
    # )
    parser.add_argument(
        "--save_checkpoint",
        type=lambda x: bool(strtobool(x)),
        default=False,
        help="whether to save the checkpointed model",
    )

    args, _ = parser.parse_known_args()
    config = load_config(args.config_fname, args.seed_id, args.lrate)
    # config.train_config.num_train_steps *= 10
    main(
        config.train_config,
        mle_log=None,
        log_ext=str(args.lrate) if args.lrate != 5e-04 else "",
        scale=args.scale,
        SIRENs=args.SIRENs,
        # count_switch=args.count_switch,
        use_wandb=args.wandb,
        # high_freq_multiplier=args.high_freq_multiplier,
        save_checkpoint=args.save_checkpoint,
    )

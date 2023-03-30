import jax

from utils.helpers import load_config, save_pkl_object
from utils.models import get_model_ready
import wandb
import numpy as np
from distutils.util import strtobool


def main(
    config,
    mle_log,
    scale,
    count_switch,
    high_freq_multiplier,
    log_ext="",
    use_wandb: bool = False,
):
    """Run training with ES or PPO. Store logs and agent ckpt."""
    rng = jax.random.PRNGKey(config.seed_id)
    # Setup the model architecture
    rng, rng_init = jax.random.split(rng)

    model, params = get_model_ready(
        rng_init, config, scale, count_switch, high_freq_multiplier
    )

    config.scale = scale
    config.count_switch = count_switch
    config.high_freq_multiplier = high_freq_multiplier
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
        log_return_test,
        log_td_error_train,
        log_td_error_test,
        log_mean_novelty_train,
        log_mean_novelty_test,
        network_ckpt,
    ) = train_fn(rng, config, model, params, mle_log, use_wandb)

    if use_wandb:
        log_return_train = [np.array(i) for i in log_return_train]
        log_return_test = [np.array(i) for i in log_return_test]
        log_td_error_train = [np.array(i) for i in log_td_error_train]
        log_td_error_test = [np.array(i) for i in log_td_error_test]
        log_mean_novelty_train = [np.array(i) for i in log_mean_novelty_train]
        log_mean_novelty_test = [np.array(i) for i in log_mean_novelty_test]
        for i in range(len(log_return_train)):
            wandb.log(
                {
                    "steps": log_steps[i],
                    "return_train": log_return_train[i],
                    "test_return": log_return_test[i],
                    "td_error_train": log_td_error_train[i],
                    "td_error_test": log_td_error_test[i],
                    "novelty_train": log_mean_novelty_train[i],
                    "novelty_test": log_mean_novelty_test[i],
                }
            )
        wandb.log({"total_return_train": np.sum(log_return_train)})
        wandb.log({"total_return_train": np.sum(log_return_test)})
    data_to_store = {
        "log_steps": log_steps,
        "log_return_train": log_return_train,
        "log_return_test": log_return_test,
        "network": network_ckpt,
        "train_config": config,
    }

    save_pkl_object(
        data_to_store,
        f"agents/{config.env_name}/{config.train_type.lower()}{log_ext}.pkl",
    )


if __name__ == "__main__":
    # Use MLE-Infrastructure if available (e.g. for parameter search)
    # try:
    #     from mle_toolbox import MLExperiment

    #     mle = MLExperiment(config_fname="configs/cartpole/ppo.yaml")
    #     main(mle.train_config, mle_log=mle.log)
    # # Otherwise use simple logging and config loading
    # except Exception:
    import argparse
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-config",
        "--config_fname",
        type=str,
        default="agents/FourRooms-misc/ppo.yaml",
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
        default=10,
        help="Scale of the frequency in the SIREN network",
    )
    parser.add_argument(
        "--wandb",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="whether to log with wandb",
    )
    parser.add_argument(
        "--high_freq_multiplier",
        type=float,
        default=1000,
        help="Multiplier for the high frequency in the SIREN network",
    )
    parser.add_argument(
        "--count_switch",
        type=int,
        default=10000000,
        help="Number of steps before switching to high frequency",
    )

    args, _ = parser.parse_known_args()
    config = load_config(args.config_fname, args.seed_id, args.lrate)
    main(
        config.train_config,
        mle_log=None,
        log_ext=str(args.lrate) if args.lrate != 5e-04 else "",
        scale=args.scale,
        count_switch=args.count_switch,
        use_wandb=args.wandb,
        high_freq_multiplier=args.high_freq_multiplier,
    )

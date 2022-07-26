from models import make_model
from components.uncert_agents import make_agent
from shared.components.dataset import Dataset
from shared.components.logger import Logger
from shared.envs.env import Env
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import uuid
from termcolor import colored
import warnings

import sys

sys.path.append('..')

warnings.simplefilter(action='ignore', category=FutureWarning)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Post train VAE agent",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    train_config = parser.add_argument_group("Train config")
    train_config.add_argument(
        "-DT",
        "--dataset",
        type=str,
        default="./dataset.hdf5",
        help='Path to dataset',
    )
    train_config.add_argument(
        "-M",
        "--model",
        type=str,
        default="./param/best_vae.pkl",
        help='Path to model',
    )
    train_config.add_argument(
        "-BS",
        "--batch-size",
        type=int,
        default=256,
        help='Batch size',
    )
    train_config.add_argument(
        "-TTP",
        "--train-test-prop",
        type=float,
        default=0.9,
        help='Train test proportion',
    )
    train_config.add_argument(
        "-E",
        "--epochs",
        type=int,
        default=20,
        help='Training epochs',
    )
    train_config.add_argument(
        "-KLD",
        "--kld-weight",
        type=float,
        default=1,
        help='KLD Weight',
    )
    train_config.add_argument(
        "-D",
        "--device",
        type=str,
        default="auto",
        help='Which device use: "cpu" or "cuda", "auto" for autodetect',
    )
    train_config.add_argument(
        "-S",
        "--seed",
        type=float,
        default=1,
        help='Pytorch seed',
    )

    # Agent Config
    agent_config = parser.add_argument_group("Agent config")
    agent_config.add_argument(
        "-SS", "--state-stack", type=int, default=6, help="Number of state stack as observation"
    )
    agent_config.add_argument(
        "-A",
        "--architecture",
        type=str,
        default="1024",
        help='Base network architecture',
    )

    args = parser.parse_args()

    run_id = uuid.uuid4()
    run_name = f"vae_model"

    # Whether to use cuda or cpu
    if args.device == "auto":
        torch.cuda.empty_cache()
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        torch.manual_seed(args.seed)
        if use_cuda:
            torch.cuda.manual_seed(args.seed)
    else:
        device = args.device
    print(colored(f"Using: {device}", "green"))

    # Init logger
    logger = Logger("highway-ddqn", args.model, run_name,
                    str(run_id), args=vars(args))
    config = logger.get_config()

    # Init Agent and Environment
    print(colored("Initializing agent", "blue"))
    architecture = [int(l) for l in config["architecture"].split("-")]
    env = Env(
        state_stack=config["state_stack"],
        action_repeat=1,
        seed=0,
        version=1,
    )
    model1 = make_model(
        model='vae',
        state_stack=config["state_stack"],
        input_dim=env.observation_dims,
        output_dim=len(env.actions),
        architecture=architecture,
    ).to(device)
    model2 = make_model(
        model='vae',
        state_stack=config["state_stack"],
        input_dim=env.observation_dims,
        output_dim=len(env.actions),
        architecture=architecture,
    ).to(device)
    agent = make_agent(
        agent='vae',
        model1=model1,
        model2=model2,
        gamma=0,
        buffer=None,
        logger=logger,
        actions=env.actions,
        epsilon=None,
        device=device,
        lr=0,
    )
    # agent.load(config["model"])
    model1.eval()
    model2.eval()
    agent._vae.train()
    env.close()
    print(colored("Agent and environments created successfully", "green"))

    for name, param in config.items():
        print(colored(f"{name}: {param}", "cyan"))

    dataset = Dataset('dataset.hdf5', overwrite=False)

    agent.update_vae(
        dataset,
        logger,
        batch_size=config["batch_size"],
        train_test_prop=config["train_test_prop"],
        epochs=config["epochs"],
        kld_weight=config["kld_weight"],
        eval_every=10000,
    )

    # agent.save(0, 'param/best_vae_trained')

    logger.close()

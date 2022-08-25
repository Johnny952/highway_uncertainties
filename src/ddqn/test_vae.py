import argparse
import torch
import torch.utils.data as data
import torch.optim as optim
import uuid
from termcolor import colored
import warnings
import json

import sys

sys.path.append('..')
from models import make_model
from components.uncert_agents import make_agent
from shared.components.dataset import Dataset
from shared.components.logger import Logger
from shared.models.vae import VAE
from shared.envs.env import Env, load_env

warnings.simplefilter(action='ignore', category=FutureWarning)

def save_options(dictionary, path='param/vae.json'):
    with open(path, 'w') as f:
        json.dump(dictionary, f)

def load_options(path='param/vae.json'):
    with open(path) as json_file:
        data = json.load(json_file)
    return data


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
        default="./dataset_train.hdf5",
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
        "-MO",
        "--mode",
        type=str,
        default="E",
        help='VAE mode, whether it is for epistemic uncertainty "E" or aleatoric uncertainty "A"',
        choices=["E", "A"]
    )
    train_config.add_argument(
        "-BS",
        "--batch-size",
        type=int,
        default=128,
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
        "-KLD",
        "--kld-weight",
        type=float,
        default=1,
        help='KLD Weight',
    )
    train_config.add_argument(
        "-ALW",
        "--act-loss-weight",
        type=float,
        default=0.2,
        help='Action Loss Weight',
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

    # VAE Config
    vae_config = parser.add_argument_group("VAE config")
    vae_config.add_argument(
        "-LR", "--learning-rate", type=float, default=0.0001, help="Learning Rate"
    )

    # Agent Config
    agent_config = parser.add_argument_group("Agent config")
    agent_config.add_argument(
        "-SS", "--state-stack", type=int, default=4, help="Number of state stack as observation"
    )
    agent_config.add_argument(
        "-A",
        "--architecture",
        type=str,
        default="512-512",
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
    logger = Logger("highway-ddqn", f'vae-{args.mode}', run_name,
                    str(run_id), args=vars(args))
    config = logger.get_config()

    # Init Agent and Environment
    print(colored("Initializing agent", "blue"))
    architecture = [int(l) for l in config["architecture"].split("-")]
    load_env()
    env = Env(
        state_stack=config["state_stack"],
        action_repeat=1,
        seed=0,
        version=1,
    )

    dataset = Dataset(config["dataset"], overwrite=False)
    train_length = int(len(dataset) * config["train_test_prop"])
    train_set, val_set = data.random_split(
        dataset, [train_length, len(dataset) - train_length])
    train_loader = data.DataLoader(
        train_set, batch_size=config["batch_size"], shuffle=True)
    val_loader = data.DataLoader(
        val_set, batch_size=config["batch_size"], shuffle=True)
    
    vae_config = load_options()
    vae = VAE(
        state_stack=vae_config["state_stack"],
        obs_dim=vae_config["obs_dim"],
        nb_actions=vae_config["nb_actions"],
        obs_encoder_arc=vae_config["obs_encoder_arc"],
        act_encoder_arc=vae_config["act_encoder_arc"],
        shared_encoder_arc=vae_config["shared_encoder_arc"],
        obs_decoder_arc=vae_config["obs_decoder_arc"],
        act_decoder_arc=vae_config["act_decoder_arc"],
        shared_decoder_arc=vae_config["shared_decoder_arc"],
        latent_dim=vae_config["latent_dim"],
        beta=vae_config["beta"],
        gamma=vae_config["gamma"],
        max_capacity=vae_config["max_capacity"],
        Capacity_max_iter=vae_config["Capacity_max_iter"],
        loss_type=vae_config["loss_type"],
        act_loss_weight=vae_config["act_loss_weight"],
    ).to(torch.float)
    optimizer = optim.Adam(vae.parameters(), lr=config["learning_rate"])

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
    checkpoint = torch.load(config["model"])
    vae.load_state_dict(checkpoint["vae_state_dict"])
    model1.load_state_dict(checkpoint["model1_state_dict"])
    model2.load_state_dict(checkpoint["model2_state_dict"])
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
        save_obs=False,
        vae1=vae,
        vae1_optimizer=optimizer,
    )

    model1.eval()
    model2.eval()
    vae.eval()
    env.close()
    print(colored("Agent and environments created successfully", "green"))

    for name, param in config.items():
        print(colored(f"{name}: {param}", "cyan"))
    agent.eval_vae(
        'E', val_loader, kld_weight=config["kld_weight"],
    )

    logger.close()

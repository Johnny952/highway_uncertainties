from models import make_model
from components.uncert_agents import make_agent
from shared.components.dataset import Dataset
from shared.components.logger import Logger
from shared.envs.env import Env
import argparse
import torch
import torch.utils.data as data
import torch.optim as optim
import uuid
from termcolor import colored
import warnings

import sys

from shared.models.vae import VAE

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
        "--mode",
        type=str,
        default='E',
        help='VAE model mode, aleatoric "A" or epistemic "E"',
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

    # VAE Config
    vae_config = parser.add_argument_group("VAE config")
    vae_config.add_argument(
        "-EA", "--encoder-arc", type=str, default='256,128,64', help="VAE Encoder architecture comma separated"
    )
    vae_config.add_argument(
        "-DA", "--decoder-arc", type=str, default='64,128,256', help="VAE Decoder architecture comma separated"
    )
    vae_config.add_argument(
        "-LD", "--latent-dim", type=int, default=32, help="VAE latent dimensions"
    )
    vae_config.add_argument(
        "-B", "--beta", type=float, default=4, help="VAE KLD scale when loss type equal 'H'"
    )
    vae_config.add_argument(
        "-G", "--gamma", type=float, default=100, help="VAE KLD scale when loss type equal 'B'"
    )
    vae_config.add_argument(
        "-MC", "--max-capacity", type=float, default=25, help="Max capacity"
    )
    vae_config.add_argument(
        "-LT", "--loss-type", type=str, default='B', help="VAE loss type, can be 'B' or 'H'"
    )
    vae_config.add_argument(
        "-LR", "--learning-rate", type=float, default=0.001, help="Learning Rate"
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

    dataset = Dataset(config["dataset"], overwrite=False)
    train_length = int(len(dataset) * config["train_test_prop"])
    train_set, val_set = data.random_split(
        dataset, [train_length, len(dataset) - train_length])
    train_loader = data.DataLoader(
        train_set, batch_size=config["batch_size"], shuffle=True)
    val_loader = data.DataLoader(
        val_set, batch_size=config["batch_size"], shuffle=True)

    Capacity_max_iter = len(train_loader) * config["epochs"]
    vae = VAE(
        state_stack=config["state_stack"],
        input_dim=env.observation_dims,
        encoder_arc=config["encoder_arc"],
        encoder_arc=config["decoder_arc"],
        latent_dim=config["latent_dim"],
        beta=config["beta"],
        gamma=config["gamma"],
        max_capacity=config["max_capacity"],
        Capacity_max_iter=Capacity_max_iter,
        loss_type=config["loss_type"],
    )
    vae.to(device)
    optimizer = optim.Adam(
            vae.parameters(), lr=config["learning_rate"])
    if config["mode"] == "E":
        agent._vae_lr = config["learning_rate"]
        agent._vae = vae
        agent._vae_optimizer = optimizer
    elif config["mode"] == "A":
        agent._vae2_lr = config["learning_rate"]
        agent._vae2 = vae
        agent._vae2_optimizer = optimizer
    else:
        raise NotImplementedError('Mode not implemented')

    agent.update_vae(
        config["mode"],
        train_loader,
        val_loader,
        logger,
        epochs=config["epochs"],
        kld_weight=config["kld_weight"],
        eval_every=10000,
    )

    # agent.save(0, 'param/best_vae_trained')

    logger.close()

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
        default=10,
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
        "-SEA", "--shared-encoder-arc", type=str, default='256-128-64', help="VAE Encoder architecture comma separated"
    )
    vae_config.add_argument(
        "-OEA", "--obs-encoder-arc", type=str, default='64-16', help="VAE Encoder architecture comma separated"
    )
    vae_config.add_argument(
        "-AEA", "--act-encoder-arc", type=str, default='16', help="VAE Encoder architecture comma separated"
    )
    vae_config.add_argument(
        "-SDA", "--shared-decoder-arc", type=str, default='64-128-256', help="VAE Decoder architecture comma separated"
    )
    vae_config.add_argument(
        "-ODA", "--obs-decoder-arc", type=str, default='16-64', help="VAE Decoder architecture comma separated"
    )
    vae_config.add_argument(
        "-ADA", "--act-decoder-arc", type=str, default='16', help="VAE Decoder architecture comma separated"
    )
    vae_config.add_argument(
        "-LD", "--latent-dim", type=int, default=32, help="VAE latent dimensions"
    )
    vae_config.add_argument(
        "-B", "--beta", type=float, default=4, help="VAE KLD scale when loss type equal 'H'"
    )
    vae_config.add_argument(
        "-G", "--gamma", type=float, default=1, help="VAE KLD scale when loss type equal 'B'"
    )
    vae_config.add_argument(
        "-MC", "--max-capacity", type=float, default=25, help="Max capacity"
    )
    vae_config.add_argument(
        "-LT", "--loss-type", type=str, default='B', help="VAE loss type, can be 'B' or 'H'"
    )
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

    Capacity_max_iter = len(train_loader) * config["epochs"]
    
    obs_encoder_arc = [int(l) for l in config["obs_encoder_arc"].split("-")]
    act_encoder_arc = [int(l) for l in config["act_encoder_arc"].split("-")]
    shared_encoder_arc = [int(l) for l in config["shared_encoder_arc"].split("-")]
    obs_decoder_arc = [int(l) for l in config["obs_decoder_arc"].split("-")]
    act_decoder_arc = [int(l) for l in config["act_decoder_arc"].split("-")]
    shared_decoder_arc = [int(l) for l in config["shared_decoder_arc"].split("-")]
    vae = VAE(
        state_stack=config["state_stack"],
        obs_dim=env.observation_dims,
        nb_actions=len(env.actions),
        obs_encoder_arc=obs_encoder_arc,
        act_encoder_arc=act_encoder_arc,
        shared_encoder_arc=shared_encoder_arc,
        obs_decoder_arc=obs_decoder_arc,
        act_decoder_arc=act_decoder_arc,
        shared_decoder_arc=shared_decoder_arc,
        latent_dim=config["latent_dim"],
        beta=config["beta"],
        gamma=config["gamma"],
        max_capacity=config["max_capacity"],
        Capacity_max_iter=Capacity_max_iter,
        loss_type=config["loss_type"],
        act_loss_weight=1,
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
    model1.load_state_dict(checkpoint["model1_state_dict"])
    model2.load_state_dict(checkpoint["model2_state_dict"])
    if args.mode == "E":
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
    elif args.mode == "A":
        options = load_options(path='param/vae-E.json')
        vae2 = VAE(
            state_stack=options["state_stack"],
            obs_dim=options["obs_dim"],
            nb_actions=options["nb_actions"],
            obs_encoder_arc=options["obs_encoder_arc"],
            act_encoder_arc=options["act_encoder_arc"],
            shared_encoder_arc=options["shared_encoder_arc"],
            obs_decoder_arc=options["obs_decoder_arc"],
            act_decoder_arc=options["act_decoder_arc"],
            shared_decoder_arc=options["shared_decoder_arc"],
            latent_dim=options["latent_dim"],
            beta=options["beta"],
            gamma=options["gamma"],
            max_capacity=options["max_capacity"],
            Capacity_max_iter=options["Capacity_max_iter"],
            loss_type=options["loss_type"],
            act_loss_weight=options["act_loss_weight"],
        ).to(torch.float)
        optimizer2 = optim.Adam(vae2.parameters(), lr=config["learning_rate"])
        vae2.eval()
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
            vae1=vae2,
            vae1_optimizer=optimizer2,
            vae2=vae,
            vae2_optimizer=optimizer,
        )

    model1.eval()
    model2.eval()
    env.close()
    print(colored("Agent and environments created successfully", "green"))

    for name, param in config.items():
        print(colored(f"{name}: {param}", "cyan"))

    agent.update_vae(
        args.mode,
        train_loader,
        val_loader,
        logger,
        epochs=config["epochs"],
        kld_weight=config["kld_weight"],
        eval_every=10000,
    )

    agent.save(0, 'param/best_vae_trained')
    save_options({
        "state_stack": config["state_stack"],
        "obs_dim": env.observation_dims,
        "nb_actions": len(env.actions),
        "obs_encoder_arc": obs_encoder_arc,
        "act_encoder_arc": act_encoder_arc,
        "shared_encoder_arc": shared_encoder_arc,
        "obs_decoder_arc": obs_decoder_arc,
        "act_decoder_arc": act_decoder_arc,
        "shared_decoder_arc": shared_decoder_arc,
        "latent_dim": config["latent_dim"],
        "beta": config["beta"],
        "gamma": config["gamma"],
        "max_capacity": config["max_capacity"],
        "Capacity_max_iter": Capacity_max_iter,
        "loss_type": config["loss_type"],
        "act_loss_weight": config["act_loss_weight"],
    }, path=f'param/vae-{args.mode}.json')

    logger.close()

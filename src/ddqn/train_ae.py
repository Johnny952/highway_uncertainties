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
from shared.models.ae import AE
from shared.envs.env import Env, load_env

warnings.simplefilter(action='ignore', category=FutureWarning)

def save_options(dictionary, path='param/ae.json'):
    with open(path, 'w') as f:
        json.dump(dictionary, f)

def load_options(path='param/ae.json'):
    with open(path) as json_file:
        data = json.load(json_file)
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Post train AE agent",
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
        default="./param/best_ae.pkl",
        help='Path to model',
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
        "-E",
        "--epochs",
        type=int,
        default=10,
        help='Training epochs',
    )
    train_config.add_argument(
        "-NBE",
        "--number-evaluations",
        type=int,
        default=100,
        help='Number of evaluations steps during training',
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

    # AE Config
    ae_config = parser.add_argument_group("AE config")
    ae_config.add_argument(
        "-SEA", "--shared-encoder-arc", type=str, default='512-512', help="AE Encoder architecture comma separated"
    )
    ae_config.add_argument(
        "-OEA", "--obs-encoder-arc", type=str, default='64-16', help="AE Encoder architecture comma separated"
    )
    ae_config.add_argument(
        "-AEA", "--act-encoder-arc", type=str, default='4-16', help="AE Encoder architecture comma separated"
    )
    ae_config.add_argument(
        "-SDA", "--shared-decoder-arc", type=str, default='512-512', help="AE Decoder architecture comma separated"
    )
    ae_config.add_argument(
        "-ODA", "--obs-decoder-arc", type=str, default='16-64', help="AE Decoder architecture comma separated"
    )
    ae_config.add_argument(
        "-ADA", "--act-decoder-arc", type=str, default='16-4', help="AE Decoder architecture comma separated"
    )
    ae_config.add_argument(
        "-LD", "--latent-dim", type=int, default=8, help="AE latent dimensions"
    )
    ae_config.add_argument(
        "-OLW", "--obs-loss-weight", type=float, default=1, help="Observation reconstruction loss weight"
    )
    ae_config.add_argument(
        "-ALW", "--act-loss-weight", type=float, default=1, help="Action reconstruction loss weight"
    )
    ae_config.add_argument(
        "-PLW", "--prob-loss-weight", type=float, default=1, help="Probability loss weight"
    )
    ae_config.add_argument(
        "-LR", "--learning-rate", type=float, default=0.0001, help="Learning Rate"
    )
    ae_config.add_argument(
        "-N", "--name", type=str, default='e', help="File sufix"
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
    run_name = f"ae_model"

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
    logger = Logger("highway-ddqn", 'ae', run_name,
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
    ae = AE(
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
        act_loss_weight=config["act_loss_weight"],
        obs_loss_weight=config["obs_loss_weight"],
        prob_loss_weight=config["prob_loss_weight"],
    ).to(torch.float)
    optimizer = optim.Adam(ae.parameters(), lr=config["learning_rate"])

    model1 = make_model(
        model='base',
        state_stack=config["state_stack"],
        input_dim=env.observation_dims,
        output_dim=len(env.actions),
        architecture=architecture,
    ).to(device)
    model2 = make_model(
        model='base',
        state_stack=config["state_stack"],
        input_dim=env.observation_dims,
        output_dim=len(env.actions),
        architecture=architecture,
    ).to(device)
    checkpoint = torch.load(config["model"])
    model1.load_state_dict(checkpoint["model1_state_dict"])
    model2.load_state_dict(checkpoint["model2_state_dict"])
    agent = make_agent(
        agent='ae',
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
        ae=ae,
        ae_optimizer=optimizer,
    )

    model1.eval()
    model2.eval()
    env.close()
    print(colored("Agent and environments created successfully", "green"))

    for name, param in config.items():
        print(colored(f"{name}: {param}", "cyan"))

    total_updates = len(train_loader) * config["epochs"]

    agent.update_ae(
        train_loader,
        val_loader,
        logger,
        epochs=config["epochs"],
        eval_every=total_updates // config["number_evaluations"],
    )

    agent.save(0, f'param/best_ae_trained-{args.name}.pkl')
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
        "act_loss_weight": config["act_loss_weight"],
        "obs_loss_weight": config["obs_loss_weight"],
        "prob_loss_weight": config["prob_loss_weight"],
    }, path=f'param/ae-{args.name}.json')

    logger.close()

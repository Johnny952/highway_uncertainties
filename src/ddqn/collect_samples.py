import argparse
import torch
import uuid
from termcolor import colored
import warnings
import sys
sys.path.append('..')

from models import make_model
from components.uncert_agents import make_agent
from shared.components.logger import SimpleLogger
from shared.envs.env import Env, load_env
from components.trainer import Trainer

warnings.simplefilter(action='ignore', category=FutureWarning)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Environment samples collector",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    train_config = parser.add_argument_group("Config")
    train_config.add_argument(
        "-DT",
        "--dataset",
        type=str,
        default="./dataset_test.hdf5",
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
        "-E",
        "--episodes",
        type=int,
        default=10000,
        help='Episodes',
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
        type=int,
        default=2,
        help='Pytorch seed',
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

    # Environment Config
    env_config = parser.add_argument_group("Environment config")
    env_config.add_argument(
        "-AR", "--action-repeat", type=int, default=1, help="repeat action in N frames"
    )

    args = parser.parse_args()

    run_id = uuid.uuid4()
    run_name = f"vae_model_test"

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
    logger = SimpleLogger("highway-ddqn", args.model, run_name, str(run_id), args=vars(args))
    config = logger.get_config()

    # Init Agent and Environment
    print(colored("Initializing agent", "blue"))
    architecture = [int(l) for l in config["architecture"].split("-")]
    load_env()
    env = Env(
        state_stack=config["state_stack"],
        action_repeat=config["action_repeat"],
        seed=config["seed"],
        version=1,
        #path_render='render/test',
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
        agent='base',
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

    agent.load(config["model"])
    model1.eval()
    model2.eval()
    print(colored("Agent and environments created successfully", "green"))

    for name, param in config.items():
        print(colored(f"{name}: {param}", "cyan"))

    trainer = Trainer(
        agent=agent,
        env=None,
        eval_env=env,
        logger=logger,
        steps=1,
        nb_evaluations=config["episodes"],
        eval_interval=1,
        model_name=run_name,
        checkpoint_every=10,
        debug=True,
        save_obs=False,
        save_obs_test=True,
        dataset_path=config["dataset"],
    )

    trainer.eval(0)

    logger.close()
    env.close()
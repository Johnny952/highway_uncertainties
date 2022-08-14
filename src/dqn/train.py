import os
import argparse
import torch
import uuid
import glob
from termcolor import colored
from pyvirtualdisplay import Display
from collections import namedtuple
import warnings
from stable_baselines3 import DQN

import sys

sys.path.append('..')
from shared.utils.uncert_file import init_uncert_file
from shared.envs.env import Env

warnings.simplefilter(action='ignore', category=FutureWarning)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a DDQN agent for Highway Env",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Environment Config
    env_config = parser.add_argument_group("Environment config")
    env_config.add_argument(
        "-AR", "--action-repeat", type=int, default=1, help="repeat action in N frames"
    )
    env_config.add_argument(
        "-TS",
        "--train-seed",
        type=float,
        default=0,
        help="Train Environment Random seed",
    )
    env_config.add_argument(
        "-ES",
        "--eval-seed",
        type=float,
        default=10,
        help="Evaluation Environment Random seed",
    )

    # Agent Config
    agent_config = parser.add_argument_group("Agent config")
    agent_config.add_argument(
        "-M",
        "--model",
        type=str,
        default="base",
        help='Type of uncertainty model: "base", "sensitivity", "dropout", "bootstrap", "aleatoric", "bnn" or "custom"',
    )
    agent_config.add_argument(
        "-G", "--gamma", type=float, default=0.99, help="discount factor"
    )
    agent_config.add_argument(
        "-SS", "--state-stack", type=int, default=2, help="Number of state stack as observation"
    )
    agent_config.add_argument(
        "-A",
        "--architecture",
        type=str,
        default="512-512",
        help='Base network architecture',
    )

    # Training Config
    train_config = parser.add_argument_group("Train config")
    train_config.add_argument(
        "-S", "--steps", type=int, default=20000, help="Number of training steps"
    )
    train_config.add_argument(
        "-TF",
        "--train-freq",
        type=int,
        default=1,
        help="Train frequency",
    )
    train_config.add_argument(
        "-D",
        "--device",
        type=str,
        default="auto",
        help='Which device use: "cpu" or "cuda", "auto" for autodetect',
    )
    train_config.add_argument(
        "-EI",
        "--eval-interval",
        type=int,
        default=200,
        help="Interval between evaluations",
    )
    train_config.add_argument(
        "-EV",
        "--evaluations",
        type=int,
        default=3,
        help="Number of evaluations episodes every x training episode",
    )
    train_config.add_argument(
        "-ER",
        "--eval-render",
        action="store_true",
        help="render the environment on evaluation",
    )
    train_config.add_argument(
        "-DB",
        "--debug",
        action="store_true",
        help="debug mode",
    )

    # Update
    update_config = parser.add_argument_group("Update config")
    update_config.add_argument(
        "-BC", "--buffer-capacity", type=int, default=15000, help="Buffer Capacity"
    )
    update_config.add_argument(
        "-BS", "--batch-size", type=int, default=32, help="Batch Capacity"
    )
    update_config.add_argument(
        "-LR", "--learning-rate", type=float, default=5e-4, help="Learning Rate"
    )

    args = parser.parse_args()
    
    run_id = uuid.uuid4()
    run_name = f"{args.model}_{run_id}"
    # run_name = args.model
    render_path = "render"
    render_eval_path = f"{render_path}/eval"
    render_eval__model_path = f"{render_eval_path}/{run_name}"
    param_path = "param"
    uncertainties_path = "uncertainties"
    uncertainties_eval_path = f"{uncertainties_path}/eval"
    uncertainties_eval_model_path = f"{uncertainties_eval_path}/{run_name}.txt"

    print(colored("Initializing data folders", "blue"))
    # Init model checkpoint folder and uncertainties folder
    if not args.debug:
        if not os.path.exists(param_path):
            os.makedirs(param_path)
        if not os.path.exists(uncertainties_path):
            os.makedirs(uncertainties_path)
        if not os.path.exists(render_path):
            os.makedirs(render_path)
        if not os.path.exists(render_eval_path):
            os.makedirs(render_eval_path)
        if not os.path.exists(render_eval__model_path):
            os.makedirs(render_eval__model_path)
        else:
            files = glob.glob(f"{render_eval__model_path}/*")
            for f in files:
                os.remove(f)
        if not os.path.exists(uncertainties_eval_path):
            os.makedirs(uncertainties_eval_path)
        init_uncert_file(file=uncertainties_eval_model_path)
    print(colored("Data folders created successfully", "green"))

    # Virtual display
    display = Display(visible=0, size=(1400, 900))
    display.start()

    # Whether to use cuda or cpu
    if args.device == "auto":
        torch.cuda.empty_cache()
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        torch.manual_seed(args.train_seed)
        if use_cuda:
            torch.cuda.manual_seed(args.train_seed)
    else:
        device = args.device
    print(colored(f"Using: {device}", "green"))

    # Init Agent and Environment
    print(colored("Initializing agent and environments", "blue"))
    env = Env(
        state_stack=args.state_stack,
        action_repeat=args.action_repeat,
        seed=args.train_seed,
        version=1,
    )
    eval_env = Env(
        state_stack=args.state_stack,
        action_repeat=args.action_repeat,
        seed=args.eval_seed,
        path_render=render_eval__model_path if args.eval_render else None,
        evaluations=args.evaluations,
        version=1,
    )
    agent = DQN('MlpPolicy', "highway-fast-v0",
                policy_kwargs=dict(net_arch=[256, 256]),
                learning_rate=5e-4,
                buffer_size=15000,
                learning_starts=200,
                batch_size=32,
                gamma=0.8,
                train_freq=1,
                gradient_steps=1,
                target_update_interval=50,
                exploration_fraction=0.7)
    print(colored("Agent and environments created successfully", "green"))

    steps = config["steps"]
    for name, param in config.items():
        print(colored(f"{name}: {param}", "cyan"))

    trainer = Trainer(
        agent=agent,
        env=env,
        eval_env=eval_env,
        logger=logger,
        steps=steps,
        nb_evaluations=config["evaluations"],
        eval_interval=config["eval_interval"],
        model_name=run_name,
        checkpoint_every=500,
        debug=config["debug"],
        save_obs=config["model"] == "vae",
        learning_start=200,
        train_freq=config["train_freq"],
    )

    trainer.run()
    env.close()
    eval_env.close()
    logger.close()
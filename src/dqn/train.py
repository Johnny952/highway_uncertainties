import os
import argparse
import torch
import uuid
import glob
from termcolor import colored
from pyvirtualdisplay import Display
from wandb.integration.sb3 import WandbCallback
import warnings
import gym
from stable_baselines3 import DQN
from gym.wrappers import RecordVideo

import sys

sys.path.append('..')
from shared.utils.uncert_file import init_uncert_file
from shared.components.logger import Logger
from shared.envs.env import RecorderWrapper, load_env

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
    # run_name = f"{args.model}_{run_id}"
    run_name = args.model
    render_path = "render"
    render_eval_path = f"{render_path}/eval"
    render_eval_model_path = f"{render_eval_path}/{run_name}"
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
        if not os.path.exists(render_eval_model_path):
            os.makedirs(render_eval_model_path)
        else:
            files = glob.glob(f"{render_eval_model_path}/*")
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

    # Init logger
    logger = Logger(
        "highway-dqn",
        args.model,
        run_name,
        str(run_id),
        args=vars(args),
        sync_tensorboard=True,
        # monitor_gym=True,
        save_code=True,
    )
    config = logger.get_config()

    # Init Agent and Environment
    print(colored("Initializing agent and environments", "blue"))
    load_env()
    env = RecorderWrapper(
        gym.make('highway-v1'),
        dataset_path='train_dataset.hdf5'
    )
    env.seed(config["train_seed"])

    eval_env = RecordVideo(
        gym.make('highway-v1'),
        video_folder=render_eval_model_path,
        # episode_trigger=lambda e: e % config["evaluations"] == config["evaluations"] // 2
    )
    eval_env.seed(config["eval_seed"])
    eval_env.configure({"simulation_frequency": 15}) 
    eval_env.unwrapped.set_record_video_wrapper(eval_env)

    env.reset()
    eval_env.reset()
    agent = DQN('MlpPolicy', env,
                policy_kwargs=dict(net_arch=[256, 256]),
                learning_rate=5e-4,
                buffer_size=30000,
                learning_starts=200,
                batch_size=32,
                gamma=0.8,
                train_freq=1,
                gradient_steps=1,
                target_update_interval=50,
                verbose=1,
                tensorboard_log=f"runs/{run_id}"
    )
    print(colored("Agent and environments created successfully", "green"))

    agent.learn(total_timesteps=int(4e4), callback=WandbCallback())
    agent.save("param/model")
    del agent

    model = DQN.load(f"{param_path}/model", env=eval_env)

    for videos in range(10):
        done = False
        obs = eval_env.reset()
        while not done:
            # Predict
            action, _states = model.predict(obs, deterministic=True)
            # Get reward
            obs, reward, done, info = eval_env.step(action)
            # Render
            # eval_env.render()

    env.close()
    eval_env.close()
    logger.close()
import os
import argparse
import torch
import uuid
import glob
from termcolor import colored
from pyvirtualdisplay import Display
from collections import namedtuple
import warnings

import sys

sys.path.append('..')
from shared.utils.uncert_file import init_uncert_file
from shared.envs.env import Env, load_env
from shared.utils.replay_buffer import ReplayMemory
from shared.components.logger import SimpleLogger
from components.uncert_agents import make_agent
from components.epsilon import Epsilon
from components.trainer import Trainer
from models import make_model
from train_ae import load_options
from shared.models.ae import AE

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
        type=int,
        default=0,
        help="Train Environment Random seed",
    )
    env_config.add_argument(
        "-ES",
        "--eval-seed",
        type=int,
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
        help='Type of uncertainty model: "base", "vae"',
    )
    agent_config.add_argument(
        "-G", "--gamma", type=float, default=0.7, help="discount factor"
    )
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
    agent_config.add_argument(
        "-FC",
        "--from-checkpoint",
        type=str,
        default="param/best_model",
        help='Path to trained model',
    )
    agent_config.add_argument(
        "-C",
        "--config",
        type=str,
        default="param/ae.json",
        help='Path to config model',
    )

    # Epsilon Config
    epsilon_config = parser.add_argument_group("Epsilon config")
    epsilon_config.add_argument(
        "-EM",
        "--epsilon-method",
        type=str,
        default="linear",
        help="Epsilon decay method: constant, linear, exp or inverse_sigmoid",
    )
    epsilon_config.add_argument(
        "-EMa",
        "--epsilon-max",
        type=float,
        default=1,
        help="The minimum value of epsilon, used this value in constant",
    )
    epsilon_config.add_argument(
        "-EMi",
        "--epsilon-min",
        type=float,
        default=0.05,
        help="The minimum value of epsilon",
    )
    epsilon_config.add_argument(
        "-EF",
        "--epsilon-factor",
        type=float,
        default=7,
        help="Factor parameter of epsilon decay, only used when method is exp or inverse_sigmoid",
    )
    epsilon_config.add_argument(
        "-EMS",
        "--epsilon-max-steps",
        type=int,
        default=15000,
        help="Max Epsilon Steps parameter, when epsilon is close to the minimum",
    )

    # Training Config
    train_config = parser.add_argument_group("Train config")
    train_config.add_argument(
        "-S", "--steps", type=int, default=120000, help="Number of training steps"
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
        default=1200,
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
        "-BC", "--buffer-capacity", type=int, default=30000, help="Buffer Capacity"
    )
    update_config.add_argument(
        "-BS", "--batch-size", type=int, default=32, help="Batch Capacity"
    )
    update_config.add_argument(
        "-LR", "--learning-rate", type=float, default=5e-4, help="Learning Rate"
    )

    # BNN
    bnn_config = parser.add_argument_group("BNN config")
    bnn_config.add_argument(
        "-NN", "--nb-nets", type=int, default=10, help="Number of forward passes while selecting action"
    )
    bnn_config.add_argument(
        "-SNB", "--sample-nbr", type=int, default=50, help="Number of forward passes while computing loss"
    )
    bnn_config.add_argument(
        "-CCW", "--complexity-cost-weight", type=float, default=50, help="KLD complexity loss weight"
    )
    bnn_config.add_argument(
        "-PS", "--prior-sigma", type=float, default=1, help="Prior gaussian variance"
    )

    args = parser.parse_args()
    
    run_id = uuid.uuid4()
    #run_name = f"{args.model}_{run_id}"
    run_name = args.model
    render_path = "render"
    render_eval_path = f"{render_path}/test"
    render_eval__model_path = f"{render_eval_path}/{run_name}"
    param_path = "param"
    uncertainties_path = "uncertainties"
    uncertainties_eval_path = f"{uncertainties_path}/test"
    uncertainties_eval_model_path = f"{uncertainties_eval_path}/{run_name}.txt"

    uncertainties_eval_path2 = f"{uncertainties_path}/test0"
    uncertainties_eval_model_path2 = f"{uncertainties_eval_path2}/{run_name}.txt"

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
        if not os.path.exists(uncertainties_eval_path2):
            os.makedirs(uncertainties_eval_path2)
        init_uncert_file(file=uncertainties_eval_model_path)
        init_uncert_file(file=uncertainties_eval_model_path2)
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
    logger = SimpleLogger("highway-ddqn", args.model, run_name, str(run_id), args=vars(args))
    config = logger.get_config()

    # Init Agent and Environment
    print(colored("Initializing agent and environments", "blue"))
    load_env()
    env = Env(
        state_stack=config["state_stack"],
        action_repeat=config["action_repeat"],
        seed=config["train_seed"],
        version=1,
    )
    eval_env = Env(
        state_stack=config["state_stack"],
        action_repeat=config["action_repeat"],
        seed=config["eval_seed"],
        path_render=render_eval__model_path if config["eval_render"] else None,
        evaluations=1,
        version=1,
    )
    Transition = namedtuple(
        "Transition", ("state", "action", "next_state", "reward", "done")
    )
    buffer = ReplayMemory(
        config["buffer_capacity"],
        config["batch_size"],
        Transition,
    )
    epsilon = Epsilon(
        max_steps=config["epsilon_max_steps"],
        method=config["epsilon_method"],
        epsilon_max=config["epsilon_max"],
        epsilon_min=config["epsilon_min"],
        factor=config["epsilon_factor"],
    )
    architecture = [int(l) for l in config["architecture"].split("-")]
    model1 = make_model(
        model=config["model"],
        state_stack=config["state_stack"],
        input_dim=env.observation_dims,
        output_dim=len(env.actions),
        architecture=architecture,

        prior_mu=0,
        prior_sigma=config["prior_sigma"],
    ).to(device)
    model2 = make_model(
        model=config["model"],
        state_stack=config["state_stack"],
        input_dim=env.observation_dims,
        output_dim=len(env.actions),
        architecture=architecture,

        prior_mu=0,
        prior_sigma=config["prior_sigma"],
    ).to(device)
    ae1 = None
    ae1_optimizer = None
    if args.model.lower().strip() == 'ae':
        options = load_options(path=config["config"])
        ae1 = AE(
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
            act_loss_weight=options["act_loss_weight"],
            obs_loss_weight=options["obs_loss_weight"],
            prob_loss_weight=options["prob_loss_weight"],
        ).to(torch.float)
        ae1_optimizer = torch.optim.Adam(ae1.parameters(), lr=config["learning_rate"])

    agent = make_agent(
        agent=config["model"],
        model1=model1,
        model2=model2,
        gamma=config["gamma"],
        buffer=buffer,
        logger=logger,
        actions=env.actions,
        epsilon=epsilon,
        device=device,
        lr=config["learning_rate"],
        save_obs=False,

        nb_nets=config["nb_nets"],
        sample_nbr=config["sample_nbr"],
        complexity_cost_weight=config['complexity_cost_weight'],

        ae=ae1,
        ae_optimizer=ae1_optimizer,
    )
    agent.load(config["from_checkpoint"], eval_mode=True)
    print(colored("Agent and environments created successfully", "green"))

    for name, param in config.items():
        print(colored(f"{name}: {param}", "cyan"))

    trainer = Trainer(
        agent=agent,
        env=env,
        eval_env=eval_env,
        logger=logger,
        steps=config["steps"],
        nb_evaluations=config["evaluations"],
        eval_interval=config["eval_interval"],
        model_name=run_name,
        checkpoint_every=500,
        debug=config["debug"],
        save_obs=False,
        learning_start=200,
        train_freq=config["train_freq"],
    )

    trainer.eval_spawn_vehicle(mode='test')
    trainer.eval_change_act(mode='test0')
    env.close()
    eval_env.close()
    logger.close()
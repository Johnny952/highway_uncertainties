import numpy as np
from tqdm import tqdm

from shared.utils.uncert_file import save_uncert
from shared.envs.env import Env
from shared.components.logger import Logger
from shared.components.dataset import Dataset
from components.uncert_agents.base_agent import BaseAgent

class Trainer:
    def __init__(
        self,
        agent: BaseAgent,
        env: Env,
        eval_env: Env,
        logger: Logger,
        steps: int,
        nb_evaluations: int = 1,
        eval_interval: int = 10,
        model_name: str="base",
        checkpoint_every: int=1000,
        debug: bool=False,
        render: bool=False,
        save_obs: bool=False,
        save_obs_test: bool=False,
        dataset_path: str='dataset.hdf5',
    ) -> None:
        self._logger = logger
        self._agent = agent
        self._env = env
        self._eval_env = eval_env
        self._nb_steps = steps
        self._nb_evaluations = nb_evaluations
        self._eval_interval = eval_interval
        self._model_name = model_name
        self._checkpoint_every = checkpoint_every
        self._debug = debug
        self._render = render

        self.best_model_path = f"param/best_{model_name}.pkl"
        self.checkpoint_model_path = f"param/checkpoint_{model_name}.pkl"

        self._save_obs = save_obs
        self._save_obs_test = save_obs_test
        if save_obs or save_obs_test:
            self._dataset = Dataset(dataset_path, overwrite=True)

        self._best_score = -1e10
        self._eval_nb = 0
        self._max_running_score = 0
        self._global_step = 0

    def run(self):
        running_score = 0
        i_ep = 0

        while self._global_step  < self._nb_steps:
            print(f"\rTraning episode: {i_ep}",end="")
            metrics = {
                "Train Episode": i_ep,
                "Episode Running Score": float(running_score),
                "Episode Score": 0,
                "Episode Steps": 0,
                "Episode Mean Speed": 0,
            }
            state = self._env.reset()
            rewards = []
            speeds = []
            done = False
            i_step = 0

            while not done:
                if self._save_obs:
                    self._dataset.push(state, i_ep, i_step)
                action, action_idx = self._agent.select_action(state)[:2]
                next_state, reward, done, info = self._env.step(action)
                if self._agent.store_transition(
                    state, action_idx, next_state, reward, done
                ):
                    self._agent.update()
                metrics["Episode Score"] += reward
                metrics["Episode Steps"] += 1
                speeds.append(info["forward_speed"])
                rewards.append(reward)

                # Eval agent
                if (self._global_step + 1) % self._eval_interval == 0:
                    eval_score = self.eval(self._global_step)

                    if eval_score > self._best_score and not self._debug:
                        self._agent.save(self._global_step, path=self.best_model_path)
                        self._best_score = eval_score

                # Save checkpoint
                if (i_ep + 1) % self._checkpoint_every == 0 and not self._debug:
                    self._agent.save(i_ep, path=self.checkpoint_model_path)


                state = next_state
                self._global_step += 1
                i_step += 1
                self._agent.epsilon_step()

            running_score = running_score * 0.99 + metrics["Episode Score"] * 0.01
            if running_score > self._max_running_score:
                self._max_running_score = running_score
            metrics["Episode Mean Speed"] = np.mean(speeds)
            metrics["Episode Std Speed"] = np.std(speeds)
            metrics["Episode Max Speed"] = np.max(speeds)
            metrics["Episode Min Speed"] = np.min(speeds)
            metrics["Episode Running Score"] = running_score
            metrics["Max Episode Running Score"] = self._max_running_score
            metrics["Epsilon"] = self._agent.get_epsilon()
            metrics["Episode Min Reward"] = float(np.min(rewards))
            metrics["Episode Max Reward"] = float(np.max(rewards))
            metrics["Episode Mean Reward"] = float(np.mean(rewards))
            # metrics["Episode Noise"] = float(info["noise"])
            self._logger.log(metrics)

            i_ep += 1
        
        eval_score = self.eval(self._global_step)
        if eval_score > self._best_score and not self._debug:
            self._agent.save(self._global_step, path=self.best_model_path)
            self._best_score = eval_score
        self._agent.save(i_ep, path=self.checkpoint_model_path)

    def eval(self, episode_nb, mode='eval'):
        assert mode in ['train', 'eval', 'test0', 'test']
        # self._agent.eval_mode()
        wandb_mode = mode.title()
        metrics = {
            f"{wandb_mode} Episode": self._eval_nb,
            f"{wandb_mode} Mean Score": 0,
            f"{wandb_mode} Mean Epist Uncert": 0,
            f"{wandb_mode} Mean Aleat Uncert": 0,
            f"{wandb_mode} Mean Steps": 0,
            f"{wandb_mode} Mean Speed": 0,
        }
        mean_uncert = np.array([0, 0], dtype=np.float64)

        for i_val in tqdm(range(self._nb_evaluations), f'{wandb_mode} step {episode_nb}'):
            score = 0
            steps = 0
            state = self._eval_env.reset()
            die = False
            speeds = []

            uncert = []
            while not die:
                if self._save_obs_test:
                    evaluation = self._eval_nb * self._nb_evaluations + i_val
                    self._dataset.push(state, evaluation, steps)
                action, _, (epis, aleat) = self._agent.select_action(state, eval=True)
                uncert.append(
                    [epis.view(-1).cpu().numpy()[0], aleat.view(-1).cpu().numpy()[0]]
                )
                next_state, reward, die, info = self._eval_env.step(action)[:4]

                score += reward
                state = next_state
                steps += 1
                speeds.append(info["forward_speed"])

            uncert = np.array(uncert)
            if not self._debug:
                save_uncert(
                    episode_nb,
                    i_val,
                    score,
                    uncert,
                    file=f"uncertainties/{mode}/{self._model_name}.txt",
                    # sigma=self._eval_env.random_noise,
                )

            mean_uncert += np.mean(uncert, axis=0) / self._nb_evaluations
            metrics[f"{wandb_mode} Mean Score"] += score / self._nb_evaluations
            metrics[f"{wandb_mode} Mean Steps"] += steps / self._nb_evaluations
            metrics[f"{wandb_mode} Mean Speed"] += np.mean(speeds) / self._nb_evaluations
        metrics[f"{wandb_mode} Mean Epist Uncert"] = mean_uncert[0]
        metrics[f"{wandb_mode} Mean Aleat Uncert"] = mean_uncert[1]

        self._logger.log(metrics)
        self._eval_nb += 1

        return metrics[f"{wandb_mode} Mean Score"]

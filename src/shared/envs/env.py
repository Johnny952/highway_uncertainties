import gym
import numpy as np
from gym.envs.registration import register
#from gym.wrappers.monitoring.video_recorder import VideoRecorder
from gym.wrappers.record_video import RecordVideo
from collections import deque

from shared.components.dataset import Dataset

def load_env():
    register(
        id='highway-v1',
        entry_point='shared.envs.custom_highway_env:CustomHighwayEnv',
    )

class Env():
    """
    Environment wrapper for CarRacing 
    """

    def __init__(self, state_stack, action_repeat, seed=None, path_render=None, evaluations=1, version=1):
        gym.logger.set_level(200)

        self.path_render = path_render
        self._render = path_render is not None
        self.env = gym.make(f'highway-v{version}')
        if self._render:
            self.evaluations = evaluations
            self.idx_val = evaluations // 2
            self.env = RecordVideo(self.env, video_folder=self.path_render, episode_trigger=lambda e: e % self.evaluations == self.idx_val)
            self.env.unwrapped.set_record_video_wrapper(self.env)
        self.env.seed(seed)
        self.fps = 15

        # self.env.metadata["render_modes"] = self.env.metadata["render.modes"]
        self.env.metadata["video.frames_per_second"] = self.fps
        self.env.metadata["render_fps"] = self.fps

        self.state_stack = deque([], maxlen=state_stack)
        self.action_repeat = action_repeat
        self.action_type = self.env.action_type
        self.action_space = self.env.action_space
        self.observation_type = self.env.observation_type
        self.actions = [i for i in range(5)]
        self.observation_dims = 5*5

        self.episode = -1
    
    def close(self):
        self.env.close()
        
    def reset(self):
        self.episode += 1
        state = self.env.reset()

        for _ in range(self.state_stack.maxlen):
            self.state_stack.append(state)
        return np.array(self.state_stack)

    def step(self, action):
        total_reward = 0
        total_steps = 0
        info = {}
        for _ in range(self.action_repeat):
            state, reward, die, info = self.env.step(action)
            total_steps += 1
            total_reward += reward
            if die:
                break
        
        info["steps"] = total_steps
        self.state_stack.append(state)
        assert len(self.state_stack) == self.state_stack.maxlen

        return np.array(self.state_stack), total_reward, die, info

    def render(self, *arg):
        return self.env.render(*arg)

    def spawn_vehicle(self):
        self.env.env.env.spawn_vehicle()

class ObservationStacker(gym.Wrapper):
    def __init__(self, env: Env, new_step_api: bool = False, state_stack=1):
        super().__init__(env, new_step_api)
        self.state_stack = deque([], maxlen=state_stack)
    
    def reset(self, **kwargs):
        state = super().reset(**kwargs)
        for _ in range(self.state_stack.maxlen):
            self.state_stack.append(state)
        return np.array(self.state_stack)
    
    def step(self, action):
        next_state, reward, done, info = super().step(action)
        self.state_stack.append(next_state)
        return np.array(self.state_stack), reward, done, info

class RecorderWrapper(gym.Wrapper):
    def __init__(self, env: Env, new_step_api: bool = False, dataset_path: str='dataset.hdf5',):
        super().__init__(env, new_step_api)
        self._dataset = Dataset(dataset_path, overwrite=True)

        self._step = 0
        self._episode = 0
    
    def step(self, action):
        next_state, reward, done, info = super().step(action)
        self._dataset.push(next_state, self._step, self._episode)
        self._step += 1
        return next_state, reward, done, info

    def reset(self, **kwargs):
        self._episode += 1
        self._step = 0
        return super().reset(**kwargs)
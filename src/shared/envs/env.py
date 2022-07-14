import gym
import numpy as np
from gym.envs.registration import register
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from collections import deque

class Env():
    """
    Environment wrapper for CarRacing 
    """

    def __init__(self, state_stack, action_repeat, seed=None, path_render=None, evaluations=1, version=1):
        gym.logger.set_level(200)

        register(
            id='highway-v1',
            entry_point='shared.envs.custom_highway_env:CustomHighwayEnv',
        )

        self.path_render = path_render
        self._render = path_render is not None
        self.env = gym.make(f'highway-v{version}')
        if self._render:
            self.evaluations = evaluations
            self.idx_val = evaluations // 2
            # self.env = Monitor(gym.make('highway-v1', verbose=0), path_render,
            #                    video_callable=lambda episode_id: episode_id % evaluations == self.idx_val, force=True)
        # self.reward_threshold = self.env.spec.reward_threshold
        self.env.seed(seed)
        self.env.config["duration"] = 600
        self.env.config["offroad_terminal"] = True
        self.fps = 7.5
        self.env.config["policy_frequency"] = 1/self.fps
        self.env.config["simulation_frequency"] = self.fps

        self.env.metadata["render_modes"] = self.env.metadata["render.modes"]
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
        self.recorder_initialized = False
    
    def close(self):
        self.env.close()
        if self._render:
            self.recorder.close()

    def create_recorder(self):
        if self.recorder_initialized:
            self.recorder.close()

        self.recorder_initialized = True
        self.recorder = VideoRecorder(
            self.env,
            base_path=f"{self.path_render}/{str(self.episode)}",
            enabled=self.episode % self.evaluations == self.idx_val,
        )
        self.recorder.frames_per_sec = 15
        
    def reset(self):
        self.episode += 1
        state = self.env.reset()
        if self._render:
            self.create_recorder()

        for _ in range(self.state_stack.maxlen):
            self.state_stack.append(state)
        return np.array(self.state_stack)

    def step(self, action):
        total_reward = 0
        total_steps = 0
        info = {}
        for _ in range(self.action_repeat):
            state, reward, die, info = self.env.step(action)
            if self._render:
                self.render()
            total_steps += 1
            total_reward += reward
            if die:
                break
        
        info["steps"] = total_steps
        self.state_stack.append(state)
        assert len(self.state_stack) == self.state_stack.maxlen

        return np.array(self.state_stack), total_reward, die, info

    def render(self, *arg):
        return self.recorder.capture_frame()

    def spawn_vehicle(self):
        self.env.env.env.spawn_vehicle()
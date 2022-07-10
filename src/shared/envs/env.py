import gym
import numpy as np
from gym.envs.registration import register
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

        self.render_path = path_render is not None
        if not self.render_path:
            self.env = gym.make(f'highway-v{version}')
        else:
            from gym.wrappers.monitoring.video_recorder import VideoRecorder

            self.evaluations = evaluations
            self.idx_val = evaluations // 2
            self.env = gym.make(f'highway-v{version}')
            metadata = {
                "render_fps": 60
                # "video.frames_per_second": 60
            }
            self.recorder = VideoRecorder(self.env, base_path=path_render, enabled=lambda episode_id: episode_id % evaluations == self.idx_val, metadata=metadata)
            # self.env = Monitor(gym.make('highway-v1', verbose=0), path_render,
            #                    video_callable=lambda episode_id: episode_id % evaluations == self.idx_val, force=True)
        # self.reward_threshold = self.env.spec.reward_threshold
        self.env.seed(seed)
        self.env.config["duration"] = 60
        self.env.config["offroad_terminal"] = True
        self.env.config["policy_frequency"] = 1/15
        self.env.config["simulation_frequency"] = 15

        self.state_stack = deque([], maxlen=state_stack)
        self.action_repeat = action_repeat
        self.action_type = self.env.action_type
        self.action_space = self.env.action_space
        self.observation_type = self.env.observation_type
        self.actions = [i for i in range(5)]
        self.observation_dims = 5*5

    
    def close(self):
        self.env.close()
        if self.render_path:
            self.recorder.close()

    def reset(self):
        state = self.env.reset()
        for _ in range(self.state_stack.maxlen):
            self.state_stack.append(state)
        return np.array(self.state_stack)

    def step(self, action):
        total_reward = 0
        total_steps = 0
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
        self.recorder.capture_frame()
        return self.env.render(*arg)

    def spawn_vehicle(self):
        self.env.env.env.spawn_vehicle()
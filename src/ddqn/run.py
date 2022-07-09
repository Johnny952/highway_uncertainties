import gym
# import highway_env
from pyvirtualdisplay import Display
from tqdm import tqdm
import warnings

import sys
sys.path.append('..')
from shared.envs.env import Env

warnings.simplefilter(action='ignore', category=FutureWarning)

# Virtual display
display = Display(visible=0, size=(1400, 900))
display.start()

env = Env(0, 0, path_render='render/render')

done = False
obs = env.reset()

for i in tqdm(range(50)):
    obs, reward, done, info = env.step(1)
    env.render()
    if i == 25:
        env.spawn_vehicle()
    # print(reward)

env.close()
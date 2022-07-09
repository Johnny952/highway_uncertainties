import os
import psutil
import torch
import gym
import numpy as np
import nvidia_smi
from pyvirtualdisplay import Display

if __name__ == "__main__":
    nvidia_smi.nvmlInit()
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)

    display = Display(visible=0, size=(1400, 900))
    display.start()

    #gym.logger.set_level(40)

    if torch.cuda.is_available():
        print("Using Cuda")

    env = gym.make('highway-v0')
    running_score = 0

    for i in range(10000):

        state = env.reset()

        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        gpu_memory = int(info.used / info.total * 100)
        process = psutil.Process(os.getpid())
        print("Memory usage at {} epoch: {} GB".format(i, process.memory_info().rss/1e9))
        print("Memory usage {}%".format(psutil.virtual_memory().percent))
        print("CPU usage {}".format(psutil.cpu_percent()))
        print(f'GPU usage: {gpu_memory}', )

        if process.memory_info().rss >= 1e10:   # if memory usage is over 10GB
            break

        score = 0
        for t in range(1000):
            state_, reward, done, info = env.step(np.array([1., 0., 0.]))
            score += reward
            state = state_
            if done:
                break
        #env.close()    # Con esto entrena el doble de iteraciones, pero el memory leak persiste
        running_score = running_score * 0.99 + score * 0.01

    env.close()
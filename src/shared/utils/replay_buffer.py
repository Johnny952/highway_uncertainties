from collections import deque
import random

class ReplayMemory(object):

    def __init__(self, capacity, batch_size, Transition):
        """Constructor of Replay Buffer

        Args:
            capacity (int): Maximum number of experiences
            batch_size (int): Number of experiences to sample
            Transition (namedtuple): Transition schema
        """        
        self.memory = deque([], maxlen=int(capacity))
        self._capacity = capacity
        self.batch_size = batch_size
        self._Transition = Transition

    def push(self, *args):
        """Save a experiences"""
        self.memory.append(self._Transition(*args))
    
    def empty(self):
        """Empty memory"""        
        self.memory.clear()

    def sample(self):
        """Sample experiences

        Raises:
            Exception: Number of experiences is less than the required

        Returns:
            list: Sample of experiences (state, action, next state, reward, done)
        """        
        if len(self) < self.batch_size:
            raise Exception('Number of experiences is less than the required')
        random_samp =  random.sample(self.memory, self.batch_size)
        return self._Transition(*zip(*random_samp))
    
    def dataset(self):
        data =  self._Transition(*zip(*self.memory))
        return data

    def __len__(self):
        return len(self.memory)
    
    def able_sample(self):
        return len(self) >= self.batch_size

    def is_memory_full(self):
        return len(self) == self._capacity


if __name__ == "__main__":
    import numpy as np
    from collections import namedtuple

    buffer = ReplayMemory(50, 16, namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done')))

    for i in range(60):
        state = np.random.randn(96, 96, 4)
        action = 3
        next_state = np.random.randn(96, 96, 4)
        reward = -1
        done = False

        buffer.push(state, action, next_state, reward, done)

        #print(f"Experiences: {i+1}\tSaved: {len(buffer)}")
    
    #print("\nSample:", len(buffer.sample()))
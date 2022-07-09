from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.distributions import Beta
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np

from shared.utils.replay_buffer import ReplayMemory
from shared.components.logger import Logger
from ddqn.components.epsilon import Epsilon

class BaseAgent:
    def __init__(
        self,
        model1: nn.Module,
        model2: nn.Module,
        gamma,
        buffer: ReplayMemory,
        logger: Logger,
        actions: "list",
        epsilon: Epsilon,
        device="cpu",
        lr=1e-3,
        nb_nets=None,
        clip_grad: bool=False,
        **kwargs,
    ):
        self._logger = logger
        self._device = device
        self._actions = actions
        self._epsilon = epsilon
        self._gamma = gamma
        self.nb_nets = nb_nets
        self._clip_grad = clip_grad

        self._buffer = buffer
        self._criterion = nn.MSELoss()

        self._model1 = model1
        self._model2 = model2

        self.lr = lr
        if self._model1 is not list or self._model1 is not dict:
            self._optimizer1 = optim.Adam(self._model1.parameters(), lr=lr)
            self._optimizer2 = optim.Adam(self._model2.parameters(), lr=lr)
        self._nb_update = 0
        self.training_step = 0

        logger.watch(self._model1)
    
    def get_epsilon(self):
        return self._epsilon.epsilon()

    def epsilon_step(self):
        return self._epsilon.step()
    
    def select_action(self, state: np.ndarray, eval=False):
        aleatoric = torch.Tensor([0])
        epistemic = torch.Tensor([0])
        if eval or np.random.rand() > self._epsilon.epsilon():
            # Select action greedily
            with torch.no_grad():
                index, (epistemic, aleatoric) = self.get_uncert(
                    (torch.from_numpy(state).unsqueeze(dim=0).float()).to(
                        self._device
                    )
                )
        else:
            # Select random action
            index = torch.randint(0, len(self._actions), size=(1,))
        return self._actions[index], index.cpu(), (epistemic, aleatoric)

    def chose_action(self, state: torch.Tensor):
        values = self._model1(state)
        _, index = torch.max(values, dim=-1)
        return index

    def get_uncert(self, state: torch.Tensor):
        values = self._model1(state)
        _, index = torch.max(values, dim=-1)

        epistemic = torch.Tensor([0])
        aleatoric = torch.Tensor([0])
        return index, (epistemic, aleatoric)

    def store_transition(self, state, action_idx, next_state, reward, done):
        self._buffer.push(
            torch.from_numpy(np.array(state, dtype=np.float32)).unsqueeze(dim=0),
            action_idx.unsqueeze(dim=0),
            torch.from_numpy(np.array(next_state, dtype=np.float32)).unsqueeze(dim=0),
            torch.Tensor([reward]),
            torch.Tensor([done]),
        )
        return self._buffer.able_sample()

    def save(self, epoch, path="param/ppo_net_params.pkl"):
        tosave = {
            "epoch": epoch,
            "model1_state_disct": self._model1.state_dict(),
            "model2_state_disct": self._model2.state_dict(),
            "optimizer1_state_dict": self._optimizer1.state_dict(),
            "optimizer2_state_dict": self._optimizer2.state_dict(),
        }
        torch.save(tosave, path)

    def load(self, path, eval_mode=False):
        checkpoint = torch.load(path)
        self._model1.load_state_dict(checkpoint["model1_state_disct"])
        self._model2.load_state_dict(checkpoint["model2_state_disct"])
        self._optimizer1.load_state_dict(checkpoint["optimizer1_state_dict"])
        self._optimizer2.load_state_dict(checkpoint["optimizer2_state_dict"])

        if eval_mode:
            self._model1.eval()
            self._model2.eval()
        else:
            self._model1.train()
            self._model2.train()
        return checkpoint["epoch"]

    def sample_buffer(self):
        dataset = self._buffer.sample()

        states = torch.cat(dataset.state).float().to(self._device)
        action_idx = torch.cat(dataset.action).type(torch.int64).to(self._device)
        next_states = torch.cat(dataset.next_state).float().to(self._device)
        rewards = torch.cat(dataset.reward).to(self._device)
        dones = torch.cat(dataset.done).to(self._device)

        return states, action_idx, next_states, rewards, dones

    def update(self):
        states, actions, next_states, rewards, dones = self.sample_buffer()
        loss1, loss2 = self.compute_loss(states, actions, next_states, rewards, dones)

        self._optimizer1.zero_grad()
        loss1.backward()
        if self._clip_grad:
            for param in self._model1.parameters():
                param.grad.data.clamp_(-1, 1)
        self._optimizer1.step()

        self._optimizer2.zero_grad()
        loss2.backward()
        if self._clip_grad:
            for param in self._model2.parameters():
                param.grad.data.clamp_(-1, 1)
        self._optimizer2.step()

        losses = {
            "Loss 1": loss1.item(),
            "Loss 2": loss2.item(),
            "Update Step": self._nb_update,
        }
        self._logger.log(losses)
        self._nb_update += 1

    def compute_loss(self, states, actions, next_states, rewards, dones):
        curr_Q1 = self._model1(states).gather(1, actions).squeeze(dim=-1)
        curr_Q2 = self._model2(states).gather(1, actions).squeeze(dim=-1)

        next_Q = torch.min(
            torch.max(self._model1(next_states), 1)[0],
            torch.max(self._model2(next_states), 1)[0],
        ).squeeze(dim=-1)
        expected_Q = rewards + (1 - dones) * self._gamma * next_Q

        loss1 = self._criterion(curr_Q1, expected_Q.detach())
        loss2 = self._criterion(curr_Q2, expected_Q.detach())

        return loss1, loss2

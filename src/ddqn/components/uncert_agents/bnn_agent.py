import torch
import torchbnn as bnn

from .base_agent import BaseAgent

class BNNAgent(BaseAgent):
    def __init__(self, nb_nets=10, sample_nbr=50, complexity_cost_weight=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nb_nets = nb_nets
        self.sample_nbr = sample_nbr
        self.complexity_cost_weight = complexity_cost_weight
        self._kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False)

    def chose_action(self, state: torch.Tensor):
        values_list = []
        for _ in range(self.nb_nets):
            values = self._model1(state)
            values_list.append(values)
        values_list = torch.stack(values_list)
        values = torch.mean(values_list, dim=0)
        _, index = torch.max(values, dim=-1)
        return index

    def get_uncert(self, state: torch.Tensor):
        values_list = []
        for _ in range(self.nb_nets):
            values = self._model1(state)
            values_list.append(values)
        values_list = torch.stack(values_list)
        values = torch.mean(values_list, dim=0)
        _, index = torch.max(values, dim=-1)
        epistemic = torch.var(values_list, dim=0)
        aleatoric = torch.Tensor([0])
        return index, epistemic, aleatoric
    
    def compute_loss(self, states, actions, next_states, rewards, dones):
        loss1 = torch.tensor([0.], dtype=torch.float, requires_grad=True).to(self._device)
        loss2 = torch.tensor([0.], dtype=torch.float, requires_grad=True).to(self._device)

        for _ in range(self.sample_nbr):
            curr_Q1 = self._model1(states).gather(1, actions).squeeze(dim=-1)
            curr_Q2 = self._model2(states).gather(1, actions).squeeze(dim=-1)

            next_Q = torch.min(
                torch.max(self._model1(next_states), 1)[0],
                torch.max(self._model2(next_states), 1)[0],
            ).squeeze(dim=-1)
            expected_Q = rewards + (1 - dones) * self._gamma * next_Q

            value_loss1 = self._criterion(curr_Q1, expected_Q.detach())
            value_loss2 = self._criterion(curr_Q2, expected_Q.detach())

            kld_loss1 = self._kl_loss(self._model1) * self.complexity_cost_weight
            kld_loss2 = self._kl_loss(self._model2) * self.complexity_cost_weight
            loss1 += kld_loss1 + value_loss1
            loss2 += kld_loss2 + value_loss2

            self._logger.log(
                {
                    "Value Loss 1": float(value_loss1),
                    "Value Loss 2": float(value_loss2),
                    "KLD Loss 2": float(kld_loss1),
                    "KLD Loss 2": float(kld_loss2),
                }
            )

        return loss1, loss2
    

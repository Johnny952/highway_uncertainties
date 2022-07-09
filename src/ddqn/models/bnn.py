import torch
import torch.nn as nn
import torchbnn as bnn
from shared.models.bnn import BNNBase

class BNN(nn.Module):
    def __init__(self, state_stack, input_dim=11, output_dim=1, architecture=[256, 128, 64], **kwargs):
        super(BNN, self).__init__()

        self.base = BNNBase(state_stack, input_dim, architecture=architecture)

        self.v = nn.Sequential(
            bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=architecture[-1], out_features=output_dim),
            nn.Softplus()
        )
    
    def forward(self, x: torch.Tensor):
        x = x.view(x.shape[0], -1)
        x = self.base(x)
        v = self.v(x)
        return v

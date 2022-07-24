import torch.nn as nn
from shared.models.base import Base

class Model(nn.Module):
    def __init__(self, state_stack, input_dim=11, output_dim=1, architecture=[256, 128, 64], p=None, **kwargs):
        super(Model, self).__init__()
        self.state_stack = state_stack
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.architecture = architecture
        self.base = Base(state_stack, input_dim, architecture=architecture, dropout=p)
        self.v = nn.Sequential(
            nn.Linear(architecture[-1], output_dim),
            nn.Softplus()
        )
    
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.base(x)
        v = self.v(x)
        return v

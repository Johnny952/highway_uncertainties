import torch.nn as nn
import torchbnn as bnn

class BNNBase(nn.Module):
    def __init__(self, state_stack, input_dim, architecture=[256, 128, 64]):
        super(BNNBase, self).__init__()

        modules = [
            bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=state_stack*input_dim, out_features=architecture[0]),
            nn.ReLU(),
        ]
        for i in range(len(architecture) -1):
            modules += [
                bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=architecture[i], out_features=architecture[i+1]),
                nn.ReLU(),
            ]
        self.fc = nn.Sequential(*modules)
    
    def forward(self, x):
        return self.fc(x)
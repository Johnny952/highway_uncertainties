import torch.nn as nn

class Base(nn.Module):
    def __init__(self, state_stack, input_dim, architecture=[256, 128, 64], dropout=None):
        super(Base, self).__init__()

        modules = [
            nn.Linear(state_stack*input_dim, architecture[0]),
            nn.ReLU(),
        ]
        for i in range(len(architecture) -1):
            if dropout:
                modules.append(nn.Dropout(p=dropout))
            modules += [
                nn.Linear(architecture[i], architecture[i+1]),
                nn.ReLU(),
            ]
        self.fc = nn.Sequential(*modules)
    
    def forward(self, x):
        return self.fc(x)
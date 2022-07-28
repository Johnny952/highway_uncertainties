import torch
import torch.nn as nn
from .base import Base


class InverseBase(nn.Module):
    def __init__(self, state_stack, input_dim, architecture=[64, 128, 256], dropout=None):
        super(InverseBase, self).__init__()

        modules = []
        for i in range(len(architecture) - 1):
            if dropout:
                modules.append(nn.Dropout(p=dropout))
            modules += [
                nn.Linear(architecture[i], architecture[i+1]),
                nn.ReLU(),
            ]
        modules += [
            nn.Linear(architecture[-1], state_stack*input_dim),
            nn.ReLU(),
        ]
        self.fc = nn.Sequential(*modules)

    def forward(self, x):
        return self.fc(x)


class VAE(nn.Module):
    def __init__(self,
        state_stack: int,
        input_dim: int,
        encoder_arc: "list[int]"=[256, 128, 64],
        decoder_arc: "list[int]"=[64, 128, 256],
        latent_dim: int=32,
        beta: float=4,
        gamma: float=100,
        max_capacity: int=25,
        Capacity_max_iter: int = 1e5,
        loss_type:str = 'B',
    ):
        super(VAE, self).__init__()

        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.encoder_arc = encoder_arc
        self.decoder_arc = decoder_arc
        self.beta = beta
        self.gamma = gamma
        self.max_capacity = max_capacity
        self.Capacity_max_iter = Capacity_max_iter
        assert loss_type in ['B', 'H']
        self.loss_type = loss_type
        self.num_iter = 0

        self.C_max = torch.Tensor([max_capacity])
        self.C_stop_iter = Capacity_max_iter

        self.decoder_loss = nn.MSELoss()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            Base(state_stack, input_dim, architecture=encoder_arc)
        )

        self.mu = nn.Linear(encoder_arc[-1], latent_dim)
        self.log_var = nn.Linear(encoder_arc[-1], latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, decoder_arc[0]),
            nn.ReLU(),
            InverseBase(state_stack, input_dim, architecture=decoder_arc),
        )

    def encode(self, x):
        x = self.encoder(x)
        mu = self.mu(x)
        log_var = self.log_var(x)
        return [mu, log_var]

    def decode(self, x):
        x = self.decoder(x)
        return x

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), x, mu, log_var]

    def loss_function(self, *args, **kwargs) -> dict:
        self.num_iter += 1
        recons = args[0]
        x = args[1]
        mu = args[2]
        log_var = args[3]
        kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        
        recons_loss = self.decoder_loss(recons, torch.flatten(x, start_dim=1))

        kld_loss = torch.mean(-0.5 * torch.sum(1 +
                              log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        l = {}

        if self.loss_type == 'H': # https://openreview.net/forum?id=Sy2fzU9gl
            loss = recons_loss + self.beta * kld_weight * kld_loss
        elif self.loss_type == 'B': # https://arxiv.org/pdf/1804.03599.pdf
            self.C_max = self.C_max.to(x.device)
            C = torch.clamp(self.C_max/self.C_stop_iter * self.num_iter, 0, self.C_max.data[0])
            loss = recons_loss + self.gamma * kld_weight * (kld_loss - C).abs()

        loss = recons_loss + kld_weight * kld_loss
        l['loss'] = loss
        l['Reconstruction_Loss'] = recons_loss.detach()
        l['kld_loss'] = kld_loss.detach()
        return l

    def sample(self, num_samples: int, current_device: int, **kwargs):
        z = torch.randn(num_samples, self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x, **kwargs):

        return self.forward(x)[0]

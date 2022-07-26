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
    def __init__(self, state_stack, input_dim, encoder_arc=[256, 128, 64], decoder_arc=[64, 128, 256], latent_dim=32):
        super(VAE, self).__init__()

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
        recons = args[0]
        x = args[1]
        mu = args[2]
        log_var = args[3]

        # Account for the minibatch samples from the dataset
        kld_weight = kwargs['M_N']
        recons_loss = self.decoder_loss(recons, torch.flatten(x, start_dim=1))

        kld_loss = torch.mean(-0.5 * torch.sum(1 +
                              log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss': recons_loss.detach(), 'kld_loss': kld_loss.detach()}

    def sample(self, num_samples: int, current_device: int, **kwargs):
        z = torch.randn(num_samples, self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x, **kwargs):

        return self.forward(x)[0]

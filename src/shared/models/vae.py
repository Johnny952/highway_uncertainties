import torch
import torch.nn as nn
from .base import Base


class InverseBase(nn.Module):
    def __init__(self, state_stack, obs_dim, architecture=[64, 128, 256], dropout=None):
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
            nn.Linear(architecture[-1], state_stack*obs_dim),
            nn.ReLU(),
        ]
        self.fc = nn.Sequential(*modules)

    def forward(self, x):
        return self.fc(x)


class VAE(nn.Module):
    def __init__(self,
        state_stack: int,
        obs_dim: int,
        nb_actions: int,
        obs_encoder_arc: "list[int]"=[64, 16],
        act_encoder_arc: "list[int]"=[16],
        shared_encoder_arc: "list[int]"=[256, 128, 64],
        obs_decoder_arc: "list[int]"=[16, 64],
        act_decoder_arc: "list[int]"=[16],
        shared_decoder_arc: "list[int]"=[64, 128, 256],
        latent_dim: int=32,
        beta: float=4,
        gamma: float=100,
        max_capacity: int=25,
        Capacity_max_iter: int = 1e5,
        loss_type:str = 'B',
        act_loss_weight: float = 1,
    ):
        super(VAE, self).__init__()

        self.latent_dim = latent_dim
        self.obs_dim = obs_dim
        self.nb_actions = nb_actions
        self.obs_encoder_arc = obs_encoder_arc
        self.act_encoder_arc = act_encoder_arc
        self.obs_decoder_arc = obs_decoder_arc
        self.act_decoder_arc = act_decoder_arc
        self.act_loss_weight = act_loss_weight
        self.beta = beta
        self.gamma = gamma
        self.max_capacity = max_capacity
        self.Capacity_max_iter = Capacity_max_iter
        assert loss_type in ['B', 'H']
        self.loss_type = loss_type
        self.num_iter = 0

        self.C_max = torch.Tensor([max_capacity])
        self.C_stop_iter = Capacity_max_iter

        self.obs_loss = nn.MSELoss()
        self.act_loss = nn.CrossEntropyLoss()

        # Encoders
        self.obs_encoder = nn.Sequential(
            nn.Flatten(),
            Base(state_stack, obs_dim, architecture=obs_encoder_arc)
        )
        self.act_encoder = nn.Sequential(
            nn.Flatten(),
            Base(1, 1, architecture=act_encoder_arc)
        )
        self.shared_encoder = nn.Sequential(
            Base(1, act_encoder_arc[-1] + obs_encoder_arc[-1], architecture=shared_encoder_arc)
        )

        # Encoding distribution parameters
        self.mu = nn.Linear(shared_encoder_arc[-1], latent_dim)
        self.log_var = nn.Linear(shared_encoder_arc[-1], latent_dim)

        # Decoders
        self.shared_decoder = nn.Sequential(
            nn.Linear(latent_dim, shared_decoder_arc[0]),
            nn.ReLU(),
            InverseBase(1, obs_decoder_arc[0] + act_decoder_arc[0], architecture=shared_decoder_arc),
        )
        self.obs_decoder = InverseBase(state_stack, obs_dim, architecture=obs_decoder_arc)
        self.act_decoder = InverseBase(1, nb_actions, architecture=act_decoder_arc)
        

    def encode(self, obs, act):
        x = self.obs_encoder(obs)
        y = self.act_encoder(act)
        z = torch.cat((x, y), dim=-1)
        z = self.shared_encoder(z)
        mu = self.mu(z)
        log_var = self.log_var(z)
        return [mu, log_var]

    def decode(self, x):
        x = self.shared_decoder(x)
        obs = self.obs_decoder(x[:, :self.obs_decoder_arc[0]])
        act = self.act_decoder(x[:, -self.act_decoder_arc[0]:])
        return obs, act

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, obs, act):
        mu, log_var = self.encode(obs, act)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), (obs, act), mu, log_var]

    def loss_function(self, *args, **kwargs) -> dict:
        self.num_iter += 1
        recons_obs, recons_act = args[0]
        obs, act = args[1]
        mu = args[2]
        log_var = args[3]
        kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        
        recons_obs_loss = self.obs_loss(recons_obs, torch.flatten(obs, start_dim=1))
        recons_act_loss = self.act_loss(recons_act, act)
        recons_loss = recons_obs_loss + self.act_loss_weight * recons_act_loss

        kld_loss = torch.mean(-0.5 * torch.sum(1 +
                              log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        l = {}

        if self.loss_type == 'H': # https://openreview.net/forum?id=Sy2fzU9gl
            loss = recons_loss + self.beta * kld_weight * kld_loss
        elif self.loss_type == 'B': # https://arxiv.org/pdf/1804.03599.pdf
            self.C_max = self.C_max.to(obs.device)
            C = torch.clamp(self.C_max/self.C_stop_iter * self.num_iter, 0, self.C_max.data[0])
            loss = recons_loss + self.gamma * kld_weight * (kld_loss - C).abs()

        loss = recons_loss + kld_weight * kld_loss
        l['loss'] = loss
        l['Reconstruction_Loss'] = recons_loss.detach()
        l['kld_loss'] = kld_loss.detach()
        l['Obs_loss'] = recons_obs_loss.detach()
        l['Act_loss'] = recons_act_loss.detach()
        return l

    def sample(self, num_samples: int, current_device: int, **kwargs):
        z = torch.randn(num_samples, self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x, **kwargs):

        return self.forward(x)[0]

import torch
import torch.nn as nn
import numpy as np
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


class AE(nn.Module):
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
        act_loss_weight: float = 1,
        obs_loss_weight: float = 1,
        prob_loss_weight: float = 1,
    ):
        super(AE, self).__init__()

        self.latent_dim = latent_dim
        self.state_stack = state_stack
        self.obs_dim = obs_dim
        self.nb_actions = nb_actions
        self.obs_encoder_arc = obs_encoder_arc
        self.act_encoder_arc = act_encoder_arc
        self.obs_decoder_arc = obs_decoder_arc
        self.act_decoder_arc = act_decoder_arc

        self.prob_loss_weight = prob_loss_weight
        self.act_loss_weight = act_loss_weight
        self.obs_loss_weight = obs_loss_weight
        self.obs_loss = nn.MSELoss()
        self.act_loss = nn.CrossEntropyLoss()
        self.loss = nn.GaussianNLLLoss()

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
        self.encoding = (
            Base(1, shared_encoder_arc[-1], architecture=[latent_dim])
        )

        # Decoders
        self.shared_decoder = nn.Sequential(
            nn.Linear(latent_dim, shared_decoder_arc[0]),
            nn.ReLU(),
            InverseBase(1, obs_decoder_arc[0] + act_decoder_arc[0], architecture=shared_decoder_arc),
        )
        self.obs_decoder = InverseBase(1, obs_decoder_arc[-1], architecture=obs_decoder_arc[:-1])
        self.act_decoder = InverseBase(1, act_decoder_arc[-1], architecture=act_decoder_arc[:-1])
        
        # Decoding distribution parameters}
        self.obs_mu = nn.Linear(obs_decoder_arc[-1], state_stack * obs_dim)
        self.obs_log_var = nn.Linear(obs_decoder_arc[-1], state_stack * obs_dim)

        self.act_mu = nn.Linear(act_decoder_arc[-1], nb_actions)
        self.act_log_var = nn.Linear(act_decoder_arc[-1], nb_actions)

    def encode(self, obs, act):
        x = self.obs_encoder(obs)
        y = self.act_encoder(act)
        z = torch.cat((x, y), dim=-1)
        z = self.shared_encoder(z)
        return self.encoding(z)

    def decode(self, x):
        x = self.shared_decoder(x)
        obs = self.obs_decoder(x[:, :self.obs_decoder_arc[0]])
        act = self.act_decoder(x[:, -self.act_decoder_arc[0]:])

        obs_mu = self.obs_mu(obs)
        obs_log_var = self.obs_log_var(obs)
        act_mu = self.act_mu(act)
        act_log_var = self.act_log_var(act)
        return (obs_mu, obs_log_var), (act_mu, act_log_var)

    def forward(self, obs, act):
        z = self.encode(obs, act)
        reconst_obs, reconst_act = self.decode(z)
        return [reconst_obs, reconst_act, (obs, act)]

    def multivariate_normal_distribution(self, x, d, mean, covariance, epsilon=1e-10):
        x_m = x - mean
        return 1.0 / (torch.sqrt((2 * np.pi)**d * (torch.linalg.det(covariance) + epsilon))) * torch.exp(-0.5 * x_m @ torch.linalg.solve(covariance, x_m.T))
    
    def prob(self, obs, act):
        (obs_mu, obs_log_var), (act_mu, act_log_var) = self(obs, act)[:2]
        one_hot_act = nn.functional.one_hot(act.squeeze(dim=1).long(), num_classes=self.nb_actions)
        target_ = torch.cat((torch.flatten(obs, start_dim=1), one_hot_act), dim=-1)
        mu = torch.cat((obs_mu, act_mu), dim=-1)
        covar = torch.exp(torch.cat((obs_log_var, act_log_var), dim=-1)) * torch.eye(self.obs_dim * self.state_stack + self.nb_actions).to(obs.device)
        return self.multivariate_normal_distribution(target_, self.obs_dim * self.state_stack + self.nb_actions, mu, covar)

    def sum_var(self, obs, act):
        (_, obs_log_var), (_, act_log_var) = self(obs, act)[:2]
        obs_sum_var = torch.sum(torch.exp(obs_log_var)) / self.obs_dim / self.state_stack
        act_sum_var = torch.sum(torch.exp(act_log_var)) / self.nb_actions
        return obs_sum_var + act_sum_var

    def log_prob(self, obs, act):
        (obs_mu, obs_log_var), (act_mu, act_log_var) = self(obs, act)[:2]
        one_hot_act = nn.functional.one_hot(act.squeeze(dim=1).long(), num_classes=self.nb_actions)
        target_ = torch.cat((torch.flatten(obs, start_dim=1), one_hot_act), dim=-1)
        mu = torch.cat((obs_mu, act_mu), dim=-1)
        covar = torch.exp(torch.cat((obs_log_var, act_log_var), dim=-1)) * torch.eye(self.obs_dim * self.state_stack + self.nb_actions).to(obs.device)
        distribution = torch.distributions.multivariate_normal.MultivariateNormal(mu, covar.unsqueeze(dim=0))
        return distribution.log_prob(target_)

    def loss_function(self, *args, **kwargs) -> dict:
        obs_mu, obs_log_var = args[0]
        act_mu, act_log_var = args[1]
        obs, act = args[2]

        obs_loss = self.obs_loss(obs_mu, torch.flatten(obs, start_dim=1))
        act_loss = self.act_loss(act_mu, act.squeeze(dim=1).long())

        one_hot_act = nn.functional.one_hot(act.squeeze(dim=1).long(), num_classes=self.nb_actions)

        target_ = torch.cat((torch.flatten(obs, start_dim=1), one_hot_act), dim=-1)
        mu = torch.cat((obs_mu, act_mu), dim=-1)
        log_var = torch.cat((obs_log_var, act_log_var), dim=-1)
        prob_loss = self.loss(mu, target_, torch.exp(log_var))

        loss = self.act_loss_weight * act_loss + self.obs_loss_weight * obs_loss + prob_loss * self.prob_loss_weight

        l = {
            'loss': loss,
            'Obs Loss': obs_loss.detach(),
            'Act Loss': act_loss.detach(),
            'Prob Loss': prob_loss.detach(),
        }
        return l

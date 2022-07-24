import torch.optim as optim
import torch

from .base_agent import BaseAgent
from shared.models.vae import VAE


class VAEAgent(BaseAgent):
    def __init__(self,
                 *args,
                 **kwargs,
                 ):
        super().__init__(*args, **kwargs)

        self._vae = VAE(
            state_stack=self._model1.state_stack,
            input_dim=self._model1.input_dim,
            encoder_arc=[256, 128, 64],
            decoder_arc=[64, 128, 256],
            latent_dim=32,
        )
        self._vae.to(self._device)
        self._vae.eval()
        self._vae_lr = 1e-3

        self._vae_optimizer = optim.Adam(
            self.vae.parameters(), lr=self._vae_lr)

    def get_uncert(self, state: torch.Tensor):
        values = self._model1(state)
        _, index = torch.max(values, dim=-1)

        epistemic = torch.Tensor([0])
        aleatoric = torch.Tensor([0])
        return index, (epistemic, aleatoric)

    def get_uncert(self, state: torch.Tensor):
        index, (_, aleatoric) = super().get_uncert(state)

        epistemic = torch.exp(self._vae.encode(state)[1])
        return index, (epistemic, aleatoric)

    def save(self, epoch, path="param/ppo_net_params.pkl"):
        tosave = {
            "epoch": epoch,
            "model1_state_dict": self._model1.state_dict(),
            "model2_state_dict": self._model2.state_dict(),
            "vae_state_dict": self._vae.state_dict(),
            "optimizer1_state_dict": self._optimizer1.state_dict(),
            "optimizer2_state_dict": self._optimizer2.state_dict(),
            "vae_optimizer_state_dict": self._vae_optimizer.state_dict(),
        }
        torch.save(tosave, path)

    def load(self, path, eval_mode=False):
        checkpoint = torch.load(path)
        self._model1.load_state_dict(checkpoint["model1_state_dict"])
        self._model2.load_state_dict(checkpoint["model2_state_dict"])
        self._vae.load_state_dict(checkpoint["vae_state_dict"])
        self._optimizer1.load_state_dict(checkpoint["optimizer1_state_dict"])
        self._optimizer2.load_state_dict(checkpoint["optimizer2_state_dict"])
        self._vae_optimizer.load_state_dict(checkpoint["vae_optimizer_state_dict"])

        if eval_mode:
            self._model1.eval()
            self._model2.eval()
            self._vae.eval()
        else:
            self._model1.train()
            self._model2.train()
            self._vae.train()
        return checkpoint["epoch"]
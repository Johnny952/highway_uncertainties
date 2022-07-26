import torch.optim as optim
import torch.nn as nn
import torch
from tqdm import tqdm
from torch.utils import data

from .base_agent import BaseAgent
from shared.models.vae import VAE
from shared.components.dataset import Dataset
from shared.components.logger import Logger


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
            self._vae.parameters(), lr=self._vae_lr)
        self._vae_criterion = nn.MSELoss()

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
        self._vae_optimizer.load_state_dict(
            checkpoint["vae_optimizer_state_dict"])

        if eval_mode:
            self._model1.eval()
            self._model2.eval()
            self._vae.eval()
        else:
            self._model1.train()
            self._model2.train()
            self._vae.train()
        return checkpoint["epoch"]

    def update_vae(self, dataset: Dataset, logger: Logger, batch_size: int = 256, train_test_prop: int = 0.9, epochs: int = 10, kld_weight=1, eval_every=1000):
        train_length = int(len(dataset) * train_test_prop)
        train_set, val_set = data.random_split(
            dataset, [train_length, len(dataset) - train_length])
        train_loader = data.DataLoader(
            train_set, batch_size=batch_size, shuffle=True)
        val_loader = data.DataLoader(
            val_set, batch_size=batch_size, shuffle=True)

        index = 0
        eval_idx = 0

        for epoch in tqdm(range(epochs), 'Training Epoch'):
            metrics = {
                'Epoch': epoch,
                'Running Loss': 0.0,
                'Running Reconst': 0.0,
                'Running KLD': 0.0,
            }
            for i, batch in tqdm(enumerate(train_loader, 0), 'Training Batch'):
                self._vae_optimizer.zero_grad()
                outputs = self._vae(batch.to(self._device))
                loss = self._vae.loss_function(*outputs, M_N=kld_weight)
                loss['loss'].backward()
                self._vae_optimizer.step()

                metrics["Running Loss"] += loss['loss'].item()
                metrics["Running Reconst"] += loss['Reconstruction_Loss']
                metrics["Running KLD"] += loss['kld_loss']

                if i % 1000 == 0:
                    logger.log({
                        'Update': index,
                        'Loss': loss['loss'].item(),
                        'Reconst': loss['Reconstruction_Loss'],
                        'KLD': loss['kld_loss'],
                    })

                if i % eval_every == 0:
                    m = self.eval_vae(val_loader, kld_weight)
                    m['Eval Idx'] = eval_idx
                    logger.log(m)
                    eval_idx += 1
                index += 1

            logger.log(metrics)

        print('Finished Training')

    def eval_vae(self, loader, kld_weight):
        metrics = {
            'Eval Loss': 0.0,
            'Eval Reconst': 0.0,
            'Eval KLD': 0.0,
        }
        for i, batch in tqdm(enumerate(loader, 0), 'Eval Batch'):
            with torch.no_grad():
                outputs = self._vae(batch.to(self._device))
                loss = self._vae.loss_function(*outputs, M_N=kld_weight)
                metrics['Eval Loss'] += loss['loss'].item()
                metrics['Eval Loss'] += loss['Reconstruction_Loss']
                metrics['Eval Loss'] += loss['kld_loss']
        return metrics

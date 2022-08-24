import torch.optim as optim
import torch
from tqdm import tqdm
from torch.utils import data

from .base_agent import BaseAgent
from shared.models.vae import VAE
from shared.components.logger import Logger
from shared.components.dataset import Dataset


class VAEAgent(BaseAgent):
    def __init__(self,
                vae1=None,
                vae1_optimizer=None,
                save_obs: bool=True,
                vae2=None,
                vae2_optimizer=None,
                *args,
                **kwargs,
                ):
        super().__init__(*args, **kwargs)

        self._dataset = Dataset('dataset_update.hdf5', overwrite=save_obs)

        if (vae1 and vae1_optimizer) and (vae2 and vae2_optimizer):
            self.load_vae(vae1, vae1_optimizer)
            self.load_vae2(vae2, vae2_optimizer)

        elif vae2 and vae2_optimizer:
            self.load_vae(vae2, vae2_optimizer)
            self.load_vae2(vae2, vae2_optimizer)

        elif vae1 and vae1_optimizer:
            self.load_vae(vae1, vae1_optimizer)
            self.load_vae2(vae1, vae1_optimizer)
        
        else:
            raise NotImplementedError('At least one vae and one optimizer must be provided')

    def load_vae(self, vae, vae_optimizer):
        self._vae = vae
        self._vae.to(self._device)
        self._vae.eval()
        self._vae_optimizer = vae_optimizer

    def load_vae2(self, vae, vae_optimizer):
        self._vae2 = vae
        self._vae_optimizer = vae_optimizer
        self._vae2.to(self._device)
        self._vae2.eval()

    def sample_buffer(self):
        dataset = self._buffer.sample()

        for idx, (state, action) in enumerate(zip(dataset.state, dataset.action)):
            self._dataset.push(state.squeeze().numpy(), action.squeeze().numpy(), self._nb_update, idx)

        states = torch.cat(dataset.state).float().to(self._device)
        action_idx = torch.cat(dataset.action).type(
            torch.int64).to(self._device)
        next_states = torch.cat(dataset.next_state).float().to(self._device)
        rewards = torch.cat(dataset.reward).to(self._device)
        dones = torch.cat(dataset.done).to(self._device)

        return states, action_idx, next_states, rewards, dones

    def get_uncert(self, state: torch.Tensor):
        index = super().get_uncert(state)[0]

        epistemic = torch.exp(self._vae.encode(state)[1])
        aleatoric = torch.exp(self._vae2.encode(state)[1])
        return index, (epistemic, aleatoric)

    def save(self, epoch, path="param/ppo_net_params.pkl"):
        tosave = {
            "epoch": epoch,
            "model1_state_dict": self._model1.state_dict(),
            "model2_state_dict": self._model2.state_dict(),
            "vae_state_dict": self._vae.state_dict(),
            "vae2_state_dict": self._vae2.state_dict(),
            "optimizer1_state_dict": self._optimizer1.state_dict(),
            "optimizer2_state_dict": self._optimizer2.state_dict(),
            "vae_optimizer_state_dict": self._vae_optimizer.state_dict(),
            "vae2_optimizer_state_dict": self._vae2_optimizer.state_dict(),
        }
        torch.save(tosave, path)

    def load(self, path, eval_mode=False):
        checkpoint = torch.load(path)
        self._model1.load_state_dict(checkpoint["model1_state_dict"])
        self._model2.load_state_dict(checkpoint["model2_state_dict"])
        self._vae.load_state_dict(checkpoint["vae_state_dict"])
        self._vae2.load_state_dict(checkpoint["vae2_state_dict"])
        self._optimizer1.load_state_dict(checkpoint["optimizer1_state_dict"])
        self._optimizer2.load_state_dict(checkpoint["optimizer2_state_dict"])
        self._vae_optimizer.load_state_dict(
            checkpoint["vae_optimizer_state_dict"])
        self._vae2_optimizer.load_state_dict(
            checkpoint["vae2_optimizer_state_dict"])

        if eval_mode:
            self._model1.eval()
            self._model2.eval()
            self._vae.eval()
        else:
            self._model1.train()
            self._model2.train()
            self._vae.train()
        return checkpoint["epoch"]

    def update_vae(self, mode, train_loader: data.DataLoader, val_loader: data.DataLoader, logger: Logger, epochs: int = 10, kld_weight=1, eval_every=1000):
        index = 0
        eval_idx = 0

        for epoch in tqdm(range(epochs), 'Training Epoch'):
            metrics = {
                'Epoch': epoch,
                'Running Loss': 0.0,
                'Running Reconst': 0.0,
                'Running KLD': 0.0,
            }
            for i, (obs, act) in tqdm(enumerate(train_loader, 0), 'Training Batch'):
                if mode == 'E':
                    self._vae_optimizer.zero_grad()
                    outputs = self._vae(obs.to(self._device), act.to(self._device))
                    loss = self._vae.loss_function(*outputs, M_N=kld_weight)
                    loss['loss'].backward()
                    self._vae_optimizer.step()
                else:
                    self._vae2_optimizer.zero_grad()
                    outputs = self._vae2(obs.to(self._device), act.to(self._device))
                    loss = self._vae2.loss_function(*outputs, M_N=kld_weight)
                    loss['loss'].backward()
                    self._vae2_optimizer.step()

                metrics["Running Loss"] += loss['loss'].item()
                metrics["Running Reconst"] += loss['Reconstruction_Loss']
                metrics["Running KLD"] += loss['kld_loss']

                if i % 1000 == 0:
                    logger.log({
                        'Update': index,
                        'Loss': loss['loss'].item(),
                        'Reconst': loss['Reconstruction_Loss'],
                        'KLD': loss['kld_loss'],
                        'Obs Loss': loss['Obs_loss'],
                        'Act Loss': loss['Act_loss'],
                    })

                if i % eval_every == 0:
                    m = self.eval_vae(mode, val_loader, kld_weight)
                    m['Eval Idx'] = eval_idx
                    logger.log(m)
                    eval_idx += 1
                index += 1

            logger.log(metrics)

        print('Finished Training')

    def eval_vae(self, mode, loader, kld_weight):
        metrics = {
            'Eval Loss': 0.0,
            'Eval Reconst': 0.0,
            'Eval KLD': 0.0,
        }
        for i, (obs, act) in tqdm(enumerate(loader, 0), 'Eval Batch'):
            with torch.no_grad():
                if mode == 'E':
                    outputs = self._vae(obs.to(self._device), act.to(self._device))
                    loss = self._vae.loss_function(*outputs, M_N=kld_weight)
                else:
                    outputs = self._vae2(obs.to(self._device), act.to(self._device))
                    loss = self._vae2.loss_function(*outputs, M_N=kld_weight)
                metrics['Eval Loss'] += loss['loss'].item()
                metrics['Eval Reconst'] += loss['Reconstruction_Loss']
                metrics['Eval KLD'] += loss['kld_loss']
                metrics['Eval Obs'] += loss['Obs_loss']
                metrics['Eval Act'] += loss['Act_loss']
        return metrics

import torch
from tqdm import tqdm
from torch.utils import data

from .base_agent import BaseAgent
from shared.components.logger import Logger
from shared.components.dataset import Dataset
from shared.models.vae import VAE


class VAEAgent(BaseAgent):
    def __init__(self,
                vae1: VAE=None,
                vae1_optimizer=None,
                save_obs: bool=True,
                vae2: VAE=None,
                vae2_optimizer=None,
                *args,
                **kwargs,
                ):
        super().__init__(*args, **kwargs)

        if save_obs:
            self._dataset = Dataset('dataset_update.hdf5')

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

    def load_vae(self, vae: VAE, vae_optimizer):
        self._vae = vae
        self._vae.to(self._device)
        self._vae.eval()
        self._vae_optimizer = vae_optimizer

    def load_vae2(self, vae: VAE, vae_optimizer):
        self._vae2 = vae
        self._vae2_optimizer = vae_optimizer
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

        epistemic = torch.sum(torch.exp(self._vae.encode(state, index.unsqueeze(dim=0).float())[1]))
        aleatoric = torch.sum(torch.exp(self._vae2.encode(state, index.unsqueeze(dim=0).float())[1]))
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
            for i, (obs, act) in enumerate(tqdm(train_loader, f'Training Batch epoch {epoch}')):
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
                    m = self.eval_vae(mode, val_loader, kld_weight, idx=eval_idx)
                    m['Eval Idx'] = eval_idx
                    logger.log(m)
                    eval_idx += 1
                index += 1

            logger.log(metrics)

        print('Finished Training')

    def eval_vae(self, mode, loader, kld_weight, idx=0):
        metrics = {
            'Eval Loss': 0.0,
            'Eval Reconst': 0.0,
            'Eval KLD': 0.0,
            'Eval Obs': 0.0,
            'Eval Act': 0.0,
            'Eval Accuracy': 0.0,
            'Eval MSE': 0.0,
            'Eval MAE': 0.0,
            'Eval log std': 0.0,
        }
        t = 0
        mse_pixels = 0
        for i, (obs, act) in enumerate(tqdm(loader, f'Eval {idx}')):
            with torch.no_grad():
                if mode == 'E':
                    outputs = self._vae(obs.to(self._device), act.to(self._device))
                    loss = self._vae.loss_function(*outputs, M_N=kld_weight)
                    recons = self._vae.decode(outputs[2])
                    metrics['Eval log std'] += torch.sum(torch.exp(self._vae.encode(obs.to(self._device), act.to(self._device))[1])) / self._vae.latent_dim
                else:
                    outputs = self._vae2(obs.to(self._device), act.to(self._device))
                    loss = self._vae2.loss_function(*outputs, M_N=kld_weight)
                    recons = self._vae2.decode(outputs[2])
                    metrics['Eval log std'] += torch.sum(torch.exp(self._vae2.encode(obs.to(self._device), act.to(self._device))[1])) / self._vae2.latent_dim
                metrics['Eval Loss'] += loss['loss'].item()
                metrics['Eval Reconst'] += loss['Reconstruction_Loss']
                metrics['Eval KLD'] += loss['kld_loss']
                metrics['Eval Obs'] += loss['Obs_loss']
                metrics['Eval Act'] += loss['Act_loss']
                metrics['Eval Accuracy'] += acc(recons[1], act.to(self._device))
                metrics['Eval MSE'] += mse(recons[0], obs.to(self._device))
                metrics['Eval MAE'] += mae(recons[0], obs.to(self._device))
                t += act.numel()
                mse_pixels += obs.numel()
        metrics['Eval Accuracy'] /= t
        metrics['Eval MSE'] /= mse_pixels
        metrics['Eval MAE'] /= mse_pixels
        metrics['Eval log std'] /= t
        return metrics

def acc(pred, target):
    inp = torch.argmax(pred, dim=1)
    return torch.sum(target.squeeze() == inp)

def mse(pred, target):
    return torch.nn.functional.mse_loss(pred, torch.flatten(target, start_dim=1), reduction='sum')

def mae(pred, target):
    return torch.nn.functional.l1_loss(pred, torch.flatten(target, start_dim=1), reduction='sum') 
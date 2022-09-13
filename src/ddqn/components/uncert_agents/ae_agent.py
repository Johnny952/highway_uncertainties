import torch
from tqdm import tqdm
from torch.utils import data

from .base_agent import BaseAgent
from shared.components.logger import Logger
from shared.components.dataset import Dataset
from shared.models.ae import AE


class AEAgent(BaseAgent):
    def __init__(self,
                ae: AE=None,
                ae_optimizer=None,
                save_obs=False,
                *args,
                **kwargs,
                ):
        super().__init__(*args, **kwargs)

        if save_obs:
            self._dataset = Dataset('dataset_update.hdf5')
        self._ae = ae
        self._ae.to(self._device)
        self._ae.eval()
        self._ae_optimizer = ae_optimizer

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

        epistemic = self._ae.prob(state, index.unsqueeze(dim=0).float())
        #print(epistemic.shape)
        #raise Exception
        aleatoric = torch.Tensor([0])
        return index, (epistemic, aleatoric)

    def save(self, epoch, path="param/ppo_net_params.pkl"):
        tosave = {
            "epoch": epoch,
            "model1_state_dict": self._model1.state_dict(),
            "model2_state_dict": self._model2.state_dict(),
            "ae_state_dict": self._ae.state_dict(),
            "optimizer1_state_dict": self._optimizer1.state_dict(),
            "optimizer2_state_dict": self._optimizer2.state_dict(),
            "ae_optimizer_state_dict": self._ae_optimizer.state_dict(),
        }
        torch.save(tosave, path)

    def load(self, path, eval_mode=False):
        checkpoint = torch.load(path)
        self._model1.load_state_dict(checkpoint["model1_state_dict"])
        self._model2.load_state_dict(checkpoint["model2_state_dict"])
        self._ae.load_state_dict(checkpoint["ae_state_dict"])
        self._optimizer1.load_state_dict(checkpoint["optimizer1_state_dict"])
        self._optimizer2.load_state_dict(checkpoint["optimizer2_state_dict"])
        self._ae_optimizer.load_state_dict(
            checkpoint["ae_optimizer_state_dict"])

        if eval_mode:
            self._model1.eval()
            self._model2.eval()
            self._ae.eval()
        else:
            self._model1.train()
            self._model2.train()
            self._ae.train()
        return checkpoint["epoch"]

    def update_ae(self, train_loader: data.DataLoader, val_loader: data.DataLoader, logger: Logger, epochs: int = 10, eval_every=1000):
        index = 0
        eval_idx = 0

        for epoch in tqdm(range(epochs), 'Training Epoch'):
            metrics = {
                'Epoch': epoch,
                'Running Loss': 0.0,
                'Running Act Loss': 0.0,
                'Running Obs Loss': 0.0,
                'Running Prob Loss': 0.0,
            }
            for i, (obs, act) in enumerate(tqdm(train_loader, f'Training epoch {epoch} Batch')):
                self._ae_optimizer.zero_grad()
                outputs = self._ae(obs.to(self._device), act.to(self._device))
                loss = self._ae.loss_function(*outputs)
                loss['loss'].backward()
                self._ae_optimizer.step()

                metrics["Running Loss"] += loss['loss'].item()
                metrics["Running Act Loss"] += loss['Act Loss']
                metrics["Running Obs Loss"] += loss['Obs Loss']
                metrics["Running Prob Loss"] += loss['Prob Loss']

                if i % 1000 == 0:
                    logger.log({
                        'Update': index,
                        'Loss': loss['loss'].item(),
                        'Act Loss': loss['Act Loss'],
                        'Obs Loss': loss['Obs Loss'],
                        'Prob Loss': loss['Prob Loss'],
                    })

                if i % eval_every == 0:
                    m = self.eval_ae(val_loader, idx=eval_idx)
                    m['Eval Idx'] = eval_idx
                    logger.log(m)
                    eval_idx += 1
                index += 1

            # logger.log(metrics)

        print('Finished Training')

    def eval_ae(self, loader, idx=0):
        metrics = {
            'Eval Loss': 0.0,
            'Eval Obs Loss': 0.0,
            'Eval Act Loss': 0.0,
            'Eval Prob Loss': 0.0,
            'Eval Accuracy': 0.0,
            'Eval MSE': 0.0,
            'Eval MAE': 0.0,
            'Eval log std': 0.0,
        }
        t = 0
        mse_pixels = 0
        for i, (obs, act) in enumerate(tqdm(loader, f'Eval {idx}')):
            with torch.no_grad():
                outputs = self._ae(obs.to(self._device), act.to(self._device))
                loss = self._ae.loss_function(*outputs)

                obs_log_var = outputs[0][1]
                act_log_var = outputs[0][1]

                metrics['Eval log std'] += torch.sum(torch.exp(obs_log_var)) + torch.sum(torch.exp(act_log_var))
                metrics['Eval Loss'] += loss['loss'].item()
                metrics['Eval Obs Loss'] += loss['Obs Loss']
                metrics['Eval Act Loss'] += loss['Act Loss']
                metrics['Eval Prob Loss'] += loss['Prob Loss']
                reconst_obs = outputs[0][0]
                reconst_act = outputs[1][0]
                metrics['Eval Accuracy'] += acc(reconst_act, act.to(self._device))
                metrics['Eval MSE'] += mse(reconst_obs, obs.to(self._device))
                metrics['Eval MAE'] += mae(reconst_obs, obs.to(self._device))
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
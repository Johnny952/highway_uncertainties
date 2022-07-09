from .model import Model
from .bnn import BNN
from shared.models.vae import VAE

class VAEModel:
    def __init__(self, **kwargs) -> None:
        self.vae = VAE(
            encoder_arc=[256, 128, 64],
            decoder_arc=[64, 128, 256],
            latent_dim=32,
            **kwargs
        )
        self.model = Model(**kwargs)

    def to(self, device):
        self.vae.to(device)
        self.model.to(device)

def make_model(
        model = 'base',
        **kwargs,
    ):
    switcher = {
        'base': Model,
        'bnn': BNN,
        'vae': VAEModel,
    }
    return switcher.get(model, Model)(**kwargs)

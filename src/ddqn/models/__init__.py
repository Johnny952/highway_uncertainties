from .model import Model
from .bnn import BNN
from shared.models.vae import VAE

def make_model(
        model = 'base',
        **kwargs,
    ):
    switcher = {
        'base': Model,
        'bnn': BNN,
        'vae': Model,
    }
    return switcher.get(model, Model)(**kwargs)

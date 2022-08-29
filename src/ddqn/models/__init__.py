from .model import Model
from .bnn import BNN

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

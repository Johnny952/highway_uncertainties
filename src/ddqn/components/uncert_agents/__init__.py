from .base_agent import BaseAgent
# from .sensitivity_agent import SensitivityAgent
# from .aleatoric_agent import AleatoricAgent
# from .dropout_agent import DropoutAgent
# from .dropout_agent2 import DropoutAgent2
# from .bootstrap_agent import BootstrapAgent
# from .bootstrap_agent2 import BootstrapAgent2
# from .bnn_agent import BNNAgent
# from .vae_agent import VAEAgent

def make_agent(agent='base', **kwargs):
    switcher = {
        'base': BaseAgent,
        # 'dropout': DropoutAgent,
        # 'dropout2': DropoutAgent2,
        # 'bootstrap': BootstrapAgent,
        # 'bootstrap2': BootstrapAgent2,
        # 'sensitivity': SensitivityAgent,
        # 'bnn': BNNAgent,
        # 'aleatoric': AleatoricAgent,
        # 'vae': VAEAgent,
    }
    return switcher.get(agent, BaseAgent)(**kwargs)

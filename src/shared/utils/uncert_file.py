import numpy as np

def save_uncert(epoch, val_episode, score, uncert, file='uncertainties/train.txt', sigma=None):
    with open(file, 'a+') as f:
        if sigma is None:
            np.savetxt(f, np.concatenate(([epoch], [val_episode], [score], uncert.T.reshape(-1))).reshape(1, -1), delimiter=',')
        else:
            np.savetxt(f, np.concatenate(([epoch], [val_episode], [score], [sigma], uncert.T.reshape(-1))).reshape(1, -1), delimiter=',')

def init_uncert_file(file='uncertainties/train.txt'):
    with open(file, 'w+') as f:
        pass

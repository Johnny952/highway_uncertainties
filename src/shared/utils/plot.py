import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

_NAN_ = -1
def scale01(array):
    max_ = np.max(array)
    min_ = np.min(array)
    if max_ == min_:
        if max_ == 0:
            return array, 0
        else:
            return array / max_, 0
    return (array - min_) / (max_ - min_), (max_ - min_)

def read_uncert(path):
    epochs = []
    val_idx = []
    reward = []
    sigma = []
    epist = []
    aleat = []
    with open(path, "r") as f:
        for row in f:
            data = np.array(row[:-1].split(",")).astype(np.float32)
            epochs.append(data[0])
            val_idx.append(data[1])
            reward.append(data[2])
            sigma.append(data[3])
            l = len(data) - 4
            epist.append(data[4 : l // 2 + 4])
            aleat.append(data[l // 2 + 4 :])
    return process(np.array(epochs), np.array(reward), epist, aleat), np.unique(sigma), sigma


def process(epochs, reward, epist, aleat):
    unique_ep = np.unique(epochs)
    mean_reward = np.zeros(unique_ep.shape, dtype=np.float32)
    mean_epist = np.zeros(unique_ep.shape, dtype=np.float32)
    mean_aleat = np.zeros(unique_ep.shape, dtype=np.float32)
    std_reward = np.zeros(unique_ep.shape, dtype=np.float32)
    std_epist = np.zeros(unique_ep.shape, dtype=np.float32)
    std_aleat = np.zeros(unique_ep.shape, dtype=np.float32)
    for idx, ep in enumerate(unique_ep):
        indexes = np.argwhere(ep == epochs).astype(int)
        mean_reward[idx] = np.mean(reward[indexes])
        std_reward[idx] = np.std(reward[indexes])
        for i in range(indexes.shape[0]):
            mean_epist[idx] += np.mean(epist[indexes[i][0]]) / indexes.shape[0]
            std_epist[idx] += np.std(epist[indexes[i][0]]) / indexes.shape[0]
            mean_aleat[idx] += np.mean(aleat[indexes[i][0]]) / indexes.shape[0]
            std_aleat[idx] += np.std(aleat[indexes[i][0]]) / indexes.shape[0]
    return (
        epochs,
        (unique_ep, mean_reward, mean_epist, mean_aleat),
        (std_reward, std_epist, std_aleat),
        (epist, aleat),
    )

def plot_time(paths, names, log_scales, uncertainties, figure='images/time_*.png', nb_eval=-1, red_lines=[10, 20]):
    for idx, (path, name, log_scale, uncertainty) in enumerate(zip(paths, names, log_scales, uncertainties)):
        (
            _,
            (unique_ep, mean_reward, mean_epist, mean_aleat),
            (std_reward, std_epist, std_aleat),
            (epist, aleat),
        ) = read_uncert(path)[0]
        uncert_label = 'Epistemic' if uncertainty == 1 else 'Aleatoric'
        if uncertainty == 1:
            uncert = epist
        elif uncertainty == 2:
            uncert = aleat
        else:
            raise NotImplementedError()
        
        plt.figure(figsize=(20, 10))
        start_idx = 5
        to_plot = uncert[nb_eval][start_idx:]
        x_indices = range(start_idx, len(uncert[nb_eval]))
        plt.plot(x_indices, to_plot)
        # plt.plot(uncert[nb_eval])
        for red_line in red_lines:
            plt.plot([red_line, red_line], [min(to_plot), max(to_plot)], 'r')
        if log_scale:
            plt.yscale('log')
        plt.xlabel('Step', fontsize=16)
        plt.ylabel(f'{uncert_label} Uncertainty', fontsize=16)
        plt.title(f"{uncert_label} Uncertainty of {name} during evaluation", fontsize=18)
        plt.savefig(figure.replace('*', f"{name}-{uncert_label}"))
        plt.close()
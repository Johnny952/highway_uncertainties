import h5py
from torch.utils import data
from pathlib import Path
from tqdm import tqdm

class Dataset(data.Dataset):
    def __init__(self, file_path: str, overwrite=False) -> None:
        super().__init__()

        self._file_path = file_path

        p = Path(file_path)
        if not p.is_file() or overwrite:
            with h5py.File(self._file_path, 'w'):
                pass
        
        self._keys = []
        self._episodes = {}
        self._read_keys()

    def push(self, observation, episode, timestamp):
        ep_name = str(episode)
        time_name = str(timestamp)
        self._push(observation, ep_name, time_name)

    def _push(self, observation, episode, timestamp):
        with h5py.File(self._file_path, 'a') as h5_file:
            if not episode in h5_file.keys():
                h5_file.create_group(episode)
            if not timestamp in h5_file[episode].keys():
                h5_file[episode].create_dataset(timestamp, data=observation)
            else:
                h5_file[episode][timestamp] = observation

    def _read_keys(self):
        with h5py.File(self._file_path, 'r') as h5_file:
            for ep_name, episode in tqdm(h5_file.items(), "Reading data info "):
                for time_name in episode.keys():
                    self._keys.append({
                        "episode": ep_name,
                        "timestamp": time_name,
                    })
                    if not ep_name in self._episodes.keys():
                        self._episodes[ep_name] = [time_name]
                    else:
                        self._episodes[ep_name].append(time_name)

    def __len__(self):
        return len(self._keys)
    
    def __getitem__(self, i):
        pointer = self._keys[i]
        episode, timestamp = pointer["episode"], pointer["timestamp"]
        with h5py.File(self._file_path, 'r') as f:
            d = f[episode][timestamp][()]
        return d

if __name__ == "__main__":
    dataset = Dataset('../../ddqn/dataset.hdf5')
    print(dataset[-1])
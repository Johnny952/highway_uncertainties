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
        self.obs_key = 'observation'
        self.act_key = 'action'

    def push(self, observation, action, episode, timestamp):
        ep_name = str(episode)
        time_name = str(timestamp)
        self._push(observation, action, ep_name, time_name)

    def _push(self, observation, action, episode, timestamp):
        with h5py.File(self._file_path, 'a') as h5_file:
            if not episode in h5_file.keys():
                ep = h5_file.create_group(episode)
            else:
                ep = h5_file[episode]
            if not timestamp in h5_file[episode].keys():
                t = ep.create_group(timestamp)
            else:
                t = ep[timestamp]
            if not self.obs_key in t.keys():
                t.create_dataset(self.obs_key, data=observation)
            else:
                t[self.obs_key] = observation
            if not self.act_key in t.keys():
                t.create_dataset(self.act_key, data=action)
            else:
                t[self.act_key] = action

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
            obs = f[episode][timestamp][self.obs_key][()]
            act = f[episode][timestamp][self.act_key][()]
        return obs, act

if __name__ == "__main__":
    dataset = Dataset('../../ddqn/dataset_update.hdf5')
    print(len(dataset))
    #print(dataset[0][0])
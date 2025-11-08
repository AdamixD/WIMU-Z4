import numpy as np
from torch.utils.data import Dataset


class SongSequenceDataset(Dataset):
    def __init__(self, items: list[tuple[np.ndarray, np.ndarray]]):
        self.items = items
        self.input_dim = items[0][0].shape[1]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]

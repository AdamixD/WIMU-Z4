import numpy as np
from torch.utils.data import Dataset


class SongSequenceDataset(Dataset):
    """Dataset for VA regression (continuous outputs)"""

    def __init__(self, items: list[tuple[np.ndarray, np.ndarray]]):
        self.items = items
        self.input_dim = items[0][0].shape[1]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


class SongClassificationDataset(Dataset):
    """Dataset for Russell 4Q classification (discrete class outputs)"""

    def __init__(self, items: list[tuple[np.ndarray, np.ndarray]]):
        """
        Args:
            items: List of (embeddings, labels) tuples
                - embeddings: (seq_len, feat_dim) - audio embeddings over time
                - labels: (seq_len,) - class labels for each time step
        """
        self.items = items
        self.input_dim = items[0][0].shape[1]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]

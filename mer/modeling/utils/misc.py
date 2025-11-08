import random

import torch
import numpy as np


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def pad_and_mask(batch: list[tuple[np.ndarray, np.ndarray]]):
    maxT = max(x.shape[0] for x, _ in batch)
    Xs, Ys, Ms = [], [], []
    for X, Y in batch:
        T = X.shape[0]
        pad = maxT - T
        if pad > 0:
            X = np.pad(X, ((0, pad), (0, 0)), mode='edge')
            Y = np.pad(Y, ((0, pad), (0, 0)), mode='edge')
            m = np.zeros((maxT,), np.float32)
            m[:T] = 1
        else:
            m = np.ones((maxT,), np.float32)
        Xs.append(torch.from_numpy(X).float())
        Ys.append(torch.from_numpy(Y).float())
        Ms.append(torch.from_numpy(m).float())
    return torch.stack(Xs, 0), torch.stack(Ys, 0), torch.stack(Ms, 0)
